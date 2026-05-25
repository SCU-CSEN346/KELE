#!/usr/bin/env python3
"""Build DPO preference pairs for Stage 3 of the SocratTeachLLM pipeline.

Three sources per docs/TRAINING_PLAN.md §3:

  Source 1 — judge-mined from Stage 2b checkpoint. Generates K candidate
             responses per turn from the fine-tuned model, scores each with
             the 4-axis LLM judge, keeps pairs where
             judge(preferred) - judge(rejected) ≥ --judge-threshold.
             Requires a Stage 2b checkpoint — scaffolded inert until that
             exists. The CLI flag is wired; the body returns early.

  Source 2 — SocratTeachLLM-as-rejected on synthetic data. Pairs the synthetic
             ground-truth teacher (preferred) against STL's response on the
             same turn (rejected). Requires per-turn STL output logs on the
             synthetic split, which the current STL synthetic runs only
             persist as aggregated summaries. Scaffolded inert until a
             dialogue-level log exists.

  Source 3 — Programmatic perturbations. Applies the anti-pattern catalogue
             from src/project/tournament_utilizations.py (preamble bloat,
             multi-question, off-topic explainer) plus two plan-prescribed
             extensions (direct-answer reveal, stage skip) to ground-truth
             teacher turns from SocratDataset-EN. Zero API cost, fully
             functional in this PR.

Output: a Parquet file with one row per pair. Columns include the prompt,
chosen + rejected responses, source/sub-source labels, ground-truth state and
stage, and a `holdout` flag (5% stratified by stage × source). TRL's DPOTrainer
ingests Parquet directly via datasets.Dataset.from_parquet.

Usage:
  uv run python scripts/build_dpo_pairs.py --dry-run
  uv run python scripts/build_dpo_pairs.py --source 3 --output data/dpo_pairs/source3.parquet
  uv run python scripts/build_dpo_pairs.py --source all --output data/dpo_pairs/all.parquet
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Pair schema
# ---------------------------------------------------------------------------


@dataclass
class DPOPair:
    """One preference pair. Columns map directly to Parquet output."""

    pair_id: str
    source: int  # 1, 2, or 3
    sub_source: str  # e.g. "preamble" | "multi-question" | "stl" | "judge-mined"
    dialogue_id: str
    turn_idx: int
    stage: str  # a | b | c | d | e (state[0])
    state: str  # full state code (a0, b3, c19, ...)
    prompt: str  # user turn (with consultant marker per §0.2)
    chosen: str  # preferred teacher response
    rejected: str  # rejected teacher response
    judge_chosen: float = float("nan")
    judge_rejected: float = float("nan")
    judge_delta: float = float("nan")
    seed: int = 0
    holdout: bool = False


# ---------------------------------------------------------------------------
# Stage / source target distributions
# ---------------------------------------------------------------------------

# Softened from the natural SocratDataset distribution (which is heavy on b/c).
# Plan agent design §4.7 — full uniform is too aggressive; this keeps the rare
# stages prominent without 4× oversampling.
STAGE_TARGET = {"a": 0.12, "b": 0.22, "c": 0.30, "d": 0.24, "e": 0.12}

# Source weights for the final pair set composition. Plan agent design §4.6.
# Until Sources 1 and 2 land, Source 3 takes the full 100%.
SOURCE_WEIGHT = {1: 0.50, 2: 0.10, 3: 0.40}


# ---------------------------------------------------------------------------
# Anti-pattern perturbations (Source 3)
# ---------------------------------------------------------------------------
#
# Each perturbation takes (turn dict, ground_truth teacher response) and returns
# a corrupted teacher response that exhibits a specific anti-pattern. Empirical
# justification: tournament_utilizations.py:244-261 hand-crafted three of these
# and they delivered +0.29 composite as in-context negatives — strong enough to
# justify mechanical application here.

_PREAMBLE_PREFIXES = (
    "嗯，这是一个很好的问题。让我想想... ",
    "好的，让我先解释一下背景。",
    "首先，我们需要回顾一下相关知识点。",
    "我来给你详细分析一下。",
)

_OFF_TOPIC_FILLERS = (
    "其实，这个话题让我想到了一个有趣的故事。",
    "顺便说一句，类似的概念在其他领域也有应用。",
    "我们先放下这个问题，看看一些相关的背景知识。",
)


def perturb_preamble(turn: dict, gt_teacher: str) -> str | None:
    """Prepend a long-winded preamble before the actual question.

    Anti-pattern #1 from tournament_utilizations.py: "嗯，这是一个很好的问题。
    让我想想..." style padding. Trained models that emit these score lower on
    the question-form-fidelity axis."""
    if not gt_teacher:
        return None
    preamble = _PREAMBLE_PREFIXES[hash(turn.get("student", "")) % len(_PREAMBLE_PREFIXES)]
    return preamble + gt_teacher


def perturb_multi_question(turn: dict, gt_teacher: str) -> str | None:
    """Append a second question, violating one-question-per-turn discipline.

    Anti-pattern #2: from the b-stage anti-example, where the teacher follows
    up the main question with "另外你还想了解什么吗？" or similar."""
    if not gt_teacher:
        return None
    if gt_teacher.count("？") + gt_teacher.count("?") >= 2:
        return None  # already multi-question — skip
    tail = "另外，你还有什么想问的吗？"
    if gt_teacher.endswith("？") or gt_teacher.endswith("?"):
        return gt_teacher + tail
    return gt_teacher + "？" + tail


def perturb_off_topic(turn: dict, gt_teacher: str) -> str | None:
    """Insert an off-topic tangent before returning to the question.

    Anti-pattern #3: from the c-stage anti-example, where the teacher drifts
    into background explanation instead of staying focused on the student's
    current line of reasoning."""
    if not gt_teacher:
        return None
    tangent = _OFF_TOPIC_FILLERS[hash(turn.get("teacher", "")) % len(_OFF_TOPIC_FILLERS)]
    return tangent + " " + gt_teacher


def perturb_direct_answer_reveal(turn: dict, gt_teacher: str) -> str | None:
    """Reveal the answer directly instead of asking a question.

    Anti-pattern #4 (plan extension): the most catastrophic Socratic failure
    — gives away the answer rather than scaffolding discovery. Only meaningful
    on stages b/c/d where the answer hasn't been confirmed yet."""
    state = turn.get("state", "") or ""
    stage = state[0] if state else ""
    if stage not in {"b", "c", "d"}:
        return None
    # Strip the trailing question and replace with a declarative reveal.
    student = turn.get("student", "")[:80]
    return f"答案是这样的：{student}。这就是正确的方向。"


def perturb_stage_skip(turn: dict, gt_teacher: str) -> str | None:
    """Use a stage-e (closure) move when the current state is b (probing).

    Anti-pattern #5 (plan extension): premature closure — summarises and
    confirms understanding before the student has actually worked through
    the misconception."""
    state = turn.get("state", "") or ""
    stage = state[0] if state else ""
    if stage != "b":
        return None
    return "让我们总结一下：你已经掌握了这个概念。这就是答案，对吗？"


PERTURBATIONS = [
    ("preamble", perturb_preamble),
    ("multi-question", perturb_multi_question),
    ("off-topic", perturb_off_topic),
    ("direct-answer", perturb_direct_answer_reveal),
    ("stage-skip", perturb_stage_skip),
]


# ---------------------------------------------------------------------------
# Source 3: programmatic perturbations from SocratDataset-EN ground truth
# ---------------------------------------------------------------------------


def _build_user_prompt(student: str, state: str, action: str | None) -> str:
    """Mirror the Pattern A (long-label) user-turn format from dataset.py."""
    user_content = student
    if state and action:
        user_content += (
            f"\n\n苏格拉底教学顾问评估结果: 学生处于 {state} 状态\n"
            f"苏格拉底教学顾问建议的操作: {action}"
        )
    return user_content


def build_source3_pairs(max_dialogues: int | None = None) -> Iterator[DPOPair]:
    """Yield Source 3 pairs from SocratDataset-EN.

    For each (dialogue, turn) with a ground-truth state+action, apply every
    applicable perturbation to the ground-truth teacher response and emit a
    pair (chosen=GT, rejected=perturbed). Some perturbations are stage-gated
    (return None when not applicable) — they're silently skipped."""
    from src.project.dataset import load_socrat_en

    dialogues = load_socrat_en(split="all")
    if max_dialogues is not None:
        dialogues = dialogues[:max_dialogues]

    for d in dialogues:
        dialogue_id = str(d["id"])
        # Walk paired user→assistant turns. Each user turn carries the marker
        # already (set by the loader); we recover state+action from the
        # ground_truth_states list and the original record format.
        turns = _reconstruct_turns(d)
        for turn_idx, turn in enumerate(turns):
            gt_teacher = turn["teacher"]
            state = turn.get("state", "") or ""
            action = turn.get("action", "") or ""
            if not (state and action and gt_teacher):
                continue
            prompt = _build_user_prompt(turn["student"], state, action)
            stage = state[0]

            for sub_source, fn in PERTURBATIONS:
                rejected = fn(turn, gt_teacher)
                if rejected is None or rejected == gt_teacher:
                    continue
                yield DPOPair(
                    pair_id=f"3-{dialogue_id}-{turn_idx}-{sub_source}",
                    source=3,
                    sub_source=sub_source,
                    dialogue_id=dialogue_id,
                    turn_idx=turn_idx,
                    stage=stage,
                    state=state,
                    prompt=prompt,
                    chosen=gt_teacher,
                    rejected=rejected,
                )


def _reconstruct_turns(record: dict) -> list[dict]:
    """Recover {student, teacher, state, action} per turn from a loaded record.

    The loader stores state inside the user-turn marker text and action just
    after it. Rather than parse those back out, we re-hit the HF dataset
    cache for the original record — cheap on the second call since hf_load
    caches to disk."""
    from datasets import load_dataset as hf_load

    if not hasattr(_reconstruct_turns, "_cache"):
        raw = hf_load("ulises-c/SocratDataset-EN", split="train")
        _reconstruct_turns._cache = {str(r["id"]): r for r in raw}  # type: ignore[attr-defined]
    cache = _reconstruct_turns._cache  # type: ignore[attr-defined]
    raw_rec = cache.get(record["id"])
    if not raw_rec:
        return []
    out = []
    for turn in raw_rec.get("dialogue", []) or []:
        out.append(
            {
                "student": turn.get("student", ""),
                "teacher": turn.get("teacher", ""),
                "state": turn.get("state", ""),
                "action": turn.get("action", ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Source 1: judge-mined (inert scaffold)
# ---------------------------------------------------------------------------


def build_source1_pairs(
    checkpoint: str | None,
    judge_threshold: float,
    k: int,
    max_judge_cost: float,
) -> Iterator[DPOPair]:
    """Generate K candidates per turn from Stage 2b, score with the 4-axis
    judge, keep pairs above the delta threshold. Inert until a Stage 2b
    checkpoint exists — full implementation lands in the next-next PR."""
    if not checkpoint:
        print("[Source 1] skipped — needs a Stage 2b checkpoint via --checkpoint", file=sys.stderr)
        return
        yield  # pragma: no cover  (signals to type checkers this is a generator)
    raise NotImplementedError(
        "Source 1 candidate generation + judge scoring not yet implemented. "
        "Wire this once Stage 2b training has produced a checkpoint."
    )


# ---------------------------------------------------------------------------
# Source 2: STL-as-rejected on synthetic (inert scaffold)
# ---------------------------------------------------------------------------


def build_source2_pairs(stl_dialogues_path: str | None) -> Iterator[DPOPair]:
    """Pair synthetic ground-truth teacher (chosen) against STL teacher on the
    same turn (rejected). Inert until per-turn STL synthetic dialogue logs
    are persisted — current STL synthetic runs save only aggregated summary
    JSONs (verified 2026-05-24)."""
    if not stl_dialogues_path:
        print(
            "[Source 2] skipped — needs --stl-results pointing at a directory "
            "with per-turn STL synthetic dialogue logs",
            file=sys.stderr,
        )
        return
        yield  # pragma: no cover
    if not Path(stl_dialogues_path).exists():
        print(f"[Source 2] skipped — path not found: {stl_dialogues_path}", file=sys.stderr)
        return
        yield  # pragma: no cover
    raise NotImplementedError(
        "Source 2 dialogue-level ingest not yet implemented. "
        "Capture STL responses on synthetic at per-turn granularity first."
    )


# ---------------------------------------------------------------------------
# Stratification, holdout, output
# ---------------------------------------------------------------------------


def stratify_by_stage(
    pairs: list[DPOPair], target: dict[str, float], max_pairs: int, seed: int = 42
) -> list[DPOPair]:
    """Downsample dominant stages toward the target distribution.

    With-replacement oversampling of rare stages is avoided in this PR — if a
    stage is short of its quota, we keep whatever exists and let the final
    count fall below max_pairs. Adding noisy duplicates is worse than a
    smaller balanced set."""
    by_stage: dict[str, list[DPOPair]] = defaultdict(list)
    for p in pairs:
        by_stage[p.stage].append(p)

    rng = random.Random(seed)
    out: list[DPOPair] = []
    for stage, quota_frac in target.items():
        quota = int(max_pairs * quota_frac)
        bucket = by_stage.get(stage, [])
        rng.shuffle(bucket)
        out.extend(bucket[:quota])
    rng.shuffle(out)
    return out


def assign_holdout(pairs: list[DPOPair], frac: float = 0.05, seed: int = 42) -> None:
    """Mark 5% of pairs as holdout, stratified by (stage, sub_source)."""
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[DPOPair]] = defaultdict(list)
    for p in pairs:
        by_stratum[(p.stage, p.sub_source)].append(p)
    for stratum, bucket in by_stratum.items():
        rng.shuffle(bucket)
        n_holdout = max(1, int(len(bucket) * frac)) if len(bucket) >= 20 else 0
        for p in bucket[:n_holdout]:
            p.holdout = True


def write_parquet(pairs: list[DPOPair], path: Path) -> None:
    """Write pairs to Parquet. TRL DPOTrainer ingests this via
    datasets.Dataset.from_parquet(path).filter(lambda r: not r['holdout'])."""
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit(
            "pandas is required for parquet output. Install with: uv add pandas pyarrow"
        ) from e
    df = pd.DataFrame([asdict(p) for p in pairs])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def print_pair_preview(pair: DPOPair) -> None:
    print(f"  pair_id={pair.pair_id}  stage={pair.stage}  sub={pair.sub_source}")
    print(f"    prompt: {pair.prompt[:120]!r}{'...' if len(pair.prompt) > 120 else ''}")
    print(f"    chosen: {pair.chosen[:120]!r}{'...' if len(pair.chosen) > 120 else ''}")
    print(f"    reject: {pair.rejected[:120]!r}{'...' if len(pair.rejected) > 120 else ''}")


# ---------------------------------------------------------------------------
# Dry-run + main
# ---------------------------------------------------------------------------


def dry_run() -> None:
    """Source 3 only, ~100 pairs, parquet write into a tempdir."""
    print("\n=== DRY RUN — Source 3 only, no API calls ===\n")
    pairs = list(build_source3_pairs(max_dialogues=20))
    print(f"Generated {len(pairs)} pairs from the first 20 SocratDataset-EN dialogues.")

    # Count by sub_source and stage
    sub_counter = Counter(p.sub_source for p in pairs)
    stage_counter = Counter(p.stage for p in pairs)
    print(f"\nSub-source distribution: {dict(sub_counter)}")
    print(f"Stage distribution:      {dict(stage_counter)}")

    print("\nFirst 3 pairs:")
    for p in pairs[:3]:
        print()
        print_pair_preview(p)

    assign_holdout(pairs, frac=0.05)
    n_holdout = sum(1 for p in pairs if p.holdout)
    print(f"\nHoldout assignment: {n_holdout}/{len(pairs)} pairs marked holdout=True.")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "dry_run_pairs.parquet"
        write_parquet(pairs, out_path)
        size_kb = out_path.stat().st_size / 1024
        print(f"\nParquet write OK: {out_path.name}  ({size_kb:.1f} KB on disk)")

    print("\n=== Dry run complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs for SocratTeachLLM Stage 3"
    )
    parser.add_argument(
        "--source",
        choices=["1", "2", "3", "all"],
        default="3",
        help="which source(s) to build (default: 3 — the only fully-functional one in this PR)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output Parquet path (e.g. data/dpo_pairs/source3.parquet)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200_000,
        help="cap on final pair count after stratification (default 200000)",
    )
    parser.add_argument(
        "--judge-threshold",
        type=float,
        default=2.0,
        help="Source 1: keep only pairs where judge(chosen) - judge(rejected) ≥ threshold",
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Source 1: candidates per turn (default 3)"
    )
    parser.add_argument(
        "--max-judge-cost",
        type=float,
        default=50.0,
        help="Source 1: abort if estimated judge cost exceeds (default $50)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Source 1: Stage 2b checkpoint path"
    )
    parser.add_argument(
        "--stl-results",
        type=str,
        default=None,
        help="Source 2: directory with per-turn STL synthetic dialogue logs",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.05,
        help="fraction held out for DPO eval (default 0.05)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true", help="Source 3 only, ~100 pairs, no API or large download"
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.output is None:
        print("--output PATH is required (unless --dry-run)", file=sys.stderr)
        sys.exit(2)

    all_pairs: list[DPOPair] = []
    if args.source in ("3", "all"):
        all_pairs.extend(build_source3_pairs())
    if args.source in ("2", "all"):
        all_pairs.extend(build_source2_pairs(args.stl_results))
    if args.source in ("1", "all"):
        all_pairs.extend(
            build_source1_pairs(
                args.checkpoint,
                args.judge_threshold,
                args.k,
                args.max_judge_cost,
            )
        )

    print(f"Built {len(all_pairs)} raw pairs across selected sources.")
    if not all_pairs:
        print("No pairs to write — exiting.", file=sys.stderr)
        sys.exit(1)

    stratified = stratify_by_stage(all_pairs, STAGE_TARGET, args.max_pairs, seed=args.seed)
    assign_holdout(stratified, frac=args.holdout_frac, seed=args.seed)

    print(f"After stratification + holdout: {len(stratified)} pairs.")
    print(f"  Holdout: {sum(1 for p in stratified if p.holdout)}")
    print(f"  Stages:  {dict(Counter(p.stage for p in stratified))}")
    print(f"  Sources: {dict(Counter(p.source for p in stratified))}")

    write_parquet(stratified, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
