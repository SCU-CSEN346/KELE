#!/usr/bin/env python3
"""validate_translation.py — Validate SocratDataset-EN against the Chinese source.

Two-phase design: structural checks (no LLM) gate the LLM quality eval.

Usage:
    # Phase 1 only:
    uv run python -m src.project.validate_translation --structural-only

    # Phase 1 + Phase 2 (default 5% sample, single node):
    uv run python -m src.project.validate_translation --base-url http://localhost:8080/v1

    # Two-node parallel (run simultaneously on each Mac Mini):
    uv run python -m src.project.validate_translation --shard odd  --base-url http://localhost:8080/v1
    uv run python -m src.project.validate_translation --shard even --base-url http://localhost:8080/v1

    # Merge shard results:
    uv run python -m src.project.validate_translation \
        --merge data/validate_llm_scores_odd.json data/validate_llm_scores_even.json
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL: str = "http://localhost:8080/v1"
MODEL: str = os.path.expanduser("~/.models/mlx-community--Qwen3.5-9B-4bit")

SAMPLE_SIZE: float = 0.05
SAMPLE_SEED: int = 42
THINKING_BUDGET: int = 0

ZH_HF_REPO: str = "ulises-c/SocratDataset"
EN_HF_REPO: str = "ulises-c/SocratDataset-EN"

OUTPUT_DIR: str = "data"
CHECKPOINT_EVERY: int = 10

SCORE_FLAG_THRESHOLD: int = 3  # overall_score < this → flag record
FLAG_RATE_THRESHOLD: float = 0.10  # flag rate above this → surface for review

# ── Helpers ───────────────────────────────────────────────────────────────────

_CHINESE_RE = re.compile(r"[一-鿿]")
_ROW_FIELDS = ["question", "newHint", "newKnowledgePoint", "newAnalyze"]
_TURN_FIELDS = ["student", "teacher", "evaluation"]


def _extra() -> dict:
    if THINKING_BUDGET == 0:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {"chat_template_kwargs": {"thinking_budget": THINKING_BUDGET}}


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    return re.sub(r"\s*```$", "", text)


def _parse_json(text: str) -> dict:
    text = _strip_fences(text)
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj
    except json.JSONDecodeError:
        pass
    for suffix in ('"]}', "]}", "}"):
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("unrepairable JSON", text, 0)


# ── Phase 1 — Structural checks ───────────────────────────────────────────────


@dataclass
class StructuralResult:
    passed: bool
    failures: dict[str, object] = field(default_factory=dict)

    def _fail_count(self, check: str) -> int:
        v = self.failures.get(check)
        if v is None:
            return 0
        if isinstance(v, dict):
            return sum(len(x) for x in v.values())
        return len(v)  # type: ignore[arg-type]

    def report(self) -> None:
        checks = [
            "id_coverage",
            "field_completeness",
            "state_preservation",
            "option_count",
            "dialogue_round_count",
            "no_chinese_in_en",
        ]
        width = max(len(c) for c in checks)
        print("\n── Phase 1: Structural checks ──")
        for check in checks:
            n = self._fail_count(check)
            status = "PASS" if n == 0 else f"FAIL ({n} records)"
            print(f"  {check:<{width}}  {status}")
        print()
        if self.passed:
            print("  All checks passed.\n")
        else:
            print("  Failures saved to data/validate_structural_failures.json\n")


def _has_chinese_in_en(rec: dict) -> bool:
    for f in _ROW_FIELDS:
        if _CHINESE_RE.search(str(rec.get(f, ""))):
            return True
    for opt in rec.get("options", []):
        if _CHINESE_RE.search(str(opt)):
            return True
    for turn in rec.get("dialogue", []):
        for tf in _TURN_FIELDS:
            if _CHINESE_RE.search(str(turn.get(tf, ""))):
                return True
        if _CHINESE_RE.search(str(turn.get("action", ""))):
            return True
    return False


def run_structural_checks(zh_records: list[dict], en_by_id: dict[int, dict]) -> StructuralResult:
    failures: dict[str, object] = {}
    lists: dict[str, list[int]] = defaultdict(list)

    zh_ids = {r["id"] for r in zh_records}
    en_ids = set(en_by_id.keys())
    missing = sorted(zh_ids - en_ids)
    surplus = sorted(en_ids - zh_ids)
    if missing or surplus:
        failures["id_coverage"] = {"missing_in_en": missing, "extra_in_en": surplus}

    for zh in zh_records:
        rid = zh["id"]
        en = en_by_id.get(rid)
        if en is None:
            continue

        for f in _ROW_FIELDS:
            if not en.get(f):
                lists["field_completeness"].append(rid)
                break
        else:
            for turn in en.get("dialogue", []):
                if not all(turn.get(f) for f in _TURN_FIELDS):
                    lists["field_completeness"].append(rid)
                    break

        zh_states = [t.get("state") for t in zh.get("dialogue", [])]
        en_states = [t.get("state") for t in en.get("dialogue", [])]
        if zh_states != en_states:
            lists["state_preservation"].append(rid)

        if len(zh.get("options", [])) != len(en.get("options", [])):
            lists["option_count"].append(rid)

        if len(zh.get("dialogue", [])) != len(en.get("dialogue", [])):
            lists["dialogue_round_count"].append(rid)

        if _has_chinese_in_en(en):
            lists["no_chinese_in_en"].append(rid)

    failures.update(lists)
    passed = not failures
    return StructuralResult(passed=passed, failures=failures)


# ── Phase 2 — LLM quality eval ────────────────────────────────────────────────

_EVAL_SYSTEM = """\
You are a professional translator evaluating Chinese-to-English translation quality \
for elementary school science tutoring dialogues.

Given a Chinese source record and its English translation, return ONLY valid JSON:
{
  "overall_score": <int 1-5>,
  "meaning_preserved": <bool>,
  "socratic_tone_preserved": <bool>,
  "fluency": <int 1-5>,
  "flags": [<string>, ...]
}

Scoring: 1=very poor, 3=acceptable, 5=excellent.
meaning_preserved: factual content, answer options, and correct answer are preserved.
socratic_tone_preserved: teacher's guiding, questioning style is maintained.
fluency: 1=awkward, 5=native-sounding English.
flags: brief notes on specific issues; empty list if none.\
"""

_RETRY_REMINDER = (
    "\n\nIMPORTANT: return ONLY valid JSON matching the schema. No markdown fences, no commentary."
)


def _build_eval_prompt(zh: dict, en: dict) -> str:
    def _fmt(rec: dict, lang: str) -> str:
        lines = [f"=== {lang} ==="]
        lines.append(f"question: {rec.get('question', '')}")
        lines.append(f"options: {rec.get('options', [])}")
        lines.append(f"newHint: {rec.get('newHint', '')}")
        lines.append(f"newKnowledgePoint: {rec.get('newKnowledgePoint', '')}")
        lines.append(f"newAnalyze: {rec.get('newAnalyze', '')}")
        lines.append("dialogue:")
        for i, turn in enumerate(rec.get("dialogue", []), 1):
            lines.append(f"  [{i}] student:    {turn.get('student', '')}")
            lines.append(f"      teacher:    {turn.get('teacher', '')}")
            lines.append(f"      evaluation: {turn.get('evaluation', '')}")
        return "\n".join(lines)

    return _fmt(zh, "SOURCE (Chinese)") + "\n\n" + _fmt(en, "TRANSLATION (English)")


def _validate_eval_result(result: dict) -> dict:
    required = {"overall_score", "meaning_preserved", "socratic_tone_preserved", "fluency", "flags"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"missing keys: {missing}")
    overall = result["overall_score"]
    if isinstance(overall, bool) or not isinstance(overall, int) or not 1 <= overall <= 5:
        raise ValueError(f"overall_score must be int 1-5, got {overall!r}")
    fluency = result["fluency"]
    if isinstance(fluency, bool) or not isinstance(fluency, int) or not 1 <= fluency <= 5:
        raise ValueError(f"fluency must be int 1-5, got {fluency!r}")
    if not isinstance(result["meaning_preserved"], bool):
        raise ValueError(f"meaning_preserved must be bool, got {result['meaning_preserved']!r}")
    if not isinstance(result["socratic_tone_preserved"], bool):
        raise ValueError(
            f"socratic_tone_preserved must be bool, got {result['socratic_tone_preserved']!r}"
        )
    if not isinstance(result["flags"], list):
        raise ValueError(f"flags must be list, got {type(result['flags']).__name__}")
    return result


def eval_record(client: OpenAI, model: str, zh: dict, en: dict, retries: int = 3) -> dict:
    prompt = _build_eval_prompt(zh, en)
    reminder = ""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _EVAL_SYSTEM},
                    {"role": "user", "content": prompt + reminder},
                ],
                temperature=0.1,
                max_tokens=512,
                extra_body=_extra(),
            )
            result = _parse_json(resp.choices[0].message.content or "")
            return _validate_eval_result(result)
        except Exception as e:
            last_err = e
            reminder = _RETRY_REMINDER
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"id={zh['id']} eval failed after {retries} attempts: {last_err}")


def _is_flagged(score: dict) -> bool:
    return (
        score.get("overall_score", 5) < SCORE_FLAG_THRESHOLD
        or not score.get("meaning_preserved", True)
        or not score.get("socratic_tone_preserved", True)
    )


# ── Sampling ──────────────────────────────────────────────────────────────────


def stratified_sample(records: list[dict], fraction: float, seed: int, shard: str) -> list[dict]:
    """Sample proportionally from each mission×grade stratum within the shard."""
    if shard == "odd":
        records = [r for r in records if r["id"] % 2 == 1]
    elif shard == "even":
        records = [r for r in records if r["id"] % 2 == 0]

    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        strata[(r.get("mission", ""), r.get("grade", ""))].append(r)

    sample: list[dict] = []
    for _, group in sorted(strata.items()):
        n = max(1, round(len(group) * fraction))
        sample.extend(rng.sample(group, min(n, len(group))))

    return sample


# ── Checkpoint I/O ────────────────────────────────────────────────────────────


def _ckpt_path(output_dir: Path, shard: str) -> Path:
    return output_dir / f"validate_llm_checkpoint_{shard}.json"


def _load_checkpoint(path: Path) -> tuple[set[int], list[dict]]:
    if path.exists():
        state = json.loads(path.read_text())
        return set(state["processed_ids"]), state["results"]
    return set(), []


def _save_checkpoint(path: Path, processed_ids: set[int], results: list[dict]) -> None:
    path.write_text(json.dumps({"processed_ids": list(processed_ids), "results": results}))


# ── Merge ─────────────────────────────────────────────────────────────────────


def merge_shards(paths: list[str], output_dir: Path) -> None:
    all_scores: list[dict] = []
    for p in paths:
        all_scores.extend(json.loads(Path(p).read_text()))

    all_scores.sort(key=lambda r: r["id"])

    scores_path = output_dir / "validate_llm_scores_all.json"
    scores_path.write_text(json.dumps(all_scores, indent=2))

    flagged = [s for s in all_scores if _is_flagged(s["score"])]
    flagged_path = output_dir / "validate_llm_flagged_all.json"
    flagged_path.write_text(json.dumps(flagged, indent=2, ensure_ascii=False))

    _print_summary(all_scores, flagged)
    print(f"\nMerged {len(all_scores)} records → {scores_path}")


# ── Summary ───────────────────────────────────────────────────────────────────


def _print_summary(scores: list[dict], flagged: list[dict]) -> None:
    n = len(scores)
    if n == 0:
        print("No scores to summarise.")
        return

    overall = [s["score"]["overall_score"] for s in scores]
    fluency = [s["score"]["fluency"] for s in scores]
    meaning = sum(1 for s in scores if s["score"].get("meaning_preserved", True))
    tone = sum(1 for s in scores if s["score"].get("socratic_tone_preserved", True))
    flag_rate = len(flagged) / n

    print(f"\n── Phase 2 summary ({n} records) ──")
    print(
        f"  overall_score  avg={sum(overall) / n:.2f}  dist={dict(sorted((k, overall.count(k)) for k in set(overall)))}"
    )
    print(f"  fluency        avg={sum(fluency) / n:.2f}")
    print(f"  meaning_preserved       {meaning}/{n} ({meaning / n:.0%})")
    print(f"  socratic_tone_preserved {tone}/{n} ({tone / n:.0%})")
    print(
        f"  flag_rate      {flag_rate:.1%}  ({'⚠ above threshold' if flag_rate > FLAG_RATE_THRESHOLD else 'ok'})"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SocratDataset-EN translation quality")
    parser.add_argument("--structural-only", action="store_true")
    parser.add_argument("--shard", choices=["all", "odd", "even"], default="all")
    parser.add_argument(
        "--sample",
        type=float,
        default=SAMPLE_SIZE,
        metavar="FRAC",
        help="Fraction of dataset to evaluate (default: %(default)s)",
    )
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--api-key", default="no-key")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--merge", nargs="+", metavar="PATH", help="Merge shard result JSON files and print summary"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_shards(args.merge, output_dir)
        return

    print("Loading datasets from HuggingFace…")
    from datasets import load_dataset as _load  # type: ignore

    zh_records: list[dict] = _load(ZH_HF_REPO, split="train").to_list()
    en_records: list[dict] = _load(EN_HF_REPO, split="train").to_list()
    en_by_id: dict[int, dict] = {r["id"]: r for r in en_records}
    print(f"  ZH: {len(zh_records)} records  EN: {len(en_records)} records")

    # Phase 1
    result = run_structural_checks(zh_records, en_by_id)
    result.report()

    failures_path = output_dir / "validate_structural_failures.json"
    if result.failures:
        failures_path.write_text(json.dumps(result.failures, indent=2))

    if not result.passed:
        print("Phase 1 failed — fix structural issues before running Phase 2.", file=sys.stderr)
        sys.exit(1)

    if args.structural_only:
        return

    # Phase 2
    sample = stratified_sample(zh_records, args.sample, SAMPLE_SEED, args.shard)
    print("\n── Phase 2: LLM quality eval ──")
    print(f"  shard={args.shard}  sample={args.sample:.0%}  records={len(sample)}")
    print(f"  model={args.model}\n")

    ckpt = _ckpt_path(output_dir, args.shard)
    processed_ids, llm_results = _load_checkpoint(ckpt)
    initial_done = len(processed_ids)
    if initial_done:
        print(f"  Resuming from checkpoint: {initial_done} already done")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    start = time.time()
    errors = 0

    for zh in sample:
        if zh["id"] in processed_ids:
            continue
        en = en_by_id[zh["id"]]
        try:
            score = eval_record(client, args.model, zh, en)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            errors += 1
            continue

        llm_results.append(
            {"id": zh["id"], "grade": zh.get("grade"), "mission": zh.get("mission"), "score": score}
        )
        processed_ids.add(zh["id"])

        done = len(processed_ids)
        new_done = done - initial_done
        elapsed = time.time() - start
        rate = new_done / elapsed if elapsed > 0 and new_done > 0 else 0
        eta = (len(sample) - done) / rate / 60 if rate > 0 else float("inf")
        eta_str = f"{eta:.0f}m" if eta < 60 else f"{eta / 60:.1f}h"
        flagged_str = "⚑" if _is_flagged(score) else " "
        print(
            f"[{done:>4}/{len(sample)}] id={zh['id']:>5}  score={score['overall_score']}  {flagged_str}  ETA {eta_str}"
        )

        if new_done % CHECKPOINT_EVERY == 0:
            _save_checkpoint(ckpt, processed_ids, llm_results)

    _save_checkpoint(ckpt, processed_ids, llm_results)

    suffix = args.shard
    scores_path = output_dir / f"validate_llm_scores_{suffix}.json"
    scores_path.write_text(json.dumps(llm_results, indent=2))

    flagged = [r for r in llm_results if _is_flagged(r["score"])]
    flagged_path = output_dir / f"validate_llm_flagged_{suffix}.json"
    flagged_path.write_text(json.dumps(flagged, indent=2, ensure_ascii=False))

    _print_summary(llm_results, flagged)
    print(f"\nScores → {scores_path}  ({errors} errors)")
    if flagged:
        print(f"Flagged → {flagged_path}")


if __name__ == "__main__":
    main()
