"""
Convergence analysis — at what n does sample-size noise become negligible?

For each n=681 run, we ask: if you had only chosen N random dialogues from the
test split, how close would your metric estimate be to the n=681 truth?

Method:
1. Load all dialogue JSONs for the run.
2. Precompute per-turn signals (state-correctness, ROUGE-1/2/L F1, sentence-BLEU-4)
   so bootstrap iterations become O(n) array operations, not O(n) re-scoring.
3. For each n in {grid}, draw B random subsamples of n DIALOGUES (without
   replacement), pool the constituent turns, and aggregate the per-turn signals
   into n-sized metric estimates.
4. Report bootstrap mean + std + |deviation| from the truth at each n.
5. Sweep across runs; find the smallest n where ALL configs satisfy a
   tolerance threshold (e.g., 95% of bootstraps within ±X of the full-n truth).

Output:
  results/convergence/<run>/convergence.json
  results/convergence/aggregate.json
  docs/figures/convergence_curves.{pdf,png}
  docs/figures/convergence_summary.{pdf,png}

Note on BLEU: the canonical metric in metrics.py is sacrebleu's *corpus* BLEU-4,
which is not a per-turn average. For convergence analysis we use the per-turn
sentence-BLEU-4 average as a fast proxy with the same convergence behaviour up
to a constant offset. The deviation curves are interpreted relative to the
sentence-BLEU truth at full n, NOT the corpus-BLEU reported number.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

# ── Repo paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project.metrics import _ZhCharTokenizer  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
RUNS: list[tuple[str, str]] = [
    ("GPT-4o + SocratTeachLLM (baseline)", "results/baseline"),
    ("A3B fusion-think (prior locked)", "results/qwen35b-a3b-local-unified"),
    ("Gemma 31B standalone (retracted)", "results/gemma4-31b-local-unified"),
    ("BERT + Gemma + 10-shot (LOCKED)", "results/bert-consultant-fewshot10-gemma-full"),
    ("BERT + A3B + 10-shot", "results/bert-consultant-fewshot10-a3b-full"),
    ("Sonnet + BERT + top-3", "results/bert-claude-sonnet-top3-n681"),
    ("Opus + BERT + top-3", "results/bert-claude-opus-top3-n681"),
]

# n-grid: dense at small n where convergence is fast, coarser at large n
N_GRID: list[int] = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600]

# Bootstrap config
B = 500
RNG_SEED = 42

# Tolerance candidates (we'll pick the right one after looking at the data)
TOLERANCES: dict[str, list[float]] = {
    "state_acc": [0.5, 1.0, 2.0],  # percentage points
    "rouge1": [0.5, 1.0, 2.0],
    "rouge2": [0.5, 1.0, 2.0],
    "bleu4_sent": [0.25, 0.5, 1.0],
}


# ── Per-turn precomputation ──────────────────────────────────────────────────
def load_run(
    run_dir: Path,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all dialogues; return per-turn signals + dialogue_id-per-turn array.

    Returns
    -------
    dialogue_ids   : sorted list of unique dialogue IDs (length D)
    turn_dlg_idx   : (T,) int array — dialogue index for each turn (into dialogue_ids)
    state_correct  : (T,) bool array
    rouge1_per     : (T,) float — per-turn ROUGE-1 F1
    rouge2_per     : (T,) float
    bleu4_per      : (T,) float — sentence-level BLEU-4
    """
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2"], use_stemmer=False, tokenizer=_ZhCharTokenizer()
    )
    bleu_sent = BLEU(effective_order=True, tokenize="zh")

    dialogue_ids: list[int] = []
    turn_dlg_idx: list[int] = []
    state_correct: list[bool] = []
    rouge1_per: list[float] = []
    rouge2_per: list[float] = []
    bleu4_per: list[float] = []

    files = sorted(run_dir.glob("dialogues/*.json"))
    for f in files:
        data = json.loads(f.read_text())
        if "error" in data:
            continue
        dlg_id = int(f.stem)
        dlg_idx = len(dialogue_ids)
        dialogue_ids.append(dlg_id)
        for turn in data.get("dialogue", []):
            gt_state = turn.get("ground_truth_state", "")
            pred_state = turn.get("state", "")
            pred_resp = turn.get("teacher_response", "")
            gt_resp = turn.get("ground_truth_teacher", "")
            if not gt_state or not pred_resp or not gt_resp:
                continue
            turn_dlg_idx.append(dlg_idx)
            state_correct.append(pred_state == gt_state)

            r = rouge.score(gt_resp, pred_resp)
            rouge1_per.append(r["rouge1"].fmeasure)
            rouge2_per.append(r["rouge2"].fmeasure)

            # Sentence-level BLEU-4 (single ref, single hyp)
            try:
                s = bleu_sent.sentence_score(pred_resp, [gt_resp]).score
            except Exception:
                s = 0.0
            bleu4_per.append(s)

    return (
        dialogue_ids,
        np.asarray(turn_dlg_idx, dtype=np.int32),
        np.asarray(state_correct, dtype=bool),
        np.asarray(rouge1_per, dtype=np.float32) * 100,
        np.asarray(rouge2_per, dtype=np.float32) * 100,
        np.asarray(bleu4_per, dtype=np.float32),
    )


# ── Bootstrap ────────────────────────────────────────────────────────────────
def bootstrap_run(
    label: str,
    turn_dlg_idx: np.ndarray,
    state_correct: np.ndarray,
    rouge1_per: np.ndarray,
    rouge2_per: np.ndarray,
    bleu4_per: np.ndarray,
    n_grid: Iterable[int],
    b: int = B,
    seed: int = RNG_SEED,
) -> dict:
    """For each n in n_grid, draw b random dialogue subsamples (without replacement)
    and aggregate per-turn signals into metric estimates."""
    rng = np.random.default_rng(seed)
    n_dialogues = int(turn_dlg_idx.max()) + 1

    # Truth at full n
    truth = {
        "state_acc": float(state_correct.mean() * 100),
        "rouge1": float(rouge1_per.mean()),
        "rouge2": float(rouge2_per.mean()),
        "bleu4_sent": float(bleu4_per.mean()),
    }

    # Precompute: for fast "which turns belong to which sampled dialogues" we
    # use the sorted turn_dlg_idx → an index where turns of each dialogue live
    # in contiguous slabs (the files were enumerated in dialogue order).
    # Build a boundary table: starts[d], ends[d] = slice of turns for dialogue d.
    starts = np.searchsorted(turn_dlg_idx, np.arange(n_dialogues), side="left")
    ends = np.searchsorted(turn_dlg_idx, np.arange(n_dialogues), side="right")

    per_n: dict[int, dict] = {}
    for n in n_grid:
        n = min(n, n_dialogues)
        boot_metrics = {k: np.zeros(b, dtype=np.float64) for k in truth}
        for i in range(b):
            sampled = rng.choice(n_dialogues, size=n, replace=False)
            # Build mask of which turns to include
            slab_starts = starts[sampled]
            slab_ends = ends[sampled]
            # Flatten ranges
            turn_indices_list = [
                np.arange(s, e, dtype=np.int64) for s, e in zip(slab_starts, slab_ends)
            ]
            if not turn_indices_list:
                continue
            turn_idx = np.concatenate(turn_indices_list)
            boot_metrics["state_acc"][i] = state_correct[turn_idx].mean() * 100
            boot_metrics["rouge1"][i] = rouge1_per[turn_idx].mean()
            boot_metrics["rouge2"][i] = rouge2_per[turn_idx].mean()
            boot_metrics["bleu4_sent"][i] = bleu4_per[turn_idx].mean()

        # Summary
        summary = {}
        for metric, vals in boot_metrics.items():
            tval = truth[metric]
            dev = vals - tval
            summary[metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "p2_5": float(np.percentile(vals, 2.5)),
                "p97_5": float(np.percentile(vals, 97.5)),
                "abs_dev_mean": float(np.abs(dev).mean()),
                "abs_dev_p95": float(np.percentile(np.abs(dev), 95)),
            }
        per_n[n] = summary

    return {"label": label, "truth": truth, "n_dialogues": n_dialogues, "per_n": per_n}


# ── Driver ────────────────────────────────────────────────────────────────────
def _process_one(args):
    label, rel_path = args
    run_dir = ROOT / rel_path
    print(f"[{label}] loading {run_dir.name}...", flush=True)
    (dlg_ids, turn_dlg_idx, sc, r1, r2, b4) = load_run(run_dir)
    print(
        f"[{label}] loaded D={len(dlg_ids)} dialogues, T={sc.size} turns; bootstrapping...",
        flush=True,
    )
    out = bootstrap_run(label, turn_dlg_idx, sc, r1, r2, b4, N_GRID)
    print(f"[{label}] done.", flush=True)
    return rel_path, out


def main() -> None:
    out_dir = ROOT / "results" / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(7, len(RUNS))) as ex:
        futures = {ex.submit(_process_one, r): r for r in RUNS}
        for fut in as_completed(futures):
            rel_path, out = fut.result()
            results[rel_path] = out
            # Per-run dump
            run_out_dir = out_dir / Path(rel_path).name
            run_out_dir.mkdir(parents=True, exist_ok=True)
            (run_out_dir / "convergence.json").write_text(json.dumps(out, indent=2))

    # Aggregate dump
    (out_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "n_grid": N_GRID,
                "B": B,
                "rng_seed": RNG_SEED,
                "runs": results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_dir / 'aggregate.json'}")
    print(f"Per-run files in {out_dir}/")


if __name__ == "__main__":
    main()
