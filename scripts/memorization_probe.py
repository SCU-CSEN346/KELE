"""Per-turn ROUGE-1 distribution analysis for SocratTeachLLM contamination probe.

Computes character-level ROUGE-1 between each generated teacher response and
its ground-truth counterpart, then reports the distribution shape (mean, p50,
p90, p99, max), the exact-match count, the near-verbatim count (ROUGE >= 80),
and the high-overlap count (ROUGE >= 60).

Usage:
    uv run python scripts/memorization_probe.py <results_dir_1> [<results_dir_2> ...]

The smoking-gun pattern for benchmark contamination is:
    - A heavy right tail (ROUGE > 80 turns at non-negligible rate)
    - Exact matches (ROUGE = 100)
    - Train/test distributions that are statistically identical

A clean (uncontaminated) model on a never-seen benchmark shows:
    - Smooth distribution concentrated around the mean
    - Max well below 80
    - Zero exact matches
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def char_rouge1(hyp: str, ref: str) -> float:
    """Character-level ROUGE-1 F1 (Chinese-friendly; matches kele.py tokenization)."""
    hyp_chars = list(hyp.strip())
    ref_chars = list(ref.strip())
    if not hyp_chars or not ref_chars:
        return 0.0
    h = Counter(hyp_chars)
    r = Counter(ref_chars)
    overlap = sum((h & r).values())
    p = overlap / len(hyp_chars)
    rec = overlap / len(ref_chars)
    if p + rec == 0:
        return 0.0
    return 2 * p * rec / (p + rec) * 100


def per_turn_rouges(results_dir: str | Path) -> tuple[list[float], int, int, int]:
    rouges: list[float] = []
    exact = 0
    near_verbatim = 0
    high = 0
    for fp in sorted(glob.glob(f"{results_dir}/dialogues/*.json")):
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)
        if "dialogue" not in d:
            continue
        for turn in d["dialogue"]:
            hyp = turn.get("teacher_response", "").strip()
            ref = turn.get("ground_truth_teacher", "").strip()
            if not hyp or not ref:
                continue
            r = char_rouge1(hyp, ref)
            rouges.append(r)
            if hyp == ref:
                exact += 1
            if r >= 80:
                near_verbatim += 1
            if r >= 60:
                high += 1
    return rouges, exact, near_verbatim, high


def report(results_dir: str | Path) -> None:
    rouges, exact, nv, ho = per_turn_rouges(results_dir)
    if not rouges:
        print(f"{results_dir}: NO DATA")
        return
    rouges.sort()
    n = len(rouges)
    mean = sum(rouges) / n
    print(f"=== {Path(results_dir).name} ===")
    print(
        f"  n_turns={n}  mean={mean:.2f}  p50={rouges[n // 2]:.2f}  "
        f"p90={rouges[int(n * 0.90)]:.2f}  p99={rouges[min(int(n * 0.99), n - 1)]:.2f}  "
        f"max={rouges[-1]:.2f}"
    )
    print(f"  EXACT MATCHES:                {exact:4d}  ({100 * exact / n:5.1f}%)")
    print(f"  NEAR-VERBATIM (rouge1 >= 80): {nv:4d}  ({100 * nv / n:5.1f}%)")
    print(f"  HIGH-OVERLAP  (rouge1 >= 60): {ho:4d}  ({100 * ho / n:5.1f}%)")
    buckets = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]
    for lo, hi in zip(buckets, buckets[1:]):
        n_b = sum(1 for x in rouges if lo <= x < hi)
        bar = "#" * int(50 * n_b / n)
        print(f"  [{lo:3.0f}-{hi:3.0f})  {n_b:4d} ({100 * n_b / n:5.1f}%) {bar}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="One or more results directories with dialogues/")
    args = p.parse_args()
    for d in args.dirs:
        report(d)


if __name__ == "__main__":
    main()
