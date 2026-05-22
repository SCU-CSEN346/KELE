"""Semantic similarity replacement for surface ROUGE-1.

Uses BAAI/bge-m3 (multilingual; handles both Chinese SocratDataset and English
SocratDataset-EN) to embed teacher_response and ground_truth_teacher, then
reports cosine similarity. Score range [0, 1] — higher is more semantically
similar to the ground truth, regardless of n-gram overlap.

This is a memorization-resistant replacement for surface R-1: a paraphrase
that says the same thing as the reference gets a high score even though
its n-gram overlap with the reference is low.

Usage:
  uv run python scripts/semantic_r1.py <results_dir>
  uv run python scripts/semantic_r1.py <results_dir1> <results_dir2> ...

Output: per-dir <results_dir>/semantic_r1.json with mean/median similarity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Semantic similarity replacement for ROUGE-1")
    parser.add_argument("results_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--model",
        default="BAAI/bge-m3",
        help="HF sentence-embedding model (default bge-m3, multilingual).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading embedding model: {args.model} ...", flush=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)

    import numpy as np

    for rdir in args.results_dirs:
        dialogues_dir = rdir / "dialogues"
        if not dialogues_dir.is_dir():
            print(f"  skip (no dialogues/): {rdir}", file=sys.stderr)
            continue

        files = sorted(dialogues_dir.glob("*.json"))
        if not files:
            print(f"  skip (no dialogue files): {rdir}", file=sys.stderr)
            continue

        print(f"\n=== {rdir} ({len(files)} dialogues) ===", flush=True)
        t0 = time.time()

        teacher_responses = []
        gt_responses = []
        gt_states = []
        for f in files:
            d = json.loads(f.read_text())
            for t in d.get("dialogue", []):
                tr = t.get("teacher_response", t.get("teacher", ""))
                gt = t.get("ground_truth_teacher", "")
                if not tr or not gt:
                    continue
                teacher_responses.append(tr)
                gt_responses.append(gt)
                gt_states.append(t.get("ground_truth_state", "?"))

        n = len(teacher_responses)
        print(f"  {n} turn pairs to embed", flush=True)

        if n == 0:
            continue

        # Embed both lists in batches; bge-m3 outputs are L2-normalized so
        # dot product = cosine sim.
        tr_emb = model.encode(
            teacher_responses, batch_size=args.batch_size, show_progress_bar=False,
            normalize_embeddings=True,
        )
        gt_emb = model.encode(
            gt_responses, batch_size=args.batch_size, show_progress_bar=False,
            normalize_embeddings=True,
        )
        sims = (tr_emb * gt_emb).sum(axis=1)  # element-wise cosine via dot of normed

        mean = float(sims.mean())
        median = float(np.median(sims))
        p10 = float(np.quantile(sims, 0.10))
        p90 = float(np.quantile(sims, 0.90))

        # Per-stage breakdown
        per_stage: dict[str, dict[str, float]] = {}
        for s in "abcde":
            mask = np.array([gs and gs[0] == s for gs in gt_states])
            if mask.any():
                per_stage[s] = {
                    "n": int(mask.sum()),
                    "mean": float(sims[mask].mean()),
                    "median": float(np.median(sims[mask])),
                }

        summary = {
            "results_dir": str(rdir),
            "embed_model": args.model,
            "n_pairs": n,
            "mean_cosine": mean,
            "median_cosine": median,
            "p10_cosine": p10,
            "p90_cosine": p90,
            "per_stage": per_stage,
            "wall_clock_seconds": time.time() - t0,
        }

        out_path = rdir / "semantic_r1.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(
            f"  mean cosine: {mean:.4f}  median: {median:.4f}  p10/p90: {p10:.4f}/{p90:.4f}"
            f"  ({time.time() - t0:.1f}s)",
            flush=True,
        )
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
