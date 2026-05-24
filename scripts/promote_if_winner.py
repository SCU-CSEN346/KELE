#!/usr/bin/env python3
"""Read the 4 Qwen-27B grid metrics, pick the winner, decide promotion.

Composite = state_accuracy + 0.5 * rouge1  (same formula used by
scripts/aggregate_tournament_leaderboard.py).

Promotion threshold: locked BERT-integration headline composite at n=681 is
48.15 + 0.5 * 36.78 = 66.54. We promote to a full n=681 run only if the
best n=50 cell beats locked by at least +2 composite points (mini→full
typically loses ~1-2 pts, so +2 at n=50 ≈ break-even at n=681).

Writes results/_orchestrator_logs/promote_decision.json + prints a single
sourceable env-var block to stdout for the orchestrator to consume.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LOCKED_COMPOSITE = 48.15 + 0.5 * 36.78  # 66.54
PROMOTE_DELTA = 2.0
RESULTS = Path("results")

# Cell -> (script, consultant, teacher_mode, bert_ckpt)
CELLS = {
    "bge-small-bert-qwen27b-fewshot10-n50-fixed": (
        "eval_bert_qwen27b_fewshot10_full.sh",
        "bge-small",
        "think",
        "results/state_classifier_v1/final",
    ),
    "t4-bert-qwen27b-fewshot10-n50-fixed": (
        "eval_bert_qwen27b_fewshot10_full.sh",
        "t4",
        "think",
        "results/state-clf-qwen3.5-0.8b-lora/final",
    ),
    "bge-small-bert-qwen27b-nothink-fewshot10-n50-fixed": (
        "eval_bert_qwen27b_nothink_fewshot10_full.sh",
        "bge-small",
        "nothink",
        "results/state_classifier_v1/final",
    ),
    "t4-bert-qwen27b-nothink-fewshot10-n50-fixed": (
        "eval_bert_qwen27b_nothink_fewshot10_full.sh",
        "t4",
        "nothink",
        "results/state-clf-qwen3.5-0.8b-lora/final",
    ),
}


def composite(metrics: dict) -> float:
    state = metrics.get("state_accuracy", {}).get("overall", 0.0)
    r1 = metrics.get("rouge1", 0.0)
    return state + 0.5 * r1


def main() -> int:
    summary = []
    for cell, (script, consultant, mode, ckpt) in CELLS.items():
        mfile = RESULTS / cell / "metrics_summary.json"
        if not mfile.exists():
            summary.append({"cell": cell, "status": "MISSING", "metrics_path": str(mfile)})
            continue
        m = json.loads(mfile.read_text())
        c = composite(m)
        summary.append(
            {
                "cell": cell,
                "status": "OK",
                "script": script,
                "consultant": consultant,
                "teacher_mode": mode,
                "bert_ckpt": ckpt,
                "state_accuracy": m["state_accuracy"]["overall"],
                "rouge1": m["rouge1"],
                "composite": round(c, 2),
                "delta_vs_locked": round(c - LOCKED_COMPOSITE, 2),
            }
        )

    ok = [s for s in summary if s["status"] == "OK"]
    if not ok:
        print("ERROR: no Qwen-27B cells have metrics_summary.json yet", file=sys.stderr)
        return 2

    winner = max(ok, key=lambda s: s["composite"])
    promote = winner["composite"] >= LOCKED_COMPOSITE + PROMOTE_DELTA

    decision = {
        "locked_composite": round(LOCKED_COMPOSITE, 2),
        "promote_threshold": round(LOCKED_COMPOSITE + PROMOTE_DELTA, 2),
        "winner": winner,
        "promote": promote,
        "cells_evaluated": len(ok),
        "cells_missing": len(summary) - len(ok),
        "all": summary,
    }

    out_dir = RESULTS / "_orchestrator_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "promote_decision.json"
    out_file.write_text(json.dumps(decision, indent=2))

    # Human-readable summary to stderr
    print("=" * 60, file=sys.stderr)
    print(f"Qwen-27B grid decision (locked composite = {LOCKED_COMPOSITE:.2f})", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for s in sorted(ok, key=lambda x: -x["composite"]):
        marker = "★" if s["cell"] == winner["cell"] else " "
        print(
            f"  {marker} {s['cell']:<60s}  "
            f"state={s['state_accuracy']:5.2f}  R1={s['rouge1']:5.2f}  "
            f"composite={s['composite']:5.2f}  Δ={s['delta_vs_locked']:+5.2f}",
            file=sys.stderr,
        )
    print(
        f"  Decision: {'PROMOTE to n=681' if promote else 'DEFER to Layer-2 n=400'}",
        file=sys.stderr,
    )
    print(f"  Written: {out_file}", file=sys.stderr)
    print(file=sys.stderr)

    # Machine-readable env vars to stdout (orchestrator does `eval $(promote_if_winner.py)`)
    print(f"WINNER_CELL={winner['cell']}")
    print(f"WINNER_SCRIPT={winner['script']}")
    print(f"WINNER_CONSULTANT={winner['consultant']}")
    print(f"WINNER_TEACHER_MODE={winner['teacher_mode']}")
    print(f"WINNER_BERT_CKPT={winner['bert_ckpt']}")
    print(f"WINNER_COMPOSITE={winner['composite']}")
    print(f"PROMOTE={'true' if promote else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
