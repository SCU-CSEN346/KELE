"""Aggregate Phase 1 tournament cell results into a ranked leaderboard.

Reads metrics_summary.json from each results/tournament-cell-*/ directory and
prints a leaderboard ranked by the composite score `state_acc + 0.5 * R-1`
(per docs/PROMPT_ENGINEERING_PLAN.md §4, Phase 2 selection rule).

Usage:
  uv run python scripts/aggregate_tournament_leaderboard.py
  uv run python scripts/aggregate_tournament_leaderboard.py results/tournament-cell-*

The Phase 0 reference (BERT + Gemma + 10-shot at n=50) is included as the
baseline row whenever it is on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Phase 0 reference (BERT + Gemma + 10-shot at n=50, locked baseline).
PHASE_0_REFERENCE_PATH = Path("results/bert-consultant-fewshot10-gemma-n50/metrics_summary.json")


def load_metrics(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def parse_cell_label(cell_dir: Path) -> tuple[str, str]:
    """Extract cell number and name from a directory like
    `results/tournament-cell-9-persona/` → ('9', 'persona')."""
    name = cell_dir.name
    parts = name.split("-")
    # Expected: 'tournament-cell-<NUM>-<NAME...>'
    if len(parts) >= 4 and parts[0] == "tournament" and parts[1] == "cell":
        num = parts[2]
        label = "-".join(parts[3:])
        return num, label
    return "?", name


def composite_score(m: dict) -> float:
    state = m.get("state_accuracy", {}).get("overall", 0.0)
    r1 = m.get("rouge1", 0.0)
    return state + 0.5 * r1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cell_dirs",
        nargs="*",
        type=Path,
        help="Tournament cell directories (default: auto-discover results/tournament-cell-*)",
    )
    args = parser.parse_args()

    if args.cell_dirs:
        cell_dirs = args.cell_dirs
    else:
        cell_dirs = sorted(Path("results").glob("tournament-cell-*"))

    if not cell_dirs:
        print("No tournament cells found. Run scripts/eval_prompt_tournament.sh first.")
        return 1

    rows: list[dict] = []

    # Phase 0 baseline as anchor row if present
    base = load_metrics(PHASE_0_REFERENCE_PATH)
    if base:
        rows.append(
            {
                "rank_label": "ref",
                "cell": "0",
                "name": "baseline (BERT + Gemma + 10-shot, n=50)",
                "metrics": base,
            }
        )

    for d in cell_dirs:
        m = load_metrics(d / "metrics_summary.json")
        if not m:
            continue
        num, label = parse_cell_label(d)
        rows.append({"cell": num, "name": label, "metrics": m})

    # Rank everything except the reference by composite score
    rankable = [r for r in rows if r.get("rank_label") != "ref"]
    rankable.sort(key=lambda r: composite_score(r["metrics"]), reverse=True)
    for i, r in enumerate(rankable, 1):
        r["rank_label"] = str(i)

    # Anchor reference at top, then ranked cells
    ordered = [r for r in rows if r.get("rank_label") == "ref"] + rankable

    # Print table
    print()
    print("Phase 1 prompt-engineering tournament leaderboard")
    print("(ranked by composite = state_acc + 0.5 × ROUGE-1)")
    print()
    header = (
        f"{'Rank':>4}  {'Cell':>4}  {'Utilization':<28}  "
        f"{'State':>6}  {'R-1':>6}  {'R-2':>6}  {'B-4':>6}  {'Composite':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in ordered:
        m = r["metrics"]
        state = m.get("state_accuracy", {}).get("overall", 0.0)
        r1 = m.get("rouge1", 0.0)
        r2 = m.get("rouge2", 0.0)
        b4 = m.get("bleu4", 0.0)
        composite = composite_score(m)
        rank = r["rank_label"]
        cell = r["cell"]
        name = r["name"][:28]
        print(
            f"{rank:>4}  {cell:>4}  {name:<28}  "
            f"{state:>6.2f}  {r1:>6.2f}  {r2:>6.2f}  {b4:>6.2f}  {composite:>9.2f}"
        )

    # Phase 2 candidate suggestion
    if len(rankable) >= 3:
        print()
        print("Top-3 single utilizations (Phase 2 composition candidates):")
        for r in rankable[:3]:
            print(f"  - cell {r['cell']} ({r['name']})")
        print()
        print("Phase 2 composition rules (from docs/PROMPT_ENGINEERING_PLAN.md §4):")
        print("  - Mutex: at most one of {#6, #7, #8} in any composition")
        print("  - Compose top non-mutex utilizations + exactly one mutex-pick if any")
        print("  - Validate the composition at n=50 before promoting to Phase 3 (full n=681)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
