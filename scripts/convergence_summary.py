"""Identify the plateau n from results/convergence/aggregate.json and produce
a concise summary + figure.

We define the plateau n* for a given (run, metric, tolerance ε) as the smallest
n in the grid such that the 95th percentile of |bootstrap_dev| stays at or
below ε for THIS n AND every larger n in the grid (monotone-bounded).

A run-aggregate plateau (for a chosen ε) is the largest per-run n* — the
sample size that guarantees the tolerance simultaneously across all 7 runs.

We then sweep ε over a few candidates and pick a defensible one based on
what the data shows."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AGG_PATH = ROOT / "results" / "convergence" / "aggregate.json"

METRICS = [
    ("state_acc", "State accuracy (%)", 1.0),
    ("rouge1", "ROUGE-1 F1 (%)", 1.0),
    ("rouge2", "ROUGE-2 F1 (%)", 0.75),
    ("bleu4_sent", "Sentence-BLEU-4 (%)", 0.5),
]


def plateau_n(per_n: dict[str, dict], metric: str, eps: float) -> int | None:
    """Smallest n in the grid such that p95 of |dev| stays <= eps for THIS n
    AND every larger n in the grid. Monotone bounded — guards against
    pre-mature accidental dips."""
    ns_sorted = sorted(int(k) for k in per_n)
    devs = [per_n[str(n)][metric]["abs_dev_p95"] for n in ns_sorted]
    # Find first index from the right where dev > eps; the position after it
    # is the plateau n.
    plateau_idx = None
    for i in range(len(ns_sorted) - 1, -1, -1):
        if devs[i] > eps:
            plateau_idx = i + 1
            break
        if i == 0:
            plateau_idx = 0
    if plateau_idx is None or plateau_idx >= len(ns_sorted):
        return None  # never plateaus within grid
    return ns_sorted[plateau_idx]


def main() -> None:
    data = json.loads(AGG_PATH.read_text())
    runs = data["runs"]
    n_grid: list[int] = data["n_grid"]

    # ── Per-run, per-metric, per-tolerance plateau table ─────────────────────
    print("=" * 90)
    print("Per-run plateau n* (smallest n where p95(|dev|) stays <= eps thereafter)")
    print("=" * 90)
    print()
    plateaus_by_metric: dict[str, dict[float, list[int | None]]] = {}
    run_labels: list[str] = []

    for metric, _label, _default_eps in METRICS:
        print(f"### {metric}")
        for eps in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            row = []
            for rel_path, payload in runs.items():
                p = plateau_n(payload["per_n"], metric, eps)
                row.append(p)
                if metric == "state_acc" and eps == 0.5:
                    run_labels.append(payload["label"])
            plateaus_by_metric.setdefault(metric, {})[eps] = row
            row_str = ", ".join(str(x) if x is not None else "—" for x in row)
            print(f"  eps={eps:>4.2f}   per-run: [{row_str}]   max: {_safe_max(row)}")
        print()

    # ── Aggregate (max over runs) ────────────────────────────────────────────
    print("=" * 90)
    print("Aggregate plateau n* = MAX across all 7 runs (so threshold holds simultaneously)")
    print("=" * 90)
    print()
    print(f"{'tolerance ε':>12} | " + " | ".join(f"{m:>12}" for m, *_ in METRICS))
    print("-" * 90)
    for eps in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        cells = []
        for metric, *_ in METRICS:
            v = _safe_max(plateaus_by_metric[metric][eps])
            cells.append(f"{v if v is not None else '—':>12}")
        print(f"{eps:>10.2f}    | " + " | ".join(cells))
    print()

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()
    for ax, (metric, label, default_eps) in zip(axes, METRICS):
        for rel_path, payload in runs.items():
            per_n = payload["per_n"]
            ns = sorted(int(k) for k in per_n)
            devs = [per_n[str(n)][metric]["abs_dev_p95"] for n in ns]
            ax.plot(
                ns,
                devs,
                marker="o",
                markersize=4,
                linewidth=1.2,
                label=payload["label"],
                alpha=0.85,
            )
        ax.axhline(default_eps, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel("n (dialogues sampled)")
        ax.set_ylabel("p95(|bootstrap dev from truth|)")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle(
        "Bootstrap convergence — p95(|dev from full-n truth|) vs. sample size n\n"
        "Dashed = chosen tolerance",
        fontsize=11,
    )
    fig.tight_layout()
    out_pdf = ROOT / "docs" / "figures" / "convergence_curves.pdf"
    out_png = ROOT / "docs" / "figures" / "convergence_curves.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


def _safe_max(xs: Iterable[int | None]) -> int | None:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return max(vals)


if __name__ == "__main__":
    main()
