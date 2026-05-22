#!/usr/bin/env python3
"""Bar chart comparing few-shot N variants on A3B fusion-think mini.

Produces docs/figures/fewshot_n_sweep.{pdf,png}:
- One group per N ∈ {0 (locked), 3 (legacy), 5, 7, 10}
- Bars within group: state acc, R-1
- Per-stage breakdown shown as line plot overlay (optional)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGS = Path("docs/figures")


def load_metrics(name: str) -> dict | None:
    p = RESULTS / name / "metrics_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    runs = [
        ("0 (locked)", "qwen35b-a3b-local-mini-unified"),
        ("3 (legacy b/c/d)", "qwen35b-a3b-local-mini-unified-fewshot"),
        ("5 (balanced)", "qwen35b-a3b-local-mini-unified-fewshot5"),
        ("7 (balanced+2)", "qwen35b-a3b-local-mini-unified-fewshot7"),
        ("10 (full)", "qwen35b-a3b-local-mini-unified-fewshot10"),
    ]

    state_accs = []
    rouge1s = []
    labels = []
    for label, dirname in runs:
        m = load_metrics(dirname)
        if m is None:
            state_accs.append(None)
            rouge1s.append(None)
        else:
            state_accs.append(m["state_accuracy"]["overall"])
            rouge1s.append(m["rouge1"])
        labels.append(label)

    x = np.arange(len(labels))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - width / 2,
        [s if s is not None else 0 for s in state_accs],
        width,
        label="State accuracy",
        color="steelblue",
    )
    bars2 = ax2.bar(
        x + width / 2,
        [r if r is not None else 0 for r in rouge1s],
        width,
        label="ROUGE-1",
        color="darkorange",
    )

    for bar, val in zip(bars1, state_accs):
        if val is not None:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.5,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="steelblue",
            )
    for bar, val in zip(bars2, rouge1s):
        if val is not None:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.3,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="darkorange",
            )

    ax1.set_xlabel("Number of teacher exemplars (N)")
    ax1.set_ylabel("State accuracy (%)", color="steelblue")
    ax2.set_ylabel("ROUGE-1", color="darkorange")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(0, 55)
    ax2.set_ylim(0, 40)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Few-shot N-sweep on Qwen 35B-A3B fusion-think mini (n=25 dialogues)")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")

    FIGS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGS / "fewshot_n_sweep.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "fewshot_n_sweep.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✓ {FIGS}/fewshot_n_sweep.pdf+png")

    # Per-stage breakdown
    fig, ax = plt.subplots(figsize=(10, 5))
    stages = ["a", "b", "c", "d", "e"]
    bar_x = np.arange(len(stages))
    bw = 0.16
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
    for i, ((label, dirname), color) in enumerate(zip(runs, colors)):
        m = load_metrics(dirname)
        if m is None:
            continue
        ps = [m["state_accuracy"]["per_stage"][s] for s in stages]
        offset = (i - len(runs) / 2 + 0.5) * bw
        ax.bar(bar_x + offset, ps, bw, label=label, color=color)

    ax.set_xticks(bar_x)
    ax.set_xticklabels(stages)
    ax.set_xlabel("SocRule stage")
    ax.set_ylabel("Per-stage state accuracy (%)")
    ax.set_title("Per-stage breakdown across few-shot N")
    ax.legend(loc="upper right", fontsize=8, title="N (exemplar selection)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(FIGS / "fewshot_n_sweep_per_stage.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "fewshot_n_sweep_per_stage.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✓ {FIGS}/fewshot_n_sweep_per_stage.pdf+png")


if __name__ == "__main__":
    main()
