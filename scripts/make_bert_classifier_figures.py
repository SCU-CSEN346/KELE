#!/usr/bin/env python3
"""Confusion matrices + per-stage bars for the BERT classifier(s).

Reads results/{stage,state}_classifier_v1/test_eval.json and generates:
- bert_stage_classifier_confusion.{pdf,png}
- bert_state_classifier_confusion_5way.{pdf,png}  (collapsed to stages)
- bert_classifier_per_stage_comparison.{pdf,png}  (vs LLM consultants)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIGS = Path("docs/figures")
FIGS.mkdir(parents=True, exist_ok=True)

STAGES = ["a", "b", "c", "d", "e"]


def stage_confusion_from_5way() -> None:
    p = Path("results/stage_classifier_v1/test_eval.json")
    if not p.exists():
        print(f"  ✗ {p} missing — skip")
        return
    d = json.loads(p.read_text())
    cm = np.array(d["confusion_matrix"])  # 5x5

    # Normalize rows
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pct,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"pred {s}" for s in STAGES],
        yticklabels=[f"gt {s}" for s in STAGES],
        cbar_kws={"label": "% of gt-row"},
        vmin=0,
        vmax=100,
        ax=ax,
    )
    ax.set_title("5-stage BERT classifier confusion (test split, n=4,304)")
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("Ground-truth stage")
    plt.tight_layout()
    plt.savefig(FIGS / "bert_stage_classifier_confusion.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "bert_stage_classifier_confusion.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ bert_stage_classifier_confusion.pdf+png")


def per_stage_comparison() -> None:
    p = Path("results/stage_classifier_v1/test_eval.json")
    if not p.exists():
        return
    d = json.loads(p.read_text())
    bert_stage = [d["per_stage"][s]["acc"] * 100 for s in STAGES]

    # Reference numbers from existing runs
    systems = {
        "GPT-4o (n=681)": [95.15, 36.93, 4.70, 5.04, 11.92],
        "A3B locked (n=681)": [91.78, 39.29, 17.57, 14.78, 56.83],
        "A3B + 10-shot (n=50)": [96.0, 40.68, 18.95, 27.45, 63.64],
        "BERT 5-stage classifier": bert_stage,
    }

    x = np.arange(len(STAGES))
    width = 0.20
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(systems))
    for i, ((sysname, vals), color) in enumerate(zip(systems.items(), colors)):
        offset = (i - len(systems) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=sysname, color=color)

    ax.set_xlabel("SocRule stage")
    ax.set_ylabel("Per-stage accuracy (%)")
    ax.set_title("Per-stage stage accuracy by consultant (test-split-comparable)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "a\n(questioning)",
            "b\n(concept probe)",
            "c\n(inductive reason)",
            "d\n(resolution)",
            "e\n(closure)",
        ]
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "bert_classifier_per_stage_comparison.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "bert_classifier_per_stage_comparison.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ bert_classifier_per_stage_comparison.pdf+png")


def state_classifier_per_state() -> None:
    """Horizontal bar chart of per-state accuracy from the 34-state classifier."""
    p = Path("results/state_classifier_v1/test_eval.json")
    if not p.exists():
        print(f"  ✗ {p} missing — skip")
        return
    d = json.loads(p.read_text())

    per_state = d["test_per_state_accuracy"]
    n_per_state = d["test_per_state_n"]
    # Sort by stage then state index
    states_with_data = [(s, per_state[s], n_per_state[s]) for s in per_state if n_per_state[s] > 0]
    # Sort within each stage by state number
    states_with_data.sort(key=lambda x: (x[0][0], int(x[0][1:])))

    states = [s for s, _, _ in states_with_data]
    accs = [a * 100 for _, a, _ in states_with_data]
    ns = [n for _, _, n in states_with_data]

    colors_by_stage = {
        "a": "#1f77b4",
        "b": "#ff7f0e",
        "c": "#2ca02c",
        "d": "#d62728",
        "e": "#9467bd",
    }
    bar_colors = [colors_by_stage[s[0]] for s in states]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(states)), accs, color=bar_colors)
    # Annotate n on top
    for bar, n in zip(bars, ns):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.5,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90,
        )

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("State (grouped by stage; only states in test set shown)")
    ax.set_ylabel("Per-state accuracy (%)")
    ax.set_title(
        f"34-state BERT classifier per-state accuracy on test split "
        f"(overall: {d['test_state_accuracy'] * 100:.2f}%, n={d['n_test_turns']:,} turns)"
    )
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elems = [Patch(facecolor=c, label=f"Stage {s}") for s, c in colors_by_stage.items()]
    ax.legend(handles=legend_elems, loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGS / "bert_state_classifier_per_state.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "bert_state_classifier_per_state.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ bert_state_classifier_per_state.pdf+png")


def main() -> None:
    print("[1] BERT stage classifier confusion matrix")
    stage_confusion_from_5way()
    print("\n[2] Per-stage comparison vs LLM consultants")
    per_stage_comparison()
    print("\n[3] 34-state BERT classifier per-state accuracy")
    state_classifier_per_state()


if __name__ == "__main__":
    main()
