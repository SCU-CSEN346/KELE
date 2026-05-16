#!/usr/bin/env python3
"""Cascade comparison figure: every n=50 configuration plotted on (state, R-1).

Designed as the final summary plot for the paper, showing the full
experimental landscape from baseline through the integration combos.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIGS = Path("docs/figures")
FIGS.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict | None:
    p = Path(f"results/{name}/metrics_summary.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


# (label, dirname, marker, group)
configs = [
    ("GPT-4o baseline (n=681)",                  "baseline",                                  "*", "ref"),
    ("A3B locked think (n=681)",                 "qwen35b-a3b-local-unified",                 "s", "locked"),
    ("A3B locked think (n=50)",                  "qwen35b-a3b-local-n50-unified",             "s", "locked"),
    ("A3B + 3-shot (n=50)",                      "qwen35b-a3b-local-n50-unified-fewshot",     "o", "prompt-eng"),
    ("A3B + 10-shot (n=50)",                     "qwen35b-a3b-local-n50-unified-fewshot10",   "o", "prompt-eng"),
    ("BERT + A3B (placeholder, n=50)",           "bert-consultant-a3b-n50",                   "^", "bert-integration"),
    ("BERT + A3B + 10-shot (n=50)",              "bert-consultant-fewshot10-n50",             "D", "bert-integration"),
    ("BERT v2 + A3B + 10-shot (n=50)",           "bert-v2-consultant-fewshot10-n50",          "D", "bert-integration"),
    ("BERT + A4B + 10-shot (n=50)",              "bert-consultant-fewshot10-a4b-n50",         "D", "bert-integration"),
    ("BERT + Gemma 31B + 10-shot (n=50)",        "bert-consultant-fewshot10-gemma-n50",       "D", "bert-integration"),
]

colors_by_group = {
    "ref":               "#7f7f7f",
    "locked":            "#1f77b4",
    "prompt-eng":        "#ff7f0e",
    "bert-integration":  "#2ca02c",
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, dirname, marker, group in configs:
        m = load(dirname)
        if m is None:
            print(f"  ✗ skip: {dirname}")
            continue
        sa = m["state_accuracy"]["overall"]
        r1 = m["rouge1"]
        color = colors_by_group[group]
        size = 220 if group == "ref" else 140
        ax.scatter(r1, sa, marker=marker, s=size, color=color, edgecolor="black", linewidth=0.7, zorder=3)
        # Smart label placement
        offset_x = -1 if "Gemma" in label or "(n=681)" in label or "GPT-4o" in label else 0.6
        offset_y = -1.5 if "Gemma" in label or "GPT-4o" in label else 1
        ha = "right" if offset_x < 0 else "left"
        ax.annotate(label, (r1, sa), xytext=(r1 + offset_x, sa + offset_y),
                    fontsize=8.5, ha=ha,
                    arrowprops=None)

    ax.set_xlabel("ROUGE-1 (surface form)")
    ax.set_ylabel("State accuracy (%)")
    ax.set_title("Configurations tested on the SocratDataset test split (n=50 matched)\n"
                 "Higher right = better on both axes; GPT-4o (top-right) is the reference")

    # Reference lines
    gpt4o = load("baseline")
    if gpt4o:
        ax.axhline(gpt4o["state_accuracy"]["overall"], color="gray", linestyle="--", alpha=0.3)
        ax.axvline(gpt4o["rouge1"], color="gray", linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="*", linestyle="", markersize=14, color=colors_by_group["ref"], label="GPT-4o reference"),
        Line2D([], [], marker="s", linestyle="", markersize=9, color=colors_by_group["locked"], label="A3B locked (no improvement)"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=colors_by_group["prompt-eng"], label="Prompt-engineering (10-shot)"),
        Line2D([], [], marker="D", linestyle="", markersize=9, color=colors_by_group["bert-integration"], label="BERT consultant integration"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(25, 47)
    ax.set_ylim(20, 55)

    plt.tight_layout()
    plt.savefig(FIGS / "cascade_comparison.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "cascade_comparison.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✓ {FIGS}/cascade_comparison.pdf+png")


if __name__ == "__main__":
    main()
