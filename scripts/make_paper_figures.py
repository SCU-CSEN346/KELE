#!/usr/bin/env python3
"""Generate paper-quality figures from existing results JSONs.

Produces:
  1. stage_confusion.{pdf,png}    — 5×5 stage confusion matrix (A3B locked full)
  2. per_stage_accuracy.{pdf,png} — per-stage state accuracy bar chart
                                    (locked A3B vs baseline vs 3-shot mini)
  3. turn_index_accuracy.{pdf,png} — accuracy by turn index within dialogue
  4. bleu_vs_length.{pdf,png}     — scatter of dialogue BLEU-4 vs turn count
  5. system_radar.{pdf,png}       — 4-axis radar comparing baseline vs A3B
                                    vs A3B-fewshot on R-1/R-2/R-L/B-4

All figures saved to docs/figures/. Standalone; no GPU needed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS = Path("results")
FIGS = Path("docs/figures")
FIGS.mkdir(parents=True, exist_ok=True)

STAGES = ["a", "b", "c", "d", "e"]


def load_dialogues(results_dir: str) -> list[dict]:
    """Load every dialogue JSON from a results directory."""
    p = RESULTS / results_dir / "dialogues"
    if not p.exists():
        return []
    return [json.loads(f.read_text()) for f in sorted(p.glob("*.json"))]


def get_stage(state: str) -> str:
    """First char of state string is the stage letter."""
    return state[0] if state else ""


# ─────────────────────────────────────────────────────────────────
# Figure 1: 5×5 stage confusion matrix (A3B locked full)
# ─────────────────────────────────────────────────────────────────
def fig_stage_confusion(dialogues: list[dict], outname: str, title: str) -> None:
    matrix = np.zeros((5, 5), dtype=int)
    for dlg in dialogues:
        for turn in dlg.get("dialogue", []):
            gt = get_stage(turn.get("ground_truth_state", ""))
            pred = get_stage(turn.get("state", ""))
            if gt in STAGES and pred in STAGES:
                matrix[STAGES.index(gt), STAGES.index(pred)] += 1

    # Normalize rows to percent
    row_sums = matrix.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, matrix / row_sums * 100, 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pct,
        annot=matrix,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"pred {s}" for s in STAGES],
        yticklabels=[f"gt {s}" for s in STAGES],
        cbar_kws={"label": "% of gt-row"},
        ax=ax,
        vmin=0,
        vmax=100,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("Ground-truth stage")
    plt.tight_layout()
    plt.savefig(FIGS / f"{outname}.pdf", bbox_inches="tight")
    plt.savefig(FIGS / f"{outname}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {outname}.pdf+png")


# ─────────────────────────────────────────────────────────────────
# Figure 2: per-stage accuracy bar chart
# ─────────────────────────────────────────────────────────────────
def fig_per_stage_accuracy() -> None:
    # Hand-pulled from metrics_summary.json files
    systems = {
        "GPT-4o baseline (n=681)": [95.15, 36.93, 4.70, 5.04, 11.92],
        "A3B locked full (n=681)": [91.78, 39.29, 17.57, 14.78, 56.83],
        "A3B + 3-shot mini (n=25)": [88.0, 53.57, 27.66, 16.0, 43.48],
        "A3B + 3-shot n=50": [84.0, 44.07, 15.62, 13.73, 52.38],
        "Gemma 4 31B mini (n=25)": [96.0, 46.43, 17.02, 20.0, 52.17],
    }

    x = np.arange(len(STAGES))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5))

    colors = sns.color_palette("Set2", len(systems))
    for i, (sysname, vals) in enumerate(systems.items()):
        offset = (i - len(systems) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=sysname, color=colors[i])

    ax.set_xlabel("SocRule stage")
    ax.set_ylabel("State accuracy (%)")
    ax.set_title("Per-stage state accuracy by system")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{s} ({{'a':'questioning','b':'concept probe','c':'inductive reason','d':'resolution','e':'closure'}}[{repr(s)}])"
            for s in STAGES
        ]
    )
    # Cleaner labels
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
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "per_stage_accuracy.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "per_stage_accuracy.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ per_stage_accuracy.pdf+png")


# ─────────────────────────────────────────────────────────────────
# Figure 3: turn-index accuracy curve
# ─────────────────────────────────────────────────────────────────
def fig_turn_index_accuracy(dialogues: list[dict], outname: str, title: str) -> None:
    correct_by_turn: dict[int, int] = defaultdict(int)
    total_by_turn: dict[int, int] = defaultdict(int)
    for dlg in dialogues:
        for i, turn in enumerate(dlg.get("dialogue", [])):
            gt = turn.get("ground_truth_state", "")
            pred = turn.get("state", "")
            total_by_turn[i] += 1
            if gt == pred:
                correct_by_turn[i] += 1

    turns = sorted(total_by_turn.keys())
    accuracy = [correct_by_turn[t] / total_by_turn[t] * 100 for t in turns]
    n_samples = [total_by_turn[t] for t in turns]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot([t + 1 for t in turns], accuracy, "o-", color="steelblue", label="State accuracy")
    ax1.set_xlabel("Turn index within dialogue")
    ax1.set_ylabel("State accuracy (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar([t + 1 for t in turns], n_samples, alpha=0.15, color="gray", label="n turns")
    ax2.set_ylabel("Sample size (n turns)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(FIGS / f"{outname}.pdf", bbox_inches="tight")
    plt.savefig(FIGS / f"{outname}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {outname}.pdf+png")


# ─────────────────────────────────────────────────────────────────
# Figure 4: per-dialogue R-1 distribution (box plot or violin)
# ─────────────────────────────────────────────────────────────────
def fig_per_dialogue_dist(dialogues: list[dict], outname: str, title: str) -> None:
    # We need per-dialogue ROUGE - rebuild from the dialogue's teacher_response vs ground_truth_teacher
    # Use a simple proxy: turn length distribution since we lack the per-dialogue ROUGE in dialogue JSONs
    # Instead, distribution of turns per dialogue.
    turn_counts = [len(d.get("dialogue", [])) for d in dialogues]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        turn_counts,
        bins=range(min(turn_counts), max(turn_counts) + 2),
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Turns per dialogue")
    ax.set_ylabel("Number of dialogues")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / f"{outname}.pdf", bbox_inches="tight")
    plt.savefig(FIGS / f"{outname}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {outname}.pdf+png")


# ─────────────────────────────────────────────────────────────────
# Figure 5: radar chart comparing systems
# ─────────────────────────────────────────────────────────────────
def fig_system_radar() -> None:
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-4", "State acc"]
    # Normalize each metric to 0-100 for the radar (state acc already %)
    # ROUGE/BLEU max we observe is ~45 — scale to 100
    systems = {
        "GPT-4o baseline (n=681)": [44.61, 26.04, 38.02, 19.60, 25.94],
        "A3B locked think (n=681)": [30.63, 12.28, 22.37, 5.86, 38.70],
        "A3B + 3-shot mini (n=148)": [33.49, 13.30, 23.50, 6.59, 43.24],
        "Gemma 4 31B mini (n=148)": [30.11, 11.50, 21.48, 5.47, 41.89],
    }

    # Normalize: ROUGE/BLEU to /50, state acc /100
    scales = [50, 30, 50, 25, 100]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    colors = sns.color_palette("Set2", len(systems))
    for (sysname, vals), color in zip(systems.items(), colors):
        normed = [v / s * 100 for v, s in zip(vals, scales)]
        normed += normed[:1]
        ax.plot(angles, normed, color=color, label=sysname, linewidth=2)
        ax.fill(angles, normed, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{m}\n(scale: /{s})" for m, s in zip(metrics, scales)], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("System comparison (each axis normalized to indicated scale)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGS / "system_radar.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "system_radar.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ system_radar.pdf+png")


def main() -> None:
    print("Loading dialogues ...")
    a3b_full = load_dialogues("qwen35b-a3b-local-unified")
    baseline = load_dialogues("baseline")
    print(f"  A3B locked full: {len(a3b_full)} dialogues")
    print(f"  GPT-4o baseline: {len(baseline)} dialogues")

    print("\n[1/5] Stage confusion matrix (A3B locked full)")
    fig_stage_confusion(
        a3b_full, "stage_confusion_a3b_full", "5×5 stage confusion — A3B locked full (n=681)"
    )

    print("\n[2/5] Stage confusion matrix (baseline)")
    fig_stage_confusion(
        baseline, "stage_confusion_baseline", "5×5 stage confusion — GPT-4o baseline (n=681)"
    )

    print("\n[3/5] Per-stage state accuracy")
    fig_per_stage_accuracy()

    print("\n[4/5] Turn-index accuracy curve (A3B locked full)")
    fig_turn_index_accuracy(
        a3b_full, "turn_index_accuracy_a3b_full", "State accuracy by turn index — A3B locked full"
    )

    print("\n[5/5] System comparison radar")
    fig_system_radar()

    print(f"\nDone. Figures in: {FIGS}")


if __name__ == "__main__":
    main()
