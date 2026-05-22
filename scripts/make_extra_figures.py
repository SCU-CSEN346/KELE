#!/usr/bin/env python3
"""Additional figures beyond make_paper_figures.py.

Produces:
  6. state_frequency.{pdf,png}     — bar chart of all 34 states (sorted by count)
                                     across the SocratDataset
  7. dialogue_length_hist.{pdf,png} — histogram of dialogueRound (5–12)
  8. stage_turn_count.{pdf,png}    — total turns per stage in the dataset
  9. fewshot_n_sweep.{pdf,png}     — placeholder for the many-shot sweep result
                                     (regenerates once 5/7/10-shot runs land)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REFS = Path("references/KELE")
RESULTS = Path("results")
FIGS = Path("docs/figures")
FIGS.mkdir(parents=True, exist_ok=True)

STAGES = ["a", "b", "c", "d", "e"]


def fig_state_frequency() -> None:
    """All 34 states sorted by ground-truth turn count in train+test."""
    counts: Counter[str] = Counter()
    data = json.loads((REFS / "SocratDataset.json").read_text())
    for dlg in data:
        for turn in dlg.get("dialogue", []):
            state = turn.get("state", "")
            if state:
                counts[state] += 1

    sorted_states = sorted(counts.items(), key=lambda x: (x[0][0], -x[1]))

    states, freqs = zip(*sorted_states)
    colors_by_stage = {
        "a": "#1f77b4",
        "b": "#ff7f0e",
        "c": "#2ca02c",
        "d": "#d62728",
        "e": "#9467bd",
    }
    bar_colors = [colors_by_stage[s[0]] for s in states]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(states)), freqs, color=bar_colors)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("State (grouped by stage)")
    ax.set_ylabel("Turn count in full SocratDataset")
    ax.set_title(f"State-label frequency ({sum(freqs):,} total turns, {len(states)} states)")
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    legend_elems = [Patch(facecolor=c, label=f"Stage {s}") for s, c in colors_by_stage.items()]
    ax.legend(handles=legend_elems, loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGS / "state_frequency.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "state_frequency.png", bbox_inches="tight", dpi=200)
    plt.close()
    print("  ✓ state_frequency.pdf+png")


def fig_dialogue_length_hist() -> None:
    data = json.loads((REFS / "SocratDataset.json").read_text())
    lengths = [dlg.get("dialogueRound", 0) for dlg in data]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        lengths,
        bins=range(min(lengths), max(lengths) + 2),
        color="steelblue",
        edgecolor="white",
        align="left",
    )
    ax.axvline(mean_len, color="red", linestyle="--", label=f"mean = {mean_len:.2f}")
    ax.set_xlabel("Turns per dialogue (dialogueRound)")
    ax.set_ylabel("Number of dialogues")
    ax.set_title(
        f"Dialogue length distribution ({len(data):,} dialogues; μ={mean_len:.2f}, σ={std_len:.2f})"
    )
    ax.set_xticks(range(min(lengths), max(lengths) + 1))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "dialogue_length_hist.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "dialogue_length_hist.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(
        f"  ✓ dialogue_length_hist.pdf+png  (μ={mean_len:.2f}, range {min(lengths)}–{max(lengths)})"
    )


def fig_stage_turn_count() -> None:
    """Total turns per stage across the dataset."""
    data = json.loads((REFS / "SocratDataset.json").read_text())
    counts = {s: 0 for s in STAGES}
    for dlg in data:
        for turn in dlg.get("dialogue", []):
            state = turn.get("state", "")
            if state and state[0] in counts:
                counts[state[0]] += 1

    colors_by_stage = {
        "a": "#1f77b4",
        "b": "#ff7f0e",
        "c": "#2ca02c",
        "d": "#d62728",
        "e": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        STAGES,
        [counts[s] for s in STAGES],
        color=[colors_by_stage[s] for s in STAGES],
    )
    for bar, stage in zip(bars, STAGES):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{counts[stage]:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("SocRule stage")
    ax.set_ylabel("Total annotated turns")
    ax.set_title(f"Stage turn distribution (SocratDataset, {sum(counts.values()):,} turns)")
    ax.set_xticklabels(
        [
            "a\n(questioning)",
            "b\n(concept probe)",
            "c\n(inductive reason)",
            "d\n(resolution)",
            "e\n(closure)",
        ]
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "stage_turn_count.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "stage_turn_count.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(
        f"  ✓ stage_turn_count.pdf+png  (c={counts['c']:,} is ~{counts['c'] / min(counts.values()):.1f}× any other stage)"
    )


def fig_per_dialogue_turn_count(results_dir: str, outname: str, title: str) -> None:
    """Turn count per dialogue from a results directory."""
    dialogues = []
    p = RESULTS / results_dir / "dialogues"
    if not p.exists():
        print(f"  ✗ {results_dir} not found, skip {outname}")
        return
    for f in sorted(p.glob("*.json")):
        dialogues.append(json.loads(f.read_text()))
    turn_counts = [d.get("num_turns_generated", len(d.get("dialogue", []))) for d in dialogues]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        turn_counts,
        bins=range(min(turn_counts), max(turn_counts) + 2),
        color="seagreen",
        edgecolor="white",
        align="left",
    )
    ax.set_xlabel("Turns per dialogue (generated)")
    ax.set_ylabel("Number of dialogues")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / f"{outname}.pdf", bbox_inches="tight")
    plt.savefig(FIGS / f"{outname}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {outname}.pdf+png")


def main() -> None:
    print("[1] State-label frequency (full dataset)")
    fig_state_frequency()
    print("\n[2] Dialogue length histogram")
    fig_dialogue_length_hist()
    print("\n[3] Stage turn count")
    fig_stage_turn_count()
    print("\n[4] Per-dialogue turn count — A3B locked full")
    fig_per_dialogue_turn_count(
        "qwen35b-a3b-local-unified",
        "turn_count_a3b_full",
        "Generated turns per dialogue — A3B locked full (n=681)",
    )


if __name__ == "__main__":
    main()
