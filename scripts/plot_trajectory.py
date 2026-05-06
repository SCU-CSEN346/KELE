"""Plot the running-metrics trajectory for the A3B fusion-think full run.

Reads results/qwen35b-a3b-local-unified/trajectory.json (produced by
compute_trajectory.py) and writes:

  results/qwen35b-a3b-local-unified/trajectory_headline.png   (single panel)
  results/qwen35b-a3b-local-unified/trajectory_full.png       (three panels)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# gpt-4o baseline for reference lines (results/baseline/metrics_summary.json).
BASELINE = {
    "state_acc_overall": 25.94,
    "per_stage": {"a": 95.15, "b": 36.93, "c": 4.70, "d": 5.04, "e": 11.92},
    "rouge1": 44.61,
    "rouge2": 26.04,
    "rougeL": 38.02,
    "bleu4": 19.60,
}

STAGE_LABELS = {
    "a": "a · problem detection",
    "b": "b · early reasoning",
    "c": "c · hard misconception",
    "d": "d · resolution",
    "e": "e · closure",
}

STAGE_COLORS = {
    "a": "#1f77b4",
    "b": "#ff7f0e",
    "c": "#d62728",
    "d": "#9467bd",
    "e": "#2ca02c",
}


def load_trajectory(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def plot_headline(traj: list[dict], out_path: Path) -> None:
    """Single-panel: overall state acc vs dialogues, with gpt-4o baseline."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    xs = [cp["n_dialogues"] for cp in traj]
    ys = [cp["state_acc_overall"] for cp in traj]
    final = traj[-1]

    ax.plot(xs, ys, color="#0d6efd", linewidth=2.2, label="A3B fusion-think (running)")
    ax.fill_between(
        xs,
        BASELINE["state_acc_overall"],
        ys,
        where=[y >= BASELINE["state_acc_overall"] for y in ys],
        alpha=0.12,
        color="#0d6efd",
        interpolate=True,
        label="lift over baseline",
    )

    ax.axhline(
        BASELINE["state_acc_overall"],
        color="#666",
        linestyle="--",
        linewidth=1.4,
        label=f"gpt-4o baseline ({BASELINE['state_acc_overall']:.2f}%)",
    )

    # Annotate peak
    peak_idx = max(range(len(ys)), key=lambda i: ys[i])
    ax.annotate(
        f"peak {ys[peak_idx]:.2f}%\n@ dlg {xs[peak_idx]}",
        xy=(xs[peak_idx], ys[peak_idx]),
        xytext=(xs[peak_idx] + 50, ys[peak_idx] + 1.5),
        fontsize=9,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#444", linewidth=0.8),
    )

    # Annotate final
    ax.annotate(
        f"final {final['state_acc_overall']:.2f}%\n(+{final['state_acc_overall'] - BASELINE['state_acc_overall']:.2f} vs baseline)",
        xy=(final["n_dialogues"], final["state_acc_overall"]),
        xytext=(final["n_dialogues"] - 200, final["state_acc_overall"] - 5),
        fontsize=9,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#444", linewidth=0.8),
    )

    ax.set_xlabel("Dialogues processed", fontsize=11)
    ax.set_ylabel("State accuracy (%)", fontsize=11)
    ax.set_title(
        "Running state-accuracy trajectory · A3B fusion-think · n=681",
        fontsize=13,
        pad=12,
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_ylim(20, 50)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_xlim(0, 700)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_full(traj: list[dict], out_path: Path) -> None:
    """Three-panel: overall state acc, per-stage state acc, ROUGE/BLEU."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), dpi=150, sharex=True)

    xs = [cp["n_dialogues"] for cp in traj]
    final = traj[-1]

    # ─── Panel 1: overall state acc ────────────────────────────────
    ys = [cp["state_acc_overall"] for cp in traj]
    ax1.plot(xs, ys, color="#0d6efd", linewidth=2.2, label="A3B fusion-think")
    ax1.axhline(
        BASELINE["state_acc_overall"],
        color="#666",
        linestyle="--",
        linewidth=1.3,
        label=f"gpt-4o baseline ({BASELINE['state_acc_overall']:.1f}%)",
    )
    ax1.fill_between(
        xs,
        BASELINE["state_acc_overall"],
        ys,
        where=[y >= BASELINE["state_acc_overall"] for y in ys],
        alpha=0.12,
        color="#0d6efd",
        interpolate=True,
    )
    ax1.set_ylabel("State accuracy (%)", fontsize=11)
    ax1.set_title("Overall state accuracy", fontsize=12, pad=8)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.25, linestyle=":")
    ax1.set_ylim(20, 50)
    ax1.annotate(
        f"final {final['state_acc_overall']:.2f}% (+{final['state_acc_overall'] - BASELINE['state_acc_overall']:.2f})",
        xy=(final["n_dialogues"], final["state_acc_overall"]),
        xytext=(final["n_dialogues"] - 200, final["state_acc_overall"] + 3),
        fontsize=9,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#444", linewidth=0.8),
    )

    # ─── Panel 2: per-stage state accuracies ────────────────────────
    for stage in "abcde":
        ys = [cp["per_stage"][stage] for cp in traj]
        ax2.plot(
            xs,
            ys,
            color=STAGE_COLORS[stage],
            linewidth=1.8,
            label=STAGE_LABELS[stage],
        )
        # Baseline reference (dashed, same color)
        ax2.axhline(
            BASELINE["per_stage"][stage],
            color=STAGE_COLORS[stage],
            linestyle=":",
            linewidth=1.0,
            alpha=0.5,
        )
    ax2.set_ylabel("Per-stage state accuracy (%)", fontsize=11)
    ax2.set_title(
        "Per-stage state accuracy (solid = A3B running, dotted = gpt-4o baseline)",
        fontsize=12,
        pad=8,
    )
    ax2.legend(loc="center right", framealpha=0.9, fontsize=8, ncols=1)
    ax2.grid(True, alpha=0.25, linestyle=":")
    ax2.set_ylim(0, 100)

    # ─── Panel 3: ROUGE / BLEU ──────────────────────────────────────
    metrics_to_plot = [
        ("rouge1", "ROUGE-1", "#1f77b4"),
        ("rouge2", "ROUGE-2", "#ff7f0e"),
        ("rougeL", "ROUGE-L", "#2ca02c"),
        ("bleu4", "BLEU-4", "#d62728"),
    ]
    for key, label, color in metrics_to_plot:
        ys = [cp[key] for cp in traj]
        ax3.plot(xs, ys, color=color, linewidth=1.8, label=f"{label} (A3B)")
        ax3.axhline(
            BASELINE[key],
            color=color,
            linestyle=":",
            linewidth=1.0,
            alpha=0.6,
            label=f"{label} baseline ({BASELINE[key]:.1f})",
        )
    ax3.set_xlabel("Dialogues processed", fontsize=11)
    ax3.set_ylabel("Text-overlap score", fontsize=11)
    ax3.set_title(
        "Text-overlap metrics (solid = A3B, dotted = gpt-4o baseline)",
        fontsize=12,
        pad=8,
    )
    ax3.legend(loc="center right", framealpha=0.9, fontsize=8, ncols=2)
    ax3.grid(True, alpha=0.25, linestyle=":")
    ax3.set_ylim(0, 50)
    ax3.set_xlim(0, 700)

    fig.suptitle(
        "Running-metrics trajectory · A3B fusion-think · n=681 (CSEN 346)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    BASE = Path("results/qwen35b-a3b-local-unified")
    traj = load_trajectory(BASE / "trajectory.json")
    print(f"Loaded {len(traj)} checkpoints from {BASE / 'trajectory.json'}")
    plot_headline(traj, BASE / "trajectory_headline.png")
    plot_full(traj, BASE / "trajectory_full.png")
