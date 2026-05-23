"""Generate the figures for the paper / README that visualize the
two-leaderboard inversion + cross-metric comparisons.

Reads results/master_leaderboard.json and writes:
  docs/figures/leaderboard_inversion.{pdf,png}   — side-by-side rankings
  docs/figures/pareto_inversion.{pdf,png}         — scatter: state acc vs surface sum
  docs/figures/ngram_gap_widening.{pdf,png}       — R-1/R-2/B-4 SocratTeachLLM-gap by n-gram order
  docs/figures/judge_vs_surface.{pdf,png}         — LLM-judge vs surface metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGS = Path("docs/figures")
FIGS.mkdir(parents=True, exist_ok=True)

# Short labels for display
SHORT = {
    "GPT-4o + SocratTeachLLM (paper baseline)": "GPT-4o + STL (paper)",
    "Gemma 4 31B + top-3 stack": "Gemma + top-3",
    "Qwen 35B-A3B + top-3 stack": "A3B + top-3",
    "Gemma 4 31B + 10-shot (n=50 ref)": "Gemma 10-shot (n=50)",
    "Gemma 4 31B + 10-shot (n=681 LOCKED)": "Gemma 10-shot (n=681)",
    "Qwen 35B-A3B + 10-shot (n=681)": "A3B 10-shot (n=681)",
    "Sonnet 4.6 + BERT + top-3": "Sonnet + top-3",
    "Opus 4.6 + BERT + top-3": "Opus + top-3",
    "Sonnet 4.6 + BERT + 10-shot only": "Sonnet 10-shot",
    "Opus 4.6 + BERT + 10-shot only": "Opus 10-shot",
    "Sonnet 4.6 + BERT raw (no exemplars)": "Sonnet raw",
    "Opus 4.6 + BERT raw (no exemplars)": "Opus raw",
    "Opus 4.6 + BERT + top-3 (n=681 Phase 3)": "Opus + top-3 (n=681)",
    "Sonnet 4.6 + BERT + top-3 (n=681 Phase 3)": "Sonnet + top-3 (n=681)",
    "Gemma 4 31B + top-3 (n=200 validation)": "Gemma + top-3 (n=200)",
    "Sonnet 4.6 consultant + SocratTeachLLM (n=50)": "Sonnet→STL (artifact)",
    "Sonnet 4.6 consultant + SocratTeachLLM (n=50 clean rerun)": "Sonnet→STL (clean)",
    "Opus 4.6 consultant + SocratTeachLLM (n=50)": "Opus→STL",
    "Sonnet 4.6 consultant + SocratTeachLLM (EN)": "Sonnet→STL (EN)",
    "Opus 4.6 consultant + SocratTeachLLM (EN)": "Opus→STL (EN)",
    "Opus 4.6 + BERT + top-3 (EN)": "Opus + top-3 (EN)",
}


def color_for(name: str) -> str:
    if "SocratTeachLLM" in name or "STL" in SHORT.get(name, ""):
        return "#d62728"  # red — memorization
    if "raw" in name.lower():
        return "#bcbd22"  # yellow — no scaffolding
    if "top-3" in name and "Gemma" in name and "(n=200)" not in name and "(n=681" not in name:
        return "#2ca02c"  # green — best open-weight
    if "Opus" in name and "top-3" in name:
        return "#1f77b4"  # blue — frontier+top-3
    if "Sonnet" in name and "top-3" in name:
        return "#9467bd"  # purple — frontier+top-3
    return "#7f7f7f"  # gray — default


def load_leaderboard() -> dict:
    p = Path("results/master_leaderboard.json")
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def fig_leaderboard_inversion(data: dict):
    """Side-by-side bar charts: surface-form ranking vs state-acc ranking."""
    rows = [r for r in data["rows"] if r.get("available")]
    if not rows:
        return

    # Two rankings
    by_surface = sorted(rows, key=lambda r: r["surface_sum"], reverse=True)[:10]
    by_state = sorted(rows, key=lambda r: r["state"], reverse=True)[:10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: surface-form
    names = [SHORT.get(r["name"], r["name"]) for r in by_surface]
    vals = [r["surface_sum"] for r in by_surface]
    colors = [color_for(r["name"]) for r in by_surface]
    axes[0].barh(range(len(names)), vals, color=colors)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("R-1 + R-2 + BLEU-4 (sum)")
    axes[0].set_title("Surface-form ranking (KELE-paper-style)")
    for i, v in enumerate(vals):
        axes[0].text(v + 0.5, i, f"{v:.2f}", va="center", fontsize=8)

    # Right: state acc
    names2 = [SHORT.get(r["name"], r["name"]) for r in by_state]
    vals2 = [r["state"] for r in by_state]
    colors2 = [color_for(r["name"]) for r in by_state]
    axes[1].barh(range(len(names2)), vals2, color=colors2)
    axes[1].set_yticks(range(len(names2)))
    axes[1].set_yticklabels(names2, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("State accuracy (%)")
    axes[1].set_title("Pedagogical ranking (state acc)")
    for i, v in enumerate(vals2):
        axes[1].text(v + 0.5, i, f"{v:.2f}%", va="center", fontsize=8)

    fig.suptitle("Two leaderboards, one inversion — top 10 configs", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "leaderboard_inversion.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "leaderboard_inversion.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGS}/leaderboard_inversion.[pdf,png]")


def fig_pareto_inversion(data: dict):
    """Scatter: state acc (x) vs surface-form sum (y). Anti-correlation = broken benchmark."""
    rows = [r for r in data["rows"] if r.get("available")]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    xs = [r["state"] for r in rows]
    ys = [r["surface_sum"] for r in rows]
    colors = [color_for(r["name"]) for r in rows]

    ax.scatter(xs, ys, c=colors, s=120, alpha=0.85, edgecolor="black", linewidth=0.5)

    for r, x, y in zip(rows, xs, ys):
        label = SHORT.get(r["name"], r["name"])
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Pedagogical: state accuracy (%) →")
    ax.set_ylabel("Surface-form: R-1 + R-2 + BLEU-4 (sum) →")
    ax.set_title(
        "Surface-form metrics vs pedagogical state accuracy\n(SocratTeachLLM-using configs in red; tracks anti-correlation in upper-left)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "pareto_inversion.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "pareto_inversion.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGS}/pareto_inversion.[pdf,png]")


def fig_ngram_gap_widening(data: dict):
    """For SocratTeachLLM-using configs vs Opus+top-3, show R-1, R-2, BLEU-4 gaps widening."""
    rows = [r for r in data["rows"] if r.get("available")]
    # Get reference Opus+top-3
    opus_top3 = next((r for r in rows if r["name"] == "Opus 4.6 + BERT + top-3"), None)
    if not opus_top3:
        return

    # SocratTeachLLM configs
    stl_configs = [r for r in rows if "SocratTeachLLM" in r["name"]]
    if not stl_configs:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    metrics = ["R-1", "R-2", "BLEU-4"]
    x = np.arange(len(metrics))
    width = 0.8 / (len(stl_configs) + 1)

    # Opus+top-3 reference
    opus_vals = [opus_top3["r1"], opus_top3["r2"], opus_top3["b4"]]
    ax.bar(
        x - 0.4 + width / 2,
        opus_vals,
        width,
        label=SHORT.get(opus_top3["name"], opus_top3["name"]),
        color="#1f77b4",
    )

    for i, stl in enumerate(stl_configs):
        vals = [stl["r1"], stl["r2"], stl["b4"]]
        # Clamp alpha to [0.4, 1.0] regardless of how many STL configs there are
        alpha = min(0.95, 0.5 + 0.08 * i)
        ax.bar(
            x - 0.4 + width / 2 + (i + 1) * width,
            vals,
            width,
            label=SHORT.get(stl["name"], stl["name"]),
            color="#d62728",
            alpha=alpha,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title(
        "SocratTeachLLM advantage widens with n-gram length\n(memorization signature — higher-order n-grams capture phrase-level fingerprinting)"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "ngram_gap_widening.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "ngram_gap_widening.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGS}/ngram_gap_widening.[pdf,png]")


def fig_judge_vs_surface(data: dict):
    """Scatter: LLM-judge (x) vs surface R-1 (y). Cluster of judged configs."""
    rows = [r for r in data["rows"] if r.get("available") and r["judge_overall"] is not None]
    if not rows:
        print("(no judged configs yet)")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [r["judge_overall"] for r in rows]
    ys = [r["r1"] for r in rows]
    colors = [color_for(r["name"]) for r in rows]
    ax.scatter(xs, ys, c=colors, s=120, alpha=0.85, edgecolor="black", linewidth=0.5)

    for r, x, y in zip(rows, xs, ys):
        ax.annotate(
            SHORT.get(r["name"], r["name"]),
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_xlabel("LLM-judge composite (0-10, memorization-resistant)")
    ax.set_ylabel("Surface ROUGE-1")
    ax.set_title("LLM-judge vs surface ROUGE-1 — orthogonal axes of evaluation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "judge_vs_surface.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "judge_vs_surface.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGS}/judge_vs_surface.[pdf,png]")


def fig_4metric_panel(data: dict):
    """Bar chart: 4 metrics (state, R-1, semantic R-1, LLM-judge) for all judged configs."""
    rows = [r for r in data["rows"] if r.get("available") and r["judge_overall"] is not None]
    if not rows:
        return

    rows.sort(key=lambda r: r["judge_overall"], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    names = [SHORT.get(r["name"], r["name"]) for r in rows]
    x = np.arange(len(names))
    width = 0.2

    # Normalize all to 0-100 scale for comparable bars
    state = [r["state"] for r in rows]
    r1 = [r["r1"] for r in rows]
    sem = [(r["semantic_r1"] * 100) if r["semantic_r1"] is not None else 0 for r in rows]
    judge = [r["judge_overall"] * 10 for r in rows]

    ax.bar(x - 1.5 * width, state, width, label="State acc (%)", color="#2ca02c")
    ax.bar(x - 0.5 * width, r1, width, label="Surface ROUGE-1", color="#ff7f0e")
    ax.bar(x + 0.5 * width, sem, width, label="Semantic R-1 ×100", color="#9467bd")
    ax.bar(x + 1.5 * width, judge, width, label="LLM-judge ×10", color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Four-metric evaluation panel (sorted by LLM-judge)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "four_metric_panel.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "four_metric_panel.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGS}/four_metric_panel.[pdf,png]")


def main():
    data = load_leaderboard()
    fig_leaderboard_inversion(data)
    fig_pareto_inversion(data)
    fig_ngram_gap_widening(data)
    fig_judge_vs_surface(data)
    fig_4metric_panel(data)


if __name__ == "__main__":
    main()
