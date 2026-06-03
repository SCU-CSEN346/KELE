"""Chart G' - Unified-score leaderboard with frontier ceiling + paper baseline.

Data from paper Tables 2 and 4 (§3.1, §3.2). All n=681. Our locked headline
overtakes the best frontier configuration we measured by +2.18 unified points,
AND the paper's GPT-4o + SocratTeachLLM baseline by +19.25 unified points
(now that the baseline LLM-judge pass has landed — see EXPERIMENT_LOG.md
2026-06-02 entry).

Baseline: stage_bal 30.75, judge 7.52, unified 52.99 (rank 38 of 42 unified
cells in the master leaderboard). The paper's headline configuration sits in
the bottom decile under a memorization-resistant evaluation.
"""

import matplotlib.pyplot as plt
import poster_style as ps

ps.apply()

# (label, unified score, group). All values are real unified = ½·stage_bal + ½·judge×10.
rows = [
    ("GPT-4o + SocratTeachLLM  (paper baseline)", 52.99, "baseline"),
    ("BERT + Gemma + 10-shot  (our prior)", 68.65, "ours_prior"),
    ("BERT + Claude-Opus + top-3  (frontier)", 69.37, "frontier"),
    ("BERT + Claude-Sonnet + top-3  (frontier ceiling)", 70.06, "frontier_ceiling"),
    ("Qwen3.5-LoRA + Gemma + 10-shot  (LOCKED · ours)", 72.24, "ours_locked"),
]

color_map = {
    "baseline": ps.SLATE,
    "ours_prior": ps.RED,
    "ours_locked": ps.RED,
    "frontier": ps.GRAY,
    "frontier_ceiling": ps.NAVY,
}
alpha_map = {
    "baseline": 1.0,
    "ours_prior": 0.70,
    "ours_locked": 1.0,
    "frontier": 0.92,
    "frontier_ceiling": 1.0,
}

# Single-axis layout — no top-strip, no footnote (judge now measured for every row)
fig, ax = plt.subplots(figsize=(15, 7.0))
fig.subplots_adjust(top=0.90, bottom=0.10, left=0.30, right=0.97)

# --- Main bar chart ---
labels = [r[0] for r in rows]
values = [r[1] for r in rows]
groups = [r[2] for r in rows]

bars = ax.barh(
    range(len(rows)), values, color=[color_map[g] for g in groups], edgecolor=ps.NAVY, linewidth=0.9
)
for bar, g in zip(bars, groups):
    bar.set_alpha(alpha_map[g])

# Value labels — placed just past the bar end
for i, v in enumerate(values):
    ax.text(
        v + 0.30,
        i,
        f"{v:.2f}",
        va="center",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=ps.NAVY,
    )

ax.set_yticks(range(len(rows)))
ax.set_yticklabels(labels, fontsize=13)
ax.set_xlabel("Unified score   (½ · stage_bal  +  ½ · judge × 10)", fontsize=14, labelpad=10)
ax.set_xlim(48, 96)
ax.set_title("Memorization-resistant unified-score leaderboard  (n = 681)", pad=14, fontsize=18)

# Vertical dotted reference at frontier ceiling
ax.axvline(70.06, color=ps.NAVY, linestyle=":", linewidth=1.1, alpha=0.40, zorder=0)

# --- Gap callouts (right side, staggered horizontally with clear separation) ---
i_baseline = next(i for i, r in enumerate(rows) if r[2] == "baseline")
i_ceiling = next(i for i, r in enumerate(rows) if r[2] == "frontier_ceiling")
i_locked = next(i for i, r in enumerate(rows) if r[2] == "ours_locked")

# Small gap (+2.18 vs frontier ceiling) — close to bars
small_arrow_x = 76.0
small_box_x = small_arrow_x + 1.0
ax.annotate(
    "",
    xy=(small_arrow_x, i_locked),
    xytext=(small_arrow_x, i_ceiling),
    arrowprops=dict(arrowstyle="<->", color=ps.GOLD, lw=2.2),
)
ax.text(
    small_box_x,
    (i_ceiling + i_locked) / 2,
    "+2.18\nvs frontier\nceiling",
    ha="left",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=ps.GOLD,
    bbox=dict(boxstyle="round,pad=0.45", facecolor=ps.LIGHT_GOLD, edgecolor=ps.GOLD, linewidth=1.3),
)

# Big gap (+19.25 vs paper baseline) — its own column on the far right
big_arrow_x = 88.0
big_box_x = big_arrow_x + 1.0
big_gap = values[i_locked] - values[i_baseline]
ax.annotate(
    "",
    xy=(big_arrow_x, i_locked),
    xytext=(big_arrow_x, i_baseline),
    arrowprops=dict(arrowstyle="<->", color=ps.RED, lw=2.6),
)
ax.text(
    big_box_x,
    (i_baseline + i_locked) / 2,
    f"+{big_gap:.2f}\nvs paper\nbaseline",
    ha="left",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=ps.RED,
    bbox=dict(boxstyle="round,pad=0.50", facecolor=ps.LIGHT_RED, edgecolor=ps.RED, linewidth=1.5),
)

ax.grid(axis="x", color=ps.DIVIDER, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

ps.save_fig(fig, "G_unified_ceiling", __file__)
