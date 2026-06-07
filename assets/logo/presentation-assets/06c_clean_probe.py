"""Chart 6c - Clean-probe collapse on held-out synthetic data.

Data from paper §3.3 and App C. STL stage-balanced macro collapses from 63.40
(published test) to 32.86 (SocratDataset-SYNTHETIC, n=75 Sonnet-generated,
verifiably disjoint). Baseline Gemma + 10-shot scores 56.13 on the same SYNTH
set - STL underperforms its own baseline on truly held-out data.
"""

import matplotlib.pyplot as plt
import numpy as np
import poster_style as ps

ps.apply()

labels = [
    "STL on\npublished test",
    "STL on\nclean SYNTH",
    "Baseline Gemma\non clean SYNTH",
]
colors = [ps.SLATE, ps.SLATE, ps.RED]
alphas = [0.85, 0.35, 0.85]

stage_bal = [63.40, 32.86, 56.13]
rouge_1 = [48.07, 35.72, 38.76]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.0))
fig.subplots_adjust(top=0.80, wspace=0.30, left=0.07, right=0.97, bottom=0.14)


def draw(ax, vals, ylabel, title, *, ymax):
    xs = np.arange(len(labels))
    bars = ax.bar(xs, vals, 0.6, color=colors, edgecolor=ps.NAVY, linewidth=0.9)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    for x, v in zip(xs, vals):
        ax.text(
            x,
            v + ymax * 0.018,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=ps.NAVY,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(0, ymax)
    ax.set_title(title, pad=12, fontsize=15)
    ax.grid(axis="y", color=ps.DIVIDER, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


draw(
    ax1,
    stage_bal,
    "Stage-balanced macro state acc (%)",
    "Stage routing collapses on unseen data",
    ymax=92,
)
draw(ax2, rouge_1, "ROUGE-1 (%)", "ROUGE-1 collapses with it", ymax=60)

# --- Left panel: drop annotation ---
# Place a clean STRAIGHT horizontal arrow above bars 0 and 1, well above the
# bar tops (y = 75) — does not cross the dashed reference line (y = 56.13).
drop1 = stage_bal[0] - stage_bal[1]
arrow_y = 75
ax1.annotate(
    "",
    xy=(1, arrow_y),
    xytext=(0, arrow_y),
    arrowprops=dict(arrowstyle="-|>", color=ps.GOLD, lw=2.4, mutation_scale=22),
)
# Drop-label sits in a card centered above the arrow
ax1.text(
    0.5,
    arrow_y + 3.5,
    f"Δ = −{drop1:.2f} pp",
    ha="center",
    va="bottom",
    fontsize=14,
    fontweight="bold",
    color=ps.GOLD,
    bbox=dict(boxstyle="round,pad=0.5", facecolor=ps.LIGHT_GOLD, edgecolor=ps.GOLD, linewidth=1.3),
)

# --- Title and subtitle in figure-level top strip ---
fig.text(
    0.5,
    0.94,
    "SocratTeachLLM collapses on truly unseen data",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=ps.NAVY,
)
fig.text(
    0.5,
    0.88,
    "75 Sonnet-generated dialogues, disjoint from STL's training set. "
    "Even ROUGE-1 (the surface-form metric STL was trained to win) collapses on unseen data.",
    ha="center",
    va="center",
    fontsize=12,
    color=ps.GRAY,
    style="italic",
)

ps.save_fig(fig, "06c_clean_probe", __file__)
