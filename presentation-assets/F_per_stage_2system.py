"""Chart F (reworked) - Per-stage state accuracy: GPT-4o baseline vs ours.

Data from paper Table 3 (sec:results §3.2). 2-system version with multiplier
annotations on stages c/d/e. Multipliers reach 7.5x / 9.2x / 6.6x on the
cognitively hard middle/closure stages.
"""
import numpy as np
import matplotlib.pyplot as plt
import poster_style as ps

ps.apply()

stages = [
    ("A", "Questioning"),
    ("B", "Early reasoning"),
    ("C", "Induction (22 states)"),
    ("D", "Resolution"),
    ("E", "Closure"),
]

gpt4o = [95.15, 36.93,  4.70,  5.04, 11.92]
ours  = [100.0, 46.39, 35.42, 46.36, 78.45]

xs = np.arange(len(stages))
bar_w = 0.36

fig, ax = plt.subplots(figsize=(14, 8.0))
fig.subplots_adjust(top=0.82, bottom=0.18, left=0.07, right=0.96)

ax.bar(xs - bar_w/2, gpt4o, bar_w, color=ps.SLATE, alpha=0.85,
       edgecolor=ps.NAVY, linewidth=0.9,
       label="GPT-4o + SocratTeachLLM  (paper baseline)")
ax.bar(xs + bar_w/2, ours, bar_w, color=ps.RED, alpha=0.95,
       edgecolor=ps.NAVY, linewidth=0.9,
       label="Qwen3.5-LoRA + Gemma 4 31B + 10-shot")

# value labels — small offset above the bar top
for x, v in zip(xs - bar_w/2, gpt4o):
    ax.text(x, v + 1.0, f"{v:.1f}", ha="center", va="bottom",
            fontsize=11, fontweight="medium", color=ps.NAVY)
for x, v in zip(xs + bar_w/2, ours):
    ax.text(x, v + 1.0, f"{v:.1f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=ps.RED)

# Multiplier callouts placed in the dedicated band y in [128..148] — well above the tallest bar
# This keeps the "headline" annotations in their own visual layer.
ANNOT_Y = 130
for i, (g, o) in enumerate(zip(gpt4o, ours)):
    delta = o - g
    mult = o / g if g > 0 else float("inf")
    is_dominant = i >= 2
    color = ps.GOLD if is_dominant else ps.GRAY
    weight = "bold" if is_dominant else "medium"
    fontsize = 13 if is_dominant else 11
    label = f"+{delta:.1f} pp\n{mult:.1f}×"
    ax.text(i, ANNOT_Y, label, ha="center", va="center",
            fontsize=fontsize, color=color, fontweight=weight,
            bbox=dict(boxstyle="round,pad=0.40",
                      facecolor=ps.LIGHT_GOLD if is_dominant else "white",
                      edgecolor=color, linewidth=1.2 if is_dominant else 0.8))

# Connect each callout to its stage with a faint guide line
for i in range(len(stages)):
    ax.plot([i, i], [max(gpt4o[i], ours[i]) + 3.5, ANNOT_Y - 6],
            color=ps.DIVIDER, linewidth=0.8, linestyle=":", zorder=0)

ax.set_xticks(xs)
ax.set_xticklabels([f"{code}\n{name}" for code, name in stages],
                   fontsize=13, fontweight="medium")
ax.set_ylabel("Per-stage state accuracy (%)", fontsize=14)
ax.set_ylim(0, 168)
ax.set_yticks([0, 20, 40, 60, 80, 100])

# Headline and subtitle in figure-level top strip — outside the data area
fig.text(0.5, 0.95,
         "Per-stage state accuracy:  paper baseline vs. open-weight integration  (n = 681)",
         ha="center", va="center", fontsize=18, fontweight="bold", color=ps.NAVY)
fig.text(0.5, 0.90,
         "Multipliers of 6.6× – 9.2× on stages C, D, and E account for the +29.45 pp overall lift",
         ha="center", va="center", fontsize=12, color=ps.GRAY, style="italic")

# Legend at the BOTTOM, full-width, two rows so it doesn't compete with the overall callout
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
          fontsize=12, frameon=False, ncol=2)

ax.grid(axis="y", color=ps.DIVIDER, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# Overall lift callout — placed INSIDE the plot, top-right, anchored to axes
# (clear separation from the title block above + from the multiplier strip below).
ax.text(0.99, 0.98,
        r"Overall accuracy:  25.94%  $\rightarrow$  55.39%" "\n"
        r"($\Delta$ = +29.45 pp,  2.14$\times$ multiplier)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=14, fontweight="bold", color=ps.RED,
        bbox=dict(boxstyle="round,pad=0.60", facecolor=ps.LIGHT_RED,
                  edgecolor=ps.RED, linewidth=1.5))

ps.save_fig(fig, "F_per_stage_2system", __file__)
