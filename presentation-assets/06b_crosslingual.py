"""Chart 6b - Cross-lingual translation reproduces the published headline.

Data from paper Table 9 (app:critique) and §3.3. Running SocratTeachLLM against
SocratDataset-EN reproduces R-1 = 57.40 / R-2 = 33.63 within 1.5 R-1 points.
The R-2/BLEU-4 ratio jumps 1.3 (ZH) -> ~9 (EN), the textbook 4-gram-loss
signature under translation.
"""

import matplotlib.pyplot as plt
import numpy as np
import poster_style as ps

ps.apply()

# (metric, published_ZH, ours_EN)
metrics = ["R-1", "R-2", "BLEU-4"]
published_ZH = [57.40, 33.63, 41.96]
ours_EN = [55.85, 33.79, 3.56]

# Build figure with breathing room above and to the right
fig, ax = plt.subplots(figsize=(12.5, 6.8))
fig.subplots_adjust(top=0.78, right=0.78, left=0.08, bottom=0.12)

x = np.arange(len(metrics))
bar_w = 0.34

bars_pub = ax.bar(
    x - bar_w / 2,
    published_ZH,
    bar_w,
    color=ps.SLATE,
    alpha=0.85,
    edgecolor=ps.NAVY,
    linewidth=0.9,
    label="Chinese (paper baseline)  ·  GPT-4o consultant + SocratTeachLLM teacher",
)
bars_ours = ax.bar(
    x + bar_w / 2,
    ours_EN,
    bar_w,
    color=ps.RED,
    alpha=0.85,
    edgecolor=ps.NAVY,
    linewidth=0.9,
    label="English (our translation)  ·  Sonnet consultant + SocratTeachLLM teacher",
)

# value labels — positioned with clear vertical separation from any annotation
for xv, v in zip(x - bar_w / 2, published_ZH):
    ax.text(
        xv,
        v + 1.5,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=ps.NAVY,
    )
for xv, v in zip(x + bar_w / 2, ours_EN):
    ax.text(
        xv,
        v + 1.5,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=ps.NAVY,
    )


# Δ annotations placed ABOVE the value labels in dedicated airspace
# Reserve y in [66..80] for these annotations.
def delta_label(ax, idx, delta, y, color=ps.NAVY):
    xl = idx - bar_w / 2
    xr = idx + bar_w / 2
    ax.annotate(
        "", xy=(xr, y), xytext=(xl, y), arrowprops=dict(arrowstyle="<->", color=color, lw=1.6)
    )
    sign = "+" if delta >= 0 else "−"
    ax.text(
        idx,
        y + 1.4,
        f"Δ = {sign}{abs(delta):.2f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=color,
    )


delta_label(ax, 0, ours_EN[0] - published_ZH[0], 66, color=ps.NAVY)
delta_label(ax, 1, ours_EN[1] - published_ZH[1], 42, color=ps.RED)
delta_label(ax, 2, ours_EN[2] - published_ZH[2], 50, color=ps.GOLD)

# Right-side commentary box — placed in the figure-level whitespace outside ax
fig.text(
    0.79,
    0.55,
    "4-gram fingerprints\nare lost in translation\nBLEU-4 collapses 92%",
    ha="left",
    va="center",
    fontsize=12,
    color=ps.GOLD,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.55", facecolor=ps.LIGHT_GOLD, edgecolor=ps.GOLD, linewidth=1.2),
)

# Headline and subtitle in the figure-level top strip
fig.text(
    0.5,
    0.93,
    "Translating the dataset reproduces the published headline within 1.5 R-1 pts",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=ps.NAVY,
)
fig.text(
    0.5,
    0.87,
    "R-2 / BLEU-4 ratio jumps:  ≈ 1.3 (ZH)  to  ≈ 9 (EN)  ·  exactly the n-gram length where memorization lives",
    ha="center",
    va="center",
    fontsize=12,
    color=ps.GRAY,
    style="italic",
)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=15, fontweight="medium")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 80)

# Legend below the plot area, full-width — no overlap risk
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=12, frameon=False, ncol=2)

ax.grid(axis="y", color=ps.DIVIDER, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

ps.save_fig(fig, "06b_crosslingual", __file__)
