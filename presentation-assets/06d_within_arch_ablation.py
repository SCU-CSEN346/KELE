"""Chart 6d - Within-architecture base-model ablation.

Data from paper Table 6 (sec:results-critique). Matched n=681; the only variable
between paired rows is the LoRA adapter. STL adds ~8pp state accuracy but ~15pp
ROUGE-1 over its own pre-trained base. Surface-form inflation = 1.68-1.85x.
"""

import matplotlib.pyplot as plt
import numpy as np
import poster_style as ps

ps.apply()

# Table 6 numbers (state acc %, R-1 %)
data = {
    "bert-fixed": {"base": (43.71, 32.30), "stl": (52.66, 47.30)},
    "qwen3.5": {"base": (48.72, 33.03), "stl": (56.50, 47.45)},
}

consultants = list(data.keys())
fig, ax = plt.subplots(figsize=(13, 7.5))
fig.subplots_adjust(top=0.84, bottom=0.18, left=0.07, right=0.97)

group_w = 0.85
bar_w = group_w / 4
xs = np.arange(len(consultants))

offsets = {
    "base_state": -1.5 * bar_w,
    "stl_state": -0.5 * bar_w,
    "base_r1": 0.5 * bar_w,
    "stl_r1": 1.5 * bar_w,
}

for i, c in enumerate(consultants):
    base_state, base_r1 = data[c]["base"]
    stl_state, stl_r1 = data[c]["stl"]

    ax.bar(
        i + offsets["base_state"],
        base_state,
        bar_w,
        color=ps.NAVY,
        edgecolor=ps.NAVY,
        linewidth=0.8,
        label="GLM-base · State acc" if i == 0 else None,
    )
    ax.bar(
        i + offsets["stl_state"],
        stl_state,
        bar_w,
        color=ps.RED,
        alpha=0.55,
        edgecolor=ps.NAVY,
        linewidth=0.8,
        label="STL · State acc" if i == 0 else None,
    )
    ax.bar(
        i + offsets["base_r1"],
        base_r1,
        bar_w,
        color=ps.NAVY,
        edgecolor=ps.NAVY,
        linewidth=0.8,
        hatch="//",
        label="GLM-base · ROUGE-1" if i == 0 else None,
    )
    ax.bar(
        i + offsets["stl_r1"],
        stl_r1,
        bar_w,
        color=ps.RED,
        edgecolor=ps.NAVY,
        linewidth=0.8,
        hatch="//",
        label="STL · ROUGE-1" if i == 0 else None,
    )

    # value labels (just above each bar top)
    for off_key, val in [
        ("base_state", base_state),
        ("stl_state", stl_state),
        ("base_r1", base_r1),
        ("stl_r1", stl_r1),
    ]:
        ax.text(
            i + offsets[off_key],
            val + 0.6,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=ps.NAVY,
            fontweight="medium",
        )

    # Δ-state bracket — placed clearly above the bars
    dx_left = i + offsets["base_state"]
    dx_right = i + offsets["stl_state"]
    bracket_y_state = max(base_state, stl_state) + 7.0
    ax.annotate(
        "",
        xy=(dx_right, bracket_y_state),
        xytext=(dx_left, bracket_y_state),
        arrowprops=dict(arrowstyle="<->", color=ps.NAVY, lw=1.4),
    )
    d_state = stl_state - base_state
    ax.text(
        (dx_left + dx_right) / 2,
        bracket_y_state + 1.4,
        f"Δstate  +{d_state:.2f} pp",
        ha="center",
        va="bottom",
        fontsize=11,
        color=ps.NAVY,
        fontweight="bold",
    )

    # Δ-R1 bracket — placed at a different height to avoid collision with Δstate
    dx_left = i + offsets["base_r1"]
    dx_right = i + offsets["stl_r1"]
    bracket_y_r1 = max(base_r1, stl_r1) + 7.0
    ax.annotate(
        "",
        xy=(dx_right, bracket_y_r1),
        xytext=(dx_left, bracket_y_r1),
        arrowprops=dict(arrowstyle="<->", color=ps.AMBER, lw=1.7),
    )
    d_r1 = stl_r1 - base_r1
    ratio = d_r1 / d_state
    ax.text(
        (dx_left + dx_right) / 2,
        bracket_y_r1 + 1.4,
        f"ΔR-1  +{d_r1:.2f} pp   ({ratio:.2f}x)",
        ha="center",
        va="bottom",
        fontsize=11,
        color=ps.GOLD,
        fontweight="bold",
    )

# group separator
ax.axvline(0.5, color=ps.DIVIDER, linewidth=1.0, linestyle=":")

ax.set_xticks(xs)
_CONSULTANT_LABEL = {"bert-fixed": "BERT-fixed", "qwen3.5": "Qwen3.5"}
ax.set_xticklabels(
    [f"Consultant: {_CONSULTANT_LABEL[c]}" for c in consultants], fontsize=14, fontweight="medium"
)
ax.set_ylabel("Score (%)", fontsize=14)
ax.set_ylim(0, 80)

# Title and subtitle in figure-level top strip
fig.text(
    0.5,
    0.95,
    "Did SocratTeachLLM actually learn to teach better than its base model?",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=ps.NAVY,
)
fig.text(
    0.5,
    0.90,
    "Within-architecture ablation (n = 681)  ·  SocratTeachLLM = GLM-4-9B-Chat (base) + a LoRA adapter trained on KELE's SocratDataset",
    ha="center",
    va="center",
    fontsize=12,
    color=ps.GRAY,
    style="italic",
)

# Headline takeaway box — top-right, in clean airspace
fig.text(
    0.97,
    0.88,
    "Result: LoRA adds +8 pp state acc\nbut +15 pp ROUGE-1 over its base.\nThe extra surface-form inflation\nIS the memorization overlay.",
    ha="right",
    va="top",
    fontsize=11,
    color=ps.GOLD,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.55", facecolor=ps.LIGHT_GOLD, edgecolor=ps.GOLD, linewidth=1.3),
)

# Legend at the bottom, 4-col, frameless
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=11, ncol=4, frameon=False)
ax.grid(axis="y", color=ps.DIVIDER, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

ps.save_fig(fig, "06d_within_arch_ablation", __file__)
