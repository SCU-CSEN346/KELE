"""Diagram D - Per-turn integration algorithm box (paper Algorithm 1).

The 7-line algorithm formalizing the per-turn loop: classify, monotonicity snap,
deterministic SocRule lookup, stage-balanced exemplar sampling, generate.
Rendered as a code-card for the poster.
"""

import matplotlib.pyplot as plt
import poster_style as ps
from matplotlib.patches import FancyBboxPatch

ps.apply()

# Wider + taller canvas + uniform vertical rhythm
fig, ax = plt.subplots(figsize=(15.5, 9.0))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 9.0)
ax.set_aspect("auto")
ax.axis("off")

# --- Title strip ---
ax.text(
    7.75,
    8.55,
    "Per-turn integration pipeline",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color=ps.NAVY,
)
ax.text(
    7.75,
    8.05,
    "Algorithm 1  ·  7 deterministic steps  ·  No JSON schema  ·  CPU classifier",
    ha="center",
    va="center",
    fontsize=12,
    color=ps.GRAY,
    style="italic",
)

# --- Card boundary ---
# SCU Red fill, black text — makes the algorithm pipeline visually pop on the poster.
BLACK = "#000000"
card_x = 0.6
card_y = 0.65
card_w = 14.3
card_h = 6.55
card = FancyBboxPatch(
    (card_x, card_y),
    card_w,
    card_h,
    boxstyle="round,pad=0.04,rounding_size=0.22",
    facecolor=ps.LIGHT_RED,
    edgecolor=ps.RED,
    linewidth=1.8,
)
ax.add_patch(card)

# --- Column anchors (everything aligned to these) ---
# Left margin inside the card
x_lineno = card_x + 0.40  # 1.00
x_label = card_x + 0.95  # 1.55 — "Require:", "Ensure:", line numbers
x_code = card_x + 2.50  # 3.10
x_comment = card_x + 7.80  # 8.40

# --- Require / Ensure (header rows inside the card) ---
y_require_a = card_y + card_h - 0.65  # 6.55
y_require_b = card_y + card_h - 1.10  # 6.10
y_ensure = card_y + card_h - 1.65  # 5.55

ax.text(
    x_label,
    y_require_a,
    "Require:",
    ha="left",
    va="center",
    fontsize=12,
    fontweight="bold",
    color=BLACK,
    family="monospace",
)
ax.text(
    x_code,
    y_require_a,
    "dialogue history H,  utterance u_t,  classifier C,  teacher LLM T,",
    ha="left",
    va="center",
    fontsize=11,
    color=BLACK,
    family="monospace",
)
ax.text(
    x_code,
    y_require_b,
    "state-action map A,  exemplar pool E",
    ha="left",
    va="center",
    fontsize=11,
    color=BLACK,
    family="monospace",
)

ax.text(
    x_label,
    y_ensure,
    "Ensure:",
    ha="left",
    va="center",
    fontsize=12,
    fontweight="bold",
    color=BLACK,
    family="monospace",
)
ax.text(
    x_code,
    y_ensure,
    "teacher response r_t,  next state s_t",
    ha="left",
    va="center",
    fontsize=11,
    color=BLACK,
    family="monospace",
)

# --- Divider line (horizontal, separating Require/Ensure header from steps) ---
y_div = card_y + card_h - 2.10  # 5.10
ax.plot(
    [card_x + 0.30, card_x + card_w - 0.30], [y_div, y_div], color=BLACK, linewidth=1.2, alpha=0.45
)

# --- Numbered steps with uniform vertical rhythm ---
steps = [
    ("1:", "s_t = C(format(H, u_t))", "34-way SocRule state from classifier"),
    ("2:", "if s_t.stage < current_stage(H):", "Enforce monotonic stage progression"),
    ("3:", "    s_t = snap_forward(s_t)", ""),
    ("4:", "a_t = A[s_t]", "Deterministic SocRule strategy lookup"),
    ("5:", "E_10 = stage_balanced_sample(E, k=10)", "10 exemplars, 2 per stage"),
    ("6:", "r_t = T(H, u_t, a_t, E_10)", "Teacher generates response (no JSON)"),
    ("7:", "return (r_t, s_t)", ""),
]

y0 = y_div - 0.55  # 4.55
dy = 0.62  # uniform line height

for i, (lineno, code, comment) in enumerate(steps):
    y = y0 - i * dy
    ax.text(
        x_lineno,
        y,
        lineno,
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=BLACK,
        family="monospace",
        alpha=0.65,
    )
    ax.text(
        x_code,
        y,
        code,
        ha="left",
        va="center",
        fontsize=12,
        color=BLACK,
        family="monospace",
        fontweight="medium",
    )
    if comment:
        ax.text(
            x_comment,
            y,
            "// " + comment,
            ha="left",
            va="center",
            fontsize=11,
            color=BLACK,
            style="italic",
            family="monospace",
            alpha=0.80,
        )

# --- Footer callout ---
ax.text(
    7.75,
    0.30,
    "Line 1 replaces the LLM-consultant call  ·  Line 4 replaces JSON-schema routing",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color=ps.GOLD,
)

ps.save_fig(fig, "D_algorithm_box", __file__)
