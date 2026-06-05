"""Diagram D - Per-turn integration algorithm box (paper Algorithm 1).

Two-column layout with clean spacing:
  · Left: the 7-line algorithm as a code card (the WHAT)
  · Right: 'What changes vs. KELE' sidebar with 4 before-after contrasts
    (the WHY and HOW it differs from KELE's LLM-consultant + JSON-schema)

Inline code comments removed - explanations live in the sidebar instead, so
the algorithm card is uncluttered and every contrast block has room to breathe.
"""

import matplotlib.pyplot as plt
import poster_style as ps
from matplotlib.patches import FancyBboxPatch

ps.apply()

fig, ax = plt.subplots(figsize=(15.5, 9.0))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 9.0)
ax.set_aspect("auto")
ax.axis("off")

BLACK = "#000000"

# =============================================================================
# Title strip
# =============================================================================
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
    "Replaces KELE's LLM-consultant + JSON-schema call with a CPU classifier + deterministic SocRule lookup",
    ha="center",
    va="center",
    fontsize=11,
    color=ps.GRAY,
    style="italic",
)

# =============================================================================
# LEFT: Algorithm card (no inline comments - clean code only)
# =============================================================================
card_x = 0.40
card_y = 0.95
card_w = 7.20
card_h = 6.25

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

# Card title
ax.text(
    card_x + card_w / 2,
    card_y + card_h - 0.40,
    "Algorithm 1  ·  Per-turn pipeline",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
    color=ps.RED,
)

# Column anchors
x_label = card_x + 0.45
x_code = card_x + 1.65

# Require / Ensure (header rows)
y_req_a = card_y + card_h - 1.05
y_req_b = card_y + card_h - 1.50
y_ens = card_y + card_h - 2.05

ax.text(
    x_label,
    y_req_a,
    "Require:",
    ha="left",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=BLACK,
    family="monospace",
)
ax.text(
    x_code,
    y_req_a,
    "dialogue H, utterance u_t,",
    ha="left",
    va="center",
    fontsize=10,
    color=BLACK,
    family="monospace",
)
ax.text(
    x_code,
    y_req_b,
    "classifier C, teacher T, map A, exemplars E",
    ha="left",
    va="center",
    fontsize=10,
    color=BLACK,
    family="monospace",
)

ax.text(
    x_label,
    y_ens,
    "Ensure:",
    ha="left",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=BLACK,
    family="monospace",
)
ax.text(
    x_code,
    y_ens,
    "response r_t, next state s_t",
    ha="left",
    va="center",
    fontsize=10,
    color=BLACK,
    family="monospace",
)

# Divider
y_div = card_y + card_h - 2.45
ax.plot(
    [card_x + 0.30, card_x + card_w - 0.30],
    [y_div, y_div],
    color=BLACK,
    linewidth=1.0,
    alpha=0.4,
)

# Algorithm steps (no inline comments - see sidebar for WHY)
x_lineno = card_x + 0.40
x_step = card_x + 1.05

steps = [
    ("1:", "s_t = C(format(H, u_t))"),
    ("2:", "if s_t.stage < current_stage(H):"),
    ("3:", "    s_t = snap_forward(s_t)"),
    ("4:", "a_t = A[s_t]"),
    ("5:", "E_10 = stage_balanced_sample(E, k=10)"),
    ("6:", "r_t = T(H, u_t, a_t, E_10)"),
    ("7:", "return (r_t, s_t)"),
]

y_step_start = y_div - 0.50
dy = 0.50

for i, (lineno, code) in enumerate(steps):
    y = y_step_start - i * dy
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
        alpha=0.55,
    )
    ax.text(
        x_step,
        y,
        code,
        ha="left",
        va="center",
        fontsize=12,
        color=BLACK,
        family="monospace",
    )

# =============================================================================
# RIGHT: Sidebar - "What changes vs. KELE" (4 clean contrast blocks)
# =============================================================================
side_x = 7.85
side_y = 0.95
side_w = 7.25
side_h = 6.25

side = FancyBboxPatch(
    (side_x, side_y),
    side_w,
    side_h,
    boxstyle="round,pad=0.04,rounding_size=0.22",
    facecolor=ps.LIGHT_BG,
    edgecolor=ps.NAVY,
    linewidth=1.6,
)
ax.add_patch(side)

# Sidebar header
ax.text(
    side_x + side_w / 2,
    side_y + side_h - 0.40,
    "What the algorithm guarantees",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    color=ps.NAVY,
)

# Four algorithm-specific insights (NOT architectural - those live in panel C).
# Each block shows: title + line reference + 2-line description of what the
# algorithm structurally enforces that wouldn't happen without it.
insights = [
    (
        "Monotonic stage progression",
        "lines 2–3",
        "Socratic teaching can only move forward. snap_forward "
        "enforces this even if the classifier predicts an earlier stage.",
    ),
    (
        "Stage-balanced exemplar sampling",
        "line 5",
        "2 exemplars per SocRule stage. Without balancing, stage c "
        "(33.7% of the corpus) would dominate the teacher's example pool.",
    ),
    (
        "Explicit state-machine threading",
        "line 7",
        "Each turn returns s_t for the next turn's monotonicity check. "
        "Dialogue is a debuggable state machine, not an LLM-implicit buffer.",
    ),
    (
        "End-to-end determinism",
        "lines 1, 4, 5",
        "Classifier + state-action lookup + fixed-seed sampling. "
        "Same input always produces the same output. Reproducible and audit-able.",
    ),
]

# Block layout - give each block clear vertical space
y_block_top = side_y + side_h - 1.00
y_block_bot = side_y + 0.35
total_block_area = y_block_top - y_block_bot
block_h = total_block_area / len(insights)

x_text = side_x + 0.40
x_text_right = side_x + side_w - 0.30

for i, (title, line_ref, description) in enumerate(insights):
    y_block = y_block_top - i * block_h

    # Block title (red bold)
    ax.text(
        x_text,
        y_block - 0.05,
        title,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=ps.RED,
    )
    # Line reference (small italic, right-aligned)
    ax.text(
        x_text_right,
        y_block - 0.05,
        line_ref,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=ps.RED,
        style="italic",
    )

    # Description (navy, 2 lines via textwrap)
    import textwrap

    wrapped = textwrap.fill(description, width=68)
    ax.text(
        x_text + 0.05,
        y_block - 0.50,
        wrapped,
        ha="left",
        va="top",
        fontsize=10,
        color=ps.NAVY,
        linespacing=1.35,
    )

    # Divider between blocks (except last)
    if i < len(insights) - 1:
        y_sep = y_block - block_h + 0.10
        ax.plot(
            [side_x + 0.30, side_x + side_w - 0.30],
            [y_sep, y_sep],
            color=ps.DIVIDER,
            linewidth=0.8,
        )

# =============================================================================
# Footer
# =============================================================================
ax.text(
    7.75,
    0.45,
    "Takeaway: each numbered step encodes a structural guarantee that an LLM-consultant pipeline cannot provide",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color=ps.RED,
)

ps.save_fig(fig, "D_algorithm_box", __file__)
