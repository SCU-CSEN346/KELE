"""Graphic N - 10-lever prompt-engineering tournament table.

Companion to M_method's 500-dialogue tournament row. Detail panel: ranks all 10
levers by state accuracy at n=50, with ROUGE-1, BLEU-4, and a horizontal state-
accuracy bar. Winner (length budget) promoted to full n=681 evaluation.

Data: results/tournament-cell-*/metrics_summary.json (10 levers, n=50 each,
500 dialogues total).
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import poster_style as ps
from matplotlib.patches import FancyBboxPatch, Rectangle

ps.apply()

# Tournament results - ranked by state accuracy (descending)
levers = [
    # (lever_name, blurb, state_acc, rouge1, bleu4)
    ("Length budget", "Cap tokens per turn", 51.96, 39.91, 12.37),
    ("Persona", "Add teacher persona prefix", 51.27, 39.60, 12.38),
    ("CoT scaffold", "Chain-of-thought prompt", 50.89, 38.98, 9.55),
    ("Per-state exemplars", "1 exemplar per 34-state", 50.88, 39.42, 10.90),
    ("Negative exemplars", "Anti-patterns in prompt", 50.71, 39.82, 10.88),
    ("Compressed history", "Summarize prior turns", 47.50, 38.79, 9.82),
    ("N-best rerank", "Sample 5, pick best", 47.33, 38.27, 10.02),
    ("Style-matched exemplars", "Match exemplars to style", 46.72, 42.17, 12.04),
    ("Format retry", "Retry on schema failure", 46.45, 39.60, 10.55),
    ("Lexical priors", "Topic glossary in prompt", 43.89, 39.12, 9.88),
]

# Baseline reference: 10-shot stage-balanced (no extra lever) baseline observed
# across smoke-and-mini runs in this teacher class is ~46-47 state-acc; the
# tournament tests whether each lever ADDS lift on top of that base.
BASELINE_HINT = 46.71  # smoke-and-mini projected, mentioned in §2.2 of the paper

fig, ax = plt.subplots(figsize=(15.5, 9.0))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 9.0)
ax.set_aspect("auto")
ax.axis("off")

# =============================================================================
# Title strip
# =============================================================================
ax.text(
    7.75,
    8.55,
    "10-lever prompt-engineering tournament",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
    color=ps.NAVY,
)
ax.text(
    7.75,
    8.10,
    "500 dialogues across 10 prompt levers (n=50 each)  ·  ranked by state accuracy  ·  winner promoted to full n=681",
    ha="center",
    va="center",
    fontsize=12.5,
    color=ps.GRAY,
    style="italic",
)

# =============================================================================
# Table layout
# =============================================================================
# Column anchors
X_RANK = 0.55
X_LEVER = 1.10
X_BLURB = 4.35
X_STATE = 7.85  # state accuracy number (right-aligned)
X_BAR_L = 8.20  # bar start
X_BAR_R = 12.60  # bar end (full width)
X_R1 = 13.30  # rouge-1 (right-aligned)
X_BLEU = 14.65  # bleu-4  (right-aligned)
X_TABLE_R = 15.10

# Row band
y_header = 7.45
y_row_top = 7.00
row_h = 0.55

# Bar normalization: range over the 10 cells
state_min = min(row[2] for row in levers)
state_max = max(row[2] for row in levers)
bar_w_max = X_BAR_R - X_BAR_L


def bar_width(state_acc):
    # Anchor 40% -> 0 width, top value -> full width
    lo = 40.0
    hi = state_max + 0.5
    return bar_w_max * (state_acc - lo) / (hi - lo)


# --- Header row ---
ax.text(
    X_RANK, y_header, "#", ha="left", va="center", fontsize=11, fontweight="bold", color=ps.SLATE
)
ax.text(
    X_LEVER,
    y_header,
    "Lever",
    ha="left",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=ps.SLATE,
)
ax.text(
    X_BLURB,
    y_header,
    "What it does",
    ha="left",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=ps.SLATE,
)
ax.text(
    X_STATE,
    y_header,
    "State acc",
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=ps.SLATE,
)
ax.text(
    (X_BAR_L + X_BAR_R) / 2,
    y_header,
    "(visual)",
    ha="center",
    va="center",
    fontsize=10,
    color=ps.GRAY,
    style="italic",
)
ax.text(
    X_R1,
    y_header,
    "ROUGE-1",
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=ps.SLATE,
)
ax.text(
    X_BLEU,
    y_header,
    "BLEU-4",
    ha="right",
    va="center",
    fontsize=11,
    fontweight="bold",
    color=ps.SLATE,
)

# --- Data rows ---
for i, (lever, blurb, state, r1, bleu) in enumerate(levers):
    y = y_row_top - i * row_h
    is_winner = i == 0

    # Winner row highlight
    if is_winner:
        highlight = FancyBboxPatch(
            (X_RANK - 0.18, y - row_h / 2 + 0.04),
            X_TABLE_R - X_RANK + 0.10,
            row_h - 0.08,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=ps.LIGHT_RED,
            edgecolor=ps.RED,
            linewidth=1.4,
            zorder=0,
        )
        ax.add_patch(highlight)

    # Subtle alt-row banding for non-winner rows
    elif i % 2 == 1:
        band = Rectangle(
            (X_RANK - 0.10, y - row_h / 2 + 0.04),
            X_TABLE_R - X_RANK + 0.05,
            row_h - 0.08,
            facecolor=ps.LIGHT_SLATE,
            edgecolor="none",
            alpha=0.30,
            zorder=0,
        )
        ax.add_patch(band)

    row_color = ps.RED if is_winner else ps.NAVY
    row_weight = "bold" if is_winner else "normal"

    # Rank
    rank_str = f"{i + 1}"
    ax.text(
        X_RANK, y, rank_str, ha="left", va="center", fontsize=14, fontweight="bold", color=row_color
    )

    # Lever name. Row highlight + bold red text already signal the winner.
    ax.text(
        X_LEVER,
        y,
        lever,
        ha="left",
        va="center",
        fontsize=13.5,
        fontweight=row_weight,
        color=row_color,
    )

    # Blurb (one-line description)
    ax.text(
        X_BLURB, y, blurb, ha="left", va="center", fontsize=11.5, color=ps.SLATE, style="italic"
    )

    # State acc (right-aligned)
    ax.text(
        X_STATE,
        y,
        f"{state:.2f}",
        ha="right",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=row_color,
    )

    # Bar
    bw = bar_width(state)
    bar = Rectangle(
        (X_BAR_L, y - 0.13),
        bw,
        0.26,
        facecolor=ps.RED if is_winner else ps.SLATE,
        edgecolor="none",
        alpha=0.95 if is_winner else 0.65,
    )
    ax.add_patch(bar)

    # ROUGE-1 (right-aligned)
    ax.text(X_R1, y, f"{r1:.2f}", ha="right", va="center", fontsize=12.5, color=row_color)

    # BLEU-4 (right-aligned)
    ax.text(X_BLEU, y, f"{bleu:.2f}", ha="right", va="center", fontsize=12.5, color=row_color)

# =============================================================================
# Takeaway footer
# =============================================================================
y_takeaway = 0.55
ax.text(
    7.75,
    y_takeaway + 0.30,
    "Top-5 levers cluster within 1.3 pp - many help, few dominate.  "
    "Length-budget winner promoted to full n=681; the locked headline uses it on top of 10-shot stage-balanced exemplars.",
    ha="center",
    va="center",
    fontsize=12,
    color=ps.NAVY,
)
ax.text(
    7.75,
    y_takeaway - 0.20,
    "ROUGE-1 and BLEU-4 disagree with state accuracy:  style-matched exemplars top ROUGE-1 (42.17) while finishing 8th on pedagogy - prefigures the contamination panel.",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color=ps.RED,
    style="italic",
)

# --- Save preserving full canvas ---
here = Path(__file__).resolve().parent
png_dir = here / "png"
pdf_dir = here / "pdf"
png_dir.mkdir(exist_ok=True)
pdf_dir.mkdir(exist_ok=True)
with mpl.rc_context({"savefig.bbox": "standard"}):
    fig.savefig(png_dir / "N_tournament.png", dpi=300)
    fig.savefig(pdf_dir / "N_tournament.pdf")
print("wrote png/N_tournament.png + pdf/N_tournament.pdf  (15.5x9.0)")
