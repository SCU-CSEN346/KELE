"""Graphic E - Evaluation panel: four LLM-judge axes (minimal version).

Four horizontal SCU-red cards, one per axis. Axis name is the hero; the
score range sits beneath it as the description. No title, no footer — the
cards speak for themselves.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import poster_style as ps
from pathlib import Path

ps.apply()

fig, ax = plt.subplots(figsize=(22, 9))
ax.set_xlim(0, 22)
ax.set_ylim(0, 9)
ax.set_aspect("auto")
ax.axis("off")

# =============================================================================
#  Four axis cards
# =============================================================================
axes_data = [
    {"name": "Socratic\nValidity", "range": "0 - 3"},
    {"name": "Learning\nAdvancement", "range": "0 - 3"},
    {"name": "Age-\nAppropriateness", "range": "0 - 2"},
    {"name": "Question-Form\nFidelity", "range": "0 - 2"},
]

card_w = 5.10
card_gap = 0.30
y_card_top = 8.50
y_card_bot = 0.50
card_h = y_card_top - y_card_bot  # 8.00

cards_total_w = len(axes_data) * card_w + (len(axes_data) - 1) * card_gap
cards_start = (22 - cards_total_w) / 2

for i, axis in enumerate(axes_data):
    card_x = cards_start + i * (card_w + card_gap)

    # Card background — solid SCU red with white text
    card = FancyBboxPatch(
        (card_x, y_card_bot),
        card_w,
        card_h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=ps.RED,
        edgecolor=ps.RED,
        linewidth=2.0,
    )
    ax.add_patch(card)

    # Axis name — hero text (much larger, multi-line, white)
    # Capped at 38pt so the longest single word ("Appropriateness", 15 chars)
    # clears the 5.10" card width with breathing room.
    ax.text(
        card_x + card_w / 2,
        y_card_bot + card_h * 0.70,
        axis["name"],
        ha="center",
        va="center",
        fontsize=38,
        fontweight="bold",
        color="white",
        linespacing=0.95,
    )

    # Score range — large display number
    ax.text(
        card_x + card_w / 2,
        y_card_bot + card_h * 0.25,
        axis["range"],
        ha="center",
        va="center",
        fontsize=66,
        fontweight="bold",
        color="#FBE4EA",
    )

# Save preserving full canvas
here = Path(__file__).resolve().parent
png_dir = here / "png"
pdf_dir = here / "pdf"
png_dir.mkdir(exist_ok=True)
pdf_dir.mkdir(exist_ok=True)
with mpl.rc_context({"savefig.bbox": "standard"}):
    fig.savefig(png_dir / "E_evaluation.png", dpi=300)
    fig.savefig(pdf_dir / "E_evaluation.pdf")
print("wrote png/E_evaluation.png + pdf/E_evaluation.pdf  (22×7 four-axis card row)")
