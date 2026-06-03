"""Graphic M.2 - Method stat-card banner.

Alternative to text-heavy M_method. Three pillars rendered as columns of stat
cards (4 cards each), with a 5-card hero results strip at the bottom. Wide-
banner format (28×16") for full-width placement on the poster with comfortable
vertical breathing room.

Card hierarchy (top → bottom in each card):
  Hero number  (large, SCU red)
  Primary label (bold navy)
  Sub-label    (small italic gray)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import poster_style as ps
from pathlib import Path

ps.apply()

fig, ax = plt.subplots(figsize=(28, 16))
ax.set_xlim(0, 28)
ax.set_ylim(0, 16)
ax.set_aspect("auto")
ax.axis("off")

# =============================================================================
#  Top half: three pillar sections, each with four supporting stat cards
# =============================================================================
section_w = 28 / 3  # 9.33

sections = [
    {
        "title": "Deterministic-classifier integration",
        "cards": [
            ("0.8B", "Classifier params", "Qwen3.5-LoRA"),
            ("1.6 GB", "CPU footprint", "no GPU for routing"),
            ("67.6%", "Test state acc", "4,304-turn split"),
            ("0%", "JSON fallback rate", "from 21% at scale"),
        ],
    },
    {
        "title": "10-shot exemplar tournament",
        "cards": [
            ("500", "Dialogues scored", "10 cells × n=50"),
            ("10", "Prompt levers", "CoT, persona, N-best..."),
            ("10-shot", "Winning template", "stage-balanced"),
            ("+1.58", "Top-lever gain", "length budget"),
        ],
    },
    {
        "title": "Open-weight SOTA teacher",
        "cards": [
            ("31B", "Teacher params", "Gemma 4 31B"),
            ("32 GB", "Consumer GPU", "RTX 5090"),
            ("$0", "Per-run API cost", "vs ~$15 frontier"),
            ("0", "Vendor lock-in", "open weights"),
        ],
    },
]

# Hero results strip — the bottom-line achievements
hero = [
    ("2.14×", "State-acc multiplier", "vs paper baseline"),
    ("+29.45 pp", "Absolute state-acc lift", "25.94% to 55.39%"),
    ("+19.25", "Unified-score lead", "vs paper baseline"),
    ("+2.18", "Unified-score lead", "vs frontier ceiling"),
    ("9.2×", "Peak per-stage multiplier", "stage d (resolution)"),
]

# --- Top-half layout coords (with generous vertical room) ---
y_section_header = 15.30
y_card_top = 14.40
y_card_bot = 8.40
card_h = y_card_top - y_card_bot  # 6.00

card_inner_w = 2.20
card_gap = 0.08

for s_idx, section in enumerate(sections):
    section_left = s_idx * section_w
    section_center = section_left + section_w / 2

    # Section title
    ax.text(
        section_center,
        y_section_header,
        section["title"],
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=ps.RED,
    )

    # 4 supporting cards in a row
    cards_total_w = 4 * card_inner_w + 3 * card_gap
    cards_start = section_left + (section_w - cards_total_w) / 2

    for c_idx, (number, label, sub) in enumerate(section["cards"]):
        card_x = cards_start + c_idx * (card_inner_w + card_gap)

        card = FancyBboxPatch(
            (card_x, y_card_bot),
            card_inner_w,
            card_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=ps.LIGHT_RED,
            edgecolor=ps.RED,
            linewidth=1.6,
        )
        ax.add_patch(card)

        # Hero number (large SCU red) — sized so longest tokens fit (e.g. "10-shot", "+1.58")
        ax.text(
            card_x + card_inner_w / 2,
            y_card_bot + card_h * 0.75,
            number,
            ha="center",
            va="center",
            fontsize=36,
            fontweight="bold",
            color=ps.RED,
        )

        # Primary label (bold navy)
        ax.text(
            card_x + card_inner_w / 2,
            y_card_bot + card_h * 0.42,
            label,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=ps.NAVY,
        )

        # Sub-label (italic gray)
        ax.text(
            card_x + card_inner_w / 2,
            y_card_bot + card_h * 0.18,
            sub,
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
            color=ps.GRAY,
        )

# Vertical dividers between sections (subtle)
for s_idx in range(1, 3):
    x_divider = s_idx * section_w
    ax.plot(
        [x_divider, x_divider],
        [y_card_bot - 0.40, y_card_top + 0.70],
        color=ps.DIVIDER,
        linewidth=1.0,
    )

# =============================================================================
#  Bottom: hero results strip — 5 solid-SCU-red cards
# =============================================================================
y_hero_header = 7.55
y_hero_card_top = 6.65
y_hero_card_bot = 0.45
hero_card_h = y_hero_card_top - y_hero_card_bot  # 6.20

# Hero strip header
ax.text(
    14,
    y_hero_header,
    "Headline outcome",
    ha="center",
    va="center",
    fontsize=24,
    fontweight="bold",
    color=ps.NAVY,
)

hero_card_w = 5.00
hero_gap = 0.45
hero_total_w = len(hero) * hero_card_w + (len(hero) - 1) * hero_gap
hero_start = (28 - hero_total_w) / 2

for h_idx, (number, label, sub) in enumerate(hero):
    card_x = hero_start + h_idx * (hero_card_w + hero_gap)

    card = FancyBboxPatch(
        (card_x, y_hero_card_bot),
        hero_card_w,
        hero_card_h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=ps.RED,
        edgecolor=ps.RED,
        linewidth=2.0,
    )
    ax.add_patch(card)

    # Hero number — bold white on SCU red
    ax.text(
        card_x + hero_card_w / 2,
        y_hero_card_bot + hero_card_h * 0.66,
        number,
        ha="center",
        va="center",
        fontsize=60,
        fontweight="bold",
        color="white",
    )

    # Hero label
    ax.text(
        card_x + hero_card_w / 2,
        y_hero_card_bot + hero_card_h * 0.30,
        label,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="white",
    )

    # Sub
    ax.text(
        card_x + hero_card_w / 2,
        y_hero_card_bot + hero_card_h * 0.14,
        sub,
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        color="#FBE4EA",
    )

# Save preserving full canvas (avoid bbox="tight" auto-cropping)
here = Path(__file__).resolve().parent
png_dir = here / "png"
pdf_dir = here / "pdf"
png_dir.mkdir(exist_ok=True)
pdf_dir.mkdir(exist_ok=True)
with mpl.rc_context({"savefig.bbox": "standard"}):
    fig.savefig(png_dir / "M2_method.png", dpi=300)
    fig.savefig(pdf_dir / "M2_method.pdf")
print("wrote png/M2_method.png + pdf/M2_method.pdf  (28×16 stat-card banner)")
