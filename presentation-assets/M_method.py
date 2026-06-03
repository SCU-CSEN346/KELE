"""Graphic M - Method section (3 hero-metric rows).

Three method choices, each paired with the measurable outcome it produced.
Hero metrics on the left anchor the eye; method choice + body on the right.
Pipeline axes: consultant (row 1) -> prompt engineering (row 2) -> teacher (row 3).

Sized to match H_conclusion + D_algorithm_box (15.5" x 9.0").

The 500-dialogue tournament detail (lever sweep + per-lever results) lives in
the companion N graphic. Campaign scale (143 variants) lives there too.
"""
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import poster_style as ps
from pathlib import Path

ps.apply()

fig, ax = plt.subplots(figsize=(15.5, 9.0))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 9.0)
ax.set_aspect("auto")
ax.axis("off")

# Each row: method CHOICE -> measurable OUTCOME. Rows correspond to the three
# pipeline axes (consultant / prompt-engineering / teacher).
rows = [
    {
        "metric": "30×",
        "metric_label": "faster routing per turn",
        "headline": "Deterministic classifier consultant",
        "body": (
            "A 0.8B-parameter Qwen3.5-LoRA on CPU replaces KELE's LLM consultant. "
            "~50 ms forward pass vs. ~1.5 s API round-trip. "
            "Routing is deterministic and reproducible, not sampled."
        ),
    },
    {
        "metric": "500",
        "metric_label": "dialogues across 10 levers",
        "headline": "10-shot stage-balanced exemplar tournament",
        "body": (
            "Swept length budget, persona, CoT scaffold, N-best rerank, +6 more. "
            "Winner: 10-shot stage-balanced exemplars, 2 per SocRule stage. "
            "Plain text generation removes KELE's JSON schema and its 21% fallback at scale."
        ),
    },
    {
        "metric": "$0",
        "metric_label": "per-run evaluation cost",
        "headline": "Open-weight teacher on one consumer GPU",
        "body": (
            "Gemma 4 31B served via vLLM on a single RTX 5090 (32 GB). "
            "No proprietary API, no rented hardware. "
            "Frontier API teachers cost ~$15 per full n=681 evaluation."
        ),
    },
]

# --- Layout: 3 hero rows, breathing room top + bottom ---
top_margin = 0.45
bottom_margin = 0.45
gap = 0.35
n = len(rows)
available = 9.0 - top_margin - bottom_margin - (n - 1) * gap
row_h = available / n  # ~2.47

# Column anchors
CARD_X = 0.20
CARD_W = 15.10
HERO_CENTER_X = 1.90
DIVIDER_X = 3.65
TEXT_X = 3.95

BODY_WRAP = 88

for i, row in enumerate(rows):
    y_top = 9.0 - top_margin - i * (row_h + gap)
    y_bot = y_top - row_h
    y_center = (y_top + y_bot) / 2

    # Card background (light red fill, red border)
    card = FancyBboxPatch(
        (CARD_X, y_bot),
        CARD_W,
        row_h,
        boxstyle="round,pad=0.04,rounding_size=0.22",
        facecolor=ps.LIGHT_RED,
        edgecolor=ps.RED,
        linewidth=1.6,
    )
    ax.add_patch(card)

    # --- Hero metric (left) ---
    ax.text(
        HERO_CENTER_X,
        y_center + 0.28,
        row["metric"],
        ha="center",
        va="center",
        fontsize=58,
        fontweight="bold",
        color=ps.RED,
    )
    ax.text(
        HERO_CENTER_X,
        y_center - 0.78,
        row["metric_label"],
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="medium",
        color=ps.SLATE,
        style="italic",
    )

    # Vertical divider between hero metric and right column
    ax.plot(
        [DIVIDER_X, DIVIDER_X],
        [y_bot + 0.30, y_top - 0.30],
        color=ps.RED,
        linewidth=1.2,
        alpha=0.55,
    )

    # --- Headline + body (right) ---
    ax.text(
        TEXT_X,
        y_top - 0.55,
        row["headline"],
        ha="left",
        va="center",
        fontsize=21,
        fontweight="bold",
        color=ps.RED,
    )

    body_wrapped = textwrap.fill(row["body"], width=BODY_WRAP)
    ax.text(
        TEXT_X,
        y_top - 1.15,
        body_wrapped,
        ha="left",
        va="top",
        fontsize=14,
        color=ps.NAVY,
        linespacing=1.45,
    )

# Save preserving full canvas
here = Path(__file__).resolve().parent
png_dir = here / "png"
pdf_dir = here / "pdf"
png_dir.mkdir(exist_ok=True)
pdf_dir.mkdir(exist_ok=True)
with mpl.rc_context({"savefig.bbox": "standard"}):
    fig.savefig(png_dir / "M_method.png", dpi=300)
    fig.savefig(pdf_dir / "M_method.pdf")
print("wrote png/M_method.png + pdf/M_method.pdf  (15.5x9.0 matching H + D)")
