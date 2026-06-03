"""Graphic H - Conclusion section (3 bullets).

Architectural and methodological contributions, plus limitations and future
work. Self-contained text card for the poster's column-3 footer. SCU palette.
"""
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import poster_style as ps

ps.apply()

fig, ax = plt.subplots(figsize=(14, 10.0))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10.0)
ax.set_aspect("auto")
ax.axis("off")

# --- Title block (centered, with SCU-red separator strip beneath) ---
ax.text(7, 9.50, "Conclusion",
        ha="center", va="center", fontsize=26, fontweight="bold", color=ps.NAVY)

ax.plot([5.5, 8.5], [9.05, 9.05],
        color=ps.RED, linewidth=4.0, solid_capstyle="round")

# --- Section data (no em dashes; short sentences for readability) ---
sections = [
    (
        "Architectural contribution",
        "A 0.8B-parameter CPU classifier replaces the LLM consultant. "
        "This eliminates the 21% JSON-schema fallback rate at scale, "
        "lifts state accuracy by 2.14× (+29.45 pp), and raises the unified "
        "score by +19.25 points over the GPT-4o + SocratTeachLLM paper "
        "baseline. All at $0 API cost on one 32 GB consumer GPU.",
    ),
    (
        "Methodological contribution",
        "Five converging diagnostics establish that ROUGE and BLEU on this "
        "benchmark systematically reward training-data memorization rather "
        "than transferable teaching capability. The five diagnostics are "
        "surface-form rank inversion, monotonic n-gram-length scaling, "
        "cross-lingual translation reproduction, clean-probe collapse, "
        "and within-architecture base-model ablation.",
    ),
    (
        "Limitations and future work",
        "Result is specific to this teacher, prompt, and consultant "
        "(Chinese-language SocratDataset). Stage 2b QLoRA adapter on the "
        "Gemma teacher trained successfully, but evaluation is blocked by "
        "a train/serve prompt-format mismatch. Native English-domain "
        "benchmark and bilingual classifier co-training are next.",
    ),
]

# --- Layout per section ---
y_top_band = 8.75
y_bot_band = 1.20
band_total = y_top_band - y_bot_band
band_h = band_total / len(sections)

BODY_WRAP_WIDTH = 100

for i, (header, body) in enumerate(sections):
    y_band_top = y_top_band - i * band_h
    y_band_bot = y_band_top - band_h

    # Header row (badge + section header), positioned near band top
    y_header = y_band_top - 0.45

    # Numbered badge: solid SCU-red circle with white numeral
    badge = Circle((0.70, y_header), 0.35,
                   facecolor=ps.RED, edgecolor="none", zorder=2)
    ax.add_patch(badge)
    ax.text(0.70, y_header, str(i + 1),
            ha="center", va="center",
            fontsize=18, fontweight="bold", color="white")

    # Section header in SCU red
    ax.text(1.40, y_header, header,
            ha="left", va="center",
            fontsize=17, fontweight="bold", color=ps.RED)

    # Body text: textwrap-wrapped, navy, generous line spacing
    wrapped = textwrap.fill(body, width=BODY_WRAP_WIDTH)
    ax.text(1.40, y_band_top - 1.20, wrapped,
            ha="left", va="top",
            fontsize=12, color=ps.NAVY,
            linespacing=1.65)

    # Subtle divider between sections (except after the last)
    if i < len(sections) - 1:
        ax.plot([0.4, 13.6], [y_band_bot, y_band_bot],
                color=ps.DIVIDER, linewidth=0.8)

# --- Footer tagline (centered, italic, SCU red) ---
ax.text(7, 0.45,
        "The same five-diagnostic audit applies to any Socratic-teaching benchmark.",
        ha="center", va="center",
        fontsize=13, style="italic", fontweight="bold", color=ps.RED)

ps.save_fig(fig, "H_conclusion", __file__)
