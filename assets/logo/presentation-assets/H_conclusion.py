"""Graphic H - Conclusion section (4 bullets).

Architectural and methodological contributions, limitations and future work,
and real-classroom deployment. Sized to match D_algorithm_box (15.5"×9.0").
"""

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import poster_style as ps
from matplotlib.patches import Circle

ps.apply()

# Match D_algorithm_box dimensions exactly.
fig, ax = plt.subplots(figsize=(15.5, 9.0))
ax.set_xlim(0, 15.5)
ax.set_ylim(0, 9.0)
ax.set_aspect("auto")
ax.axis("off")

# --- Section data (no em dashes; short sentences for readability) ---
sections = [
    (
        "Architectural contribution",
        "A 0.8B-parameter CPU classifier replaces the LLM consultant. "
        "Routing runs 30× faster per turn (~50 ms vs. ~1.5 s API round-trip) "
        "and eliminates the 21% JSON-schema fallback rate at scale. State "
        "accuracy lifts 2.14× (+29.45 pp) and the unified score rises +19.25 "
        "points over the GPT-4o + SocratTeachLLM paper baseline. All at $0 "
        "API cost on one 32 GB consumer GPU.",
    ),
    (
        "Methodological contribution",
        "Five converging diagnostics establish that ROUGE/BLEU on this "
        "benchmark reward training-data memorization, not transferable "
        "teaching capability. Three are shown above (cross-lingual translation, "
        "clean-probe collapse, within-architecture ablation); two more (rank "
        "inversion and n-gram-length scaling) are detailed in the paper.",
    ),
    (
        "Limitations and future work",
        "Result is specific to this teacher, prompt, and consultant "
        "(Chinese-language SocratDataset). The +2.18 frontier-overtaking "
        "margin is supported but bounded by the n=400 variance budget. "
        "A native English-domain benchmark and bilingual classifier "
        "co-training are next.",
    ),
    (
        "Real classrooms, today",
        "Elementary-school classrooms anywhere are within reach today. "
        "The full pipeline fits on a single consumer-grade GPU at near-zero "
        "per-run cost, and hardware like Nvidia's newly released N1 chip "
        "enables deployment on shared classroom rigs and individual student "
        "laptops alike. Completing the scaling work to every language and "
        "subject is how every student gets access, not only those at "
        "well-funded schools.",
    ),
]

# --- Layout ---
y_top_band = 8.85
y_bot_band = 0.15
band_total = y_top_band - y_bot_band
band_h = band_total / len(sections)

# Wrap width tuned so each bullet uses the full horizontal canvas and wraps
# to 3–4 lines at fontsize 13.
BODY_WRAP_WIDTH = 132

for i, (header, body) in enumerate(sections):
    y_band_top = y_top_band - i * band_h

    # Header row (badge + section header)
    y_header = y_band_top - 0.35

    badge = Circle(
        (0.45, y_header),
        0.28,
        facecolor=ps.RED,
        edgecolor="none",
        zorder=2,
    )
    ax.add_patch(badge)
    ax.text(
        0.45,
        y_header,
        str(i + 1),
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="white",
    )

    ax.text(
        0.90,
        y_header,
        header,
        ha="left",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=ps.RED,
    )

    wrapped = textwrap.fill(body, width=BODY_WRAP_WIDTH)
    ax.text(
        0.90,
        y_band_top - 0.78,
        wrapped,
        ha="left",
        va="top",
        fontsize=13,
        color=ps.NAVY,
        linespacing=1.40,
    )

# Save preserving full canvas (avoid bbox="tight" auto-cropping right margin).
here = Path(__file__).resolve().parent
png_dir = here / "png"
pdf_dir = here / "pdf"
png_dir.mkdir(exist_ok=True)
pdf_dir.mkdir(exist_ok=True)
with mpl.rc_context({"savefig.bbox": "standard"}):
    fig.savefig(png_dir / "H_conclusion.png", dpi=300)
    fig.savefig(pdf_dir / "H_conclusion.pdf")
print("wrote png/H_conclusion.png + pdf/H_conclusion.pdf  (15.5×9.0 to match D)")
