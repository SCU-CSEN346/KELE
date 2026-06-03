"""Shared style for poster figures.

SCU-branded palette: SCU Bronco Red as the primary "our work" color, SCU
Mission Gold as the accent, slate for STL/contamination/legacy comparisons,
navy for text and frontier ceiling.
"""

from __future__ import annotations

import matplotlib as mpl

# --- Primary palette (SCU brand) ---
RED = "#A32035"  # SCU Bronco Red (primary brand)
GOLD = "#DDB94E"  # SCU Mission Gold — accent / callouts / highlights

# --- Supporting palette ---
NAVY = "#0F172A"  # text, axes, frontier ceiling
SLATE = "#475569"  # STL / contamination / legacy "bad" comparison color
GRAY = "#6B7280"  # neutral baseline (older / non-headline configs)
TEAL = "#0F766E"  # tertiary (code comments in D — IDE convention)
AMBER = "#D97706"  # kept as fallback for warning callouts

# --- Backgrounds / structure ---
LIGHT_BG = "#F8FAFC"
DIVIDER = "#E5E7EB"
LIGHT_RED = "#FBE4EA"  # light SCU-red tint for our boxes
LIGHT_GOLD = "#FAF0D0"  # light SCU-gold tint for accent backgrounds
LIGHT_SLATE = "#E2E8F0"  # light slate for STL/legacy boxes

# Inter not installed locally; Helvetica Neue is the closest modern-sans match.
FONT_STACK = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def apply():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.labelsize": 15,
            "axes.labelweight": "medium",
            "axes.edgecolor": NAVY,
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "text.color": NAVY,
        }
    )


def annotate_bar(ax, x, y, text, *, color=NAVY, weight="bold", offset=0.6, fontsize=12):
    ax.text(
        x,
        y + offset,
        text,
        ha="center",
        va="bottom",
        color=color,
        fontsize=fontsize,
        fontweight=weight,
    )


def save_fig(fig, name, caller_file):
    """Save fig as png (under png/) and pdf (under pdf/) at 300 dpi.
    Pass `__file__` from the calling script; we resolve to its parent dir."""
    from pathlib import Path

    here = Path(caller_file).resolve().parent
    png_dir = here / "png"
    pdf_dir = here / "pdf"
    png_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)
    fig.savefig(png_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{name}.pdf", bbox_inches="tight")
    print(f"wrote png/{name}.png + pdf/{name}.pdf")
