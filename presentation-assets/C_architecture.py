"""Diagram C - Architecture comparison: KELE original vs ours.

Top row: KELE two-agent stack (GPT-4o consultant + SocratTeachLLM teacher
fine-tuned on SocratDataset). Bottom row: our deterministic-classifier
integration (Qwen3.5-LoRA classifier + Gemma 4 31B teacher + 10-shot exemplars).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import poster_style as ps

ps.apply()

# Wider canvas so arrows have real horizontal runway
fig, ax = plt.subplots(figsize=(17, 9.5))
ax.set_xlim(0, 17)
ax.set_ylim(0, 11.5)
ax.set_aspect("equal")
ax.axis("off")

box_h = 1.55

# Y-bands (top down):
#   10.5–11.0 : title
#   9.6–10.2  : subtitle
#   8.3–6.75  : KELE row (boxes)
#   6.20      : KELE warning text
#   5.50–3.95 : Ours row (boxes)
#   3.40      : Ours green text
#   2.70–0.30 : comparison table
y_top = 6.75
y_bot = 3.95

def add_box(x, y, w, h, label, sub, fill, edge, lw=1.4, fs=12, fsub=10,
            text_color=ps.NAVY, sub_color=ps.GRAY):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.04,rounding_size=0.18",
                         facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, label,
            ha="center", va="center", fontsize=fs, fontweight="bold",
            color=text_color)
    if sub:
        ax.text(x + w / 2, y + h * 0.25, sub,
                ha="center", va="center", fontsize=fsub, color=sub_color,
                style="italic")

def add_arrow(x1, y, x2, color=ps.NAVY, lw=2.4):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=24))

# Row labels (left side)
ax.text(0.6, y_top + box_h / 2, "KELE\n(original)",
        ha="left", va="center", fontsize=15, fontweight="bold", color=ps.SLATE)
ax.text(0.6, y_bot + box_h / 2, "Ours\n(integration)",
        ha="left", va="center", fontsize=15, fontweight="bold", color=ps.RED)

# Box geometry — WIDER GAPS so arrows are clearly visible.
# Layout: 4 boxes with 1.0-wide gaps between them.
bx = [2.5, 5.8, 10.6, 14.3]
bw = [2.3, 3.6, 2.7, 2.3]
# Verify: 14.3 + 2.3 = 16.6, leaves 0.4 buffer on the right
# Gaps: 5.8 - (2.5+2.3) = 1.0;  10.6 - (5.8+3.6) = 1.2;  14.3 - (10.6+2.7) = 1.0

# ===== KELE row =====
add_box(bx[0], y_top, bw[0], box_h,
        "Student\nutterance", "+ Dialogue history", ps.LIGHT_BG, ps.NAVY)
add_box(bx[1], y_top, bw[1], box_h,
        "Consultant: GPT-4o", "JSON schema  ·  API ($)",
        ps.LIGHT_SLATE, ps.SLATE)
add_box(bx[2], y_top, bw[2], box_h,
        "Teacher:\nSocratTeachLLM", "GLM-4-9B + LoRA",
        ps.LIGHT_SLATE, ps.SLATE)
add_box(bx[3], y_top, bw[3], box_h,
        "Teacher\nresponse", "", ps.LIGHT_BG, ps.NAVY)

for i in range(3):
    add_arrow(bx[i] + bw[i] + 0.10, y_top + box_h / 2,
              bx[i + 1] - 0.10)

# KELE-row warning in its own clear strip ABOVE the boxes
ax.text((bx[1] + bw[1] + bx[2]) / 2, y_top + box_h + 0.62,
        "JSON schema  ·  21% Fallback rate at scale",
        ha="center", va="bottom", fontsize=11, color=ps.SLATE, style="italic",
        fontweight="bold")
ax.text((bx[1] + bw[1] + bx[2]) / 2, y_top + box_h + 0.28,
        "Fallback = teacher emits malformed JSON; system reverts to a two-call recovery path",
        ha="center", va="bottom", fontsize=10, color=ps.GRAY, style="italic")

# ===== Ours row =====
# Primary boxes (classifier + teacher response) take solid SCU red fill with
# white text for maximum brand pop. The center Gemma teacher box keeps the
# softer LIGHT_RED tint, providing breathing room and visual hierarchy.
add_box(bx[0], y_bot, bw[0], box_h,
        "Student\nutterance", "+ Dialogue history", ps.LIGHT_BG, ps.NAVY)
add_box(bx[1], y_bot, bw[1], box_h,
        "Classifier:\nQwen3.5-0.8B-LoRA",
        "1.6 GB  ·  CPU  ·  67.6% State acc",
        ps.RED, ps.RED, lw=1.8,
        text_color="white", sub_color="#FBE4EA")
add_box(bx[2], y_bot, bw[2], box_h,
        "Teacher:\nGemma 4 31B",
        "+ 10-shot exemplars",
        ps.LIGHT_RED, ps.RED)
add_box(bx[3], y_bot, bw[3], box_h,
        "Teacher\nresponse", "", ps.RED, ps.RED, lw=1.8,
        text_color="white", sub_color="white")

for i in range(3):
    add_arrow(bx[i] + bw[i] + 0.10, y_bot + box_h / 2,
              bx[i + 1] - 0.10)

# Ours-row note in clear strip BELOW the boxes
ax.text((bx[1] + bw[1] + bx[2]) / 2, y_bot - 0.30,
        "Deterministic SocRule lookup  ·  No JSON, no sampling variance",
        ha="center", va="top", fontsize=11, color=ps.RED, style="italic")

# ===== Title + subtitle =====
ax.text(8.5, 10.75,
        "Architecture:  LLM consultant  vs  supervised classifier consultant",
        ha="center", va="center", fontsize=18, fontweight="bold", color=ps.NAVY)
ax.text(8.5, 9.95,
        "Replace the consultant axis; isolate routing from response generation",
        ha="center", va="center", fontsize=13, color=ps.GRAY, style="italic")

# ===== Comparison table =====
tbl_x  = 0.6
tbl_y  = 0.30
tbl_w  = 15.8
tbl_h  = 2.30
col_label_w = 5.6
col_kele_w  = 5.1
col_ours_w  = 5.1
cx_label = tbl_x + col_label_w / 2
cx_kele  = tbl_x + col_label_w + col_kele_w / 2
cx_ours  = tbl_x + col_label_w + col_kele_w + col_ours_w / 2

tbl_border = FancyBboxPatch((tbl_x, tbl_y), tbl_w, tbl_h,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor=ps.LIGHT_BG,
                            edgecolor=ps.DIVIDER, linewidth=1.0)
ax.add_patch(tbl_border)

header_h = 0.42
header = Rectangle((tbl_x, tbl_y + tbl_h - header_h), tbl_w, header_h,
                   facecolor="#E2E8F0", edgecolor="none")
ax.add_patch(header)

for x_div in (tbl_x + col_label_w,
              tbl_x + col_label_w + col_kele_w):
    ax.plot([x_div, x_div], [tbl_y + 0.05, tbl_y + tbl_h - 0.05],
            color=ps.DIVIDER, linewidth=0.8)

header_y = tbl_y + tbl_h - header_h / 2
ax.text(cx_label, header_y, "Dimension",
        ha="center", va="center", fontsize=13, fontweight="bold", color=ps.NAVY)
ax.text(cx_kele, header_y, "KELE",
        ha="center", va="center", fontsize=14, fontweight="bold", color=ps.SLATE)
ax.text(cx_ours, header_y, "Ours",
        ha="center", va="center", fontsize=14, fontweight="bold", color=ps.RED)

rows = [
    ("Per-run eval API cost",  "~$15 / 1k turns",       "$0  (CPU classifier)"),
    ("Consultant footprint",   "API call",              "1.6 GB on CPU"),
    ("Teacher footprint",      "9B  ·  ~19 GB GPU",     "31B  ·  32 GB GPU"),
    ("Routing determinism",    "LLM-sampled",           "Deterministic"),
]

n_rows = len(rows)
data_top = tbl_y + tbl_h - header_h - 0.12
data_bot = tbl_y + 0.08
row_dy = (data_top - data_bot) / n_rows

for i, (lab, k, o) in enumerate(rows):
    y_row = data_top - (i + 0.5) * row_dy
    ax.text(cx_label, y_row, lab, ha="center", va="center",
            fontsize=11, color=ps.NAVY)
    ax.text(cx_kele, y_row, k, ha="center", va="center",
            fontsize=11, color=ps.SLATE)
    ax.text(cx_ours, y_row, o, ha="center", va="center",
            fontsize=11, color=ps.RED, fontweight="bold")
    if i < n_rows - 1:
        y_sep = y_row - row_dy / 2
        ax.plot([tbl_x + 0.05, tbl_x + tbl_w - 0.05],
                [y_sep, y_sep],
                color=ps.DIVIDER, linewidth=0.5)

ps.save_fig(fig, "C_architecture", __file__)
