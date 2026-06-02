"""Diagram A - SocRule 5-stage pipeline flow.

Data from paper Table 1 and §2.1. Five strictly-ordered stages: Student
Questioning, Concept Probing, Inductive Reasoning, Rule Construction, Teacher
Summary. Each box reports state count, turn count, and share of total turns
(percent of the 42,892-turn dataset). Stage C is the binding constraint -
22 states and 33.7% of all turns.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import poster_style as ps

ps.apply()

# (code, name, n_states, n_turns)
stages = [
    ("a", "Student\nQuestioning",  2,  6803),
    ("b", "Concept\nProbing",      6,  7480),
    ("c", "Inductive\nReasoning", 22, 14445),
    ("d", "Rule\nConstruction",    4,  7361),
    ("e", "Teacher\nSummary",      1,  6803),
]
TOTAL_TURNS = sum(s[3] for s in stages)  # 42,892

# Slightly taller boxes to host the new percent pill.
fig, ax = plt.subplots(figsize=(16, 6.2))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6.0)
ax.set_aspect("equal")
ax.axis("off")

box_w, box_h = 2.30, 3.45
gap = 0.80
total_w = len(stages) * box_w + (len(stages) - 1) * gap
x0 = (16 - total_w) / 2
y_box = 1.10

for i, (code, name, n_states, n_turns) in enumerate(stages):
    x = x0 + i * (box_w + gap)
    is_hard = code == "c"

    fill = ps.LIGHT_GOLD if is_hard else ps.LIGHT_RED
    edge = ps.GOLD if is_hard else ps.RED
    lw   = 2.2 if is_hard else 1.6

    box = FancyBboxPatch((x, y_box), box_w, box_h,
                         boxstyle="round,pad=0.04,rounding_size=0.18",
                         facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(box)

    # Header strip
    header_h = 0.55
    header = FancyBboxPatch((x, y_box + box_h - header_h), box_w, header_h,
                            boxstyle="round,pad=0.0,rounding_size=0.18",
                            facecolor=edge, edgecolor=edge, linewidth=0)
    ax.add_patch(header)
    ax.text(x + box_w / 2, y_box + box_h - header_h / 2 - 0.04,
            f"Stage {code}",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="white")

    # Stage name
    ax.text(x + box_w / 2, y_box + box_h - header_h - 0.55, name,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=ps.NAVY)

    # States pill
    ax.text(x + box_w / 2, y_box + 1.35,
            f"{n_states} state{'s' if n_states > 1 else ''}",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.30", facecolor=edge,
                      edgecolor="none"))

    # Turn count
    ax.text(x + box_w / 2, y_box + 0.78, f"{n_turns:,} turns",
            ha="center", va="center", fontsize=11, color=ps.NAVY)

    # Percent-of-total pill (new) — emphasised on the binding-constraint stage
    pct = n_turns / TOTAL_TURNS * 100
    pct_face   = ps.LIGHT_GOLD if is_hard else "#FFFFFF"
    pct_edge   = ps.GOLD if is_hard else ps.RED
    pct_text_c = ps.GOLD if is_hard else ps.RED
    pct_weight = "bold"
    pct_lw     = 1.4 if is_hard else 1.2
    ax.text(x + box_w / 2, y_box + 0.30,
            f"{pct:.1f}% of total",
            ha="center", va="center", fontsize=10.5, color=pct_text_c,
            fontweight=pct_weight,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=pct_face,
                      edgecolor=pct_edge, linewidth=pct_lw))

    # Arrow to the next box
    if i < len(stages) - 1:
        x_arr_start = x + box_w + 0.10
        x_arr_end   = x0 + (i + 1) * (box_w + gap) - 0.10
        ax.annotate("",
                    xy=(x_arr_end,   y_box + box_h / 2),
                    xytext=(x_arr_start, y_box + box_h / 2),
                    arrowprops=dict(arrowstyle="-|>", color=ps.NAVY,
                                    lw=2.4, mutation_scale=26))

# Title + subtitle
ax.text(8, 5.55,
        "SocRule: 5 strictly-ordered Socratic teaching stages, 34 cognitive states",
        ha="center", va="center", fontsize=18, fontweight="bold", color=ps.NAVY)
ax.text(8, 5.05,
        "6,803 multi-turn Chinese dialogues  ·  42,892 annotated turns  ·  Elementary-school science",
        ha="center", va="center", fontsize=12, color=ps.GRAY, style="italic")

ps.save_fig(fig, "A_socrule_flow", __file__)
