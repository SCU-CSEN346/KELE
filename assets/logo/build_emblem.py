#!/usr/bin/env python3
"""Generate the MELE emblem as a clean flat vector SVG (hand-authored).

Inspired by test_logo.png: laurel wreath + torch, Socrates & android busts
facing each other on an Ionic pedestal, an open book (human text <-> binary),
and the MELE wordmark. SCU palette: red #A00037, gold #FAC300.
"""

import math
import random

from txt2path import text_to_path

RED = "#A00037"
GOLD = "#FAC300"
GREY = "#D9DCE3"
GREYD = "#AAB0BC"
INK = "#3a3f4a"
CREAM = "#FBF7EC"
EB = "/usr/share/fonts/TTF/OpenSans-ExtraBold.ttf"
RG = "/usr/share/fonts/TTF/OpenSans-Regular.ttf"
BD = "/usr/share/fonts/TTF/OpenSans-Bold.ttf"

W, H = 1200, 1180
CX = 600

out = []


def add(s):
    out.append(s)


# ---------------------------------------------------------------- neural net bg
def neural_net():
    # Full-canvas graph behind the emblem: nodes + edges = pure geometry,
    # kept low-opacity so it reads as background texture, not foreground.
    rnd = random.Random(346)
    cols, rows = 7, 7
    nodes = []
    for r in range(rows):
        for c in range(cols):
            x = 70 + c * (W - 140) / (cols - 1) + rnd.uniform(-34, 34)
            y = 70 + r * (H - 140) / (rows - 1) + rnd.uniform(-34, 34)
            nodes.append((x, y))

    edges = []
    for i, (xi, yi) in enumerate(nodes):
        dists = sorted(
            ((math.hypot(xi - xj, yi - yj), j) for j, (xj, yj) in enumerate(nodes) if j != i)
        )
        for _, j in dists[:3]:
            if (min(i, j), max(i, j)) not in edges:
                edges.append((min(i, j), max(i, j)))

    g = ['<g opacity="0.16">']
    for a, b in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        g.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{GREYD}" stroke-width="2"/>'
        )
    for k, (x, y) in enumerate(nodes):
        fill = RED if k % 5 == 0 else (GOLD if k % 5 == 2 else GREYD)
        g.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 if k % 5 in (0, 2) else 4}" fill="{fill}"/>'
        )
    g.append("</g>")
    return "\n".join(g)


# ---------------------------------------------------------------- wreath
def leaf(px, py, deg, L=52, Wd=17, fill=RED):
    d = (
        f"M0,0 C {Wd},{-L * 0.30} {Wd},{-L * 0.72} 0,{-L} "
        f"C {-Wd},{-L * 0.72} {-Wd},{-L * 0.30} 0,0 Z"
    )
    return (
        f'<path transform="translate({px:.1f},{py:.1f}) rotate({deg:.1f})" d="{d}" fill="{fill}"/>'
    )


def wreath():
    cx, cy, R = CX, 470, 300
    n = 11
    g = ["<g>"]
    # stem arcs
    for side in (-1, 1):
        a0, a1 = math.radians(255), math.radians(95)  # left arc default
        if side == 1:
            a0, a1 = math.radians(-75), math.radians(85)
        for i in range(n):
            t = i / (n - 1)
            a = a0 + (a1 - a0) * t
            px = cx + R * math.cos(a)
            py = cy - R * math.sin(a)
            # tangent toward top
            tx, ty = -math.sin(a), -math.cos(a)
            if side == -1:
                tx, ty = math.sin(a), math.cos(a)
            # tilt outward a touch
            rad_out = (math.cos(a), -math.sin(a))
            dx = tx * 0.75 + rad_out[0] * 0.45 * side * 0 + rad_out[0] * 0.4
            dy = ty * 0.75 + rad_out[1] * 0.4
            nrm = math.hypot(dx, dy)
            dx, dy = dx / nrm, dy / nrm
            deg = math.degrees(math.atan2(dx, -dy))
            fill = GOLD if i % 2 == 0 else RED
            sz = 56 if 2 <= i <= n - 2 else 42
            g.append(leaf(px, py, deg, L=sz, Wd=sz * 0.32, fill=fill))
    g.append("</g>")
    return "\n".join(g)


# ---------------------------------------------------------------- torch
def torch():
    return f'''<g>
  <path d="M{CX - 9},190 L{CX - 15},250 L{CX + 15},250 L{CX + 9},190 Z" fill="{GREYD}"/>
  <rect x="{CX - 20}" y="183" width="40" height="12" rx="4" fill="{GREY}"/>
  <path d="M{CX},120 C {CX + 34},150 {CX + 30},178 {CX},186
           C {CX - 30},178 {CX - 34},150 {CX},120 Z" fill="{GOLD}"/>
  <path d="M{CX},140 C {CX + 16},158 {CX + 14},176 {CX},184
           C {CX - 14},176 {CX - 16},158 {CX},140 Z" fill="{RED}"/>
</g>'''


# ---------------------------------------------------------------- pedestal
def pedestal():
    return f'''<g>
  <!-- abacus / slab top -->
  <rect x="{CX - 150}" y="452" width="300" height="26" rx="4" fill="{GREY}" stroke="{GREYD}" stroke-width="2"/>
  <!-- volutes -->
  <g fill="none" stroke="{GREYD}" stroke-width="9">
    <path d="M{CX - 120},480 a18,18 0 1 0 0.1,0" fill="{GREY}"/>
    <path d="M{CX + 120},480 a18,18 0 1 1 -0.1,0" fill="{GREY}"/>
  </g>
  <rect x="{CX - 130}" y="478" width="260" height="16" fill="{GREY}" stroke="{GREYD}" stroke-width="2"/>
  <!-- fluted shaft -->
  <rect x="{CX - 95}" y="494" width="190" height="120" fill="{GREY}" stroke="{GREYD}" stroke-width="2"/>
  <g stroke="{GREYD}" stroke-width="3">
    {"".join(f'<line x1="{CX - 70 + i * 28}" y1="500" x2="{CX - 70 + i * 28}" y2="608"/>' for i in range(6))}
  </g>
  <!-- base -->
  <rect x="{CX - 110}" y="612" width="220" height="20" rx="4" fill="{GREY}" stroke="{GREYD}" stroke-width="2"/>
</g>'''


# ---------------------------------------------------------------- busts
def socrates():
    # profile FACING RIGHT (toward center): the silhouette's right edge IS
    # the forehead->nose->lip->beard profile.
    sil = (
        "M468,250 "
        "C 506,248 536,262 548,288 "  # forehead leaning out
        "C 550,300 549,308 547,314 "  # brow
        "C 560,326 566,334 552,344 "  # nose tip out then back
        "C 547,350 549,356 545,362 "  # philtrum / upper lip
        "C 558,378 560,408 548,434 "  # beard front bulge
        "C 536,456 504,466 474,458 "  # beard bottom
        "C 456,452 448,436 446,418 "  # jaw / neck back
        "C 444,398 442,380 430,366 "  # back of head lower
        "C 414,340 414,294 430,270 "  # back of skull
        "C 442,256 454,250 468,250 Z"
    )
    return f'''<g>
  <path d="{sil}" fill="{GREY}"/>
  <!-- beard mass (darker) -->
  <path d="M470,372 C 470,420 500,452 538,440 C 552,410 556,384 545,362
           C 520,392 492,392 470,372 Z" fill="{GREYD}"/>
  <g fill="none" stroke="{RED}" stroke-width="4" stroke-linecap="round">
    <path d="M486,404 q16,14 36,8"/>
    <path d="M484,420 q18,16 42,8"/>
  </g>
  <!-- crown curls -->
  <g fill="{GREYD}">
    <circle cx="446" cy="276" r="13"/><circle cx="470" cy="264" r="13"/>
    <circle cx="496" cy="266" r="12"/><circle cx="430" cy="300" r="12"/>
  </g>
  <!-- brow + eye -->
  <path d="M512,304 q14,-4 24,2" fill="none" stroke="{RED}" stroke-width="4" stroke-linecap="round"/>
  <circle cx="522" cy="314" r="4.5" fill="{RED}"/>
</g>'''


def android():
    # bust FACING LEFT (toward center): angular robotic head + neck on slab,
    # sized and baselined to match Socrates (mirror, faces near x636).
    sil = (
        "M724,250 "
        "C 760,248 786,262 792,288 "  # forehead/back leaning out
        "C 794,300 793,308 791,314 "  # crown back
        "L792,360 "  # back of head
        "C 792,384 784,402 770,414 "  # back of jaw down
        "L770,432 L740,432 "  # neck right
        "C 740,448 742,452 742,452 "  # neck base
        "L668,452 "  # shoulders to center
        "C 668,452 670,448 670,432 "  # neck left base
        "L640,432 L640,414 "  # neck left
        "C 626,402 620,384 620,360 "  # jaw front (center side)
        "C 620,348 626,342 634,338 "  # chin
        "C 622,330 614,322 624,312 "  # nose notch out toward center
        "C 630,306 634,300 632,292 "  # bridge
        "C 634,272 648,256 672,250 "  # forehead toward center
        "C 690,247 708,247 724,250 Z"
    )
    return f'''<g>
  <!-- antenna -->
  <line x1="718" y1="252" x2="718" y2="220" stroke="{GREYD}" stroke-width="6"/>
  <circle cx="718" cy="215" r="8" fill="{RED}"/>
  <path d="{sil}" fill="{GREY}"/>
  <!-- circuit panel on skull -->
  <rect x="700" y="280" width="78" height="52" rx="8" fill="{INK}"/>
  <g fill="{GOLD}">
    {"".join(f'<circle cx="{712 + (i % 5) * 15}" cy="{293 + (i // 5) * 15}" r="4.2"/>' for i in range(15))}
  </g>
  <!-- visor eye (center-facing) -->
  <rect x="630" y="318" width="40" height="11" rx="5.5" fill="{RED}"/>
  <!-- mouth grille -->
  <g stroke="{INK}" stroke-width="3.5">
    <line x1="634" y1="366" x2="664" y2="366"/>
    <line x1="636" y1="374" x2="662" y2="374"/>
  </g>
  <!-- neck seams + trace -->
  <g stroke="{GREYD}" stroke-width="3">
    <line x1="648" y1="440" x2="732" y2="440"/>
  </g>
  <path d="M778,346 h-16 v18" stroke="{RED}" stroke-width="4" fill="none"/>
  <path d="M704,332 v10 h18" stroke="{RED}" stroke-width="4" fill="none"/>
</g>'''


# ---------------------------------------------------------------- book
def book():
    # shallow open book: two near-flat page blocks meeting at a center spine.
    # left page text lines (human), center binary, right page lines (human).
    def lines(x0, w, count, y0=702):
        return "".join(
            f'<rect x="{x0 + (10 if i == 0 else 0):.0f}" y="{y0 + i * 13}" '
            f'width="{(w - 20) if i == 0 else w}" height="4.5" rx="2" '
            f'fill="{RED if i == 0 else GREYD}"/>'
            for i in range(count)
        )

    binary = "".join(
        f'<text x="{CX}" y="{712 + j * 15}" font-family="DejaVu Sans Mono" '
        f'font-size="13" letter-spacing="1" fill="{RED}" text-anchor="middle">{row}</text>'
        for j, row in enumerate(["10110", "01001", "11010", "00101", "10101"])
    )
    return f'''<g>
  <!-- gold cover rim (sits behind pages) -->
  <path d="M{CX - 262},688 L{CX},676 L{CX + 262},688
           L{CX + 262},702 C {CX + 150},792 {CX + 40},794 {CX},792
           C {CX - 40},794 {CX - 150},792 {CX - 262},702 Z" fill="{GOLD}"/>
  <!-- left page -->
  <path d="M{CX - 248},690 L{CX - 4},680 L{CX - 4},778
           C {CX - 120},776 {CX - 200},762 {CX - 248},752 Z"
        fill="{CREAM}" stroke="{GREYD}" stroke-width="1.5"/>
  <!-- right page -->
  <path d="M{CX + 248},690 L{CX + 4},680 L{CX + 4},778
           C {CX + 120},776 {CX + 200},762 {CX + 248},752 Z"
        fill="{CREAM}" stroke="{GREYD}" stroke-width="1.5"/>
  <!-- center spine -->
  <path d="M{CX},676 L{CX},792" stroke="{GOLD}" stroke-width="6"/>
  {lines(CX - 232, 150, 6)}
  {lines(CX + 82, 150, 6)}
  {binary}
</g>'''


# ---------------------------------------------------------------- wordmark
def wordmark():
    g = ["<g>"]
    d, w = text_to_path("MELE", EB, 158, 0, 0)
    x0 = CX - w / 2
    d, _ = text_to_path("MELE", EB, 158, x0, 930)
    g.append(f'<path d="{d}" fill="{RED}"/>')

    # subtitle: two lines, all red, bold acronym letters M E L E
    def line(runs, y, size):
        total = 0
        widths = []
        for txt, fp in runs:
            _, ww = text_to_path(txt, fp, size, 0, 0)
            widths.append(ww)
            total += ww
        x = CX - total / 2
        for (txt, fp), ww in zip(runs, widths):
            dd, _ = text_to_path(txt, fp, size, x, y)
            g.append(f'<path d="{dd}" fill="{RED}"/>')
            x += ww

    line([("M", BD), ("emorization resistant ", RG), ("E", BD), ("valuation", RG)], 980, 33)
    line(
        [("for ", RG), ("L", BD), ("arge language model ", RG), ("E", BD), ("ducators", RG)],
        1022,
        33,
    )
    g.append("</g>")
    return "\n".join(g)


# ---------------------------------------------------------------- assemble
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add(neural_net())
add(wreath())
add(torch())
add(pedestal())
add(socrates())
add(android())
add(book())
add(wordmark())
add("</svg>")

with open("emblem.svg", "w") as f:
    f.write("\n".join(out))
print("wrote emblem.svg")
