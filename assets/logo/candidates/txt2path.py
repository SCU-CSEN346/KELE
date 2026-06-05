#!/usr/bin/env python3
"""Outline text into an SVG <path d="..."> string using a TTF font.

Glyphs are laid out left-to-right by advance width, scaled to font_size,
y-flipped to SVG coordinates, baseline placed at (x, y).
"""

import sys

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

_cache = {}


def _load(font_path):
    if font_path not in _cache:
        f = TTFont(font_path)
        _cache[font_path] = (f, f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm, f["hmtx"])
    return _cache[font_path]


def text_to_path(text, font_path, font_size, x, y, letter_spacing=0.0):
    font, gset, cmap, upm, hmtx = _load(font_path)
    s = font_size / upm
    pen_x = x
    d = []
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:
            pen_x += font_size * 0.5 + letter_spacing
            continue
        spen = SVGPathPen(gset)
        tpen = TransformPen(spen, (s, 0, 0, -s, pen_x, y))
        gset[gname].draw(tpen)
        seg = spen.getCommands()
        if seg:
            d.append(seg)
        pen_x += hmtx[gname][0] * s + letter_spacing
    return " ".join(d), pen_x - x


def text_width(text, font_path, font_size, letter_spacing=0.0):
    _, w = text_to_path(text, font_path, font_size, 0, 0, letter_spacing)
    return w


if __name__ == "__main__":
    # quick self-test: render MELE
    fp = "/usr/share/fonts/TTF/OpenSans-ExtraBold.ttf"
    d, w = text_to_path("MELE", fp, 200, 20, 220)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(w) + 40}" '
        f'height="260" viewBox="0 0 {int(w) + 40} 260">'
        f'<rect width="100%" height="100%" fill="#f2efe9"/>'
        f'<path d="{d}" fill="#A00037"/></svg>'
    )
    sys.stdout.write(svg)
