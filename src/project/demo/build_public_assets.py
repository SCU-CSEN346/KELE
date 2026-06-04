"""Regenerate the Chainlit demo image assets in public/ from the source logos.

Edit the CONFIG block below and run:

    uv run python assets/logo/build_public_assets.py

After regenerating, bump the cache version in .chainlit/config.toml
(custom_css = "/public/custom.css?v=N") and the in-CSS image URL so browsers
refetch — Chainlit serves these statically and browsers cache them hard.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "assets" / "logo"
PUBLIC = ROOT / "public"

# ---------------------------------------------------------------------------
# CONFIG — tune these
# ---------------------------------------------------------------------------

# Source artwork (in assets/logo/).
EMBLEM_SRC = "logo_v2_no_bg.png"  # transparent full emblem
FAVICON_SRC = "favicon.jpg"  # circular Socrates/android bust

# --- White-disc landing seal (public/logo_landing.png) ---
# The emblem composited onto a filled circle so the dark-red wordmark/line-art
# read on the grey dark theme.
DISC_COLOR = (255, 255, 255, 255)  # seal background; try a tint e.g. (250,250,245,255)
DISC_PAD = 1.03  # circle radius vs. content's circumscribing radius (1.0 = tight)
DISC_CONTENT_ALPHA = 120  # px with alpha above this count as "content" when sizing
SEAL_SIZE = 760  # output px (square)

# --- Header / welcome logos (public/logo_light.png, logo_dark.png) ---
HEADER_LOGO_SIZE = 512  # max px; transparent emblem, used as-is in the header

# --- Favicon + chat avatar (public/favicon.png, public/avatars/mele.png) ---
# Chroma threshold that separates the red disc (high chroma) from the white
# square corners; raise to keep more, lower to mask more aggressively.
FAVICON_DISC_CHROMA = 40
FAVICON_SIZE = 256
AVATAR_SIZE = 128


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _circular_mask(side: int, supersample: int = 4) -> Image.Image:
    big = Image.new("L", (side * supersample, side * supersample), 0)
    ImageDraw.Draw(big).ellipse([0, 0, side * supersample - 1, side * supersample - 1], fill=255)
    return big.resize((side, side), Image.LANCZOS)


def build_seal() -> None:
    em = Image.open(SRC / EMBLEM_SRC).convert("RGBA")
    w, h = em.size
    alpha = em.getchannel("A").load()
    xs, ys = [], []
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if alpha[x, y] > DISC_CONTENT_ALPHA:
                xs.append(x)
                ys.append(y)
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    radius = DISC_PAD * max(
        math.hypot(x0 - cx, y0 - cy),
        math.hypot(x1 - cx, y0 - cy),
        math.hypot(x0 - cx, y1 - cy),
        math.hypot(x1 - cx, y1 - cy),
    )
    side = int(2 * radius)
    base = Image.new("RGBA", (side, side), DISC_COLOR)
    base.alpha_composite(em, (int(radius - cx), int(radius - cy)))
    base.putalpha(_circular_mask(side))
    base.resize((SEAL_SIZE, SEAL_SIZE), Image.LANCZOS).save(PUBLIC / "logo_landing.png")
    print(f"logo_landing.png  {SEAL_SIZE}px  disc={DISC_COLOR} pad={DISC_PAD}")


def build_header_logos() -> None:
    em = Image.open(SRC / EMBLEM_SRC).convert("RGBA")
    em.thumbnail((HEADER_LOGO_SIZE, HEADER_LOGO_SIZE), Image.LANCZOS)
    em.save(PUBLIC / "logo_light.png")
    em.save(PUBLIC / "logo_dark.png")
    print(f"logo_light.png / logo_dark.png  {em.size[0]}px")


def build_favicon_and_avatar() -> None:
    src = Image.open(SRC / FAVICON_SRC).convert("RGB")
    w, h = src.size
    px = src.load()

    def is_disc(r: int, g: int, b: int) -> bool:
        return (max(r, g, b) - min(r, g, b)) > FAVICON_DISC_CHROMA or (r + g + b) < 3 * 180

    cy, cx = h // 2, w // 2
    rows = [x for x in range(w) if is_disc(*px[x, cy])]
    cols = [y for y in range(h) if is_disc(*px[cx, y])]
    x0, x1 = (min(rows), max(rows)) if rows else (0, w - 1)
    y0, y1 = (min(cols), max(cols)) if cols else (0, h - 1)
    dcx, dcy = (x0 + x1) / 2, (y0 + y1) / 2
    rad = min(x1 - x0, y1 - y0) / 2 + 1

    side = int(2 * rad)
    mask = _circular_mask(side)
    disc = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    disc.paste(src.crop((int(dcx - rad), int(dcy - rad), int(dcx + rad), int(dcy + rad))), (0, 0))
    disc.putalpha(mask)

    disc.resize((FAVICON_SIZE, FAVICON_SIZE), Image.LANCZOS).save(PUBLIC / "favicon.png")
    (PUBLIC / "avatars").mkdir(exist_ok=True)
    disc.resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS).save(PUBLIC / "avatars" / "mele.png")
    print(
        f"favicon.png {FAVICON_SIZE}px / avatars/mele.png {AVATAR_SIZE}px  chroma={FAVICON_DISC_CHROMA}"
    )


if __name__ == "__main__":
    build_seal()
    build_header_logos()
    build_favicon_and_avatar()
    print("done →", PUBLIC)
