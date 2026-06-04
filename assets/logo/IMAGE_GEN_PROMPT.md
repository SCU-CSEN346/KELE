# Image-gen prompt — clean line-art MELE emblem (for vectorizing)

Goal: regenerate the `test_logo` concept as **flat line-art** so it traces
razor-sharp to vector. The previous version blurred only because it was a *soft,
shaded* render. We want the opposite: hard edges, uniform outlines, flat colors.

**Generate the EMBLEM ONLY — no "MELE" text, no subtitle.** AI text comes out
garbled; I'll add the wordmark as crisp font outlines afterward.

---

## Prompt (paste this)

> Flat vector line-art logo emblem, clean and modern, thick uniform black
> outlines, bold cel-shaded flat colors, NO gradients, NO soft shading, NO
> drop shadows, NO texture, hard crisp edges, sticker / vector decal style.
>
> Subject: a circular laurel wreath made of two symmetric arcs of pointed laurel
> leaves with a small lit torch at the very top center. Inside the wreath, two
> busts in profile FACING EACH OTHER on top of a classical Ionic column pedestal
> (with scroll volutes and vertical fluting). Left bust: Socrates — a bearded
> ancient-Greek philosopher with curly hair, classical and dignified. Right
> bust: a sleek humanoid android / AI robot head in profile, with a visible
> circuit-board / neural-net pattern on its skull and a small antenna. The two
> heads face inward toward each other as if in dialogue. Below the busts, a large
> open book with two pages: the left page shows lines of human writing, the
> center shows a column of binary digits (0s and 1s), the right page shows lines
> of human writing again — symbolizing human text translated to binary and back.
>
> Color palette: deep crimson red (#A00037) and gold-yellow (#FAC300) as the two
> brand colors, with light grey / off-white for the busts and column and the book
> pages. Limit to about 4–5 flat colors total. The book covers are gold (#FAC300).
> The laurel leaves alternate red and gold. The torch flame is gold and red.
>
> Pure solid WHITE background, no scene, no border, centered composition,
> symmetrical, high resolution, square framing.

## Negative / avoid (if your tool has a negative-prompt field)

> gradients, soft shading, drop shadow, glow, bloom, 3D render, photorealistic,
> realistic, painterly, watercolor, texture, grain, noise, blurry edges,
> anti-aliasing, background scenery, text, letters, words, watermark, signature

---

## Settings tips
- **Highest resolution available** (≥1024², ideally 2048²). More pixels → cleaner trace.
- **Pure white background**, not off-white, not gradient (it tiled a beige→peach
  gradient last time — that's what made tracing messy).
- **No text** in the image. If your tool insists on adding "MELE", regenerate or
  crop it out; I replace it with vector text anyway.
- Generate a few; pick the one with the **flattest colors and hardest edges**
  (least shading), not the prettiest-shaded one — flatness is what vectorizes well.

## When you have one you like
Drop it in `assets/logo/candidates/mele_lineart.png` (or just tell me the path /
push it) and I'll: vectorize it sharp, snap every color to exact SCU hex, make the
background transparent, composite the crisp font wordmark + bolded acronym
letters under it, and export the final `public/` PNGs + wire Chainlit.
