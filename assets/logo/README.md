# MELE logo — hand-authored vector (draft v1)

Built **from scratch** as clean flat vector (not a trace of `test_logo.png` —
the trace splined straight lines and corners, so it was rebuilt by hand using
`test_logo.png` only as inspiration). SCU palette: red `#A00037`, gold `#FAC300`.

| File | What |
|---|---|
| `emblem.svg` | The full emblem master: laurel wreath + torch, Socrates & android busts facing each other on an Ionic pedestal, open book (human text ↔ binary), MELE wordmark. |
| `b_avatar.svg` | The small chat-avatar mark (the "B" speech-bubble + M you picked). The full emblem is too detailed for the ~36px message avatar; this is its small-size companion. |
| `build_emblem.py` | **Editable source** for the emblem — procedurally places wreath leaves, circuit grid, binary, and outlines the wordmark. Run `python3 build_emblem.py` to regenerate `emblem.svg`. |
| `txt2path.py` | Helper: outlines Open Sans glyphs into SVG paths so the wordmark is crisp and font-independent. |
| `_PREVIEW_light.png` / `_PREVIEW_dark.png` | Emblem on light and on the Chainlit dark theme. |
| `_PREVIEW_avatar40px.png` | The B avatar at true 40px size (4× nearest-neighbor zoom). |

## Wordmark
"MELE" in Open Sans ExtraBold, SCU red. Subtitle all red, two lines, with the
acronym letters bolded: **M**emorization resistant **E**valuation for **L**arge
language model **E**ducators (L = **L**arge, not LLM).

## Neural-net background
Full-canvas graph (nodes + edges) behind the wreath, low opacity (~0.16) so it
reads as background texture. Pure geometry, so it's authored directly in vector
(`neural_net()` in `build_emblem.py`) rather than image-gen — see
`IMAGE_GEN_PROMPT.md`: the generated emblem must stay on plain white, and this
net is composited behind it.

## Known follow-ups (pre-final)
- Dark theme: the red wordmark is low-contrast on dark — will ship a gold/white
  wordmark variant for the Chainlit dark UI.
- Busts are deliberately flat/iconic (organic bearded vs. angular robotic) — easy
  to refine further per feedback.
- Final step once approved: export PNGs into `public/`, wire `.chainlit/config.toml`
  (verify the Chainlit 2.4 logo/avatar mechanism first).
