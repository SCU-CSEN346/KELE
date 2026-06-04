# MELE logo — candidate round 1

Four directions in the SCU palette (red `#A00037`, gold `#FAC300`) on a dark
background — the same dark theme Chainlit will use. Designed to stay legible at
avatar size (~36–40px next to chat messages), not just at full size.

Open **`_CONTACT_SHEET.png`** to compare all four at once:
top row = full size, bottom row = actual avatar size.

| | Direction | Idea |
|---|---|---|
| **A** | `a_apple` | SCU-red apple, gold leaf. *"mele"* = Italian for apples + the universal teaching symbol. Most legible small; best booth read. **Recommended.** |
| **B** | `b_bubble` | Red speech bubble holding a white **M** with a gold ∴ ("therefore"). Leans into the Socratic-dialogue concept; a touch busy when shrunk. |
| **C** | `c_monogram` | Bold white **M** on a rounded SCU-red plate with a gold ∴ reasoning glyph. Cleanest at tiny size, most "tech brand" feel. |
| **D** | `d_hybrid` | Apple that doubles as a speech bubble with reasoning dots inside. Conceptually richest (teaching + dialogue), most fragile small. |

Each `*.svg` is the editable source; `*_full.png` is a 256px render on dark.

Next step once a direction is picked: produce the final square avatar + a
horizontal wordmark lockup, drop into `public/`, and wire `.chainlit/config.toml`.
