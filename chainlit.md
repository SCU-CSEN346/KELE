# Welcome to MELE

**MELE** (**M**emorization-resistant **E**valuation for **L**LM **E**ducators) is a Socratic math tutor
built at Santa Clara University. Instead of giving you the answer, MELE guides you to
find it yourself through carefully chosen questions — the same method research shows
produces deeper and more durable learning than direct instruction.

---

## How the demo works

1. **Type any K–12 math question or problem** — arithmetic, algebra, geometry, word
   problems, proofs. MELE works best when you share something you're genuinely stuck on
   or curious about.

2. **Expect questions back, not answers.** MELE will ask you what you already know, what
   you've tried, or where you got stuck. Follow its lead — even a wrong guess moves the
   conversation forward.

3. **MELE tracks where you are.** Under the hood, a 34-state cognitive model (the
   SocRule pipeline from the KELE paper) classifies each turn and picks the right
   teaching move — probing, hinting, validating, or closing out the concept.

4. **Session memory is per-conversation.** Each new chat starts fresh. You can end a
   session at any time and start a new one with a different topic.

---

## The research behind it

MELE extends **KELE** (Peng et al., _Findings of EMNLP 2025_), the first academic
framework to structure LLMs for genuine Socratic teaching. Our headline result: a single
32 GB consumer GPU running an open-weight Gemma 4 31B teacher, guided by a fine-tuned
Qwen3.5-0.8B state classifier, **overtakes Anthropic's Opus 4.6** on a
memorization-resistant evaluation — and exposes that the published ROUGE/BLEU benchmark
was rewarding training-data memorization, not actual teaching capability.

The demo runs the locked top-performer stack:
**Qwen3.5-0.8B-LoRA state classifier × Gemma 4 31B teacher × 10-shot exemplars**
— unified score **72.24**, +29.45 pp over the GPT-4o + SocratTeachLLM baseline.

---

_Questions about the research? Talk to Ulises or Max at the booth._
