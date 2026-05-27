# Final Presentation Outline
# CSEN 346 — Reproducing & Extending KELE: Structured Socratic Teaching with LLMs

**Team:** Cyberdyne — Ulises Chavarria · Maximilian Khan (2 members)

**Two deliverable dates** (per instructor, 2026-05-25 — see `STATUS_REPORT.md`):
- **2026-05-26 — In-class presentation.** 5 min per member × 2 = **10 min talk** + **10 min Q&A**. **No demo.**
- **2026-06-04 — Final submission.** Paper + code + HF artifacts + **poster + recorded demo**.

This file scopes the **May 26 talk** (8-slide compressed deck) and the **June 4 demo + poster**. The paper's long-form content lives in `deliverables/overleaf/latex/acl_latex.tex`; do not duplicate it here.

---

## Speaker Assignments (May 26)

| Slides | Speaker | Time |
|---|---|---|
| 1–4 | Ulises | ~5 min |
| 5–8 | Max | ~5 min |
| Q&A | both | 10 min |

Hard cap is 10 min talk — *"you won't be allowed to continue after."* Target dry run ≤9:45.

---

# Part A — May 26 In-Class Deck (8 slides)

## Slide 1 — Title

**Title:** "Beyond Memorization: Reproducing and Extending KELE for Socratic Teaching with LLMs"

- Cyberdyne — Ulises Chavarria · Maximilian Khan
- CSEN 346 · Santa Clara University · Spring 2026
- Paper reproduced: Peng et al., *KELE: A Multi-Agent Framework for Structured Socratic Teaching*, EMNLP 2025 Findings
- Repo: github.com/ulises-c/csen-346 · HF: huggingface.co/ulises-c

**Speaker (Ulises):** One sentence on project identity. Transition: "Why we picked this paper, what we built, and a finding that surprised us."

---

## Slide 2 — Problem + KELE in One Slide

**Headline:** AI tutors should teach, not tell — and KELE is the structured framework that tries to enforce that.

- **The Socratic ask:** guide the student to the answer through structured questioning, not direct disclosure.
- **SocRules** — the 5-stage framework KELE operates on: greeting (a) → questioning (b) → correction (c) → affirmation (d) → closure (e).
- **KELE architecture** (Peng et al., EMNLP 2025): two-agent framework — a **Consultant** (GPT-4o) classifies the current SocRule stage, then a **Teacher** (SocratTeachLLM, GLM-4 9B fine-tuned) generates the response.
- **Benchmark:** SocratDataset — 6,803 Chinese dialogues; 681-dialogue test split.
- **Published headline:** SocratTeachLLM surpasses GPT-4o on ROUGE-1/2/BLEU-4.

**Visual:** Left = 5-box SocRule diagram with one-line student/teacher exchange. Right = clean two-box Consultant → Teacher diagram.

**Speaker notes:** Say "surpasses on their metrics" deliberately — this sets up slide 4.

---

## Slide 3 — Our Contributions (3 Pillars)

**Headline:** We reproduced KELE, then improved it in three ways.

| Pillar | What we did | Why it matters |
|---|---|---|
| 🔧 **Architecture** | Replaced the GPT-4o consultant with a deterministic supervised classifier — 24M-param Chinese BERT initially, post-fix upgraded to an ~800M-param Qwen3.5-0.8B-LoRA classifier in the current locked headline | Zero per-run API cost; faster; deterministic routing |
| 📐 **Benchmark** | Showed ROUGE/BLEU invert the true ranking; built a memorization-resistant **unified score** | The "best" model by the paper's metric is the worst pedagogically |
| 🌐 **Extension** | Cross-lingual transfer (Chinese → English); Stage 2 SFT pipeline shipped | Generalizes beyond Chinese tutoring data |

**Speaker notes:** This is the map of the talk. Slide 4 lives on Pillar 2; Slide 5 lives on Pillar 1; Slide 7 lives on Pillar 3.

---

## Slide 4 — The Benchmark Inversion (Teaser → Punchline)

**Headline:** Same configurations. Opposite rankings.

| | ROUGE/BLEU ranking | Pedagogical ranking (unified) |
|---|---|---|
| 🥇 #1 | GPT-4o + SocratTeachLLM | **Gemma 31B + Qwen3.5-LoRA consultant (our system, n=681)** |
| 🥈 #2 | Claude Opus + BERT | Gemma 31B + BERT consultant (our system, n=50) |
| 🥉 #3 | … | Claude Sonnet + BERT (frontier, n=681) |
| 🚨 Last | Claude Opus raw | **GPT-4o + SocratTeachLLM** |

**Why it inverts:** ROUGE and BLEU reward token-level mimicry of training data — not pedagogical correctness.

- The 9B SocratTeachLLM was **trained on 90% of the test data** (paper §4.3) — and the split was never released. Any third-party evaluator draws a different random sample and gets a memorization score.
- Diagnostic signature: the gap *widens* with n-gram order (ROUGE-1 +1.59 → ROUGE-2 +4.92 → BLEU-4 +4.07). Memorization.

**Visual:** Two-column bar chart — left bars (ROUGE-1) put STL first; right bars (unified) put STL last.

**Speaker notes (Ulises):** Land the inversion as a stand-alone result. Don't promise to "come back to it" — Max's slide 7 closes the loop with the synthetic-test-set evidence.

---

## Slide 5 — BERT Consultant + Unified Metric (Architecture)

**Headline:** Two changes to KELE — one in the consultant, one in how we measure.

- **Consultant swap:** replace GPT-4o with fine-tuned `bge-small-zh` (24M params, frozen base + 5-way + 34-way heads). Matches or exceeds GPT-4o on state routing at 0 API cost.
- **Unified score:** `unified = 0.5 × stage_balanced + 0.5 × (judge × 10)`
  - **stage_balanced** = macro-F1 over 5 SocRule stages → corrects for stage-frequency imbalance (questioning dominates; closure is rare-but-critical).
  - **LLM-judge** = Claude Sonnet 4.6 on 4 rubric axes (Socratic validity, stage advancement, age-appropriateness, question form, 0–10).
  - Two orthogonal, memorization-resistant axes — neither rewards token-level mimicry of the training set.

| Consultant | Teacher | State Acc | Δ vs original |
|---|---|---|---|
| GPT-4o (original KELE) | SocratTeachLLM | 25.94% | baseline |
| BERT (ours, prior locked headline 2026-05-18) | Gemma 31B | 48.15% | **+22.21 pp** |
| BERT (ours) | Claude Sonnet | 49.97% | **+24.03 pp** |
| **Qwen3.5-LoRA (ours, current locked headline 2026-05-26)** | **Gemma 31B** | **55.39%** | **+29.45 pp** |

**Visual:** Two-column diagram (KELE original vs. our system) + one bar chart of the state-acc lift.

**Speaker notes (Max):** "A 24M-param classifier matches GPT-4o on a 5-class routing problem. The Socratic task isn't *that* hard for the consultant — it's classification, not generation."

---

## Slide 6 — Master Leaderboard + Local OVERTAKES Frontier (2026-05-26)

**Headline:** On a fair metric, a fine-tuned local model on a single consumer GPU now overtakes the best frontier configuration we tested.

| Rank | Config | n | Unified | Stage_bal | Judge |
|---:|---|---:|---:|---:|---:|
| 🏆🥇 1 | **`qwen3.5 × Gemma-31B · fewshot10 · n=681`** ← **LOCKED HEADLINE (2026-05-26)** | 3974 | **72.24** | 61.32 | 8.32 |
| 🥈 2 | bert × Gemma-31B · top3 · n=50 | 278 | 70.08 | 58.48 | 8.17 |
| 🥉 3 | bert × Claude-Sonnet · top3 · n=681 (best frontier we tested) | 3840 | 70.06 | 58.17 | 8.19 |
| 4 | bert × Claude-Opus · fewshot10 · n=50 | 271 | 69.79 | 58.73 | 8.08 |
| 8 | bert × Gemma-31B · fewshot10 · n=681 (prior locked headline 2026-05-18) | 3834 | 68.65 | 55.42 | 8.19 |
| ⚠️ 9 | qwen3.5 × SocratTeachLLM · n=50 | 288 | 68.21 | 63.40 | 7.30 |
| ⚠️ last (38) | Claude-Sonnet × SocratTeachLLM · EN · n=50 | 303 | 44.36 | 22.55 | 6.62 |

- 🏆 = **current locked paper headline** (full n=681, certified, reproducible from `results/`).
- ⚠️ = SocratTeachLLM-based; contamination-inflated.
- 143 configs measured; 38 with full unified score.

**The overtaking claim:** best frontier we tested (`bert × Claude-Sonnet · top3 · n=681`, unified 70.06) vs. our current locked headline (`qwen3.5 × Gemma-31B · fewshot10 · n=681`, unified 72.24). **The local cell leads by +2.18 unified points — at the same canonical n=681 sample size.** The 2026-05-23 "parity" framing has inverted: open-weight on a single 32 GB consumer GPU at $0 eval API cost now beats the best Anthropic teacher we measured on a memorization-resistant evaluation.

**Speaker notes (Max):** "Open-weight, 32 GB consumer GPU, zero dollars per inference run, beats Anthropic's best frontier teacher under matched consultant infrastructure by 2.18 unified points at the canonical 681-dialogue test split. Same memorization-resistant metric. Same number of test dialogues. The gap is not noise — it's a lead."

---

## Slide 7 — Contamination Evidence + Bilingual Transfer

**Headline:** When we evaluate the "winner" on clean data, it collapses — and the SocRule framework still works across languages.

- **Contamination probe:** evaluate SocratTeachLLM on a **fully synthetic** test set (generated by Claude, never in SocratDataset). Stage_bal drops **63.4 → 32.86** — worse than our dumbest baseline. The 30-point gap *is* the memorization.
- **Cross-lingual transfer:** we translated all 6,803 dialogues to English (`SocratDataset-EN`, on HF). Our `qwen3.5` consultant — trained only on Chinese — scores **unified 65.11** at n=400 on the English test split. The SocRule routing behavior is language-invariant.
- **PR #79 update:** contamination probe now exists in both languages — `SocratDataset-SYNTHETIC-EN` (n=75) is live on HF alongside the original `SocratDataset-SYNTHETIC` extended to n=75.

**Visual:** Two-bar chart — STL on original test (63.4) vs. STL on synthetic clean test (32.86). Caption: "30-point drop = memorization."

**Speaker notes (Max):** "On data it has never seen, the model the paper calls its hero performs worse than random."

---

## Slide 8 — Conclusion + Takeaways

**Three findings to take home:**

1. **A 24M-param BERT classifier replaces a multi-billion-parameter LLM consultant** — matching state-routing accuracy at zero per-run API cost.
2. **ROUGE/BLEU are misleading for Socratic teaching.** The "winning" model in the original paper ranks last on pedagogical accuracy. The unified score (stage_bal + judge) fixes this.
3. **Open-weight local models OVERTAKE the best frontier teacher we tested on a memorization-resistant benchmark.** Gemma 31B + Qwen3.5-LoRA consultant + 10-shot prompting at canonical n=681 scores unified 72.24, beating the best Claude configuration (Sonnet + top-3 + BERT, n=681 unified 70.06) by +2.18 unified points on a single 32 GB consumer GPU at $0 per-run eval API cost.

> "If you're evaluating educational AI, measure pedagogy — not n-gram overlap."

**In flight, gated for June 4 paper:** Stage 2 SFT pipeline shipped (PR #79); training in progress on **Gemma 4 31B-IT** (pivoted from Qwen3.6-27B — ROCm/Triton 3.6.0 issue on gfx1201). Lands as a new row in the leaderboard if it beats the current headline; documented as future work otherwise.

**Links:**
- GitHub: github.com/ulises-c/csen-346
- HF: huggingface.co/ulises-c (SocratDataset, SocratDataset-EN, SocratDataset-SYNTHETIC, SocratDataset-SYNTHETIC-EN, SocratTeachLLM mirror)
- Demo (June 4): TBD — link in paper

**Speaker notes (Max):** Close with the one-liner. Hand to Q&A.

---

# Part B — June 4 Deliverables

## B.1 — Demo (3-min recorded screencast)

**Goal:** Show the BERT consultant + Gemma 31B teacher pipeline running end-to-end on a single dialogue, then regenerate the leaderboard from raw results.

**Script:**
1. **Single-dialogue trace** (~60s) — `uv run python -m src.project.kele test --bert-consultant results/state_classifier_v1/final --n 1 --output results/demo` against a pre-seeded SocratDataset dialogue. Show per-turn stage prediction (a → b → c → d → e) and the teacher's response.
2. **Leaderboard regeneration** (~30s) — `make eval-summary` (or `scripts/backtest_stage_balanced.py`) regenerating `master_leaderboard.md` from `results/`.
3. **Voiceover beats** (~90s) — hit the three findings: BERT/Qwen3.5-LoRA classifier replaces the LLM consultant; ROUGE inversion; local OVERTAKES frontier at canonical n=681.

**Hosting:** HF Space *or* YouTube *or* Google Drive — all three accepted per rubric. Link goes in the paper's §Evaluation (currently `[TBD]`) and on the poster's QR code.

**Production rule:** record in the same session that produces the slide deck's data figures — they share the same `kele test` invocation.

---

## B.2 — Poster (5 panels, "more images, minimal text")

| Panel | Source figure |
|---|---|
| 1 — Problem motivation | Slide 2's 5-box SocRule + tutoring example |
| 2 — KELE architecture (original vs. ours) | Slide 5's two-column diagram |
| 3 — Benchmark inversion / contamination | Slide 4's inversion table + slide 7's 63.4 → 32.86 bar |
| 4 — Unified leaderboard top-5 | Slide 6 table with overtaking callout (+2.18 unified pts vs. frontier) |
| 5 — Cross-lingual + future work (SFT) | Slide 7 bilingual result + slide 8's SFT-in-flight footer |

Pull figures from the slide deck; **do not regenerate from scratch.** Print PDF + arrange physical delivery per course logistics.

---

# Anticipated Q&A

The 10-min Q&A is half the slot. Pre-write 1–2 lines per question; both speakers field.

**Q: SocratTeachLLM's authors *do* claim a proper train/test split. Why is contamination the issue?**
A: Because the split was never released. Any third party evaluating their model on a random 10% subsample gets a memorization score. We show this empirically: stage_bal collapses 63.4 → 32.86 on the synthetic clean test set.

**Q: Isn't it unfair to compare a fine-tuned model (SocratTeachLLM) to prompted models?**
A: Yes — and that's exactly our critique. The paper frames it as "SocratTeachLLM surpasses GPT-4o" without noting one is fine-tuned on the training data and the other is zero-shot. We report both clearly.

**Q: Why is `bge-small-zh` good enough as a state classifier?**
A: Because the task is classification, not generation. The 5 SocRule stages are well-defined; 24M params has enough capacity to learn the lexical/structural cues. Our results show it matches GPT-4o.

**Q: What is the unified metric's theoretical basis?**
A: Equal-weight average of two orthogonal, memorization-resistant axes. We defend 50/50 in the paper — no principled prior for weighting one over the other, and any other weighting requires defending why one axis dominates.

**Q: Where is your fine-tuned teacher?** *(updated for PR #79)*
A: Two answers depending on timing. The **consultant side** is fully shipped — 5 trained classifiers, two of them (`state_classifier_v1` and `state-clf-qwen3.5-0.8b-lora`) on the headline path; the Qwen3.5-LoRA classifier is the current locked-headline consultant as of 2026-05-26. The **teacher side** is in flight: pipeline shipped via PR #79 (format fix, three Stage 2 loaders, three Stage 1 loaders, DPO pair builder with Source 3 functional), training the Gemma 4 31B-IT base mid-sprint. The decision to *not* race to a SocratTeachLLM replacement was deliberate — once contamination evidence landed, the right target became "win on memorization-resistant metrics," which the current locked headline now does **decisively** at canonical scale (unified 72.24 vs. best frontier 70.06, a +2.18 lead at n=681).

**Q: Why pivot from Qwen3.6-27B to Gemma 4 31B for SFT?** *(new)*
A: Qwen3.6-27B uses gated DeltaNet layers that require `flash-linear-attention` (FLA). FLA's Triton 3.6.0 kernels page-fault at runtime on AMD gfx1201 (RDNA4 / R9700) and the torch fallback OOMs above 512 tokens — well below the 1280-token requirement for SocratDataset. Gemma 4 31B-IT uses standard softmax attention (no FLA dependency), fits in 32 GB VRAM at seq=1280 with QLoRA (~21–23 GB peak), and is already the **#1 open-weight teacher in our leaderboard** (unified 70.08). The pivot is documented in PR #79.

**Q: Does the model work in real classrooms?**
A: We've shown it works on 681 held-out Chinese math/science dialogues. Real-classroom deployment would require teacher training, privacy review, and latency work (Gemma 31B is ~3s/turn on our GPU). Out of scope for this paper; listed as future work.

**Q: What are the ethical implications?**
A: Bias in training data (SocratDataset is Chinese elementary science — may generalize poorly to other subjects/cultures), risk of over-reliance on AI tutors, and student privacy. All three addressed in the paper's ethics section.

---

# Slide Design Notes

- **Font:** clean sans-serif (Inter, Lato, or similar). Min 24pt body, 36pt headings.
- **Color scheme:** white background, #2563EB (blue) accents. Pick one and stick with it.
- **Three must-have visuals** (everything else is text):
  1. **Slide 4 — inversion bar chart:** ROUGE ranking vs. unified ranking, same 10 configs.
  2. **Slide 5 — two-column architecture diagram:** KELE original vs. our system, plus the state-acc lift bars.
  3. **Slide 6 — leaderboard top-5 table** with the overtaking callout (+2.18 unified pts at n=681).
- **Slide budget:** ≤6 bullets per slide; one key number/claim per slide; no acronyms without first-use definition; no >5-column tables.

---

# Timing Checklist

- [ ] Lock the 8-slide compression with Max (mostly done; sanity-check pillar emphasis)
- [ ] Build the deck (PPTX / Keynote / Slidev) — est. 3–5 h
- [ ] Produce the 3 must-have figures (don't regenerate ones that already exist in `results/`)
- [ ] Dry run with stopwatch — target **≤9:45** (10-min cap is hard)
- [ ] Q&A defensive material — 1–2 lines per question above (especially the two new ones for PR #79)
- [ ] PDF backup export
- [ ] Confirm no network needed if presenting locally (no live demo on May 26)
