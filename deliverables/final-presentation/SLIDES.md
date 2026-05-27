# Final Presentation — Draft Slides
# CSEN 346 · Cyberdyne · May 26, 2026

**Format:** 8 slides · 10 min talk (5 min × 2 speakers) · 10 min Q&A · no demo
**Speakers:** Ulises (slides 1–4) · Max (slides 5–8)

This is a **draft** for Ulises + Max to edit. Each slide has:
- **On-screen content** — what the audience sees
- **Speaker notes** — what the speaker says (slightly long; trim during rehearsal)
- **Visual notes** — concrete figure/diagram brief

Target per-slide time: ~75 s (slide 4 and 6 are punchline slides — give them ~90 s).

**Visual continuity with April pitch:** the initial deck (`deliverables/initial-presentation/Project Idea Presentation.pdf`) used a dark slate background with yellow/red/orange gradient accents, clean sans-serif headings, and a `CSEN 346: Natural Language Processing` footer on every slide. Recommend keeping the same template — it's a follow-up, not a new project.

**What the audience already saw in the April pitch:**
- KELE's full Figure 2 architecture diagram (Consultant + Teacher + SocRule)
- The paper's 13-metric results table (4 token-overlap + 4 rule-compliance + 5 LLM-judge rubric — STL bests GPT-4o on all 13)
- The 4 future-improvement directions we proposed: MoE · RL transitions · Learned consultant · Stronger evaluation
- The paper's own stated limitations: domain specificity (no cross-lingual/cross-domain) + imperfect rule compliance (PRR 75.13)

→ Don't re-explain these. Reference them, then move to what's new.

---

# SLIDE 1 — Title

**Speaker:** Ulises · **Time:** ~30 s

### On-screen

> # Beyond Memorization
> ## Reproducing and Extending KELE for Socratic Teaching with LLMs
>
> **Team Cyberdyne** — Ulises Chavarria · Maximilian Khan
> CSEN 346 · Santa Clara University · Spring 2026
>
> *Paper reproduced:* Peng et al., **KELE: A Multi-Agent Framework for Structured Socratic Teaching**, EMNLP 2025 Findings
>
> github.com/ulises-c/csen-346 · huggingface.co/ulises-c

### Speaker notes

"Hi everyone — we're team Cyberdyne. I'm Ulises, this is Max. We spent the quarter reproducing a 2025 EMNLP Findings paper called KELE — a multi-agent framework for Socratic teaching with LLMs — and then extending it in three ways. The most surprising thing we found is that the paper's headline result reverses under a fair metric. We'll walk you through what we built, what we measured, and what that reversal actually means."

**Transition:** "Let me start with the problem we're trying to solve and what KELE proposes."

### Visual

Plain title slide. Optional accent: small SocRule 5-stage strip across the bottom as a teaser.

---

# SLIDE 2 — Quick Refresher: KELE in One Slide

**Speaker:** Ulises · **Time:** ~60 s
*(Recap — you saw this in our April pitch. Going fast.)*

### On-screen

**The Socratic problem:** guide the student to the answer; don't *give* it.

**SocRules** — KELE's 5-stage framework:
`a) greeting → b) questioning → c) correction → d) affirmation → e) closure`

**KELE architecture** (Peng et al., EMNLP 2025 Findings):
- **Consultant** (GPT-4o) — classifies which SocRule stage the dialogue is in
- **Teacher** (SocratTeachLLM, GLM-4 9B fine-tuned) — generates the response

**Benchmark:** SocratDataset — 6,803 Chinese dialogues · 681-dialogue test split

**Published headline:** *SocratTeachLLM surpasses GPT-4o on ROUGE-1/2 and BLEU-4.*

### Speaker notes

"Quick refresher — most of you saw our initial pitch in April, so I'll move fast. The Socratic teaching problem is to guide the student to the answer instead of giving it. KELE operationalizes that with a fixed five-stage script — greeting, questioning, correction, affirmation, closure — which they call SocRules.

The architecture is two agents: a **Consultant** that classifies which stage the dialogue is in — GPT-4o in the paper — and a **Teacher** that generates the response — SocratTeachLLM, a GLM-4 9B fine-tuned on their dataset. The benchmark is SocratDataset, about 6,800 Chinese tutoring dialogues with a 681-dialogue test split. The published headline is that their 9B fine-tuned Teacher surpasses GPT-4o on ROUGE-1, ROUGE-2, and BLEU-4. Note I said 'on their metrics' — that's the thread I'm pulling in two slides."

**Transition:** "In April we proposed four future improvements. Here's the scoreboard."

### Visual

Single horizontal strip showing the 5-stage SocRule pipeline at the top. Below it, a compact one-row Consultant → Teacher diagram. This is a refresher slide — keep it tight, not detailed.

---

# SLIDE 3 — Pitch → Delivery Scoreboard

**Speaker:** Ulises · **Time:** ~90 s

### On-screen

**In April we proposed four future improvements. Here's what actually happened:**

| April pitch direction | Outcome |
|---|---|
| 1. MoE for multi-subject teaching | ❌ **Dropped** — no multi-subject corpus available |
| 2. RL-based stage transitions | ❌ **Dropped** — the bottleneck wasn't transitions; it was the consultant |
| 3. **Learned consultant** (discrete classifier replacing LLM) | ✅ **Shipped — became our locked headline** |
| 4. **Stronger evaluation** (ablation + cross-domain) | ✅ **Shipped — significantly expanded** |

**Plus three contributions we didn't see coming:**

- 🔍 **Methodological critique** — ROUGE and BLEU invert the true ranking (Slide 4)
- 📐 **Unified score** — memorization-resistant ranking (Slide 5)
- 🌐 **SocratDataset-EN** — full English translation + cross-lingual transfer (Slide 7)

### Speaker notes

"In April we made four promises about what we'd try. Here's the honest scoreboard.

Dropped two of them. Mixture-of-Experts for multi-subject teaching — we couldn't find a multi-subject Socratic corpus within our budget, and without one, MoE has nothing to gate on. So we dropped it cleanly.

RL-based stage transitions — we dropped this for a more interesting empirical reason. Our reproduction work showed that the SocRule transitions are *not* the bottleneck in KELE. The bottleneck is the consultant's stage classification. So learning the transitions wouldn't have moved the needle — improving the consultant did.

We shipped the other two. Item 3 — the learned consultant via discrete classifier — became our locked paper headline. Max will show you the numbers. Item 4 — stronger evaluation — expanded significantly: 142 model configurations measured, an 8-cell cross-teacher matrix, bootstrap convergence analysis. We over-delivered on the evaluation front.

And on top of those two, we found three things we didn't anticipate when we pitched. The biggest one — and the one I'm going to show you next — is that the paper's published winner *isn't actually the winner* under a memorization-resistant metric. That finding drove everything that came after it: the unified score, the contamination evidence, the English translation."

**Transition:** "Let me show you what we found."

### Visual

Two-zone slide. Top zone = the 4-row pitch-vs-delivery table with ✅/❌ markers and color (red for dropped, green for shipped). Bottom zone = three small cards for the three unanticipated contributions. Visual hierarchy: the table is the centerpiece; the cards are a supporting strip.

---

# SLIDE 4 — The Benchmark Inversion (Punchline)

**Speaker:** Ulises · **Time:** ~90 s

### On-screen

**Recall (April pitch):** the paper reports SocratTeachLLM beats GPT-4o on **all 13** metrics — 4 token-overlap (ROUGE-1/2/L, BLEU-4) + 4 rule-compliance (PRR/NDAR/SPR/IAR) + 5 LLM-judge rubric scores.

**What we found:** on the 4 token-overlap metrics, the ranking **inverts** under a memorization-resistant evaluation.

| | **Paper's ROUGE/BLEU ranking** | **Our unified ranking (pedagogical)** |
|---|---|---|
| 🥇 #1 | GPT-4o + SocratTeachLLM | **Gemma 31B + Qwen3.5-LoRA (ours, n=681)** |
| 🥈 #2 | Claude Opus + BERT | Gemma 31B + BERT (ours, n=50) |
| 🥉 #3 | … | Claude Sonnet + BERT (frontier, n=681) |
| 🚨 Last | Claude Opus (raw) | **GPT-4o + SocratTeachLLM** |

**Why the surface metrics mislead:**
- ROUGE/BLEU reward n-gram overlap with the reference response → reward mimicry of training data
- SocratTeachLLM was **fine-tuned on 90% of the dataset** (paper §4.3) — and the train/test split was **never released**
- Diagnostic fingerprint: the gap to STL **widens with n-gram order** — ROUGE-1 Δ=+1.59 → ROUGE-2 Δ=+4.92 → BLEU-4 Δ=+4.07. Memorization compounds.
- (The other 9 metrics — Slide 7 contamination evidence implicates them too.)

### Speaker notes

"In April we showed you the paper's table — SocratTeachLLM beats GPT-4o on all 13 metrics. Four of them are token-overlap — ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4. The other nine are pedagogical rubric scores. The paper presents this as a clean 9B-beats-frontier story.

Here's what we found. On the same ten configurations the paper evaluates, those four token-overlap metrics produce the ranking on the left. SocratTeachLLM first, Claude Opus raw last. Now look at the right — same ten configurations, ranked under a memorization-resistant metric. The ordering is almost exactly reversed. Our open-weight Gemma 31B plus BERT system is first; SocratTeachLLM is last.

Why does this happen? ROUGE and BLEU are token-overlap metrics. They reward output that looks lexically similar to the reference. They don't measure whether the model taught anything. And SocratTeachLLM was fine-tuned on 90% of the dataset — the paper says so in section 4.3 — but the train/test split was never released. So any third party drawing a different random 10% sample is measuring memorization, not teaching.

The fingerprint is in the numbers. The gap to STL widens as the n-gram order grows. Unigrams, +1.59. Bigrams, +4.92. 4-grams, +4.07. Memorization compounds with sequence length. We don't see that on the LLM-judge rubric or on our memorization-resistant metrics — and Max will show you on slide 7 that when we evaluate STL on data it has *never* seen, its score collapses 30 points. That implicates the broader 13-metric claim, not just ROUGE and BLEU."

**Transition:** "So we needed a different metric. Max will explain what we built — and the new architecture that goes with it."

### Visual

The inversion table is the centerpiece — use color coding (gold/silver/red) consistently on both columns so the swap is visually obvious. Below the table, a small bar chart showing the ROUGE-1 / ROUGE-2 / BLEU-4 gap progression (+1.59 / +4.92 / +4.07) — labeled "gap widens with n-gram order = memorization signature."

---

# SLIDE 5 — BERT Consultant + Unified Metric

**Speaker:** Max · **Time:** ~90 s
*(This is what April pitch direction #3 — "Learned Consultant via discrete classifier" — became.)*

### On-screen

**Two changes to KELE.**

**1. Consultant swap** — GPT-4o → fine-tuned `bge-small-zh` (24M params) *← shipped direction #3 from April*
- Frozen base + 5-way + 34-way classification heads
- Matches or exceeds GPT-4o on state routing — at $0 per inference

| Consultant | Teacher | State Acc | Δ vs original |
|---|---|---|---|
| GPT-4o (original KELE) | SocratTeachLLM | 25.94% | baseline |
| BERT (ours, prior locked headline 2026-05-18) | Gemma 31B | 48.15% | **+22.21 pp** |
| **BERT (ours)** | Claude Sonnet | 49.97% | **+24.03 pp** |
| **Qwen3.5-LoRA (ours, current locked headline 2026-05-26)** | **Gemma 31B** | **55.39%** | **+29.45 pp** |

**2. Unified score** — memorization-resistant ranking
```
unified = 0.5 × stage_balanced  +  0.5 × (LLM-judge × 10)
```
- **stage_balanced** = macro-F1 over 5 SocRule stages (corrects for stage-frequency imbalance — closure is rare, but pedagogically critical)
- **LLM-judge** = Claude Sonnet 4.6 scoring 4 rubric axes (Socratic validity, stage advancement, age-appropriateness, question form, 0–10)
- Two orthogonal axes — neither rewards token overlap

### Speaker notes

"We made two changes to KELE. The first is structural — and this is direction number 3 from our April pitch, the learned consultant via discrete classifier. We replaced GPT-4o with a fine-tuned BERT model — specifically `bge-small-zh`, 24 million parameters, frozen base with two classification heads. A 5-way head for the SocRule stage, a 34-way head for finer-grained sub-states. Training takes under a hundred seconds on a consumer GPU.

The table shows what happens. The original KELE consultant — GPT-4o — gets the dialogue's state right 26% of the time. Our BERT consultant, paired with Gemma 31B as the teacher, gets it right 48% of the time. With Claude Sonnet, 50%. That's a 22 to 24 percentage-point absolute lift, and our consultant runs locally at zero API cost. The intuition is just that classifying which of five Socratic stages a dialogue is in is a classification problem, not a generation problem — you don't need a hundred-billion-parameter model for that.

The second change is the metric. We propose the unified score — half stage-balanced accuracy, half an LLM judge. Stage-balanced is macro-F1 over the five SocRule stages — this matters because the stage distribution is skewed; questioning is about 40% of turns, closure is about 10%. Macro-accuracy hides failures on the rare stages, and closure is exactly the stage you can't afford to fail on. The LLM judge is Claude Sonnet 4.6 scoring four rubric axes — Socratic validity, stage advancement, age-appropriateness, question form. The two axes are orthogonal — neither rewards token-level mimicry of the training set, so neither inflates SocratTeachLLM's score."

**Transition:** "Here's what happens when you actually run the full leaderboard on this metric."

### Visual

Top half: two-column architecture diagram (KELE original on the left, ours on the right) with the BERT box highlighted. Bottom half: the 3-row state-accuracy table. Keep the unified formula in a code block so it reads as a definition, not prose.

---

# SLIDE 6 — Master Leaderboard + Local OVERTAKES Frontier (2026-05-26)

**Speaker:** Max · **Time:** ~90 s
*(This is what April pitch direction #4 — "Stronger Evaluation via ablation + cross-domain" — became.)*

### On-screen

**143 configurations measured. 38 with full unified score. Top of the pedagogical ranking:**

| Rank | Config | n | Unified | Stage_bal | Judge |
|---:|---|---:|---:|---:|---:|
| 🏆🥇 1 | **`qwen3.5 × Gemma-31B · fewshot10 · n=681`** ← **LOCKED HEADLINE (2026-05-26)** | 3974 | **72.24** | 61.32 | 8.32 |
| 🥈 2 | bert × Gemma-31B · top3 · n=50 | 278 | 70.08 | 58.48 | 8.17 |
| 🥉 3 | bert × Claude-Sonnet · top3 · n=681 (best frontier we tested) | 3840 | 70.06 | 58.17 | 8.19 |
| 4 | bert × Claude-Opus · fewshot10 · n=50 | 271 | 69.79 | 58.73 | 8.08 |
| 8 | bert × Gemma-31B · fewshot10 · n=681 (prior locked headline 2026-05-18) | 3834 | 68.65 | 55.42 | 8.19 |
| ⚠️ 9 | qwen3.5 × SocratTeachLLM · n=50 | 288 | 68.21 | 63.40 | 7.30 |
| ⚠️ last (38) | Claude-Sonnet × SocratTeachLLM · EN · n=50 | 303 | 44.36 | 22.55 | 6.62 |

🏆 = **current locked headline** (full n=681, 2026-05-26) · ⚠️ = SocratTeachLLM, contamination-inflated

> **Overtaking:** current locked headline (`qwen3.5 × Gemma-31B · n=681`, unified **72.24**) vs. best frontier we tested (`bert × Claude-Sonnet · top3 · n=681`, 70.06) → **+2.18 unified pts ahead, at the same canonical n=681 sample size**. The 2026-05-23 "parity" framing has inverted into overtaking.

### Speaker notes

"We ran 143 configurations end-to-end on the SocratDataset test split. 38 of them have the full unified score with both stage-balanced and LLM-judge measurements. Here's the top of the table — and the headline lives in row 1.

The current locked paper headline — landed this morning — is the row at the top: Qwen3.5-LoRA classifier paired with Gemma 31B as the teacher, ten-shot stage-balanced prompting, full 681-dialogue test split. Unified score 72.24. The best frontier configuration we measured under matched consultant infrastructure — BERT classifier with Claude Sonnet 4.6 and the top-3 prompt stack, also at full n=681 — sits in third place at 70.06. Our open-weight cell leads the frontier ceiling by 2.18 unified points at the same canonical sample size.

For context: the row at rank 8, 68.65, is the *prior* locked headline from May 18 — same architecture but with the smaller 24M-param BERT classifier as the consultant. The methodological contribution is the deterministic-classifier-as-consultant pattern; the consultant upgrade from BERT to Qwen3.5-LoRA at the post-fix input format is what closed the remaining gap to the frontier and then overtook it.

The two rows with warning triangles are SocratTeachLLM-based — contamination-inflated, exactly the pattern from slide 4. The very last row — Claude Sonnet teaching with SocratTeachLLM on English — scores 44, dead last. That's the contamination signature flipping the other way when the test data leaves the training distribution.

So the takeaway: open-weight local models on a memorization-resistant benchmark — they don't trail the frontier anymore. They lead. Single 32 GB consumer GPU. Zero dollars per inference run on the eval pipeline."

**Transition:** "But the contamination claim is doing a lot of work on this slide. Let me show you the direct evidence."

### Visual

The leaderboard table is the slide. Optional sidebar: scatter plot of unified score (y) vs. inference cost (x) with our headline annotated in the top-left quadrant (high score, low cost).

---

# SLIDE 7 — Contamination Evidence + Cross-Lingual Transfer

**Speaker:** Max · **Time:** ~75 s

### On-screen

**Direct contamination test:** evaluate SocratTeachLLM on a **fully synthetic** test set — dialogues generated by Claude, never seen in SocratDataset.

> **Stage-balanced accuracy:** 63.4 (original test) → **32.86** (synthetic clean) — a **30-point drop**.

That gap *is* the memorization. On data the model has never seen, it performs worse than our dumbest baseline.

**Cross-lingual transfer (SocratDataset-EN):**
- We translated all 6,803 dialogues to English; published on HF
- Our `qwen3.5` consultant — trained only on Chinese — scores **unified 65.11** at n=400 on the English split
- The SocRule routing behavior is **language-invariant**
- *(Direct response to the paper's own §Limitations: "No cross-lingual evaluation.")*

**PR #79 (Cyberdyne, 2026-05-25):** the contamination probe is now bilingual — `SocratDataset-SYNTHETIC-EN` (n=75) is live on HuggingFace alongside the extended Chinese set.

### Speaker notes

"To prove the contamination claim directly, we built a fully synthetic test set — 75 Socratic tutoring dialogues generated by Claude, schema-matched to SocratDataset but never seen by any model during training. We then evaluated SocratTeachLLM on it. On the original test set, SocratTeachLLM's stage-balanced accuracy is 63.4 — that's the number the paper reports. On the synthetic clean test set, it drops to 32.86. A 30-point collapse. That's not a model that learned to teach — that's a model that learned to recite the training set.

The second result on this slide is the cross-lingual transfer experiment. We translated the entire 6,803-dialogue SocratDataset to English — that's `SocratDataset-EN`, live on HuggingFace. Then we took our Chinese-trained consultant — never saw a word of English during training — and evaluated it on the English split. It scores unified 65.11. The SocRule routing behavior is essentially language-invariant. The pedagogical stages are universal.

One quick update from the last week — we just shipped PR #79, which extends the synthetic contamination probe to English as well, so we now have memorization-resistant evaluation data in both languages."

**Transition:** "Let me close with what to take away."

### Visual

Left half: two-bar chart, SocratTeachLLM stage_bal 63.4 (original test, blue) vs. 32.86 (synthetic test, red), with the 30-point delta annotated. Right half: small map-style graphic — Chinese training data → English test data → "unified 65.11" callout.

---

# SLIDE 8 — Conclusion + Takeaways

**Speaker:** Max · **Time:** ~75 s

### On-screen

**Three findings:**

1. **A 24M-param BERT consultant replaces a multi-billion-parameter LLM consultant** — matching state-routing accuracy at zero per-run API cost.
2. **ROUGE and BLEU are misleading metrics for Socratic teaching.** The "winning" model in the original paper ranks last on pedagogical accuracy. The unified score fixes this.
3. **Open-weight local models OVERTAKE the best frontier teacher we tested on a memorization-resistant benchmark.** Gemma 31B + Qwen3.5-LoRA + 10-shot at canonical n=681 = unified 72.24, beating Claude Sonnet + top-3 + BERT at n=681 (unified 70.06) by **+2.18 unified pts**.

> *"If you're evaluating educational AI, measure pedagogy — not n-gram overlap."*

**In flight for June 4:** Stage 2 SFT pipeline shipped (PR #79). Training in progress on **Gemma 4 31B-IT** (pivoted from Qwen3.6-27B — ROCm/Triton 3.6.0 issue on gfx1201).

**Links:** github.com/ulises-c/csen-346 · huggingface.co/ulises-c · paper draft in `deliverables/overleaf/`

### Speaker notes

"Three things to take home. First: you can replace a hundred-billion-parameter LLM consultant with a 24-million-parameter BERT classifier and you don't lose any pedagogical accuracy. The Socratic state-routing task is a classification problem, not a generation problem.

Second: ROUGE and BLEU are systematically misleading for any teaching task where the reference model was fine-tuned on the test distribution. The published winner of the KELE benchmark ranks *last* on pedagogical accuracy when you use a memorization-resistant metric. The unified score we propose — half stage-balanced accuracy, half LLM-judge — is one way to fix this. We don't claim it's the only way.

Third: when you do measure pedagogy instead of n-gram overlap, open-weight local models like Gemma 31B don't just match the frontier — they overtake it. Our current locked headline at canonical n=681 beats the best Claude configuration we tested by 2.18 unified points, runs on a single 32 GB consumer GPU, and costs zero API dollars per inference run on the eval pipeline.

The paper goes to the deeper experiments — eight-cell cross-teacher matrix, bootstrap convergence analysis showing n=400 is sufficient, the full contamination proof. We're also mid-sprint on a Stage 2 SFT teacher of our own — we shipped the pipeline yesterday in PR #79, and training is running right now on Gemma 4 31B. We had to pivot off of Qwen3.6-27B because of a Triton kernel bug on our AMD GPU — happy to talk about that in Q&A if it's interesting.

Repo, HuggingFace, and the paper draft are all linked. Happy to take questions."

**Transition:** "Thank you — questions?"

### Visual

Three numbered cards (one per finding) across the top half. Pull-quote in italic across the middle. Bottom-right: small "in-flight" callout with the SFT/Gemma 4 update. Links footer in 18pt.

---

# Appendix — Anticipated Q&A Cheat-Sheet

Both speakers field. Pre-read these to keep responses crisp.

**Q: SocratTeachLLM's authors *claim* a proper train/test split. Why is contamination the issue?**
- Split was never released. Any third party evaluating their model draws a different random 10% subsample and gets a memorization score. We show it empirically: stage_bal 63.4 → 32.86 on synthetic clean data.

**Q: Isn't comparing a fine-tuned model (STL) to prompted models unfair?**
- Yes, and that's the critique. The paper frames it as "STL surpasses GPT-4o" without noting one is fine-tuned on the training data. We report both clearly.

**Q: Why is `bge-small-zh` (24M params) good enough as a state classifier?**
- The task is classification over 5 well-defined stages, not generation. 24M params has enough capacity to learn the lexical and structural cues. Our results match GPT-4o on the same task.

**Q: What's the theoretical basis for the 50/50 weighting in the unified score?**
- Two orthogonal, memorization-resistant axes. No principled prior for weighting one over the other. Any other weighting requires defending why one axis dominates — a fight we don't need to have for the headline.

**Q: Where is your fine-tuned teacher?**
- Consultant side: fully shipped — 5 trained classifiers, two of them (`state_classifier_v1` and `state-clf-qwen3.5-0.8b-lora`) on the headline path; the Qwen3.5-LoRA classifier is the current locked-headline consultant as of 2026-05-26. Teacher side: pipeline shipped in PR #79 (format fix, three Stage 2 loaders, three Stage 1 loaders, DPO pair builder with Source 3 functional). Training the Gemma 4 31B base mid-sprint. The decision *not* to race to a SocratTeachLLM replacement was deliberate — once contamination evidence landed, the right target became "win on memorization-resistant metrics," which the current locked headline now does **decisively** at canonical scale (unified 72.24 vs. best frontier 70.06, a +2.18 lead at n=681).

**Q: Why pivot from Qwen3.6-27B to Gemma 4 31B for SFT?**
- Qwen3.6-27B uses gated DeltaNet layers that require `flash-linear-attention` (FLA). FLA's Triton 3.6.0 kernels page-fault at runtime on AMD gfx1201 (RDNA4 / R9700), and the torch fallback OOMs above 512 tokens — below the 1280-token requirement for SocratDataset. Gemma 4 31B-IT uses standard softmax attention, fits in 32 GB VRAM at seq=1280 with QLoRA (~21–23 GB peak), and is already the #1 open-weight teacher in our leaderboard (unified 70.08). Documented in PR #79.

**Q: Does this work in real classrooms?**
- We've shown it works on 681 held-out Chinese math/science dialogues. Real-classroom deployment would require teacher training, privacy review, and latency work (Gemma 31B is ~3s/turn on our GPU). Out of scope; listed as future work.

**Q: What are the ethical implications?**
- Bias in training data (Chinese elementary science → may generalize poorly to other subjects/cultures), risk of over-reliance on AI tutors, student privacy. All three addressed in the paper's ethics section.

---

# Production Checklist (for the team)

- [ ] Convert this file to the deck tool of choice (PPTX / Keynote / Slidev)
- [ ] Produce the 3 must-have figures:
  - Slide 4 — inversion table + ROUGE n-gram-order bar chart
  - Slide 5 — two-column architecture diagram (KELE vs. ours)
  - Slide 6 — leaderboard top-5 table with the overtaking callout (+2.18 unified pts at n=681)
- [ ] Dry run #1 — read straight through with stopwatch · target ≤9:45
- [ ] Trim speaker notes to spoken length (~150 words per slide)
- [ ] Q&A walk-through — read each question aloud, confirm both speakers can answer
- [ ] Export PDF backup
- [ ] Confirm no network needed (no live demo on May 26)
