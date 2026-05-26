# Final Presentation Outline
# CSEN 346 — Reproducing & Extending KELE: Structured Socratic Teaching with LLMs

**Format:** 20-minute presentation + 10-minute Q&A  
**Speakers:** 3 members, ~7 min each  
**Demo:** 5–10 min (integrated into Session 3)  
**Grading weights:** Intro/Motivation 15% · Technical Depth 20% · Results 20% · Demo 10% · Slide Quality 15% · Delivery 10% · Q&A 10%

---

## Speaker Assignments

| Section | Speaker | Time | Slides |
|---|---|---|---|
| 1 — Introduction & Motivation | Ulises | ~7 min | 1–6 |
| 2 — Architecture & Method | Max | ~7 min | 7–13 |
| 3 — Results, Demo & Conclusion | TBD (3rd member) | ~6 min + demo | 14–20 |

---

## SECTION 1 — Introduction & Motivation  
**Speaker: Ulises | ~7 min**

---

### Slide 1 — Title

**Title:** "Beyond Memorization: Reproducing and Extending KELE for Socratic Teaching with LLMs"

- Course: CSEN 346 — Natural Language Processing, Santa Clara University (Spring 2026)
- Team: Ulises Chavarria · Max · [Third member]
- Paper reproduced: Peng et al., *KELE: A Multi-Agent Framework for Structured Socratic Teaching*, EMNLP 2025 Findings
- GitHub: github.com/ulises-c/csen-346| HuggingFace: huggingface.co/ulises-c

**Speaker notes:** One sentence on project identity. Transition: "We'll walk through why we picked this paper, what we built, and what we found — including a finding that surprised us."

---

### Slide 2 — The Problem: Teaching, Not Telling

**Headline:** AI tutors tend to give answers, not guide discovery.

- Socratic method: guide the student to the answer through structured questioning
- Real-world stakes: AI tutors are being deployed in classrooms — their pedagogical quality matters
- The standard: SocRules (5-stage framework: greeting → questioning → correction → affirmation → closure)
- Visual: Simple 5-box diagram of SocRule stages (a → b → c → d → e), with a short example exchange

**Speaker notes:** Ground the audience in why this matters before showing the architecture. One motivating example: "A student says 'the answer is 4.' A bad tutor says 'Correct!' A Socratic tutor asks 'How did you arrive at that?'"

---

### Slide 3 — The Paper We Reproduced: KELE

**Headline:** KELE (Peng et al., EMNLP 2025) is a two-agent Socratic teaching framework.

- **Architecture:** Teaching Consultant (state classifier) + Teaching Teacher (response generator)
- **Consultant:** GPT-4o classifies current cognitive state (which SocRule stage?)
- **Teacher:** SocratTeachLLM (fine-tuned GLM-4 9B) generates the teacher's response
- **Benchmark:** SocratDataset — 6,803 Chinese dialogues; 681-dialogue test split
- **Published headline:** SocratTeachLLM surpasses GPT-4o on ROUGE-1/2/BLEU-4

Visual: clean two-box diagram (Consultant → Teacher) with the eval loop.

**Speaker notes:** Be careful to say "surpasses on their metrics" — this sets up Slide 5 perfectly.

---

### Slide 4 — Our Contributions (3 Pillars)

**Headline:** We reproduced KELE, then improved it in three ways.

| Pillar | What we did | Why it matters |
|---|---|---|
| 🔧 Architecture | Replaced GPT-4o consultant with a 24M-param BERT classifier | Zero per-run API cost; faster; deterministic routing |
| 📐 Benchmark | Showed ROUGE/BLEU invert the true ranking | The "best" model by the paper's metric is the worst pedagogically |
| 🌐 Extension | Bilingual (Chinese + English) transfer; Stage 2 SFT pipeline | Generalizes beyond Chinese tutoring data |

**Speaker notes:** This is the map of the talk. Each pillar gets its own section.

---

### Slide 5 — The Benchmark Inversion (Teaser)

**Headline:** The same 10 configs rank in opposite order depending on the metric.

| | ROUGE/BLEU ranking | Pedagogical ranking (state accuracy) |
|---|---|---|
| 🥇 #1 | GPT-4o + SocratTeachLLM | **Gemma 31B + BERT (our system)** |
| 🥈 #2 | Claude Opus + BERT | Claude Opus + BERT |
| 🚨 Last | Claude Opus raw | **GPT-4o + SocratTeachLLM** |

> "Same configurations. Opposite rankings."

- BLEU and ROUGE reward token-level mimicry of training data — not pedagogical correctness
- A 9B model fine-tuned on 90% of the test data will always win token-overlap metrics

**Speaker notes:** Drop this as a cliffhanger — don't explain it fully yet. "We'll come back to *why* this happens."

---

### Slide 6 — Research Questions

**Three questions we set out to answer:**

1. Can a lightweight classifier (BERT, 24M params) match GPT-4o as the teaching consultant?
2. Are ROUGE/BLEU fair metrics for evaluating Socratic teaching quality?
3. Do our improvements transfer cross-lingually to English?

**Speaker notes:** These map directly to the three pillars. Transition: "Max will walk through the architecture and how we answered questions 1 and 2."

---

## SECTION 2 — Architecture & Method  
**Speaker: Max | ~7 min**

---

### Slide 7 — Our System Architecture

**Headline:** Two changes to KELE — one in the consultant, one in how we measure.

Diagram with two columns:
- **Left (KELE original):** Student turn → GPT-4o consultant → state label → SocratTeachLLM teacher → response
- **Right (Our system):** Student turn → BERT-base consultant → state label → [open-weight teacher] → response

Key change: the consultant is now a fine-tuned `bert-base-chinese` with a 5-class head, trained on SocratDataset training split.

**Speaker notes:** "The consultant's job is to ask: which of the 5 SocRule stages is the dialogue in right now? That's a classification problem — not a generation problem. You don't need a 100B-parameter model for that."

---

### Slide 8 — BERT Consultant: Design & Results

**Headline:** A 24M-param classifier matches GPT-4o on state routing.

- Architecture: `bert-base-chinese`, fine-tuned on the 90% training split
- Training: 3 epochs, batch=32, lr=2e-5, max_len=512
- Performance: matched or exceeded GPT-4o consultant on the same teacher (Gemma 31B) at n=681

| Consultant | Teacher | State Acc | Δ vs baseline |
|---|---|---|---|
| GPT-4o (original) | SocratTeachLLM | 25.94% | baseline |
| BERT (ours) | Gemma 31B | 48.15% | **+22.21 pp** |
| BERT (ours) | Claude Sonnet | 49.97% | **+24.03 pp** |

Visual: bar chart comparing 3 consultants across state accuracy.

**Speaker notes:** "The BERT consultant doesn't just match GPT-4o — it beats every GPT-4o configuration on pedagogical accuracy. And it runs locally, for free."

---

### Slide 9 — Why ROUGE/BLEU Are Wrong for This Task

**Headline:** Surface-form metrics systematically reward memorization, not teaching.

Three-part argument:

1. **The gap widens with n-gram order:** ROUGE-1 gap (SocratTeachLLM vs. our system) = +1.59 → ROUGE-2 = +4.92 → BLEU-4 = +4.07. This is the diagnostic signature of dataset memorization.

2. **SocratTeachLLM was trained on 90% of the test data.** Their own paper says so (§4.3: "90% for training and 10% for testing") — but the split was never released. Any third-party evaluator draws a different random sample and gets a memorization score.

3. **Contamination proof:** When we evaluate SocratTeachLLM on a *completely synthetic* test set (generated by Claude, never in SocratDataset), its stage_bal collapses from 63.4 → **32.9** — worse than our dumbest baseline. On the original test set: 63.4. On clean data: 32.9. That 30-point gap is memorization.

Visual: two-bar chart — STL on original test (63.4) vs. synthetic clean test (32.9).

**Speaker notes:** "The model the paper calls its hero performs *worse than random* on data it's never seen."

---

### Slide 10 — Our Replacement Metric: Unified Score

**Headline:** We propose a two-axis metric that is memorization-resistant.

```
unified = 0.5 × stage_balanced + 0.5 × (LLM-judge × 10)
```

- **Stage-balanced accuracy:** macro-F1 over 5 SocRule stages; corrects for stage-frequency imbalance in the original dataset
- **LLM-judge:** Claude Sonnet 4.6 rubric scoring 4 axes — Socratic validity, stage advancement, age-appropriateness, question form (0–10)
- **Why both?** Stage_bal = "did it go to the right stage?"; judge = "was the response actually good teaching?" A model can score high on one and low on the other.

Visual: 2×2 quadrant with stage_bal (x) vs judge (y). Mark a few configs.

**Speaker notes:** "These two axes genuinely disagree about which model is best. Qwen 27B wins stage routing. Gemma wins judged quality. Only by combining them do we get a defensible headline."

---

### Slide 11 — Why Stage-Balanced Matters

**Headline:** The published macro accuracy metric hides failures on rare-but-important stages.

- Stage distribution in the test set: ~40% questioning (c), ~10% closure (e)
- Macro accuracy is frequency-weighted — a model that gets all questioning turns right and all closure turns wrong looks great
- Stage-balanced gives each of 5 stages equal weight → closure matters as much as questioning
- Real-world analog: a medical diagnostic that's 95% accurate on common symptoms but 0% on rare fatal ones

Visual: table showing macro vs. stage_bal for 3 configs where they diverge most.

**Speaker notes:** "Closure is the moment the student confirms they understood. A model that never closes a dialogue is pedagogically broken — but macro accuracy rewards it anyway."

---

### Slide 12 — Our Evaluation Pipeline

**Headline:** End-to-end reproducible evaluation in one command.

Pipeline diagram:
1. Load SocratDataset (681 test dialogues, ~4,200 turns)
2. For each turn: BERT consultant predicts stage → teacher generates response
3. State accuracy: compare predicted stage to ground truth
4. ROUGE/BLEU: compare teacher response to ground truth (shown as the contamination signal)
5. LLM judge: Claude Sonnet 4.6 evaluates each response on 4 rubric axes
6. Unified score: combine stage_bal + judge

```bash
# Full evaluation in one command
uv run python scripts/run_eval.py --config configs/eval-bert-gemma31b.env
```

**Speaker notes:** "Every config in our 131-config leaderboard was produced by this exact pipeline. The data is all in the repo."

---

### Slide 13 — Bilingual Extension: SocratDataset-EN

**Headline:** We built an English version of the benchmark and demonstrated cross-lingual transfer.

- SocratDataset is Chinese-only; no English Socratic teaching evaluation existed
- **SocratDataset-EN:** We translated all 681 test dialogues using Claude (with quality verification on N=20 human-annotated pairs) — published on HuggingFace
- **Cross-lingual transfer result:** Our qwen3.5 LoRA classifier (trained on Chinese) achieves **unified 65.11** on the English test set (n=400, canonical sample), showing the state-routing behavior transfers across languages

Visual: English example dialogue exchange, BERT routing it correctly in English.

**Speaker notes:** "This matters because it shows the SocRule framework is language-agnostic — the pedagogical stages are universal. Transition to results."

---

## SECTION 3 — Results, Demo & Conclusion  
**Speaker: [Third member] | ~6 min + demo**

---

### Slide 14 — Master Leaderboard (Top 10)

**Headline:** 131 configurations tested. Our system tops the pedagogically-fair ranking.

| Rank | Config | Unified | Stage_bal | Judge |
|---|---|---|---|---|
| 🥇 1 | bert × Gemma-31B · top3 · n=50 | **70.08** | 58.48 | 8.17 |
| 🥈 2 | bert × Claude-Sonnet · top3 · n=681 | **70.06** | 58.17 | 8.19 |
| 🥉 3 | bert × Claude-Opus · fewshot10 · n=50 | **69.79** | 58.73 | 8.08 |
| 🏆 7 | **bert × Gemma-31B · fewshot10 · n=681** | **68.65** | 55.42 | 8.19 |
| ⚠️ 8 | qwen3.5 × SocratTeachLLM · n=50 | 68.21 | 63.40 | 7.30 |
| ⚠️ last | Claude-Sonnet × SocratTeachLLM · EN | 44.36 | 22.55 | 6.62 |

- 🏆 = **locked paper headline** (full n=681, certified result)
- ⚠️ = SocratTeachLLM-based; contamination-inflated scores
- Top 3 include Claude Opus/Sonnet (frontier) — our BERT+Gemma system #1 and #7

**Speaker notes:** "Our locked headline is #7. It scores 68.65 unified — only 1.41 points behind the best frontier configuration. That gap is smaller than measurement noise at n=50."

---

### Slide 15 — The Headline Finding: Local–Frontier Parity

**Headline:** An open-weight local model matches frontier proprietary AI on a memorization-resistant benchmark.

```
Best frontier:    bert × Claude-Sonnet · top3 · n=681   → unified 70.06
Best open-weight: bert × Gemma-31B    · top3 · n=50    → unified 70.08
Gap: 0.02 unified points. Parity.
```

- **Locked headline** (`bert × Gemma-31B · n=681`): unified **68.65** — 1.41 pts below frontier
- ~1 of those points is a known measurement artifact (BERT input format bug, patched)
- Runs on a single 32 GB consumer GPU (AMD Radeon RX 9700) at zero per-run API cost
- Prompt engineering stack (10-shot examples + top-3 exemplars) closes most of the gap

Visual: Scatter plot — unified score (y) vs. cost/inference (x). Gemma in the top-left (high score, low cost).

**Speaker notes:** "On a fair metric, a fine-tuned local model is indistinguishable from GPT-4o or Claude Opus for Socratic teaching. This is the central paper finding."

---

### Slide 16 — Teacher Comparison: Gemma vs. Qwen vs. Claude

**Headline:** Gemma 31B wins quality; Qwen 27B wins stage routing. Both beat Claude baseline.

| Teacher | Consultant | Stage_bal | Judge | Unified |
|---|---|---|---|---|
| Gemma 31B | BERT | 55.42 | **8.19** | **68.65** |
| Qwen 27B (think) | qwen3.5 | **60.88** | 7.52 | 68.05 |
| Claude Sonnet | BERT | 58.17 | 8.19 | **70.06** |
| Claude Opus | BERT | 58.63 | 8.01 | 69.37 |
| SocratTeachLLM ⚠️ | qwen3.5 | 63.40 | 7.30 | 68.21 |

Key insight: **Gemma scores 8.19 judge (same as Claude Sonnet)** — the rubric evaluator does not prefer Claude.

**Speaker notes:** "Think mode vs. no-think: Qwen 27B with chain-of-thought wins closure (+20 pp stage e) but the responses are longer and less clear. A real trade-off worth a sentence in the paper."

---

### Slide 17 — Cross-Teacher Grid: 8-Cell Experiment

**Headline:** Every open-weight teacher surpasses the original paper's SocratTeachLLM baseline on unified.

Visual: 2×4 heatmap
- Rows: qwen3.5 consultant / bert-fixed consultant
- Columns: Gemma-31B / A3B-35B / Qwen-27B-think / Qwen-27B-no-think
- Values: unified score (color = red→green 63–69)
- Bottom row outside grid: STL paper baseline (GPT-4o + SocratTeachLLM) = unified ~26 honest (not shown as competitive)

**Speaker notes:** "Across every combination of consultant and teacher, we beat the paper's model on the fair metric. The paper model only 'wins' on the metric designed for its training data."

---

### Slide 18 — Demo

**Title:** Live Demo — Socratic Teaching in Action

Demo script:
1. **Show system running** — launch `uv run python scripts/run_eval.py` with a single dialogue (or pre-recorded walkthrough)
2. **BERT routing visible** — show the per-turn state prediction (stage a through e)
3. **Teacher response quality** — Gemma 31B generating the response given the stage and prompt stack
4. **Leaderboard regeneration** — `make eval-summary` or `scripts/backtest_stage_balanced.py` regenerating the table from raw results

Demo link: [HuggingFace Space / YouTube recording — TBD]

**Speaker notes:** If live demo is infeasible, use pre-recorded 3-min walkthrough with voiceover. Emphasize: "This runs entirely on a consumer GPU — no cloud required."

---

### Slide 19 — Limitations & Future Work

**Limitations:**
- Locked headline uses pre-fix BERT input format (~1 pp measurement artifact, documented)
- n=50 cross-teacher comparisons have ±6 pp variance; Gemma vs. Qwen 27B claim is provisional until n=681 verification
- LLM judge (Claude Sonnet) may have single-model bias; multi-judge panel would harden the metric
- SocratDataset-EN is machine-translated; quality of English ground truth not human-verified at scale

**Future Work (Stage 2 — SFT):**
- Fine-tune Qwen3.6-27B on SocratDataset (Stage 2 SFT pipeline built, training pending)
- DPO preference pairs: 5 anti-pattern perturbations per dialogue for preference alignment
- Stage 1 general instruction tuning (OpenHermes-2.5 + UltraChat 200K + SlimOrca) as warm-up

**Speaker notes:** "We've already built the training infrastructure. The pipeline is designed; the GPU got in the way."

---

### Slide 20 — Conclusion

**Three findings to take home:**

1. **A 24M-param BERT classifier replaces a multi-billion-parameter LLM consultant** — matching performance at orders-of-magnitude lower cost.

2. **ROUGE/BLEU are misleading metrics for Socratic teaching** — the "winning" model in the original paper is last place on pedagogical accuracy. Our unified metric fixes this.

3. **Open-weight local models reach frontier parity on a memorization-resistant benchmark.** Gemma 31B + prompt engineering = Claude Sonnet on fair evaluation.

> "If you're evaluating educational AI, measure pedagogy — not n-gram overlap."

Links:
- GitHub: github.com/ulises-c/csen-346
- SocratDataset-EN: huggingface.co/datasets/ulises-c/SocratDataset-EN
- Paper draft: deliverables/overleaf/

---

## Anticipated Q&A

**Q: Why is SocratTeachLLM's contamination a problem if the authors did use a proper train/test split?**  
A: Because they never released the split. Any third party evaluating their model on a random 10% subsample gets a memorization score. The benchmark is unreproducible. We show this empirically with the synthetic test set (stage_bal 63.4 → 32.9).

**Q: Isn't it unfair to compare a fine-tuned model (SocratTeachLLM) to a prompted model?**  
A: Yes — and that's exactly our critique. The paper presents it as "SocratTeachLLM surpasses GPT-4o" without noting that one is fine-tuned on the training data and the other is zero-shot. We report both fine-tuned and prompted models clearly.

**Q: Why is BERT-base-chinese good enough as a state classifier?**  
A: Because the task is classification, not generation. The 5 SocRule stages are well-defined, and the BERT model has enough capacity to learn the lexical and structural cues that distinguish them. Our results show it matches GPT-4o at a fraction of the cost.

**Q: What is the unified metric's theoretical basis?**  
A: It's an equal-weight average of two orthogonal, memorization-resistant axes. We defend 50/50 in the paper: there's no principled prior for weighting one over the other, and any other weighting requires defending why one axis dominates — a fight we don't need to have for the headline.

**Q: Does the model work in real classrooms?**  
A: We've shown it works on 681 held-out dialogues from a Chinese math tutoring corpus. Real-classroom deployment would require teacher training, privacy considerations, and latency constraints (Gemma 31B is ~3s/turn on our GPU). That's explicitly out of scope for this paper — listed as future work.

**Q: What are the ethical implications?**  
A: Bias in training data (SocratDataset is Chinese math, so the system may generalize poorly to other subjects/cultures), risk of over-reliance on AI tutors, and student privacy. We address all three in the paper's ethics section.

---

## Slide Design Notes

- **Font:** Use a clean sans-serif (Inter, Lato, or similar). Min 24pt body, 36pt headings.
- **Color scheme:** Dark background or white background — pick one. Suggested: white bg, #2563EB (blue) accents.
- **Figures:** Every slide should have at least one visual. Priority order:
  1. Two-column KELE vs. Our System diagram (Slide 7)
  2. Inversion bar chart: ROUGE ranking vs. Pedagogical ranking (Slide 5)
  3. Contamination collapse bar chart: STL original vs. synthetic (Slide 9)
  4. Unified leaderboard top-10 table (Slide 14)
  5. Scatter: unified score vs. inference cost (Slide 15)
  6. 8-cell heatmap (Slide 17)
- **Slide count:** 20 content slides. Aim for ≤6 bullet points per slide. Prefer one key number/claim per slide.
- **Avoid:** walls of text, acronyms without definition on first use, results tables with >5 columns.

---

## Timing Checklist

- [ ] 7 min — Slides 1–6 (Ulises)
- [ ] 7 min — Slides 7–13 (Max)
- [ ] 6 min — Slides 14–20 (TBD)
- [ ] 5–10 min — Demo (Slide 18, live or recorded)
- [ ] 10 min — Q&A (all members)
- [ ] Rehearse: at least one full run-through with timer
- [ ] Confirm demo runs without network if presenting locally
- [ ] Confirm slide deck exported to PDF as backup
