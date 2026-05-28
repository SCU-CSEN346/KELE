# CSEN 346 — Final Project Status Report

**Project:** Reproducing and Extending KELE — A Multi-Agent Framework for Structured Socratic Teaching with LLMs
**Team:** Cyberdyne — Ulises Chavarria · Maximilian Khan *(2-member team; no 3rd member joined)*
**Course:** CSEN 346, Santa Clara University, Spring 2026
**Status snapshot date:** 2026-05-25 *(updated 2026-05-26 with locked-headline promotion — see §1.2)*

> **🏆 Update 2026-05-26 — Locked headline promoted; frontier overtaken.** Post-snapshot, TODO #14 cell #2 (`qwen3.5 × Gemma-31B · fewshot10 · n=681`) landed at unified **72.24** (master #1) and was promoted to the paper's locked headline, beating the best frontier configuration we tested (`bert × Claude-Sonnet · top3 · n=681`, unified 70.06) by **+2.18 unified pts** at canonical n=681. The 2026-05-23 parity framing has inverted to overtaking. The prior 2026-05-18 BERT-classifier locked headline (unified 68.65) is now at master #8 and preserved as the BERT-classifier-axis architectural result. TODO #14 status: **3 of 4 cells done**; remaining cell #1 (`bert-fixed × Gemma-31B · n=681`) is no longer load-bearing for the overtaking claim. Section-level updates in §1.2 below; older paragraphs reflect the 2026-05-25 snapshot.
**Two distinct deliverable dates (per instructor, 2026-05-25):**
- **2026-05-26 — In-class presentation:** 5 min per member × 2 members = **10 min talk** + **10 min Q&A**. **No demo on this date.**
- **2026-06-04 — Final submission:** paper + code + HF artifacts + **poster + demo**.

**Talk structure (May 26):** 2 speakers × 5 min each. The 20-slide outline in `OUTLINE.md` was budgeted for the rubric's nominal "20-min talk + demo + Q&A" format and needs to be **compressed to ~6–8 slides total** for the 10-min slot. Demo moves out of the talk and into the June 4 deliverable. See §2.1 for the compression plan.

This document audits where the project stands against the **course rubric** (`references/requirements/project_guidelines.md`), the **initial pitch** (`deliverables/initial-presentation/Project Idea Presentation.pdf`), and the **final-presentation outline** (`deliverables/final-presentation/OUTLINE.md`). Section 1 summarizes what is done. Section 2 lists the open work in priority order with deadlines. Section 3 catalogs known risks. Section 4 is a one-glance checklist.

---

## 1. What is done

### 1.1 Pivots from the initial pitch

The initial-presentation deck (2026-04-07) proposed four future-improvement directions. The campaign delivered on two of them in depth, dropped two, and added three significant contributions that were not in the original pitch.

| Initial-pitch direction | Status | Notes |
|---|---|---|
| **(1)** Mixture of Experts for multi-subject teaching (gating) | **Dropped** | SocratDataset is single-domain (elementary science); no multi-subject corpus available within budget. Not pursued. |
| **(2)** RL-based stage transitions (replace rigid rules) | **Dropped** | Replaced by a stronger empirical finding — KELE's SocRule transitions are not the bottleneck; the *consultant* classifier is. Pivot is documented in `docs/EXPERIMENT_LOG.md`. |
| **(3)** Learned consultant (discrete classifier replacing LLM) | **✅ Shipped — became the project's locked headline.** | 24M-param `bge-small-zh` BERT consultant. 86.55% stage acc / 61.64% state acc at <100s training. Replaces the GPT-4o consultant entirely; +22.21 pp absolute state-acc lift at n=681. |
| **(4)** Stronger evaluation (ablation + cross-domain) | **✅ Shipped — significantly expanded.** | 143 model variants measured, 38 with full unified score (as of 2026-05-26), 8-cell cross-teacher matrix, 4-metric memorization-resistant evaluation panel, bootstrap convergence analysis (n=400 sufficient). |

**New contributions not in the original pitch (all three load-bearing in the paper):**

1. **Methodological critique of the published benchmark** — surface-form metrics (ROUGE/BLEU) systematically reward training-data memorization over teaching capability. The same 10 configurations rank in opposite orders under surface-form vs. pedagogical metrics. Full proof in `docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md` and `docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md`.
2. **Unified score (`unified = 0.5 × stage_bal + 0.5 × (judge × 10)`)** — a memorization-resistant single-number ranking. Spec in `docs/UNIFIED_RANKING.md`. Used as the primary headline metric in the paper.
3. **SocratDataset-EN** — full English translation of all 6,803 dialogues, published on HuggingFace, with a cross-lingual transfer experiment confirming the consultant's pedagogical routing is language-invariant (unified 65.11 at canonical n=400 on EN).

### 1.2 Locked headline results

All numbers from `results/master_leaderboard.md`, `results/_orchestrator_logs/backtest_stage_balanced_latest.md`, and the per-config `metrics_summary.json` / `judge_summary.json` files. Backed by `scripts/backtest_stage_balanced.py`.

| Ranking | Cell | n | Unified | Stage_bal | Judge | R-1 |
|---:|---|---:|---:|---:|---:|---:|
| 🥇 1 | `bert × Gemma-31B · composed · top3 · n=50` | 278 | **70.08** | 58.48 | 8.17 | 41.13 |
| 🥈 2 | `bert × Claude-Sonnet · top3 · n=681` | 3840 | 70.06 | 58.17 | 8.19 | 41.93 |
| 🥉 3 | `bert × Claude-Opus · fewshot10 · n=50` | 271 | 69.79 | 58.73 | 8.08 | 42.77 |
| 🏆🥇 1 | **`qwen3.5 × Gemma-31B · fewshot10 · n=681`** ← **CURRENT LOCKED HEADLINE (2026-05-26)** | 3974 | **72.24** | 61.32 | 8.32 | 37.65 |
| 8 | `bert × Gemma-31B · fewshot10 · n=681` ← prior locked headline (2026-05-18); preserved as BERT-classifier-axis result | 3834 | 68.65 | 55.42 | 8.19 | 36.78 |
| ⚠️ 8 | `qwen3.5 × SocratTeachLLM · fewshot10 · n=50` (contamination-driven) | 288 | 68.21 | 63.40 | 7.30 | 48.07 |
| ⚠️ last (37) | `Claude-Sonnet × SocratTeachLLM · EN · n=50` | 303 | 44.36 | 22.55 | 6.62 | 55.85 |

**Local-vs-frontier at canonical n=681 (3 of 4 TODO #14 cells landed):** `qwen3.5 × Gemma-31B` leads by **+2.18 pts** (current locked headline, overtaking); `qwen3.5 × A3B-35B` trails by 2.25 pts; `qwen3.5 × Qwen-27B no-think` trails by 3.35 pts. The 2026-05-23 parity framing has inverted to overtaking: teacher choice is the binding driver and Gemma 31B + Qwen3.5-LoRA classifier overtakes; A3B and Qwen-27B don't. Remaining cell #1 (`bert-fixed × Gemma-31B · n=681`) is no longer load-bearing for the overtaking claim.

### 1.3 Fine-tuning progress — consultant vs. teacher

Fine-tuning was originally a single line item in the initial pitch ("learned consultant via discrete classifier replacing LLM"). It split into two streams during the campaign. **The consultant stream is fully shipped; the teacher stream is plumbed but unrun.** Treat this as the single biggest pre-existing-knowledge gap when reading the rest of the report.

#### A. Consultant-side fine-tuning — ✅ SHIPPED (5+ trained models, all in `results/`)

This is the line of work the locked headline rides on. Every checkpoint listed below was trained, evaluated on the 681-dialogue test split, and is reproducible from the scripts in the repo.

| Checkpoint | Base | Method | Params | Train time | Test state acc | Test stage acc | Role |
|---|---|---|---:|---:|---:|---:|---|
| `state_classifier_v1` | `bge-small-zh-v1.5` | Frozen base + 5-way + 34-way heads | 24M | 92s + 148s | 61.64% | 86.55% | **Locked-headline consultant** (paper §4.6; the `bert` cells in master leaderboard) |
| `state_classifier_v2` | `bge-small-zh-v1.5` | + inverse-frequency class weighting + label smoothing | 24M | 211s | 59.94% | — | Rare-state-recovery ablation; documents the state-acc vs. R-1 tradeoff axis (paper §4.6 ablation paragraph) |
| `state-clf-qwen3-emb-0.6b-frozen` (T1) | `Qwen3-Embedding-0.6B` | Frozen base + head | 600M | — | (frozen-probe baseline) | — | Funnel rung 1 — locked in HF publishing plan |
| `state-clf-qwen3-emb-0.6b-lora` (T2) | `Qwen3-Embedding-0.6B` | LoRA r=16 | 600M | — | — | — | Funnel rung 2 |
| `state-clf-qwen3.5-0.8b-frozen` (T3) | `Qwen3.5-0.8B` | Frozen base + head | 800M | — | — | — | Funnel rung 3 |
| `state-clf-qwen3.5-0.8b-lora` (T4) | `Qwen3.5-0.8B` | LoRA r=16 | 800M | — | **67.57%** | a:100 / b:90.9 / c:85.1 / d:75.9 / e:96.8 | **Funnel winner** — used as the `qwen3.5` consultant in every post-2026-05-22 cell incl. bilingual probes and 3 of 4 TODO #14 cells |

The full consultant-upgrade funnel is documented in `docs/CONSULTANT_UPGRADE_LOG.md` (49 KB log of decisions and ablations). Training script: `scripts/train_state_classifier_34way.py`. Both T4 and `state_classifier_v1` are the consultants behind the leaderboard's `qwen3.5 × …` and `bert × …` cells respectively — they are not abstractions, they are checkpoints sitting in `results/state-clf-qwen3.5-0.8b-lora/final/` and `results/state_classifier_v1/final/` right now.

#### B. Teacher-side fine-tuning — ⏳ PIPELINE READY · EXECUTING IN NEXT SPRINT (gated, Jun 4 target)

The original KELE paper's "headline" was `SocratTeachLLM` — a GLM4-9B fine-tuned on SocratDataset. Our equivalent ambition (Slide 19 of the outline; §`sec:nextsteps` of the paper) is to fine-tune our own teacher checkpoint. **This is Ulises's individual contribution** (base-model search → SFT pipeline design → execution); per the 2026-05-25 scope decision, it has been **promoted from stretch goal to a Jun 4 paper deliverable** under hard gates (§2.8). Current state: **everything is wired except the training run itself.**

| Component | Status | Path / Evidence |
|---|---|---|
| Plan-of-record | ✅ Complete | `docs/TRAINING_PLAN.md` — 515 lines, 3-stage curriculum: scoped general SFT → Socratic SFT (2a breadth → 2b structural) → DPO |
| Base-model decision | ✅ Locked | Qwen3.6-27B-Instruct (Phase 1, velocity choice); Gemma 4 31B planned head-to-head (Phase 2) — TRAINING_PLAN §0.1 |
| QLoRA training script | ✅ Written | `scripts/train_sft.py` (310 lines) |
| Training config | ✅ Created | `configs/train-sft-qwen36-27b-qlora.env` — QLoRA r=16, 3 epochs, lr=5e-5, batch 1×16, max_seq 2048 |
| VRAM budget verified | ✅ | 20–23 GB peak on 32 GB R9700 with grad checkpointing |
| Datasets staged | ✅ All 6 on HF | `socrat-zh`, `socrat-en`, `socrateach-multi`, `socrateach-single`, `socraticmath`, `socrat-synthetic` |
| Synthetic baseline (contamination control) | ✅ n=37 measured | `results/synthetic-baseline/` — Qwen 27B Q4 no-think (23.44%), Qwen 27B Q4 think-4096 (26.92%), Gemma 4 31B Q5 (27.23%) — Wilson 95% CIs overlap by ~7pp |
| Branch with WIP code | ✅ | `feat/stage2-sft-pipeline-design` |
| **`dataset.py` SFT format fix** (TRAINING_PLAN §0.2) | **🔥 BLOCKER** — not yet applied | Current `src/project/dataset.py:103,167` prepends `[State: X] [Action: Y]` to the assistant target; this would train the model to emit literal `[State: c12]` strings in user-facing replies. Fix is specified (move state/action to user turn), not yet committed. |
| Synthetic baseline extension to n=75 / n=100 | ⏳ Planned (E1) | `n=37` Wilson SE is ~3pp overall and ~9pp per-stage — too loose to measure the post-SFT lift with confidence |
| LLM-judge on synthetic baselines | ⏳ Planned (E2, ~$5) | Places baselines on `unified` so the post-SFT lift is reportable on the project's primary metric |
| Stage 1 dataset loaders (OpenHermes-2.5, UltraChat-200K, SlimOrca) | ⏳ Planned | TRAINING_PLAN §2 Stage 1 — ~30–50k scoped records |
| Stage 2a → 2b training launch | ⏳ Pending | ~6–12 GPU-h on R9700 |
| Stage 3 DPO data build (`scripts/build_dpo_pairs.py`) | ⏳ Not written | Plan: judge-mined pairs + STL-as-rejected on synthetic + 5 programmatic perturbations → ~150–200k pairs, zero API regen cost |
| Stage 3 DPO training (`scripts/train_dpo.py`) | ⏳ Not written | TRL `DPOTrainer`; beta=0.1, lr=1e-5, 1–2 epochs |
| Phase 2 — Gemma 4 31B head-to-head SFT | ⏳ Planned | Mirror config with `TRAIN_BASE_MODEL=google/gemma-4-31b-it` |

**Why teacher fine-tuning is *not* the locked headline.** The campaign uncovered that the original SocratTeachLLM's apparent excellence is largely training-data memorization (`docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md`: stage_bal collapses 63.4 → 32.86 on the synthetic clean test set). Once that landed, the value proposition of "re-do SocratTeachLLM with a stronger base" weakened: the right comparison is no longer "match SocratTeachLLM's ROUGE-1" but "lift `unified` on the synthetic-clean evaluation past Gemma + 10-shot's 56.13." That target is achievable but is **strictly future work** for the paper; the locked headline does not need it.

**Bottom line on fine-tuning.** Consultant side is done in depth — five fine-tuned classifiers, two of which (`v1` and T4-LoRA) are the consultants that produce every locked-headline number in the paper. Teacher side is the **next sprint's deliverable** (§2.8): Ulises's individual contribution path, gated on (a) May 26 deck being rehearsed and (b) synthetic-baseline extension to n=75 completing. If either gate fails, teacher SFT slips to documented-future-work (paper-as-is is already an A submission); if both pass, Stage 2a → 2b runs overnight and lands in the paper as a new row in Tables 6/14 before the Jun 4 paper-trim sprint.

### 1.4 Course rubric coverage

Cross-checked against the seven graded sections in `references/requirements/project_guidelines.md`. Three artifacts (paper, code, presentation) each count 20%; poster counts 10%.

#### Paper (`deliverables/overleaf/latex/acl_latex.tex`, 770 lines, ACL template)

| Section | Required for | Status |
|---|---|---|
| Abstract | All deadlines | ✅ Written |
| §1 Introduction & Motivation | Apr 14 | ✅ Written |
| §2 Related Work | Apr 14 | ✅ Written |
| §3 Dataset & Methodology | Apr 23 | ✅ Written (SocRule, SocratDataset, SocratDataset-EN, reproduction setup) |
| §4 Evaluation & Results | May 5 | ✅ Written (smoke/mini/full protocol, locked headlines, frontier ceiling, contamination proof, n=400 convergence) |
| §5 Conclusion | May 14 | ✅ Written |
| Limitations | May 14 | ✅ Written (7 paragraphs incl. cross-architecture scaling, n=50 vs n=681, schema fallbacks, language scope, pre-fix consultant artifact) |
| Ethics Statement | May 14 | ✅ Written |
| References (`custom.bib`) | All deadlines | ✅ Populated |

**Length & dual-target strategy:** The current `acl_latex.tex` draft (~12+ compiled pages) serves as the **long-form submission-candidate paper** (§2.2b stretch — for an actual venue submission post-graduation). A **separate 6-page `acl_latex_6page.tex`** must be derived from it as the class deliverable (§2.2 BLOCKING). The class version is a compressed subset; the long-form is preserved and polished separately for ACL Rolling Review / EMNLP Findings.

#### Code (`src/project/`, `scripts/`, `configs/`, `tests/`)

| Requirement | Status | Evidence |
|---|---|---|
| Clear function/variable names | ✅ | `src/project/{kele,socratic_teaching_system,socratic_teaching_unified,evaluate,metrics}.py` |
| Docstrings on functions & classes | ✅ | Confirmed in source; PLAN.md checked |
| Inline comments where needed | ✅ | Present in non-trivial logic |
| External library justification | ✅ | `pyproject.toml` + README dependency note |
| README — model description | ✅ | `README.md` §1–§3 |
| README — installation instructions | ✅ | `README.md` §Python Environment, §Dependencies |
| README — usage / reproduction | ✅ | `README.md` §Running the Experiments, §Common Commands |
| README — expected output | ✅ | Per-config `metrics_summary.json` schema documented |
| README — member contributions | **⏳ Missing** | Add a "Member Contributions" section before Jun 4 |
| `.env.example` | ✅ | Present, documented in PLAN.md |
| Test suite | ✅ | `tests/` — 11 test files, CI badge in README, coverage badge live |

#### Presentation (`deliverables/final-presentation/`)

| Element | Status |
|---|---|
| Slide-by-slide outline (`OUTLINE.md`) | ✅ exists, but **scoped to 20 min + demo** — must be compressed to ~6–8 slides for the 10-min May 26 slot |
| Compressed deck (10-min, 2 speakers × 5 min) | **⏳ Not started** |
| Slide deck (PPTX/PDF) | **⏳ Not started** |
| Speaker split | ✅ 2-member team confirmed; natural split is Ulises = motivation + method, Max = results + conclusion |
| Rehearsal timing run (≤10 min) | ⏳ Pending |
| Backup PDF export | ⏳ Pending |
| Demo (June 4, not May 26) | ⏳ Scripted in OUTLINE.md Slide 18; recording/Space not produced — moved to June 4 deliverable |

#### Poster (Jun 4)

| Element | Status |
|---|---|
| Layout & content | **⏳ Not started** |
| Key panels identified | Per PLAN.md §Poster: problem · KELE architecture · our improvement · results comparison |
| Following "more images, minimal text" rule | N/A yet |

#### HuggingFace artifacts (data + model release)

| Artifact | URL | Status |
|---|---|---|
| `ulises-c/SocratTeachLLM` (model mirror) | live | ✅ |
| `ulises-c/SocratDataset` (Chinese, 6,803 dialogues) | live | ✅ |
| `ulises-c/SocratDataset-EN` (English translation) | live | ✅ |
| `ulises-c/SocratDataset-SYNTHETIC` (contamination probe) | live | ✅ |
| State-classifier funnel (5 models) | per `docs/HF_PUBLISHING_PLAN.md` | **⏳ Plan written, tooling specced, not yet published** — blocked on Max's HF account confirmation + execution-day checklist |
| Stage 2 SFT checkpoint (Qwen3.6-27B QLoRA) | n/a | **⏳ Pipeline designed, not yet trained** (see §2.5) |

### 1.4 Experimental campaign at a glance

- **143 model variants** measured end-to-end on the SocratDataset 681-dialogue test split (and subsets), 38 with full unified score (as of 2026-05-26).
- **37 configurations** have both stage_balanced AND LLM-judge scores (the unified-ranking tier).
- **7 full-scale n=681 runs** (Gemma 31B, A3B-35B, Qwen-27B no-think, Claude Sonnet, Claude Opus, BERT integrations, plus the GPT-4o + SocratTeachLLM baseline reproduction).
- **8-cell cross-teacher matrix at n=50** (post-fix consultant) for like-for-like comparison.
- **Bootstrap convergence analysis** across all seven n=681 runs → **n=400 is sufficient** at ε=2pp for all four memorization-resistant metrics (41% compute saving with no ranking loss).
- **Compute economics:** locked headline runs at 12.9 GPU-hours / $0 API per full evaluation on a single 32 GB consumer GPU (RTX 5090 / AMD R9700).

---

## 2. Open work, ranked by deadline

Two-week sprint to **Jun 4 final submission**. Items are ordered by what blocks the most downstream work.

### 2.1 (BLOCKING) — Compress and build the May 26 deck (10 min total, 5 per speaker)

The existing `OUTLINE.md` is 20 slides scoped for a 20-min talk with integrated demo. The May 26 slot is **10 min talk + 10 min Q&A, no demo**. Compression is the actual blocker.

**Proposed 8-slide compression (Ulises 5 min, Max 5 min):**

| # | Slide | Speaker | Source in OUTLINE.md |
|---|---|---|---|
| 1 | Title + team | Ulises | Slide 1 |
| 2 | Problem + KELE in one slide | Ulises | Slides 2–3 collapsed |
| 3 | Our contributions (3 pillars) | Ulises | Slide 4 |
| 4 | **The benchmark-inversion finding** (teaser → punchline) | Ulises | Slides 5 + 9 collapsed; the one slide most likely to draw Q&A |
| 5 | BERT consultant + unified metric (architecture in one breath) | Max | Slides 7–8 + 10 collapsed |
| 6 | **Master leaderboard top-5 + the local–frontier parity finding** | Max | Slides 14–15 collapsed |
| 7 | Bilingual + contamination evidence (one slide) | Max | Slides 9 + 13 collapsed |
| 8 | Conclusion + 3 takeaways + links | Max | Slide 20 |

- [ ] **Lock the 8-slide compression** with Max before slide production starts (decide if pillar 3 = bilingual or fine-tuning).
- [ ] **Build the deck** (PPTX / Keynote / Slidev). Estimated 3–5 hours.
- [ ] **Build the 3 must-have visuals** (the 5+ in OUTLINE.md will not fit; pick the highest-impact three):
  1. Two-column KELE vs. our system diagram (Slide 5 in new deck)
  2. Inversion bar chart: ROUGE ranking vs. pedagogical ranking (Slide 4 in new deck)
  3. Unified leaderboard top-5 table with parity callout (Slide 6 in new deck)
- [ ] **Timing rehearsal — single run with a stopwatch.** Target: ≤9:45 to leave margin for transition. The 10-min cap is hard ("you won't be allowed to continue after").
- [ ] **Prepare Q&A defensive material** (10 min Q&A is half the slot — preparation matters). Use `OUTLINE.md §Anticipated Q&A` as the source; expand 1–2 lines on each anticipated question. Key topics: contamination claim, BERT vs. LLM consultant, unified-metric weighting, real-classroom deployment, ethics.
- [ ] **Export deck to PDF as backup.**

### 2.2 (BLOCKING for June 4) — Class-submission paper (6-page ACL trim)

The class rubric requires a **4–6 page research paper (excluding references)**. The current `acl_latex.tex` is ~12+ compiled pages — built as a long-form draft to serve both this 6-page class deliverable *and* the longer-form submission candidate in §2.2b. The 6-page trim is the required artifact for grading.

**Strategy:** maintain two LaTeX targets from the same content base:
- `deliverables/overleaf/latex/acl_latex.tex` — long-form, conference-submission candidate (see §2.2b)
- `deliverables/overleaf/latex/acl_latex_6page.tex` (new) — class deliverable, 6 pages body

The cleanest setup is a new top-level `.tex` that `\input{}`s the same shared sections (intro, dataset, methodology, conclusion, ethics) and a slimmed Results section with the trim cuts inlined. Alternatively, branch the file and trim aggressively — easier to ship, harder to keep in sync. Given the deadline, **branch-and-trim is the recommended path** for the 6-page version.

- [ ] **Create `acl_latex_6page.tex`** from `acl_latex.tex` and trim to ≤6 body pages.
- [ ] **Identify aggressive cuts.** Likely candidates: the §4.5 tournament 13-model table → 3-row best-of summary; §4.7 (Gemma pivot retraction) → one paragraph; §4.12 (cross-architecture scaling discussion) → Limitations footnote; multiple per-stage tables → fold into one.
- [ ] **Re-run [Agentic Reviewer](https://paperreview.ai/)** on the 6-page version before submission (per project guidelines and `docs/PLAN.md`).
- [ ] **Add a "Member Contributions" paragraph** (rubric requirement under code submission; safest to also include in the paper).
- [ ] **Final BibTeX cleanup** — ensure every `\citep` in the 6-page version resolves; check `custom.bib` against the cited keys.
- [ ] (Conditional) **Add the SFT row + paragraph** if §2.8 lands — must fit within the 6-page budget.

### 2.2b (STRETCH) — Full-length paper submission candidate

The current `acl_latex.tex` long-form draft is already a strong basis for an actual venue submission (ACL Rolling Review, EMNLP Findings, or similar). The 6-page class version is necessarily a compressed subset; the long-form retains the methodological depth — full smoke/mini/full gating protocol, complete cross-teacher matrix, contamination-proof appendix, bootstrap convergence analysis, frontier-ceiling discussion — that strengthens external review.

This is a *post-graduation* artifact, not a Jun 4 deliverable. But the trim work in §2.2 *removes* content from the 6-page version, which means the long-form draft must be preserved as a separate file before trimming, not edited in place.

- [ ] **Preserve `acl_latex.tex` as the long-form file** before §2.2 begins (do not branch-and-trim *in place*).
- [ ] **Post-Jun 4 polish pass** on the long-form: tighten abstract, harden Limitations §, ensure all claims are reproducible from `results/` artifacts, run a second Agentic Reviewer pass.
- [ ] **Target a venue** — ACL Rolling Review (continuous), EMNLP 2026 Findings (~Aug deadline), or NAACL 2026 Industry/Findings track. The benchmark-critique angle and the BERT-consultant + open-weight-parity finding are both venue-worthy on their own; together they form a complete contribution.
- [ ] **Anonymize + reformat** per target venue's submission guidelines (often the bottleneck — anonymized HF repo mirror, anonymous GitHub fork).
- [ ] (Optional) **Add SFT results across both papers** if §2.8 lands, with extended ablation tables that don't fit in the 6-page version.

**Why stretch:** Class grade does not depend on venue submission. Adds prestige and lifetime value of the project, but only after the Jun 4 package is complete.

### 2.3 (BLOCKING for June 4) — Poster

Per the rubric: "more images, minimal text." Per instructor confirmation (2026-05-25): the poster is a Jun 4 deliverable, not May 26.

- [ ] **Draft poster layout** — 5 key panels: problem motivation · KELE architecture · our improvement (BERT consultant + 10-shot) · contamination/inversion finding · unified-metric leaderboard with parity callout.
- [ ] Pull figures from the same set produced for the slide deck (§2.1) — do not regenerate from scratch.
- [ ] Print PDF + arrange physical poster delivery per course logistics.

### 2.4 (BLOCKING for June 4) — Demo

Per instructor (2026-05-25): demo is a Jun 4 deliverable, not part of the May 26 in-class slot. This frees the team to record (not perform live) and submit a polished artifact.

- [ ] **Record a 3-min screencast** showing a single SocratDataset dialogue going through the BERT consultant → Gemma 31B teacher pipeline turn-by-turn. Suggested takes:
  - Show `uv run python -m src.project.kele test --bert-consultant results/state_classifier_v1/final --n 1 --output results/demo` against a pre-seeded dialogue.
  - Cut in `make eval-summary` or `scripts/backtest_stage_balanced.py` regenerating the leaderboard from raw results.
  - Voiceover hits the three findings (BERT replaces the LLM consultant, ROUGE inversion, local–frontier parity).
- [ ] **Host** the recording on HuggingFace Spaces, YouTube, or Google Drive — any of the three is accepted per rubric.
- [ ] **Reference the demo link in the paper** (currently `[TBD]` in the paper's §Evaluation section).
- [ ] Also reference in poster QR code if room.

### 2.5 (BLOCKING for June 4) — README final polish

- [ ] Add **Member Contributions** section to `README.md` (rubric requirement; currently missing per `docs/PLAN.md` checklist). The actual split must be confirmed by the team, but as a starting point: **Ulises** — base-model search for teacher fine-tuning (Qwen3.6-27B vs. Gemma 4 31B head-to-head on synthetic baseline; SFT pipeline design in `docs/TRAINING_PLAN.md`), plus shared contributions across campaign work; **Max** — BERT consultant funnel (`bge-small-zh` baseline + Qwen3-emb T1/T2 + Qwen3.5-0.8B T3/T4 LoRA), prompt-engineering tournament, n=681 parity cells, HF publishing plan. Other workstreams (fusion architecture, evaluation infrastructure, contamination critique, SocratDataset-EN, paper draft, leaderboard tooling) are shared and should be attributed per actual ownership.
- [ ] Verify all HF links resolve.
- [ ] Add a "Demo link" line under the Hugging Face section once the demo recording is hosted (after §2.4).

### 2.6 (STRETCH, not blocking submission) — Publish state-classifier funnel to HF

Plan-of-record: `docs/HF_PUBLISHING_PLAN.md`. Five model repos under `maxjkh/…` plus an HF Collection.

- [ ] Confirm Max's HF namespace (`maxjkh`).
- [ ] Implement `scripts/build_model_cards.py` (~150 lines) and `scripts/publish_to_hf.py` (~50 lines) per the plan.
- [ ] Dry-run, then real publish — start with the smallest (`bge-small-zh`, ~95 MB) before the 2.3 GB Qwen3-family ones.

**Why stretch:** the locked headline uses the `state_classifier_v1` BERT checkpoint, which is in `results/` and reproducible from `scripts/train_state_classifier_34way.py`. Public HF release strengthens the submission but is not gating.

### 2.7 (STRETCH, not blocking submission) — TODO #14 remaining cell

`docs/EXPERIMENT_LOG.md` (2026-05-25 + 2026-05-26 entries) — **3 of 4 cells landed**. Cell #2 (`qwen3.5 × Gemma-31B · fewshot10 · n=681`) landed 2026-05-26 at unified 72.24 and was promoted to the locked headline (overtaking frontier by +2.18 unified pts). One outstanding apples-to-apples n=681 cell:

- [ ] `bert-fixed × Gemma-31B · fewshot10 · n=681` (~12 GPU-h, ~$15 judge spend) — closes the BERT-vs-Qwen3.5-LoRA classifier comparison at canonical scale under matched post-fix conditions. **No longer load-bearing for the overtaking claim** — only confirms the BERT-classifier-axis result at canonical scale.

Outcome would tighten the Limitations §"pre-fix consultant input-format artifact" claim. Strictly an evidence upgrade — not gating for the paper or presentation.

### 2.8 (BLOCKING-GATED for June 4) — Teacher SFT (Stage 2b on Gemma 4 31B QLoRA, executed on 5090)

**Pivot 2026-05-26 — read this first.** The prior plan had Ulises executing Qwen3.6-27B QLoRA → fallback Gemma 4 31B on his R9700. That has been **superseded**:

- **Qwen3.6-27B SFT is DROPPED.** Not enough time before June 4, and Gemma 4 31B is the headline-teacher (the locked headline at unified 72.24 uses base Gemma 4 31B — fine-tuning *that* model is the natural extension).
- **Gemma 4 31B SFT moves from Ulises's R9700 to Max's 5090.** RDNA4 has FLA Triton page faults (Qwen) and HF loader RAM-storm risk (Gemma) — both eliminated on the 5090's CUDA stack. 5090 box verified: torch 2.11.0+cu130, bitsandbytes 0.49.2, peft 0.19.1, trl 1.4.0, transformers 5.9.0, 122 GB RAM, 1.3 TB disk free.
- **Track B (Qwen 27B head-to-head)** stays in `acl_latex.tex` §`sec:nextsteps` as documented future work. Configs `train-sft-qwen36-27b-qlora.env` and `train-sft-stage2-socratic.env` remain in the repo for that path.

**Owner split (revised):**
- **Ulises** — SFT pipeline design (still the bulk of his individual contribution): dataset format Option B fix, Stage 1 ShareGPT loaders, Stage 2 Socratic loaders, DPO pair builder, train configs for both Gemma 4 31B and Qwen3.6-27B, base-model selection process, HF dataset publishing (PRs **#79** and **#90**).
- **Max** — SFT **execution** on the headline-teacher (Gemma 4 31B QLoRA on 5090), post-SFT eval pipeline (LoRA→GGUF merge + reuse existing eval infra), result integration into paper Tables 6/14 + locked-headline reconciliation.

This is a cleaner contributions split than the prior framing — pipeline design vs. headline-model execution. Both substantial, both publishable.

**Plan-of-record:** `docs/TRAINING_PLAN.md` (Stage 2 §4), `docs/HANDOFF_GEMMA4_SFT.md` (on PR #79 branch). The handoff was written for Ulises's R9700 path; the steps below adapt it to the 5090.

**Hard gates (all three must hold before Stage 2b launches):**

- **Gate A** — May 26 class talk delivered (§2.1 closed). ✅ DONE (2026-05-26).
- **Gate B** — PRs **#79** and **#90** reviewed and merged into `main`.
- **Gate C** — `google/gemma-4-31b-it` BF16 weights (~60 GB) downloaded to HF cache + dry-run sanity passes (`uv run python scripts/train_sft.py --config configs/train-sft-stage2-gemma4-31b.env --dry-run` yields 12,244 train / 1,362 eval records with long-label markers).

If any gate fails by 2026-05-30, **SFT slips to documented future work** and §2.8 collapses back to "see §1.3.B for status." The locked headline (unified 72.24) is already a strong A submission and does not depend on SFT landing.

**Calibrated execution path (5090, in order):**

| Step | Effort | Target date | Outcome |
|---|---|---|---|
| Review and merge PR #90 (Ulises's dataset loaders + HF docs — independent) | ~30 min human | 2026-05-27 AM | Clean main; loaders available |
| Review and merge PR #79 (SFT pipeline design); cherry-pick or land Ulises's uncommitted `low_cpu_mem_usage=True` + dry-run `bf16=False` fixes in `scripts/train_sft.py`, and the `download-gemma4-31b` Makefile target | ~1 h human | 2026-05-27 AM | Gate B satisfied |
| `make download-gemma4-31b` (~60 GB to `~/.cache/huggingface/hub/`); then dry-run sanity on `train-sft-stage2-gemma4-31b.env` | ~30 min wall + ~5 min human | 2026-05-27 midday | Gate C satisfied |
| ✅ **Stage 2b DONE (2026-05-28 04:44 PDT)** — QLoRA NF4 r=16 on `socrat-zh,socrat-en`, 3 epochs, lr=5e-5, batch 1×16=16. **seq forced 1280→1024** on 5090 (VRAM ceiling in CE; see `feedback_sft_resume_oom_fragmentation.md` and `EXPERIMENT_LOG.md` 2026-05-28). Required env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_SEQ_LEN=1024`. | 10h49m GPU (8 launches; 3 OOM modes resolved) | 2026-05-28 AM | Final adapter at `outputs/sft-stage2-gemma4-31b/final/` (468 MB safetensors, 122 M trainable params). Final train_loss 0.4359, mean_token_accuracy 0.8555. |
| Merge LoRA adapter → GGUF **Q5_K_M** with **KELE-tagged filename** (Q5_K_XL was Unsloth Dynamic; llama-quantize natively only ships Q5_K_S/M — Q5_K_M chosen as standard equivalent at ~5.33 bpw, +0.057 ppl on Llama-3-8B). Three-step pipeline: (1) `peft.merge_and_unload()` via `scripts/merge_lora_gemma4_sft.py` → `outputs/sft-stage2-gemma4-31b/merged/` (HF checkpoint, project-local for safety); (2) `scripts/convert_gemma4_sft_to_gguf.sh` runs llama.cpp `convert_hf_to_gguf.py` → f16 intermediate → `llama-quantize` → `outputs/sft-stage2-gemma4-31b/gemma-4-31B-kele-socratic-sft-Q5_K_M.gguf` (verify locally first); (3) once verified, copy to `~/Documents/models/weights/gemma-4-31B-kele-socratic-sft-Q5_K_M.gguf`. **Do NOT overwrite any existing `gemma-4-31b-it-*.gguf` in the shared models dir** — the KELE filename must coexist with base-model weights used by other projects. | ~1 h human + GPU | 2026-05-28 AM | Project-local artifact + clearly-named shared copy. Drop-in compatible with `serve_gemma4_31b_q5.sh` via `GEMMA4_31B_WEIGHT_FILE=gemma-4-31B-kele-socratic-sft-Q5_K_M.gguf` override. |
| Eval — `bash scripts/eval_bert_gemma_fewshot10_full.sh` with the SFT weights, on **n=681 canonical first** (matches locked-headline n; required for leaderboard comparability), **then synthetic n=75 separately** (`data/synthetic_extension/socrat_synthetic_en_75.json`). Two distinct runs; do not concatenate. | ~3.5 h GPU + ~$2 judge (n=681 ≈ 3h + $1.50; n=75 ≈ 30 min + $0.30) | 2026-05-28 PM / 2026-05-29 AM | Paper-grade numbers on `unified` for both cells. The n=681 cell goes into the master ranked list; the n=75 cell is the contamination-control probe. |
| Aggregate via `scripts/backtest_stage_balanced.py`; write paper paragraph (~150 words) + table rows in Tables 6 and 14. Decompose: master-ranking position from n=681; contamination signature from n=75 vs n=681 delta. | ~2 h human | 2026-05-29 | SFT row locked into `acl_latex.tex` |

**Why no Stage 2a (breadth) warm-start.** Original plan had `socrateach-multi + socrateach-single` as a 2a step before 2b. With time compressed, jump straight to 2b on `socrat-zh + socrat-en` — this matches the original KELE paper's training recipe and avoids spending ~3 GPU-h on a warm-start whose marginal value is unproven. The Stage 2a configs stay in repo as documented future work.

**Locked decisions (carry forward from PR #79 — do not relitigate):**

1. **SFT format = Pattern A + long inference-matching labels** (`苏格拉底教学顾问评估结果:` / `苏格拉底教学顾问建议的操作:`). Chinese markers in both `socrat-zh` and `socrat-en`.
2. **`socrat-synthetic` is eval-only.** Never in `TRAIN_SOURCES`.
3. **DPO Sources 1 & 2 inert** until Stage 2b checkpoint exists. Source 3 is functional.
4. **`assistant_only_loss=True`** in `SFTConfig` — the format fix guarantees the assistant turn is clean teacher output.

**Explicitly out of scope for Jun 4:**

- ❌ **Stage 1 general SFT** — Stage 2b warm-starts from base `gemma-4-31b-it`, not from a Stage 1 checkpoint. Documented future work.
- ❌ **Stage 3 DPO** — Source 1 unblocks once Stage 2b checkpoint exists; decide on 2026-05-29 based on remaining time. Sources 1 & 2 require additional plumbing. Default-stay-as-future-work.
- ❌ **Qwen3.6-27B SFT** — DROPPED for Jun 4. Configs preserved. Future work.

**Outcome interpretation matrix.** All four outcomes are publishable:

| Outcome on `unified` (test, n=681) | Interpretation | Paper framing |
|---|---|---|
| Fine-tuned Gemma **wins** vs `qwen3.5 × Gemma-31B · n=681` (72.24, ≥+1 pt) | Pareto upgrade on the locked headline | Promote SFT as the **new locked headline**; original headline becomes the prompt-engineering baseline |
| Fine-tuned Gemma **ties** (±1 pt) | Consultant axis dominates teacher axis | "Consultant routing — not teacher capacity — is the binding constraint" — strengthens the BERT-consultant thesis |
| Fine-tuned Gemma **wins on test but loses on synthetic n=75** | Contamination signature on our own training data | Powerful confirmation of `docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md` thesis applied to *our own* training, not just STL |
| Fine-tuned Gemma **loses both** | Either undertraining or data-format issue | Document honestly in §Limitations; locked headline stands at 72.24 |

**Risks specific to the 5090 path:**

| Risk | Mitigation |
|---|---|
| GPU contention with concurrent eval/serve runs | Sequence explicitly; `nvidia-smi` check before each phase; `pkill llama-server` if leftover |
| `bitsandbytes 0.49.2` NF4 + Gemma 4 31B (newer arch) edge case | Standard NVIDIA path is well-trodden. Fallback: `scripts/prequant_gemma4.py` for explicit pre-quantisation. |
| LoRA-merged GGUF gives different numerics than transformers+PEFT eval | Sanity eval: round-trip the **base** Gemma through GGUF first; must match the locked-headline numbers within sampling noise. If not, eval via transformers+PEFT directly (Option B from original handoff). |
| `transformers 5.9.0` newer than PR #79 was developed against | Dry-run first; pin to PR #79's working versions in a fresh venv if API drift |
| SFT result is uninspiring AND we already spent the time | All four outcomes in the matrix above are publishable. No wasted-effort scenario. |

### 2.9 (STRETCH) — Prompt-engineering tournament

`docs/PROMPT_ENGINEERING_PLAN.md` — 10-utilization tournament on the locked baseline. Top-3 composition has already landed (the Gemma + top-3 cell that tops the unified ranking at n=50). Remaining: full n=681 promotion of the top-3 stack on Gemma. Already partially captured by frontier-teacher cells; the locked-headline-equivalent open-weight cell is in `TODO #14`.

---

## 3. Known risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **OUTLINE.md (20-slide, 20-min) overruns the actual 10-min slot** | **High** if not compressed | §2.1 — lock the 8-slide compression first, then build. The hard cap is 10 min — *"you won't be allowed to continue after."* |
| 10 min of Q&A is half the slot and unmoderated | **High** | §2.1 last item — pre-write 1–2 lines on each anticipated question (contamination, BERT consultant rationale, unified metric weighting, classroom deployment, ethics). Both speakers field Q&A. |
| Paper exceeds 6-page ACL limit after compile | **High** — 770 LaTeX lines now | §2.2 — aggressive cuts (tournament table, Gemma-pivot section, per-stage redundancy). Verify appendix policy with TA before assuming overflow can live there. |
| Demo recording slips past Jun 4 | Medium | Record in the same hour the slide deck's data figures are produced — they share the same `kele test` invocation. Submit Drive/YouTube link if HF Space build is not viable. |
| Locked headline's pre-fix BERT artifact challenged in Q&A | Low | Documented in §Limitations of the paper and the Anticipated Q&A in OUTLINE.md. Post-fix `bert-fixed × Gemma · n=50` lands at unified 67.65 — ~1 pt below locked headline, within sampling noise. |
| Q&A challenges SocratTeachLLM contamination claim | Low | `docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md` carries the full case; backed by the synthetic-test-set collapse (stage_bal 63.4 → 32.9) and the cross-lingual translation experiment. |
| Q&A challenges "where is your fine-tuned teacher?" | Medium | Two answers depending on §2.8 outcome. If SFT lands by Jun 4: "Stage 2a→2b Qwen3.6-27B QLoRA — here are the numbers on test + synthetic." If it slips: "Consultant-side fine-tuning shipped (5 checkpoints); teacher-side designed and partially trained (synthetic baseline complete, Stage 2 SFT pipeline in `feat/stage2-sft-pipeline-design` branch). The decision to *not* ship a full SocratTeachLLM-replacement was deliberate after contamination evidence landed." |
| **Teacher SFT runs but underperforms BERT+Gemma+10shot baseline** | Medium | Already mitigated by the outcome-interpretation matrix in §2.8 — every outcome (win / tie / synthetic-divergence / lose-both) has a publishable framing. Worst case (lose-both) goes into §Limitations; locked headline stands. |
| **Teacher SFT slips past Jun 1 paper-trim deadline** | Medium | Hard gate in §2.8 — if Gate A or Gate B fails by 2026-05-30, SFT slips to documented future work. No new paper text required in that path; paper-as-is is already complete. |
| HF artifacts pulled / link rot before grading | Low | All four datasets and STL mirror live; backup via `master_leaderboard.md` and `results/` raw artifacts checked into the repo |
| GPU unavailable during final week | Medium | All headline numbers are locked and in `results/`. Remaining GPU work (§2.6–§2.8) is stretch — no replication needed for submission. |

---

## 4. One-glance checklist — split by deadline

```
═══════════════ MAY 26 (in-class presentation: 10 min talk + 10 min Q&A) ═══════════════

PRESENTATION
  [x] OUTLINE.md drafted (but scoped to 20 min — must compress)
  [ ] 8-slide compression locked (Ulises 4 slides, Max 4 slides)   ← §2.1 BLOCKING
  [ ] Slide deck built                                              ← §2.1 BLOCKING
  [ ] 3 must-have figures produced (KELE diagram, inversion, top-5) ← §2.1 BLOCKING
  [ ] Timing rehearsal ≤9:45                                        ← §2.1 BLOCKING
  [ ] Q&A defensive material prepped (1-2 lines/question)          ← §2.1 BLOCKING
  [ ] PDF backup export

═══════════════ JUNE 4 (final submission: paper + code + HF + poster + demo) ═════════

TEACHER SFT  (Ulises's individual contribution slice — §2.8 BLOCKING-GATED)
  Gate A — May 26 deck rehearsed                   (must close by 5/27)
  Gate B — Synthetic baseline n=37 → n=75 done    (must close by 5/30)
  [ ] Fix dataset.py SFT format (§0.2)             5/26 post-talk
  [ ] Synthetic baseline n=75 (E1)                 5/27
  [ ] LLM-judge on 3 baselines (E2, ~$5)          5/28
  [ ] Stage 2a (breadth) training                  5/28 evening
  [ ] Stage 2b (structural) training               5/29-30 overnight
  [ ] Eval on test n=400 + synthetic n=75          5/30
  [ ] Paper paragraph + tables row                 6/1

PAPER — class deliverable (6 pages, BLOCKING)
  [x] Long-form draft in acl_latex.tex (12+ pages — source for both targets)
  [x] Intro & Related Work
  [x] Dataset & Methodology
  [x] Evaluation & Results
  [x] Conclusion, Limitations, Ethics
  [ ] Preserve acl_latex.tex unchanged BEFORE trim begins (§2.2b)
  [ ] Create acl_latex_6page.tex                   ← §2.2 BLOCKING
  [ ] Trim to ≤6 body pages                        ← §2.2 BLOCKING
  [ ] (conditional) SFT row added to Tables 6/14   ← gated on §2.8 landing
  [ ] Agentic Reviewer pass on 6-page version
  [ ] Member-contributions paragraph (2 members)
  [ ] Final bib cleanup on 6-page version

CODE  (in repo, already shipped)
  [x] Docstrings, .env.example, tests, README sections
  [ ] Member Contributions section in README        ← §2.5
  [ ] Demo link added under Hugging Face section
  [ ] Verify HF links resolve

POSTER
  [ ] Layout draft (5 panels)                       ← §2.3 BLOCKING
  [ ] Print / submit

DEMO
  [ ] Record 3-min screencast                       ← §2.4 BLOCKING
  [ ] Host (HF Space / YouTube / Drive)
  [ ] Reference link in paper

═══════════════ STRETCH (only if BLOCKING items are clean by Jun 1) ═════════════════
  [ ] Full-length paper submission candidate         ← §2.2b (post-Jun 4)
       [ ] Preserve acl_latex.tex (long-form) intact
       [ ] Post-Jun 4 polish pass
       [ ] Target venue (ACL RR / EMNLP Findings / NAACL 2026)
       [ ] Anonymize + reformat for venue
  [ ] HF state-classifier funnel publish (5 models, §2.6)
  [ ] HF teacher-SFT checkpoint publish              (depends on §2.8 landing)
  [ ] TODO #14 cells 1+2 — n=681 parity (§2.7)
  [ ] Stage 3 DPO + Phase 2 Gemma head-to-head      (out of Jun 4 scope, future work)
  [ ] Prompt-engineering tournament n=681 promotion (§2.9)
```

(HF data artifacts — SocratDataset, SocratDataset-EN, SocratDataset-SYNTHETIC, SocratTeachLLM mirror — are already live and listed in §1.4 / §1.3.)

**Bottom line.** The science is locked: 143 model variants measured, **current locked headline at unified 72.24 (overtaking the best frontier teacher we tested by +2.18 unified pts at canonical n=681; promoted 2026-05-26)** with five fine-tuned consultant checkpoints, four HF datasets live, paper drafted end-to-end, complete experimental campaign in `results/`. **Three near-term sprints stand between now and grading:**

1. **May 26 (in ~24h) — May 26 talk.** Compress the 20-slide OUTLINE.md to an 8-slide / 10-min deck for 5 min × 2 speakers. Build 3 must-have figures, rehearse to ≤9:45, prep Q&A defensive material. No demo, no poster.
2. **May 27 → 30 — Teacher SFT sprint (Ulises-owned, §2.8).** Hard-gated on the talk being clean and the synthetic baseline extending to n=75. If both gates pass: fix `dataset.py`, run Stage 2a → 2b QLoRA on Qwen3.6-27B overnight, eval on test (n=400) + synthetic (n=75), add one paragraph + one row to the paper. If either gate fails: SFT slips to documented future work — the locked headline is already an A submission.
3. **Jun 1 → 4 — Paper trim + poster + demo + README polish.** Preserve the long-form `acl_latex.tex` intact (it's the seed of the §2.2b stretch submission candidate), then derive a separate `acl_latex_6page.tex` for the class deliverable. Agentic Reviewer pass on the 6-page version, build 5-panel poster, record 3-min demo, add Member Contributions to README, verify HF links. Poster + demo + 6-page paper land together on Jun 4.

**Two paper targets:** the 6-page class deliverable is required and graded; the full-length venue-submission candidate (ACL Rolling Review / EMNLP Findings / NAACL 2026) is a post-graduation stretch — preserved from the same long-form draft, polished after Jun 4, anonymized per target venue.

**Fine-tuning honesty disclosure** (likely Q&A topic on May 26): consultant-side fine-tuning is fully shipped (5 trained classifiers, two of them on the headline path; the Qwen3.5-LoRA classifier is the current locked-headline consultant as of 2026-05-26); teacher-side fine-tuning is mid-sprint, decisively gated, and either lands in the Jun 4 paper or remains documented future work. The pivot toward consultant-side fine-tuning was deliberate — contamination evidence (`docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md`) shifted the right target away from "beat SocratTeachLLM on its memorization metric" toward "win on memorization-resistant metrics," which the current locked headline now does **decisively** (unified 72.24 vs. best frontier 70.06, +2.18 lead at canonical n=681).
