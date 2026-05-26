# CSEN 346 — Final Project Status Report

**Project:** Reproducing and Extending KELE — A Multi-Agent Framework for Structured Socratic Teaching with LLMs
**Team:** Cyberdyne — Ulises Chavarria · Maximilian Khan *(2-member team; no 3rd member joined)*
**Course:** CSEN 346, Santa Clara University, Spring 2026
**Status snapshot date:** 2026-05-25
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
| **(4)** Stronger evaluation (ablation + cross-domain) | **✅ Shipped — significantly expanded.** | 142 model variants measured, 37 with full unified score, 8-cell cross-teacher matrix, 4-metric memorization-resistant evaluation panel, bootstrap convergence analysis (n=400 sufficient). |

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
| 🏆 7 | **`bert × Gemma-31B · fewshot10 · n=681`** ← **LOCKED HEADLINE** | 3834 | **68.65** | 55.42 | 8.19 | 36.78 |
| ⚠️ 8 | `qwen3.5 × SocratTeachLLM · fewshot10 · n=50` (contamination-driven) | 288 | 68.21 | 63.40 | 7.30 | 48.07 |
| ⚠️ last (37) | `Claude-Sonnet × SocratTeachLLM · EN · n=50` | 303 | 44.36 | 22.55 | 6.62 | 55.85 |

**Local–frontier parity gap envelope at canonical n=681:** 1.41 pts (legacy locked headline) → 2.25 pts (qwen3.5 × A3B-35B) → 3.35 pts (qwen3.5 × Qwen-27B no-think). The 4-cell `TODO #14` sub-leaderboard has 2/4 cells landed.

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

#### B. Teacher-side fine-tuning — ⏳ PIPELINE READY, NOT YET TRAINED

The original KELE paper's "headline" was `SocratTeachLLM` — a GLM4-9B fine-tuned on SocratDataset. Our equivalent ambition (Slide 19 of the outline; §`sec:nextsteps` of the paper) is to fine-tune our own teacher checkpoint. Current state: **everything is wired except the training run itself.**

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

**Bottom line on fine-tuning.** Consultant side is done in depth — five fine-tuned classifiers, two of which (`v1` and T4-LoRA) are the consultants that produce every locked-headline number in the paper. Teacher side is a designed-but-undeployed extension — listed in the paper as `\S sec:nextsteps` item, in the presentation as Slide 19 "Future Work," and in this report as §2.8 stretch work.

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

**Length:** The current draft is **substantially over the 4–6 page limit** (770 lines of LaTeX, ~12+ pages compiled). Heavy trimming required before final submission.

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

- **142 model variants** measured end-to-end on the SocratDataset 681-dialogue test split (and subsets).
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

### 2.2 (BLOCKING) — Trim the paper to the 4–6 page ACL limit (target: 2026-06-02)

The current `acl_latex.tex` is ~12+ compiled pages. Required cuts to meet the rubric's "**4-6 page research paper (excluding references)**" limit:

- [ ] **Compile current draft and measure actual page count.** Target: ≤6 body pages (refs are excluded from the limit).
- [ ] **Identify aggressive cuts.** Likely candidates: the §4.5 tournament 13-model table can collapse to a 3-row best-of summary; §4.7 (Gemma pivot retraction) can compress to one paragraph; §4.12 (cross-architecture scaling discussion) can move detail to a Limitations footnote; the multiple per-stage tables can fold into one.
- [ ] **Re-run [Agentic Reviewer](https://paperreview.ai/) before submission** (per the project guidelines and `docs/PLAN.md`).
- [ ] **Add a "Member Contributions" appendix** (rubric requirement under the code submission section but typically also in the paper).
- [ ] **Final BibTeX cleanup** — ensure every `\citep` resolves; check `custom.bib` against the cited keys.

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

### 2.7 (STRETCH, not blocking submission) — TODO #14 remaining cells

`docs/EXPERIMENT_LOG.md` (2026-05-25 entry) — 2 of 4 cells landed. Two outstanding apples-to-apples n=681 cells:

- [ ] `bert-fixed × Gemma-31B · fewshot10 · n=681` (~12 GPU-h, ~$15 judge spend) — closes the pre-fix consultant input-format artifact for the locked headline.
- [ ] `qwen3.5 × Gemma-31B · fewshot10 · n=681` (~13 GPU-h, ~$15) — completes the parity sub-leaderboard.

Outcome would tighten the Limitations §"pre-fix consultant input-format artifact" claim and the §"local–frontier parity verified only at n=50 for the post-fix cells" claim. Strictly an evidence upgrade — not gating for the paper or presentation.

### 2.8 (STRETCH, not blocking submission) — Stage 2 SFT pipeline (teacher fine-tuning)

`docs/TRAINING_PLAN.md` — Qwen3.6-27B QLoRA fine-tune on a 3-stage pipeline (general SFT → Socratic SFT → DPO). **Pipeline is fully designed; no training has been launched.**

- [ ] Fix `src/project/dataset.py` SFT format to match BERT-consultant inference format (§0.2 of TRAINING_PLAN).
- [ ] Extend synthetic baseline from n=37 to n=75 (E1) so the post-SFT lift measurement is defensible.
- [ ] Run LLM-judge on the three synthetic baselines (E2, ~$5).
- [ ] Stage 2a → 2b training runs (~6–12 GPU-h on R9700).
- [ ] DPO Stage 3 (~3 GPU-h).
- [ ] Gemma 4 31B head-to-head Phase 2 (mirror config).

**Why stretch:** Listed as "Future Work" in the OUTLINE.md and §`sec:nextsteps` of the paper. Shipping this expands the contribution significantly but does **not** unblock the final-submission package. **Earliest realistic completion = ~2 weeks of GPU time + tooling.** Given the May 26 / Jun 4 deadlines, attempt this only if the BLOCKING items are clean by Jun 1.

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
| Q&A challenges "where is your fine-tuned teacher?" | Medium | Be honest: consultant-side fine-tuning shipped (5 checkpoints); teacher-side fine-tuning is designed and infrastructured but not executed (`docs/TRAINING_PLAN.md`). Pivot was *deliberate* — once contamination evidence landed, beating SocratTeachLLM on its memorization metric stopped being the right target. |
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

PAPER
  [x] Intro & Related Work
  [x] Dataset & Methodology
  [x] Evaluation & Results
  [x] Conclusion, Limitations, Ethics
  [ ] Trim to 4-6 pages                            ← §2.2 BLOCKING
  [ ] Agentic Reviewer pass
  [ ] Member-contributions paragraph (2 members)
  [ ] Final bib cleanup

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
  [ ] HF state-classifier funnel publish (5 models, §2.6)
  [ ] TODO #14 cells 1+2 — n=681 parity (§2.7)
  [ ] Stage 2 SFT teacher fine-tune (§2.8)
  [ ] Prompt-engineering tournament n=681 promotion (§2.9)
```

(HF data artifacts — SocratDataset, SocratDataset-EN, SocratDataset-SYNTHETIC, SocratTeachLLM mirror — are already live and listed in §1.4 / §1.3.)

**Bottom line.** The science is locked: 142 model variants measured, locked headline at unified 68.65 with five fine-tuned consultant checkpoints, four HF datasets live, paper drafted end-to-end, complete experimental campaign in `results/`. **Two near-term packaging sprints stand between now and grading:**

1. **May 26 (in ~24h):** *compress* the 20-slide OUTLINE.md to an 8-slide / 10-min deck for 5 min × 2 speakers, build the 3 must-have figures, rehearse, and prep Q&A defensive material. No demo, no poster — just the talk and the room.
2. **Jun 4:** trim the paper to 4–6 pages, run it through Agentic Reviewer, record the 3-min demo, build the poster, add Member Contributions to the README, verify HF links. The poster + demo + final paper land together.

**Fine-tuning honesty disclosure** (likely Q&A topic): consultant-side fine-tuning is fully shipped (5 trained classifiers, two of them on the headline path); teacher-side fine-tuning is plumbed but not run (TRAINING_PLAN.md + train_sft.py + QLoRA config + synthetic baseline all exist; no SFT/DPO checkpoints have been produced). The pivot was deliberate — once contamination evidence landed (`docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md`), beating SocratTeachLLM on its memorization metric stopped being the right target, and prompt-engineering plus the BERT consultant closed the gap on the metrics that actually matter (unified 68.65 vs. frontier 70.06).
