# Experiment Tiers — open work for the KELE project

Reference doc for prioritizing additional experiments. Generated 2026-05-15 after the 8h autonomous run completed and PR #52 was opened. Branch this doc lives on: `mk/level-up-experiments`.

Update this list as items move from `pending` → `in_progress` → `done`. Move done items to the bottom of their tier with a `[✓ commit-hash, date]` annotation.

---

## Tier S — Paper-shipping experiments (the three Next Steps items)

| # | Experiment | Time | Risk | Expected impact | Status |
|---|---|---|---|---|---|
| S.1 | **Gemma 4 31B full run (n=681)** | ~30 h | Low | Verifies/refutes the 46.71% headline projection. Single most important open experiment. | pending |
| S.2 | **LoRA fine-tune on A3B teacher** (or SocratTeachLLM 9B) on `(history, teacher_response)` train pairs | ~30 min training + 1 h eval (after pipeline build, ~1 day) | Low (well-understood pipeline) | Remaining surface-form lever after prompt-eng's bounded recovery. Could close another 5–8 pts of the ROUGE-1 gap. | pending |
| S.3 | **Hierarchical Chinese-BERT state classifier** (5-way stage head + within-stage state head; trained on 42K labeled turns) | ~1 h training + 2 h eval (~half-day for code + integration) | Medium (integration with kele.py) | Targets stage-c specifically (22-way classification). Could lift stage-c from 17% to 50%+. Headline-changing for the consultant component. | pending |

## Tier A — Strong incremental experiments (under 4 hours each, GPU)

| # | Experiment | Time | Why it matters |
|---|---|---|---|
| A.4 | **Many-shot prompt-eng sweep** (5-shot, 7-shot, 10-shot) at smoke/mini/n=50 | ~3 h | Current 3-shot delivered +1.5 R-1 mean. Test if more examples compound or saturate. |
| A.5 | **Stage-aware exemplars** (one exemplar per SocRule stage a/b/c/d/e) | ~3 h | Static 3-shot covers only b/c/d. Adding stage-a and stage-e exemplars may unlock missing-stage gains. |
| A.6 | **Negative-example prompt-eng** (show bad teacher responses + corrections) | ~2 h | Underexplored direction; contrastive prompting works in PEFT, may work in prompt. |
| A.7 | **Anti-paraphrastic instructions** (direct instruction "match GT phrasing closely") | ~1 h | Orthogonal to exemplars, may compose. |
| A.8 | **27B Q4 vs Q5 comparison** at smoke + mini | ~1.5 h | Tournament showed Q4 (30.67) > Q5 (28.62) at no-think; no Max-side Q4 data exists. May reveal a Q4 sweet spot. |
| A.9 | **Mistral Small 24B in think mode** at smoke + mini | ~1.5 h | Tournament: 21.85 no-think. Test if its think mode unlocks Qwen-family-style gains. |
| A.10 | **Qwen3 14B Q4 in think mode** at smoke + mini | ~30 min | Faster than 27B but smaller; potential cost-efficient surprise. |
| A.11 | **Phi-4 14B or DeepSeek R1 14B in think mode** | ~1 h each | DeepSeek-R1 is reasoning-distilled, so think mode might dramatically shift it from the tournament's last place. |
| A.12 | **Sampling parameter sweep** on A3B fusion-think (temperature 0/0.3/0.7, top_p tuning) | ~2 h | Currently using defaults. Tuning may improve state acc and/or ROUGE. |
| A.13 | **Verify A4B's "always-on reasoning" claim** by running A4B with `--nothink` kwarg | 30 min | Tests whether the chat-template kwarg can actually disable Gemma's reasoning emission. |

## Tier B — Observability & visualization (NO GPU NEEDED)

| # | Item | Time |
|---|---|---|
| B.14 | 5×5 stage confusion matrix on locked A3B full run | 30 min |
| B.15 | 34×34 state confusion matrix | 1 h |
| B.16 | Per-dialogue R-1 distribution (box plots, identify outliers) | 30 min |
| B.17 | Turn-index accuracy curve (does accuracy degrade for late turns?) | 30 min |
| B.18 | BLEU-4 vs dialogue length scatter | 30 min |
| B.19 | State frequency bar chart for all 34 states | 30 min |
| B.20 | Radar chart comparing systems on R-1/R-2/R-L/B-4 (paper-quality figure) | 45 min |
| B.21 | Trajectory error analysis: which dialogues completely collapsed? | 1 h |
| B.22 | Per-turn schema-fallback investigation (38 fallbacks in A3B full): root cause + recoverability | 1 h |
| B.23 | Dialogue length distribution histogram (`dialogueRound` 5–12) | 30 min |
| B.24 | Stage turn-count bar chart (motivates why stage c is hardest) | 30 min |
| B.25 | Question-type breakdown (multiple choice vs true/false) | 30 min |

## Tier C — Methodological / dataset-level

| # | Item | Time | Notes |
|---|---|---|---|
| C.26 | **SocratDataset-EN translation verification** | 1 h | The only pre-existing `docs/TODO.md` item. Sample audit for fidelity. |
| C.27 | **A3B + 3-shot at full n=681** | ~16 h | Definitive ROUGE-recovery measurement at locked-headline scale. |
| C.28 | **Counterfactual dialogue augmentation** for stage-c rare misconceptions | ~1 week | IMPROVEMENT_PLAN.md #7. Generate variants, then LoRA. |
| C.29 | **Persistent student memory** (JSON-tracked student state across turns) | ~4 days | IMPROVEMENT_PLAN.md #3. Composable with everything. |
| C.30 | **Retrieval-augmented teacher** (BAAI/bge-large-zh-v1.5 index over train dialogues) | ~1 week | IMPROVEMENT_PLAN.md #1. Per-dialogue dynamic exemplars. |

## Tier D — Research bets (probably post-paper)

| # | Item | Time | IMPROVEMENT_PLAN ref |
|---|---|---|---|
| D.31 | Multi-Turn RL with outcome reward | ~3–4 weeks | #8 |
| D.32 | Student simulator self-play | ~3 weeks | #9 |
| D.33 | Socratic moves as tool calls (architectural rewrite) | ~3–4 weeks | #10 |
| D.34 | Process Reward Model + Best-of-N sampling | ~2 weeks | #5 |
| D.35 | Semantic reward fine-tuning (BLEU → BERTScore/embedding loss) | ~1 week | #6 |
| D.36 | Consultant–teacher fusion (single end-to-end fine-tune) | ~2 weeks | #4 (already partially done via the fusion architecture, but not fine-tuned) |

---

## Snapshot of completed work (as of 2026-05-15 09:53 PDT)

**On `mk/8h-autonomous-extensions` (PR #52):**
- Paper fix: Gemma row corrected in Table 7
- 11 new evaluation datapoints (A4B smoke + mini, A3B n=50 think/no-think, A3B + 3-shot smoke/mini/n=50, 27B Q5 mini think/no-think, Qwopus smoke/mini)
- Plumbing: eval/serve scripts for A4B, Qwopus, A3B think, 27B think
- Opt-in 3-shot teacher exemplars in `socratic_teaching_unified.py`
- Paper updates: Table 7 expanded, new §4.7.2 ROUGE recovery subsection, abstract/conclusion/limitations/next-steps revised

**Current branch: `mk/level-up-experiments`** (this doc lives here)
