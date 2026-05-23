# Unified ranking metric for KELE Socratic-teaching configurations

**Author:** Max + Claude Opus 4.7 (1M ctx)
**Date introduced:** 2026-05-23
**Status:** Active. Replaces ad-hoc per-metric leaderboards as the single-number ranking for paper headlines and overnight promote-or-defer decisions on judged cells.

## TL;DR

```
unified_score = 0.5 × stage_balanced + 0.5 × (judge_overall × 10)
```

Both inputs on [0, 100]; output on [0, 100]. Higher is better. Only defined for cells that have both a state-acc evaluation (`metrics_summary.json`) **and** an LLM-judge rubric run (`judge_summary.json`).

A second variant — `unified_ped` — substitutes `pedagogically_weighted` for `stage_balanced` and is reported as a defensible alternative when the pedagogical-weights argument is acceptable.

## Why we need a unified metric

By 2026-05-23 the project tracks at least eight per-cell metrics: macro state accuracy, stage-balanced macro, pedagogically-weighted macro, frequency-inverse weighted macro, LLM-judge overall (with four sub-axes), and four surface-form metrics (R-1, R-2, R-L, BLEU-4). Six are in active use; two (R-2, R-L) are reported only as memorization-signature triangulation.

Reporting six numbers per cell is right for the methodology section but **wrong for the headline.** The paper needs a single defensible rank per configuration so:

- Promote-or-defer decisions don't require argument over which metric to prefer.
- The cross-teacher matrix has one ordering, not five.
- Future runs can be ranked against the locked headline without recomputing argumentation.

The challenge: each of our metrics individually has known failure modes. Picking any one as "the" ranking would carry its weakness into the headline.

## Why these two metrics specifically

**`stage_balanced`** (Proposal 7 in `BENCHMARK_CRITIQUE_AND_PROPOSAL.md`) measures **per-turn pedagogical correctness** — does the consultant correctly route the dialogue through the SocRule stages — with each stage weighted equally regardless of its frequency in the test split. This corrects the published `macro state acc`'s structural under-weighting of rare-but-important stages (especially closure, which is ~10% of turns but the moment that pedagogically anchors the learning).

**`judge_overall`** is the Claude Sonnet 4.6 rubric score (averaged across Socratic validity, advancement, age-appropriateness, and question form). It measures **per-turn pedagogical quality** — given the teacher's response, would a frontier LLM evaluator score it as good Socratic teaching? It is memorization-resistant by construction: the rubric checks teaching moves and form, not phrasing match to ground truth.

The two are *orthogonal*. `stage_balanced` answers "did the system route to the right stage?" and `judge` answers "given the response, is it good?". A model can be high on one and low on the other:

- High `stage_bal` + low `judge` = correctly routed but poorly-written responses
- Low `stage_bal` + high `judge` = beautifully written but stage-misrouted (the SocratTeachLLM failure mode)

Both are needed to claim a configuration is a good Socratic teacher.

## Why 50/50 weighting (and not something else)

Three considerations argue for equal weight:

1. **No principled prior on which axis dominates.** Both correctness and quality are necessary; neither is sufficient. The paper's central methodological argument is that any single-axis ranking is misleading.

2. **Memorization resistance is roughly equal.** `stage_bal` resists memorization via per-stage breakdown (it's hard to memorize stage-by-stage SocRule routing without learning the underlying pedagogy); `judge` resists via rubric-based evaluation (it's hard to game a structured rubric without actually doing the teaching well). Neither is dramatically more robust than the other.

3. **Defensibility.** Equal weight requires no justification beyond the two metrics themselves; any other weighting requires defending why one matters more, which is a battle the paper does not need to fight in the headline.

If the reviewers later argue we should weight `judge` more heavily because of its rubric-based design, the alternative weightings are trivial to recompute from the per-cell metrics already in the repo.

## What gets REJECTED from the unified score

- **`macro` state acc.** Frequency-weighted, hides closure dominance (Proposal 7). Still reported as the "test-distribution-matched" secondary number.
- **R-1, R-2, R-L, BLEU-4.** Surface-form metrics shown inversely correlated with pedagogical capability on this benchmark (`BENCHMARK_CRITIQUE_AND_PROPOSAL.md`'s central claim). Including any of these in the unified score would re-introduce the memorization signature we're trying to escape.
- **`pedagogically_weighted` (in primary unified).** Defensible but opinionated — requires a citation-backed argument about relative stage importance. We report it as `unified_ped` for cases where that argument is acceptable; not the primary headline.
- **`frequency_inverse`.** Mathematically clean but over-corrects — gives stage e ~3× the weight of stage c despite no pedagogical argument for that ratio.

## How rankings differ across metrics (cross-teacher 8-cell complete matrix, 2026-05-23 PM)

All 8 cells now judged:

| sb# | u# | Cell | macro | stage_bal | judge | **unified** |
|:-:|:-:|---|---:|---:|---:|---:|
| 1 | 4 | T4 × Qwen27B-think | 53.19 | 58.68 | 7.51 | **66.89** |
| 2 | 3 | T4 × A3B 35B | 54.86 | 58.62 | 7.52 | **66.91** |
| 3 | 5 | bge × Qwen27B-think | 49.08 | 57.15 | 7.41 | **65.65** |
| 4 | 🥇 1 | **T4 × Gemma 31B** | 51.58 | 56.13 | **8.18** | **68.94** |
| 5 | 6 | T4 × Qwen27B-no-think | 51.89 | 55.45 | 7.56 | **65.54** |
| 6 | 🥈 2 | **bge × Gemma 31B** | 45.94 | 52.73 | **8.26** | **67.65** |
| 7 | 7 | bge × Qwen27B-no-think | 46.85 | 52.62 | 7.59 | **64.25** |
| 8 | 8 | bge × A3B 35B | 45.36 | 52.48 | 7.49 | **63.70** |

**The headline shifts.** Under stage_bal alone, T4 × Qwen 27B-think narrowly wins (58.68); under unified, **T4 × Gemma 31B wins decisively** (68.94 vs Qwen-think's 66.89 — a 2.05-point margin, well outside n=50 noise on the judge axis). Even bge × Gemma 31B (67.65) beats both Qwen-think configurations on unified.

The driver is judge score: Gemma sits at **8.18–8.26** across both consultants while Qwen sits at **7.41–7.59**. That ~0.65-point judge gap (×10 = 6.5 unified-pp) more than swallows the stage_bal advantage Qwen had on closure.

**Three honest readings of the same data:**

1. **"Qwen 27B is the best stage-routing teacher"** — true under stage_bal, especially on closure (+12 pp stage e). This is the closure-dominance finding from `EXPERIMENT_LOG.md 2026-05-23`.

2. **"Gemma 31B is the best perceived-quality teacher"** — true under judge. Sonnet rates Gemma's question form and clarity higher.

3. **"Gemma 31B wins the unified headline"** — combining both axes, the quality margin overwhelms the routing margin. **This is the paper's recommended headline ranking.**

The think-vs-no-think tension we flagged is now quantified: T4 × Qwen 27B-think (66.89) and T4 × Qwen 27B-no-think (65.54) differ by 1.35 unified points — bigger than I estimated yesterday but still small relative to the Qwen-vs-Gemma gap (~2 points).

## How to read a unified score

For interpretation reference:

- **Locked headline (BERT + Gemma 31B + 10-shot, n=681):** stage_bal 55.42, no judge yet — would need to be judged to land on the unified scale. Estimated at ~62-64 unified based on current judged-cell distribution.
- **Frontier ceiling (BERT + Claude Opus + top3, n=681):** stage_bal 58.63, judge 8.01 — unified ≈ **69.46**.
- **SocratTeachLLM at its peak surface form (R-1 = 55.85 on EN translation):** stage_bal 22.55, judge would be low (closure is broken at 16-25%) — unified would be in the 30s. This is exactly the desired property: the worst-pedagogy / best-surface model gets ranked at the bottom.

A unified score above **65** is, as of 2026-05-23, a serious paper headline candidate. Above **70** would shift the locked headline. Below **55** is a memorization-dominated configuration that the rubric correctly punishes.

## Implementation

`scripts/backtest_stage_balanced.py` computes both `unified` and `unified_ped` columns automatically. The unified column is `None` for cells that lack `judge_summary.json` — these cells appear in the broader stage-balanced leaderboard but NOT in the master ranked list (we don't fake-rank an unjudged cell).

To bring a cell onto the unified leaderboard:

```bash
.venv/bin/python scripts/llm_judge_eval.py results/<cell> --model claude-sonnet-4-6 --workers 10
.venv/bin/python scripts/backtest_stage_balanced.py
```

About 5 minutes and ~$0.10 per cell. The output `backtest_stage_balanced_latest.md` symlink in `results/_orchestrator_logs/` always points at the freshest run.

## What changes when we apply unified going forward

1. **Promote-or-defer decisions** (`scripts/promote_if_winner.py`) currently use `composite = state + 0.5 × R-1`. **TODO:** swap to `unified = 0.5 × stage_bal + 0.5 × (judge × 10)` once we are willing to gate live runs on judge availability. Until then, promote uses the composite at decision time and the unified is reported post-hoc.

2. **Paper §4 (Results)** swaps the headline table from macro-only to unified-primary with the full per-metric breakdown beneath.

3. **Paper §5 (Limitations)** picks up a paragraph explaining the unified formula and why two-axis aggregation is the honest move on a benchmark with known per-axis pathologies.

4. **Future runs** report unified as their primary number whenever the run is judged; otherwise report stage_bal as the headline and flag that judge is pending.

## What this does NOT replace

- The full per-metric table (macro, stage_bal, pedagogical, freq_inv, judge axes, R-1/R-2/BLEU-4, per-stage breakdown) stays as the methodology-section artifact. Unified is the single-number ranking; the full table is the supporting evidence.
- Per-stage analysis (especially stage e dominance findings) is reported separately because no scalar can capture stage-specific narratives.
- The benchmark critique (`BENCHMARK_CRITIQUE_AND_PROPOSAL.md`) stands on its own — unified is the *response* to the critique, not a replacement for it.

## Open questions for the paper-writing pass

- Should the unified threshold for "headline candidate" be locked at >65 or >70? Currently soft.
- Should we report `unified` and `unified_ped` side-by-side, or pick one as primary and relegate the other to an appendix?
- Should the LLM-judge model be Claude Sonnet 4.6 only (current), or expand to a multi-judge panel (Sonnet + Opus + GPT-4o) for the published headline? The latter would harden the metric against single-model bias but adds ~3× cost.
