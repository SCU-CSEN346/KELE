# Unified ranking metric for KELE Socratic-teaching configurations

**Author:** Max + Claude Opus 4.7 (1M ctx)
**Date introduced:** 2026-05-23
**Status:** Active. Replaces ad-hoc per-metric leaderboards as the single-number ranking for paper headlines and overnight promote-or-defer decisions on judged cells.
**Companion:** Cell labels in this doc use the format defined in [`docs/NAMING_CONVENTION.md`](NAMING_CONVENTION.md) — `<consultant> × <teacher> [· <variant>...] · n=<N>`.

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

All 8 cells now judged. Cell labels follow `docs/NAMING_CONVENTION.md`:

| sb# | u# | Cell | macro | stage_bal | judge | **unified** |
|:-:|:-:|---|---:|---:|---:|---:|
| 1 | 4 | `qwen3.5 × Qwen-27B · think · fewshot10 · n=50` | 53.19 | 58.68 | 7.51 | **66.89** |
| 2 | 3 | `qwen3.5 × A3B-35B · fewshot10 · n=50` | 54.86 | 58.62 | 7.52 | **66.91** |
| 3 | 5 | `bert-fixed × Qwen-27B · think · fewshot10 · n=50` | 49.08 | 57.15 | 7.41 | **65.65** |
| 4 | 🥇 1 | **`qwen3.5 × Gemma-31B · fewshot10 · n=50`** | 51.58 | 56.13 | **8.18** | **68.94** |
| 5 | 6 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=50` | 51.89 | 55.45 | 7.56 | **65.54** |
| 6 | 🥈 2 | **`bert-fixed × Gemma-31B · fewshot10 · n=50`** | 45.94 | 52.73 | **8.26** | **67.65** |
| 7 | 7 | `bert-fixed × Qwen-27B · no-think · fewshot10 · n=50` | 46.85 | 52.62 | 7.59 | **64.25** |
| 8 | 8 | `bert-fixed × A3B-35B · fewshot10 · n=50` | 45.36 | 52.48 | 7.49 | **63.70** |

**The headline shifts.** Under stage_bal alone, T4 × Qwen 27B-think narrowly wins (58.68); under unified, **qwen3.5 × Gemma 31B wins decisively** (68.94 vs Qwen-think's 66.89 — a 2.05-point margin, well outside n=50 noise on the judge axis). Even bert-fixed × Gemma 31B (67.65) beats both Qwen-think configurations on unified.

The driver is judge score: Gemma sits at **8.18–8.26** across both consultants while Qwen sits at **7.41–7.59**. That ~0.65-point judge gap (×10 = 6.5 unified-pp) more than swallows the stage_bal advantage Qwen had on closure.

**Three honest readings of the same data:**

1. **"Qwen 27B is the best stage-routing teacher"** — true under stage_bal, especially on closure (+12 pp stage e). This is the closure-dominance finding from `EXPERIMENT_LOG.md 2026-05-23`.

2. **"Gemma 31B is the best perceived-quality teacher"** — true under judge. Sonnet rates Gemma's question form and clarity higher.

3. **"Gemma 31B wins the unified headline"** — combining both axes, the quality margin overwhelms the routing margin. **This is the paper's recommended headline ranking.**

The think-vs-no-think tension we flagged is now quantified: T4 × Qwen 27B-think (66.89) and T4 × Qwen 27B-no-think (65.54) differ by 1.35 unified points — bigger than I estimated yesterday but still small relative to the Qwen-vs-Gemma gap (~2 points).

## How to read a unified score

For interpretation reference:

- **Locked headline (BERT + Gemma 31B + 10-shot, n=681):** stage_bal 55.42, judge 8.19 — unified **68.65** (#7 master, 🏆). Judged after this doc was first written; replaces the prior ~62-64 estimate.
- **Frontier ceiling (BERT + Claude Sonnet + top3, n=681):** stage_bal 58.17, judge 8.19 — unified **70.06** (#2 master, 🥈). The Opus n=681 cell is at unified 69.37 (#4).
- **First canonical-scale local parity cell (BERT + Qwen 35B-A3B + 10-shot, n=681):** stage_bal 60.02, judge 7.56 — unified **67.81** (#11 master, TODO #14 cell #3, landed 2026-05-25). Locks the canonical-scale parity gap at 2.25 unified pts behind frontier (1.41 pts via the legacy locked-headline anchor).
- **SocratTeachLLM at its peak surface form (R-1 = 55.85 on EN translation):** stage_bal 22.55, judge would be low (closure is broken at 16-25%) — unified would be in the 30s. This is exactly the desired property: the worst-pedagogy / best-surface model gets ranked at the bottom.

A unified score above **65** is a serious paper headline candidate. Above **70** would shift the locked headline. Below **55** is a memorization-dominated configuration that the rubric correctly punishes.

## Implementation

`scripts/backtest_stage_balanced.py` computes both `unified` and `unified_ped` columns automatically. The unified column is `None` for cells that lack `judge_summary.json` — these cells appear in the broader stage-balanced leaderboard but NOT in the master ranked list (we don't fake-rank an unjudged cell).

To bring a cell onto the unified leaderboard:

```bash
.venv/bin/python scripts/llm_judge_eval.py results/<cell> --model claude-sonnet-4-6 --workers 10
.venv/bin/python scripts/backtest_stage_balanced.py
```

About 5 minutes and ~$0.10 per cell. The output `backtest_stage_balanced_latest.md` symlink in `results/_orchestrator_logs/` always points at the freshest run.

## What changes when we apply unified going forward

1. **Paper headline** uses `unified` as the primary single-number ranking (landed 2026-05-23). Paper §`sec:unified-ranking-parity` defines the metric; Limitations flags the pre-fix bert artifact and the n=50-vs-n=681 verification asymmetry. Master ranked list in `results/_orchestrator_logs/backtest_stage_balanced_latest.md`.

2. **Promote-or-defer decisions** (`scripts/promote_if_winner.py`) still use the older `composite = state + 0.5 × R-1` because live-chain decisions require the metric at-decision-time and judge runs are post-hoc. **Open TODO:** wire the chain to run LLM-judge in-line before promote so the chain can decide on unified directly. ~30 lines plus a wait-for-judge step in the orchestrator. Not blocking the paper.

3. **Future runs** report unified as their primary number whenever the run is judged. Cells without judge appear in the stage_bal leaderboard but NOT in the master ranked list (we don't fake-rank an unjudged cell). To bring a new cell onto the master list, run `scripts/llm_judge_eval.py results/<cell> --model claude-sonnet-4-6 --workers 10` (~5 min, ~$0.10) and re-run the backtest.

4. **Cross-references in other docs.** Per the audit on 2026-05-23, `docs/EXPERIMENT_LOG.md` (2026-05-23 entry), `docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md` (Proposal 8), and the paper (`sec:unified-ranking-parity`) now all reference this doc as the canonical definition. `docs/PROMPT_ENGINEERING_PLAN.md` and other Phase 1/2 docs that use the older `composite` should be read as historical — the metric they used was correct for their decision-window but is superseded for paper-headline purposes.

## Caveat — n=50 first-N vs random-sample bias (added 2026-05-23 PM)

The cross-teacher 8-cell matrix above uses **first-50-by-sorted-ID** for the n=50 cells (legacy `--limit 50` behavior; the `--sample-seed` plumbing landed mid-day on 2026-05-23). The retry-chain re-runs of the Qwen 27B cells at random-sample seed=42 reveal that **first-N systematically under-samples the Qwen 27B variants by ~2 stage_bal points**:

- `qwen3.5 × Qwen-27B · no-think`: n=50 first-N sb 55.45 → n=200 random seed=42 sb 57.45 (+2.00)
- `qwen3.5 × Qwen-27B · think`: n=50 first-N sb 58.68 → n=100 random seed=42 sb ~60.55 partial @ 52/100 (+1.87 so far)

Under the unified metric (judge-axis assumed stable across $n$), this implies the Qwen 27B cells appear ~2 unified points lower than their random-sample truth. The "Gemma wins cross-teacher by ~2 unified pts over Qwen 27B" gap reported above could close to roughly 0 once both cells are re-run at random sample — **Gemma and Qwen 27B would then be statistically tied on this benchmark at the screening tier**, with the unified-metric winner depending on stage-c (Gemma's slight edge) vs stage-e/closure (Qwen 27B's clear edge).

We have NOT verified whether the first-N vs random bias is Qwen-specific or uniform across teachers. The other six cross-teacher cells (bert-fixed × Gemma/A3B/Qwen-27B-both-modes, qwen3.5 × Gemma/A3B) haven't been re-run at random sample yet. **The full-n=681 sub-leaderboard TODO (item 7 in `docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md`) eliminates this concern entirely** — at n=681 the sample IS the whole test split and first-N vs random is a vacuous distinction.

Until then: report the cross-teacher matrix with this caveat in the paper, and treat the "Gemma wins by 2 pts" claim as provisional until either (a) the four cross-teacher cells re-run at random-sample n≥100, or (b) the full-n=681 sub-leaderboard lands. Option (b) is the planned-of-record fix.

## What this does NOT replace

- The full per-metric table (macro, stage_bal, pedagogical, freq_inv, judge axes, R-1/R-2/BLEU-4, per-stage breakdown) stays as the methodology-section artifact. Unified is the single-number ranking; the full table is the supporting evidence.
- Per-stage analysis (especially stage e dominance findings) is reported separately because no scalar can capture stage-specific narratives.
- The benchmark critique (`BENCHMARK_CRITIQUE_AND_PROPOSAL.md`) stands on its own — unified is the *response* to the critique, not a replacement for it.

## Open questions for the paper-writing pass

- Should the unified threshold for "headline candidate" be locked at >65 or >70? Currently soft.
- Should we report `unified` and `unified_ped` side-by-side, or pick one as primary and relegate the other to an appendix?
- Should the LLM-judge model be Claude Sonnet 4.6 only (current), or expand to a multi-judge panel (Sonnet + Opus + GPT-4o) for the published headline? The latter would harden the metric against single-model bias but adds ~3× cost.
