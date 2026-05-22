# Sample-size convergence — n=400 is the canonical ground-truth size

**Date:** 2026-05-22
**Decision (locked):** **n = 400 dialogues is the canonical "ground truth" sample size for Socratic-teaching evaluation going forward.** It satisfies our chosen tolerance of ε ≤ 2 percentage points on all four primary metrics (state acc, R-1, R-2, sentence-BLEU-4) across every configuration we tested. It saves **≈41% compute, wall-clock, and API spend** versus the n=681 test split with no loss of decision precision on configuration ranking.

**Question answered:** For a new dialogue-sequence dataset (and for future evaluation runs), what's the minimum sample size such that the metric estimate from n random dialogues stays within a chosen tolerance of the n=681 truth?

**Headline numbers:**
- **n = 200** — Max's original intuition; satisfies ε ≤ 1 pp on surface-form metrics (R-1, R-2, BLEU-4), but state accuracy has ~2-3 pp uncertainty. Useful as a **screening tier** during prompt-engineering tournaments.
- **n = 400 (canonical)** — satisfies ε ≤ 2 pp on **all four metrics** across all 7 configurations tested. Defensible for any leaderboard ranking where the decision boundary is ≥ 2 pp.
- **n = 600** — state acc within 1 pp; marginal precision gain over n=400 is small for a ~50% wall-clock increase.

**Compute / time / energy savings at n=400 vs n=681:**

| Track | n=681 cost | n=400 cost | Saving |
|---|---:|---:|---:|
| Anthropic API (Sonnet 4.6, our Phase 3 frontier ceiling) | ~$5 | **~$2.94** | 41% |
| Anthropic API (Opus 4.6) | ~$8 | **~$4.70** | 41% |
| Open-weight local (Gemma 4 31B on RTX 5090) | 12.9 GPU-h | **~7.6 GPU-h** | 41% |
| Open-weight local (A3B fusion-think n=681) | 16.5 GPU-h | **~9.7 GPU-h** | 41% |
| LLM-judge eval (full panel) | ~$3-4/config | **~$1.75-2.35** | 41% |

For the campaign so far this represents roughly **\$20–\$30 of API spend saved per future Phase-N round** and roughly **5–10 GPU-hours per local headline run** — both directly translating into power, time, and the ability to run more configurations in the same budget.

## Method

For each of the **7 full n=681 runs** in the repo, we:

1. Precompute per-turn signals (state correctness; ROUGE-1, ROUGE-2, sentence-BLEU-4 F1) so bootstrap iterations become O(n) array operations instead of O(n) re-scoring.
2. For each n in a grid `{25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600}`, draw **B=500** random subsamples of n DIALOGUES (without replacement), pool the constituent turns, and aggregate the per-turn signals into n-sized metric estimates.
3. For each (run, n, metric) report the 95th percentile of `|bootstrap_estimate − full-n truth|`.
4. Define the **plateau n\*** for (run, metric, tolerance ε) as the smallest n in the grid such that `p95(|dev|)` stays ≤ ε for this n AND every larger n. This guards against accidental dips.
5. The **aggregate plateau n\*** for (metric, ε) is the max of the per-run plateaus — the n that satisfies ε simultaneously across all 7 configurations.

Note on BLEU: the canonical metric in `metrics.py` is sacrebleu's *corpus* BLEU-4, which is not a per-turn average. For convergence analysis we use sentence-level BLEU-4 averaged across turns as a fast proxy with the same convergence behavior up to a constant offset. Deviation curves are interpreted relative to sentence-BLEU truth, not the reported corpus-BLEU number.

The 7 runs analyzed:

- GPT-4o + SocratTeachLLM (baseline, 4294 turns)
- A3B fusion-think (prior locked headline, 4171 turns)
- Gemma 31B standalone (retracted, 4246 turns)
- BERT + Gemma + 10-shot (LOCKED open-weight headline, 3834 turns)
- BERT + A3B + 10-shot (3762 turns)
- Sonnet + BERT + top-3 (3840 turns)
- Opus + BERT + top-3 (3794 turns)

Reproducibility: `scripts/convergence_analysis.py` + `scripts/convergence_summary.py`. Raw output at `results/convergence/aggregate.json`. Figure at `docs/figures/convergence_curves.png`.

## Aggregate plateau table (max across all 7 runs)

```
tolerance ε  |  state_acc  |  ROUGE-1  |  ROUGE-2  |  sent-BLEU-4
─────────────┼─────────────┼───────────┼───────────┼─────────────
   0.25 pp   |     —       |    600    |    600    |     600
   0.50 pp   |    600      |    500    |    500    |     400
   0.75 pp   |    600      |    400    |    300    |     250
   1.00 pp   |    600      |    200    |    200    |     175
   1.50 pp   |    500      |    125    |    100    |     100
   2.00 pp   |    400      |     75    |     75    |      75
```

(Cells are n-of-dialogues; `—` = no plateau within the grid.)

## Recommendation

**State accuracy is the binding constraint.** The four surface-form metrics converge quickly (R-1/R-2/BLEU-4 are all within 1.0 pp of truth by n=200), but state acc is a per-turn binary classification accuracy with higher variance per dialogue and converges substantially slower.

**The canonical decision:**

- **n = 400** is the canonical ground-truth size for any future full-scale evaluation in this project. Tolerance ε ≤ 2 pp on all four primary metrics across all 7 configurations tested. Cheaper, faster, and indistinguishable from n=681 at any decision boundary ≥ 2 pp.
- **n = 200** remains useful as the *screening tier* — fast enough to support apples-to-apples cell comparisons in prompt-engineering tournaments where you only need to identify a clear winner (Δ ≥ 3 pp); not strong enough for headline ranking where 1-2 pp differences matter.
- **n = 600+** is the *high-precision tier* — use only when a paper-level decision requires state-acc resolution ≤ 1 pp.

**Per-metric pattern (across all 7 runs, ε ≤ 1 pp):**

| Metric | Median plateau n\* | Worst-case run |
|---|---:|---|
| ROUGE-1 | 150 | Sonnet+BERT+top-3 needs 200 |
| ROUGE-2 | 125 | A3B fusion-think needs 200 |
| sent-BLEU-4 | 75 | A3B fusion-think needs 175 |
| State accuracy | 500 | GPT-4o + SocratTeachLLM needs 600 |

## Why state accuracy converges slower

State acc is a binary per-turn indicator (correct/incorrect across 34 classes). At ~25–50% accuracy with ~6 turns per dialogue, each dialogue gives you ~6 binary draws whose variance is ~p(1−p) ≈ 0.21. ROUGE/BLEU are continuous and bounded; their per-turn variance is empirically lower (most teacher responses cluster in a narrow R-1 band of 0.3-0.5 for an open-weight teacher). So per-turn ROUGE has a tighter individual-observation distribution, and the law of large numbers brings the mean in faster.

The orange curve in the figure (GPT-4o + SocratTeachLLM baseline) is consistently the highest variance — its low 25.94% state acc sits closer to the most-uncertain p=0.5 floor than any other run.

## Implication for the new dialogue-sequence dataset

The new Socratic-teaching dialogue dataset we're constructing should target **n = 400 dialogues** as its canonical evaluation tier — this is the smallest sample that lets a downstream user rank configurations on all four primary metrics with ≤ 2 pp resolution, and it requires ~41% less compute, time, and energy per evaluation than the original SocratDataset's 681 test dialogues.

**Stratification opportunity.** The bootstrap above samples random dialogues. A **stage-and-subject-stratified** n=400 dataset (force balanced coverage of stages a-e and across chemistry / biology / physics chapters) would converge faster than the random estimates above — likely halving the n required for the same tolerance, putting **n = 200 stratified** within reach of the ε ≤ 2 pp guarantee. This is a worthwhile extension when constructing the new dataset: stratify by (stage × subject × difficulty) to get more signal per dialogue.

**Memorization-detection caveat.** If a future dataset is explicitly designed to detect training-data contamination (per [`BENCHMARK_CRITIQUE_AND_PROPOSAL.md`](BENCHMARK_CRITIQUE_AND_PROPOSAL.md)), the cross-lingual judge signal that revealed SocratTeachLLM's memorization fingerprint is most robust at n ≥ 300 per configuration. n=400 satisfies this without further adjustment; n=200 may be too noisy at the per-config delta level (we saw -1.0 vs -0.07 cross-lingual judge deltas at n=50, but the asymmetry compresses at smaller n).

**Tldr for dataset construction:**
- Target size: **n = 400 dialogues** (random) or **n = 200-300 dialogues** (stratified)
- Each dialogue retains its full 5-stage SocRule trajectory + GPT-4-generated ground-truth labels
- Include topic / subject metadata to enable stratified evaluation downstream
- Reserve ~20% as a held-out chapter-level split for the explicit memorization-detection test (per Proposal 5 in `BENCHMARK_CRITIQUE_AND_PROPOSAL.md`)
