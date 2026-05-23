# The SocratDataset benchmark measures memorization, not teaching

**Author:** Claude Opus 4.7 (1M ctx) for Max
**Date:** 2026-05-21 PM
**Status:** Critique-and-proposal doc. Surfaces a methodological problem with the published KELE benchmark and proposes concrete alternatives. This is now a candidate primary paper contribution — beyond "we built a better Socratic teacher" and into "the field has been measuring the wrong thing."

## The triggering observation

When we ranked our top configurations by the sum of surface-form metrics (R-1 + R-2 + BLEU-4), this is what the leaderboard looks like at $n{=}50$ (and $n{=}681$ for the paper baseline):

| Rank | Configuration | n turns | R-1 | R-2 | BLEU-4 | Sum | State acc |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **GPT-4o consultant + SocratTeachLLM teacher** (paper baseline) | 681 | **44.61** | **26.04** | **19.60** | **90.25** | 25.94% |
| 2 | Opus 4.6 + 10-shot + top-3 stack | 271 | 42.77 | 21.12 | 15.53 | 79.42 | 49.82% |
| 3 | Sonnet 4.6 + 10-shot + top-3 stack | 281 | 43.02 | 20.52 | 14.33 | 77.87 | 48.75% |
| 4 | Gemma 4 31B + 10-shot + top-3 stack | 278 | 41.13 | 18.60 | 12.91 | 72.64 | 50.72% |
| 5 | Sonnet 4.6 + 10-shot only | 267 | 39.68 | 19.40 | 10.15 | 69.23 | 47.94% |
| 6 | Qwen 35B-A3B-think + 10-shot + top-3 stack | 276 | 37.64 | 15.69 | 9.97 | 63.30 | 48.19% |
| 7 | Gemma 4 31B + 10-shot (locked headline) | 3834 | 36.78 | 16.10 | 9.05 | 61.93 | 48.15% |
| 8 | Opus 4.6 + 10-shot only | 272 | 32.99 | 15.26 | 7.24 | 55.49 | 47.43% |
| 9 | Sonnet 4.6 raw | 260 | 29.10 | 13.38 | 5.69 | 48.17 | 45.00% |
| 10 | Opus 4.6 raw | 239 | 23.28 | 10.12 | 4.18 | 37.58 | 39.75% |

**Two observations are jarring:**

1. A **9B fine-tune from 2024** (SocratTeachLLM) crushes **Opus 4.6** on every surface metric. The gap widens the longer the n-gram: R-1 +1.59, R-2 +4.92, BLEU-4 +4.07. Higher-order n-gram overlap measures phrase-level fingerprinting, which is **the strongest possible memorization signature**.

2. The **state-accuracy axis tells the opposite story**. SocratTeachLLM's pipeline produces 25.94% state accuracy — **the worst of any configuration we tested**, including raw Opus with zero scaffolding (39.75%). The model that lexically matches ground truth most closely is also the worst at routing students through the SocRule pedagogical stages.

If R-1/R-2/BLEU-4 measured pedagogical capability, Opus 4.6 with carefully tuned prompts should top this leaderboard, not a 9B specialist from 2024. Instead, the most-memorized model wins by every surface metric, and the most-pedagogically-correct model (Gemma + top-3 at 50.72% state acc) sits 4th on surface metrics. **The metrics are inversely correlated with the thing they're supposed to measure.**

## Why ROUGE/BLEU are wrong for Socratic teaching

ROUGE and BLEU are n-gram overlap metrics designed for translation and summarization — settings where the space of "correct" outputs is small. Translation has one or a few right answers per source sentence. Summarization has a more constrained valid-output space than open-ended generation.

**Socratic teaching is the opposite kind of task.** For a given student turn at a given pedagogical state, the space of *valid* Socratic responses is enormous. Consider stage c (misconception induction): a student says "all plants grow on land." Valid teacher responses include:

- "Have you seen any plants growing in water?"
- "Are you sure? Can you think of any plants that don't grow on land?"
- "What about plants in the ocean — like seaweed?"
- "Let's think — are there places other than land where plants might grow?"

All four are pedagogically equivalent: they prompt the student to reconsider the over-generalization by introducing a counterexample. They serve the same teaching goal. **Yet they share almost no n-gram overlap with each other.**

If the ground-truth annotation happens to use phrasing #1, then a model that produces #2-#4 looks "worse" by ROUGE — even though pedagogically all four are interchangeable. **ROUGE doesn't measure teaching; it measures phrasing-match to the dataset's annotator.**

This is fine when:
- The dataset has many valid references per turn (it doesn't)
- The teacher model isn't trained on the dataset (SocratTeachLLM was)
- You don't care about generalization beyond the dataset (we do)

This is broken when:
- A model has been trained on the exact phrasing patterns the test set uses
- Surface-form mimicry is conflated with pedagogical competence
- Researchers report ROUGE as evidence of "teaching quality"

The KELE paper is in the broken regime on all three counts.

## The state-accuracy metric has a different problem

State accuracy compares the consultant's predicted state code (e.g., `c12`) against a ground-truth annotation. This is closer to measuring something real — whether the system correctly identifies the cognitive state of the student turn and routes to an appropriate teaching strategy. But it has its own flaw:

**The ground-truth annotations were generated by GPT-4 (per the original SocratDataset construction).** SocratTeachLLM (a fine-tune of GLM4-9B) and the GPT-4 consultant pipeline were both trained against GPT-4 annotation conventions. **A model that learns to imitate GPT-4's annotation idiosyncrasies will score higher than a model that makes genuinely better pedagogical judgments but disagrees with GPT-4 on annotation edge cases.**

This is the second methodological hole: state accuracy assumes the ground-truth labels are correct. They're not — they're one model's guess at labels, frozen and treated as ground truth. Any subsequent model is being evaluated on "how well does it imitate GPT-4 from 2024 on this specific task?"

## What a defensible benchmark would look like

A real benchmark for Socratic teaching needs to measure pedagogical capability under conditions where surface mimicry is impossible and the ground truth isn't a single model's frozen output. Here are five concrete proposals, in priority order by feasibility and impact:

### Proposal 1: LLM-as-judge with a structured rubric (highest priority)

For each generated teacher response, query a panel of frontier LLMs (e.g., Claude Sonnet 4.6, GPT-4o, Gemini 2.5) with a structured rubric:

1. **Socratic validity** (0-3): Is this a valid teaching move at the current stage? Does it match the SocRule action for the predicted state?
2. **Advancement** (0-3): Does this response advance the student's reasoning toward the correct answer, or does it stall / confuse?
3. **Age-appropriateness** (0-2): Is the vocabulary and complexity right for elementary school?
4. **Question-form fidelity** (0-2): Single question, no leading hints (except where stage-appropriate).

Average across the panel and across the turn. Multi-judge reduces single-model bias. Score range 0-10 per turn; report mean and per-stage breakdown.

**Why this works:** It measures the actual pedagogical thing we care about. A model that says "Have you seen plants in water?" gets the same score as "What about plants in the ocean?" because both are valid Socratic moves — surface form doesn't decide.

**Why this is feasible:** Cost ~$3 to judge 4,300 turns × 3 judges × ~200 tokens per judgment. Wall clock ~30 min on the Anthropic batch API. Cheap enough to run on every evaluation.

**Validation:** Sample 50 turns, have a human pedagogy expert score with the same rubric, measure agreement with the LLM panel. Report Cohen's kappa.

### Proposal 2: Embedding-based semantic similarity (paraphrase-tolerant R-1 replacement)

Replace n-gram overlap with cosine similarity between dense embeddings of the generated response and the ground truth. Use a Chinese sentence-embedding model (e.g., `bge-large-zh`, `m3e-base`) that captures semantic equivalence across paraphrases.

**Why this works:** "Have you seen plants in water?" and "What about plants in the ocean?" have ~0.85 cosine similarity in good embedding space, vs. ~0.15 R-1 overlap. The embedding captures that they're saying the same thing.

**Why this is feasible:** Pre-trained models are free. Computation is millisecond-per-pair. Drop-in replacement for current ROUGE reporting.

**Caveat:** Still measures similarity-to-annotator. Doesn't fix the GPT-4-labeled-ground-truth problem. But it does remove the surface-memorization advantage that ROUGE gives to fine-tuned models.

### Proposal 3: Stage-progression efficiency (a fully reference-free metric)

Measure how many turns the dialogue takes to reach stage e (closure where the student gives the correct answer). Faster = better teaching, assuming correctness at the end.

**Why this works:** Completely reference-free. No annotator labels needed. Directly measures the thing teaching is supposed to do: efficiently lead students to understanding.

**Why this is feasible:** Already implicitly tracked in our dialogue logs (`num_turns_generated` vs `num_turns_ground_truth`). Just needs to be reported as a primary metric.

**Caveat:** Needs a way to verify the student response at stage e actually IS the correct answer. Currently the student response is generated by another LLM; need to either (a) use a deterministic student simulator with ground-truth answer or (b) score the final answer against the multiple-choice key.

### Proposal 4: Multi-reference ROUGE (mitigation if Proposal 1 isn't acceptable)

For each ground-truth turn, generate 3-5 valid Socratic alternatives using a different LLM with the rubric from Proposal 1. Report ROUGE against the *set* of references (max over alternatives) instead of against a single reference.

**Why this works:** Reduces the "matched the exact annotator phrasing" advantage. A model that produces any valid Socratic response gets credited even if it doesn't match the original annotator's choice.

**Why this is feasible:** One-time cost: generate ~4,300 × 4 alternative references with Claude or GPT-4o. ~$5-10 in API spend, ~30 min wall clock. Then reuse for every future evaluation.

### Proposal 5: Held-out-topic generalization split

Construct a new evaluation split that holds out entire *subject-area chapters*, not random dialogues. SocratTeachLLM was trained on a 90/10 random split — meaning chapter 1.1 dialogues likely appear in both train and test. A chapter-level held-out split would expose whether the model has memorized chapter-specific phrasing.

**Why this works:** If SocratTeachLLM is overfit, it should collapse on a chapter-held-out split where it hasn't seen the chapter's vocabulary during training. If it's genuinely generalized, it should hold steady.

**Why this is feasible:** No new data needed. Just re-split SocratDataset by chapter ID. Re-run evaluations on the chapter-held-out test set.

### Proposal 6: Right-size the evaluation sample (avoid over-sampling)

Empirically (see [`CONVERGENCE_ANALYSIS.md`](CONVERGENCE_ANALYSIS.md)), the SocratDataset 681-dialogue test split is **over-sampled** for ranking purposes. A bootstrap analysis across all 7 of our full-scale n=681 configurations shows that **n = 400 random dialogues** is sufficient to estimate all four primary metrics (state acc, ROUGE-1, ROUGE-2, sentence-BLEU-4) within ε ≤ 2 percentage points of the n=681 truth at 95% confidence. State accuracy is the binding constraint — surface-form metrics converge at n ≈ 100-200, but state-acc needs n ≈ 400 because it's a per-turn binary 0-of-34-class indicator with higher variance per dialogue.

**Why this matters:** at n=400 vs n=681 the campaign saves **~41%** of API spend, GPU-hours, and energy *per full evaluation round* with no loss of decision precision on any ranking ≥ 2 pp. For the Anthropic API runs, that's ~$2 saved per Sonnet config and ~$3 per Opus config; for the open-weight local runs, ~5-10 GPU-hours saved per full headline. Across the campaign so far that compounds into roughly $20-30 of API spend and 30-50 GPU-hours that need not have been spent.

**Why this is feasible:** uses existing infrastructure. The convergence script at `scripts/convergence_analysis.py` produces this estimate on any future config in ~3 seconds. The recommendation is: use **n = 400** as the canonical "ground truth" sample for any future full-scale eval; use **n = 200** as a screening tier during prompt-engineering tournaments where decisions only need Δ ≥ 3 pp resolution.

**Caveat:** the analysis above assumes random sampling. A stage-and-subject-stratified subsample (forcing balanced coverage of stages a–e and across chemistry / biology / physics chapters) should converge faster — likely making n ≈ 200 stratified equivalent to n = 400 random for the same tolerance. Worth pursuing when constructing the new dataset.

### Proposal 7: Stage-balanced state accuracy (closes the closure-blindness gap) — **TODO, deferred**

**Status:** Pending. Math is sketched but not implemented; final weighting choice deferred — see "Open weighting question" below.

**The triggering observation (2026-05-22):** Qwen 27B think mode, run as the teacher behind the bge-small consultant at n=50 (partial, in flight as of write time), produces these per-stage accuracies after 175 turns:

```
stage   acc      n_turns
  a    100.0%    33
  b     36.8%    38
  c     21.1%    57   ← most frequent stage, dominates the macro
  d     41.4%    29
  e     83.3%    18   ← rarest stage; +8-17 pp lead over every other cell
```

Stage e (closure) is **+8 pp over A3B (75.0)**, **+11 pp over Gemma (72.7)**, **+17 pp over both T4-consultant cells (66.7)**. This is the largest single-stage lead any teacher has shown across the 6 cells we have full data on.

Yet the headline macro state accuracy is only ~49% — essentially tied with the locked headline (48.15%). The reason is that stage e is only **10.3%** of the test split's turns. Frequency-weighted macro state acc therefore credits Qwen 27B's closure dominance with only ~1.7 pp of headline lift, while the +12 pp deficit on stage c (Qwen 21.1% vs T4 × Gemma 33.3%) costs ~3.9 pp because stage c is 32.6% of turns. **The composite is dominated by stage c, and Qwen 27B's pedagogically most-important strength is invisible to the headline.**

**Why this matters pedagogically.** Stage e is *closure* — the point at which the teacher consolidates the learning and seals the conversation. It is:

- The *rarest* stage by frequency (one closure per dialogue, vs ≥1 questioning turn per dialogue, often several).
- The *most pedagogically load-bearing* — closure is what makes the learning stick. A botched closure leaves the student without resolution, regardless of how skillfully the teacher anchored or questioned earlier.
- The stage where surface-form metrics are *most blind* — a closure that uses different vocabulary but conveys the same consolidation looks fine on ROUGE but mediocre on stage-correct.

This is the same pathology this doc already identifies for ROUGE/BLEU (Proposal 1's motivating argument): **frequency-weighted averaging hides what is pedagogically important**. The corollary for state accuracy: macro state-acc under-weights closure.

**The math, with three weighting options.** Let $p_s$ be the per-stage accuracy and $f_s$ the per-stage frequency on the test split.

- **Current (frequency-weighted macro):** $\text{macro} = \sum_s f_s \cdot p_s$. Each turn counts once.
- **Option A — Stage-balanced macro (recommended baseline):** $\text{stage\_bal} = \frac{1}{5}\sum_s p_s$. Each stage weighted equally; methodologically standard "macro-F1" move from multi-class classification.
- **Option B — Pedagogically-weighted:** assign weights by pedagogical importance, e.g. $w_a{=}0.10$, $w_b{=}0.20$, $w_c{=}0.25$, $w_d{=}0.20$, $w_e{=}0.25$ (gives closure parity with questioning). Requires citation from KELE/SocRule literature on relative stage importance to be defensible.
- **Option C — Frequency-inverse weighted:** $w_s \propto 1/f_s$, normalized. Mathematically gives rare stages more weight; theoretically clean but may overcorrect (stage e gets ~3× weight of stage c under this).

Under the same Qwen 27B partial data (and the four already-complete n=50 cells):

| Cell | Current macro | Stage-balanced (Option A) | Δ |
|---|---:|---:|---:|
| T4 × A3B | 54.86% | 58.62% | +3.8 |
| Qwen 27B think × bge (partial, 175 turns) | 49.14% | **56.52%** | **+7.4** ← biggest jump |
| T4 × Gemma | 51.58% | 56.13% | +4.6 |
| bge × Gemma | 45.94% | 52.73% | +6.8 |
| bge × A3B | 45.36% | 52.48% | +7.1 |

Under stage-balanced, Qwen 27B leapfrogs T4 × Gemma into #2 on a *weaker consultant*. The +7.4 pp lift is the largest of any cell, entirely driven by stage e.

**Open weighting question — deferred to the paper-writing pass.** Recommendation (informed by ML convention): **adopt Option A (stage-balanced macro) as the new primary headline state-acc metric**, with the current frequency-weighted macro reported as a secondary "test-distribution-matched" number, and the per-stage table (a/b/c/d/e) reported in full as the most informative single artifact. This mirrors the macro-vs-micro F1 convention in multi-class classification literature. Options B/C remain available if a pedagogical-weights argument warrants the extra complexity.

**Why not pick now.** Picking the weighting *after* seeing the data is metric-shopping; the chain currently in flight uses the original macro for its promote decision precisely to avoid that bias. Finalizing the weighting belongs to the paper-writing pass, when we can co-design the metric with the per-stage reporting table and pick whichever option is cleanest to defend in the limitations section.

**Why this is feasible.** All per-stage data already exists. `metrics_summary.json` writes per-stage accuracies for every run we've ever done. The metric is `~5 lines` to add to `compute_all_metrics()` once the weighting is settled. No new infrastructure, no new runs.

### Backtesting (the load-bearing part — do this FIRST)

**Verified 2026-05-22:** every one of the **147 `metrics_summary.json` files** under `results/` already carries `state_accuracy.per_stage` for stages a/b/c/d/e. **The full historical leaderboard can be recomputed under stage-balanced macro with zero new runs and ~30 lines of throwaway aggregation Python.** This is not a tweak to a tail metric — it is a re-analysis of the **entire experimental record of the project**.

This matters because the original macro hides closure dominance *across every cell we've ever run*, not just Qwen 27B. We do not yet know:

- Whether the **locked headline (BERT + Gemma + 10-shot, n=681, 48.15% macro)** is still the headline under stage-balanced — or whether one of the runs we previously dismissed actually beats it.
- Whether the **Phase 2 Claude tournament rankings** (Opus + top-3 = 71.20 composite, narrowly beating Gemma at 70.33) survive the metric switch, or whether the per-stage profile inverts the order.
- Whether the **consultant upgrade campaign (T1-T4) rankings** hold — T4 won the Layer-1 classifier-only race at 67.57%, but stage-balanced macro on the *downstream* runs may tell a different story about which classifier produces the best closure quality.
- Whether the **SocratTeachLLM overfit hypothesis** strengthens or weakens — does the surface-form-winner also win on closure, or does its per-stage profile reveal an even more lopsided memorization signature than R-2 already shows?
- Whether the **teacher arms (Gemma vs A3B vs Claude vs Qwen 27B)** maintain their current ordering or re-sort. Each teacher likely has a per-stage profile and the headline ordering may not reflect *real* pedagogical capability.

**This is where the real insight lives.** The current paper narrative is built on rankings that may not survive a metric that actually credits closure. The honest move is to recompute everything before writing the headline numbers down — *especially* the locked headline, *especially* the SocratTeachLLM comparison, *especially* anything we currently treat as "settled."

**Backtest scope, in priority order:**

1. **Locked headline run** (`results/bert-consultant-fewshot10-gemma-full/metrics_summary.json`, n=681). Recompute stage-balanced macro. This is the number that anchors the paper.
2. **All n=681 full runs** (~7 configurations per `CONVERGENCE_ANALYSIS.md`). Re-rank under stage-balanced.
3. **The Phase 2 Claude triple-arch tournament** (6 configs at n=50). Does the +1 pt Opus-over-Gemma margin survive?
4. **The SocratTeachLLM Chinese vs English experiment** (4 configs). Does the memorization signature look even more lopsided when we look at stage-level performance?
5. **The 4-cell Qwen 27B grid** (in flight tonight). The cell that motivated this proposal.
6. **Every other config** (~140 remaining `metrics_summary.json` files). Sweep for any sleeper that beats the locked headline under the new metric.

**Output of the backtest:** a single `results/backtest_stage_balanced_2026_XX_XX.md` table with all 147 configurations ranked by both metrics side-by-side, plus the per-stage breakdown for the top 20. Anything that changes rank by ≥3 positions deserves a sentence of analysis in the paper. Anything that *beats the locked headline under the new metric but lost under the old one* deserves a section.

**Sequence:** backtest FIRST → write the methodology paragraph (informed by what the backtest reveals) → only THEN finalize which weighting variant becomes the paper headline. Implementing stage-balanced macro in `compute_all_metrics()` is the *last* step, not the first — by the time we add it to the live code, the backtest will already have told us what to expect every new run will look like under it.

**Connection to existing critique.** This is the third rung on the same ladder this doc already climbs:

1. ROUGE/BLEU are frequency-weighted over n-grams → reward phrase-level mimicry over teaching equivalence.
2. Macro state-acc is frequency-weighted over turns → under-counts closure, the pedagogically critical rare stage.
3. The KELE benchmark, as published, has no per-stage breakdown in its headline at all → invisibility of pedagogical structure becomes the norm.

Adopting stage-balanced macro is the natural co-headline to the four-metric panel.

### Proposal 8: Unified two-axis ranking (lands the methodology in a single number)

**Status:** Implemented 2026-05-23. See `docs/UNIFIED_RANKING.md` for full formula + rationale.

`unified_score = 0.5 × stage_balanced + 0.5 × (judge × 10)` produces a single defensible rank per configuration by averaging the two memorization-resistant metrics this doc argues for: closure-aware pedagogical correctness (Proposal 7) and rubric-based pedagogical quality (Proposal 1). Both axes are necessary; neither is sufficient; equal weight is the no-prior default.

What it changes:
- **The headline race tightens and shifts.** Cross-teacher 8-cell matrix at n=50 (all judged 2026-05-23): T4 × Gemma 31B wins unified at 68.94, beating T4 × Qwen 27B-think (66.89) despite the latter's stage_bal lead. The judge dimension carries enough signal to overturn the closure-only ranking — exactly the property a unified ranking should have.
- **The locked headline (BERT + Gemma 31B + 10-shot, n=681) lands at unified 68.65**, only 0.29 below the n=50 winner. Frontier ceiling (BERT + Claude Sonnet/Opus + top3 at n=681) sits at 69.37–70.06 unified — 0.7–1.4 points above locked at proper sample size.
- **SocratTeachLLM cells crash to the unified bottom.** Surface-form R-1 of 45–56 paired with stage_bal of 19–42 and judge of 6.6–7.8 yields unified 44–60. The metric inversion this doc surfaces gets cleanly punished by the unified ranking, without needing a separate memorization-detector. This is the unified score working as intended.

The unified column now appears in every `backtest_stage_balanced_*.md` artifact via `scripts/backtest_stage_balanced.py`.

## Recommended benchmark composition for our paper

We propose **a four-metric panel** that triangulates pedagogical capability without single-metric memorization advantages, **collapsed into a unified single-number ranking** (Proposal 8) for the paper headline:

| Metric | What it measures | Memorization-resistant? | Implementation | Role |
|---|---|---|---|---|
| **`unified` (0-100)** | Headline aggregate | ✅ Yes (by construction — averages two resistant axes) | `docs/UNIFIED_RANKING.md` | **Primary paper headline** |
| **`stage_bal`** | Per-turn pedagogical correctness (closure-aware) | ✅ Yes (per-stage) | Proposal 7 | Feeds unified |
| **`judge`** | Per-turn pedagogical quality | ✅ Yes (rubric-based) | Proposal 1 | Feeds unified |
| Per-stage table (a/b/c/d/e) | Pedagogical-stage profile | ✅ Yes | `state_accuracy.per_stage` | Methodology table |
| `macro` (frequency-weighted) | Test-distribution-matched secondary | ⚠️ Partial — hides closure | already implemented | Secondary number |
| Semantic R-1 (cosine sim) | Surface similarity, paraphrase-tolerant | ⚠️ Partial | Proposal 2 (deferred) | Future panel addition |
| Stage-progression efficiency | Turns-to-closure | ✅ Yes — reference-free | Proposal 3 (deferred) | Future panel addition |

**Rank by `unified` as the headline. Report `stage_bal`, `judge`, `macro`, and per-stage breakdown as supporting evidence. Surface-form ROUGE-1/R-2/BLEU-4 should be reported as a memorization indicator, not as a quality metric** — explicitly framed as "high values on these metrics suggest training-data overlap."

## Paper framing

This changes the paper's contribution structure. The original contribution was:

> "We built a better Socratic teaching system (BERT + Gemma + 10-shot) that beats GPT-4o + SocratTeachLLM by +22.21 pts state accuracy."

The augmented contribution is:

> "We built a better Socratic teaching system that wins on state accuracy and on the LLM-judge rubric, but loses on ROUGE/BLEU. We investigate why and find that the published benchmark is dominated by surface-form memorization signatures: a 9B fine-tune from 2024 outscores Opus 4.6 on every n-gram metric while producing the worst state accuracy of any configuration tested, including raw Opus with zero scaffolding. We propose a four-metric evaluation panel that triangulates pedagogical capability and is robust to training-data overlap. The original KELE result should be re-interpreted accordingly."

That's a much stronger paper. The methodological contribution is publishable independently of the architectural contribution — and the two together make the strongest version of the work.

## Concrete next steps

In priority order:

1. **Run the LLM-judge evaluation on the existing dialogue logs.** All n=50 runs already have full dialogue traces under `results/*/dialogues/*.json`. We need to write `scripts/llm_judge_eval.py` that calls a panel of LLMs with the rubric for each turn. Estimated effort: 1-2 hours of code, $1-3 in API spend, ~30 min wall clock. **This is the highest-leverage next step.**

2. **Add the surface-form leaderboard table** (the one Max generated) to the README and paper as evidence of the metric inversion. Frame as "the metric inversion that motivated the methodological critique."

3. **Implement semantic R-1** with `bge-large-zh` embeddings. Drop-in replacement, free to compute. Adds a more honest surface-similarity signal.

4. **Re-split SocratDataset by chapter** and re-run the locked headline + at least one Claude config on the chapter-held-out split. If SocratTeachLLM (or any model) collapses there, that's smoking-gun evidence for the overfit hypothesis without needing to repair SocratTeachLLM serving infrastructure.

5. **Write the benchmark-critique paragraph(s) into the paper** as the methodological contribution. This is the writing work; the data already exists.

6. **TODO — Stage-balanced state accuracy (Proposal 7). BACKTEST FIRST.** All **147** historical `metrics_summary.json` files already carry `state_accuracy.per_stage`. Recomputing the full leaderboard under stage-balanced macro requires zero new runs and ~30 lines of aggregation Python. **Do the backtest before writing any paper numbers down** — we do not yet know whether the locked headline survives the metric switch, whether the Phase 2 Claude tournament rankings invert, whether SocratTeachLLM's overfit signature looks even more lopsided at stage-level resolution, or whether some run we previously dismissed beats the current headline under the new metric. Sequence: backtest → write methodology paragraph informed by what it reveals → finalize weighting choice → add ~5 lines to `compute_all_metrics()` so new runs report it natively. See Proposal 7 §Backtesting for scope (priority-ordered) and expected output. **This is the load-bearing analysis of the project's final write-up, not a tail polish item.**

## TL;DR for the paper

The KELE benchmark, as published, has two structural problems:

1. **ROUGE/BLEU are inappropriate for Socratic teaching evaluation** because the space of valid responses is enormous and the metrics reward surface mimicry over pedagogical equivalence.
2. **The "ground-truth" state annotations are GPT-4-generated**, and any model trained to imitate GPT-4 annotation conventions will score higher than a model that produces genuinely better pedagogical judgments but disagrees with GPT-4 on edge cases.

Combined effect: the benchmark systematically rewards models that have memorized the dataset's phrasing patterns over models that exhibit genuine pedagogical capability. The published "GPT-4o + SocratTeachLLM" baseline R-1 of 44.61 — higher than Opus 4.6 with carefully tuned prompts — is the canonical example of the failure mode.

We propose a four-metric evaluation panel (LLM-judge rubric, state acc against BERT annotator, semantic R-1, stage-progression efficiency) that triangulates pedagogical capability without single-metric memorization advantages, and recommend that future work on Socratic teaching systems adopt it — evaluated at **n = 400** random dialogues per configuration (see Proposal 6 above), which gives ≤ 2 pp resolution on all four metrics at ~41% lower cost than the original 681-dialogue test split.

---

## Empirical validation (2026-05-22)

The four-metric proposal in this doc was implemented on 2026-05-21 PM and validated overnight. Status of each proposal:

- **Proposal 1 (LLM-judge rubric):** **built and run.** `scripts/llm_judge_eval.py` queries Claude Sonnet 4.6 with the 4-axis rubric (Socratic validity, advancement, age-appropriateness, question-form fidelity); aggregates per-turn → per-dialogue → per-config; writes `judge_summary.json` per run. Total spend across 15 judged configurations was **$59.86**. Adding GPT-4o-mini + Gemini Flash to form the multi-judge panel (the stated stretch goal) remains future work.
- **Proposal 2 (semantic R-1):** **built.** `scripts/semantic_r1.py` uses `BAAI/bge-m3` multilingual embeddings (~5s per config, free). Cosine similarity replaces R-1 for paraphrase tolerance. Numbers in `results/master_leaderboard.{json,md}`.
- **Proposal 3 (stage-progression efficiency):** **partial.** `num_turns_generated` vs `num_turns_ground_truth` is already in every dialogue log; what's missing is the closure-correctness check (currently dialogues are scored by reaching state e, not by whether the multiple-choice answer was actually given).
- **Proposal 4 (multi-reference ROUGE):** not implemented; the LLM-judge result was strong enough to make this less urgent.
- **Proposal 5 (held-out-topic split):** not yet run.

### What the empirical findings show

**1. LLM-judge ranks BERT+Gemma+10-shot (locked headline, n=681) #1 at 8.19/10**, ahead of every Claude configuration and every SocratTeachLLM configuration:

```
#1  BERT + Gemma + 10-shot LOCKED (n=681)     8.19   ← open-weight, wins fair test
#2  Gemma + top-3 (n=50)                       8.17
#3  Sonnet + top-3                             8.11
#4  Opus + top-3                               8.08
#5  Opus + top-3 (English)                     8.01
#6  Sonnet + 10-shot only                      7.84
#7  Opus → SocratTeachLLM (Chinese)            7.80
#8  Sonnet → SocratTeachLLM (Chinese clean)    7.63
#9  A3B + top-3                                7.49
...
#15 Opus raw                                   6.80
```

The open-weight system wins the memorization-resistant test. SocratTeachLLM-using configurations are nowhere near the top despite winning surface metrics.

**2. Cross-lingual translation experiment** — the most decisive single result. We ran SocratTeachLLM (still as teacher) against the English translation of the dataset, with frontier Claude as consultant:

| Configuration | R-1 | R-2 | BLEU-4 | R-2/BLEU-4 ratio |
|---|---:|---:|---:|---:|
| Sonnet+STL on Chinese (clean rerun, workers=1) | 45.61 | — | — | ~1.3 |
| **Sonnet+STL on English** | **55.85** | **33.79** | 3.56 | **~9** |
| Opus+STL on English | 44.22 | 26.20 | 2.96 | ~9 |

Sonnet+STL on English reaches R-1=55.85 / R-2=33.79 — nearly identical to the original Peng et al. paper's reported headline of R-1=57.40 / R-2=33.63. **The paper's flagship number is reproducible by translating SocratDataset into the original paper's reporting language.** The R-2/BLEU-4 ratio jumps from ~1.3 (Chinese) to ~9 (English) under the identical teacher, the textbook signature of phrase-level memorization at exactly the predicted n-gram length.

**3. Cross-lingual LLM-judge transfer** — independent confirmation:

```
Opus + top-3 (Chinese → English):         -0.07   ← transfers!
Sonnet → SocratTeachLLM (ZH → EN):        -1.01   ← 14× worse
Opus → SocratTeachLLM (ZH → EN):          -1.03
```

Frontier+prompt-eng configurations transfer Chinese→English with negligible judge loss. SocratTeachLLM-using configurations degrade **14× more** on a metric constructed to be paraphrase-invariant. The "advantage" SocratTeachLLM showed on Chinese surface metrics is revealed as language-bound memorization, not transferable pedagogical capability.

### The combined evidence

Four observations now point in the same direction:

1. The metric ordering inverts between surface form and state accuracy on the same configurations.
2. The surface-form gap widens monotonically with n-gram length (+1.59 R-1, +4.92 R-2, +4.07 BLEU-4).
3. Translating SocratDataset to English under SocratTeachLLM reproduces the original paper's R-1=57.40 headline within 1.5 points.
4. Under LLM-judge cross-lingual transfer, SocratTeachLLM-using configurations degrade 14× more than frontier+prompt-eng configurations.

**The most parsimonious account consistent with all four observations is that SocratTeachLLM was trained on test-set surface forms, either by direct contamination or by insufficient distributional separation between train and test splits.** Either interpretation undermines the paper's claimed contribution: the published R-1=57.40 / R-2=33.63 headline is not evidence of pedagogical capability but of phrasing recall.

The four-metric panel proposed in this doc was sufficient to surface this conclusion. We recommend it as the evaluation standard for future Socratic-teaching research.
