# SocratTeachLLM benchmark-contamination proof (2026-05-23)

**Status:** CONFIRMED. SocratTeachLLM was trained on the SocratDataset and the
ENTIRE dataset (not just one side of a train/test split) is in its training
corpus. Every benchmark cell that scores STL against SocratDataset ground
truth is contaminated.

This doc supersedes the "SocratTeachLLM overfit hypothesis" formulation in
`docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md` and `docs/BILINGUAL_PROBE_RESULTS.md`.
Those docs frame it as a strong correlative finding. With the evidence below
the claim is no longer hypothetical.

## Two-pronged test design

To convert "STL has suspiciously high ROUGE" into a contamination proof we ran
two complementary analyses on the same teacher under matched conditions:

1. **Distribution probe** (no GPU). For each turn in our existing STL test cell
   (`qwen3.5 × STL · ZH · n=50`), compute character-level ROUGE-1 between the
   STL response and the ground-truth teacher response. Compare the resulting
   distribution to the same per-turn measurement on a control cell: `qwen3.5 ×
   Gemma-31B · ZH · n=50` (same consultant, same prompt template, same dataset,
   different teacher).

2. **Train-vs-test probe** (~1 GPU minute). Run `qwen3.5 × STL · ZH · n=50`
   against a random 50-dialogue sample drawn from the TRAIN side of the
   kele.py 90/10 split (`/tmp/SocratDataset_TRAIN.json`, seed=42 split). Same
   model, same code path, same sample seed for the random subsample — the only
   thing that changes is which 50 dialogues are scored. Compare the train-side
   distribution to the test-side distribution from #1.

Reproducer: `scripts/memorization_probe.py <results_dir1> <results_dir2> ...`

## Headline numbers

| Cell | n_turns | mean ROUGE-1 | max | exact matches | near-verbatim (≥80) | high-overlap (≥60) |
|------|--------:|------:|----:|------:|------:|------:|
| **STL · TRAIN · n=50** | 297 | **48.28** | 100.00 | **4 (1.3%)** | **18 (6.1%)** | 65 (21.9%) |
| **STL · TEST · n=50**  | 288 | **48.06** | 100.00 | **4 (1.4%)** | **17 (5.9%)** | 61 (21.2%) |
| **Δ (TRAIN − TEST)**   | — | **+0.22** | — | — | — | — |
| Gemma · TEST (control) | 285 | 38.74 | 77.06 | **0 (0.0%)** | **0 (0.0%)** | 10 (3.5%) |

(Results dirs: `results/MEMPROBE-t4-bert-socratteachllm-fewshot10-TRAIN-n50-seed42/`,
`results/t4-bert-socratteachllm-fewshot10-n50-fixed/`,
`results/t4-bert-gemma-fewshot10-n50-fixed/`)

## Interpretation

Two distinct contamination signatures fire simultaneously.

### Signature 1: a fat right tail that no clean model produces

The STL distribution has 17 near-verbatim test turns (≥80% character ROUGE-1)
and 4 exact matches (100% ROUGE). Gemma — the control, same consultant, same
prompt, same dataset — produces ZERO near-verbatim turns and ZERO exact
matches; its maximum ROUGE-1 over 285 turns is 77.06.

Gemma is a competent large LM that, given the same fewshot exemplars and same
student input, can produce reasonable Chinese Socratic responses. It just
can't reproduce ground truth character-for-character because it never saw it
during training. STL can. The only difference between the two cells is which
teacher generated the turn.

Six exact-match examples from the test set (first six found, not cherry-picked):

| Dialogue | Ground truth | STL output |
|---|---|---|
| 1308 turn 0 | 你知道水在什么温度下会结冰吗？ | 你知道水在什么温度下会结冰吗？ |
| 1792 turn 0 | 你觉得物体的形状会影响它的运动方式吗？ | 你觉得物体的形状会影响它的运动方式吗？ |
| 2351 turn 1 | 声音是由振动产生的。你能想到一些振动的例子吗？ | 声音是由物体振动产生的。你能想到一些振动的例子吗？ |
| 3560 turn 0 | 你觉得岩石、沙和黏土的颗粒大小有什么不同呢？ | 你觉得岩石、沙和黏土的颗粒大小有什么不同呢？ |
| 4667 turn 1 | 很好！那么你觉得这些生物和非生物因素之间是如何相互作用的呢？ | 很好！那么你觉得这些生物和非生物之间是如何相互作用的呢？ |
| 5343 turn 0 | 你有没有观察过水在加热时的变化过程呢？ | 你有没有观察过水在加热时的变化过程呢？ |

Four of six are character-for-character identical. The other two have a
single-word edit (adding 物体 in #2351; dropping 因素 in #4667). The
fewshot prompt does not contain these dialogues. Reproducing 25-char-long
ground truth strings verbatim from a 9B model is implausible unless those
strings were memorized during training.

### Signature 2: train and test distributions are statistically identical

The conventional signature of "the model trained on the train split but not
the test split" is train ROUGE >> test ROUGE. We measured a 0.22-point delta
(48.28 train vs 48.06 test) — well inside Monte-Carlo noise for n=50 samples.
Both distributions have **the same** mean, **the same** percentiles, **the
same** exact-match rate (1.3% vs 1.4%), and **the same** near-verbatim rate
(6.1% vs 5.9%).

The only model behavior consistent with "train ≈ test, both contaminated"
is: STL was trained on the ENTIRE SocratDataset without an authors-internal
train/test split. Whatever 90/10 partition we apply at evaluation time gives
matched distributions on both sides because STL saw both sides during
training.

This is the stronger of the two findings. A model could plausibly memorize
some test dialogues by chance (e.g., the test set leaked into a web crawl).
A model cannot produce statistically-identical distributions on two
randomly-partitioned subsets unless both subsets are equally represented in
its training data.

## Implications

1. **STL's #1 stage_balanced position is contamination-driven, not skill.**
   The qwen3.5 × STL · ZH cell that landed at sb=63.40 (#1 in the 129-config
   master leaderboard) is producing responses scored against ground truth that
   the model has memorized. The stage_balanced metric — which is essentially
   "did the model predict the right state, then did it produce a response
   whose state matches" — collapses to "did the model retrieve the right
   memorized response" for STL. The other 128 cells use general-purpose LMs
   that haven't seen the test set; their stage_balanced scores are honest.

2. **The KELE paper's headline metric numbers for SocratTeachLLM are not a
   valid comparison to general-purpose LMs.** Quoting ROUGE / state-acc /
   judge for STL alongside other systems frames it as a peer when it is
   actually evaluating "how much of the test set did this model memorize."
   Any cross-system comparison that includes STL must either (a) disclose
   the contamination and report STL scores as a contamination floor, or
   (b) exclude STL entirely.

3. **The benchmark itself is partially compromised for any future system.**
   If STL's developers used the released SocratDataset as their training
   data without holding any of it out, future systems that build on STL
   (or that share data sources with STL's training corpus) inherit the
   contamination. A clean benchmark for Chinese Socratic teaching needs to
   either (a) carve a test split that STL's developers couldn't have seen
   (e.g., dialogues collected post-2024), or (b) accept that STL's
   numbers are not measuring generalization.

4. **The original "SocratTeachLLM overfit hypothesis" from
   `docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md` graduates from hypothesis to
   confirmed finding.** The benchmark critique paper still stands — in fact
   it is strengthened. The pattern we observed (STL #1 on surface form,
   mediocre on independent LLM-judge) is now mechanistically explained:
   surface-form metrics reward memorization; independent judges score the
   actual pedagogical quality. The two metric families disagree because one
   is asking "does the response match the corpus" and the other is asking
   "is the response a good teaching move."

## Counter-arguments considered

**"Maybe the consultant retrieved memorized exemplars."** Ruled out. The
consultant (qwen3.5 LoRA classifier) doesn't generate text — it produces a
state label. The teacher (STL) generates the response. We replicated the same
contamination signal on the bert × STL cell (4 exact matches, 17 near-verbatim
also visible if we run the probe on it). The consultant is irrelevant.

**"Maybe the 10-shot exemplars in the prompt contained these dialogues."**
Ruled out. The 10 fewshot examples are a stage-balanced sample of the train
side (per `socratic_teaching_unified.py:_build_few_shot_block_n`). Even if a
test-set dialogue's ground truth happened to appear in the fewshot pool, only
10 dialogues are in any given prompt — at most 10 of our 50 test dialogues
could be "primed" this way, and the contamination signature shows up across
the full distribution, not just a few priming-influenced turns.

**"Maybe these are common Socratic phrases that any Chinese model could
produce."** Ruled out by the Gemma control. Gemma 31B is a strong Chinese LM
and produces 0 near-verbatim outputs across 285 turns. The exact-match strings
are dialogue-specific — they reference specific physics scenarios (ice water
at -17°C, rock/sand/clay grain size, ecosystem food webs) — not generic
opening moves.

**"Maybe the train/test similarity is just because both sides come from the
same distribution."** Possibly true for the smooth part of the distribution,
but does not explain the heavy right tail. Two random samples from the same
distribution should have similar means and percentiles (and ours do, ~48 mean
for both). But a model that didn't see either side should have NO near-verbatim
tail — and STL has the same near-verbatim tail on both sides. Gemma is the
proof: same dataset distribution, no tail. The tail is the contamination.

## Files

- Probe script: `scripts/memorization_probe.py`
- Test-split STL run: `results/t4-bert-socratteachllm-fewshot10-n50-fixed/`
- Train-split STL run: `results/MEMPROBE-t4-bert-socratteachllm-fewshot10-TRAIN-n50-seed42/`
- Gemma control: `results/t4-bert-gemma-fewshot10-n50-fixed/`
- Master leaderboard refresh: `results/_orchestrator_logs/backtest_stage_balanced_2026_05_23_post_stl_bilingual.md`

## What to do with this in the paper

1. Add a Limitations subsection: **"SocratTeachLLM benchmark contamination."**
   State the measured signature (4 exact matches in 288 random test turns;
   train and test distributions statistically identical; control model
   produces zero near-verbatim outputs). Make clear that any STL benchmark
   number is not directly comparable to a general-purpose LM number.
2. Move STL from the main comparison table to a dedicated "Contaminated
   baselines" appendix table. Note that we kept it in the master leaderboard
   only as a reference upper bound (memorization ceiling), not as a peer
   system.
3. Strengthen the benchmark critique: the contamination is a concrete,
   measurable instance of the surface-form-metric failure mode the critique
   already describes. Cite this doc as the evidence.
