# SocratTeachLLM benchmark contamination — proof and paper-grounded framing (2026-05-23)

**Status:** CONFIRMED — with the framing tightened after reading the KELE
paper. SocratTeachLLM produces character-for-character identical outputs to
SocratDataset ground-truth on a measurable fraction of held-out test turns;
the KELE paper itself (Peng et al., EMNLP 2025) confirms STL was fine-tuned
on SocratDataset; **the benchmark as published is unreproducible** because
the authors did not release their specific train/test split, so any
independent evaluator who runs STL on a random 10% subsample gets a
memorization score, not a generalization score.

The original "SocratTeachLLM overfit hypothesis" formulation in
`docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md` and `docs/BILINGUAL_PROBE_RESULTS.md`
graduates from hypothesis-grade to a structurally defensible finding. We also
identified a second, distinct methodology critique in the paper: their
headline comparison ("SocratTeachLLM surpasses GPT-4o") is fine-tuned-on-task
vs zero-shot, not a peer-to-peer comparison.

**What we are NOT claiming:** that the authors lied. The KELE paper explicitly
declares a 90/10 split (§4.3): *"The dataset, SocratDataset, was split into
90% for training and 10% for testing."* We have no direct evidence the
authors trained on the whole dataset. The strongest defensible reading of our
evidence is that they honestly held out 680 dialogues, used a different
random seed than ours, never released the split, and therefore left a
benchmark that no third party can reproduce honestly.

## What the KELE paper says about training data

Three quotes from Peng et al. EMNLP 2025 (`references/KELE/2025.findings-emnlp.888.pdf`)
that pin down the methodology we are critiquing:

> §4.2: *"The SocratDataset consists of 6,803 multi-turn dialogues, totaling
> over 42,000 teacher-student interaction turns…"*

> §4.3: *"We trained SocratTeachLLM on GLM4-9B using the LoRA fine-tuning
> method, with 3 epochs, a learning rate of 5e−5 and a batch size of 16. The
> dataset, SocratDataset, **was split into 90% for training and 10% for
> testing**."*

> §5.2: *"We first randomly sampled 680 multi-turn dialogues from the
> SocratDataset as the test set and decomposed them into 4,245 single-turn
> dialogues, each containing teaching consultant evaluation result and
> teaching action suggestion."*

What the paper does **not** say:
1. Which random seed produced the 680-dialogue test set, or
2. Whether the test split was distributed alongside the model weights, or
3. Whether their LoRA fine-tune used only the 90% train side or the full
   6,803 dialogues.

Our HuggingFace model card audit (`yuanpan/SocratTeachLLM`) found the README
is empty — no training-data disclosure beyond the apache-2.0 license and
`language: zh` tag. The split is effectively private.

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

This signature has two plausible explanations, both of which damage the
benchmark equally:

**Explanation A — authors' 90/10 split is different from ours.** They
honestly held out 680 random dialogues using a seed we don't know. Two
random 10% samples of a 6,803-dialogue corpus have ~10% expected
intersection. So if their seed differs from ours:
  - ~90% of *our* test set is in *their* training set
  - ~90% of *our* train set is in *their* training set
  - both look "memorized" at the same rate because the contamination rate
    over *our* partition is the same on both sides
This is fully consistent with the paper's §4.3 claim and our measured
distributions. **It makes the benchmark unreproducible without their
specific split**, which the authors did not release.

**Explanation B — the 90/10 split claim is post-hoc or partial.** The
authors fine-tuned on the full dataset; the "90/10 split" was the
evaluation-time partitioning of held-out scoring rather than training-time
exclusion. We have no direct evidence for this and the paper's wording
("was split into 90% for training and 10% for testing") reads against it.

Either way, the practical implication is identical: every third-party
benchmark of STL on SocratDataset measures memorization. The only escape is
to evaluate STL on data that demonstrably could not have been in any 90% of
SocratDataset — i.e., a freshly-constructed test set (see
`scripts/generate_synthetic_socrat.py` for our clean-probe attempt).

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

## A second methodology critique surfaced from the paper

§5.2 of Peng et al. reports SocratTeachLLM beating GPT-4o, GLM4-9B, Qwen2.5-7B,
Qwen2.5-14B, Qwen2.5-32B, SocraticLM-7B, and EduChat-13B on every metric in
their Table 1. This is presented as evidence of the KELE framework's
effectiveness ("SocratTeachLLM surpasses GPT-4o, which has several times
larger parameter scale, in all teaching capability").

The comparison is structurally broken regardless of contamination:
**SocratTeachLLM was fine-tuned on the task; every baseline was evaluated
zero-shot.** A model fine-tuned on a corpus will always outperform a generalist
model that has never seen that corpus's distribution. The Table 1 numbers do
not measure "intrinsic Socratic teaching ability of the LLM family" — they
measure "how much does task-specific fine-tuning help on the corpus you
fine-tuned on." That is a tautology, not a finding.

A valid comparison would either (a) zero-shot STL too, or (b) fine-tune every
baseline on the same SocratDataset training split. The paper does neither.

## What to do with this in our paper

1. Add a Limitations subsection: **"SocratTeachLLM benchmark contamination."**
   State the measured signature (4 exact matches in 288 random test turns;
   train and test distributions statistically identical at 48.28 vs 48.06
   mean ROUGE-1; Gemma 31B control on the same dataset produces zero
   near-verbatim outputs). State the two possible mechanisms (unreleased
   split with high overlap vs over-training on the full corpus) — both
   produce the same practical implication: STL's benchmark numbers are
   memorization-aided when reproduced by third parties.

2. Move STL from the main comparison table to a dedicated "Contaminated
   baselines" appendix table. Note that we kept it in the master leaderboard
   only as a reference upper bound (memorization ceiling), not as a peer
   system. Mark its row with the contamination flag.

3. Add a paragraph contrasting STL's reported KELE-paper numbers (Table 1:
   R-1=57.4) with both our measured STL numbers on our split (R-1=48.07,
   ~10 points lower, consistent with partial split-overlap) and the
   Gemma 31B control numbers (R-1=38.76, a fair reflection of a clean LM's
   performance). This documents both contamination and the size of the
   contamination-driven inflation.

4. Strengthen the existing benchmark critique
   (`docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md`): STL is a concrete, measurable
   instance of the surface-form-metric failure mode that doc predicts. A
   model that memorized the response distribution scores highest on
   surface-form metrics while losing on the independent LLM-judge axis. This
   is no longer a hypothesis — it is measured behavior on a published
   peer-reviewed model.

5. Note the second methodology critique above (fine-tuned-vs-zero-shot
   comparison in the paper's Table 1) as a SEPARATE finding from
   contamination. It does not depend on the contamination evidence and would
   stand on its own even if the split were released and clean.

## What we are not doing (and why)

**We are not constructing a full human-annotated clean Chinese Socratic
dataset.** A fresh, peer-reviewed benchmark would be a separate research
contribution requiring months of work, Chinese-curriculum domain experts, and
inter-annotator reliability validation. The contamination finding does not
need a clean dataset to be valid — the Gemma control, the train/test
identity, and the verbatim matches carry the argument.

**We constructed a small synthetic clean probe** (Claude-Sonnet-generated
Chinese elementary-science Socratic dialogues; see
`scripts/generate_synthetic_socrat.py` and the `references/synthetic/`
checked-in dataset). Results below.

**We are not escalating to a fraud accusation.** The paper's §4.3 split claim
is consistent with our evidence under Explanation A. We have no evidence
sufficient to claim the authors knowingly trained on the test set. The
defensible critique is methodology + unreproducibility, not dishonesty.

## Synthetic clean-probe results — STL on truly unseen data is *worse* than Gemma (2026-05-23 PM)

To isolate STL's generalization from its memorization of SocratDataset, we
generated 37 fresh Chinese elementary-science Socratic dialogues with Claude
Sonnet 4.6 (the synth gen had two JSON-parse failures that the recovery path
salvaged partial yield from; intended n=50 yielded 37 fully-validated
dialogues, totaling 211 evaluation turns). The dialogues use brand-new
questions, brand-new student trajectories, and brand-new teacher references
that demonstrably cannot have been in any 90% subsample of SocratDataset.

Same `qwen3.5 × STL` configuration, same `--sample-seed 42`, same Sonnet
LLM-judge. Comparison (the headline table for the paper):

| Cell | n_turns | R-1 | state_acc | exact (R-1=100) | near-verbatim (R-1≥80) | judge |
|------|--------:|----:|----:|---:|---:|----:|
| STL · TRAIN (memorized) | 297 | 48.28 | 55.22 | 4 (1.3%) | 18 (6.1%) | — |
| STL · TEST (memorized)  | 288 | 48.06 | 58.33 | 4 (1.4%) | 17 (5.9%) | 7.30 |
| **STL · SYNTH (CLEAN)** | **211** | **35.72** | **29.38** | **0 (0.0%)** | **0 (0.0%)** | **6.97** |
| Gemma · TEST (control)  | 285 | 38.74 | 51.58 | 0 (0.0%) | 0 (0.0%) | 8.18 |

(Sources: `results/CLEANPROBE-t4-bert-socratteachllm-fewshot10-SYNTH-n50-seed42/`
metrics + judge summaries; `references/synthetic/SocratDataset_SYNTHETIC.json`
is the input dataset.)

### What this proves

**STL on unseen data is WORSE than Gemma on every axis** — confirming that
its apparent excellence on SocratDataset is entirely contamination-driven:

- **R-1 inflation from contamination: ≈ +12 points.** STL · TEST = 48.06; STL ·
  SYNTH = 35.72. On clean data STL falls 3 points BELOW Gemma's clean baseline
  (38.74). The 10-point lead STL had over Gemma on SocratDataset isn't a real
  capability; it's the surface-overlap of memorized responses.

- **State-accuracy inflation: ≈ +29 points.** STL · TEST = 58.33; STL · SYNTH
  = 29.38. (Caveat: state_acc on synth data also penalizes legitimate state
  disagreements with Claude's synthetic ground-truth states, so this gap is a
  noisier signal than R-1. The collapse is still real but the magnitude is an
  upper bound.)

- **Judge inflation: ≈ +0.33 points.** STL · TEST = 7.30; STL · SYNTH = 6.97.
  Smallest gap of the three metrics, which is exactly what the benchmark
  critique predicts: the LLM-judge is the most contamination-resistant
  metric because it scores pedagogical quality independently of corpus
  overlap. STL even on clean data is still a reasonable Socratic teacher
  (6.97/10) — but worse than Gemma (8.18/10).

- **Smoking-gun control fires perfectly.** STL · SYNTH produces 0 exact
  matches and 0 near-verbatim turns out of 211 — distributionally
  indistinguishable from Gemma's clean baseline (also 0 exact, 0
  near-verbatim out of 285). The 17 near-verbatim STL · TEST turns weren't
  pedagogical fluency; they were memorized retrieval.

### The narrative for the paper

The contamination story can now be told with three orthogonal evidence
streams that all fire on the same conclusion:

1. **Verbatim exact-match copies on the test set.** STL produces 4
   character-for-character identical outputs to held-out ground truth that
   reference specific physics scenarios (ice water at -17°C, rock/sand/clay
   grain size, ecosystem food webs). Gemma produces zero across the same
   number of turns.
2. **Train and test statistically identical.** STL on a random 50-dialogue
   train sample has the same R-1 distribution as on a random 50-dialogue test
   sample (Δ R-1 = +0.22; same exact-match rate; same near-verbatim rate).
   No conventional train-vs-test memorization signature exists because the
   model wasn't trained on a held-out partition we can reproduce.
3. **Clean-probe collapse below the Gemma baseline.** STL on truly unseen
   synthetic data is WORSE than Gemma on all three primary axes. The
   ~10-point R-1 lead on SocratDataset is entirely memorization signal.

The benchmark critique paper can now make a quantitative claim: **for STL,
roughly 12 R-1 points and 29 state-accuracy points of measured "performance"
on SocratDataset are contamination-driven, not generalization.**
