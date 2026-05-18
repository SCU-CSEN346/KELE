# Prompt-engineering plan for the post-BERT-baseline phase

**Author:** Claude Opus 4.7 (1M ctx) for Max
**Date:** 2026-05-17
**Status:** Plan-of-record. Execution gated on the BERT + Gemma + 10-shot full run (in flight, expected ~11 AM PDT Mon 2026-05-18).

## Mission

Back-engineer the benchmark from the ground-truth teacher distribution, then design 10 concrete prompt utilizations that push the open-weight integrated system toward (or past) GPT-4o-with-SocratTeachLLM on **both** state accuracy and ROUGE-1. The 10 utilizations run on the model we choose forward (the winner of the BERT-integration n=681 run — currently Gemma 4 31B candidate; A3B fallback).

## 1. What does a "successful result" look like?

The benchmark has two axes that need to move:

| Axis | GPT-4o baseline (n=681) | A3B locked (n=681) | BERT + Gemma + 10-shot (n=50) | Aspirational target |
|---|---:|---:|---:|---:|
| State accuracy | 25.94% | 38.70% | 51.06% | **≥ 55%** |
| ROUGE-1 | 44.61 | 30.63 | 38.53 | **≥ 42** |

State accuracy is decision-relevant; ROUGE-1/-2/-L/BLEU-4 measure surface mimicry of SocratDataset's ground-truth teacher phrasing. SocratTeachLLM (the GPT-4o baseline's teacher) was *trained* on that phrasing, which is why it owns the ROUGE band; no open-weight base model reaches it by zero-shot prompting alone. To beat 42 R-1 we have to *manufacture* style adherence at the prompt layer.

## 2. Back-engineering the ground truth

A scan of all $43{,}892$ teacher turns in `references/KELE/SocratDataset.json` reveals five exploitable patterns:

### Length, per stage (chars)

| Stage | n | Mean | Median | p10 | p90 | Pedagogical role |
|---|---:|---:|---:|---:|---:|---|
| a | 6803 | 21.9 | 21 | 14 | 31 | Opening probe — short, single question |
| b | 7480 | 28.2 | 26 | 18 | 42 | Build-on-answer — slight transitional |
| c | 14445 | 29.9 | 28 | 19 | 43 | Misconception induction — terse redirect |
| d | 7361 | 39.1 | 37 | 24 | 56 | Resolution — confirm + extend |
| e | 6803 | 43.2 | 42 | 31 | 56 | Closure — summarize + check |

**Open-weight teachers consistently overshoot these lengths** by $1.5$–$3\times$ (preamble, multi-question, explanation). The single biggest lever for R-1 is forcing length compliance per stage.

### Top opening 4-grams (lexical priors)

| Count | Opener | Translation/role |
|---:|---|---|
| 6496 | 很好！那 | "Great! So..." — confirm + transition |
| 4125 | 非常好！ | "Excellent!" |
| 1436 | 很好！所 | "Great! So..." (alt) |
| 1412 | 你能告诉 | "Can you tell me..." |
| 953 | 太棒了！ | "Awesome!" |
| 804 | 很好！你 | "Great! You..." |
| 653 | 你有没有 | "Do you have/have you..." |
| 623 | 完全正确 | "Completely correct" |
| 602 | 很好，那 | "Great, so..." |
| 596 | 那你觉得 | "So what do you think..." |

**Confirmatory + transitional** ("X! 那 / 所" pattern) dominates. Question-type openers ("你能/你有/那你") are the second-most-common cluster. Open-weight teachers rarely emit these openers — they default to neutral explanatory phrasing.

### Structural rules

1. Always ends in a Chinese question mark `？` (one question, no double-questions in $>$95% of turns)
2. No preamble / no "Let me..." / no Markdown / no bullets
3. Acknowledgment + transition + question is the dominant 3-part structure for stages b–e (stage a often skips acknowledgment)
4. Average $\approx 30$ chars across the corpus — anything over 100 chars is almost certainly off-distribution

## 3. Which model do we use?

**Decision gates on the BERT + Gemma + 10-shot full run currently in flight:**

| Outcome | Decision |
|---|---|
| BERT+Gemma+10shot full lands ≥ 48% state acc | Use **Gemma 4 31B** teacher for the 10 prompt utilizations |
| BERT+Gemma+10shot full lands 40–48% | Run **BERT+A3B+10shot full** (~16h) to compare; pick the higher state-acc winner |
| BERT+Gemma+10shot full lands < 40% (collapse) | Fall back to **A3B fusion-think** teacher; the schema-fallback risk we saw for Gemma extends to the integration path |

The 10 utilizations below are model-agnostic in form (same env vars / wrapper script structure). Only the served backbone changes.

## 4. The ten prompt utilizations

Each is a discrete prompt-engineering lever. All ten are conceived as **stackable** (utilization #N+1 is meant to compose with #N), and a final headline run combines the top performers from the smoke/mini gates.

| # | Name | What it adds | Predicted gain |
|---:|---|---|---|
| 1 | Length-budgeted system prompt (per-stage) | Caps teacher output to stage-specific char budgets from §2 | R-1: +1.5 to +3.0 |
| 2 | Lexical-prior priming | Inserts the top-10 opener 4-grams as "preferred openings" | R-1: +0.5 to +1.5 |
| 3 | Style-matched exemplars (vs random 10-shot) | Replaces stage-balanced 10-shot with exemplars selected by **stylistic similarity** (length, opener, structure) to the closest train-split teacher turns | R-1: +1.0 to +2.5; State: +0.5 |
| 4 | Per-state few-shot routing (BERT-conditional) | When BERT predicts state $X$, inject 1–3 exemplars **specifically from train dialogues that landed in state $X$** | State: +1.5 to +3.0; R-1: +0.5 |
| 5 | Negative-exemplar contrast | Pair each positive exemplar with a deliberately wrong style sibling, with a brief rationale label | R-1: +1.0; risk of over-correction |
| 6 | Output-format hard constraint | "Format: One Chinese sentence. End with `？`. No preamble. No explanation." + post-hoc regex check & retry | R-1: +1.0 to +2.0 |
| 7 | Chain-of-thought scaffold (internal, then output) | Teacher reasons (1) student misconception, (2) smallest probing question, then emits **only** the final question | State: +1.0 to +2.0 |
| 8 | Self-critique / N-best re-rank | Generate 3 candidate responses, score each by (a) length match, (b) opener-match, (c) one-question-only; output highest-scored | R-1: +0.5 to +1.5; cost: 3× inference |
| 9 | Persona / role-anchored teacher | "你是苏老师，一位有20年经验的苏格拉底式教师，以问一个能让学生顿悟的问题闻名" — anchors style in a fictional senior teacher persona | R-1: +0.5; State: +0.5 (heuristic) |
| 10 | Compressed dialogue-history prompt | Summarize the last $k$ student turns into 1 sentence before the teacher call. Reduces distractor context that drives off-distribution responses | R-1: +0.5; State: +0.5; secondary: cuts wall clock 10–15% |

### Detailed implementation notes (utilizations 1–10)

#### #1 — Length-budgeted system prompt (per-stage)

The teacher's system prompt is extended with a stage-conditional length budget computed from the table in §2. When BERT (or the LLM consultant) predicts stage $s$, the teacher prompt receives a directive:

```
本轮处于阶段 {s}。回答长度应在 {p10[s]}–{p90[s]} 字之间（典型 {median[s]} 字）。
只问一个问题，以"？"结束。
```

(Translation: "This turn is in stage {s}. Response length should be {p10}–{p90} chars (typically {median} chars). Ask exactly one question, ending with `？`.")

Implementation: env var `KELE_STAGE_LENGTH_BUDGET=1`. New code path in `socratic_teaching_bert_consultant.py` that reads the BERT-predicted stage and rewrites the teacher system prompt accordingly. Single integration point; no schema changes.

#### #2 — Lexical-prior priming

The top-10 opener 4-grams from §2 are inserted into the teacher system prompt as preferred openings:

```
优秀教师的开头通常是：很好！那 / 非常好！ / 你能告诉 / 太棒了！ / ...
请采用其中一种风格开始你的问题。
```

Implementation: env var `KELE_LEXICAL_PRIORS=1`. Same integration point as #1.

#### #3 — Style-matched exemplars

Today's 10-shot exemplar pool (`src/project/socratic_teaching_unified.py`) is stage-balanced but otherwise random. Replace with a similarity-weighted selection: for each query turn, retrieve the $k=10$ train-split turns whose teacher responses have **maximum stylistic similarity** (cosine of length-normalized character-n-gram vector) to the typical response for the predicted stage. This trades the "stage balance" guarantee for "style precision."

Implementation: precompute the train-split style-vector index (one-time, ~30s). New env var `KELE_STYLE_MATCHED_EXEMPLARS=1`. Replaces the stage-balanced exemplar selection path.

#### #4 — Per-state few-shot routing (BERT-conditional)

Currently the 10 exemplars cover the 5 stages (2 per stage). BERT predicts the *34-state* code, not just the stage. Use BERT's state prediction to retrieve $k=3$ exemplars from train dialogues that ended on that exact state.

For state codes with $<3$ train-split exemplars, back off to stage-balanced selection. This composes naturally with #3.

Implementation: env var `KELE_PER_STATE_EXEMPLARS=1`. Requires building a `train_split_by_state` index. The BERT classifier's per-turn state prediction (already in the integration pipeline) feeds the index lookup.

#### #5 — Negative-exemplar contrast

For each positive exemplar, generate one **anti-example** by perturbing it: e.g., add preamble, multiply length by 2, add a second question. Label as "✗ 不推荐" (not recommended) in the prompt:

```
好的示例：很好！那你觉得种子需要一个什么样的地方？
不推荐：嗯，这是一个很好的问题。让我想想... 种子需要一个什么样的地方？另外你觉得它还需要什么？
```

Implementation: env var `KELE_NEGATIVE_EXEMPLARS=1`. Doubles exemplar token count — watch context window. Single env-gated mode.

#### #6 — Output-format hard constraint + post-hoc retry

Add to system prompt: `输出格式：只一句中文。以"？"结尾。不要有前言。不要解释。`

After generation, check (regex) for: ends-with-`？`, char count ≤ p90[stage], contains exactly one `？`. If any check fails, retry once with the error appended ("你上次的回答 {fault}，请重新回答。"). This is the *only* utilization that adds a second LLM call per turn, but only when the constraint fails (estimated <10% retry rate after constraint compliance climbs).

Implementation: env var `KELE_FORMAT_RETRY=1`. New code in `socratic_teaching_bert_consultant.py` that wraps the teacher call with a validation loop.

#### #7 — Chain-of-thought scaffold (internal then output)

Two-pass teacher prompt:

1. **Pass 1 (hidden):** "Briefly identify (a) the student's misconception, (b) the smallest probing question that exposes it. Do not output anything else."
2. **Pass 2 (final):** "Given the analysis above, output ONLY the probing question, formatted per the constraint."

The pass-1 output is internal; only pass-2 is returned to the dialogue. This adds inference cost (2× per turn) but Gemma is fast enough at this stage.

Implementation: env var `KELE_TEACHER_COT=1`. Requires a two-call orchestration in the teacher path. Mutex with utilization #6 (use one or the other; both adds 3× cost).

#### #8 — Self-critique / N-best re-rank

Generate 3 candidate teacher responses (temperature 0.8 to ensure diversity). Score each on a $0$–$3$ scale:

- $+1$ if length within stage p10–p90
- $+1$ if opens with one of the top-10 4-grams
- $+1$ if ends with exactly one `？` and contains no preamble markers (`嗯`, `让我想想`, etc.)

Output the highest-scored candidate; tie-break by lowest length. Costs 3× inference.

Implementation: env var `KELE_NBEST_RERANK=3`. New scoring helper in `src/project/style_critic.py`. Mutex with #6 and #7 (otherwise costs balloon).

#### #9 — Persona / role-anchored teacher

System prompt prefix:

```
你是苏老师（Teacher Su），一位有20年经验的苏格拉底式儿童教师。
你的特点是：每次只问一个能让学生顿悟的问题，从不解释，从不说废话。
你说话简短、温和、循序渐进。
```

Cheapest of the 10 to implement; pure prompt-string change. Implementation: env var `KELE_TEACHER_PERSONA=1`.

#### #10 — Compressed dialogue-history prompt

Before the teacher call, run a 1-shot summarization of the last $k=3$ student turns: "用一句话总结学生最近的回答："  This summary replaces the verbatim history in the teacher prompt. Reduces context distractors that drive open-weight models off-distribution at scale.

Trade-off: requires one extra small LLM call per turn (the summarizer can be the same backbone but with `max_tokens=80`).

Implementation: env var `KELE_COMPRESSED_HISTORY=1`. New helper in `socratic_teaching_bert_consultant.py`. Composes with all others.

## 5. Sequencing + evaluation protocol

The smoke→mini→full gating protocol from §4.2 of the paper applies. Each utilization gets:

1. **Smoke (n=5)** — sanity check (~5 min with BERT+Gemma teacher at 55 dlg/hr)
2. **Mini (n=25)** — promotion gate (~25 min)
3. **n=50** — apples-to-apples comparison against the locked 51.06% / 38.53 R-1 integration baseline
4. **Full (n=681)** — committed only after smoke/mini average suggests Pareto win

Order of investigation (parallelizable on smoke/mini; serialize on n=50):

```
Phase A (n=5 + n=25 smoke/mini for all 10):  ~6 hours total
Phase B (n=50 apples-to-apples for top 5):   ~5 hours
Phase C (n=681 full for top 2 composed):     ~24 hours
```

Total ~35 hours = one extended weekend slot.

## 6. Stacking strategy

After Phase B's n=50 sweep, pick the top 5 utilizations by **state acc + 0.5 × R-1** composite (state-weighted since it's the harder gain). Then compose:

1. **Headline candidate A (style-heavy):** #1 + #2 + #3 + #6 + #9 (cheap stack, no inference multiplier)
2. **Headline candidate B (reasoning-heavy):** #1 + #4 + #7 + #9 (one CoT pass; per-state routing)
3. **Headline candidate C (max-cost):** #1 + #4 + #7 + #8 + #10 (N-best re-rank + CoT — 4× inference per turn but maximum quality ceiling)

Phase C runs the best of these at full scale.

## 7. What we explicitly will NOT do

- **Re-introduce standalone Gemma fusion** at scale — the 21% schema-fallback rate is intrinsic and not closable by prompt-eng.
- **LoRA fine-tuning** of the teacher — orthogonal to prompt-eng; deferred per the paper's Next Steps. Would compose well with this plan but is a separate phase.
- **Re-train the BERT classifier** — its 61.64% state / 86.55% stage is already saturating the consultant axis. Hierarchical 5+22 head is a separate plan.
- **Change the SocratDataset split or metric definitions** — we report apples-to-apples vs the locked baselines.

## 8. Open risks

1. **Stacking may not compose linearly.** Most likely interaction: #5 (negative exemplars) and #3 (style-matched) reinforce surface-form bias but possibly drag state accuracy if the model overfits to style at the expense of routing nuance. Watch for state-acc regressions in the n=50 phase.
2. **Per-state exemplar pool may be sparse** for the rarest of the 34 states. Already mitigated by stage-balanced fallback (#4).
3. **CoT pass (#7) is mutex with N-best (#8)** for cost reasons. Phase B should determine which one earns its inference cost.
4. **If BERT+Gemma collapses at full scale**, the whole plan retargets to A3B teacher. All 10 utilizations remain valid; only the underlying serving backbone changes. The schema-fallback risk does not apply since BERT replaces the consultant.

## 9. References + open questions for Max

- Ground-truth corpus: `references/KELE/SocratDataset.json` (43,892 teacher turns)
- Existing 10-shot implementation: `src/project/socratic_teaching_unified.py` (lines around 217–225)
- BERT integration entry: `src/project/socratic_teaching_bert_consultant.py`
- Style critic: not yet implemented (proposed in #8)

**Open questions for Max before Phase A:**

1. Do you want all 10 implemented behind env vars, or should we prototype only the top 5 by expected gain (#1, #3, #4, #6, #9) and add the rest if those clear smoke?
2. Is the **35-hour total budget** acceptable for Phase A+B+C, or should Phase C land on a separate weekend slot?
3. For utilization #9 (persona), the proposed "苏老师" persona is arbitrary — do you want to tie it to a named pedagogical authority (e.g., 孔子, since this is Chinese-language tutoring), or keep it generic?
