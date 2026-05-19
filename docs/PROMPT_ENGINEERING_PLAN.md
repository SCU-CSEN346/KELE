# Prompt-engineering tournament for the post-BERT-baseline phase

**Author:** Claude Opus 4.7 (1M ctx) for Max
**Date:** 2026-05-18 (revised 2× from 2026-05-17 — adds Phase 0.5 teacher-choice ablation)
**Status:** Plan-of-record. Phase 0 complete (BERT + Gemma + 10-shot full landed 48.15% / 36.78 R-1, new locked headline). Phase 0.5 staged but paused for power-management; tournament gated on Phase 0.5 winner.
**Protocol:** Tournament-style, mirroring the §4.7 13-model no-think tournament. **n=50 × 10 cells = 500 dialogues** in Phase 1.

## Mission

Back-engineer the benchmark from the ground-truth teacher distribution, then design **10 concrete prompt utilizations** and evaluate them apples-to-apples at $n{=}50$ each. The top 2–3 single utilizations compose into a stacked headline candidate; the winner of the composed test goes to $n{=}681$ for the paper headline.

## Phase 0 — Gating run completion + result documentation ✅ COMPLETE

The BERT + Gemma + 10-shot full run completed Mon 2026-05-18 11:36:52 PDT (12h 53m). Result: **48.15% state / 36.78 R-1 at n=681** — a Pareto win over the prior A3B locked headline on both axes (+9.45 state, +6.15 R-1) and +22.21 over GPT-4o (1.86×). The schema-fallback hypothesis is confirmed (n=50→n=681 attenuation only -2.91 pts vs standalone Gemma's -10.5 pts collapse).

All Phase 0 documentation hooks landed in commit `1037e3b`: README headline, paper Abstract + §4.8.1 + Table 11 + Takeaways + Conclusion + Limitations, briefing update, new memory `bert_integration_full_2026_05_18.md`, `project_overview` refresh, `MEMORY.md` index update.

## Phase 0.5 — Teacher-choice ablation at full scale (NEXT TEST)

**Before the prompt tournament starts**, we run BERT + Qwen 35B-A3B-think + 10-shot at $n{=}681$ to validate the teacher choice.

### Why this matters

The Phase 0 headline locked Gemma 4 31B as the teacher. But the underlying premise — "Gemma is the right teacher because it won at $n{=}50$" — rests on a single small-$n$ datapoint, and the campaign just got burned on this exact failure mode (Gemma standalone mini → full collapse).

Two competing hypotheses:

1. **Gemma-teacher hypothesis (current locked):** at $n{=}50$, BERT + Gemma + 10-shot = 51.06%, beating BERT + A3B + 10-shot's 48.19% by +2.87 state. The BERT integration removed the schema-fallback risk, so the teacher comparison reflects pure response-generation quality, where Gemma's dense parameter count wins.

2. **A3B-teacher hypothesis (untested at scale):** standalone A3B beat standalone Gemma at $n{=}681$ by +7.31 state. The teacher-only path may inherit some of that scale advantage. Combined with A3B's higher serving speed (~50 dlg/hr vs Gemma's 53 here — similar), BERT + A3B + 10-shot at full scale could overtake.

### Procedure

```bash
# Pre-staged wrapper (commit pending this section):
bash scripts/eval_bert_a3b_fewshot10_full.sh
```

- **n** = 681 (full test split)
- **Backbone serve:** `scripts/serve_qwen35b_a3b_think.sh` (the think variant — the default serve has `--reasoning off` baked in for the no-think tournament)
- **BERT checkpoint:** `results/state_classifier_v1/final` (same as Phase 0)
- **Env vars:** `KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=10`
- **Output:** `results/bert-consultant-fewshot10-a3b-full/`
- **Projected wall clock:** ~13–14 h (n=50 reference was 60 min for 50 dialogues; per-item resume crash-safe)
- **VRAM:** ~30 GB

### Decision rule

| BERT + A3B full result | Decision |
|---|---|
| ≥ 48.15% (beats Gemma full) | Switch locked headline to A3B; tournament uses A3B teacher |
| 45–48% | Gemma stays locked; tournament uses Gemma; teacher choice validated |
| < 45% | Same as above; Gemma's teacher-side advantage is larger than the n=50 suggested |

Either outcome strengthens the paper: we've justified the teacher backbone with a full-scale ablation instead of assuming it from $n{=}50$.

### Result documentation hooks (to land before tournament launch)

- [ ] `docs/level_up_evening_briefing.md` — Phase 0.5 update with the BERT+A3B full result + decision committed
- [ ] `README.md` — add the BERT+A3B full row alongside BERT+Gemma; bold whichever wins
- [ ] Paper Table 11 — add `BERT + A3B + 10-shot (full)` row
- [ ] Paper §4.8.1 — append the teacher-ablation paragraph (whichever wins becomes the headline)
- [ ] Paper Conclusion + Next Steps — update if headline changes
- [ ] Memory — append Phase 0.5 result to `bert_integration_full_2026_05_18.md` (or create sibling memory if headline switches)

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

## 3. The ten prompt utilizations (tournament cells)

Each is a discrete prompt-engineering lever, evaluated as a single-cell at $n{=}50$ against the Phase 0 reference baseline. All cells are env-var-gated; the underlying backbone is whichever model wins Phase 0.

| # | Name | What it adds | Predicted Δ vs ref |
|---:|---|---|---|
| 1 | Length-budgeted system prompt (per-stage) | Caps teacher output to stage-specific char budgets from §2 | R-1: +1.5 to +3.0 |
| 2 | Lexical-prior priming | Inserts the top-10 opener 4-grams as "preferred openings" | R-1: +0.5 to +1.5 |
| 3 | Style-matched exemplars (vs random) | Replaces stage-balanced 10-shot with exemplars selected by **stylistic similarity** (length, opener, structure) | R-1: +1.0 to +2.5; State: +0.5 |
| 4 | Per-state few-shot routing (BERT-conditional) | When BERT predicts state $X$, inject 1–3 exemplars from train dialogues that landed in state $X$ | State: +1.5 to +3.0; R-1: +0.5 |
| 5 | Negative-exemplar contrast | Pair each positive exemplar with an anti-example + label | R-1: +1.0; risk of over-correction |
| 6 | Output-format hard constraint + retry | Strict template + post-hoc regex check & one retry | R-1: +1.0 to +2.0 |
| 7 | CoT scaffold (hidden then output) | Teacher reasons internally, emits only the final question | State: +1.0 to +2.0 |
| 8 | N-best re-rank with style critic | Generate 3 candidates, score by length/opener/format | R-1: +0.5 to +1.5; cost: 3× |
| 9 | Persona / role-anchored teacher | "你是苏老师..." — anchors style in a senior pedagogical persona | R-1: +0.5; State: +0.5 |
| 10 | Compressed dialogue-history prompt | Summarize last $k$ student turns to 1 sentence before teacher call | R-1: +0.5; State: +0.5; secondary: cuts wall clock 10-15% |

### Detailed implementation notes (utilizations 1–10)

#### #1 — Length-budgeted system prompt (per-stage)

The teacher's system prompt is extended with a stage-conditional length budget computed from the table in §2. When BERT predicts stage $s$, the teacher prompt receives:

```
本轮处于阶段 {s}。回答长度应在 {p10[s]}–{p90[s]} 字之间（典型 {median[s]} 字）。
只问一个问题，以"？"结束。
```

Env var: `KELE_STAGE_LENGTH_BUDGET=1`. New code path in `socratic_teaching_bert_consultant.py`.

#### #2 — Lexical-prior priming

Top-10 opener 4-grams from §2 inserted as preferred openings:

```
优秀教师的开头通常是：很好！那 / 非常好！ / 你能告诉 / 太棒了！ / ...
请采用其中一种风格开始你的问题。
```

Env var: `KELE_LEXICAL_PRIORS=1`. Same integration point as #1.

#### #3 — Style-matched exemplars

Replace stage-balanced 10-shot with similarity-weighted selection: for each query turn, retrieve $k=10$ train-split turns whose teacher responses have **maximum stylistic similarity** (cosine of length-normalized character-n-gram vector) to the typical response for the predicted stage.

Env var: `KELE_STYLE_MATCHED_EXEMPLARS=1`. Requires a one-time index build (~30s).

#### #4 — Per-state few-shot routing (BERT-conditional)

Use BERT's 34-state prediction to retrieve $k=3$ exemplars from train dialogues that ended on that exact state. Fall back to stage-balanced for sparse states (<3 train-split exemplars).

Env var: `KELE_PER_STATE_EXEMPLARS=1`. Composes naturally with #3.

#### #5 — Negative-exemplar contrast

For each positive exemplar, generate one anti-example (preamble added, length 2×, second question). Label as `不推荐`:

```
好的示例：很好！那你觉得种子需要一个什么样的地方？
不推荐：嗯，这是一个很好的问题。让我想想... 种子需要一个什么样的地方？另外你觉得它还需要什么？
```

Env var: `KELE_NEGATIVE_EXEMPLARS=1`. Doubles exemplar token count.

#### #6 — Output-format hard constraint + post-hoc retry

System prompt: `输出格式：只一句中文。以"？"结尾。不要有前言。不要解释。`

Post-generation regex check: ends-with-`？`, char count ≤ p90[stage], exactly one `？`. If any check fails, retry once with the error appended.

Env var: `KELE_FORMAT_RETRY=1`. New validation wrapper in the teacher call path. Estimated retry rate <10%.

#### #7 — Chain-of-thought scaffold (internal then output)

Two-pass teacher: (1) internal reasoning about misconception + smallest probing question, (2) emit only the question.

Env var: `KELE_TEACHER_COT=1`. Two-call orchestration in the teacher path. **Mutex with #8** (use one or the other; both = 3× cost).

#### #8 — Self-critique / N-best re-rank

Generate 3 candidates (temperature 0.8). Score $0$–$3$:
- $+1$ if length within stage p10–p90
- $+1$ if opens with one of the top-10 4-grams
- $+1$ if ends with exactly one `？` and contains no preamble markers

Env var: `KELE_NBEST_RERANK=3`. New scoring helper in `src/project/style_critic.py`. **Mutex with #6 and #7**.

#### #9 — Persona / role-anchored teacher

System prompt prefix:

```
你是苏老师（Teacher Su），一位有20年经验的苏格拉底式儿童教师。
你的特点是：每次只问一个能让学生顿悟的问题，从不解释，从不说废话。
你说话简短、温和、循序渐进。
```

Cheapest of the 10 to implement. Env var: `KELE_TEACHER_PERSONA=1`.

#### #10 — Compressed dialogue-history prompt

Before the teacher call, run a 1-shot summarization of the last $k=3$ student turns ("用一句话总结学生最近的回答："). The summary replaces verbatim history in the teacher prompt.

Env var: `KELE_COMPRESSED_HISTORY=1`. One extra small LLM call per turn (same backbone, `max_tokens=80`). Composes with all others.

## 3.5 Parallel-eval infrastructure (NEW — accelerates everything downstream)

The current sequential eval loop uses only **1 of 6** llama-server KV slots. A `ThreadPoolExecutor` client layer (env-gated `KELE_PARALLEL_WORKERS=N`, default 1 for backward compat) lets concurrent dialogues fill the remaining slots. Implementation landed in `src/project/kele.py`; per-worker `SocraticTeachingSystem` isolation, crash-safe per-dialogue file output preserved, error-write + continue matches sequential behavior. CLI flag `--workers N` or env `KELE_PARALLEL_WORKERS=N`.

**Throughput projection (5090, A3B-think + BERT integration):**

| Workers | dlg/hr | 681 dlg | 500 dlg (Phase 1) |
|---:|---:|---:|---:|
| 1 (sequential) | 60 | 11 h | 8 h |
| 4 (recommended) | ~220 | 3 h | 2.3 h |
| 6 (server max) | ~280 | 2.4 h | 1.8 h |

**Validation gate:** Phase C of the parallelization work requires (a) determinism check (n=10 sequential vs parallel, metrics within sampling noise) and (b) throughput check (n=50 at N=4). Until validated, wrappers default to N=1; flip to N=4 by default once green.

**Per-model worker caps:** all three model families inherit `serve_qwen27b.sh`/`serve_gemma4_31b.sh` defaults of `-np 6 --kv-unified -ctk q4_0`. Unified KV means VRAM is paid for at server boot regardless of slot count → adding workers costs only compute, not memory. Recommended N=4 across Qwen 27B, Qwen 35B-A3B, Gemma 4 31B (leaves headroom for transient KV peaks).

## 4. Tournament protocol

**Single tier: $n{=}50$ per cell.** No smoke, no mini, no early gating. Each cell is one self-contained evaluation against the locked Phase 0 reference. Mirrors the §4.7 13-model no-think tournament that produced reviewable apples-to-apples data.

| Stage | Cells | Dialogues | Wall clock | Output |
|---|---:|---:|---:|---|
| **Phase 1 — Single-utilization tournament** | 10 | 500 | ~10-11h | One $n{=}50$ run per utilization; ranked leaderboard |
| **Phase 2 — Composed headline candidate** | 1-2 | 50-100 | ~1-2h | Top 2-3 single utilizations composed; n=50 to validate stacking |
| **Phase 3 — Full-scale headline** | 1 | 681 | ~12-14h | The Phase 2 winner at full scale → paper headline |

**Total budget: 500 + 50-100 + 681 = ~1,200-1,300 dialogues, ~24-27h wall clock.**

### Phase 1 sequencing

Cells are independent and can run back-to-back in any order. Mutex constraints from §3:
- #6, #7, #8 are mutually exclusive in any single composition (don't stack within one cell)
- All others stack freely

Suggested execution order (cheapest first → costliest last, so we get more leaderboard signal earlier in the run):

1. #9 persona (cheapest — pure prompt-string change)
2. #1 length-budget
3. #2 lexical priors
4. #6 format-retry (cheap; ~5-10% overhead from retries)
5. #3 style-matched exemplars (requires one-time index)
6. #4 per-state routing (requires one-time index)
7. #5 negative-exemplar contrast
8. #10 compressed history (one extra small call per turn)
9. #7 CoT scaffold (2× inference)
10. #8 N-best re-rank (3× inference)

### Phase 2 — Composition test

After Phase 1 lands, pick the top 2-3 by `state_acc + 0.5 × R-1` composite. Build composed configurations honoring mutex constraints:

- **Composed-A (cheap stack):** top winners from {#1, #2, #3, #4, #5, #9, #10}, plus exactly one of {#6, #7, #8}
- **Composed-B (max quality):** all of the top winners, with the costliest mutex pick

Run each composed config once at $n{=}50$ against the same reference. Pick the higher.

### Phase 3 — Full-scale headline

The Phase 2 winner runs at $n{=}681$ (~12h with BERT-consultant, ~22h if standalone fusion). Documented in the paper as the final headline.

## 5. What we explicitly will NOT do

- **Re-introduce standalone Gemma fusion** at scale — the 21% schema-fallback rate is intrinsic and not closable by prompt-eng.
- **LoRA fine-tuning** — orthogonal to prompt-eng; deferred per the paper's Next Steps.
- **Re-train the BERT classifier** — its 61.64% state / 86.55% stage is already saturating the consultant axis. The hierarchical 5+22 head is a separate plan.
- **Change the SocratDataset split or metric definitions** — we report apples-to-apples vs the locked baselines.
- **Smoke/mini gating** for individual cells — we learned in the 8h-run campaign that mini overstated the 3-shot finding by ~8 points; $n{=}50$ is the gate.

## 6. Open risks

1. **Stacking may not compose linearly.** Most likely interaction: #5 (negative exemplars) and #3 (style-matched) reinforce surface-form bias but possibly drag state accuracy. Phase 2 will surface this; we accept the risk in exchange for tournament-style cleanness.
2. **Per-state exemplar pool may be sparse** for the rarest of the 34 states. Mitigated by stage-balanced fallback (#4).
3. **CoT (#7) and N-best (#8) are 2-3× cost** — if either wins Phase 1, we accept the inference multiplier in the headline.
4. **If BERT+Gemma collapses at full scale**, the tournament retargets to A3B teacher. All 10 utilizations remain valid; only the underlying serving backbone changes. The schema-fallback risk does not apply since BERT replaces the consultant.

## 7. Status checklist

- [x] Phase 0 — BERT + Gemma + 10-shot full run completes (✅ 48.15% / 36.78 R-1)
- [x] Phase 0 — All result documentation hooks land (commit `1037e3b`)
- [ ] **Phase 0.5 — BERT + A3B + 10-shot full run (PAUSED for power; wrapper staged at `scripts/eval_bert_a3b_fewshot10_full.sh`)**
- [ ] Phase 0.5 — Decision rule applied; locked headline confirmed or switched
- [ ] Phase 0.5 — Documentation hooks land
- [ ] Phase 1 — Implement env-var-gated code paths for #1–#10
- [ ] Phase 1 — Tournament wrapper script (`scripts/eval_prompt_tournament.sh`)
- [ ] Phase 1 — Run 10 cells, n=50 each (~2-3h at N=4 parallel; ~10-11h sequential)
- [ ] Phase 1 — Leaderboard committed; top 2-3 identified
- [ ] Phase 2 — Composed config(s) run at n=50
- [ ] Phase 2 — Winner committed
- [ ] Phase 3 — Full n=681 headline run
- [ ] Phase 3 — Paper + README + memory updates

## 8. Open questions for Max

1. **Order of implementation:** should we implement all 10 env-var paths before launching, or can we batch them (e.g., implement 5, launch first half, implement remaining 5 while first half runs)?
2. **Mutex defaults:** in Phase 2 composition, my default is to test the cheap stack first and only run the max-cost stack if the cheap one underperforms. Confirm or override?
3. **Persona name (#9):** the proposed "苏老师" is arbitrary — tie to a named pedagogical authority (e.g., 孔子, since this is Chinese-language tutoring), or keep generic?

## References

- Ground-truth corpus: `references/KELE/SocratDataset.json` (43,892 teacher turns)
- Existing 10-shot implementation: `src/project/socratic_teaching_unified.py` (lines around 217–225)
- BERT integration entry: `src/project/socratic_teaching_bert_consultant.py`
- Tournament precedent in the paper: §4.7, Table 7
- Phase 0 active wrapper: `scripts/eval_bert_gemma_fewshot10_full.sh`
- Phase 0 output (in flight): `results/bert-consultant-fewshot10-gemma-full/`
