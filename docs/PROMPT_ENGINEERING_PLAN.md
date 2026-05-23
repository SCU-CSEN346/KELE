# Prompt-engineering tournament for the post-BERT-baseline phase

**Author:** Claude Opus 4.7 (1M ctx) for Max
**Date:** 2026-05-18 (revised 6× — adds Phase 0.5 teacher-choice ablation; updated 2026-05-19 with Phase 0.5 result; updated 2026-05-19 PM with Phase 1 tournament results; updated 2026-05-21 with Phase 2-Claude parallel front; updated 2026-05-21 PM — Composed-B dropped, A3B "for-fun" variant added)
**Status:** Plan-of-record. **Phase 0 complete** (BERT + Gemma + 10-shot full = 48.15% / 36.78 R-1, locked headline). **Phase 0.5 complete** (BERT + A3B + 10-shot full = 46.57% / 33.27 R-1, Gemma stays locked). **Phase 1 complete** (10-cell n=50 tournament; length_budget #1 wins, +1.58 composite; phase 2 stack identified). **Phase 2 now runs three tracks in parallel** — Composed-Gemma (locked teacher + top-3 prompts), Composed-A3B (fast teacher + top-3 prompts, opportunistic), Claude API teacher swap. Ready for all three.
**Protocol:** Tournament-style, mirroring the §4.7 13-model no-think tournament. **n=50 × 10 cells = 500 dialogues** in Phase 1.

## Mission

Back-engineer the benchmark from the ground-truth teacher distribution, then design **10 concrete prompt utilizations** and evaluate them apples-to-apples at $n{=}50$ each. The top 2–3 single utilizations compose into a stacked headline candidate; the winner of the composed test goes to $n{=}681$ for the paper headline.

## Phase 0 — Gating run completion + result documentation ✅ COMPLETE

The BERT + Gemma + 10-shot full run completed Mon 2026-05-18 11:36:52 PDT (12h 53m). Result: **48.15% state / 36.78 R-1 at n=681** — a Pareto win over the prior A3B locked headline on both axes (+9.45 state, +6.15 R-1) and +22.21 over GPT-4o (1.86×). The schema-fallback hypothesis is confirmed (n=50→n=681 attenuation only -2.91 pts vs standalone Gemma's -10.5 pts collapse).

All Phase 0 documentation hooks landed in commit `1037e3b`: README headline, paper Abstract + §4.8.1 + Table 11 + Takeaways + Conclusion + Limitations, briefing update, new memory `bert_integration_full_2026_05_18.md`, `project_overview` refresh, `MEMORY.md` index update.

## Phase 0.5 — Teacher-choice ablation at full scale ✅ COMPLETE (2026-05-19)

**Result:** BERT + Qwen 35B-A3B-think + 10-shot at $n{=}681$ = **46.57% state / 33.27 R-1** (3,762 turns; 3h 15m wall clock with parallel-eval at $N{=}4$).

**Decision rule applied:** Gemma stays locked (A3B fell in the 45–48% "teacher choice validated by ablation" bucket).

**Per-stage split (the load-bearing finding):** A3B beats Gemma on b (+1.31) and e (+1.28) — the simpler dialogue-act stages; Gemma beats A3B on c (-3.95) and d (-2.16) — the cognitive heavy-lift stages. Consistent with the dense-vs-MoE hypothesis (Gemma's ~31B always-active params absorb harder reasoning; A3B's ~3B active-per-token MoE shines on lower-cognitive-load acts). **This directly motivates utilization #4 (per-state few-shot routing) for the tournament.**

**Methodological observation:** The matched-$n{=}50$ leaderboard is a reliable predictor of full-scale teacher ranking within the BERT-integration architecture. Attenuation was -1.62 state for A3B vs -2.91 state for Gemma — both within sampling variance, both far smaller than the +15.32-pt overshoot that destroyed the standalone-Gemma projection. The schema-fallback collapse mode is structurally absent in the integration architecture.

All documentation hooks landed (briefing, README, paper §4.8.1 ablation paragraph, `tab:allruns` row, memory sibling). Detailed result + per-stage table preserved in the previous version of this section below.

---

## Phase 1 — Prompt-utilization tournament ✅ COMPLETE (2026-05-19 PM)

**Result:** 10 single-utilization cells run at n=50 against the locked BERT + Gemma 4 31B + 10-shot baseline (51.06% state / 38.53 R-1, composite 70.33). Wall clock ~6h 09m across three sub-runs (a session crash mid-cell-10 forced a recovery run; cells 10/7/8 re-ran cleanly with no contamination of the completed cells).

### Final leaderboard (ranked by composite = state + 0.5×R-1)

| Rank | Cell | Utilization | State | R-1 | R-2 | B-4 | Composite | Δ vs base | Wall |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| — | 0 | baseline (BERT+Gemma+10shot) | 51.06 | 38.53 | 16.93 | 9.68 | 70.33 | — | — |
| 1 | **1** | **length_budget** | **51.96** | **39.91** | **17.93** | **12.37** | **71.91** | **+1.58** | 33m |
| 2 | **9** | persona | 51.27 | 39.60 | 17.89 | 12.38 | **71.07** | +0.74 | 25m |
| 3 | 5 | negative_exemplars | 50.71 | 39.82 | 18.05 | 10.88 | 70.62 | +0.29 | 27m |
| 4 | 4 | per_state_exemplars | 50.88 | 39.42 | 17.78 | 10.90 | 70.59 | +0.26 | 33m |
| 5 | 7 | cot_scaffold | 50.89 | 38.98 | 17.13 | 9.55 | 70.38 | +0.05 | 45m |
| 6 | 3 | style_matched_exemplars | 46.72 | **42.17** | **20.10** | 12.04 | 67.81 | −2.52 | 33m |
| 7 | 10 | compressed_history | 47.50 | 38.79 | 17.07 | 9.82 | 66.89 | −3.44 | 41m |
| 8 | 8 | nbest_rerank | 47.33 | 38.27 | 16.48 | 10.02 | 66.47 | −3.86 | 96m |
| 9 | 6 | format_retry | 46.45 | 39.60 | 17.82 | 10.55 | 66.25 | −4.08 | 11m |
| 10 | 2 | lexical_priors | 43.89 | 39.12 | 17.74 | 9.88 | 63.45 | −6.88 | 25m |

### Load-bearing findings

1. **Length budget (#1) is the single biggest single-prompt lever** — +0.9 state, +1.38 R-1. The §2 hypothesis that open-weight teachers overshoot stage-typical character lengths by 1.5–3× was correct: forcing per-stage budgets simultaneously lifts surface mimicry (R-1) and stage routing (state acc). This is the cleanest result of the tournament.
2. **The expensive multi-call cells underperformed.** Of the {#6 format-retry, #7 CoT, #8 N-best} mutex group only #7 cleared baseline, by +0.05 composite — i.e., noise. #8 was the worst of all multi-call cells (−3.86) at 3× inference cost. The intuition that hidden reasoning would lift state accuracy did not bear out at this composition layer.
3. **Style-matched exemplars (#3) is the only utilization that pushed R-1 above 40** (42.17 vs 38.53 base) — but state accuracy collapsed 4.3 pts. Pure surface-form optimization is anti-correlated with stage routing at this baseline. Not a Phase 2 default; reserved for an R-1-specialist composition only.
4. **Per-state routing (#4) did not deliver the Phase 0.5-predicted lift.** Despite the per-stage split that motivated this utilization (A3B wins b/e, Gemma wins c/d), BERT-conditional retrieval barely moved the needle (+0.26 composite, within sampling noise at n=50). Likely interpretation: the dense Gemma teacher already absorbs the BERT state-name signal in the prompt itself, so explicit per-state retrieval is redundant.
5. **Lexical-prior priming (#2) is actively harmful** (−6.88 composite). Listing preferred opener 4-grams as "preferred openings" appears to bias the teacher away from correct state-conditional response form — it's gaming a surface pattern at the expense of pedagogical content.

### Phase 2 composition plan — three-track structure

Phase 1 identified the top-3 prompt utilizations as the single concatenated stack worth testing. Phase 2 runs that stack across three teacher backbones in parallel — two local (Gemma locked-headline, A3B for-fun reference), one frontier (Claude API):

| Track | Teacher | Cost | Wall clock |
|---|---|---|---|
| **Phase 2-Gemma** | Locked headline teacher (Gemma 4 31B local) | $0 | ~25 min |
| **Phase 2-A3B** | Fast MoE teacher (Qwen 35B-A3B local) | $0 | ~10 min |
| **Phase 2-Claude** | Frontier teachers (Sonnet 4.6 + Opus 4.6 via Anthropic API) | $0.80–$1.20 (cached) | ~10–12 min/cell |

The three tracks are **orthogonal** in resource: Gemma and A3B alternate on the local 5090 (one at a time on port 8080); Claude runs on the API and can fire alongside either local run. **Phase 3 promotes whichever track winners clear the gates** to full n=681.

---

#### The top-3 prompt stack (used by all three tracks)

Aggregator-selected top-3 non-mutex utilizations from Phase 1, all stack-compatible, all pure prompt-string layering (no extra inference calls):

- **#1 length_budget** (+1.58)
- **#9 persona** (+0.74)
- **#5 negative_exemplars** (+0.29)

**Sum-of-effects upper bound:** +2.61 composite (73.94 from 70.33 baseline), assuming linear stacking. Realistic outcome is sub-linear; we accept the test.

**Why drop Composed-B (CoT add-on, #7):** Phase 1 result for #7 alone was +0.05 composite — essentially noise. The original justification (CoT may interact non-linearly with length budget) is speculative; testing it doubles inference cost for a borderline expected lift. Drop it from Phase 2; revisit only if Composed-Gemma underperforms.

**Why skip #4 (per-state exemplars) despite being 4th-best:** #4 only delivered +0.26 composite (also noise-band), and its hypothesized mechanism — per-stage exemplar routing — is plausibly redundant with #5 (negative exemplars already give the teacher anti-pattern guidance per stage). Diminishing returns; not worth the additional config complexity.

**Shared env-var prefix for all three tracks:**
```bash
KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=10 \
KELE_STAGE_LENGTH_BUDGET=1 KELE_TEACHER_PERSONA=1 KELE_NEGATIVE_EXEMPLARS=1
```

---

#### Phase 2-Gemma — Top-3 stack on the locked teacher

The apples-to-apples test: does the top-3 prompt composition lift the locked Gemma headline? Result baseline to beat is the Phase 1 length-budget cell (#1 alone): 51.96 / 39.91 / 71.91 composite.

**Promotion to Phase 3-Gemma (n=681) if composite ≥ 72.5.**

#### Phase 2-A3B — Opportunistic fast-teacher test (for-fun)

Same top-3 stack, but with Qwen 35B-A3B as the teacher instead of Gemma. Why bother:

1. A3B is ~4× faster than Gemma at inference — ~10 min for n=50 vs ~25 min. Cheap to add.
2. Phase 0.5 ablation showed A3B has a per-stage profile *opposite* to Gemma: A3B wins on stages b/e (simpler dialogue acts), Gemma wins on c/d (cognitive heavy-lift). The top-3 prompt stack — especially #1 length_budget — might disproportionately help A3B on the harder stages where it currently loses.
3. If A3B + top-3 stack closes more than half the c/d gap vs Gemma, it's a genuine surprise and worth a paper paragraph.

**Promotion to Phase 3-A3B (n=681) if composite > Gemma full headline (70.33 equivalent at full n).** Otherwise document as ablation.

#### Phase 2-Claude — Frontier-teacher swap

Same top-3 stack, but with Claude Sonnet 4.6 or Opus 4.6 as the teacher via Anthropic API. Spec lives in `docs/CLAUDE_API_TEACHER_PLAN.md`.

**Mission sequence (run in order, gate at each step):**

1. **Mission 1 — n=5 token-calibration probe** (~$0.05, ~5 min). Validate the 0.7× BERT→Claude token-ratio estimate against real `usage.input_tokens`. Decision gate: ±30% of estimate.
2. **Mission 2 — Sonnet 4.6 at n=50 with top-3 stack** (~$0.40 cached, ~10 min). Decision gate: composite ≥ 70.33 → proceed to Mission 3; composite ≥ 72.5 → schedule full n=681 immediately.
3. **Mission 3 — Opus 4.6 at n=50 with top-3 stack** (~$0.60 cached, ~12 min). Decision gate: must beat both Gemma headline AND Sonnet result by ≥ +1.0 composite to justify the 1.67× premium.

Pricing snapshot (verified 2026-05-21):

| Model | n=50 cached | n=681 cached | n=681 batch+cached |
|---|---:|---:|---:|
| Sonnet 4.6 | ~$0.40 | ~$5.01 | ~$2.50 |
| Opus 4.6 | ~$0.60 | ~$8.35 | ~$4.20 |

Full cost tables and wiring details in `docs/CLAUDE_API_TEACHER_PLAN.md`.

---

#### Phase 2-Claude — Frontier-teacher swap

**Hypothesis:** the BERT-routed consultant is doing its job (state acc ceiling lifted), so what's left on the table is *teacher-response quality* — especially on stage c (only 30.31% even in the locked headline). A frontier teacher with stronger reasoning may crack that ceiling. Full spec in `docs/CLAUDE_API_TEACHER_PLAN.md`.

**Mission sequence (run in order, gate at each step):**

1. **Mission 1 — n=5 token-calibration probe** (~$0.05, ~5 min). Validate the 0.7× BERT→Claude token-ratio estimate against real `usage.input_tokens` from the API. Decision gate: ±30% of estimate.
2. **Mission 2 — Sonnet 4.6 at n=50** (~$0.40 cached, ~10 min). Drop-in swap for Gemma in the locked architecture. Decision gate: composite ≥ 70.33 → proceed to Mission 3; composite ≥ 72.5 → schedule full n=681 immediately.
3. **Mission 3 — Opus 4.6 at n=50** (~$0.60 cached, ~12 min). Frontier-teacher test. Decision gate: must beat both Gemma headline AND Sonnet result by ≥ +1.0 composite to justify the 1.67× premium.

**Pricing snapshot** (verified 2026-05-21):

| Model | n=50 cached | n=681 cached | n=681 batch+cached |
|---|---:|---:|---:|
| Sonnet 4.6 | ~$0.40 | ~$5.01 | ~$2.50 |
| Opus 4.6 | ~$0.60 | ~$8.35 | ~$4.20 |

Full cost tables and the OpenAI-client-compatible wiring path in `docs/CLAUDE_API_TEACHER_PLAN.md`.

---

#### Phase 3 (split) — Full-scale promotion

**Phase 3-Local:** Whichever Composed-A/B clears 72.5 composite at n=50 → run at n=681 on Gemma teacher. Wall clock ~13h.

**Phase 3-Claude:** Whichever Claude teacher clears the locked Gemma headline by ≥ +2.0 composite at n=50 → run at n=681 via the Anthropic API. Cost ~$5–$8 cached. Wall clock ~3–4h (much faster than local because of API parallelism + no model loading).

Either or both may promote. The paper headline goes to whichever Phase 3 hits the highest composite at full scale.

---

#### Phase 4 — Combined stack (conditional)

**Trigger:** both Phase 2-Local AND Phase 2-Claude produced wins.

Stack the Phase 2-Local-winning prompt composition on top of the Phase 2-Claude-winning teacher. This is the upper-bound combined optimization: BERT consultant × frontier teacher × tournament-winning prompt stack. Re-run at n=50 first; if composite ≥ 75, promote to n=681 immediately.

**Why this is the most exciting bet:** Composed-A's hypothesized lift is +2.61 (sub-linear realistic: +1.5). Frontier-teacher hypothesized lift is similar order (+1 to +3 vs Gemma). If they're independent, the combined lift is additive — potentially clearing 55% state acc, which is the aspirational target.

---

### Artifact pointers (Phase 1)

- Per-cell results: `results/tournament-cell-{1..10}-*/metrics_summary.json`
- Per-cell dialogue traces: `results/tournament-cell-{1..10}-*/dialogues/`
- Leaderboard aggregator: `scripts/aggregate_tournament_leaderboard.py`
- Tournament wrapper: `scripts/eval_prompt_tournament.sh`
- Run logs: `logs/tournament_2026-05-19T{16-21-39,18-26-00,23-17-13}.log`

### Artifact pointers (Phase 2-Claude)

- Full plan: `docs/CLAUDE_API_TEACHER_PLAN.md`
- Pricing memory: `~/.claude/projects/.../memory/claude_api_pricing_2026_05_21.md`
- Result staging convention: `results/sonnet46-bert-fewshot10-n{50,681}/`, `results/opus46-bert-fewshot10-n{50,681}/`

---

## Phase 0.5 — Teacher-choice ablation at full scale (ORIGINAL PLAN, archived)

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

- [ ] `docs/archive/level_up_evening_briefing.md` — Phase 0.5 update with the BERT+A3B full result + decision committed
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
- [x] **Phase 0.5 — BERT + A3B + 10-shot full run** ✅ 46.57% / 33.27 R-1 (3h 15m wall, N=4 parallel)
- [x] Phase 0.5 — Decision rule applied (45–48% bucket → Gemma stays locked)
- [x] Phase 0.5 — Documentation hooks land (briefing, README, paper §4.8.1, `tab:allruns`, memory sibling)
- [x] Phase 1 — Implement env-var-gated code paths for #1–#10 (commit `b71f6b6`)
- [x] Phase 1 — Tournament wrapper script (`scripts/eval_prompt_tournament.sh`)
- [x] Phase 1 — Run 10 cells, n=50 each (✅ ~6h 09m total across crash + recovery sub-runs)
- [x] Phase 1 — Leaderboard committed; top 3 identified (#1 length_budget, #9 persona, #5 negative_exemplars)
- [ ] Phase 2 — Composed-A (#1+#9+#5) run at n=50
- [ ] Phase 2 — Composed-B (Composed-A + #7 mutex pick) run at n=50
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
