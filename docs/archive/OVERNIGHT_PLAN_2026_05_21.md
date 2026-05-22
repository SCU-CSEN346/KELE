# Overnight run plan — 2026-05-21 PM → 2026-05-22 08:00 AM

**Author:** Claude Opus 4.7 for Max
**Window:** ~9h 30m from now (22:23 PDT) to 08:00 PDT tomorrow
**Paper deadline:** 2026-06-04 (14 days out — overnight session is for paper-readiness, not crunch)
**Total API budget consumed so far:** ~$2.55 of available top-up

## State at plan time

**Running right now:**
- A-Sonnet (Sonnet 4.6 consultant + SocratTeachLLM teacher, n=50, task `bck0vqmhs`) — ETA ~3 min
- A-Opus (Opus 4.6 consultant + SocratTeachLLM teacher, n=50, task `bxf7gni2i`) — ETA ~7 min
- SocratTeachLLM HF-Transformers serving on :8001 (LD_LIBRARY_PATH workaround for libnvrtc)

**Locked or quasi-locked at n=50:**
- Gemma 4 31B + top-3 (composite 71.28) — Phase 3 candidate
- Opus 4.6 + top-3 (composite 71.20) — alternate Phase 3 candidate
- Sonnet 4.6 + top-3 (composite 70.26)
- Qwen 35B-A3B + top-3 (composite 67.01) — underperformed
- Plus 4 ablation configs (raw + 10-shot-only for Sonnet/Opus)

**Locked at n=681:**
- BERT + Gemma 4 31B + 10-shot (48.15% / 36.78 R-1) — the locked paper headline
- BERT + A3B + 10-shot (46.57% / 33.27 R-1) — teacher-ablation
- GPT-4o + SocratTeachLLM (25.94% / 44.61 R-1) — the suspicious baseline

## The non-negotiable: time-window math

| Run | Wall clock | Fits before 8 AM? |
|---|---:|:---:|
| Gemma + top-3 at n=681 (local) | ~13h | ❌ finishes ~11 AM |
| Opus 4.6 + top-3 at n=681 (API) | ~3-4h | ✅ |
| Sonnet 4.6 + top-3 at n=681 (API) | ~3-4h | ✅ |
| A3B + top-3 at n=681 (local) | ~3h | ✅ if no Gemma full running |
| LLM-judge eval over all n=50 logs (API, ~30K judgments) | ~30 min | ✅ |
| Semantic R-1 over all dialogue logs (CPU/GPU local) | ~10 min | ✅ |
| Figure generation (all new charts) | ~30-45 min | ✅ |
| Doc passes after each milestone | ~15 min each | ✅ |

**Hard constraint:** Gemma full at 13h overflows the window. Two ways to handle:
1. Skip it tonight. Use n=50 number for the paper Phase 3 narrative. (Lowest risk.)
2. Run it anyway as fire-and-forget — finishes during the day, can be folded in later.
3. Run a middle-ground n=200 or n=300 Gemma run (~4-5h) to get full-scale signal without 13h commitment.

Recommend Option 3 (n=200 Gemma) — see Phase B plan below.

## The plan — four phases

### PHASE A (22:25 → 22:55, ~30 min) — Land experiment-A and update docs

**Goal:** Finish the literal GPT-4o-baseline-mirror runs and integrate into all docs.

- 22:25–22:35: Wait for A-Sonnet + A-Opus to complete (background, no action needed)
- 22:35–22:45: Read both metrics_summary.json. **Critical check:** does SocratTeachLLM's R-1 hold ~44 regardless of consultant? If yes → overfit hypothesis CONFIRMED. If R-1 drops → original 44.61 was partly a consultant-prompting artifact.
- 22:45–22:55: Update README + paper + benchmark-critique doc with the experiment-A results table. Commit + push.

**Output:** Final 4-config × 2-axis ranking inversion fully evidenced.

### PHASE B (22:55 → 02:00, ~3 hours) — Full-scale Phase 3 promotions in parallel

**Three parallel tracks. No resource contention.**

**Track B1 — API track 1: Opus 4.6 + top-3 at n=681** (~3-4h, ~$8 cached)
```bash
KELE_STAGE_LENGTH_BUDGET=1 KELE_TEACHER_PERSONA=1 KELE_NEGATIVE_EXEMPLARS=1 \
  ./scripts/eval_bert_claude_fewshot10.sh opus  # no --n flag = full n=681
```

**Track B2 — API track 2: Sonnet 4.6 + top-3 at n=681** (~3-4h, ~$5 cached)
```bash
KELE_STAGE_LENGTH_BUDGET=1 KELE_TEACHER_PERSONA=1 KELE_NEGATIVE_EXEMPLARS=1 \
  ./scripts/eval_bert_claude_fewshot10.sh sonnet
```

**Track B3 — Local GPU: Gemma + top-3 at n=200** (~5-6h fire-and-forget) OR A3B + top-3 at n=681 (~3h)

Tradeoff: Gemma full would be the ideal Phase 3 headline but doesn't fit cleanly. n=200 gives mid-scale signal in ~5h. A3B full takes ~3h and gives a full-scale data point for the A3B+top-3 ablation.

**Recommend Track B3 = Gemma + top-3 at n=200.** Reasons:
- n=200 = 4× the n=50 sample → much tighter confidence interval on the 71.28 composite
- If it holds, Phase 3-Gemma full at n=681 becomes a low-risk overnight tomorrow
- A3B+top-3 underperformed at n=50; full-scale ablation isn't a priority

**Rate-limit considerations:** We just saw 135 retries on Sonnet n=50 with 4 workers; at n=681 that scales to ~1800 retries with the same 4 workers. Need to either:
- Reduce KELE_PARALLEL_WORKERS=2 (halves throughput but reduces 529s)
- Stagger the two API runs (start Opus first, start Sonnet 30 min later when Opus has settled into a rhythm)
- Use Anthropic's prompt caching to reduce per-call input tokens

**Recommend: stagger by 30 min and use workers=2 each.** Total wall clock unchanged (~3.5h serial overlap) but cleaner.

### PHASE C (02:00 → 06:00, ~4 hours) — Memorization-resistant evaluation panel

**Goal:** Implement and run the 4-metric replacement panel from `docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md` against all existing dialogue logs. This is the methodological-paper contribution validation.

**Track C1 — LLM-judge eval** (~$2, ~30-60 min)
Write `scripts/llm_judge_eval.py` that:
- Reads dialogue logs from `results/*/dialogues/*.json`
- For each turn, queries Claude Haiku 4.5 ($1/M input, $5/M output — cheap) with the 4-axis rubric from BENCHMARK_CRITIQUE_AND_PROPOSAL.md
- Aggregates per-turn → per-dialogue → per-config scores
- Writes per-config `judge_summary.json`
- Output: ranking by LLM-judge composite across all 10+ configs we have

**Track C2 — Semantic R-1** (~10 min, free)
Write `scripts/semantic_r1.py` using `bge-large-zh` from HF:
- Embed each `teacher_response` and `ground_truth_teacher`
- Compute cosine similarity per turn
- Aggregate per-config
- Compare ranking to surface R-1

**Track C3 — Stage-progression efficiency** (~5 min, free)
Already mostly in dialogue logs:
- `num_turns_generated` vs `num_turns_ground_truth` — efficiency ratio
- Need to also score "did the student reach stage e correctly?" — for that, parse the final state in dialogue and check it's `e34`

Combine into one consolidated leaderboard with 4 metrics ranked side-by-side.

### PHASE D (06:00 → 07:00, ~1 hour) — Figures + final doc sweep

**Track D1 — Generate figures** (~30 min):
- `fig_leaderboard_inversion.pdf`: side-by-side surface-form vs state-acc ranking (the smoking gun visualization)
- `fig_4metric_panel.pdf`: bar chart with all 4 metrics per config
- `fig_pareto_inversion.pdf`: scatter of (state acc, surface-form sum) showing the inversion as anti-correlation
- `fig_ngram_gap.pdf`: bar chart showing R-1/R-2/BLEU-4 gap between SocratTeachLLM and Opus widening monotonically
- `fig_per_stage_top4.pdf`: per-stage state acc for the top-4 configs

Use `scripts/make_paper_figures.py` as the template.

**Track D2 — Final doc pass** (~30 min):
- Integrate Phase B and Phase C results into README + paper
- Add the 4-metric panel results table to paper §4
- Add the LLM-judge ranking as a new paper subsection
- Commit + push the final state

### PHASE E (07:00 → 08:00) — Buffer + status report

- 30 min buffer for any unexpected issues (rate limits stuck, server crashes, etc.)
- Write a clean status report at the bottom of `OVERNIGHT_PLAN.md` summarizing what landed
- Final commit + push

## Assumptions and unknowns to investigate

These are flagged for either Max's input or autonomous investigation as time allows:

### 1. Will Anthropic rate-limits allow two parallel n=681 runs?
- **Risk:** Tier-1 RPM is 50; with 2 runs × 6 calls/dialogue × 2 workers each = ~24 concurrent. Should fit but tight.
- **Mitigation:** Workers=2 each + 30-min stagger. If 529 rate gets above 30%, drop one run or use Anthropic batch API (50% off, ~24h delivery).

### 2. Will SocratTeachLLM stay alive overnight?
- **Risk:** The LD_LIBRARY_PATH workaround is fragile. If the server crashes, we lose experiment-A reproducibility.
- **Mitigation:** Only experiment-A needs SocratTeachLLM running, which finishes within 10 min. After that, kill it to free GPU for Phase 3-Gemma/A3B.

### 3. Was SocratTeachLLM trained on the test split?
- **Cannot determine directly without the original training code.**
- **Indirect tests we can run:**
  - Chapter-held-out split eval (deferred to tomorrow — too much engineering for tonight)
  - Check the HF model card and original paper for explicit splitting methodology
  - Verify our 90/10 train/test reproduces the paper's split with seed=42
- **For tonight:** add a paragraph to BENCHMARK_CRITIQUE doc noting what we can and cannot determine.

### 4. Why is paper R-1 = 57.40 but our reproduction = 44.61?
- 12.79-point gap on the same model + same dataset + same metric.
- Possible causes: different tokenization (char vs word-piece), different generation params (temperature, top_p, max_tokens), different consultant prompt, different test-split selection.
- **For tonight:** check `src/project/metrics.py` to confirm ROUGE tokenization. If word-level, that explains it (Chinese ROUGE is usually char-level).

### 5. Stage c bottleneck is universal at ~20-30%. Why?
- BERT classifier hits 61.64% overall state acc but the integrated system gets ~30% on stage c at full scale.
- Suggests the *teacher response*, not the BERT routing, is the c-stage bottleneck.
- **For tonight:** add per-stage analysis to LLM-judge eval to localize which axis (validity, advancement, age-appropriateness, question-form) is weak on stage c.

### 6. Does the top-3 stack work the same way at full scale?
- Phase 1 single cells were tested at n=50 and showed +1.58 composite for length_budget alone.
- Phase 2 stack at n=50 showed +0.96 over locked baseline.
- We have NO full-scale validation of the top-3 stack yet.
- **Phase 3-Opus at full n=681 will resolve this** for Claude; Phase 3-Gemma stays partial (n=200 tonight).

### 7. Is the LLM-judge eval susceptible to its own biases?
- Using Claude Haiku 4.5 as judge introduces a Claude-bias risk — it might rate Claude-generated responses more favorably.
- **Mitigation:** Use a panel (Claude Haiku + GPT-4o-mini + Gemini Flash) and report inter-judge agreement.
- **For tonight:** start with Claude Haiku only; expand to multi-judge if time allows.

### 8. Are we double-counting "memorization" by including the 10-shot exemplars?
- The 10-shot exemplars come from the TRAIN split — they should not contaminate the TEST evaluation.
- But they DO carry ground-truth phrasing patterns into the teacher prompt.
- **For tonight:** verify exemplar generation uses train-split only (`KELE_FEW_SHOT_N=10` source). Check `_build_few_shot_block_n` in `socratic_teaching_unified.py`.

## What's NOT in tonight's plan (deferred backlog)

These are real work items but don't fit the window:

1. **Chapter-held-out split** — engineering work, run tomorrow
2. **Multi-judge panel** — add GPT-4o + Gemini after tonight's Haiku-only baseline
3. **LoRA fine-tune of Gemma teacher** — ~30 min training, holds for after Phase 3 lands
4. **Hierarchical 5+22 BERT classifier** — proposed in plan doc, deferred
5. **Phase 3-Gemma full n=681** — needs ~13h, runs tomorrow during the day
6. **A3B + top-3 full n=681** — only run if Gemma full not chosen for tomorrow

## Decision points for Max

Before I autonomously start Phase B, three explicit choices:

### Q1: Anthropic spend approval for Phase B
- **Opus full + Sonnet full at n=681** = ~$13 total uncached, ~$8 cached
- Plus Phase C LLM-judge = ~$2-5
- **Total estimated spend for tonight: ~$15-20**
- Current API budget remaining: ~$2.45 + whatever you top-up to. Need ≥$25 in account to run safely.

### Q2: Track B3 local-GPU strategy
- **A. Gemma + top-3 at n=200** (~5h, mid-scale signal) ← my recommendation
- **B. A3B + top-3 at n=681** (~3h, full-scale signal on a known-underperforming config)
- **C. Skip local-GPU work tonight, all GPU runs deferred to tomorrow**
- **D. Gemma + top-3 at full n=681 anyway** (won't finish by 8 AM, ~11 AM completion)

### Q3: LLM-judge model selection
- **A. Claude Haiku 4.5 only** (~$2, fastest, single-model bias risk) ← my recommendation
- **B. Multi-judge panel: Claude Haiku + GPT-4o-mini + Gemini Flash** (~$6, better robustness, ~3× wall clock)
- **C. Claude Sonnet 4.6 as single judge** (~$3, less bias but slower)

## Recommended TL;DR

If you say "go" to my recommendations (B1+B2+B3-option-A, C1=Q3-option-A, full C2+C3+D1+D2):
- API spend: ~$15
- Wall clock: 9h 30m of structured overnight work
- By 8 AM: Phase 3-Opus full done, Phase 3-Sonnet full done, Gemma at n=200 done, 4-metric panel implemented + run on all logs, all figures generated, paper + README updated with final results, ready for final-week polishing.

If anything is contested or you'd prefer different tradeoffs, let me know which decision points need different answers and I'll adjust.
