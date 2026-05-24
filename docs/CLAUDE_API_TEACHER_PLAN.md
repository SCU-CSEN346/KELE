# Claude-API teacher experiments — pricing discovery + mission plan

**Author:** Claude Opus 4.7 (1M ctx) for Max
**Date:** 2026-05-21
**Status:** Plan-of-record (now mostly retrospective). Predecessor: `docs/PROMPT_ENGINEERING_PLAN.md` (Phase 1 tournament complete, Phase 2 composition pending at the time of writing). This plan opened a parallel front: swap the local Gemma 4 31B teacher for a frontier Claude teacher via the Anthropic API.

> **⚠ Status header (added 2026-05-23, audit pass).** All five missions in this plan have landed:
>
> - **Mission 2 (Sonnet n=50)** and **Mission 3 (Opus n=50)** — done; the n=50 frontier triple-arch landed in `results/bert-claude-{opus,sonnet}-*-n50/` and is documented in `README.md` §"Phase 2 — Frontier-teacher comparison" and `docs/EXPERIMENT_LOG.md` 2026-05-21 entry.
> - **Mission 4 (full n=681 promotion)** — done for both Sonnet + top-3 and Opus + top-3. Results in `results/bert-claude-{sonnet,opus}-top3-n681/`. Per the unified-metric leaderboard (added 2026-05-23, see `docs/UNIFIED_RANKING.md`), `bert × Claude-Sonnet · top3 · n=681` is the current **frontier ceiling** at unified 70.06, and `bert × Claude-Opus · top3 · n=681` is at 69.37.
> - **Mission 5 (composition × Claude)** — implicitly covered by the top-3 stack runs.
> - **The local–frontier parity finding (1.12 pt gap on unified)** that this Claude-teacher work helped establish is now documented in paper §`sec:unified-ranking-parity`, `docs/UNIFIED_RANKING.md`, and `memory/local_frontier_parity_2026_05_23.md`.
>
> Pricing numbers below were accurate at write time and are preserved as a reference for future API budgeting.

## TL;DR

A full n=681 evaluation with **Architecture A** (BERT consultant + Claude teacher + 10-shot exemplars) costs an estimated:

| Configuration | Sonnet 4.6 | Opus 4.6 |
|---|---:|---:|
| No caching, real-time | $13.75 | $22.91 |
| Batch API (50% off) | $6.87 | $11.46 |
| **1h prompt cache (recommended)** | **$5.01** | **$8.35** |
| Cache + batch combined | ~$2.50 | ~$4.20 |

For the price of a sandwich we can test whether a frontier teacher cracks the c-stage routing ceiling that BERT+Gemma still bottoms out at (30.31%).

## Pricing discovery — the headline surprise

**Claude Opus 4.6 is now $5 input / $25 output per MTok — not the legacy $15/$75.** The price drop landed with the Opus 4.5 generation and persists through Opus 4.6/4.7. This makes Opus only **1.67× the cost of Sonnet**, not 5× as in the older Opus 4.1 era.

Canonical rates (Anthropic API, verified 2026-05-21 against `platform.claude.com/docs/en/about-claude/pricing`):

| Model | Base input | 5m cache write | 1h cache write | Cache read | Output |
|---|---:|---:|---:|---:|---:|
| **Sonnet 4.6** | $3.00 | $3.75 | $6.00 | $0.30 | $15.00 |
| **Opus 4.6** | $5.00 | $6.25 | $10.00 | $0.50 | $25.00 |
| Haiku 4.5 (reference) | $1.00 | $1.25 | $2.00 | $0.10 | $5.00 |

**Batch API** = 50% off both input and output, asynchronous, no caching multiplier interaction needed.
**1h prompt cache** = 2× base input on first write, then 0.1× on reads. Break-even is two reads.
**Long context** = full 1M token window available on Sonnet 4.6 + Opus 4.6 at standard rates.

## Workload sizing

Measurements taken on the locked headline architecture (`BERT consultant + Gemma teacher + 10-shot exemplars`), tokenized with `bert-base-chinese` then scaled by 0.7× as a Claude-BPE proxy (Claude's tokenizer compresses Chinese better than char-level BERT).

**Dataset:** SocratDataset 90/10 split → **681 test dialogues**, mean 6.32 turns per dialogue, **~4,304 teacher calls per full eval**.

**Per-turn prompt composition (Architecture A):**

| Block | BERT-CN tokens | ≈ Claude tokens | Cacheable? |
|---|---:|---:|---|
| Teacher system prompt (persona + rules) | 313 | ~220 | ✅ static |
| 10-shot stage-balanced exemplar block | 763 | ~535 | ✅ static |
| Conversation history (running avg) | 200 | ~140 | ❌ grows |
| Consultant handoff (state + action) | 40 | ~30 | ❌ per-turn |
| Current student input | 30 | ~21 | ❌ per-turn |
| **Total input per call** | 1,346 | **~946** | — |
| Teacher output (Chinese response) | 35 | ~25 | — |

**Static cacheable prefix: ~755 Claude tokens.** That's the same across all 681 dialogues — a perfect cache target. With 1h TTL and ~50 dialogues/hour throughput, the cache stays warm with only ~4 writes per full run.

## Cost tables — full n=681 evaluation

### Architecture A: BERT-consultant + Claude teacher + 10-shot
*Drop-in for the locked headline. Apples-to-apples teacher swap (Gemma 4 31B → Claude).*

| Pricing tier | Sonnet 4.6 | Opus 4.6 |
|---|---:|---:|
| Standard real-time | $13.75 | $22.91 |
| Batch API | $6.87 | $11.46 |
| **1h prompt cache** | **$5.01** | **$8.35** |

### Architecture B: Standalone (Claude does both consultant + teacher calls)
*Replicates the GPT-4o baseline setup. Two LLM calls per turn, including the massive 34-state consultant prompt.*

| Pricing tier | Sonnet 4.6 | Opus 4.6 |
|---|---:|---:|
| Standard real-time | $36.75 | $61.25 |
| Batch API | $18.37 | $30.62 |

### Architecture C: Unified single-call (fused consultant + teacher JSON)
*One structured-output call per turn. State classification + response in one shot.*

| Pricing tier | Sonnet 4.6 | Opus 4.6 |
|---|---:|---:|
| Standard real-time | $34.31 | $57.18 |
| Batch API | $17.15 | $28.59 |
| 1h prompt cache | $10.84 | $18.07 |

### Caveats

- Token estimate is ±25% — the 0.7× BERT→Claude scaling is a heuristic, not a measurement.
- The `usage.input_tokens` / `usage.output_tokens` fields in API responses are authoritative — read them on the first calls and recalibrate.
- Rate-limit tier matters at full-eval scale: Tier 1 may bottleneck. Plan: serial through batch API if real-time hits caps.

## Mission plan

### Mission 1 — Token-calibration probe (~$0.05, ~5 min)

**Goal:** Validate the token estimate before committing to a full run.

**Procedure:**
1. Configure `KELE_TEACHER_BASE_URL=https://api.anthropic.com/v1/` and `KELE_TEACHER_MODEL=claude-sonnet-4-6` via env vars / config layer.
2. Run `bash scripts/eval_bert_gemma_fewshot10_full.sh` with `--limit 5` or equivalent smoke flag.
3. Capture actual `usage.input_tokens` / `usage.output_tokens` from each response.
4. Compute realized cost-per-dialogue; scale to 681 to confirm full-run estimate.

**Decision rule:** If realized cost is within ±30% of estimate, proceed to Mission 2. If wildly off, recompute and re-plan.

### Mission 2 — Sonnet 4.6 at n=50 (~$0.40 cached, ~10 min)

**Goal:** First single-cell tournament-style result for a Claude teacher. Apples-to-apples vs the locked Gemma headline at n=50 (51.06% / 38.53 R-1, composite 70.33).

**Configuration:**
```bash
KELE_TEACHER_BASE_URL=https://api.anthropic.com/v1/ \
KELE_TEACHER_MODEL=claude-sonnet-4-6 \
KELE_FEW_SHOT_TEACHER=1 \
KELE_FEW_SHOT_N=10 \
bash scripts/eval_bert_gemma_fewshot10_full.sh --n 50
```

**Result tracking:** Stage in `results/sonnet46-bert-fewshot10-n50/` mirroring the tournament-cell directory layout.

**Decision rule:**
- **Composite ≥ 70.33** (matches locked headline) → proceed to Mission 3.
- **Composite ≥ 72.5** (matches Phase 2 promotion threshold) → schedule full n=681 run immediately, Sonnet becomes the Phase 3 candidate.
- **Composite < 68** → Claude teacher loses; don't escalate.

### Mission 3 — Opus 4.6 at n=50 (~$0.60 cached, ~12 min)

**Goal:** Test whether the frontier model justifies its 1.67× premium for this task.

Same config as Mission 2, with `KELE_TEACHER_MODEL=claude-opus-4-6`.

**Decision rule:** Promote to full n=681 if composite beats both:
- The locked Gemma headline (70.33)
- The Sonnet 4.6 result from Mission 2

If Opus only marginally beats Sonnet (< +1.0 composite), Sonnet wins on cost-efficiency for the headline.

### Mission 4 — Full n=681 promotion run

**Trigger:** Whichever Claude model wins Mission 2/3 by ≥ +2.0 composite over Gemma.

**Configuration:** Same as the winning n=50 cell, with `--n 681` (or omit `--n`). Use 1h prompt caching. Optionally use batch API for half-price if next-day delivery is acceptable.

**Expected cost:** $5–$8 (Sonnet cached) or $8–$13 (Opus cached).

**Decision rule for paper headline:**
- **State acc ≥ 50.00** AND **R-1 ≥ 37.00** → new locked headline. Rewrite §4.8.1 / Abstract / Conclusion to feature Claude-teacher result. Document as a "frontier teacher + BERT consultant" finding.
- Anything below → keep Gemma headline; document the Claude attempt as a teacher-comparison ablation in §4.8 (the per-state stage routing analysis from Phase 0.5 already has the template).

### Mission 5 (optional) — Phase 2 composition × Claude teacher

**Only if Mission 4 promotes a Claude headline.** Re-run the Phase 2 composed stacks (Composed-A: #1+#9+#5, Composed-B: +#7 CoT) but with the winning Claude teacher instead of Gemma. Same n=50 → n=681 decision tree.

This is the upper-bound combined optimization: BERT consultant × frontier teacher × tournament-winning prompt stack. If composite ≥ 75 at n=50, run full immediately.

## Implementation notes

### Wiring Claude into the existing config

The current code uses the OpenAI Python SDK with `base_url` override (see `src/project/config.py` and the teacher client init in `socratic_teaching_system.py:393+`). The Anthropic API is OpenAI-compatible via `https://api.anthropic.com/v1/`, so the swap is just env vars — no refactor needed.

If we hit Anthropic-specific features (cache_control, structured outputs differently), prefer the native `anthropic` SDK in a thin teacher-side adapter. Don't refactor preemptively.

### Prompt caching activation

Add `cache_control={"type": "ephemeral"}` to the static portion of the messages array (system prompt + few-shot block). The Anthropic-compatible OpenAI client may or may not pass this through cleanly — verify with the first probe response's `usage.cache_creation_input_tokens` and `usage.cache_read_input_tokens` fields.

If the OpenAI client strips cache_control, switch the teacher client only to the native `anthropic` SDK. Consultant path (BERT) is unaffected.

### Rate limits

At ~50 dialogues/hour and ~6 calls each = ~300 calls/hour. Tier 1 = 50 RPM / 40K TPM Sonnet limit — sufficient. If we get throttled, drop to batch API or add 200ms inter-request delay.

### Cost-control safety net

Set `ANTHROPIC_API_COST_LIMIT` as an env var the orchestrator scripts check before each batch of calls. Default to $20 for a full run. Abort if projected cost exceeds limit based on running token usage.

## Artifact pointers

- Pricing reference (canonical): https://platform.claude.com/docs/en/about-claude/pricing
- Token-sizing probe used to build this plan: BERT-Chinese tokenizer on the live teacher prompt blocks from `socratic_teaching_unified.py:_build_few_shot_block_n(10)` and the consultant system prompt in `socratic_teaching_system.py`
- Existing locked-headline orchestrator (modify for Claude teacher): `scripts/eval_bert_gemma_fewshot10_full.sh`
- Phase 1 tournament results (apples-to-apples comparison baseline): `results/tournament-cell-1-length_budget/` through `tournament-cell-10-compressed_history/`

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Token estimate ±25% wrong → cost overrun | Mission 1 probe at n=5 first; validate before n=50 |
| OpenAI client doesn't pass `cache_control` | Verify on first response; switch to native `anthropic` SDK if needed |
| Rate limit hit during full run | Switch to batch API for half-price + async delivery |
| Claude refuses Chinese pedagogy prompt for safety reasons | Unlikely — content is benign. If it happens, log and reduce scope |
| Composite gain ≤ +1.0 — not worth paper rewrite | Document as ablation in §4.8, don't change headline |

## Why this is high-leverage

1. **Architecture A is the locked headline.** Swapping the teacher is a single env-var change, not a research pivot. Risk-bounded.
2. **The c-stage ceiling is the open scientific question.** BERT+Gemma still only routes 30.31% on c-stage states. If Sonnet/Opus's stronger reasoning lifts this, it's a publishable finding about consultant-teacher decomposition: "the consultant routing ceiling is teacher-bound, not BERT-bound."
3. **The cost is rounding error.** $5–$10 for a full run that takes hours of local compute on the 5090. We can run both Sonnet AND Opus full evals for under $20 combined.
4. **The window is open until 2026-06-04** (final paper deadline). Plenty of time to escalate or de-escalate based on Mission 2/3 signal.

**The cyber gods built Claude. Let's see what happens when Claude teaches.**
