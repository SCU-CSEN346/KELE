# Experiment Log

Engineering decisions, what we've tried, and what's next. Each entry is dated and time-ordered (newest at top).

---

## 2026-05-19 PM — Phase 1 prompt-utilization tournament COMPLETE ✅

**Ran:** 10 single-utilization cells × n=50 = 500 dialogues against the locked BERT + Gemma 4 31B + 10-shot baseline. Three sub-runs across the day (a session crash mid-cell-10 forced a clean restart of cells 10/7/8). Total wall clock ~6h 09m.

### Final leaderboard (composite = state + 0.5 × ROUGE-1)

| Rank | Cell | Utilization | State | R-1 | R-2 | B-4 | Composite | Δ vs base |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| — | 0 | baseline (BERT+Gemma+10-shot) | 51.06 | 38.53 | 16.93 | 9.68 | 70.33 | — |
| 1 | **1** | **length_budget** | **51.96** | **39.91** | 17.93 | 12.37 | **71.91** | **+1.58** |
| 2 | **9** | persona | 51.27 | 39.60 | 17.89 | 12.38 | 71.07 | +0.74 |
| 3 | 5 | negative_exemplars | 50.71 | 39.82 | 18.05 | 10.88 | 70.62 | +0.29 |
| 4 | 4 | per_state_exemplars | 50.88 | 39.42 | 17.78 | 10.90 | 70.59 | +0.26 |
| 5 | 7 | cot_scaffold | 50.89 | 38.98 | 17.13 | 9.55 | 70.38 | +0.05 |
| 6 | 3 | style_matched_exemplars | 46.72 | **42.17** | **20.10** | 12.04 | 67.81 | −2.52 |
| 7 | 10 | compressed_history | 47.50 | 38.79 | 17.07 | 9.82 | 66.89 | −3.44 |
| 8 | 8 | nbest_rerank | 47.33 | 38.27 | 16.48 | 10.02 | 66.47 | −3.86 |
| 9 | 6 | format_retry | 46.45 | 39.60 | 17.82 | 10.55 | 66.25 | −4.08 |
| 10 | 2 | lexical_priors | 43.89 | 39.12 | 17.74 | 9.88 | 63.45 | −6.88 |

### Headlines

1. **Length-budget (#1) is the cleanest result.** +0.9 state, +1.38 R-1, +1.58 composite. Confirms the §2 hypothesis: open-weight teachers overshoot stage-typical character lengths by 1.5–3×, and forcing per-stage budgets simultaneously lifts surface mimicry and stage routing.
2. **The expensive multi-call cells underperformed.** {#6 format-retry, #7 CoT, #8 N-best} is the mutex group; only #7 cleared baseline (by noise). #8 was the worst multi-call result at 3× inference cost. Verdict: hidden-reasoning and self-critique do not lift this composition layer.
3. **Style-matched exemplars (#3) bought R-1 (42.17, only cell over 40) with 4.3 pts of state acc.** Surface-form optimization is anti-correlated with stage routing.
4. **Per-state routing (#4) did not deliver the Phase 0.5-predicted lift.** Dense Gemma teacher appears to already absorb the BERT state-name signal through the prompt itself; explicit retrieval is redundant.
5. **Lexical-prior priming (#2) is actively harmful** (−6.88 composite). Listing preferred opener 4-grams biases the teacher away from correct state-conditional content.

### Phase 2 plan (next)

- **Composed-A:** `KELE_STAGE_LENGTH_BUDGET=1 KELE_TEACHER_PERSONA=1 KELE_NEGATIVE_EXEMPLARS=1` on top of the BERT+Gemma+10-shot baseline. Pure prompt-string changes, no extra inference. Run at n=50.
- **Composed-B:** Composed-A + `KELE_TEACHER_COT=1` (the only above-baseline member of the mutex group). 2× inference cost.
- Pick the higher composite at n=50; if composite ≥ 72.5, promote to Phase 3 at n=681 as the new headline candidate.

### Crash recovery note

Cell 10 (compressed_history) crashed mid-run during the second sub-run (~13:16 PDT). No data corruption to completed cells — the wrapper's per-cell `metrics_summary.json` resume gate behaved correctly. Cell 10's partial 8/50 dialogues were wiped before the recovery run to avoid mixing; the recovery sub-run completed 10 → 7 → 8 cleanly in 182m.

### Artifact pointers

- Per-cell results: `results/tournament-cell-{1..10}-*/`
- Leaderboard aggregator: `scripts/aggregate_tournament_leaderboard.py`
- Tournament wrapper: `scripts/eval_prompt_tournament.sh`
- Run logs: `logs/tournament_2026-05-19T{16-21-39,18-26-00,23-17-13}.log`
- Plan: `docs/PROMPT_ENGINEERING_PLAN.md` §3 (utilization definitions) and Phase 1 section

---

## 2026-04-14 — Baseline run #2 COMPLETE (GPT-4o consultant) ✅

**Ran:** 10:14 → 14:48 (4h 34min, 274 min). 681/681 dialogues, 4294 turns, **zero errored dialogues**. 108 rate-limit retries handled gracefully by backoff.

### Final metrics (results/baseline/metrics_summary.json)

| Metric | Paper SocratTeachLLM | Paper GPT-4o baseline | **Our run** | vs paper GPT-4o |
|---|---|---|---|---|
| ROUGE-1 | 57.4 | 48.25 | **44.61** | -3.6 |
| ROUGE-2 | 33.63 | 22.55 | **26.04** | **+3.49** ✓ |
| ROUGE-L | 50.77 | 38.27 | **38.02** | -0.25 (tied) |
| BLEU-4 | 41.96 | 29.93 | **19.60** | -10.3 |
| State acc | — | — | 25.94% | (our metric; not in paper) |
| Stage a / b / c / d / e | — | — | 95.15 / 36.93 / 4.70 / 5.04 / 11.92 | |

**Headline:** Clean reproduction. We **beat the paper's GPT-4o-as-teacher baseline on ROUGE-2** and match it on ROUGE-L. Below on BLEU-4 and ROUGE-1 vs SocratTeachLLM — the BLEU-4 gap (19.6 vs 41.96) is the biggest, likely a generation-params mismatch (paper doesn't specify temperature / max_tokens).

### OpenAI spend (gpt-4o-2024-11-20 consultant)

| | Value |
|---|---|
| **Total cost** | **$17.49** |
| Input tokens | 8,244,910 |
| Output tokens | 371,055 |
| Cost per dialogue | $0.0257 |
| Cost per turn | $0.00407 |

At list pricing ($2.50/1M input, $10/1M output) this would have been $24.32 — prompt caching on the ~2,800-token system prompt saved **~$6.80 (~28%)**. Scaling implication: each future full eval run costs ≈ $17-18. A 3-experiment campaign (baseline + Gemma-4 + BERT-consultant improvement) ≈ $50-60 in OpenAI spend.

### Smoke test results (20 dialogues, 117 turns, gpt-4o)

| Metric | Value |
|---|---|
| ROUGE-1 / 2 / L | 45.73 / 25.76 / 38.28 |
| BLEU-4 | 18.65 |
| State acc (overall) | 29.06% |
| Stage a / b / c / d / e | 100 / 45.5 / 5.3 / 0 / 11.1 |

All the metric-pipeline fixes validated end-to-end. State accuracy jumped from 1.64% (Qwen3.5-2B) → 29% (gpt-4o).

### The consultant journey today — what we tried and why

| Consultant | Outcome | Reason |
|---|---|---|
| Qwen3.5-2B (local) | State acc 1.64% | Too weak for 30-state schema — emitted bare integers instead of `"a1"`/`"b4"` |
| Qwen3.5-4B (local) | OOM | Teacher (19 GB) + 4B weights (8 GB) + KV cache exceeds 32 GB VRAM |
| gpt-4o-mini (API) | State acc 6.56% | Even mini is too weak; Stage a dropped from 100% to 30% |
| **gpt-4o-2024-11-20 (API)** | **State acc 29% smoke** | **Going with this.** Matches the paper's original setup (they used GPT-4o). |

### Added: retry-with-backoff

`references/KELE/original_CN/consultant_teacher_socratic_teaching_system_CN.py` now retries 429s up to 6 times with exponential backoff (honoring `Retry-After` header when present). Tier 1 TPM cap of 30k was dropping turns; retries fix this cleanly at the cost of slower throughput.

### Config

- `configs/baseline.env`: consultant → `https://api.openai.com/v1`, `gpt-4o-2024-11-20`
- `.env`: `CONSULTANT_API_KEY=sk-...` (gitignored, OpenAI key)
- `src/project/config.py`: load experiment config first, then `.env` (experiment wins; `.env` fills in secrets)
- Teacher vLLM still local on port 8001 at 0.60 util (no change)

---

## 2026-04-14 — Baseline run #1 post-mortem + fixes

**Run:** `results/baseline/` (2026-04-13 23:32 → 2026-04-14 04:31, 681/681 dialogues, ~5h)

**Metrics (before fixes):**
| Metric | Value | Notes |
|---|---|---|
| ROUGE-1 / 2 / L | 0.29 / 0.08 / 0.28 | raw fmeasure — effectively 0 (English vs Chinese) |
| BLEU-4 | 0.0 | zero n-gram overlap, plus wrong tokenizer |
| State acc (overall) | 15.08% | stages b/c/d near-zero |

### Problems identified

1. **Language mismatch (critical).** SocratDataset is Chinese; ground-truth teacher turns are Chinese. The KELE system prompts in `references/KELE/consultant_teacher_socratic_teaching_system.py` had been translated to English, so SocratTeachLLM replied in English. Zero overlap with references → BLEU/ROUGE collapse.
2. **sacrebleu tokenizer.** `compute_bleu` used the default `13a` (English) tokenizer. For Chinese we need `tokenize="zh"`.
3. **Consultant context overflow.** 111 consultant calls (2.8% of 3978 turns) hit the 4096-token limit on Qwen3.5-2B. Those turns fell back to "stay in current state", further depressing state accuracy.

### Fixes applied (2026-04-14)

- [x] `src/project/kele.py` — import from `references/KELE/original_CN/consultant_teacher_socratic_teaching_system_CN.py` (Chinese system prompts).
- [x] `src/project/metrics.py` — `BLEU(effective_order=True, tokenize="zh")`.
- [x] `scripts/serve_consultant.sh` — `--max-model-len 4096 → 8192`. See Option A below.

### Next

Restart consultant vLLM with the new `--max-model-len`, then rerun baseline. Expect large jumps in BLEU-4 and ROUGE once teacher outputs Chinese; state accuracy should also rise as the consultant no longer falls back on truncated turns.

---

## Decision log — Consultant context window

**Context:** Qwen3.5-2B consultant is hitting 4096-token limits on long dialogues. System prompt alone is ~2,800 tokens (stage rules + state tables) before history/input.

### Options considered

| Option | Description | Effort | Risk | Status |
|---|---|---|---|---|
| **A** | Bump `--max-model-len` from 4096 → 8192 on consultant vLLM | 1-line change in `scripts/serve_consultant.sh` | Low — Qwen3.5-2B native context is 32k; GPU has headroom at 0.32 util | **In progress (2026-04-14)** |
| B | Swap consultant to Qwen2.5-7B-Instruct (32k native context, stronger reasoning) | ~1h — download + config + rerun | Medium — more VRAM, may compete with teacher on 5090 | Not tried |
| C | Truncate / summarize history in `get_full_formatted_history` | Medium | High — undermines stage-round tracking that the consultant depends on | Rejected |

**Plan:** Try A first. If state accuracy is still low after the language fix + A, move to B (2B consultant may also simply be too weak — the paper used GPT-4o).

---

## Budget — LLM calls for the full campaign

- Each turn: 2 LLM calls (consultant + teacher)
- 3,978 turns/run × 2 ≈ **~8,000 calls per full eval run**
- Planned runs: baseline (SocratTeachLLM) + Gemma-4 extension + BERT-consultant improvement = 3 minimum
- Reserve 1-2 reruns per experiment → **~30-40k calls total**
- All local on 5090 → ~5h wall-clock per run → **~20-30 GPU-hours** for the campaign
