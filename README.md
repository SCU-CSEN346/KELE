# csen-346

[![CI](https://github.com/ulises-c/csen-346/actions/workflows/ci.yml/badge.svg)](https://github.com/ulises-c/csen-346/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ulises-c/csen-346/graph/badge.svg)](https://codecov.io/gh/ulises-c/csen-346)

Natural Language Processing — CSEN 346, Santa Clara University.

This project reproduces and extends **KELE**, a multi-agent framework for structured Socratic teaching with LLMs, and contributes along three axes: (1) a fusion structured-output architecture that collapses the two-agent stack onto a single open-weight backbone; (2) replacing the LLM consultant with a 24M-parameter BERT classifier that routes cognitive state with surgical precision; and (3) **a methodological critique of the published benchmark itself, showing that surface-form metrics (ROUGE/BLEU) systematically reward training-data memorization over teaching capability** — with a proposed four-metric replacement panel.

## Two leaderboards, one inversion

When the same 10 configurations are ranked by the KELE-paper-style surface-form metrics vs. by pedagogical state accuracy, the orderings **invert**. The model the paper celebrates (SocratTeachLLM, a 9B GLM-4 fine-tune from 2024) tops the surface-form ranking and *bottom-of-the-table* on state accuracy. Our open-weight system (Gemma + top-3 stack) sits mid-pack on surface form and **first** on pedagogical correctness:

| | Surface-form ranking (R-1+R-2+BLEU-4) | Pedagogical ranking (state acc) |
|---:|---|---|
| 🥇 #1 | GPT-4o + SocratTeachLLM (sum 90.25) | **Gemma 4 31B + top-3 (50.72%)** |
| 🥈 #2 | Opus 4.6 + top-3 (sum 79.42) | Opus 4.6 + top-3 (49.82%) |
| 🥉 #3 | Sonnet 4.6 + top-3 (sum 77.87) | Sonnet 4.6 + top-3 (48.75%) |
| #4 | Gemma 4 31B + top-3 (sum 72.64) | Qwen 35B-A3B + top-3 (48.19%) |
| #5 | Sonnet 4.6 + 10-shot only (sum 69.23) | Gemma 4 31B + 10-shot (48.15%) |
| ... | ... | ... |
| **🚨 #LAST** | Opus 4.6 raw (sum 37.58) | **GPT-4o + SocratTeachLLM (25.94%)** |

**Same nine configurations. Opposite rankings.** Frontier models with prompt scaffolding (Opus, Sonnet) should top a fair benchmark — they do top the pedagogical ranking but lose surface-form to a 9B specialist. The gap widens with n-gram length (R-1 +1.59 → R-2 +4.92 → BLEU-4 +4.07), the strongest possible memorization signature. A 9B fine-tune from 2024 lexically out-mimicking the ground truth more than Opus 4.6 with carefully tuned prompts is statistically inconsistent with normal training conditions. Full critique + four-metric replacement proposal in [`docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md`](docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md).

> **Latest (2026-05-23) — unified ranking + local–frontier parity finding.** We've since collapsed the four memorization-resistant metrics into a single defensible paper headline: `unified = 0.5 × stage_balanced + 0.5 × (judge × 10)` (see [`docs/UNIFIED_RANKING.md`](docs/UNIFIED_RANKING.md)). Under this metric the **best honest open-weight cell** (`qwen3.5 × Gemma-31B · fewshot10 · n=50`, unified **68.94**) sits **1.12 points** behind the best frontier configuration (`bert × Claude-Sonnet · top3 · n=681`, unified **70.06**) on a [0, 100] scale — **head-to-head parity** between a prompt-engineered open-weight teacher and frontier proprietary models on a memorization-resistant evaluation. The legacy locked headline (`bert × Gemma-31B · fewshot10 · n=681`, unified **68.65**) sits 1.41 points below the frontier ceiling at full sample size, with ~1 of those points being the pre-fix BERT input-format artifact (asymmetric duplication bug, patched in commit `3d68d4a`; see paper §`sec:unified-ranking-parity` and [`docs/UNIFIED_RANKING.md`](docs/UNIFIED_RANKING.md)). Full 25-config master ranked list at `results/_orchestrator_logs/backtest_stage_balanced_latest.md`. Cell-label format per [`docs/NAMING_CONVENTION.md`](docs/NAMING_CONVENTION.md).

Our **locked, full-scale headline** as of 2026-05-18 is the **BERT + Gemma 4 31B + 10-shot integration**: **+22.21 point absolute lift** in overall state accuracy over the GPT-4o + SocratTeachLLM baseline on the full 681-dialogue test split (n=681, **48.15% state acc / 36.78 ROUGE-1**, **unified 68.65**), a Pareto win over the prior A3B locked headline on both axes (+9.45 state, +6.15 R-1). The new Phase 2 winner (Gemma 4 31B + 10-shot + top-3 prompt stack at n=50, **50.72% / 41.13 R-1 / composite 71.28 / unified 70.08**) is the Phase 3 promotion candidate to n=681. Running entirely on a single 32 GB consumer GPU at zero per-run API cost. **Standalone Gemma 4 31B fusion at n=681 underperformed** (31.39% / 27.27 R-1, driven by a 21% schema-fallback rate vs A3B's 0.91%); the BERT-consultant integration removes the schema-fallback dependency entirely by routing state through a deterministic 24M-param classifier — see §4.6 and §4.8.1 of the paper for the methodological finding. The locked headline was produced under the pre-fix consultant input format; the post-fix `bert-fixed × Gemma-31B · fewshot10 · n=50` cell scores unified 67.65 — about 1 point below the locked headline on same-teacher comparison, which is the asymmetric measurement artifact (BERT-class consultants gained ~1–2 pp from the duplicated current utterance; qwen3.5-LoRA classifiers lost ~2–3 pp).

- **Paper we reproduce:** Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models", *Findings of EMNLP 2025* — [aclanthology.org/2025.findings-emnlp.888](https://aclanthology.org/2025.findings-emnlp.888/)
- **Original repository:** https://github.com/yuanpan1020/KELE
- **Our paper draft:** [`deliverables/overleaf/latex/acl_latex.tex`](deliverables/overleaf/latex/acl_latex.tex)
- **SocratTeachLLM (our HF mirror):** https://huggingface.co/ulises-c/SocratTeachLLM
- **SocratDataset (Chinese):** https://huggingface.co/datasets/ulises-c/SocratDataset
- **SocratDataset-EN (English):** https://huggingface.co/datasets/ulises-c/SocratDataset-EN

## Master leaderboard — every variant we've measured

We've evaluated **131 model variants** across this project — 8-cell cross-teacher grids, prompt-engineering tournaments, multiple full n=681 runs, frontier-model comparisons, bilingual probes, and contamination probes. The table below is the full sorted master ranking. Auto-regenerated by `scripts/backtest_stage_balanced.py`; underlying data is the per-config `metrics_summary.json` + `judge_summary.json` files in `results/`.

**Headline metric** (`unified = 0.5 × stage_balanced + 0.5 × (judge × 10)`, defined in [`docs/UNIFIED_RANKING.md`](docs/UNIFIED_RANKING.md)) is shown for the 34 configs we've LLM-judged. The remaining 97 configs are sorted by stage_balanced alone (the metric we recommend over the original macro state-acc; see [`docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md`](docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md) Proposal 7).

**Legend:** 🏆 = locked paper headline · 🥇🥈🥉 = top 3 by unified score · ⚠️ = SocratTeachLLM-based; contamination-driven scores, see [`docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md`](docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md) · Cell labels follow [`docs/NAMING_CONVENTION.md`](docs/NAMING_CONVENTION.md).

### Unified-ranked tier (34 cells with both stage_bal AND LLM-judge)

| u# | Cell | n_turns | **unified** | stage_bal | judge | macro | R-1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 🥇 1 | `bert × Gemma-31B · composed · top3 · n=50` | 278 | **70.08** | 58.48 | 8.17 | 50.72 | 41.13 |
| 🥈 2 | `bert × Claude-Sonnet · top3 · n=681` | 3840 | **70.06** | 58.17 | 8.19 | 49.97 | 41.93 |
| 🥉 3 | `bert × Claude-Opus · fewshot10 · n=50` | 271 | **69.79** | 58.73 | 8.08 | 49.82 | 42.77 |
| 4 | `bert × Claude-Opus · top3 · n=681` | 3794 | **69.37** | 58.63 | 8.01 | 49.31 | 41.63 |
| 5 | `bert × Claude-Sonnet · fewshot10 · n=50` | 281 | **69.16** | 57.18 | 8.11 | 48.75 | 43.02 |
| 6 | `qwen3.5 × Gemma-31B · fewshot10 · n=50` ← honest cross-teacher winner | 285 | **68.94** | 56.13 | 8.18 | 51.58 | 38.76 |
| 🏆 7 | `bert × Gemma-31B · fewshot10 · n=681` ← **LOCKED HEADLINE** | 3834 | **68.65** | 55.42 | 8.19 | 48.15 | 36.78 |
| ⚠️ 8 | `qwen3.5 × SocratTeachLLM · fewshot10 · n=50` | 288 | **68.21** | 63.40 | 7.30 | 58.33 | 48.07 |
| 9 | `qwen3.5 × Qwen-27B · phase3 · think · n=100 · seed=42` | 570 | **68.05** | 60.88 | 7.52 | 54.04 | 33.17 |
| 10 | `bert × Claude-Sonnet · fewshot10 · n=50` | 267 | **67.85** | 57.32 | 7.84 | 47.94 | 39.68 |
| 11 | `bert-fixed × Gemma-31B · fewshot10 · n=50` | 283 | **67.65** | 52.73 | 8.26 | 45.94 | 38.69 |
| 12 | `qwen3.5 × Gemma-31B · fewshot10 · EN · RETRY · n=100 · seed=42` ← cross-lingual Stage 1 SUCCESS | 584 | **67.30** | 52.10 | 8.25 | 46.58 | 11.66 |
| 13 | `qwen3.5 × A3B-35B · fewshot10 · n=50` | 288 | **66.91** | 58.62 | 7.52 | 54.86 | 35.67 |
| 14 | `qwen3.5 × Qwen-27B · think · fewshot10 · n=50` | 282 | **66.89** | 58.68 | 7.51 | 53.19 | 35.28 |
| 15 | `qwen3.5 × Gemma-31B · fewshot10 · EN · PARTIAL · n=61` | 351 | **66.66** | 50.84 | 8.25 | 45.01 | 10.88 |
| 16 | `qwen3.5 × Qwen-27B · phase3 · no-think · n=200 · seed=42` | 1172 | **66.55** | 57.45 | 7.56 | 52.13 | 37.11 |
| 17 | `bert × A3B-35B · composed · top3 · n=50` | 276 | **65.79** | 56.73 | 7.49 | 48.19 | 37.64 |
| 18 | `bert-fixed × Qwen-27B · think · fewshot10 · n=50` | 271 | **65.65** | 57.15 | 7.41 | 49.08 | 34.91 |
| 19 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=50` | 291 | **65.54** | 55.45 | 7.56 | 51.89 | 37.38 |
| ⚠️ 20 | `bert-fixed × SocratTeachLLM · fewshot10 · n=50` | 278 | **65.09** | 58.34 | 7.18 | 52.52 | 47.44 |
| 21 | `bert × Claude-Opus · fewshot10 · n=50` | 272 | **64.99** | 55.52 | 7.44 | 47.43 | 32.99 |
| 22 | `bert × A3B-35B · fewshot10 · n=681` | 3762 | **64.47** | 54.72 | 7.42 | 46.57 | 33.27 |
| 23 | `bert-fixed × Qwen-27B · no-think · fewshot10 · n=50` | 286 | **64.25** | 52.62 | 7.59 | 46.85 | 37.81 |
| 24 | `bert-fixed × A3B-35B · fewshot10 · n=50` | 280 | **63.70** | 52.48 | 7.49 | 45.36 | 34.99 |
| 25 | `bert × Claude-Sonnet · raw · n=50` | 260 | **62.91** | 53.32 | 7.25 | 45.00 | 29.10 |
| 26 | `bert × Claude-Opus · top3 · EN · n=50` | 270 | **60.38** | 40.69 | 8.01 | 34.44 | 0.47 |
| ⚠️ 27 | `Claude-Opus × SocratTeachLLM · n=50` | 307 | **59.86** | 41.68 | 7.80 | 38.44 | 47.58 |
| ⚠️ 28 | `bert-fixed × SocratTeachLLM · fewshot10 · EN · n=50` | 273 | **58.22** | 48.97 | 6.75 | 43.22 | 48.07 |
| ⚠️ 29 | `qwen3.5 × SocratTeachLLM · fewshot10 · EN · n=50` | 291 | **57.20** | 48.66 | 6.57 | 43.99 | 46.73 |
| 30 | `bert × Claude-Opus · raw · n=50` | 239 | **55.63** | 43.27 | 6.80 | 39.75 | 23.28 |
| ⚠️ 31 | `qwen3.5 × SocratTeachLLM · CLEANPROBE · fewshot10 · SYNTH · n=50 · seed=42` ← STL on truly unseen data | 211 | **51.29** | 32.86 | 6.97 | 29.38 | 35.72 |
| ⚠️ 32 | `Claude-Opus × SocratTeachLLM · EN · n=50` | 304 | **50.73** | 33.79 | 6.77 | 30.26 | 44.22 |
| ⚠️ 33 | `Claude-Sonnet × SocratTeachLLM · clean · n=50` | 307 | **47.53** | 18.74 | 7.63 | 18.57 | 45.61 |
| ⚠️ 34 | `Claude-Sonnet × SocratTeachLLM · EN · n=50` | 303 | **44.36** | 22.55 | 6.62 | 22.11 | 55.85 |

**Key findings visible at a glance:**

1. **Frontier ≈ open-weight at parity.** Best honest open-weight (`qwen3.5 × Gemma-31B`, #6, unified 68.94) is 1.12 unified points behind the best frontier configuration (`bert × Claude-Sonnet · top3 · n=681`, #2, 70.06). Head-to-head parity on a memorization-resistant evaluation.
2. **Our locked headline (#7, n=681) survives the metric switch** at unified 68.65 — within 0.29 of the n=50 cross-teacher winner.
3. **SocratTeachLLM's #8 ranking is contamination-driven.** On SocratDataset, `qwen3.5 × STL · ZH` lands #1 on stage_balanced (63.40, +4.72 over the next best) — but its score on clean synthetic data (cell #31, generated by Claude Sonnet, demonstrably outside SocratDataset's 90% train) collapses to 32.86 stage_bal / 35.72 R-1, *worse than Gemma 31B's 56.13/38.76 baseline*. STL's apparent excellence is ~12 R-1 points and ~29 state-acc points of memorization, not capability. Full proof in [`docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md`](docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md).
4. **Cross-lingual transfer of the qwen3.5 LoRA classifier works** (cell #12, unified 67.30 on SocratDataset-EN at n=100) — Stage 1 success; canonical-scale promotion queued.

### Stage-balanced ranking — full 131 configs

<details>
<summary>Click to expand the complete 131-row stage-balanced ranking (every model variant we've tested in this project)</summary>

| sb# | Cell | n_turns | macro | **stage_bal** | judge | R-1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `qwen3.5 × SocratTeachLLM · fewshot10 · n=50` | 288 | 58.33 | **63.40** | 7.30 | 48.07 |
| 2 | `qwen3.5 × Qwen-27B · phase3 · think · n=100 · seed=42` | 570 | 54.04 | **60.88** | 7.52 | 33.17 |
| 3 | `bert-consultant-fewshot10-mini` | 135 | 54.07 | **60.66** | — | 33.12 |
| 4 | `bert-consultant-fewshot10-a4b-mini` | 139 | 53.24 | **60.52** | — | 36.26 |
| 5 | `tournament-cell-1-length_budget` | 281 | 51.96 | **59.50** | — | 39.91 |
| 6 | `tournament-cell-7-cot_scaffold` | 281 | 50.89 | **59.02** | — | 38.98 |
| 7 | `bert × Claude-Opus · fewshot10 · n=50` | 271 | 49.82 | **58.73** | 8.08 | 42.77 |
| 8 | `tournament-cell-5-negative_exemplars` | 282 | 50.71 | **58.71** | — | 39.82 |
| 9 | `qwen3.5 × Qwen-27B · think · fewshot10 · n=50` | 282 | 53.19 | **58.68** | 7.51 | 35.28 |
| 10 | `bert × Claude-Opus · top3 · n=681` | 3794 | 49.31 | **58.63** | 8.01 | 41.63 |
| 11 | `qwen3.5 × A3B-35B · fewshot10 · n=50` | 288 | 54.86 | **58.62** | 7.52 | 35.67 |
| 12 | `bert × Gemma-31B · composed · top3 · n=50` | 278 | 50.72 | **58.48** | 8.17 | 41.13 |
| 13 | `bert × Gemma-31B · fewshot10 · n=50` | 284 | 51.06 | **58.47** | — | 38.53 |
| 14 | `bert-fixed × SocratTeachLLM · fewshot10 · n=50` | 278 | 52.52 | **58.34** | 7.18 | 47.44 |
| 15 | `bert × Claude-Sonnet · top3 · n=681` | 3840 | 49.97 | **58.17** | 8.19 | 41.93 |
| 16 | `qwen3.5 × SocratTeachLLM · MEMPROBE · fewshot10 · TRAIN · n=50 · seed=42` | 297 | 55.22 | **57.81** | — | 48.28 |
| 17 | `tournament-cell-9-persona` | 275 | 51.27 | **57.65** | — | 39.60 |
| 18 | `qwen3.5 × Qwen-27B · phase3 · no-think · n=200 · seed=42` | 1172 | 52.13 | **57.45** | 7.56 | 37.11 |
| 19 | `bert × Claude-Sonnet · fewshot10 · n=50` | 267 | 47.94 | **57.32** | 7.84 | 39.68 |
| 20 | `bert × Claude-Sonnet · fewshot10 · n=50` | 281 | 48.75 | **57.18** | 8.11 | 43.02 |
| 21 | `tournament-cell-4-per_state_exemplars` | 285 | 50.88 | **57.18** | — | 39.42 |
| 22 | `bert-fixed × Qwen-27B · think · fewshot10 · n=50` | 271 | 49.08 | **57.15** | 7.41 | 34.91 |
| 23 | `bert-consultant-fewshot10-n50` | 276 | 48.19 | **57.03** | — | 35.57 |
| 24 | `bert × A3B-35B · n=mini` | 128 | 49.22 | **56.95** | — | 26.94 |
| 25 | `bert × Gemma-31B · fewshot10 · n=mini` | 137 | 50.36 | **56.87** | — | 35.93 |
| 26 | `bert × A3B-35B · composed · top3 · n=50` | 276 | 48.19 | **56.73** | 7.49 | 37.64 |
| 27 | `qwen3.5 × Gemma-31B · fewshot10 · n=50` | 285 | 51.58 | **56.13** | 8.18 | 38.76 |
| 28 | `bert-consultant-fewshot10-a4b-n50` | 274 | 48.54 | **56.12** | — | 37.49 |
| 29 | `bert × Claude-Opus · fewshot10 · n=50` | 272 | 47.43 | **55.52** | 7.44 | 32.99 |
| 30 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=50` | 291 | 51.89 | **55.45** | 7.56 | 37.38 |
| 31 | `bert × Gemma-31B · fewshot10 · n=681` | 3834 | 48.15 | **55.42** | 8.19 | 36.78 |
| 32 | `tournament-cell-3-style_matched_exemplars` | 274 | 46.72 | **55.15** | — | 42.17 |
| 33 | `bert × A3B-35B · fewshot10 · n=681` | 3762 | 46.57 | **54.72** | 7.42 | 33.27 |
| 34 | `tournament-cell-8-nbest_rerank` | 281 | 47.33 | **53.61** | — | 38.27 |
| 35 | `bert × Claude-Sonnet · raw · n=50` | 260 | 45.00 | **53.32** | 7.25 | 29.10 |
| 36 | `tournament-cell-10-compressed_history` | 280 | 47.50 | **52.87** | — | 38.79 |
| 37 | `bert-fixed × Gemma-31B · fewshot10 · n=50` | 283 | 45.94 | **52.73** | 8.26 | 38.69 |
| 38 | `bert-consultant-richeval-mini` | 126 | 49.21 | **52.67** | — | 26.53 |
| 39 | `bert-fixed × Qwen-27B · no-think · fewshot10 · n=50` | 286 | 46.85 | **52.62** | 7.59 | 37.81 |
| 40 | `bert-fixed × A3B-35B · fewshot10 · n=50` | 280 | 45.36 | **52.48** | 7.49 | 34.99 |
| 41 | `tournament-cell-6-format_retry` | 282 | 46.45 | **52.36** | — | 39.60 |
| 42 | `bert × A3B-35B · n=50` | 261 | 44.06 | **52.31** | — | 28.36 |
| 43 | `qwen3.5 × Gemma-31B · fewshot10 · EN · RETRY · n=100 · seed=42` | 584 | 46.58 | **52.10** | 8.25 | 11.66 |
| 44 | `tournament-cell-2-lexical_priors` | 262 | 43.89 | **50.97** | — | 39.12 |
| 45 | `qwen3.5 × Gemma-31B · fewshot10 · EN · PARTIAL · n=61` | 351 | 45.01 | **50.84** | 8.25 | 10.88 |
| 46 | `qwen3.5 × Qwen-27B · phase3 · think · BROKEN · n=200 · seed=42` | 1245 | 46.51 | **50.82** | — | 15.07 |
| 47 | `qwen35b-a3b-local-mini-unified-fewshot10` | 147 | 46.94 | **50.73** | — | 34.41 |
| 48 | `bert-v2-consultant-fewshot10-n50` | 287 | 43.90 | **50.60** | — | 37.12 |
| 49 | `qwen27b-local-mini-unified` | 146 | 45.89 | **50.16** | — | 29.36 |
| 50 | `qwen35b-a3b-local-mini-unified-fewshot7` | 148 | 45.27 | **49.78** | — | 35.69 |
| 51 | `qwen35b-a3b-local-n50-unified-fewshot10` | 299 | 44.15 | **49.34** | — | 36.16 |
| 52 | `bert-v2-consultant-fewshot10-mini` | 142 | 44.37 | **49.08** | — | 33.55 |
| 53 | `bert-fixed × SocratTeachLLM · fewshot10 · EN · n=50` | 273 | 43.22 | **48.97** | 6.75 | 48.07 |
| 54 | `qwen3.5 × SocratTeachLLM · fewshot10 · EN · n=50` | 291 | 43.99 | **48.66** | 6.57 | 46.73 |
| 55 | `tournament/round1/qwen27b-q4` | 300 | 43.00 | **48.35** | — | 31.62 |
| 56 | `tournament/round1/qwen27b` | 302 | 42.72 | **48.21** | — | 30.90 |
| 57 | `gemma4-31b-local-mini-unified` | 148 | 41.89 | **46.32** | — | 30.11 |
| 58 | `qwen35b-a3b-local-mini-unified-fewshot` | 148 | 43.24 | **45.74** | — | 33.49 |
| 59 | `tournament/archive/368b6431/round1/gemma4-31b` | 305 | 40.33 | **44.99** | — | 32.88 |
| 60 | `tournament/archive/d9ac39c5/round1/gemma4-31b` | 304 | 41.12 | **44.72** | — | 32.96 |
| 61 | `qwen35b-a3b-local-unified` | 4171 | 38.70 | **44.05** | — | 30.63 |
| 62 | `tournament/archive/d9ac39c5/round1/gemma4-26b-a4b` | 303 | 39.27 | **43.49** | — | 31.60 |
| 63 | `gemma4-26b-a4b-local-mini-unified` | 147 | 38.78 | **43.31** | — | 32.04 |
| 64 | `bert × Claude-Opus · raw · n=50` | 239 | 39.75 | **43.27** | 6.80 | 23.28 |
| 65 | `tournament/archive/497374dd/round1/gemma4-26b-a4b` | 300 | 38.67 | **43.03** | — | 32.19 |
| 66 | `qwen35b-a3b-local-mini-unified-fewshot5` | 146 | 38.36 | **42.85** | — | 34.60 |
| 67 | `tournament/round1/gemma4-31b` | 303 | 38.28 | **42.19** | — | 33.04 |
| 68 | `qwen35b-a3b-local-n50-unified` | 299 | 38.13 | **42.11** | — | 32.87 |
| 69 | `qwen35b-a3b-local-n50-unified-fewshot` | 298 | 37.58 | **41.96** | — | 33.33 |
| 70 | `tournament/round1/gemma4-26b-a4b` | 303 | 37.62 | **41.78** | — | 32.48 |
| 71 | `Claude-Opus × SocratTeachLLM · n=50` | 307 | 38.44 | **41.68** | 7.80 | 47.58 |
| 72 | `tournament/archive/368b6431/round1/gemma4-26b-a4b` | 298 | 36.24 | **40.85** | — | 31.95 |
| 73 | `bert × Claude-Opus · top3 · EN · n=50` | 270 | 34.44 | **40.69** | 8.01 | 0.47 |
| 74 | `qwen35b-a3b-local-mini-unified` | 145 | 35.17 | **39.67** | — | 30.51 |
| 75 | `tournament/archive/497374dd/round1/gemma4-31b` | 304 | 35.86 | **39.58** | — | 33.03 |
| 76 | `tournament/round1/qwen35b-a3b` | 300 | 34.33 | **38.32** | — | 31.28 |
| 77 | `qwen27b-local-mini-unified-nothink` | 147 | 35.37 | **37.79** | — | 31.14 |
| 78 | `tournament/round1/qwen35-9b` | 297 | 32.66 | **36.30** | — | 34.49 |
| 79 | `tournament/archive/368b6431/round1/qwen27b-q4` | 303 | 31.68 | **36.00** | — | 32.84 |
| 80 | `gemma4-31b-local-unified` | 4246 | 31.39 | **35.60** | — | 27.27 |
| 81 | `tournament/archive/d9ac39c5/round1/qwen27b` | 307 | 31.60 | **35.51** | — | 33.33 |
| 82 | `tournament/archive/368b6431/round1/qwen27b` | 305 | 31.48 | **35.37** | — | 33.56 |
| 83 | `Claude-Opus × SocratTeachLLM · EN · n=50` | 304 | 30.26 | **33.79** | 6.77 | 44.22 |
| 84 | `tournament/archive/497374dd/round1/qwen27b-q4` | 300 | 30.67 | **33.13** | — | 33.25 |
| 85 | `qwen3.5 × SocratTeachLLM · CLEANPROBE · fewshot10 · SYNTH · n=50 · seed=42` | 211 | 29.38 | **32.86** | 6.97 | 35.72 |
| 86 | `qwopus35b-a3b-local-mini-unified` | 146 | 30.14 | **32.69** | — | 35.39 |
| 87 | `tournament/archive/497374dd/round1/qwen27b` | 304 | 28.62 | **32.35** | — | 33.42 |
| 88 | `baseline` | 4294 | 25.94 | **30.75** | — | 44.61 |
| 89 | `tournament/archive/d9ac39c5/round1/qwen27b-q4` | 302 | 26.82 | **30.02** | — | 33.32 |
| 90 | `tournament/round1/qwopus35b-a3b` | 305 | 25.90 | **28.88** | — | 35.50 |
| 91 | `tournament/archive/368b6431/round1/qwen3-14b` | 308 | 24.68 | **27.35** | — | 39.33 |
| 92 | `tournament/archive/368b6431/round1/qwen35b-a3b` | 306 | 24.18 | **27.09** | — | 31.69 |
| 93 | `tournament/archive/d9ac39c5/round1/qwen3-14b` | 308 | 23.70 | **25.89** | — | 40.44 |
| 94 | `tournament/archive/497374dd/round1/qwen3-14b` | 307 | 22.48 | **24.88** | — | 39.49 |
| 95 | `tournament/archive/d9ac39c5/round1/qwopus35b-a3b` | 301 | 21.93 | **24.43** | — | 33.38 |
| 96 | `tournament/round1/mistral-24b` | 304 | 21.38 | **23.99** | — | 37.45 |
| 97 | `tournament/archive/d9ac39c5/round1/qwen35b-a3b` | 297 | 21.55 | **23.99** | — | 31.43 |
| 98 | `tournament/archive/497374dd/round1/mistral-24b` | 302 | 21.85 | **23.61** | — | 37.06 |
| 99 | `tournament/archive/368b6431/round1/mistral-24b` | 308 | 20.13 | **23.35** | — | 37.49 |
| 100 | `Claude-Sonnet × SocratTeachLLM · EN · n=50` | 303 | 22.11 | **22.55** | 6.62 | 55.85 |
| 101 | `qwen35b-a3b-local-n50-unified-nothink` | 300 | 19.67 | **22.20** | — | 30.55 |
| 102 | `tournament/archive/d9ac39c5/round1/qwen35-9b` | 305 | 19.02 | **21.88** | — | 28.41 |
| 103 | `tournament/archive/497374dd/round1/qwen35b-a3b` | 304 | 19.74 | **21.86** | — | 31.30 |
| 104 | `wave-2026-04-21T08-59-20-892964` | 4280 | 18.93 | **21.62** | — | 43.72 |
| 105 | `tournament/archive/497374dd/round1/qwopus35b-a3b` | 307 | 18.57 | **21.41** | — | 32.81 |
| 106 | `tournament/archive/368b6431/round1/qwopus35b-a3b` | 307 | 17.92 | **20.73** | — | 33.13 |
| 107 | `tournament/archive/d9ac39c5/round1/gemma3-27b` | 306 | 18.30 | **20.15** | — | 34.35 |
| 108 | `tournament/archive/d9ac39c5/round1/mistral-24b` | 307 | 17.26 | **19.87** | — | 37.48 |
| 109 | `Claude-Sonnet × SocratTeachLLM · clean · n=50` | 307 | 18.57 | **18.74** | 7.63 | 45.61 |
| 110 | `R9700_Mac-M4` | 4262 | 15.16 | **18.41** | — | 43.57 |
| 111 | `tournament/round1/qwen3-14b` | 306 | 17.65 | **18.37** | — | 36.40 |
| 112 | `baseline_run1_en_bug` | 3978 | 15.08 | **17.92** | — | 0.29 |
| 113 | `tournament/archive/497374dd/round1/qwen35-9b` | 307 | 15.64 | **17.68** | — | 28.23 |
| 114 | `tournament/archive/368b6431/round1/gemma3-27b` | 308 | 14.61 | **17.18** | — | 34.86 |
| 115 | `tournament/archive/368b6431/round1/qwen35-9b` | 305 | 13.77 | **15.23** | — | 28.46 |
| 116 | `tournament/round1/gemma3-27b` | 307 | 14.33 | **15.14** | — | 34.61 |
| 117 | `tournament/archive/497374dd/round1/gemma3-27b` | 308 | 12.99 | **15.09** | — | 34.57 |
| 118 | `tournament/archive/368b6431/round1/glm47-23b` | 304 | 13.49 | **15.06** | — | 32.43 |
| 119 | `tournament/archive/497374dd/round1/glm47-23b` | 304 | 13.16 | **14.68** | — | 33.00 |
| 120 | `tournament/archive/368b6431/round1/phi4-14b` | 290 | 12.41 | **13.51** | — | 34.58 |
| 121 | `tournament/archive/d9ac39c5/round1/deepseek-r1-14b` | 307 | 11.40 | **13.43** | — | 34.43 |
| 122 | `tournament/archive/d9ac39c5/round1/phi4-14b` | 308 | 11.04 | **12.53** | — | 35.80 |
| 123 | `tournament/archive/497374dd/round1/phi4-14b` | 301 | 10.63 | **12.24** | — | 35.75 |
| 124 | `tournament/round1/deepseek-r1-14b` | 307 | 10.75 | **12.15** | — | 34.62 |
| 125 | `tournament/archive/368b6431/round1/deepseek-r1-14b` | 306 | 10.13 | **11.05** | — | 35.27 |
| 126 | `tournament/round1/phi4-14b` | 298 | 9.73 | **10.70** | — | 35.64 |
| 127 | `tournament/archive/d9ac39c5/round1/glm47-23b` | 305 | 9.51 | **10.67** | — | 30.90 |
| 128 | `tournament/archive/497374dd/round1/deepseek-r1-14b` | 306 | 8.17 | **9.24** | — | 35.70 |
| 129 | `Claude-Opus × SocratTeachLLM · BROKEN · n=50` | 308 | 0.00 | **0.00** | — | 13.20 |
| 130 | `Claude-Sonnet × SocratTeachLLM · n=50` | 308 | 0.00 | **0.00** | — | 43.88 |
| 131 | `Claude-Sonnet × SocratTeachLLM · BROKEN · n=50` | 308 | 0.00 | **0.00** | — | 13.20 |

</details>

The full backtest snapshot (with per-stage breakdowns + Δrank movers) is at `results/_orchestrator_logs/backtest_stage_balanced_2026_05_23.md`. The Δrank column there shows that **35 of 131 historical configs change rank by ≥3 positions** under stage-balanced macro vs the published frequency-weighted macro — concrete evidence that the closure-dominance fix is load-bearing for the rankings.

## Headline Results

All open-weight runs use a single RTX 5090 (32 GB VRAM) with one model serving both consultant and teacher roles via the fusion architecture, or the BERT consultant + LLM teacher integration (see [Architecture](#architecture)). The GPT-4o baseline uses the canonical KELE two-model stack (SocratTeachLLM teacher + GPT-4o consultant via API).

### Locked full-scale (n=681) results

| Run | n | State acc | Δ vs GPT-4o | ROUGE-1 | BLEU-4 | Wall clock | Fallback | API spend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o + SocratTeachLLM (baseline) | 681 | 25.94% | — | **44.61** | **19.60** | 4h 34m | n/a | $17.49 |
| **🏆 BERT + Gemma 4 31B + 10-shot (LOCKED HEADLINE 2026-05-18)** | **681** | **48.15%** | **+22.21 (1.86×)** | **36.78** | **9.05** | 12h 53m | **n/a (BERT skips consultant)** | **$0** |
| BERT + Qwen 35B-A3B + 10-shot (teacher ablation, 2026-05-19) | 681 | 46.57% | +20.63 (1.79×) | 33.27 | 6.96 | 3h 15m† | n/a (BERT skips consultant) | $0 |
| Qwen 3.6 35B-A3B fusion-think (prior locked headline) | 681 | 38.70% | +12.76 (1.49×) | 30.63 | 5.86 | 16h 29m | 0.91% | $0 |
| Gemma 4 31B fusion standalone (retracted, 2026-05-17) | 681 | 31.39% | +5.45 | 27.27 | 5.50 | 21h 49m | **21.0%** | $0 |

†Wall clock used the new parallel-eval client (`KELE_PARALLEL_WORKERS=4`, ~2× the throughput of the sequential path it replaced). See [Phase 0.5 teacher-ablation note](#phase-05-teacher-choice-ablation-2026-05-19) below.

**Per-stage picture for the locked BERT-integration headline vs the prior A3B locked + the disappointing Gemma standalone** (all vs GPT-4o baseline at n=681):

| Stage | GPT-4o | A3B fusion-think | Gemma standalone | **🏆 BERT+Gemma+10-shot** | Δ vs A3B | Δ vs GPT-4o |
|---|---:|---:|---:|---:|---:|---:|
| a (problem detection) | 95.15% | 91.78% | 78.71% | **99.27%** | +7.49 | +4.12 |
| b (early reasoning) | 36.93% | 39.29% | 33.11% | 23.26% | **−16.03 ⚠** | −13.67 |
| c (22-state induction) | 4.70% | 17.57% | 13.89% | **30.31%** | **+12.74** | **+25.61 (6.4×)** |
| d (resolution) | 5.04% | 14.78% | 14.23% | **41.50%** | **+26.72** | **+36.46 (8.2×)** |
| e (closure) | 11.92% | 56.83% | 38.07% | **82.77%** | **+25.94** | **+70.85 (6.9×)** |
| **Overall** | **25.94%** | **38.70%** | **31.39%** | **48.15%** | **+9.45** | **+22.21 (1.86×)** |

The BERT-integration headline is **a Pareto win over A3B on both axes** (+9.45 state, +6.15 R-1) and posts **massive multipliers on the hard middle/closure stages** (6–8× over GPT-4o on stages c/d/e — precisely where general-purpose LLMs collapse). The one weak stage is **b (early reasoning)**, where we lose 16 points to A3B — a stable property of BERT's stage-b routing distribution that also appeared at $n{=}50$, not a regression introduced at scale.

**Why Gemma standalone collapsed at full scale, and why the BERT-integration rescues it.** The standalone Gemma mini-tier (n=148 turns) predicted +20.77 state-acc; the full-tier (n=4246 turns) realized only +5.45 — a **15-point overshoot** by the smoke+mini average predictor that landed A3B within 0.10 pts. The root cause is in the schema-fallback rate: **Gemma fell back to two-call mode on 21.0% of turns** (890/4246) vs A3B's **0.91%** (38/4171). Gemma's strict-JSON adherence on stage-c-class structured output is dramatically weaker than A3B's. **The BERT-consultant integration removes this dependency entirely**: BERT (24M params, 86.55% stage / 61.64% state on the test split) routes the cognitive state deterministically, leaving Gemma to handle only response generation — a path with no JSON schema. The full-scale BERT+Gemma+10-shot run (this campaign's headline) confirms the hypothesis: the integration lands at 48.15% / 36.78 R-1 at $n{=}681$, validating that **the integration architecture isolates the consultant axis (BERT) from the response-generation axis (LLM teacher), with each axis independently optimizable**. This decomposition is now the headline methodological finding of the paper.

### Best n=50 integration leaderboard (level-up campaign)

The level-up campaign (2026-05-15) layered three orthogonal improvements onto the A3B baseline: stage-balanced 10-shot teacher exemplars, a 24M-parameter BERT consultant trained on the SocratDataset train split, and the Gemma 4 31B teacher swap. All configurations evaluated apples-to-apples at n=50.

| Rank | Configuration | State acc | R-1 | Notes |
|---:|---|---:|---:|---|
| 1 | **BERT + Gemma 4 31B + 10-shot exemplars** | **51.06%** | **38.53** | Best open-weight on both axes; 2× GPT-4o state acc at 86% R-1 |
| 2 | BERT + A4B + 10-shot exemplars | 48.54% | 37.49 | Cost-efficient alt (2× faster than Gemma) |
| 3 | BERT + A3B + 10-shot exemplars | 48.19% | 35.57 | Lowest-cost integration |
| 4 | A3B + 10-shot exemplars (LLM-only) | 44.15% | 36.16 | Zero-training Pareto win (+6.02 state, +3.29 R-1 vs locked baseline) |
| 5 | A3B locked think (matched n=50) | 38.13% | 32.87 | Reference |
| ref | GPT-4o baseline (n=681) | 25.94% | 44.61 | — |

The headline (row 1) decomposes cleanly: 10-shot exemplars recover surface form (+3.29 R-1 over locked A3B), BERT routes cognitive state with surgical precision (+4.06 state with neutral R-1), and the Gemma teacher swap adds a final +2.87 state / +2.96 R-1.

**Where the BERT integration goes from here.** The n=50 leaderboard reference held BERT + Gemma + 10-shot at 51.06%; the full $n{=}681$ run landed at 48.15% — a small attenuation (−2.91 state, −1.75 R-1) consistent with n-vs-n=50 sampling variance, well within Pareto-win territory over A3B. Next phase is the **prompt-engineering tournament** ($n{=}50 \times 10$ utilizations = 500 dialogues; see [`docs/PROMPT_ENGINEERING_PLAN.md`](docs/PROMPT_ENGINEERING_PLAN.md)) to push state acc toward 55% and R-1 toward 42 on top of this baseline. **Headline run artifacts: [`results/bert-consultant-fewshot10-gemma-full/`](results/bert-consultant-fewshot10-gemma-full/) and [`scripts/eval_bert_gemma_fewshot10_full.sh`](scripts/eval_bert_gemma_fewshot10_full.sh).**

### Phase 0.5 teacher-choice ablation (2026-05-19)

Before launching the prompt-engineering tournament, we validated the Gemma-teacher choice by running the alternative teacher (Qwen 35B-A3B-think) inside the same BERT-integration architecture at full scale: **BERT + A3B + 10-shot at n=681 = 46.57% / 33.27 R-1**, losing to the locked Gemma-teacher headline by −1.58 state, −3.51 R-1. The teachers split stage-by-stage: A3B wins the simpler dialogue acts (b +1.31, e +1.28) but loses on the cognitive heavy-lift stages (c −3.95, d −2.16), consistent with the dense-vs-MoE hypothesis (Gemma's ~31B always-active params absorb the harder reasoning; A3B's ~3B active-per-token MoE shines on lower-cognitive-load acts). This per-stage split directly motivates the **per-state few-shot routing** utilization in the Phase 1 tournament. Run artifacts: [`results/bert-consultant-fewshot10-a3b-full/`](results/bert-consultant-fewshot10-a3b-full/) and [`scripts/eval_bert_a3b_fewshot10_full.sh`](scripts/eval_bert_a3b_fewshot10_full.sh).

### Phase 2 — Frontier-teacher comparison + SocratTeachLLM overfit hypothesis (2026-05-21)

We swapped the locked Gemma teacher for two frontier Claude models (Sonnet 4.6 and Opus 4.6) via the Anthropic API to test whether teacher capacity is the binding constraint on the locked headline. The full leaderboard at $n{=}50$:

| Configuration | State acc | R-1 | R-2 | BLEU-4 | Composite | Δ vs locked Gemma |
|---|---:|---:|---:|---:|---:|---:|
| **Opus 4.6 + BERT + 10-shot + top-3 stack** | 49.82% | 42.77 | 21.12 | **15.53** | **71.20** | **+0.87** |
| Gemma 4 31B + BERT + 10-shot (locked ref) | **51.06%** | 38.53 | 16.93 | 9.68 | 70.33 | — |
| Sonnet 4.6 + BERT + 10-shot + top-3 stack | 48.75% | **43.02** | 20.52 | 14.33 | 70.26 | −0.07 |
| Sonnet 4.6 + BERT + 10-shot (no top-3) | 47.94% | 39.68 | 19.40 | 10.15 | 67.78 | −2.55 |
| Opus 4.6 + BERT + 10-shot (no top-3) | 47.43% | 32.99 | 15.26 | 7.24 | 63.92 | −6.41 |
| Sonnet 4.6 + BERT, raw (no exemplars) | 45.00% | 29.10 | 13.38 | 5.69 | 59.55 | −10.78 |
| Opus 4.6 + BERT, raw (no exemplars) | 39.75% | 23.28 | 10.12 | 4.18 | 51.39 | −18.94 |

**Key findings:**

1. **Frontier Claude models are 2-5× more prompt-engineering-sensitive than Gemma.** Raw Opus collapses to composite 51.39 (−18.94 below the locked Gemma headline). Adding 10-shot exemplars alone lifts Sonnet by +8.23 composite; adding the top-3 stack on top of that lifts Opus by another +7.28. The single-cell length-budget lever that gave Gemma only +1.58 composite delivers 2-5× larger lifts on Claude — strong evidence that the locked Gemma headline operates at a different point on its prompt-sensitivity curve.

2. **The locked Gemma headline narrowly holds.** Only Opus + the full top-3 prompt stack clears it (+0.87 composite), within sampling noise. Despite frontier model capacity, no Claude configuration delivers a decisive headline-shifting result at $n{=}50$.

3. **SocratTeachLLM-overfit hypothesis — strong preliminary evidence.** The paper's reported baseline (GPT-4o consultant + SocratTeachLLM teacher) achieves **R-1 = 44.61** — higher than ANY frontier teacher we evaluated, including Opus 4.6 with the carefully tuned top-3 prompt stack (R-1 = 42.77). A 9B GLM-4 fine-tune from 2024 lexically out-mimics ground truth better than Opus 4.6 with a stage-balanced 10-shot exemplar block AND a length-budget + persona + negative-exemplar prompt stack. **This is statistically improbable under normal circumstances** — frontier models with prompts specifically tuned for surface mimicry should reach or exceed a small specialist model's R-1, not trail it. The combination of high R-1 (44.61) with low state acc (25.94%) is the classic signature of phrasing memorization rather than generalized pedagogical capability. We are running the literal-architecture mirror (Claude as consultant + SocratTeachLLM as teacher) at $n{=}50$ to test whether SocratTeachLLM's R-1 holds with a frontier consultant — if it does, that supports the memorization interpretation; if it collapses, the original 44.61 was partly a consultant-prompting artifact.

4. **Even raw Claude (no exemplars) beats the paper's reported state accuracy.** Sonnet 4.6 raw, with no prompt engineering whatsoever, hits 45.00% state acc — almost double the 25.94% the paper reports for its GPT-4o + SocratTeachLLM pipeline. The frontier teacher with zero scaffolding delivers more pedagogically correct routing than the paper's specialized architecture.

Full briefing in [`docs/archive/CLAUDE_TRIPLE_ARCH_BRIEFING.md`](docs/archive/CLAUDE_TRIPLE_ARCH_BRIEFING.md). Plan-of-record in [`docs/CLAUDE_API_TEACHER_PLAN.md`](docs/CLAUDE_API_TEACHER_PLAN.md). Run artifacts in [`results/bert-claude-*`](results/) and [`results/bert-consultant-fewshot10-claude-*`](results/). Total API spend so far: ~$2.55.

### Phase 2 — Surface-form leaderboard reveals the benchmark is broken (2026-05-21)

When we rank all top configurations by surface-form metrics (R-1 + R-2 + BLEU-4), the leaderboard inverts: the **9B SocratTeachLLM from 2024 crushes Opus 4.6 by every n-gram measure**, while producing the **worst state accuracy of any configuration tested** (including raw Opus with zero scaffolding):

| Rank | Configuration | n turns | R-1 | R-2 | BLEU-4 | Sum | State acc |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **GPT-4o consultant + SocratTeachLLM teacher** (paper baseline) | 681 | **44.61** | **26.04** | **19.60** | **90.25** | 25.94% |
| 2 | Opus 4.6 + 10-shot + top-3 stack | 271 | 42.77 | 21.12 | 15.53 | 79.42 | 49.82% |
| 3 | Sonnet 4.6 + 10-shot + top-3 stack | 281 | 43.02 | 20.52 | 14.33 | 77.87 | 48.75% |
| 4 | Gemma 4 31B + 10-shot + top-3 stack | 278 | 41.13 | 18.60 | 12.91 | 72.64 | **50.72%** |
| 5 | Sonnet 4.6 + 10-shot only | 267 | 39.68 | 19.40 | 10.15 | 69.23 | 47.94% |
| 6 | Qwen 35B-A3B-think + 10-shot + top-3 stack | 276 | 37.64 | 15.69 | 9.97 | 63.30 | 48.19% |
| 7 | Gemma 4 31B + 10-shot (locked headline) | 3834 | 36.78 | 16.10 | 9.05 | 61.93 | 48.15% |
| 8 | Opus 4.6 + 10-shot only | 272 | 32.99 | 15.26 | 7.24 | 55.49 | 47.43% |
| 9 | Sonnet 4.6 raw | 260 | 29.10 | 13.38 | 5.69 | 48.17 | 45.00% |
| 10 | Opus 4.6 raw | 239 | 23.28 | 10.12 | 4.18 | 37.58 | 39.75% |

**By composite (state + 0.5×R-1, our existing primary metric), Phase 3 promotion candidate is Gemma 4 31B + top-3:**

| Rank | Configuration | State | R-1 | Composite |
|---:|---|---:|---:|---:|
| 1 | **🏆 Gemma 4 31B + top-3** (new Phase 3 candidate) | 50.72 | 41.13 | **71.28** |
| 2 | Opus 4.6 + top-3 | 49.82 | 42.77 | 71.20 |
| 3 | Gemma 4 31B + 10-shot (locked, baseline) | 51.06 | 38.53 | 70.33 |
| 4 | Sonnet 4.6 + top-3 | 48.75 | 43.02 | 70.26 |
| 5 | Qwen 35B-A3B-think + top-3 | 48.19 | 37.64 | 67.01 |

Gemma + top-3 stack edges Opus + top-3 by +0.08 composite (statistical tie). The frontier-model architectural advantage is fully neutralized when both teachers receive the same prompt scaffolding. Gemma + top-3 is the Phase 3 promotion target — pending Max's decision on whether to also run Opus 4.6 at full n=681 for an apples-to-apples paper comparison.

**The longer the n-gram, the wider the SocratTeachLLM gap.** R-1 +1.59 → R-2 +4.92 → BLEU-4 +4.07 over the best non-SocratTeachLLM configuration. Higher-order n-gram overlap captures phrase-level fingerprinting — the strongest possible memorization signature. **The metrics are inversely correlated with the thing they're supposed to measure**: the most pedagogically capable models (high state acc) score lowest on surface metrics; the most surface-matched model scores worst on pedagogy.

This motivated a methodological critique that we now treat as a primary paper contribution: **the KELE benchmark, as published, systematically rewards memorization over teaching capability**. ROUGE/BLEU were designed for translation/summarization where the valid-output space is small; Socratic teaching has an enormous space of pedagogically equivalent responses (paraphrases of the same teaching move), and n-gram overlap conflates surface-form match with teaching quality. The "ground-truth" state annotations are also GPT-4-generated, so any model trained to imitate GPT-4's annotation conventions has a structural advantage. The full critique and proposed alternative four-metric evaluation panel (LLM-judge rubric + BERT-annotated state acc + semantic R-1 + stage-progression efficiency) is in [`docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md`](docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md).

### Phase 3 — Cross-lingual + LLM-judge close the case (2026-05-22)

The surface-form inversion has a sharp falsifiable prediction. If SocratTeachLLM's n-gram advantage is phrase-level memorization rather than genuine teaching capability, translating the dataset into a different language should preserve word- and bigram-level overlap (translation respects lexical content at low orders) but destroy 4-gram overlap (specific phrase patterns will not survive translation). We ran the test in two complementary ways.

**Cross-lingual translation experiment.** Run SocratTeachLLM (still configured as teacher) against the [SocratDataset-EN](https://huggingface.co/datasets/ulises-c/SocratDataset-EN) translation, with frontier Claude as consultant:

| Configuration | R-1 | R-2 | BLEU-4 | R-2/BLEU-4 ratio |
|---|---:|---:|---:|---:|
| Sonnet + SocratTeachLLM on Chinese (clean rerun) | 45.61 | — | — | ~1.3 |
| **Sonnet + SocratTeachLLM on English** | **55.85** | **33.79** | 3.56 | **~9** |
| Opus + SocratTeachLLM on English | 44.22 | 26.20 | 2.96 | ~9 |

Two punchlines:

- **(a) Sonnet+STL on English (R-1=55.85 / R-2=33.79) nearly matches the original paper's R-1=57.40 / R-2=33.63.** The field's flagship surface-form number is reproducible by translating SocratDataset into the original paper's reporting language. The paper's headline may itself reflect a Chinese-to-English translation-pipeline confound rather than native-language teaching quality.
- **(b) The R-2/BLEU-4 ratio explodes 1.3 → 9** under translation. Word- and bigram-level translations survive; specific Chinese 4-gram phrase patterns do not. This is the textbook signature of phrase-level memorization at exactly the n-gram length the surface-form-inversion argument predicted.

**LLM-judge as memorization-resistant evaluator.** We built Proposal 1 from the critique doc: Claude Sonnet 4.6 scores each generated teacher response across 4 axes (Socratic validity, learning advancement, age-appropriateness, question-form fidelity) on a 0-10 composite. `scripts/llm_judge_eval.py` ran the rubric across 15 configurations for $59.86 total. Final leaderboard:

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

The locked BERT+Gemma+10-shot integration at n=681 tops the memorization-resistant ranking. **The open-weight system wins the fair test.**

**Cross-lingual judge deltas — the kill shot:**

```
Opus + top-3 (Chinese → English):         -0.07   ← transfers!
Sonnet → SocratTeachLLM (ZH → EN):        -1.01   ← 14× worse
Opus → SocratTeachLLM (ZH → EN):          -1.03
```

Frontier+prompt-eng transfers cross-lingually with negligible judge loss. SocratTeachLLM-using configurations lose **14× more** on a metric specifically constructed to be paraphrase-invariant. Even after stripping surface n-gram dependence, the SocratTeachLLM "advantage" on Chinese is revealed as language-bound memorization, not transferable pedagogical capability.

**Putting the four observations together.** The surface-form inversion, the monotonically widening n-gram gap, the translation reproducing the paper's headline within 1.5 R-1 points, and the asymmetric cross-lingual judge degradation are all explained by a single hypothesis: **SocratTeachLLM was trained on test-set surface forms, either by direct contamination or by insufficient distributional separation between train and test splits.** Each observation individually is suggestive; the four together converge on this interpretation.

### Phase 3 — Frontier ceiling at full scale (Sonnet & Opus n=681, 2026-05-22)

To bound how much pedagogical performance is gettable with paid inference, we ran the same top-3 prompt-engineering stack (`length_budget` + `teacher_persona` + `negative_exemplars` + 10-shot exemplars) with Sonnet 4.6 and Opus 4.6 as teacher at the full n=681 test split:

| Configuration | n turns | State acc | R-1 | R-2 | BLEU-4 | $/run | Δ vs locked open-weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Sonnet 4.6 + BERT + top-3 (n=681)** | 3840 | **49.97%** | **41.93** | 20.43 | 13.46 | ~$5 | +1.82 / +5.15 R-1 |
| Opus 4.6 + BERT + top-3 (n=681) | 3794 | 49.31% | 41.63 | 19.87 | 13.91 | ~$8 | +1.16 / +4.85 R-1 |
| **BERT + Gemma + 10-shot (n=681 LOCKED open-weight)** | 3834 | **48.15%** | 36.78 | 16.10 | 9.05 | **$0** | — |

Two findings:

- **The cheaper frontier model wins at full scale.** At n=50 the order was Opus > Sonnet by +0.94 composite; at n=681 Sonnet edges Opus by +0.66 state. The "frontier cap" isn't reliably the more expensive model — Sonnet 4.6's +0.66 lead at $5/run vs Opus's $8/run is the cost-efficient frontier choice.
- **The locked open-weight headline holds the cost-pareto.** The frontier ceiling is just +1.82 state / +5.15 R-1 above BERT+Gemma+10-shot, at $5+ per run. We frame Sonnet/Opus as the **frontier ceiling at full scale** — they bound the gain available from paid inference; **the locked open-weight system remains THE headline** at 48.15% state / 36.78 R-1 / **$0** per run, well within striking distance.

Per-stage at full scale shows where the frontier teachers win and lose: Sonnet n=681 = a:99.27 / b:35.61 / c:26.48 / d:45.30 / e:84.20; Opus n=681 = a:99.27 / b:36.79 / c:23.95 / d:45.36 / e:87.78; locked Gemma = a:99.27 / b:23.26 / c:30.31 / d:41.50 / e:82.77. The frontier teachers beat Gemma on **b** (early-reasoning dialogue acts) and **d/e** (resolution + closure); Gemma beats them on **c** (the 22-state misconception induction). The dense-31B-and-pre-trained-on-Chinese-pedagogy combination still wins the hardest single stage.

Run artifacts: [`results/bert-claude-sonnet-top3-n681/`](results/bert-claude-sonnet-top3-n681/), [`results/bert-claude-opus-top3-n681/`](results/bert-claude-opus-top3-n681/), [`results/bert-claude-opus-top3-EN-n50/`](results/), [`results/claude-{sonnet,opus}-consultant-socratteachllm-n50/`](results/), per-config `judge_summary.json` in each.

### Phase 3 — Sample-size calibration: n=400 is the canonical ground-truth size (2026-05-22)

The 681-dialogue test split is **over-sampled** for ranking purposes. A bootstrap convergence analysis across all 7 of our full-scale n=681 configurations (B=500 random subsamples without replacement, at each n in a 13-point grid from 25 to 600) shows the smallest n at which the 95th-percentile bootstrap deviation from the n=681 truth stays under a chosen tolerance — for every metric, simultaneously, across every configuration:

| tolerance ε | state_acc | ROUGE-1 | ROUGE-2 | sent-BLEU-4 |
|---:|---:|---:|---:|---:|
| 0.5 pp | 600 | 500 | 500 | 400 |
| **1.0 pp** | **600** | **200** | **200** | **175** |
| **2.0 pp** | **400** | **75**  | **75**  | **75**  |

**State accuracy is the binding constraint** — surface-form metrics (R-1, R-2, BLEU-4) converge ~3–4× faster than the per-turn 34-state classification accuracy because the latter is a high-variance binary indicator while the former are continuous and tightly distributed per turn.

**Canonical decision: n = 400 dialogues** is the ground-truth size for all future full-scale evaluation runs in this project. It satisfies ε ≤ 2 pp on all four primary metrics across every configuration we've tested, and saves **~41% compute, wall-clock, and API spend** vs. the n=681 test split with no loss of decision precision on any ranking ≥ 2 pp:

| Track | n=681 cost | n=400 cost | Saving |
|---|---:|---:|---:|
| Anthropic Sonnet 4.6 (full top-3 stack) | ~$5 | ~$2.94 | 41% |
| Anthropic Opus 4.6 | ~$8 | ~$4.70 | 41% |
| Gemma 4 31B local (RTX 5090) | 12.9 GPU-h | ~7.6 GPU-h | 41% |
| A3B fusion-think local | 16.5 GPU-h | ~9.7 GPU-h | 41% |
| LLM-judge eval (per config) | ~$3–4 | ~$1.75–2.35 | 41% |

**Tier structure for future evaluations:**

- **n = 200** — *screening tier*: prompt-engineering tournament cells, configuration pruning, anything where Δ ≥ 3 pp is decision-relevant.
- **n = 400** — *canonical headline tier*: any leaderboard-changing claim. Replaces n=681 as the default.
- **n = 600+** — *high-precision tier*: paper-defining decisions where state-acc resolution ≤ 1 pp matters (rare).

**The new dialogue-sequence dataset under construction should target n=400 random dialogues** as its canonical evaluation tier — or **n=200-300 dialogues stratified by stage × subject** as a tighter alternative (stratification converges faster than random sampling and likely halves the required n for the same tolerance).

Full method + tolerance sweep + per-metric per-config plateau table in [`docs/CONVERGENCE_ANALYSIS.md`](docs/CONVERGENCE_ANALYSIS.md). Reproducer: [`scripts/convergence_analysis.py`](scripts/convergence_analysis.py) (~3s for all 7 runs). Figure: [`docs/figures/convergence_curves.png`](docs/figures/convergence_curves.png).

> **Note on prior n=681 results.** The locked open-weight headline (BERT+Gemma+10-shot at 48.15% state / 36.78 R-1, n=681) and the Phase 3-Claude frontier ceilings (Sonnet 49.97% / Opus 49.31%) remain canonical — we ran the full n=681 and report the full numbers. The n=400 recommendation governs **future** runs and the new dataset we're constructing; existing n=681 results stand as documented. Aggregated leaderboard: [`results/master_leaderboard.md`](results/master_leaderboard.md). Figures: [`docs/figures/leaderboard_inversion.png`](docs/figures/), [`docs/figures/ngram_gap_widening.png`](docs/figures/), [`docs/figures/four_metric_panel.png`](docs/figures/), [`docs/figures/judge_vs_surface.png`](docs/figures/), [`docs/figures/pareto_inversion.png`](docs/figures/).

Full experimental record lives in [`deliverables/overleaf/latex/acl_latex.tex`](deliverables/overleaf/latex/acl_latex.tex) Section 4 and the per-run logs in [`results/`](results/).

## Architecture

Four key design choices distinguish our extension from a vanilla KELE reproduction:

**Unified fusion call.** KELE separates consultant (state classifier) and teacher (response generator) into two LLM calls per turn. Under our 32 GB single-GPU budget, the canonical stack (SocratTeachLLM ≈ 19 GB + 14B+ consultant) does not fit. We collapse both roles into a *single* JSON-schema-constrained LLM call that emits `{evaluation, state_code, teacher_response}` on every turn, with the schema enforced server-side via `llama.cpp`'s strict `json_schema` grammar. Empirically, fusion improves state accuracy by an average of +9.9 points over two-call across the smoke matrix (within each model and thinking mode), at roughly half the per-turn wall clock. Schema-fallback rate at full scale is 0.91% (38/4171 turns).

**24M-param BERT consultant.** A `bge-small-zh` classifier (24M params, ~95 MB) fine-tuned on the SocratDataset train split classifies dialogue context into one of 34 cognitive states (or 5 stages in the hierarchical variant). On the held-out test split, the 5-stage head scores **86.55%** and the 34-state head scores **61.64%** — dominating every LLM consultant we measured, including a +75-point lift on stage c (the hardest 22-way classification) vs GPT-4o, and +60 vs A3B+10-shot. Wired into the pipeline via the `--bert-consultant <ckpt>` flag, it replaces the LLM consultant call entirely, leaving the LLM to handle only response generation. Single-call training in 148s; zero inference VRAM contention with the LLM teacher. See [`scripts/train_state_classifier_34way.py`](scripts/train_state_classifier_34way.py).

**Stage-balanced 10-shot teacher exemplars.** A zero-training prompt-engineering layer that injects 10 stage-balanced exemplars from the train split into the teacher prompt. On A3B fusion-think at n=50: **+6.02 state, +3.29 R-1** over the locked baseline — a Pareto win along both axes. Triangulated across 5 sample sizes (n=5/15/25/50/681) — a methodological note we carry forward in the paper: any prompt-eng claim needs ≥3 sample sizes to survive small-n variance.

**Smoke → mini → full gating.** Every candidate configuration runs through three escalating evaluation tiers before any full-test-set commitment:

| Tier | n | Turns | Wall clock | Purpose |
|---|---:|---:|---:|---|
| Smoke | 5 | ~33 | 7–40 min | Cheap sanity check; rank a wide config matrix |
| Mini | 25 | ~145 | 30–60 min | Stratified-sample promotion gate |
| Full | 681 | ~4170 | 16–30 h | Canonical evaluation on held-out test split |

A central methodological observation: the smoke--mini *average* tracked our realized full-run number to within 0.10 absolute points (predicted +12.86, realized +12.76). Smoke tends optimistic, mini tends pessimistic; the average is the cheapest accurate predictor.

## Hugging Face

| Artifact | HF URL | Status |
|---|---|---|
| SocratTeachLLM (model) | [ulises-c/SocratTeachLLM](https://huggingface.co/ulises-c/SocratTeachLLM) | Live |
| SocratDataset (Chinese) | [ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset) | Live |
| SocratDataset-EN (English) | [ulises-c/SocratDataset-EN](https://huggingface.co/datasets/ulises-c/SocratDataset-EN) | Live — see [`docs/TRANSLATION_PLAN.md`](docs/TRANSLATION_PLAN.md) |

**Useful HF CLI commands:**
```bash
hf download ulises-c/SocratTeachLLM --local-dir ~/hf_models/SocratTeachLLM
hf download ulises-c/SocratDataset --repo-type dataset --local-dir ~/hf_datasets/SocratDataset
hf auth whoami
```

## Repository Structure

| Path | Description |
| --- | --- |
| `src/project/` | Reproduction pipeline + fusion (unified-call) architecture |
| `scripts/` | Per-experiment orchestrators (smoke/mini/full), serving helpers, plotting |
| `configs/` | Per-experiment `.env` files (one per model/hardware target) |
| `results/` | Per-run outputs: `metrics_summary.json`, `dialogues/*.json`, full logs, server logs |
| `deliverables/` | Course deliverables (paper draft, slides, submitted PDFs) |
| `docs/` | Working plans, exploration logs, design notes, attribution |
| `references/` | Immutable reference data — do not modify or import directly (see [`references/README.md`](references/README.md)) |
| `references/KELE/` | KELE baseline source + dataset + attribution (see [`references/KELE/ATTRIBUTION.md`](references/KELE/ATTRIBUTION.md)) |
| `references/requirements/` | Course guidelines |
| `tests/` | Pytest suites (offline by default; conditional skip for API/dep-gated tests) |

Key design docs:
- [`docs/SOCRATIC_FUSION_PLAN.md`](docs/SOCRATIC_FUSION_PLAN.md) — fusion architecture motivation and design
- [`docs/QWEN_LOCAL_EXPLORATION_LOG.md`](docs/QWEN_LOCAL_EXPLORATION_LOG.md) — full experimental trajectory
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — engineering decisions, dated entries
- [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md) — proposed extensions catalog
- [`docs/CI_FUTURE_WORK.md`](docs/CI_FUTURE_WORK.md) — deferred CI/automation items

## Running the Experiments

All open-weight runs share a single `llama.cpp` server on port 8080 (only one model fits at a time on 32 GB VRAM). The orchestrators boot the server, run the eval, optionally compare against the GPT-4o baseline, and tear down — all crash-safe via per-item resume.

**Headline locked result (BERT + Gemma 4 31B + 10-shot integration, 2026-05-18):**
```bash
# The locked headline. Bootstraps its own Gemma 4 31B server.
./scripts/eval_bert_gemma_fewshot10_full.sh
```

**Prior locked headline (Qwen 35B-A3B fusion-think, superseded 2026-05-18 by the BERT integration):**
```bash
make serve-qwen35b-a3b           # in one shell
./scripts/eval_qwen35b_a3b.sh full --unified
```

**Gemma 4 31B fusion (standalone — superseded by the integration after the 21% schema-fallback regression):**
```bash
make serve-gemma4-31b
./scripts/eval_gemma4_31b.sh smoke --unified    # n=5
./scripts/eval_gemma4_31b.sh mini  --unified    # n=25
./scripts/eval_gemma4_31b.sh full  --unified    # n=681 (~22 h actual)
```

**Best n=50 integration (BERT consultant + Gemma 4 31B teacher + 10-shot exemplars):**
```bash
# Train the BERT consultant once (~148s on a single 5090)
uv run python scripts/train_state_classifier_34way.py

# Quick n=50 reproduction:
make serve-gemma4-31b
KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=10 \
  uv run python -m src.project.kele \
    --experiment gemma4-31b-local \
    test --bert-consultant results/state_classifier_v1/final \
    --n 50 --output results/bert-consultant-fewshot10-gemma-n50

# Full n=681 evaluation (the active gating experiment, ~12h):
./scripts/eval_bert_gemma_fewshot10_full.sh
```

**Reproduction baseline (GPT-4o + SocratTeachLLM, requires `OPENAI_API_KEY`):**
```bash
make serve-socratteachllm
uv run kele --experiment baseline evaluate --output results/baseline
```

**Full smoke matrix (8 cells, ~3 hours total):**
```bash
./scripts/run_all_fusion_smokes.sh
```

| Config family | Boot script | Eval orchestrator | Notes |
|---|---|---|---|
| GPT-4o baseline (KELE canonical) | `make serve-socratteachllm` | `uv run kele --experiment baseline evaluate` | Two-model stack; needs `OPENAI_API_KEY` |
| Qwen 27B Q5 (dense) | `make serve-qwen27b` | `./scripts/eval_qwen27b.sh {smoke,mini,full} [--unified] [--nothink]` | Held in reserve as fusion-think alt |
| Qwen 3.6 35B-A3B (MoE) | `make serve-qwen35b-a3b` | `./scripts/eval_qwen35b_a3b.sh {smoke,mini,full} [--unified] [--nothink]` | **Locked headline backbone** |
| Gemma 4 31B (dense) | `make serve-gemma4-31b` | `./scripts/eval_gemma4_31b.sh {smoke,mini,full} [--unified]` | **Forward candidate** (no `--nothink`; not Gemma-supported) |

Per-run outputs land in `results/<experiment>/`: `metrics_summary.json` for headline numbers, `dialogues/*.json` for per-dialogue traces, `run_<timestamp>.log` for the full eval log, and `server_<timestamp>.log` for the `llama.cpp` server log. Use `kele-eval` to recompute or compare runs:

```bash
uv run kele-eval results/qwen35b-a3b-local-unified
uv run kele-eval --compare results/baseline results/qwen35b-a3b-local-unified
```

## Mirroring to the org repo

[ulises-c/csen-346](https://github.com/ulises-c/csen-346) is the primary development repo. It is mirrored to [SCU-CSEN346/KELE](https://github.com/SCU-CSEN346/KELE) via a dual-push remote — every `git push` publishes to both simultaneously, preserving full git history.

### How it works

`origin` is configured with two push URLs:

```
origin  git@github.com:ulises-c/csen-346.git  (fetch)
origin  git@github.com:ulises-c/csen-346.git  (push)
origin  git@github.com:SCU-CSEN346/KELE.git   (push)
```

A normal `git push` hits both. Fetch and pull still only come from `ulises-c/csen-346`.

### Setup (first time, per machine)

If you cloned from `SCU-CSEN346/KELE`, re-point your fetch remote and add the second push URL:

```bash
git remote set-url origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:SCU-CSEN346/KELE.git
```

If you cloned from `ulises-c/csen-346`, only the two push URLs are needed:

```bash
git remote set-url --add --push origin git@github.com:ulises-c/csen-346.git
git remote set-url --add --push origin git@github.com:SCU-CSEN346/KELE.git
```

Verify with `git remote -v` — you should see one fetch URL and two push URLs.

## Dependencies

[uv](https://docs.astral.sh/uv/) — Python package and project manager.

## Python Environment

This repo targets Python `3.12` and uses uv for dependency management.

### Initial setup

```bash
uv sync --group dev
```

If you want to confirm the virtualenv uv is using:

```bash
uv run python -V
uv run which pytest
```

### Install git hooks

```bash
make install-hooks
```

This copies `hooks/pre-commit` into `.git/hooks/` so that the following checks run automatically before every commit, mirroring the CI pipeline:

1. **ruff format** — auto-format check
2. **ruff check** — linting
3. **pyright** — static type checking
4. **codespell** — spell checking across source and docs
5. **shellcheck** — shell script linting (skipped gracefully if not installed; `brew install shellcheck`)
6. **pytest** — full test suite with coverage report

### Torch note

`torch` is intentionally not declared in `pyproject.toml` because the CUDA wheel installation is environment-specific. After `uv sync`, install the appropriate PyTorch build manually for your machine.

Example for CUDA 12.6:

```bash
uv run pip install --index-url https://download.pytorch.org/whl/cu126 "torch>=2.10.0"
```

## Common Commands

uv exposes the main repo entry points directly:

- `uv run kele`
- `uv run kele-eval`
- `uv run serve-teacher`

These map to the main modules in `src/project/`.

### Run tests

Run the offline/default test suite:

```bash
uv run pytest
```

Show skip reasons too:

```bash
uv run pytest -rs
```

Run a single test file:

```bash
uv run pytest tests/test_metrics.py
```

Some tests are conditional and will skip if the required dependency or runtime is not present:

- `tests/test_consultant.py` needs `openai` and relevant API credentials
- `tests/test_metrics.py` / `tests/test_evaluate.py` need `rouge-score` and `sacrebleu`
- `tests/test_serve_teacher.py` needs `fastapi`

### Run the KELE CLI

Show CLI help:

```bash
uv run kele --help
```

Quick smoke test on a few dialogues:

```bash
uv run kele --experiment baseline test --n 3 --output results/test
```

Run a full evaluation:

```bash
uv run kele --experiment baseline evaluate --output results/baseline
```

Run a partial evaluation (useful for smoke-testing the pipeline):

```bash
poetry run kele --experiment baseline evaluate --limit 5
```

Start a **fresh run** — archives any previous `run_config.json` and `metrics_summary.json`
into a timestamped `run{N}_{timestamp}/` subfolder, then clears `dialogues/` and `progress.log`:

```bash
poetry run kele --experiment baseline evaluate --limit 5 --new
```

After several `--new` runs the output directory looks like:

```
results/baseline/
├── run1_2026-04-14T10-14-09-07-00/   ← first run, archived
│   ├── run_config.json
│   └── metrics_summary.json
├── run2_2026-04-26T09-03-22-07-00/   ← second run, archived
│   └── ...
├── dialogues/                         ← current run (per-dialogue JSONs)
├── progress.log                       ← live progress (updated each dialogue)
├── run_config.json                    ← current run metadata
└── metrics_summary.json               ← current run metrics
```

Resume an interrupted run (skips already-saved dialogues automatically):

```bash
poetry run kele --experiment baseline evaluate
```

### Evaluate saved results

Recompute metrics for one run:

```bash
uv run kele-eval results/baseline
```

Compare two runs side-by-side:

```bash
uv run kele-eval --compare results/baseline results/qwen35b-a3b-local-unified
```

### Start the local teacher server

```bash
uv run serve-teacher
```

With a custom local model path:

```bash
TEACHER_LOCAL_PATH=~/hf_models/SocratTeachLLM uv run serve-teacher
```

### Run online from your own machine

If you want a public HTTPS endpoint backed by your local GPU, use the online-serving helper plus a tunnel such as Cloudflare Tunnel or ngrok.

```bash
TEACHER_LOCAL_PATH=~/hf_models/SocratTeachLLM \
TEACHER_SERVER_API_KEY=replace-this-with-a-long-random-secret \
./scripts/serve_teacher_online.sh
```

Then point your tunnel at `http://127.0.0.1:8001`.

See [`scripts/ONLINE_SETUP.md`](scripts/ONLINE_SETUP.md) for the full public-serving flow.

### Run helper scripts

The repo also includes shell scripts under `scripts/` for multi-process workflows such as model serving and long evaluation runs.

Examples:

```bash
./scripts/run_eval.sh baseline           # full eval run
./scripts/run_eval.sh baseline --limit 5 # smoke test
./scripts/serve_socratteachllm.sh
./scripts/serve_consultant.sh
./scripts/serve_both.sh
```

#### AMD Linux + Mac Mini setup

To run with a local AMD GPU teacher and a Mac Mini llama.cpp consultant:

```bash
# On the Mac Mini (once per session):
set -a && source configs/local-mac-m4.env && set +a
./scripts/serve_consultant_llamacpp.sh

# On the host PC:
./scripts/eval_amd_mac.sh              # full run (resumes if interrupted)
./scripts/eval_amd_mac.sh --limit 5   # smoke test (resumes previous)
./scripts/eval_amd_mac.sh --limit 5 --new  # fresh smoke test, archives old results
```

See [`scripts/MAC_MINI_SETUP.md`](scripts/MAC_MINI_SETUP.md) for first-time Mac Mini setup.

### Run on SCU WAVE nodes

If you have access to SCU WAVE GPU nodes, use the included cluster config and Slurm job:

```bash
sbatch scripts/slurm/wave_eval.slurm
```

See [`scripts/WAVE_SETUP.md`](scripts/WAVE_SETUP.md) for the full setup and model-path overrides.

## Configuration

Runtime settings are loaded from:

- `configs/<experiment>.env` for experiment-specific values
- `.env` for shared secrets and local overrides

Typical usage pattern:

```bash
uv run kele --experiment baseline test --n 3 --output results/test
```

This loads `configs/baseline.env` first, then fills in any missing values from `.env`.
