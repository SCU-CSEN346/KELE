# Stage-balanced backtest — 2026_05_27

Recomputed from 144 configs (filtered to n_turns ≥ 50; 31 smaller files skipped).

## Metrics

- **macro** — frequency-weighted state acc (`Σ correct / Σ turns`). The published headline.
- **stage_bal** — Option A: `(1/5) × Σ p_s`. ML-standard macro-F1 move; recommended new headline.
- **pedagogical** — Option B: weights `a=.10 b=.20 c=.25 d=.20 e=.25` giving closure parity with questioning.
- **freq_inv** — Option C: weights ∝ 1/(per-stage turn count); falls back to stage_bal when counts unavailable.
- **judge** — LLM-judge `overall_avg` from `judge_summary.json` (Claude Sonnet 4.6 rubric, 0-10 scale); `—` if not judged.
- **unified** — `0.5 × stage_bal + 0.5 × (judge × 10)`. The recommended single-number ranking for the paper headline. See `docs/UNIFIED_RANKING.md` for full rationale.
- **unified_ped** — same as `unified` but using `pedagogical` instead of `stage_bal`. Pedagogically-informed alternative.
- **Δrank** — `macro_rank − stage_bal_rank` (positive = moved UP under stage-balanced; negative = moved DOWN).

## Master ranked list (by unified score = 0.5 × stage_bal + 0.5 × (judge × 10))

Only cells with both stage_bal AND judge get a unified score (39/144 configs).

| u# | config | n | **unified** | unified_ped | stage_bal | judge | macro | R-1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `qwen3.5 × Gemma-31B · fewshot10 · n=681` | 3974 | **72.24** | 70.09 | 61.32 |  8.32 | 55.39 | 37.65 |
| 2 | `bert × Gemma-31B · composed · top3 · n=50` | 278 | **70.08** | 67.88 | 58.48 |  8.17 | 50.72 | 41.13 |
| 3 | `bert × Claude-Sonnet · top3 · n=681` | 3840 | **70.06** | 67.86 | 58.17 |  8.19 | 49.97 | 41.93 |
| 4 | `bert × Claude-Opus · fewshot10 · n=50` | 271 | **69.79** | 67.54 | 58.73 |  8.08 | 49.82 | 42.77 |
| 5 | `bert × Claude-Opus · top3 · n=681` | 3794 | **69.37** | 67.20 | 58.63 |  8.01 | 49.31 | 41.63 |
| 6 | `bert × Claude-Sonnet · fewshot10 · n=50` | 281 | **69.16** | 66.84 | 57.18 |  8.11 | 48.75 | 43.02 |
| 7 | `qwen3.5 × Gemma-31B · fewshot10 · n=50` | 285 | **68.94** | 66.44 | 56.13 |  8.18 | 51.58 | 38.76 |
| 8 | `bert-fixed × Gemma-31B · fewshot10 · n=681` | 3889 | **68.94** | 66.60 | 55.38 |  8.25 | 48.73 | 36.94 |
| 9 | `bert × Gemma-31B · fewshot10 · n=681` | 3834 | **68.65** | 66.51 | 55.42 |  8.19 | 48.15 | 36.78 |
| 10 | `qwen3.5 × SocratTeachLLM · fewshot10 · n=50` | 288 | **68.21** | 65.83 | 63.40 |  7.30 | 58.33 | 48.07 |
| 11 | `qwen3.5 × Qwen-27B · phase3 · think · n=100 · seed=42` | 570 | **68.05** | 65.84 | 60.88 |  7.52 | 54.04 | 33.17 |
| 12 | `bert × Claude-Sonnet · fewshot10 · n=50` | 267 | **67.85** | 65.86 | 57.32 |  7.84 | 47.94 | 39.68 |
| 13 | `qwen3.5 × A3B-35B · fewshot10 · n=681` | 3968 | **67.81** | 65.63 | 60.02 |  7.56 | 53.40 | 34.10 |
| 14 | `bert-fixed × Gemma-31B · fewshot10 · n=50` | 283 | **67.65** | 64.95 | 52.73 |  8.26 | 45.94 | 38.69 |
| 15 | `qwen3.5 × Gemma-31B · fewshot10 · EN · RETRY · n=100 · seed=42` | 584 | **67.30** | 65.08 | 52.10 |  8.25 | 46.58 | 11.66 |
| 16 | `qwen3.5 × A3B-35B · fewshot10 · n=50` | 288 | **66.91** | 64.57 | 58.62 |  7.52 | 54.86 | 35.67 |
| 17 | `qwen3.5 × Qwen-27B · think · fewshot10 · n=50` | 282 | **66.89** | 64.65 | 58.68 |  7.51 | 53.19 | 35.28 |
| 18 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=681` | 4010 | **66.71** | 64.38 | 58.16 |  7.53 | 53.04 | 36.63 |
| 19 | `qwen3.5 × Gemma-31B · fewshot10 · EN · PARTIAL · n=61` | 351 | **66.66** | 64.21 | 50.84 |  8.25 | 45.01 | 10.88 |
| 20 | `qwen3.5 × Qwen-27B · phase3 · no-think · n=200 · seed=42` | 1172 | **66.55** | 64.11 | 57.45 |  7.56 | 52.13 | 37.11 |
| 21 | `bert × A3B-35B · composed · top3 · n=50` | 276 | **65.79** | 63.44 | 56.73 |  7.49 | 48.19 | 37.64 |
| 22 | `bert-fixed × Qwen-27B · think · fewshot10 · n=50` | 271 | **65.65** | 63.35 | 57.15 |  7.41 | 49.08 | 34.91 |
| 23 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=50` | 291 | **65.54** | 62.86 | 55.45 |  7.56 | 51.89 | 37.38 |
| 24 | `qwen3.5 × Gemma-31B · fewshot10 · EN · canonical · n=400 · seed=42` | 2324 | **65.11** | 62.66 | 49.12 |  8.11 | 42.34 | 10.45 |
| 25 | `bert-fixed × SocratTeachLLM · fewshot10 · n=50` | 278 | **65.09** | 62.86 | 58.34 |  7.18 | 52.52 | 47.44 |
| 26 | `bert × Claude-Opus · fewshot10 · n=50` | 272 | **64.99** | 62.69 | 55.52 |  7.44 | 47.43 | 32.99 |
| 27 | `bert × A3B-35B · fewshot10 · n=681` | 3762 | **64.47** | 62.27 | 54.72 |  7.42 | 46.57 | 33.27 |
| 28 | `bert-fixed × Qwen-27B · no-think · fewshot10 · n=50` | 286 | **64.25** | 61.64 | 52.62 |  7.59 | 46.85 | 37.81 |
| 29 | `bert-fixed × A3B-35B · fewshot10 · n=50` | 280 | **63.70** | 61.10 | 52.48 |  7.49 | 45.36 | 34.99 |
| 30 | `bert × Claude-Sonnet · raw · n=50` | 260 | **62.91** | 60.46 | 53.32 |  7.25 | 45.00 | 29.10 |
| 31 | `bert × Claude-Opus · top3 · EN · n=50` | 270 | **60.38** | 56.97 | 40.69 |  8.01 | 34.44 |  0.47 |
| 32 | `Claude-Opus × SocratTeachLLM · n=50` | 307 | **59.86** | 56.63 | 41.68 |  7.80 | 38.44 | 47.58 |
| 33 | `bert-fixed × SocratTeachLLM · fewshot10 · EN · n=50` | 273 | **58.22** | 55.49 | 48.97 |  6.75 | 43.22 | 48.07 |
| 34 | `qwen3.5 × SocratTeachLLM · fewshot10 · EN · n=50` | 291 | **57.20** | 54.52 | 48.66 |  6.57 | 43.99 | 46.73 |
| 35 | `bert × Claude-Opus · raw · n=50` | 239 | **55.63** | 52.07 | 43.27 |  6.80 | 39.75 | 23.28 |
| 36 | `qwen3.5 × SocratTeachLLM · CLEANPROBE · fewshot10 · SYNTH · n=50 · seed=42` | 211 | **51.29** | 47.54 | 32.86 |  6.97 | 29.38 | 35.72 |
| 37 | `Claude-Opus × SocratTeachLLM · EN · n=50` | 304 | **50.73** | 46.81 | 33.79 |  6.77 | 30.26 | 44.22 |
| 38 | `Claude-Sonnet × SocratTeachLLM · clean · n=50` | 307 | **47.53** | 48.16 | 18.74 |  7.63 | 18.57 | 45.61 |
| 39 | `Claude-Sonnet × SocratTeachLLM · EN · n=50` | 303 | **44.36** | 43.52 | 22.55 |  6.62 | 22.11 | 55.85 |

## Stage-balanced leaderboard (all configs; sorted by stage_bal; Δrank vs macro; 39/144 have judge scores)

| sb# | macro# | Δ | config | n | macro | stage_bal | pedagogical | freq_inv | judge | R-1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | · | `qwen3.5 × SocratTeachLLM · fewshot10 · n=50` | 288 | 58.33 | **63.40** | 58.65 | 66.80 |  7.30 | 48.07 |
| 2 | 2 | · | `qwen3.5 × Gemma-31B · fewshot10 · n=681` | 3974 | 55.39 | **61.32** | 57.02 | 65.87 |  8.32 | 37.65 |
| 3 | 6 | +3 | `qwen3.5 × Qwen-27B · phase3 · think · n=100 · seed=42` | 570 | 54.04 | **60.88** | 56.46 | 66.19 |  7.52 | 33.17 |
| 4 | 5 | +1 | `bert-consultant-fewshot10-mini` | 135 | 54.07 | **60.66** | 56.96 | 66.61 |   —   | 33.12 |
| 5 | 8 | +3 | `bert-consultant-fewshot10-a4b-mini` | 139 | 53.24 | **60.52** | 56.31 | 66.61 |   —   | 36.26 |
| 6 | 7 | +1 | `qwen3.5 × A3B-35B · fewshot10 · n=681` | 3968 | 53.40 | **60.02** | 55.65 | 65.28 |  7.56 | 34.10 |
| 7 | 13 | +6 | `tournament-cell-1-length_budget` | 281 | 51.96 | **59.50** | 55.01 | 66.04 |   —   | 39.91 |
| 8 | 18 | +10 | `tournament-cell-7-cot_scaffold` | 281 | 50.89 | **59.02** | 54.78 | 66.39 |   —   | 38.98 |
| 9 | 24 | +15 | `bert × Claude-Opus · fewshot10 · n=50` | 271 | 49.82 | **58.73** | 54.22 | 67.15 |  8.08 | 42.77 |
| 10 | 21 | +11 | `tournament-cell-5-negative_exemplars` | 282 | 50.71 | **58.71** | 54.58 | 66.15 |   —   | 39.82 |
| 11 | 9 | -2 | `qwen3.5 × Qwen-27B · think · fewshot10 · n=50` | 282 | 53.19 | **58.68** | 54.22 | 63.30 |  7.51 | 35.28 |
| 12 | 25 | +13 | `bert × Claude-Opus · top3 · n=681` | 3794 | 49.31 | **58.63** | 54.29 | 66.95 |  8.01 | 41.63 |
| 13 | 4 | -9 | `qwen3.5 × A3B-35B · fewshot10 · n=50` | 288 | 54.86 | **58.62** | 53.93 | 61.58 |  7.52 | 35.67 |
| 14 | 20 | +6 | `bert × Gemma-31B · composed · top3 · n=50` | 278 | 50.72 | **58.48** | 54.07 | 65.15 |  8.17 | 41.13 |
| 15 | 17 | +2 | `bert × Gemma-31B · fewshot10 · n=50` | 284 | 51.06 | **58.47** | 54.16 | 65.04 |   —   | 38.53 |
| 16 | 11 | -5 | `bert-fixed × SocratTeachLLM · fewshot10 · n=50` | 278 | 52.52 | **58.34** | 53.88 | 63.08 |  7.18 | 47.44 |
| 17 | 23 | +6 | `bert × Claude-Sonnet · top3 · n=681` | 3840 | 49.97 | **58.17** | 53.78 | 65.20 |  8.19 | 41.93 |
| 18 | 10 | -8 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=681` | 4010 | 53.04 | **58.16** | 53.49 | 61.99 |  7.53 | 36.63 |
| 19 | 3 | -16 | `qwen3.5 × SocratTeachLLM · MEMPROBE · fewshot10 · TRAIN · n=50 · seed=42` | 297 | 55.22 | **57.81** | 52.82 | 59.35 |   —   | 48.28 |
| 20 | 16 | -4 | `tournament-cell-9-persona` | 275 | 51.27 | **57.65** | 53.33 | 63.80 |   —   | 39.60 |
| 21 | 12 | -9 | `qwen3.5 × Qwen-27B · phase3 · no-think · n=200 · seed=42` | 1172 | 52.13 | **57.45** | 52.57 | 61.38 |  7.56 | 37.11 |
| 22 | 35 | +13 | `bert × Claude-Sonnet · fewshot10 · n=50` | 267 | 47.94 | **57.32** | 53.34 | 67.76 |  7.84 | 39.68 |
| 23 | 29 | +6 | `bert × Claude-Sonnet · fewshot10 · n=50` | 281 | 48.75 | **57.18** | 52.54 | 64.77 |  8.11 | 43.02 |
| 24 | 19 | -5 | `tournament-cell-4-per_state_exemplars` | 285 | 50.88 | **57.18** | 52.83 | 62.65 |   —   | 39.42 |
| 25 | 28 | +3 | `bert-fixed × Qwen-27B · think · fewshot10 · n=50` | 271 | 49.08 | **57.15** | 52.56 | 64.76 |  7.41 | 34.91 |
| 26 | 33 | +7 | `bert-consultant-fewshot10-n50` | 276 | 48.19 | **57.03** | 52.27 | 64.68 |   —   | 35.57 |
| 27 | 26 | -1 | `bert × A3B-35B · n=mini` | 128 | 49.22 | **56.95** | 52.31 | 63.94 |   —   | 26.94 |
| 28 | 22 | -6 | `bert × Gemma-31B · fewshot10 · n=mini` | 137 | 50.36 | **56.87** | 52.56 | 62.50 |   —   | 35.93 |
| 29 | 32 | +3 | `bert × A3B-35B · composed · top3 · n=50` | 276 | 48.19 | **56.73** | 52.02 | 64.63 |  7.49 | 37.64 |
| 30 | 15 | -15 | `qwen3.5 × Gemma-31B · fewshot10 · n=50` | 285 | 51.58 | **56.13** | 51.13 | 59.63 |  8.18 | 38.76 |
| 31 | 31 | · | `bert-consultant-fewshot10-a4b-n50` | 274 | 48.54 | **56.12** | 51.75 | 63.49 |   —   | 37.49 |
| 32 | 37 | +5 | `bert × Claude-Opus · fewshot10 · n=50` | 272 | 47.43 | **55.52** | 50.93 | 63.45 |  7.44 | 32.99 |
| 33 | 14 | -19 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=50` | 291 | 51.89 | **55.45** | 50.09 | 57.90 |  7.56 | 37.38 |
| 34 | 34 | · | `bert × Gemma-31B · fewshot10 · n=681` | 3834 | 48.15 | **55.42** | 51.15 | 62.18 |  8.19 | 36.78 |
| 35 | 30 | -5 | `bert-fixed × Gemma-31B · fewshot10 · n=681` | 3889 | 48.73 | **55.38** | 50.70 | 60.89 |  8.25 | 36.94 |
| 36 | 41 | +5 | `tournament-cell-3-style_matched_exemplars` | 274 | 46.72 | **55.15** | 50.20 | 62.55 |   —   | 42.17 |
| 37 | 43 | +6 | `bert × A3B-35B · fewshot10 · n=681` | 3762 | 46.57 | **54.72** | 50.31 | 62.43 |  7.42 | 33.27 |
| 38 | 38 | · | `tournament-cell-8-nbest_rerank` | 281 | 47.33 | **53.61** | 48.89 | 59.26 |   —   | 38.27 |
| 39 | 51 | +12 | `bert × Claude-Sonnet · raw · n=50` | 260 | 45.00 | **53.32** | 48.42 | 62.13 |  7.25 | 29.10 |
| 40 | 36 | -4 | `tournament-cell-10-compressed_history` | 280 | 47.50 | **52.87** | 47.86 | 57.65 |   —   | 38.79 |
| 41 | 45 | +4 | `bert-fixed × Gemma-31B · fewshot10 · n=50` | 283 | 45.94 | **52.73** | 47.32 | 58.29 |  8.26 | 38.69 |
| 42 | 27 | -15 | `bert-consultant-richeval-mini` | 126 | 49.21 | **52.67** | 47.08 | 55.25 |   —   | 26.53 |
| 43 | 40 | -3 | `bert-fixed × Qwen-27B · no-think · fewshot10 · n=50` | 286 | 46.85 | **52.62** | 47.41 | 57.42 |  7.59 | 37.81 |
| 44 | 47 | +3 | `bert-fixed × A3B-35B · fewshot10 · n=50` | 280 | 45.36 | **52.48** | 47.28 | 58.41 |  7.49 | 34.99 |
| 45 | 44 | -1 | `tournament-cell-6-format_retry` | 282 | 46.45 | **52.36** | 47.56 | 57.97 |   —   | 39.60 |
| 46 | 54 | +8 | `bert × A3B-35B · n=50` | 261 | 44.06 | **52.31** | 47.37 | 61.44 |   —   | 28.36 |
| 47 | 42 | -5 | `qwen3.5 × Gemma-31B · fewshot10 · EN · RETRY · n=100 · seed=42` | 584 | 46.58 | **52.10** | 47.65 | 56.78 |  8.25 | 11.66 |
| 48 | 57 | +9 | `tournament-cell-2-lexical_priors` | 262 | 43.89 | **50.97** | 45.78 | 57.81 |   —   | 39.12 |
| 49 | 50 | +1 | `qwen3.5 × Gemma-31B · fewshot10 · EN · PARTIAL · n=61` | 351 | 45.01 | **50.84** | 45.94 | 55.55 |  8.25 | 10.88 |
| 50 | 39 | -11 | `qwen35b-a3b-local-mini-unified-fewshot10` | 147 | 46.94 | **50.73** | 45.23 | 53.34 |   —   | 34.41 |
| 51 | 56 | +5 | `bert-v2-consultant-fewshot10-n50` | 287 | 43.90 | **50.60** | 45.29 | 56.16 |   —   | 37.12 |
| 52 | 48 | -4 | `tournament/round4/qwen27b-q4` | 302 | 45.36 | **50.57** | 45.38 | 50.57 |   —   | 32.43 |
| 53 | 46 | -7 | `qwen27b-local-mini-unified` | 146 | 45.89 | **50.16** | 44.24 | 53.02 |   —   | 29.36 |
| 54 | 49 | -5 | `qwen35b-a3b-local-mini-unified-fewshot7` | 148 | 45.27 | **49.78** | 43.34 | 52.91 |   —   | 35.69 |
| 55 | 53 | -2 | `qwen35b-a3b-local-n50-unified-fewshot10` | 299 | 44.15 | **49.34** | 43.87 | 53.37 |   —   | 36.16 |
| 56 | 62 | +6 | `qwen3.5 × Gemma-31B · fewshot10 · EN · canonical · n=400 · seed=42` | 2324 | 42.34 | **49.12** | 44.21 | 54.84 |  8.11 | 10.45 |
| 57 | 52 | -5 | `bert-v2-consultant-fewshot10-mini` | 142 | 44.37 | **49.08** | 43.43 | 52.81 |   —   | 33.55 |
| 58 | 59 | +1 | `bert-fixed × SocratTeachLLM · fewshot10 · EN · n=50` | 273 | 43.22 | **48.97** | 43.50 | 53.53 |  6.75 | 48.07 |
| 59 | 55 | -4 | `qwen3.5 × SocratTeachLLM · fewshot10 · EN · n=50` | 291 | 43.99 | **48.66** | 43.30 | 52.24 |  6.57 | 46.73 |
| 60 | 60 | · | `tournament/round1/qwen27b-q4` | 300 | 43.00 | **48.35** | 43.35 | 48.35 |   —   | 31.62 |
| 61 | 61 | · | `tournament/round1/qwen27b` | 302 | 42.72 | **48.21** | 41.92 | 48.21 |   —   | 30.90 |
| 62 | 63 | +1 | `tournament/round2/qwen27b-q4` | 299 | 42.14 | **47.39** | 41.73 | 47.39 |   —   | 30.87 |
| 63 | 64 | +1 | `gemma4-31b-local-mini-unified` | 148 | 41.89 | **46.32** | 40.18 | 49.47 |   —   | 30.11 |
| 64 | 58 | -6 | `qwen35b-a3b-local-mini-unified-fewshot` | 148 | 43.24 | **45.74** | 40.50 | 47.39 |   —   | 33.49 |
| 65 | 66 | +1 | `tournament/archive/368b6431/round1/gemma4-31b` | 305 | 40.33 | **44.99** | 38.78 | 44.99 |   —   | 32.88 |
| 66 | 65 | -1 | `tournament/archive/d9ac39c5/round1/gemma4-31b` | 304 | 41.12 | **44.72** | 37.87 | 44.72 |   —   | 32.96 |
| 67 | 68 | +1 | `tournament/round2/gemma4-31b` | 305 | 39.67 | **44.46** | 38.14 | 44.46 |   —   | 33.11 |
| 68 | 72 | +4 | `qwen35b-a3b-local-unified` | 4171 | 38.70 | **44.05** | 38.59 | 47.88 |   —   | 30.63 |
| 69 | 69 | · | `tournament/archive/d9ac39c5/round1/gemma4-26b-a4b` | 303 | 39.27 | **43.49** | 37.04 | 43.49 |   —   | 31.60 |
| 70 | 70 | · | `gemma4-26b-a4b-local-mini-unified` | 147 | 38.78 | **43.31** | 36.89 | 46.78 |   —   | 32.04 |
| 71 | 67 | -4 | `bert × Claude-Opus · raw · n=50` | 239 | 39.75 | **43.27** | 36.15 | 43.51 |  6.80 | 23.28 |
| 72 | 73 | +1 | `tournament/archive/497374dd/round1/gemma4-26b-a4b` | 300 | 38.67 | **43.03** | 35.89 | 43.03 |   —   | 32.19 |
| 73 | 75 | +2 | `qwen35b-a3b-local-mini-unified-fewshot5` | 146 | 38.36 | **42.85** | 36.72 | 46.05 |   —   | 34.60 |
| 74 | 77 | +3 | `tournament/round3/gemma4-31b` | 304 | 38.16 | **42.37** | 35.98 | 42.37 |   —   | 32.79 |
| 75 | 76 | +1 | `tournament/round1/gemma4-31b` | 303 | 38.28 | **42.19** | 35.41 | 42.19 |   —   | 33.04 |
| 76 | 71 | -5 | `tournament/round4/gemma4-31b` | 302 | 38.74 | **42.14** | 35.22 | 42.14 |   —   | 33.03 |
| 77 | 78 | +1 | `qwen35b-a3b-local-n50-unified` | 299 | 38.13 | **42.11** | 37.75 | 45.20 |   —   | 32.87 |
| 78 | 80 | +2 | `qwen35b-a3b-local-n50-unified-fewshot` | 298 | 37.58 | **41.96** | 36.96 | 45.23 |   —   | 33.33 |
| 79 | 79 | · | `tournament/round1/gemma4-26b-a4b` | 303 | 37.62 | **41.78** | 35.46 | 41.78 |   —   | 32.48 |
| 80 | 74 | -6 | `Claude-Opus × SocratTeachLLM · n=50` | 307 | 38.44 | **41.68** | 35.22 | 44.39 |  7.80 | 47.58 |
| 81 | 81 | · | `tournament/archive/368b6431/round1/gemma4-26b-a4b` | 298 | 36.24 | **40.85** | 34.94 | 40.85 |   —   | 31.95 |
| 82 | 86 | +4 | `bert × Claude-Opus · top3 · EN · n=50` | 270 | 34.44 | **40.69** | 33.88 | 45.40 |  8.01 |  0.47 |
| 83 | 82 | -1 | `tournament/round3/qwen27b-q4` | 297 | 36.03 | **39.84** | 33.48 | 39.84 |   —   | 31.07 |
| 84 | 85 | +1 | `qwen35b-a3b-local-mini-unified` | 145 | 35.17 | **39.67** | 34.13 | 42.71 |   —   | 30.51 |
| 85 | 83 | -2 | `tournament/archive/497374dd/round1/gemma4-31b` | 304 | 35.86 | **39.58** | 32.40 | 39.58 |   —   | 33.03 |
| 86 | 87 | +1 | `tournament/round1/qwen35b-a3b` | 300 | 34.33 | **38.32** | 33.65 | 38.32 |   —   | 31.28 |
| 87 | 84 | -3 | `qwen27b-local-mini-unified-nothink` | 147 | 35.37 | **37.79** | 31.87 | 39.40 |   —   | 31.14 |
| 88 | 88 | · | `tournament/round1/qwen35-9b` | 297 | 32.66 | **36.30** | 31.94 | 36.30 |   —   | 34.49 |
| 89 | 89 | · | `tournament/archive/368b6431/round1/qwen27b-q4` | 303 | 31.68 | **36.00** | 29.87 | 36.00 |   —   | 32.84 |
| 90 | 92 | +2 | `gemma4-31b-local-unified` | 4246 | 31.39 | **35.60** | 30.33 | 38.62 |   —   | 27.27 |
| 91 | 90 | -1 | `tournament/archive/d9ac39c5/round1/qwen27b` | 307 | 31.60 | **35.51** | 30.02 | 35.51 |   —   | 33.33 |
| 92 | 91 | -1 | `tournament/archive/368b6431/round1/qwen27b` | 305 | 31.48 | **35.37** | 29.62 | 35.37 |   —   | 33.56 |
| 93 | 94 | +1 | `Claude-Opus × SocratTeachLLM · EN · n=50` | 304 | 30.26 | **33.79** | 25.97 | 36.34 |  6.77 | 44.22 |
| 94 | 93 | -1 | `tournament/archive/497374dd/round1/qwen27b-q4` | 300 | 30.67 | **33.13** | 28.47 | 33.13 |   —   | 33.25 |
| 95 | 96 | +1 | `qwen3.5 × SocratTeachLLM · CLEANPROBE · fewshot10 · SYNTH · n=50 · seed=42` | 211 | 29.38 | **32.86** | 25.37 | 34.69 |  6.97 | 35.72 |
| 96 | 95 | -1 | `qwopus35b-a3b-local-mini-unified` | 146 | 30.14 | **32.69** | 26.63 | 34.31 |   —   | 35.39 |
| 97 | 97 | · | `tournament/archive/497374dd/round1/qwen27b` | 304 | 28.62 | **32.35** | 26.66 | 32.35 |   —   | 33.42 |
| 98 | 98 | · | `synthetic-baseline/gemma4-31b` | 213 | 27.23 | **32.16** | 23.55 | 32.16 |   —   | 29.54 |
| 99 | 99 | · | `synthetic-baseline/qwen27b-q4-think-4096` | 208 | 26.92 | **31.25** | 23.24 | 31.25 |   —   | 30.84 |
| 100 | 101 | +1 | `baseline` | 4294 | 25.94 | **30.75** | 22.06 | 34.16 |   —   | 44.61 |
| 101 | 100 | -1 | `tournament/archive/d9ac39c5/round1/qwen27b-q4` | 302 | 26.82 | **30.02** | 26.07 | 30.02 |   —   | 33.32 |
| 102 | 102 | · | `tournament/round1/qwopus35b-a3b` | 305 | 25.90 | **28.88** | 24.59 | 28.88 |   —   | 35.50 |
| 103 | 103 | · | `tournament/archive/368b6431/round1/qwen3-14b` | 308 | 24.68 | **27.35** | 21.12 | 27.35 |   —   | 39.33 |
| 104 | 106 | +2 | `synthetic-baseline/qwen27b-q4` | 209 | 23.44 | **27.26** | 19.62 | 27.26 |   —   | 31.66 |
| 105 | 104 | -1 | `tournament/archive/368b6431/round1/qwen35b-a3b` | 306 | 24.18 | **27.09** | 25.29 | 27.09 |   —   | 31.69 |
| 106 | 105 | -1 | `tournament/archive/d9ac39c5/round1/qwen3-14b` | 308 | 23.70 | **25.89** | 20.36 | 25.89 |   —   | 40.44 |
| 107 | 107 | · | `tournament/archive/497374dd/round1/qwen3-14b` | 307 | 22.48 | **24.88** | 17.90 | 24.88 |   —   | 39.49 |
| 108 | 109 | +1 | `tournament/archive/d9ac39c5/round1/qwopus35b-a3b` | 301 | 21.93 | **24.43** | 18.79 | 24.43 |   —   | 33.38 |
| 109 | 112 | +3 | `tournament/round1/mistral-24b` | 304 | 21.38 | **23.99** | 18.85 | 23.99 |   —   | 37.45 |
| 110 | 111 | +1 | `tournament/archive/d9ac39c5/round1/qwen35b-a3b` | 297 | 21.55 | **23.99** | 22.48 | 23.99 |   —   | 31.43 |
| 111 | 110 | -1 | `tournament/archive/497374dd/round1/mistral-24b` | 302 | 21.85 | **23.61** | 20.03 | 23.61 |   —   | 37.06 |
| 112 | 113 | +1 | `tournament/archive/368b6431/round1/mistral-24b` | 308 | 20.13 | **23.35** | 18.60 | 23.35 |   —   | 37.49 |
| 113 | 108 | -5 | `Claude-Sonnet × SocratTeachLLM · EN · n=50` | 303 | 22.11 | **22.55** | 20.87 | 22.80 |  6.62 | 55.85 |
| 114 | 115 | +1 | `qwen35b-a3b-local-n50-unified-nothink` | 300 | 19.67 | **22.20** | 20.14 | 24.13 |   —   | 30.55 |
| 115 | 116 | +1 | `tournament/archive/d9ac39c5/round1/qwen35-9b` | 305 | 19.02 | **21.88** | 17.50 | 21.88 |   —   | 28.41 |
| 116 | 114 | -2 | `tournament/archive/497374dd/round1/qwen35b-a3b` | 304 | 19.74 | **21.86** | 20.91 | 21.86 |   —   | 31.30 |
| 117 | 117 | · | `wave-2026-04-21T08-59-20-892964` | 4280 | 18.93 | **21.62** | 16.96 | 21.62 |   —   | 43.72 |
| 118 | 119 | +1 | `tournament/archive/497374dd/round1/qwopus35b-a3b` | 307 | 18.57 | **21.41** | 16.19 | 21.41 |   —   | 32.81 |
| 119 | 121 | +2 | `tournament/archive/368b6431/round1/qwopus35b-a3b` | 307 | 17.92 | **20.73** | 15.20 | 20.73 |   —   | 33.13 |
| 120 | 120 | · | `tournament/archive/d9ac39c5/round1/gemma3-27b` | 306 | 18.30 | **20.15** | 17.18 | 20.15 |   —   | 34.35 |
| 121 | 123 | +2 | `tournament/archive/d9ac39c5/round1/mistral-24b` | 307 | 17.26 | **19.87** | 14.85 | 19.87 |   —   | 37.48 |
| 122 | 118 | -4 | `Claude-Sonnet × SocratTeachLLM · clean · n=50` | 307 | 18.57 | **18.74** | 20.00 | 18.81 |  7.63 | 45.61 |
| 123 | 125 | +2 | `R9700_Mac-M4` | 4262 | 15.16 | **18.41** | 11.89 | 18.41 |   —   | 43.57 |
| 124 | 122 | -2 | `tournament/round1/qwen3-14b` | 306 | 17.65 | **18.37** | 19.44 | 18.37 |   —   | 36.40 |
| 125 | 126 | +1 | `baseline_run1_en_bug` | 3978 | 15.08 | **17.92** | 13.90 | 19.81 |   —   |  0.29 |
| 126 | 124 | -2 | `tournament/archive/497374dd/round1/qwen35-9b` | 307 | 15.64 | **17.68** | 14.40 | 17.68 |   —   | 28.23 |
| 127 | 127 | · | `tournament/archive/368b6431/round1/gemma3-27b` | 308 | 14.61 | **17.18** | 13.68 | 17.18 |   —   | 34.86 |
| 128 | 129 | +1 | `tournament/archive/368b6431/round1/qwen35-9b` | 305 | 13.77 | **15.23** | 13.07 | 15.23 |   —   | 28.46 |
| 129 | 128 | -1 | `tournament/round1/gemma3-27b` | 307 | 14.33 | **15.14** | 13.32 | 15.14 |   —   | 34.61 |
| 130 | 132 | +2 | `tournament/archive/497374dd/round1/gemma3-27b` | 308 | 12.99 | **15.09** | 12.94 | 15.09 |   —   | 34.57 |
| 131 | 130 | -1 | `tournament/archive/368b6431/round1/glm47-23b` | 304 | 13.49 | **15.06** | 12.36 | 15.06 |   —   | 32.43 |
| 132 | 131 | -1 | `tournament/archive/497374dd/round1/glm47-23b` | 304 | 13.16 | **14.68** | 12.41 | 14.68 |   —   | 33.00 |
| 133 | 133 | · | `tournament/archive/368b6431/round1/phi4-14b` | 290 | 12.41 | **13.51** | 10.02 | 13.51 |   —   | 34.58 |
| 134 | 134 | · | `tournament/archive/d9ac39c5/round1/deepseek-r1-14b` | 307 | 11.40 | **13.43** |  9.85 | 13.43 |   —   | 34.43 |
| 135 | 135 | · | `tournament/archive/d9ac39c5/round1/phi4-14b` | 308 | 11.04 | **12.53** |  9.28 | 12.53 |   —   | 35.80 |
| 136 | 137 | +1 | `tournament/archive/497374dd/round1/phi4-14b` | 301 | 10.63 | **12.24** |  8.34 | 12.24 |   —   | 35.75 |
| 137 | 136 | -1 | `tournament/round1/deepseek-r1-14b` | 307 | 10.75 | **12.15** |  9.62 | 12.15 |   —   | 34.62 |
| 138 | 138 | · | `tournament/archive/368b6431/round1/deepseek-r1-14b` | 306 | 10.13 | **11.05** |  8.76 | 11.05 |   —   | 35.27 |
| 139 | 139 | · | `tournament/round1/phi4-14b` | 298 |  9.73 | **10.70** |  7.75 | 10.70 |   —   | 35.64 |
| 140 | 140 | · | `tournament/archive/d9ac39c5/round1/glm47-23b` | 305 |  9.51 | **10.67** |  8.59 | 10.67 |   —   | 30.90 |
| 141 | 141 | · | `tournament/archive/497374dd/round1/deepseek-r1-14b` | 306 |  8.17 | ** 9.24** |  7.05 |  9.24 |   —   | 35.70 |
| 142 | 142 | · | `Claude-Opus × SocratTeachLLM · BROKEN · n=50` | 308 |  0.00 | ** 0.00** |  0.00 |  0.00 |   —   | 13.20 |
| 143 | 143 | · | `Claude-Sonnet × SocratTeachLLM · n=50` | 308 |  0.00 | ** 0.00** |  0.00 |  0.00 |   —   | 43.88 |
| 144 | 144 | · | `Claude-Sonnet × SocratTeachLLM · BROKEN · n=50` | 308 |  0.00 | ** 0.00** |  0.00 |  0.00 |   —   | 13.20 |

## Big movers UP (≥3 ranks under stage-balanced)

| Δ | sb# | macro# | config | n | macro → stage_bal | stage_e |
|---:|---:|---:|---|---:|---|---:|
| +15 | 9 | 24 | `bert × Claude-Opus · fewshot10 · n=50` | 271 | 49.82 → 58.73 | 88.9% |
| +13 | 12 | 25 | `bert × Claude-Opus · top3 · n=681` | 3794 | 49.31 → 58.63 | 87.8% |
| +13 | 22 | 35 | `bert × Claude-Sonnet · fewshot10 · n=50` | 267 | 47.94 → 57.32 | 95.7% |
| +12 | 39 | 51 | `bert × Claude-Sonnet · raw · n=50` | 260 | 45.00 → 53.32 | 80.0% |
| +11 | 10 | 21 | `tournament-cell-5-negative_exemplars` | 282 | 50.71 → 58.71 | 90.3% |
| +10 | 8 | 18 | `tournament-cell-7-cot_scaffold` | 281 | 50.89 → 59.02 | 90.6% |
| +9 | 48 | 57 | `tournament-cell-2-lexical_priors` | 262 | 43.89 → 50.97 | 73.9% |
| +8 | 46 | 54 | `bert × A3B-35B · n=50` | 261 | 44.06 → 52.31 | 80.0% |
| +7 | 26 | 33 | `bert-consultant-fewshot10-n50` | 276 | 48.19 → 57.03 | 86.7% |
| +6 | 7 | 13 | `tournament-cell-1-length_budget` | 281 | 51.96 → 59.50 | 83.9% |
| +6 | 14 | 20 | `bert × Gemma-31B · composed · top3 · n=50` | 278 | 50.72 → 58.48 | 87.5% |
| +6 | 17 | 23 | `bert × Claude-Sonnet · top3 · n=681` | 3840 | 49.97 → 58.17 | 84.2% |
| +6 | 23 | 29 | `bert × Claude-Sonnet · fewshot10 · n=50` | 281 | 48.75 → 57.18 | 87.1% |
| +6 | 37 | 43 | `bert × A3B-35B · fewshot10 · n=681` | 3762 | 46.57 → 54.72 | 84.0% |
| +6 | 56 | 62 | `qwen3.5 × Gemma-31B · fewshot10 · EN · canonical · n=400 · seed=42` | 2324 | 42.34 → 49.12 | 72.7% |
| +5 | 32 | 37 | `bert × Claude-Opus · fewshot10 · n=50` | 272 | 47.43 → 55.52 | 84.0% |
| +5 | 36 | 41 | `tournament-cell-3-style_matched_exemplars` | 274 | 46.72 → 55.15 | 82.8% |
| +5 | 51 | 56 | `bert-v2-consultant-fewshot10-n50` | 287 | 43.90 → 50.60 | 75.0% |
| +4 | 41 | 45 | `bert-fixed × Gemma-31B · fewshot10 · n=50` | 283 | 45.94 → 52.73 | 72.7% |
| +4 | 68 | 72 | `qwen35b-a3b-local-unified` | 4171 | 38.70 → 44.05 | 56.8% |

## Big movers DOWN (≥3 ranks under stage-balanced)

| Δ | sb# | macro# | config | n | macro → stage_bal | stage_c |
|---:|---:|---:|---|---:|---|---:|
| -19 | 33 | 14 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=50` | 291 | 51.89 → 55.45 | 34.4% |
| -16 | 19 | 3 | `qwen3.5 × SocratTeachLLM · MEMPROBE · fewshot10 · TRAIN · n=50 · seed=42` | 297 | 55.22 → 57.81 | 43.8% |
| -15 | 30 | 15 | `qwen3.5 × Gemma-31B · fewshot10 · n=50` | 285 | 51.58 → 56.13 | 33.3% |
| -15 | 42 | 27 | `bert-consultant-richeval-mini` | 126 | 49.21 → 52.67 | 30.0% |
| -11 | 50 | 39 | `qwen35b-a3b-local-mini-unified-fewshot10` | 147 | 46.94 → 50.73 | 25.5% |
| -9 | 13 | 4 | `qwen3.5 × A3B-35B · fewshot10 · n=50` | 288 | 54.86 → 58.62 | 39.6% |
| -9 | 21 | 12 | `qwen3.5 × Qwen-27B · phase3 · no-think · n=200 · seed=42` | 1172 | 52.13 → 57.45 | 32.6% |
| -8 | 18 | 10 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=681` | 4010 | 53.04 → 58.16 | 35.5% |
| -7 | 53 | 46 | `qwen27b-local-mini-unified` | 146 | 45.89 → 50.16 | 21.3% |
| -6 | 28 | 22 | `bert × Gemma-31B · fewshot10 · n=mini` | 137 | 50.36 → 56.87 | 25.0% |
| -6 | 64 | 58 | `qwen35b-a3b-local-mini-unified-fewshot` | 148 | 43.24 → 45.74 | 27.7% |
| -6 | 80 | 74 | `Claude-Opus × SocratTeachLLM · n=50` | 307 | 38.44 → 41.68 | 24.0% |
| -5 | 16 | 11 | `bert-fixed × SocratTeachLLM · fewshot10 · n=50` | 278 | 52.52 → 58.34 | 33.3% |
| -5 | 24 | 19 | `tournament-cell-4-per_state_exemplars` | 285 | 50.88 → 57.18 | 29.8% |
| -5 | 35 | 30 | `bert-fixed × Gemma-31B · fewshot10 · n=681` | 3889 | 48.73 → 55.38 | 29.4% |
| -5 | 47 | 42 | `qwen3.5 × Gemma-31B · fewshot10 · EN · RETRY · n=100 · seed=42` | 584 | 46.58 → 52.10 | 30.1% |
| -5 | 54 | 49 | `qwen35b-a3b-local-mini-unified-fewshot7` | 148 | 45.27 → 49.78 | 19.1% |
| -5 | 57 | 52 | `bert-v2-consultant-fewshot10-mini` | 142 | 44.37 → 49.08 | 23.9% |
| -5 | 76 | 71 | `tournament/round4/gemma4-31b` | 302 | 38.74 → 42.14 | 19.8% |
| -5 | 113 | 108 | `Claude-Sonnet × SocratTeachLLM · EN · n=50` | 303 | 22.11 → 22.55 | 18.8% |

## Per-stage breakdown (top 25 by stage_bal)

| sb# | config | n | a | b | c | d | e |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `qwen3.5 × SocratTeachLLM · fewshot10 · n=50` | 288 | 100.00 | 66.04 | 34.69 | 46.00 | 70.27 |
| 2 | `qwen3.5 × Gemma-31B · fewshot10 · n=681` | 3974 | 100.00 | 46.39 | 35.42 | 46.36 | 78.45 |
| 3 | `qwen3.5 × Qwen-27B · phase3 · think · n=100 · seed=42` | 570 | 100.00 | 52.83 | 31.25 | 40.00 | 80.33 |
| 4 | `bert-consultant-fewshot10-mini` | 135 | 100.00 | 25.00 | 31.82 | 52.38 | 94.12 |
| 5 | `bert-consultant-fewshot10-a4b-mini` | 139 | 100.00 | 32.14 | 27.66 | 54.55 | 88.24 |
| 6 | `qwen3.5 × A3B-35B · fewshot10 · n=681` | 3968 | 100.00 | 43.89 | 32.38 | 43.54 | 80.27 |
| 7 | `tournament-cell-1-length_budget` | 281 | 100.00 | 37.29 | 26.32 | 50.00 | 83.87 |
| 8 | `tournament-cell-7-cot_scaffold` | 281 | 100.00 | 32.20 | 24.47 | 47.83 | 90.62 |
| 9 | `bert × Claude-Opus · fewshot10 · n=50` | 271 | 100.00 | 33.90 | 20.88 | 50.00 | 88.89 |
| 10 | `tournament-cell-5-negative_exemplars` | 282 | 100.00 | 30.51 | 27.08 | 45.65 | 90.32 |
| 11 | `qwen3.5 × Qwen-27B · think · fewshot10 · n=50` | 282 | 100.00 | 45.76 | 31.91 | 36.96 | 78.79 |
| 12 | `bert × Claude-Opus · top3 · n=681` | 3794 | 99.27 | 36.79 | 23.95 | 45.36 | 87.78 |
| 13 | `qwen3.5 × A3B-35B · fewshot10 · n=50` | 288 | 100.00 | 50.85 | 39.58 | 36.00 | 66.67 |
| 14 | `bert × Gemma-31B · composed · top3 · n=50` | 278 | 100.00 | 33.90 | 24.47 | 46.51 | 87.50 |
| 15 | `bert × Gemma-31B · fewshot10 · n=50` | 284 | 100.00 | 33.90 | 25.53 | 44.68 | 88.24 |
| 16 | `bert-fixed × SocratTeachLLM · fewshot10 · n=50` | 278 | 100.00 | 32.08 | 33.33 | 48.89 | 77.42 |
| 17 | `bert × Claude-Sonnet · top3 · n=681` | 3840 | 99.27 | 35.61 | 26.48 | 45.30 | 84.20 |
| 18 | `qwen3.5 × Qwen-27B · no-think · fewshot10 · n=681` | 4010 | 100.00 | 45.07 | 35.50 | 39.16 | 71.06 |
| 19 | `qwen3.5 × SocratTeachLLM · MEMPROBE · fewshot10 · TRAIN · n=50 · seed=42` | 297 | 100.00 | 52.83 | 43.81 | 36.00 | 56.41 |
| 20 | `tournament-cell-9-persona` | 275 | 100.00 | 35.59 | 30.77 | 39.13 | 82.76 |
| 21 | `qwen3.5 × Qwen-27B · phase3 · no-think · n=200 · seed=42` | 1172 | 100.00 | 47.51 | 32.61 | 37.37 | 69.78 |
| 22 | `bert × Claude-Sonnet · fewshot10 · n=50` | 267 | 100.00 | 30.51 | 24.73 | 35.71 | 95.65 |
| 23 | `bert × Claude-Sonnet · fewshot10 · n=50` | 281 | 100.00 | 33.90 | 20.21 | 44.68 | 87.10 |
| 24 | `tournament-cell-4-per_state_exemplars` | 285 | 100.00 | 27.12 | 29.79 | 45.65 | 83.33 |
| 25 | `bert-fixed × Qwen-27B · think · fewshot10 · n=50` | 271 | 100.00 | 35.59 | 23.66 | 41.86 | 84.62 |
