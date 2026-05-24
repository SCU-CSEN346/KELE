# Stage-balanced backtest — 23_post_judge

Recomputed from 120 configs (filtered to n_turns ≥ 50; 31 smaller files skipped).

## Metrics

- **macro** — frequency-weighted state acc (`Σ correct / Σ turns`). The published headline.
- **stage_bal** — Option A: `(1/5) × Σ p_s`. ML-standard macro-F1 move; recommended new headline.
- **pedagogical** — Option B: weights `a=.10 b=.20 c=.25 d=.20 e=.25` giving closure parity with questioning.
- **freq_inv** — Option C: weights ∝ 1/(per-stage turn count); falls back to stage_bal when counts unavailable.
- **judge** — LLM-judge `overall_avg` from `judge_summary.json` (Claude Sonnet 4.6 rubric, 0-10 scale); `—` if not judged.
- **Δrank** — `macro_rank − stage_bal_rank` (positive = moved UP under stage-balanced; negative = moved DOWN).

## Leaderboard (sorted by stage_bal; Δrank vs macro; 21/120 have judge scores)

| sb# | macro# | Δ | config | n | macro | stage_bal | pedagogical | freq_inv | judge | R-1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | +1 | `bert-consultant-fewshot10-mini` | 135 | 54.07 | **60.66** | 56.96 | 66.61 |   —   | 33.12 |
| 2 | 3 | +1 | `bert-consultant-fewshot10-a4b-mini` | 139 | 53.24 | **60.52** | 56.31 | 66.61 |   —   | 36.26 |
| 3 | 5 | +2 | `tournament-cell-1-length_budget` | 281 | 51.96 | **59.50** | 55.01 | 66.04 |   —   | 39.91 |
| 4 | 10 | +6 | `tournament-cell-7-cot_scaffold` | 281 | 50.89 | **59.02** | 54.78 | 66.39 |   —   | 38.98 |
| 5 | 16 | +11 | `bert-consultant-fewshot10-claude-opus-n50` | 271 | 49.82 | **58.73** | 54.22 | 67.15 |  8.08 | 42.77 |
| 6 | 13 | +7 | `tournament-cell-5-negative_exemplars` | 282 | 50.71 | **58.71** | 54.58 | 66.15 |   —   | 39.82 |
| 7 | 4 | -3 | `t4-bert-qwen27b-fewshot10-n50-fixed` | 282 | 53.19 | **58.68** | 54.22 | 63.30 |  7.51 | 35.28 |
| 8 | 17 | +9 | `bert-claude-opus-top3-n681` | 3794 | 49.31 | **58.63** | 54.29 | 66.95 |  8.01 | 41.63 |
| 9 | 1 | -8 | `t4-bert-a3b-fewshot10-n50-fixed` | 288 | 54.86 | **58.62** | 53.93 | 61.58 |   —   | 35.67 |
| 10 | 12 | +2 | `bert-gemma-composed-top3-n50` | 278 | 50.72 | **58.48** | 54.07 | 65.15 |  8.17 | 41.13 |
| 11 | 9 | -2 | `bert-consultant-fewshot10-gemma-n50` | 284 | 51.06 | **58.47** | 54.16 | 65.04 |   —   | 38.53 |
| 12 | 15 | +3 | `bert-claude-sonnet-top3-n681` | 3840 | 49.97 | **58.17** | 53.78 | 65.20 |  8.19 | 41.93 |
| 13 | 8 | -5 | `tournament-cell-9-persona` | 275 | 51.27 | **57.65** | 53.33 | 63.80 |   —   | 39.60 |
| 14 | 26 | +12 | `bert-claude-sonnet-fewshot10-n50` | 267 | 47.94 | **57.32** | 53.34 | 67.76 |  7.84 | 39.68 |
| 15 | 21 | +6 | `bert-consultant-fewshot10-claude-sonnet-n50` | 281 | 48.75 | **57.18** | 52.54 | 64.77 |  8.11 | 43.02 |
| 16 | 11 | -5 | `tournament-cell-4-per_state_exemplars` | 285 | 50.88 | **57.18** | 52.83 | 62.65 |   —   | 39.42 |
| 17 | 20 | +3 | `bge-small-bert-qwen27b-fewshot10-n50-fixed` | 271 | 49.08 | **57.15** | 52.56 | 64.76 |  7.41 | 34.91 |
| 18 | 24 | +6 | `bert-consultant-fewshot10-n50` | 276 | 48.19 | **57.03** | 52.27 | 64.68 |   —   | 35.57 |
| 19 | 18 | -1 | `bert-consultant-a3b-mini` | 128 | 49.22 | **56.95** | 52.31 | 63.94 |   —   | 26.94 |
| 20 | 14 | -6 | `bert-consultant-fewshot10-gemma-mini` | 137 | 50.36 | **56.87** | 52.56 | 62.50 |   —   | 35.93 |
| 21 | 23 | +2 | `bert-a3b-composed-top3-n50` | 276 | 48.19 | **56.73** | 52.02 | 64.63 |  7.49 | 37.64 |
| 22 | 7 | -15 | `t4-bert-gemma-fewshot10-n50-fixed` | 285 | 51.58 | **56.13** | 51.13 | 59.63 |   —   | 38.76 |
| 23 | 22 | -1 | `bert-consultant-fewshot10-a4b-n50` | 274 | 48.54 | **56.12** | 51.75 | 63.49 |   —   | 37.49 |
| 24 | 28 | +4 | `bert-claude-opus-fewshot10-n50` | 272 | 47.43 | **55.52** | 50.93 | 63.45 |  7.44 | 32.99 |
| 25 | 6 | -19 | `t4-bert-qwen27b-nothink-fewshot10-n50-fixed` | 291 | 51.89 | **55.45** | 50.09 | 57.90 |  7.56 | 37.38 |
| 26 | 25 | -1 | `bert-consultant-fewshot10-gemma-full` | 3834 | 48.15 | **55.42** | 51.15 | 62.18 |  8.19 | 36.78 |
| 27 | 32 | +5 | `tournament-cell-3-style_matched_exemplars` | 274 | 46.72 | **55.15** | 50.20 | 62.55 |   —   | 42.17 |
| 28 | 33 | +5 | `bert-consultant-fewshot10-a3b-full` | 3762 | 46.57 | **54.72** | 50.31 | 62.43 |  7.42 | 33.27 |
| 29 | 29 | · | `tournament-cell-8-nbest_rerank` | 281 | 47.33 | **53.61** | 48.89 | 59.26 |   —   | 38.27 |
| 30 | 39 | +9 | `bert-claude-sonnet-raw-n50` | 260 | 45.00 | **53.32** | 48.42 | 62.13 |  7.25 | 29.10 |
| 31 | 27 | -4 | `tournament-cell-10-compressed_history` | 280 | 47.50 | **52.87** | 47.86 | 57.65 |   —   | 38.79 |
| 32 | 35 | +3 | `bge-small-bert-gemma-fewshot10-n50-fixed` | 283 | 45.94 | **52.73** | 47.32 | 58.29 |   —   | 38.69 |
| 33 | 19 | -14 | `bert-consultant-richeval-mini` | 126 | 49.21 | **52.67** | 47.08 | 55.25 |   —   | 26.53 |
| 34 | 31 | -3 | `bge-small-bert-qwen27b-nothink-fewshot10-n50-fixed` | 286 | 46.85 | **52.62** | 47.41 | 57.42 |  7.59 | 37.81 |
| 35 | 37 | +2 | `bge-small-bert-a3b-fewshot10-n50-fixed` | 280 | 45.36 | **52.48** | 47.28 | 58.41 |   —   | 34.99 |
| 36 | 34 | -2 | `tournament-cell-6-format_retry` | 282 | 46.45 | **52.36** | 47.56 | 57.97 |   —   | 39.60 |
| 37 | 42 | +5 | `bert-consultant-a3b-n50` | 261 | 44.06 | **52.31** | 47.37 | 61.44 |   —   | 28.36 |
| 38 | 44 | +6 | `tournament-cell-2-lexical_priors` | 262 | 43.89 | **50.97** | 45.78 | 57.81 |   —   | 39.12 |
| 39 | 30 | -9 | `qwen35b-a3b-local-mini-unified-fewshot10` | 147 | 46.94 | **50.73** | 45.23 | 53.34 |   —   | 34.41 |
| 40 | 43 | +3 | `bert-v2-consultant-fewshot10-n50` | 287 | 43.90 | **50.60** | 45.29 | 56.16 |   —   | 37.12 |
| 41 | 36 | -5 | `qwen27b-local-mini-unified` | 146 | 45.89 | **50.16** | 44.24 | 53.02 |   —   | 29.36 |
| 42 | 38 | -4 | `qwen35b-a3b-local-mini-unified-fewshot7` | 148 | 45.27 | **49.78** | 43.34 | 52.91 |   —   | 35.69 |
| 43 | 41 | -2 | `qwen35b-a3b-local-n50-unified-fewshot10` | 299 | 44.15 | **49.34** | 43.87 | 53.37 |   —   | 36.16 |
| 44 | 40 | -4 | `bert-v2-consultant-fewshot10-mini` | 142 | 44.37 | **49.08** | 43.43 | 52.81 |   —   | 33.55 |
| 45 | 46 | +1 | `tournament/round1/qwen27b-q4` | 300 | 43.00 | **48.35** | 43.35 | 48.35 |   —   | 31.62 |
| 46 | 47 | +1 | `tournament/round1/qwen27b` | 302 | 42.72 | **48.21** | 41.92 | 48.21 |   —   | 30.90 |
| 47 | 48 | +1 | `gemma4-31b-local-mini-unified` | 148 | 41.89 | **46.32** | 40.18 | 49.47 |   —   | 30.11 |
| 48 | 45 | -3 | `qwen35b-a3b-local-mini-unified-fewshot` | 148 | 43.24 | **45.74** | 40.50 | 47.39 |   —   | 33.49 |
| 49 | 50 | +1 | `tournament/archive/368b6431/round1/gemma4-31b` | 305 | 40.33 | **44.99** | 38.78 | 44.99 |   —   | 32.88 |
| 50 | 49 | -1 | `tournament/archive/d9ac39c5/round1/gemma4-31b` | 304 | 41.12 | **44.72** | 37.87 | 44.72 |   —   | 32.96 |
| 51 | 54 | +3 | `qwen35b-a3b-local-unified` | 4171 | 38.70 | **44.05** | 38.59 | 47.88 |   —   | 30.63 |
| 52 | 52 | · | `tournament/archive/d9ac39c5/round1/gemma4-26b-a4b` | 303 | 39.27 | **43.49** | 37.04 | 43.49 |   —   | 31.60 |
| 53 | 53 | · | `gemma4-26b-a4b-local-mini-unified` | 147 | 38.78 | **43.31** | 36.89 | 46.78 |   —   | 32.04 |
| 54 | 51 | -3 | `bert-claude-opus-raw-n50` | 239 | 39.75 | **43.27** | 36.15 | 43.51 |  6.80 | 23.28 |
| 55 | 55 | · | `tournament/archive/497374dd/round1/gemma4-26b-a4b` | 300 | 38.67 | **43.03** | 35.89 | 43.03 |   —   | 32.19 |
| 56 | 57 | +1 | `qwen35b-a3b-local-mini-unified-fewshot5` | 146 | 38.36 | **42.85** | 36.72 | 46.05 |   —   | 34.60 |
| 57 | 58 | +1 | `tournament/round1/gemma4-31b` | 303 | 38.28 | **42.19** | 35.41 | 42.19 |   —   | 33.04 |
| 58 | 59 | +1 | `qwen35b-a3b-local-n50-unified` | 299 | 38.13 | **42.11** | 37.75 | 45.20 |   —   | 32.87 |
| 59 | 61 | +2 | `qwen35b-a3b-local-n50-unified-fewshot` | 298 | 37.58 | **41.96** | 36.96 | 45.23 |   —   | 33.33 |
| 60 | 60 | · | `tournament/round1/gemma4-26b-a4b` | 303 | 37.62 | **41.78** | 35.46 | 41.78 |   —   | 32.48 |
| 61 | 56 | -5 | `claude-opus-consultant-socratteachllm-n50` | 307 | 38.44 | **41.68** | 35.22 | 44.39 |  7.80 | 47.58 |
| 62 | 62 | · | `tournament/archive/368b6431/round1/gemma4-26b-a4b` | 298 | 36.24 | **40.85** | 34.94 | 40.85 |   —   | 31.95 |
| 63 | 66 | +3 | `bert-claude-opus-top3-EN-n50` | 270 | 34.44 | **40.69** | 33.88 | 45.40 |  8.01 |  0.47 |
| 64 | 65 | +1 | `qwen35b-a3b-local-mini-unified` | 145 | 35.17 | **39.67** | 34.13 | 42.71 |   —   | 30.51 |
| 65 | 63 | -2 | `tournament/archive/497374dd/round1/gemma4-31b` | 304 | 35.86 | **39.58** | 32.40 | 39.58 |   —   | 33.03 |
| 66 | 67 | +1 | `tournament/round1/qwen35b-a3b` | 300 | 34.33 | **38.32** | 33.65 | 38.32 |   —   | 31.28 |
| 67 | 64 | -3 | `qwen27b-local-mini-unified-nothink` | 147 | 35.37 | **37.79** | 31.87 | 39.40 |   —   | 31.14 |
| 68 | 68 | · | `tournament/round1/qwen35-9b` | 297 | 32.66 | **36.30** | 31.94 | 36.30 |   —   | 34.49 |
| 69 | 69 | · | `tournament/archive/368b6431/round1/qwen27b-q4` | 303 | 31.68 | **36.00** | 29.87 | 36.00 |   —   | 32.84 |
| 70 | 72 | +2 | `gemma4-31b-local-unified` | 4246 | 31.39 | **35.60** | 30.33 | 38.62 |   —   | 27.27 |
| 71 | 70 | -1 | `tournament/archive/d9ac39c5/round1/qwen27b` | 307 | 31.60 | **35.51** | 30.02 | 35.51 |   —   | 33.33 |
| 72 | 71 | -1 | `tournament/archive/368b6431/round1/qwen27b` | 305 | 31.48 | **35.37** | 29.62 | 35.37 |   —   | 33.56 |
| 73 | 74 | +1 | `claude-opus-consultant-socratteachllm-EN-n50` | 304 | 30.26 | **33.79** | 25.97 | 36.34 |  6.77 | 44.22 |
| 74 | 73 | -1 | `tournament/archive/497374dd/round1/qwen27b-q4` | 300 | 30.67 | **33.13** | 28.47 | 33.13 |   —   | 33.25 |
| 75 | 75 | · | `qwopus35b-a3b-local-mini-unified` | 146 | 30.14 | **32.69** | 26.63 | 34.31 |   —   | 35.39 |
| 76 | 76 | · | `tournament/archive/497374dd/round1/qwen27b` | 304 | 28.62 | **32.35** | 26.66 | 32.35 |   —   | 33.42 |
| 77 | 78 | +1 | `baseline` | 4294 | 25.94 | **30.75** | 22.06 | 34.16 |   —   | 44.61 |
| 78 | 77 | -1 | `tournament/archive/d9ac39c5/round1/qwen27b-q4` | 302 | 26.82 | **30.02** | 26.07 | 30.02 |   —   | 33.32 |
| 79 | 79 | · | `tournament/round1/qwopus35b-a3b` | 305 | 25.90 | **28.88** | 24.59 | 28.88 |   —   | 35.50 |
| 80 | 80 | · | `tournament/archive/368b6431/round1/qwen3-14b` | 308 | 24.68 | **27.35** | 21.12 | 27.35 |   —   | 39.33 |
| 81 | 81 | · | `tournament/archive/368b6431/round1/qwen35b-a3b` | 306 | 24.18 | **27.09** | 25.29 | 27.09 |   —   | 31.69 |
| 82 | 82 | · | `tournament/archive/d9ac39c5/round1/qwen3-14b` | 308 | 23.70 | **25.89** | 20.36 | 25.89 |   —   | 40.44 |
| 83 | 83 | · | `tournament/archive/497374dd/round1/qwen3-14b` | 307 | 22.48 | **24.88** | 17.90 | 24.88 |   —   | 39.49 |
| 84 | 85 | +1 | `tournament/archive/d9ac39c5/round1/qwopus35b-a3b` | 301 | 21.93 | **24.43** | 18.79 | 24.43 |   —   | 33.38 |
| 85 | 88 | +3 | `tournament/round1/mistral-24b` | 304 | 21.38 | **23.99** | 18.85 | 23.99 |   —   | 37.45 |
| 86 | 87 | +1 | `tournament/archive/d9ac39c5/round1/qwen35b-a3b` | 297 | 21.55 | **23.99** | 22.48 | 23.99 |   —   | 31.43 |
| 87 | 86 | -1 | `tournament/archive/497374dd/round1/mistral-24b` | 302 | 21.85 | **23.61** | 20.03 | 23.61 |   —   | 37.06 |
| 88 | 89 | +1 | `tournament/archive/368b6431/round1/mistral-24b` | 308 | 20.13 | **23.35** | 18.60 | 23.35 |   —   | 37.49 |
| 89 | 84 | -5 | `claude-sonnet-consultant-socratteachllm-EN-n50` | 303 | 22.11 | **22.55** | 20.87 | 22.80 |  6.62 | 55.85 |
| 90 | 91 | +1 | `qwen35b-a3b-local-n50-unified-nothink` | 300 | 19.67 | **22.20** | 20.14 | 24.13 |   —   | 30.55 |
| 91 | 92 | +1 | `tournament/archive/d9ac39c5/round1/qwen35-9b` | 305 | 19.02 | **21.88** | 17.50 | 21.88 |   —   | 28.41 |
| 92 | 90 | -2 | `tournament/archive/497374dd/round1/qwen35b-a3b` | 304 | 19.74 | **21.86** | 20.91 | 21.86 |   —   | 31.30 |
| 93 | 93 | · | `wave-2026-04-21T08-59-20-892964` | 4280 | 18.93 | **21.62** | 16.96 | 21.62 |   —   | 43.72 |
| 94 | 95 | +1 | `tournament/archive/497374dd/round1/qwopus35b-a3b` | 307 | 18.57 | **21.41** | 16.19 | 21.41 |   —   | 32.81 |
| 95 | 97 | +2 | `tournament/archive/368b6431/round1/qwopus35b-a3b` | 307 | 17.92 | **20.73** | 15.20 | 20.73 |   —   | 33.13 |
| 96 | 96 | · | `tournament/archive/d9ac39c5/round1/gemma3-27b` | 306 | 18.30 | **20.15** | 17.18 | 20.15 |   —   | 34.35 |
| 97 | 99 | +2 | `tournament/archive/d9ac39c5/round1/mistral-24b` | 307 | 17.26 | **19.87** | 14.85 | 19.87 |   —   | 37.48 |
| 98 | 94 | -4 | `claude-sonnet-consultant-socratteachllm-n50-clean` | 307 | 18.57 | **18.74** | 20.00 | 18.81 |  7.63 | 45.61 |
| 99 | 101 | +2 | `R9700_Mac-M4` | 4262 | 15.16 | **18.41** | 11.89 | 18.41 |   —   | 43.57 |
| 100 | 98 | -2 | `tournament/round1/qwen3-14b` | 306 | 17.65 | **18.37** | 19.44 | 18.37 |   —   | 36.40 |
| 101 | 102 | +1 | `baseline_run1_en_bug` | 3978 | 15.08 | **17.92** | 13.90 | 19.81 |   —   |  0.29 |
| 102 | 100 | -2 | `tournament/archive/497374dd/round1/qwen35-9b` | 307 | 15.64 | **17.68** | 14.40 | 17.68 |   —   | 28.23 |
| 103 | 103 | · | `tournament/archive/368b6431/round1/gemma3-27b` | 308 | 14.61 | **17.18** | 13.68 | 17.18 |   —   | 34.86 |
| 104 | 105 | +1 | `tournament/archive/368b6431/round1/qwen35-9b` | 305 | 13.77 | **15.23** | 13.07 | 15.23 |   —   | 28.46 |
| 105 | 104 | -1 | `tournament/round1/gemma3-27b` | 307 | 14.33 | **15.14** | 13.32 | 15.14 |   —   | 34.61 |
| 106 | 108 | +2 | `tournament/archive/497374dd/round1/gemma3-27b` | 308 | 12.99 | **15.09** | 12.94 | 15.09 |   —   | 34.57 |
| 107 | 106 | -1 | `tournament/archive/368b6431/round1/glm47-23b` | 304 | 13.49 | **15.06** | 12.36 | 15.06 |   —   | 32.43 |
| 108 | 107 | -1 | `tournament/archive/497374dd/round1/glm47-23b` | 304 | 13.16 | **14.68** | 12.41 | 14.68 |   —   | 33.00 |
| 109 | 109 | · | `tournament/archive/368b6431/round1/phi4-14b` | 290 | 12.41 | **13.51** | 10.02 | 13.51 |   —   | 34.58 |
| 110 | 110 | · | `tournament/archive/d9ac39c5/round1/deepseek-r1-14b` | 307 | 11.40 | **13.43** |  9.85 | 13.43 |   —   | 34.43 |
| 111 | 111 | · | `tournament/archive/d9ac39c5/round1/phi4-14b` | 308 | 11.04 | **12.53** |  9.28 | 12.53 |   —   | 35.80 |
| 112 | 113 | +1 | `tournament/archive/497374dd/round1/phi4-14b` | 301 | 10.63 | **12.24** |  8.34 | 12.24 |   —   | 35.75 |
| 113 | 112 | -1 | `tournament/round1/deepseek-r1-14b` | 307 | 10.75 | **12.15** |  9.62 | 12.15 |   —   | 34.62 |
| 114 | 114 | · | `tournament/archive/368b6431/round1/deepseek-r1-14b` | 306 | 10.13 | **11.05** |  8.76 | 11.05 |   —   | 35.27 |
| 115 | 115 | · | `tournament/round1/phi4-14b` | 298 |  9.73 | **10.70** |  7.75 | 10.70 |   —   | 35.64 |
| 116 | 116 | · | `tournament/archive/d9ac39c5/round1/glm47-23b` | 305 |  9.51 | **10.67** |  8.59 | 10.67 |   —   | 30.90 |
| 117 | 117 | · | `tournament/archive/497374dd/round1/deepseek-r1-14b` | 306 |  8.17 | ** 9.24** |  7.05 |  9.24 |   —   | 35.70 |
| 118 | 118 | · | `claude-opus-consultant-socratteachllm-n50-BROKEN` | 308 |  0.00 | ** 0.00** |  0.00 |  0.00 |   —   | 13.20 |
| 119 | 119 | · | `claude-sonnet-consultant-socratteachllm-n50` | 308 |  0.00 | ** 0.00** |  0.00 |  0.00 |   —   | 43.88 |
| 120 | 120 | · | `claude-sonnet-consultant-socratteachllm-n50-BROKEN` | 308 |  0.00 | ** 0.00** |  0.00 |  0.00 |   —   | 13.20 |

## Big movers UP (≥3 ranks under stage-balanced)

| Δ | sb# | macro# | config | n | macro → stage_bal | stage_e |
|---:|---:|---:|---|---:|---|---:|
| +12 | 14 | 26 | `bert-claude-sonnet-fewshot10-n50` | 267 | 47.94 → 57.32 | 95.7% |
| +11 | 5 | 16 | `bert-consultant-fewshot10-claude-opus-n50` | 271 | 49.82 → 58.73 | 88.9% |
| +9 | 8 | 17 | `bert-claude-opus-top3-n681` | 3794 | 49.31 → 58.63 | 87.8% |
| +9 | 30 | 39 | `bert-claude-sonnet-raw-n50` | 260 | 45.00 → 53.32 | 80.0% |
| +7 | 6 | 13 | `tournament-cell-5-negative_exemplars` | 282 | 50.71 → 58.71 | 90.3% |
| +6 | 4 | 10 | `tournament-cell-7-cot_scaffold` | 281 | 50.89 → 59.02 | 90.6% |
| +6 | 15 | 21 | `bert-consultant-fewshot10-claude-sonnet-n50` | 281 | 48.75 → 57.18 | 87.1% |
| +6 | 18 | 24 | `bert-consultant-fewshot10-n50` | 276 | 48.19 → 57.03 | 86.7% |
| +6 | 38 | 44 | `tournament-cell-2-lexical_priors` | 262 | 43.89 → 50.97 | 73.9% |
| +5 | 27 | 32 | `tournament-cell-3-style_matched_exemplars` | 274 | 46.72 → 55.15 | 82.8% |
| +5 | 28 | 33 | `bert-consultant-fewshot10-a3b-full` | 3762 | 46.57 → 54.72 | 84.0% |
| +5 | 37 | 42 | `bert-consultant-a3b-n50` | 261 | 44.06 → 52.31 | 80.0% |
| +4 | 24 | 28 | `bert-claude-opus-fewshot10-n50` | 272 | 47.43 → 55.52 | 84.0% |
| +3 | 12 | 15 | `bert-claude-sonnet-top3-n681` | 3840 | 49.97 → 58.17 | 84.2% |
| +3 | 17 | 20 | `bge-small-bert-qwen27b-fewshot10-n50-fixed` | 271 | 49.08 → 57.15 | 84.6% |
| +3 | 32 | 35 | `bge-small-bert-gemma-fewshot10-n50-fixed` | 283 | 45.94 → 52.73 | 72.7% |
| +3 | 40 | 43 | `bert-v2-consultant-fewshot10-n50` | 287 | 43.90 → 50.60 | 75.0% |
| +3 | 51 | 54 | `qwen35b-a3b-local-unified` | 4171 | 38.70 → 44.05 | 56.8% |
| +3 | 63 | 66 | `bert-claude-opus-top3-EN-n50` | 270 | 34.44 → 40.69 | 52.0% |
| +3 | 85 | 88 | `tournament/round1/mistral-24b` | 304 | 21.38 → 23.99 | 23.9% |

## Big movers DOWN (≥3 ranks under stage-balanced)

| Δ | sb# | macro# | config | n | macro → stage_bal | stage_c |
|---:|---:|---:|---|---:|---|---:|
| -19 | 25 | 6 | `t4-bert-qwen27b-nothink-fewshot10-n50-fixed` | 291 | 51.89 → 55.45 | 34.4% |
| -15 | 22 | 7 | `t4-bert-gemma-fewshot10-n50-fixed` | 285 | 51.58 → 56.13 | 33.3% |
| -14 | 33 | 19 | `bert-consultant-richeval-mini` | 126 | 49.21 → 52.67 | 30.0% |
| -9 | 39 | 30 | `qwen35b-a3b-local-mini-unified-fewshot10` | 147 | 46.94 → 50.73 | 25.5% |
| -8 | 9 | 1 | `t4-bert-a3b-fewshot10-n50-fixed` | 288 | 54.86 → 58.62 | 39.6% |
| -6 | 20 | 14 | `bert-consultant-fewshot10-gemma-mini` | 137 | 50.36 → 56.87 | 25.0% |
| -5 | 13 | 8 | `tournament-cell-9-persona` | 275 | 51.27 → 57.65 | 30.8% |
| -5 | 16 | 11 | `tournament-cell-4-per_state_exemplars` | 285 | 50.88 → 57.18 | 29.8% |
| -5 | 41 | 36 | `qwen27b-local-mini-unified` | 146 | 45.89 → 50.16 | 21.3% |
| -5 | 61 | 56 | `claude-opus-consultant-socratteachllm-n50` | 307 | 38.44 → 41.68 | 24.0% |
| -5 | 89 | 84 | `claude-sonnet-consultant-socratteachllm-EN-n50` | 303 | 22.11 → 22.55 | 18.8% |
| -4 | 31 | 27 | `tournament-cell-10-compressed_history` | 280 | 47.50 → 52.87 | 29.8% |
| -4 | 42 | 38 | `qwen35b-a3b-local-mini-unified-fewshot7` | 148 | 45.27 → 49.78 | 19.1% |
| -4 | 44 | 40 | `bert-v2-consultant-fewshot10-mini` | 142 | 44.37 → 49.08 | 23.9% |
| -4 | 98 | 94 | `claude-sonnet-consultant-socratteachllm-n50-clean` | 307 | 18.57 → 18.74 | 16.7% |
| -3 | 7 | 4 | `t4-bert-qwen27b-fewshot10-n50-fixed` | 282 | 53.19 → 58.68 | 31.9% |
| -3 | 34 | 31 | `bge-small-bert-qwen27b-nothink-fewshot10-n50-fixed` | 286 | 46.85 → 52.62 | 25.3% |
| -3 | 48 | 45 | `qwen35b-a3b-local-mini-unified-fewshot` | 148 | 43.24 → 45.74 | 27.7% |
| -3 | 54 | 51 | `bert-claude-opus-raw-n50` | 239 | 39.75 → 43.27 | 17.6% |
| -3 | 67 | 64 | `qwen27b-local-mini-unified-nothink` | 147 | 35.37 → 37.79 | 21.3% |

## Per-stage breakdown (top 25 by stage_bal)

| sb# | config | n | a | b | c | d | e |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `bert-consultant-fewshot10-mini` | 135 | 100.00 | 25.00 | 31.82 | 52.38 | 94.12 |
| 2 | `bert-consultant-fewshot10-a4b-mini` | 139 | 100.00 | 32.14 | 27.66 | 54.55 | 88.24 |
| 3 | `tournament-cell-1-length_budget` | 281 | 100.00 | 37.29 | 26.32 | 50.00 | 83.87 |
| 4 | `tournament-cell-7-cot_scaffold` | 281 | 100.00 | 32.20 | 24.47 | 47.83 | 90.62 |
| 5 | `bert-consultant-fewshot10-claude-opus-n50` | 271 | 100.00 | 33.90 | 20.88 | 50.00 | 88.89 |
| 6 | `tournament-cell-5-negative_exemplars` | 282 | 100.00 | 30.51 | 27.08 | 45.65 | 90.32 |
| 7 | `t4-bert-qwen27b-fewshot10-n50-fixed` | 282 | 100.00 | 45.76 | 31.91 | 36.96 | 78.79 |
| 8 | `bert-claude-opus-top3-n681` | 3794 | 99.27 | 36.79 | 23.95 | 45.36 | 87.78 |
| 9 | `t4-bert-a3b-fewshot10-n50-fixed` | 288 | 100.00 | 50.85 | 39.58 | 36.00 | 66.67 |
| 10 | `bert-gemma-composed-top3-n50` | 278 | 100.00 | 33.90 | 24.47 | 46.51 | 87.50 |
| 11 | `bert-consultant-fewshot10-gemma-n50` | 284 | 100.00 | 33.90 | 25.53 | 44.68 | 88.24 |
| 12 | `bert-claude-sonnet-top3-n681` | 3840 | 99.27 | 35.61 | 26.48 | 45.30 | 84.20 |
| 13 | `tournament-cell-9-persona` | 275 | 100.00 | 35.59 | 30.77 | 39.13 | 82.76 |
| 14 | `bert-claude-sonnet-fewshot10-n50` | 267 | 100.00 | 30.51 | 24.73 | 35.71 | 95.65 |
| 15 | `bert-consultant-fewshot10-claude-sonnet-n50` | 281 | 100.00 | 33.90 | 20.21 | 44.68 | 87.10 |
| 16 | `tournament-cell-4-per_state_exemplars` | 285 | 100.00 | 27.12 | 29.79 | 45.65 | 83.33 |
| 17 | `bge-small-bert-qwen27b-fewshot10-n50-fixed` | 271 | 100.00 | 35.59 | 23.66 | 41.86 | 84.62 |
| 18 | `bert-consultant-fewshot10-n50` | 276 | 100.00 | 33.90 | 18.09 | 46.51 | 86.67 |
| 19 | `bert-consultant-a3b-mini` | 128 | 100.00 | 25.00 | 21.43 | 52.63 | 85.71 |
| 20 | `bert-consultant-fewshot10-gemma-mini` | 137 | 100.00 | 25.00 | 25.00 | 45.45 | 88.89 |
| 21 | `bert-a3b-composed-top3-n50` | 276 | 100.00 | 32.20 | 19.57 | 45.65 | 86.21 |
| 22 | `t4-bert-gemma-fewshot10-n50-fixed` | 285 | 100.00 | 42.37 | 33.33 | 38.30 | 66.67 |
| 23 | `bert-consultant-fewshot10-a4b-n50` | 274 | 100.00 | 27.12 | 26.88 | 40.91 | 85.71 |
| 24 | `bert-claude-opus-fewshot10-n50` | 272 | 100.00 | 32.20 | 24.21 | 37.21 | 84.00 |
| 25 | `t4-bert-qwen27b-nothink-fewshot10-n50-fixed` | 291 | 100.00 | 52.54 | 34.38 | 32.00 | 58.33 |
