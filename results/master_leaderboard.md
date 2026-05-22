# Master leaderboard

Configs available: 20/21


## Sorted by composite (state + 0.5*R-1)

| Config | State | R-1 | R-2 | BLEU-4 | Semantic R-1 | LLM-judge | Composite | n_turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 31B + top-3 stack | 50.72 | 41.13 | 18.60 | 12.91 | 0.7326 | 8.17 | 71.28 | 278 |
| Opus 4.6 + BERT + top-3 | 49.82 | 42.77 | 21.12 | 15.53 | 0.7523 | 8.08 | 71.20 | 271 |
| Sonnet 4.6 + BERT + top-3 (n=681 Phase 3) | 49.97 | 41.93 | 20.43 | 13.46 | — | 8.19 | 70.94 | 3840 |
| Gemma 4 31B + 10-shot (n=50 ref) | 51.06 | 38.53 | 16.93 | 9.68 | — | — | 70.33 | 284 |
| Sonnet 4.6 + BERT + top-3 | 48.75 | 43.02 | 20.52 | 14.33 | 0.7500 | 8.11 | 70.26 | 281 |
| Opus 4.6 + BERT + top-3 (n=681 Phase 3) | 49.31 | 41.63 | 19.87 | 13.91 | — | 8.01 | 70.12 | 3794 |
| Sonnet 4.6 + BERT + 10-shot only | 47.94 | 39.68 | 19.40 | 10.15 | 0.7395 | 7.84 | 67.78 | 267 |
| Qwen 35B-A3B + top-3 stack | 48.19 | 37.64 | 15.69 | 9.97 | 0.7274 | 7.49 | 67.01 | 276 |
| Gemma 4 31B + 10-shot (n=681 LOCKED) | 48.15 | 36.78 | 16.10 | 9.05 | 0.7233 | 8.19 | 66.54 | 3834 |
| Opus 4.6 + BERT + 10-shot only | 47.43 | 32.99 | 15.26 | 7.24 | 0.7289 | 7.44 | 63.92 | 272 |
| Qwen 35B-A3B + 10-shot (n=681) | 46.57 | 33.27 | 13.89 | 6.96 | 0.7200 | 7.42 | 63.20 | 3762 |
| Opus 4.6 consultant + SocratTeachLLM (n=50) | 38.44 | 47.58 | 28.06 | 21.21 | 0.7692 | 7.80 | 62.23 | 307 |
| Sonnet 4.6 + BERT raw (no exemplars) | 45.00 | 29.10 | 13.38 | 5.69 | 0.7207 | 7.25 | 59.55 | 260 |
| Opus 4.6 consultant + SocratTeachLLM (EN) | 30.26 | 44.22 | 26.20 | 2.96 | — | 6.77 | 52.37 | 304 |
| Opus 4.6 + BERT raw (no exemplars) | 39.75 | 23.28 | 10.12 | 4.18 | 0.7170 | 6.80 | 51.39 | 239 |
| Sonnet 4.6 consultant + SocratTeachLLM (EN) | 22.11 | 55.85 | 33.79 | 3.56 | — | 6.62 | 50.03 | 303 |
| GPT-4o + SocratTeachLLM (paper baseline) | 25.94 | 44.61 | 26.04 | 19.60 | — | — | 48.25 | 4294 |
| Sonnet 4.6 consultant + SocratTeachLLM (n=50 clean rerun) | 18.57 | 45.61 | 26.70 | 20.45 | — | 7.63 | 41.38 | 307 |
| Opus 4.6 + BERT + top-3 (EN) | 34.44 | 0.47 | 0.19 | 0.01 | — | 8.01 | 34.67 | 270 |
| Sonnet 4.6 consultant + SocratTeachLLM (n=50) | 0.00 | 43.88 | 24.45 | 16.40 | 0.7437 | — | 21.94 | 308 |

## Sorted by surface-form sum (R-1 + R-2 + BLEU-4) — KELE-paper-style

| Config | R-1 | R-2 | BLEU-4 | Sum | State | LLM-judge | Semantic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opus 4.6 consultant + SocratTeachLLM (n=50) | 47.58 | 28.06 | 21.21 | 96.85 | 38.44 | 7.80 | 0.7692 |
| Sonnet 4.6 consultant + SocratTeachLLM (EN) | 55.85 | 33.79 | 3.56 | 93.20 | 22.11 | 6.62 | — |
| Sonnet 4.6 consultant + SocratTeachLLM (n=50 clean rerun) | 45.61 | 26.70 | 20.45 | 92.76 | 18.57 | 7.63 | — |
| GPT-4o + SocratTeachLLM (paper baseline) | 44.61 | 26.04 | 19.60 | 90.25 | 25.94 | — | — |
| Sonnet 4.6 consultant + SocratTeachLLM (n=50) | 43.88 | 24.45 | 16.40 | 84.73 | 0.00 | — | 0.7437 |
| Opus 4.6 + BERT + top-3 | 42.77 | 21.12 | 15.53 | 79.42 | 49.82 | 8.08 | 0.7523 |
| Sonnet 4.6 + BERT + top-3 | 43.02 | 20.52 | 14.33 | 77.87 | 48.75 | 8.11 | 0.7500 |
| Sonnet 4.6 + BERT + top-3 (n=681 Phase 3) | 41.93 | 20.43 | 13.46 | 75.82 | 49.97 | 8.19 | — |
| Opus 4.6 + BERT + top-3 (n=681 Phase 3) | 41.63 | 19.87 | 13.91 | 75.41 | 49.31 | 8.01 | — |
| Opus 4.6 consultant + SocratTeachLLM (EN) | 44.22 | 26.20 | 2.96 | 73.38 | 30.26 | 6.77 | — |
| Gemma 4 31B + top-3 stack | 41.13 | 18.60 | 12.91 | 72.64 | 50.72 | 8.17 | 0.7326 |
| Sonnet 4.6 + BERT + 10-shot only | 39.68 | 19.40 | 10.15 | 69.23 | 47.94 | 7.84 | 0.7395 |
| Gemma 4 31B + 10-shot (n=50 ref) | 38.53 | 16.93 | 9.68 | 65.14 | 51.06 | — | — |
| Qwen 35B-A3B + top-3 stack | 37.64 | 15.69 | 9.97 | 63.30 | 48.19 | 7.49 | 0.7274 |
| Gemma 4 31B + 10-shot (n=681 LOCKED) | 36.78 | 16.10 | 9.05 | 61.93 | 48.15 | 8.19 | 0.7233 |
| Opus 4.6 + BERT + 10-shot only | 32.99 | 15.26 | 7.24 | 55.49 | 47.43 | 7.44 | 0.7289 |
| Qwen 35B-A3B + 10-shot (n=681) | 33.27 | 13.89 | 6.96 | 54.12 | 46.57 | 7.42 | 0.7200 |
| Sonnet 4.6 + BERT raw (no exemplars) | 29.10 | 13.38 | 5.69 | 48.17 | 45.00 | 7.25 | 0.7207 |
| Opus 4.6 + BERT raw (no exemplars) | 23.28 | 10.12 | 4.18 | 37.58 | 39.75 | 6.80 | 0.7170 |
| Opus 4.6 + BERT + top-3 (EN) | 0.47 | 0.19 | 0.01 | 0.67 | 34.44 | 8.01 | — |

## Sorted by LLM-judge (memorization-resistant)

| Config | LLM-judge | State | R-1 | Semantic | Surface sum |
|---|---:|---:|---:|---:|---:|
| Sonnet 4.6 + BERT + top-3 (n=681 Phase 3) | 8.19 | 49.97 | 41.93 | — | 75.82 |
| Gemma 4 31B + 10-shot (n=681 LOCKED) | 8.19 | 48.15 | 36.78 | 0.7233 | 61.93 |
| Gemma 4 31B + top-3 stack | 8.17 | 50.72 | 41.13 | 0.7326 | 72.64 |
| Sonnet 4.6 + BERT + top-3 | 8.11 | 48.75 | 43.02 | 0.7500 | 77.87 |
| Opus 4.6 + BERT + top-3 | 8.08 | 49.82 | 42.77 | 0.7523 | 79.42 |
| Opus 4.6 + BERT + top-3 (n=681 Phase 3) | 8.01 | 49.31 | 41.63 | — | 75.41 |
| Opus 4.6 + BERT + top-3 (EN) | 8.01 | 34.44 | 0.47 | — | 0.67 |
| Sonnet 4.6 + BERT + 10-shot only | 7.84 | 47.94 | 39.68 | 0.7395 | 69.23 |
| Opus 4.6 consultant + SocratTeachLLM (n=50) | 7.80 | 38.44 | 47.58 | 0.7692 | 96.85 |
| Sonnet 4.6 consultant + SocratTeachLLM (n=50 clean rerun) | 7.63 | 18.57 | 45.61 | — | 92.76 |
| Qwen 35B-A3B + top-3 stack | 7.49 | 48.19 | 37.64 | 0.7274 | 63.30 |
| Opus 4.6 + BERT + 10-shot only | 7.44 | 47.43 | 32.99 | 0.7289 | 55.49 |
| Qwen 35B-A3B + 10-shot (n=681) | 7.42 | 46.57 | 33.27 | 0.7200 | 54.12 |
| Sonnet 4.6 + BERT raw (no exemplars) | 7.25 | 45.00 | 29.10 | 0.7207 | 48.17 |
| Opus 4.6 + BERT raw (no exemplars) | 6.80 | 39.75 | 23.28 | 0.7170 | 37.58 |
| Opus 4.6 consultant + SocratTeachLLM (EN) | 6.77 | 30.26 | 44.22 | — | 73.38 |
| Sonnet 4.6 consultant + SocratTeachLLM (EN) | 6.62 | 22.11 | 55.85 | — | 93.20 |

## Configs awaiting data

- Gemma 4 31B + top-3 (n=200 validation)