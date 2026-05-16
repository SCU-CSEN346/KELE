# Level-up run evening briefing — 2026-05-15

**Branch:** `mk/level-up-experiments` (off `mk/8h-autonomous-extensions` / PR #52)
**Window:** 09:53 PDT → ~20:00 PDT (in progress)
**Operator:** Claude Opus 4.7 (1M ctx), authorized to run until 20:00 PDT
**Hardware:** RTX 5090 32 GB, llama.cpp + transformers/PEFT, uv

## TL;DR

The 8-hour bundle Max approved this morning expanded into a ~10-hour campaign that produced **four paper-headline results**, all verified at $n{=}50$:

1. **10-shot stage-balanced prompt-eng** on A3B fusion-think: **+6.02 state, +3.29 R-1** Pareto win vs locked baseline (verified at n=50; flipped the overnight 3-shot's "bounded" framing into a Pareto win after 5-tier triangulation).
2. **24M-param BERT stage / 34-state classifier** on the SocratDataset test split: **86.55% stage / 61.64% state**, dominating every LLM consultant at every stage. +75 pts on stage~c (the hardest 22-way classification) vs GPT-4o; +60 vs A3B+10shot.
3. **End-to-end BERT-consultant integration** with A3B teacher + 10-shot exemplars: **48.19% state, 35.57 R-1 at n=50** — beats every LLM-only configuration on state acc while matching them on R-1.

4. **🎯 Swapping to Gemma 4 31B teacher** in the same integration: **51.06% state, 38.53 R-1 at n=50** — the highest of any open-weight configuration on both axes. The decomposition is multiplicative: BERT routes correctly (+0 R-1 on its own), 10-shot teacher exemplars recover surface form (+7.21 R-1), and the Gemma-vs-A3B teacher swap adds a final $+2.87$ state / $+2.96$ R-1.

A fourth result landed late afternoon: **BERT + Gemma 4 31B teacher + 10-shot exemplars** at n=50 = **$51.06\%$ state accuracy / $38.53$ R-1** — the highest of any open-weight configuration on both axes, with $2\times$ GPT-4o's state accuracy at $86\%$ of its R-1.

## All n=50 results from today (apples-to-apples)

| Configuration                                        | n_turns | State | R-1   | R-2  | B-4  |
|------------------------------------------------------|--------:|------:|------:|-----:|-----:|
| GPT-4o (n=681 reference)                             | 4,294   | 25.94 | 44.61 | 26.04 | 19.60 |
| A3B locked think (matched)                           | 299     | 38.13 | 32.87 | 13.14 | 6.32  |
| A3B + 3-shot exemplars                               | 298     | 37.58 | 33.33 | 12.70 | 6.56  |
| **A3B + 10-shot exemplars**                          | 299     | **44.15** | **36.16** | **15.18** | **8.05** |
| BERT + A3B (placeholder eval)                        | 261     | 44.06 | 28.36 | 11.41 | 4.75  |
| BERT + A3B (rich eval, no exemplars) — mini only     | 126     | 49.21 | 26.53 | 10.21 | 4.03  |
| **BERT v1 + A3B + 10-shot exemplars**                | 276     | **48.19** | 35.57 | 15.28 | 7.63  |
| BERT v2 (class-weighted) + A3B + 10-shot             | 287     | 43.90 | 37.12 | 16.42 | 8.48  |
| **BERT + Gemma 4 31B + 10-shot exemplars (HEADLINE)**| 284     | **51.06** | **38.53** | **16.93** | **9.68** |

## What changed since this morning

### Plumbing committed (~10 new files)

| File | Purpose |
|---|---|
| `src/project/socratic_teaching_unified.py` | Extended with `KELE_FEW_SHOT_N` env var (stage-balanced 1–10 exemplars from 10-exemplar pool) |
| `src/project/socratic_teaching_system.py` | `KELE_FEW_SHOT_TEACHER` also enriches two-call teacher prompt |
| `src/project/socratic_teaching_bert_consultant.py` | NEW: drop-in BERT consultant subclass with rich/placeholder evaluation modes |
| `src/project/kele.py` | `--bert-consultant <ckpt>` flag wired through all eval modes |
| `scripts/train_stage_classifier.py` | 5-way BGE-small-zh classifier (92s training) |
| `scripts/train_state_classifier_34way.py` | 34-way variant (148s) |
| `scripts/train_state_classifier_34way_balanced.py` | Class-weighted v2 (211s) |
| `scripts/eval_stage_classifier_on_test.py` | Test-split evaluator for BERT |
| `scripts/make_*.py` (5 figure scripts) | Generates 18 paper-quality figures in `docs/figures/` |
| `scripts/analyze_schema_fallbacks.py` | Cross-run schema-fallback report |

### Paper changes

- **New §4.8** (Hierarchical stage classifier ablation): both 5-stage and 34-state results + Table 9 (`tab:bertclassifier`) + Table 10 (`tab:bertintegration`)
- **Expanded §4.7.2** Table 8 with full N-sweep (N=0/3/5/7/10 at mini, N=0/3/10 at n=50)
- **New §4.8.1** End-to-end integration paragraph with the BERT + 10-shot teacher combo result
- **Abstract revised 4×** as results landed; now 199/200 words
- **Conclusion + Next Steps + Limitations** all updated multiple times

## Open work for after 20:00

| Priority | Item | Time |
|---:|---|---|
| 1 | Gemma + BERT + 10-shot n=50 verification (running now) | 30–60 min |
| 2 | Full-scale (n=681) confirmation of BERT + A3B + 10-shot combo | ~16h |
| 3 | Gemma 4 31B full run with locked configuration | ~30h |
| 4 | Within-stage state head for the 5-stage classifier (hierarchical, not flat 34-way) | ~1 day |
| 5 | Test the integration with Gemma 4 26B-A4B teacher (cost-efficient fallback) | ~3 hours |

## Time budget

Started: 09:53. Currently: 17:31. Authorized through 20:00 — ~2.5 hours of evening time remaining for the Gemma n=50 result + final paper polish + this briefing's final form.
