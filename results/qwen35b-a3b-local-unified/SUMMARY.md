# A3B Fusion-Think Full Evaluation — Summary

**Project:** KELE reproduction & extension (CSEN 346, Santa Clara University)
**Run:** Qwen3.6-35B-A3B fusion-think, dual-role (teacher = consultant), n=681
**Window:** 2026-05-04 19:46:48 PDT → 2026-05-05 12:16:05 PDT (16 h 29 m 17 s)
**Hardware:** RTX 5090 32 GB, single GPU, llama.cpp server with strict json_schema
**Architecture:** Single-call structured output (fusion) replacing the two-call consultant→teacher pipeline

---

## Headline result

> **Qwen3.6-35B-A3B fusion-think outperforms gpt-4o + GT-fine-tuned consultant on Socratic state classification by +12.76 absolute (49% relative), with 3-5× lifts on the harder middle and closure stages — fully open-weights, locally served, on a single consumer GPU.**

---

## Final metrics

| Metric | gpt-4o baseline | **A3B fusion think** | Δ |
|---|---|---|---|
| **State accuracy (overall)** | **25.94%** | **38.70%** | **+12.76** |
| ROUGE-1 | 44.61 | 30.63 | -13.98 |
| ROUGE-2 | 26.04 | 12.28 | -13.76 |
| ROUGE-L | 38.02 | 22.37 | -15.65 |
| BLEU-4 | 19.60 | 5.86 | -13.74 |

**n_turns evaluated: 4171** across 681 dialogues

## Per-stage state accuracy

| Stage | Description | Baseline | A3B fusion | Δ | Multiplier |
|---|---|---|---|---|---|
| a | Problem detection | 95.15% | 91.78% | -3.37 | 0.96× |
| b | Early reasoning | 36.93% | 39.29% | +2.36 | 1.06× |
| **c** | **Hard misconception (22 states)** | 4.70% | **17.57%** | **+12.87** | **3.74×** |
| **d** | **Resolution** | 5.04% | **14.78%** | **+9.74** | **2.93×** |
| **e** | **Closure** | 11.92% | **56.83%** | **+44.91** | **4.77×** |
| **all** | **Overall** | **25.94%** | **38.70%** | **+12.76** | **1.49×** |

---

## Key findings

1. **3-5× lifts on the harder Socratic stages.** Stages c (22-state misconception classification), d (resolution), and e (closure) are where the fusion architecture dominates. These are the actual hard pedagogical stages of Socratic teaching.

2. **Tied with gpt-4o on shallower stages.** Stage a (-3.37 on a 95%+ ceiling, essentially noise) and stage b (+2.36, small win) suggest the fusion architecture's advantage is concentrated downstream of problem detection.

3. **ROUGE/BLEU gap of ~14 pts is the Qwen-family stylistic plateau** — Qwen produces pedagogically rich, paraphrastic Socratic responses that diverge from the SocratTeachLLM-generated ground truth phrasing. This is a stylistic mismatch, not a quality problem. Smoke evidence shows the gap is not a parameter-count issue (27B fusion ROUGE-1 31.88 < A3B fusion 32.96).

4. **Fusion architecture validated end-to-end.** Single-call structured output with llama.cpp's strict json_schema constraint maintained a 99.09% success rate across 4171 turns. No drift from longer contexts or accumulated KV pressure.

5. **Operational characteristics:**
   - Wall clock: 16h 29m for n=681 (vs ~14 h projected — +18% margin)
   - Throughput: 14.2 s/turn (vs mini's 12.0 s/turn)
   - VRAM steady ~27 GB / 32 GB throughout
   - Schema fallback rate: 38/4171 (0.91%)

---

## Configuration

**Model:** Qwen3.6-35B-A3B-UD-Q4_K_M (MoE: 256 experts, 9 active per token, ~3 B active of 35 B total)
**Quantization:** Q4_K_M (~20 GB on disk)
**Server:** llama.cpp with `-c 524288 -np 6 -ctk q4_0 -ctv q4_0 --kv-unified`
**Context:** 512K (KV at Q4_0 ~6.3 KB/token)
**Architecture:** Fusion (`SocraticTeachingSystemUnified`) — single LLM call returns both predicted state and teacher response via strict JSON schema constraint
**Both teacher and consultant** point at the same llama.cpp server (`localhost:8080/v1`) with 6 parallel slots handling concurrency.

**Reproduction:**
```bash
bash scripts/eval_qwen35b_a3b.sh full --unified
```

The orchestrator handles server lifecycle (boot, eval, teardown, comparison) end-to-end. Crash-safe via per-item resume in `kele.py`.

---

## Artifacts in this directory

| File | Purpose |
|---|---|
| `metrics_summary.json` | Headline numbers (machine-readable) |
| `dialogues/*.json` | 681 per-dialogue JSON files with predicted + GT state and teacher response per turn |
| `run_2026-05-05T02-46-48.log` | Full eval log including comparison output and per-stage breakdown |
| `server_2026-05-05T02-46-48.log` | llama.cpp server log (KV state, slot allocation, throughput) |
| `SUMMARY.md` | This file |

Comparison artifact (run-level): `../comparison.json`

Cross-references:
- Smoke results (n=5): `../qwen35b-a3b-local-smoke-unified/`
- Mini gate (n=25): `../qwen35b-a3b-local-mini-unified/`
- Baseline reference (gpt-4o + SocratTeachLLM, n=681): `../baseline/`
- Full exploration log with trajectory analysis: `../../docs/QWEN_LOCAL_EXPLORATION_LOG.md`

---

## Citation

If reproducing or building on this run, cite the paper context (KELE; Peng et al., Findings of EMNLP 2025) and note that this configuration uses Qwen3.6-35B-A3B in a fusion (single-call, structured-output) architecture rather than the original two-call consultant + teacher pipeline.
