# Qwen Local Exploration Log

**CSEN 346 · Santa Clara University · started 2026-05-04**

## 🏆 Full run results — A3B fusion think (n=681)

**Run window:** 2026-05-04 19:46:48 PDT → 2026-05-05 12:16:05 PDT
**Total wall clock:** **16 h 29 m 17 s** (vs ~14 h projected; +18%)
**Throughput:** 4171 turns / 16.5 h = 4.21 turns/min ≈ **14.2 s/turn** (mini was 12.0 s/turn)
**Schema fallback rate:** **38 / 4171 turns (0.91%)** — well under the 5% gate
**Output dir:** `results/qwen35b-a3b-local-unified/`

### Headline result

> **Qwen3.6-35B-A3B fusion-think outperforms gpt-4o + GT-fine-tuned consultant on Socratic state classification by +12.76 absolute (49% relative), with 3-5× lifts on the harder middle and closure stages, while fully open-weights and locally served on a single RTX 5090.**

### Final metrics vs gpt-4o baseline

| Metric | gpt-4o baseline (n=681) | **A3B fusion think (n=681)** | Δ | Multiplier |
|---|---|---|---|---|
| **State acc overall** | **25.94%** | **38.70%** | **+12.76** | **1.49×** |
| ROUGE-1 | 44.61 | 30.63 | -13.98 | 0.69× |
| ROUGE-2 | 26.04 | 12.28 | -13.76 | 0.47× |
| ROUGE-L | 38.02 | 22.37 | -15.65 | 0.59× |
| BLEU-4 | 19.60 | 5.86 | -13.74 | 0.30× |

### Per-stage state accuracy

| Stage | Baseline | A3B fusion | Δ | Multiplier | Read |
|---|---|---|---|---|---|
| a (problem detection) | 95.15% | 91.78% | -3.37 | 0.96× | 🟡 essentially tied |
| b (early reasoning) | 36.93% | 39.29% | +2.36 | 1.06× | 🟢 small win |
| **c (hard misconception, 22 states)** | 4.70% | **17.57%** | **+12.87** | **3.74×** | 🟢🟢 **big win — hardest task** |
| **d (resolution)** | 5.04% | **14.78%** | **+9.74** | **2.93×** | 🟢🟢 big win |
| **e (closure)** | 11.92% | **56.83%** | **+44.91** | **4.77×** | 🟢🟢🟢 dominant |
| **Overall** | **25.94%** | **38.70%** | **+12.76** | **1.49×** | 🟢 publishable |

### Trajectory across the run (live snapshots)

| Stage of project | n | State acc | Δ baseline | Notes |
|---|---|---|---|---|
| Smoke | 5 | 42.42% | +16.48 | optimistic outlier (33 turns) |
| Mini gate | 25 | 35.17% | +9.23 | pessimistic stratified sample |
| Full @ 48 dlg | 48 | 43.06% | +17.12 | early-sample peak begins |
| Full @ 73 dlg | 73 | 43.53% | **+17.59** | **observed peak** |
| Full @ 145 dlg | 145 | 40.47% | +14.53 | decay phase |
| Full @ 502 dlg | 502 | 38.97% | +13.03 | convergence floor |
| Full @ 670 dlg | 670 | 38.79% | +12.85 | fully converged |
| **Full final** | **681** | **38.70%** | **+12.76** | **canonical** |

### Findings

1. **3-5× lifts on the harder Socratic stages.** Stages c, d, e — the actual hard pedagogical work — are where fusion-think dominates: **3.74× on the 22-state misconception classification (c), 2.93× on resolution (d), 4.77× on closure (e)**. This is the paper's core empirical claim.

2. **Stages a and b are roughly tied with gpt-4o.** Stage a -3.37 (noise-level on a 95%+ ceiling), stage b +2.36 (small win). The fusion architecture's gains are concentrated downstream of problem detection.

3. **Convergence shape was instructive — and sobering.** First 73 dialogues read +17.59 (peak); steady-state at n=681 is +12.76. Smoke (+16.48) overshot reality; mini (+9.23) undershot. **Future small-n gating should average smoke + mini as the better predictor (+12.86 ≈ true)** rather than treating either as canonical. This is a methodological note worth disclosing in the paper.

4. **ROUGE/BLEU gap (~14 pts) is the Qwen-family stylistic plateau.** Not a model-size issue (smoke evidence: 27B fusion ROUGE-1 31.88 < A3B fusion 32.96). Qwen produces pedagogically rich, paraphrastic Socratic responses that diverge from SocratTeachLLM-generated GT phrasing. **This is the target for the 5/14 prompt-engineering and LoRA experiments.**

5. **Schema enforcement is production-reliable.** llama.cpp's strict json_schema constraint kept the unified-call success rate at 99.09% across 4171 turns. Fallback rate held flat throughout the run — no drift from longer contexts or accumulated KV pressure.

6. **Wall clock came in 18% over projection** (16.5 h actual vs 14 h projected). The mini's 12.0 s/turn underestimated the full run's 14.2 s/turn — likely because the test set's later-ID dialogues are slightly longer on average (the mini sampled lower IDs only). **Future projections: use mini × 1.18 as a wall-clock margin.**

### Artifacts

| Path | Contents |
|---|---|
| `results/qwen35b-a3b-local-unified/metrics_summary.json` | Headline numbers (4171 turns, 5 stages) |
| `results/qwen35b-a3b-local-unified/dialogues/*.json` | 681 per-dialogue outputs with full state + teacher response traces |
| `results/qwen35b-a3b-local-unified/run_2026-05-05T02-46-48.log` | Full eval log: per-dialogue progress, comparison output, per-stage table |
| `results/qwen35b-a3b-local-unified/server_2026-05-05T02-46-48.log` | llama.cpp server log (KV state, slot allocation) |
| `results/qwen35b-a3b-local-unified/SUMMARY.md` | Paper-ready standalone snapshot |
| `results/comparison.json` | Side-by-side baseline vs A3B comparison |

### What this clears

- ✅ **2026-05-05 deliverable: 3rd commit + Evaluation & Results.** Headline + per-stage + per-dialogue traces all locked.
- ⏳ **2026-05-14 deliverable: 4th commit + paper draft.** ROUGE recovery via Options 2 (prompt eng) and 3 (LoRA) is the open work.
- ⏳ **2026-06-04 deliverable: Final paper + code + HuggingFace data + poster.** Run artifacts above are ready to ship.

---


## Earlier exploration history

The full 2026-05-04 → 2026-05-05 exploration log — Qwen 27B vs A3B smoke matrix, two-call-vs-fusion deltas, mini gate results, ROUGE diagnosis, decision matrices, fusion plan discussion — is preserved in [`archive/QWEN_LOCAL_EXPLORATION_LOG_full.md`](archive/QWEN_LOCAL_EXPLORATION_LOG_full.md). That archived copy is the authoritative source for the methodological reasoning that led to the A3B-fusion-think locked headline above; it is also the source the paper cites for the smoke–mini averaging finding.

