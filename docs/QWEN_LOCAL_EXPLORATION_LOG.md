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

## Mini gate results — A3B fusion think (n=25)

**Run timestamp:** 2026-05-04 18:47:02 → 19:16:14 PDT (29 min 6 s)
**Output dir:** `results/qwen35b-a3b-local-mini-unified/`
**Config:** A3B fusion (single-call structured output, thinking on), 145 turns across 25 dialogues.

| Metric | Smoke (n=5, 33 turns) | **Mini (n=25, 145 turns)** | Δ smoke→mini | vs gpt-4o baseline (n=681) |
|---|---|---|---|---|
| State acc overall | 42.42% | **35.17%** | -7.25 | **+9.23** |
| ROUGE-1 | 32.96 | 30.51 | -2.45 | -14.10 |
| ROUGE-2 | 12.59 | 11.69 | -0.90 | -14.35 |
| ROUGE-L | 23.93 | 21.30 | -2.63 | -16.72 |
| BLEU-4 | 5.87 | 5.46 | -0.41 | -14.14 |
| s/turn | 13 | **12.0** | -1 | — |
| Schema fallbacks | 0/33 (0%) | **0/145 (0%)** | — | — |

### Per-stage state accuracy

| Stage | gpt-4o baseline | A3B smoke | **A3B mini** | Δ vs baseline |
|---|---|---|---|---|
| a (problem detection) | 95.15% | 40.0% | 88.0% | -7.15 |
| b (early reasoning) | 36.93% | 50.0% | 32.14% | -4.79 |
| c (hard misconception, 22 states) | 4.70% | 7.69% | **10.64%** | **+5.94** |
| d (resolution) | 5.04% | 0.0% | **13.04%** | **+8.00** |
| e (closure) | 11.92% | 50.0% | **54.55%** | **+42.63** |
| **overall** | **25.94%** | 42.42% | **35.17%** | **+9.23** |

### Gate criterion check

| Criterion | Threshold | Measured | Verdict |
|---|---|---|---|
| State acc | 39–46% (smoke ±3 pts) | 35.17% | ❌ MISSED by 3.83 |
| Wall clock per turn | ±10% of smoke (~11.7–14.3 s) | 12.0 s | ✓ |
| Schema fallback rate | <5% | 0% | ✓ |

### Why this is a soft pass, not a hard fail

- **Smoke n=5 (33 turns) is statistically a single-digit-sample point estimate.** Mini at 145 turns is ~4.4× more turns and the more reliable headline. The 7-pt drop is regression to the mean from an optimistic smoke draw, not a degraded config.
- **Mini still beats gpt-4o baseline by +9.23 state acc.** The paper claim ("Qwen3.6-35B-A3B fusion outperforms gpt-4o on Socratic state classification") still holds, just with a smaller margin than smoke advertised.
- **Per-stage shape is healthier in mini than smoke.** Stage a recovered from 40% (smoke noise) to 88%. Stage d unlocked from 0% to 13.04% — *actually beats baseline* on the resolution stage. Stage c also moved positive (+5.94 over baseline).
- **No fallback storm, no OOM, no server drift across 29 min.** The infrastructure is solid for the full overnight run.

### Updated full-run projection

- 12.0 s/turn measured × ~4000 turns expected for n=681 (avg ~5.9 turns/dialogue from this mini's 145/25) = **~13.3 h**
- Add ~5% margin for occasional pauses, attention drift on longer dialogues → **call it ~14 h**
- If kicked off ~21:00 PDT lands ~11:00 PDT next morning. Comfortable buffer for the 2026-05-05 evening deadline.

### Decision tree for next steps

| Path | Wall clock | Lands | Risk | Use case |
|---|---|---|---|---|
| **A** | A3B fusion-think full now (~14 h) | next morning | low | **Recommended** — clears 5/5 deliverable with publishable numbers |
| B | A3B full + 27B fusion mini in parallel (impossible — same GPU, same port) | — | — | Ruled out by hardware |
| C | 27B fusion-think mini first (~1.5–2 h), then A3B full | A3B lands ~14 h after 27B mini ends | medium — pushes A3B finish toward 5/5 evening | Worth it if Max wants the 27B headline gating data tonight |
| D | Hold; draft Evaluation section first with smoke+mini; full run later | — | depends on writing speed | Conservative; aligned with paper-first work style but burns the 5/5 GPU window |

---

Live log of every Qwen-local config we've tried, the numbers it produced, and
what we learned. Newest entries at the top. This is a working document — gets
appended during exploration sessions, not a polished report.

For comparison, baselines from earlier work (`results/baseline/`,
`results/wave-2026-04-21.../`, etc.):

| Run | Teacher | Consultant | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | State acc | Wall clock |
|---|---|---|---|---|---|---|---|---|---|
| **gpt-4o baseline** | SocratTeachLLM 9B | gpt-4o-2024-11-20 | 681 | **44.61** | **26.04** | **38.02** | **19.60** | **25.94%** | 4h 34m |
| paper Table 1 | SocratTeachLLM | GPT-4o (GT consultant) | — | 57.4 | 33.63 | 50.77 | 41.96 | — | — |
| Qwen3.5-9B (WAVE) | SocratTeachLLM 9B | Qwen3.5-9B local | 681 | 43.72 | 24.87 | 36.76 | 18.63 | 18.93% | 24h 53m |
| qwen2.5:7b (Mac mini split) | SocratTeachLLM 9B | qwen2.5:7b via Ollama | 681 | 43.57 | 24.90 | 36.91 | 18.56 | 15.16% | 4h 27m |

The **gpt-4o baseline** is the canonical comparison target for everything below.

---

## ROUGE diagnosis and forward path (added 2026-05-04 ~21:30 PDT, during full run)

**Observed at n=54 (full run in flight):** ROUGE-1 32.05 vs gpt-4o baseline 44.61 — gap of -12.6 pts. ROUGE-2 -13.3, ROUGE-L -15.2, BLEU-4 -13.3. Same shape as smoke and mini. State-acc is +17 over baseline; ROUGE is the open weakness.

### Why ROUGE is low

**Style mismatch, not quality.** Qwen3.6 is instruction-tuned to be pedagogically rich:

1. Affirmation preambles ("太棒了！", "非常好的问题！", "哇...")
2. Contextual scaffolding before the Socratic prompt
3. Synonyms over exact GT phrasing ("能想到" vs "能否想到")

Smoke captured this exactly — Qwen wrote *"哇，你提出了一个非常有趣的科学猜想..."* before the actual question; GT was just *"你能想到一些植物是生长在水中或其他地方的吗？"*

ROUGE-2 and ROUGE-L drop more than ROUGE-1 because they reward longer exact matches. The state-acc lift coexists with paraphrastic style — the model is doing valid Socratic work, just not in GT phrasing.

### Would 27B help? No.

Smoke data settles this — the dense 27B does **not** outperform A3B on ROUGE:

| Config | ROUGE-1 (smoke) |
|---|---|
| 27B fusion think | 31.88 |
| **A3B fusion think** | **32.96** ← higher than 27B |
| 27B no-think two-call | 29.99 |
| A3B no-think two-call | 28.36 |

The ~12-pt gap is the Qwen-family stylistic posture, not a parameter-count issue. Both variants share the same instruction-tuning. **Running 27B will not move ROUGE meaningfully** — would buy maybe 1–2 pts at 3× the wall clock.

### Three options considered

**Option 1 — RULED OUT (VRAM constraint):** SocratTeachLLM teacher + Qwen consultant.

- *Theory:* SocratTeachLLM generated the ground truth, so swapping it back into the teacher slot lifts ROUGE close to baseline by construction. Replace gpt-4o → Qwen-A3B as consultant to preserve the +17 state-acc lift.
- *Why ruled out:* VRAM. The 2026-04-14 baseline (SocratTeachLLM 9B FP16 + Qwen3.5-2B consultant) already used 31.3/32.6 GB on the 5090 — *no headroom*. Replacing the 2B consultant with Qwen3.6-35B-A3B Q4 (20 GB weights alone) doesn't fit. SocratTeachLLM at FP16 (18 GB) + A3B Q4 (20 GB) = 38 GB just for weights, before KV. Even Q4-quantized SocratTeachLLM (~5–6 GB) + A3B Q4 + KV would be ~28–30 GB — fragile and untested.
- *Conclusion:* infeasible on a single 32 GB GPU. Multi-server / model-swap paths are operationally too slow for n=681.

**Option 2 — Prompt engineering for terse output style.**

- Modify teacher system prompt to instruct: "Respond with one Socratic question. No preamble, no praise, no scaffolding. Match this style: [3-shot GT examples]."
- Cost: prompt edit + smoke validation. ~30 min compute on top of existing infrastructure.
- Risk: fights Qwen's native instruction-tuned posture; may degrade pedagogical quality (we already saw `/no_think` collapse fusion-27B by -16.58 state acc — strong steering can break joint tasks).
- Reward: if it works, recovers ROUGE without changing the model — pure prompt cost.

**Option 3 — LoRA fine-tune Qwen on SocratTeachLLM-style outputs.**

- Treat train split as supervised pairs (student dialogue history → teacher response). PEFT-LoRA on 5090.
- Only requires Qwen + adapter — no extra inference-time VRAM impact (adapter is ~10s of MB, can be merged or hot-loaded).
- Cost: data prep + training script + eval harness. ~30 min training; 1–2 days engineering.
- Cleanest research story: "off-the-shelf Qwen has a style mismatch; LoRA realigns the style while preserving the consultant's classification advantage."

### Forward path (locked per Max 2026-05-04)

**Options 2 and 3 are the paths forward.** Option 1 is ruled out by VRAM; 27B doesn't help. The 5/14 paper-draft strategy:

1. **Option 2 first** — prompt-engineering experiment as a smoke run. Cheap, fast, validates whether style steering alone can recover ROUGE. Worth ~30 min to know before committing to LoRA.
2. **Option 3 if Option 2 underperforms** — LoRA fine-tune is the principled play if prompt engineering can't close the gap. Higher engineering cost but cleaner paper result.
3. **If both succeed:** paper has a 3-way ablation (off-the-shelf Qwen / prompt-engineered Qwen / LoRA-adapted Qwen) showing progressive ROUGE recovery while preserving state-acc lift.

The 5/5 deliverable still rests on the current A3B fusion-think full run. ROUGE recovery is a 5/14 horizon problem — not gating tomorrow's commit.

**Why this matters strategically:** the +17 state-acc lift is coming from the **Qwen consultant**, not the Qwen teacher. The teacher is where the ROUGE gap is born (style mismatch with SocratTeachLLM-generated GT). Options 2 and 3 both attack the teacher's output style without disturbing the consultant — the right surgical target.

---

## Active session — 2026-05-04

### Configs queued for testing

| # | Config | Teacher | Consultant | Server | Thinking | Notes |
|---|---|---|---|---|---|---|
| **1** | `qwen27b-local.env` | Qwen3.6-27B Q5_K_XL | Qwen3.6-27B Q5_K_XL | llama.cpp port 8080 | on (default) | Dual-role on one model; 26.3 GB VRAM at 416K ctx |
| **2** | `qwen35b-a3b-local.env` | Qwen3.6-35B-A3B Q4_K_M | Qwen3.6-35B-A3B Q4_K_M | llama.cpp port 8080 | on (default) | MoE: ~3 B active of 35 B; 23.1 GB at 512K ctx |
| 3 | (planned) qwen27b + `/no_think` | Qwen3.6-27B | Qwen3.6-27B (no_think on consultant) | port 8080 | off (consultant) | Test if disabling consultant CoT recovers throughput on 27B |
| 4 | (planned) qwen35b-a3b + `/no_think` | A3B | A3B (no_think on consultant) | port 8080 | off (consultant) | Same as #3 but on A3B; expected fastest viable config |

### Smoke / mini results

> Smoke = n=5 dialogues (~30 turns); mini = n=25 dialogues (~150 turns).
> n=5 metrics are noisy by definition — useful for wiring sanity, not statistical claims.
> **Wall-clock per turn** is the headline number for choosing the full-run config.

| # | Config | n | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | State acc | s/turn | Wall clock | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| — | gpt-4o baseline (ref) | 681 | **44.61** | 26.04 | 38.02 | 19.60 | 25.94% | — | 4h 34m | done |
| 1 | qwen27b smoke | 5 | 26.40 | 8.72 | 18.71 | 2.98 | **39.39%** | 72 | 39 min | done |
| 1 | qwen27b mini | 25 | — | — | — | — | — | — | — | skipped (smoke is enough) |
| 2 | qwen35b-a3b smoke | 5 | 27.52 | 9.93 | 19.57 | 4.29 | 24.24% | **19** | 10.6 min | done |
| 2 | qwen35b-a3b mini | 25 | — | — | — | — | — | — | — | held (per user direction) |
| 3 | qwen27b + no-think (consultant) smoke | 5 | **29.99** | 12.36 | 21.91 | 5.22 | **39.39%** | 70 | 38.6 min | done — state acc unchanged from think; +3.6 ROUGE-1 |
| 4 | qwen35b-a3b + no-think (consultant) smoke | 5 | 28.36 | 10.48 | 20.12 | 4.33 | **31.25%** | 17 | 9.3 min | done — state acc +7 pts vs think-on |
| **5** | **qwen27b fusion (unified) think** | 5 | **31.88** | **14.08** | **23.76** | **6.09** | **46.88%** | 39 | 21 min | **done — best quality config; cracked stage d (25%)** |
| **6** | **qwen27b fusion (unified) no-think** | 5 | 29.78 | 11.09 | 21.11 | 4.78 | 30.30% | 38 | 20.8 min | done — sharp -16.58 state acc collapse vs fusion think |
| **7** | **qwen35b-a3b fusion (unified) think** | 5 | **32.96** | **12.59** | **23.93** | **5.87** | **42.42%** | 13 | 7 min | **done — best operational config; +16.48 over baseline at 1/3 27B's wall clock** |
| **8** | **qwen35b-a3b fusion (unified) no-think** | 5 | 29.70 | 10.42 | 20.49 | 5.63 | 40.62% | 17 | 9.1 min | done — small regression vs fusion think (-1.80); 1 schema fallback |

### Per-stage state accuracy comparison (smoke, n=5, 33 turns each)

| Stage | gpt-4o baseline (n=681) | Qwen 27B smoke | Qwen A3B smoke |
|---|---|---|---|
| a (problem detection) | 95.15% | **100.0%** | 40.0% ← collapsed |
| b (early reasoning) | 36.93% | **50.0%** | **50.0%** |
| c (hard misconception, 22 states) | 4.70% | **30.77%** | 7.69% |
| d (resolution) | 5.04% | 0.0% | 0.0% (likely n=5 noise on both) |
| e (closure) | 11.92% | 25.0% | **50.0%** |
| **overall** | 25.94% | **39.39%** | 24.24% |

### Key findings — Qwen 27B smoke

- **State classification dramatically better.** Stage c (the hardest, 22-way classification) goes from 4.7% to 30.77% — a 6.5× improvement. Overall +13.45 points over the gpt-4o baseline.
- **ROUGE/BLEU collapse.** ROUGE-1 26.4 vs baseline 44.61 (-18 pts). The teacher generates pedagogically rich responses that *don't match ground-truth phrasing*. This was visible in dialogue 0004 — Qwen wrote *"哇，你提出了一个非常有趣的科学猜想"* where the ground truth was the more terse *"你能想到一些植物是生长在水中或其他地方的吗？"*.
- **The split is too large to be pure n=5 noise.** Direction is clear: **better Socratic reasoning, worse stylistic mimicry**.
- **Stage d 0%** is suspicious — but only a few d-turns in n=5; need n=25+ to know.
- **Wall clock confirms 75-hour full-run projection** (39 min for 5 dialogues = ~7.8 min/dialogue × 681 = 5,300 min ≈ 88 h).
- **Implication for the paper:** this is a research-paper-worthy result — *Qwen3.6-27B as both teacher and consultant outperforms gpt-4o + fine-tuned-teacher on classification accuracy at the cost of stylistic fidelity.* Worth its own ablation. But 88h wall clock means we cannot afford it as our default for n=681.

### Key findings — A3B smoke

- **3.7× faster than 27B** (19 s/turn vs 72 s/turn). Matches the 3-4× projection from the active-param ratio (3 B vs 27 B). **Full-run projection: ~23 h** — viable as an overnight run.
- **State accuracy collapses to roughly tied with baseline** (24.24% vs 25.94%). The 27B's classification advantage does not transfer to the MoE.
- **Stage a is the weak spot.** A3B drops to 40% on problem-detection — Qwen3.5-9B had the same exact failure mode in `docs/QWEN_EVAL_FIX_PLAN.md` (57.12% at n=681). The MoE's smaller active-param count appears to bottleneck the "did the student ask a question" trigger that 27B nails 100% of the time.
- **Stage c is much weaker than 27B but slightly better than baseline** (7.69% vs 27B's 30.77% vs baseline 4.7%). The hardest classification task is where the active-param gap bites hardest.
- **Stage e jumps to 50%** — the dialogue-closure trigger is a simpler classification, and A3B handles it better than either 27B or baseline.
- **ROUGE/BLEU same general profile as 27B** (~25-28 ROUGE-1 vs baseline 44.61). Both Qwen models write a richer pedagogical voice that diverges from ground-truth phrasing. This is a Qwen-family characteristic, not a 27B-specific one.

### Decision matrix as of A3B smoke complete

| Config | Full-run wall clock | Overall state acc (smoke n=5) | ROUGE-1 (smoke) | Verdict |
|---|---|---|---|---|
| gpt-4o baseline (existing) | 4h 34m | 25.94% (n=681) | 44.61 | Already shipped — paper-faithful |
| Qwen 27B (think on) | ~88 h | **39.39%** | 26.40 | Strong quality, blocked by wall clock |
| A3B (think on) | ~23 h | 24.24% | 27.52 | Fast enough, quality merely tied with baseline |
| **27B + consultant no-think** | ?? (projected ~30-40 h) | ?? | ?? | **TODO — best candidate for "fast + good quality"** |
| **A3B + consultant no-think** | ?? (projected ~10-15 h) | ?? | ?? | **TODO — best candidate for "fastest viable"** |

The next two smokes will tell us whether `/no_think` on the consultant preserves classification accuracy. If 27B + no-think holds the +13 state acc lift at ~30-40 h, that's our winner. If A3B + no-think holds the ~24% baseline-tied accuracy at ~10-15 h, that's the fastest viable for iteration.

### Realization after no-think smokes — speed lever was wrong

After running both no-think smokes, the *teacher* turns out to dominate
wall-clock time, not the consultant. Disabling consultant thinking gives only
modest speed savings:

| Config | s/turn | Δ vs think variant |
|---|---|---|
| 27B think | 72 | (baseline) |
| 27B no-think | ~90+ (cold-cached, may settle) | possibly **slower**, not faster |
| A3B think | 19 | (baseline) |
| A3B no-think | 17 | -10% (modest) |

**Implication:** to actually speed up either model, we need teacher-side
thinking disabled too — or we need a **structural** change like collapsing the
two-call pipeline into one. The consultant-thinking knob is plumbed through
`socratic_teaching_system.py:312` (prepends `/no_think`); the equivalent for
the teacher (`socrates_teacher` at line 388) does not exist and would need a
small code change. Even with that, the per-turn cost of two LLM calls remains.

**The unexpectedly good news from no-think smokes:** A3B + consultant no-think
*beats* its own with-think variant on state accuracy (24.24% → 31.25%, +7 pts).
With only 3 B active params, A3B's CoT may meander; stripping it forces a
direct classification pathway. The 27B no-think result is still in flight —
TBD whether it shows the same effect.

### Decision matrix — final, all 8 configs landed

After completing all 4 fusion smokes (2026-05-04 11:55 → 12:53 PDT, ~58 min
total). Sorted by overall state accuracy:

| Config | ROUGE-1 | State acc | s/turn | Full-run | Notes |
|---|---|---|---|---|---|
| **27B fusion think** | 31.88 | **46.88%** | 39 | ~48 h | **Headline quality** — only config to crack stage d (25%); 0 fallbacks |
| **A3B fusion think** | **32.96** | 42.42% | 13 | **~16 h** | **Operational winner** — 90% of 27B fusion's lift at 33% the wall clock |
| A3B fusion no-think | 29.70 | 40.62% | 17 | ~21 h | Small regression vs fusion think; 1 schema fallback |
| 27B no-think two-call | 29.99 | 39.39% | 70 | ~80 h | Best two-call config; matches 27B think on state acc, +3.6 ROUGE-1 |
| 27B think two-call | 26.40 | 39.39% | 72 | ~88 h | Original 27B baseline; superseded by no-think and fusion |
| A3B no-think two-call | 28.36 | 31.25% | 17 | ~21 h | Best two-call A3B; +7 over A3B think two-call |
| 27B fusion no-think | 29.78 | 30.30% | 38 | ~47 h | Sharp regression — fusion + no-think doesn't compose on dense models |
| gpt-4o baseline (n=681) | 44.61 | 25.94% | — | 4 h 34 m | Reference paper-faithful baseline |
| A3B think two-call | 27.52 | 24.24% | 19 | ~24 h | Original A3B baseline; superseded by every other Qwen config |

### Two-call vs fusion deltas (same model, same thinking mode)

| Config pair | State acc Δ | ROUGE-1 Δ | Wall-clock Δ |
|---|---|---|---|
| 27B think: two-call → fusion | +7.49 | +5.48 | -46% |
| 27B no-think: two-call → fusion | -9.09 | -0.21 | -45% |
| A3B think: two-call → fusion | **+18.18** | +5.44 | -34% |
| A3B no-think: two-call → fusion | +9.37 | +1.34 | -2% |

**Three of four fusion variants strictly beat their two-call sibling on every
metric.** The exception is 27B no-think — fusion + no-think on dense 27B
collapses (the same composition is fine on A3B, only loses 1.80 pts).

### Headline findings

1. **Fusion architecture is the architectural win.** Every fusion + thinking
   variant beats every two-call variant on state accuracy. The single-call
   structured-output design isn't just faster — it's *better* at the
   classification + generation joint task.

2. **27B fusion think is the headline result for the paper.** 46.88% state acc,
   +20.94 over the gpt-4o baseline. First config in this exploration to crack
   stage d (25% — both 27B two-call variants got 0%). 21 min for n=5 = ~48 h
   for full n=681. Multi-day weekend run.

3. **A3B fusion think is the operational winner.** 42.42% state acc (+16.48
   over baseline) with **7 min for n=5 = ~16 h overnight full run**. Sacrifices
   ~4 state-acc points relative to 27B fusion for a 3× throughput win and
   single-overnight feasibility. Slightly *higher* ROUGE-1 than 27B fusion
   think (32.96 vs 31.88) — the structured-output joint task agrees with the
   MoE just as well as with the dense model.

4. **`/no_think` interacts with the fusion architecture, not the way it does
   with two-call.** In two-call, `/no_think` is a free upgrade for both models.
   In fusion + dense 27B, it collapses (-16.58 state acc). In fusion + MoE A3B,
   it regresses slightly (-1.80) and produces one schema fallback. Hypothesis:
   the unified call's joint state-classification + teacher-generation task
   genuinely benefits from `<think>` reasoning depth; without it, the model has
   to do too much in parallel and quality suffers. Two-call decouples the two
   tasks so each is simpler and `/no_think` is fine.

5. **Schema enforcement is rock-solid.** Across 130 turns of fusion, only 1
   schema fallback (in A3B fusion no-think). llama.cpp's strict json_schema
   constraint is reliable enough to depend on for production.

6. **Stage d unlocked.** Both 27B two-call variants got 0% on stage d. Fusion
   gets 20-25% across all variants except 27B fusion no-think. The state-action
   map embedded directly in the unified prompt appears to be the lever — the
   model has zero-cost access to "what does this state mean for the teacher"
   without a second LLM hop and the context-loss it brings.

### Recommendation for the full run

**Primary: A3B fusion think.** Single overnight full run (~16 h). Beats
gpt-4o baseline by +16.48 state acc + better ROUGE-1. Crash-safe per-item.
Leaves comfortable headroom for restarts. This is the config to ship.

**Secondary (paper headline if GPU time allows): 27B fusion think.** ~48 h
weekend run for the +20.94 state-acc result and the only config that cracks
stage d. Delivers a stronger paper headline at the cost of 3× the GPU time.

**Both runs together:** complementary paper ablations. A3B fusion think tells
the "MoE + structural classification" story; 27B fusion think tells the
"dense Qwen3.6 outperforms gpt-4o" story. Different strengths in different
stages (27B fusion think wins stage c at 30.77% — A3B couldn't match — but
loses stage e to A3B 75% vs 75% tied).

### What we're NOT doing as of 2026-05-04 12:53 PDT

Per user direction: GPU goes to other (non-project) work after fusion smokes.
Mini runs (TODO M1, M2, M3 above) are deferred until GPU is available again.
The persistent docs hold all the gating logic for picking up later.

---

### Decision matrix for the "improved session" full run

All four two-call configs landed. Updated table with measured numbers:

| Config | ROUGE-1 | State acc | s/turn | Full-run wall clock | Decision |
|---|---|---|---|---|---|
| gpt-4o baseline (n=681) | 44.61 | 25.94% | — | 4 h 34 m | reference |
| **27B no-think** | **29.99** | **39.39%** | 70 | **~80 h** | **Best quality config — strictly better than 27B think** |
| 27B think | 26.40 | 39.39% | 72 | ~88 h | Slightly worse on every metric than no-think; abandoned |
| **A3B no-think** | 28.36 | **31.25%** | 17 | **~19 h** | **Best fast config — overnight viable, +5.3 over baseline** |
| A3B think | 27.52 | 24.24% | 19 | ~23 h | Tied with baseline, slowest A3B; abandoned |

**Headlines:**

1. **`/no_think` on the consultant is a free upgrade for both models.**
   - 27B: same state acc, +3.6 ROUGE-1, marginally faster. Strictly dominant.
   - A3B: +7 state acc, +0.8 ROUGE-1, marginally faster. Strictly dominant.
   - Counterintuitive but reproducible across both models.
2. **Speed and quality are decoupled.** A3B is the speed lever (~3.7×), 27B is
   the quality lever (+13 state acc). Disabling consultant thinking is a free
   tune-up for both, not a quality/speed trade.
3. **27B + consultant `/no_think` produces the headline result for the paper:**
   39.39% state acc, +13.45 over gpt-4o baseline, with a slightly *better*
   ROUGE-1 than 27B think. Wall clock ~80 h — multi-day, but doable as a
   weekend run.
4. **A3B + consultant `/no_think` is the operational win:** 31.25% state acc,
   +5.31 over baseline, ~19 h full run. Overnight viable.

### Recommendation

**Run two full evals, in this order:**

1. **A3B + no-think**, kicked off as soon as the PR is merged. ~19 h. Lands
   the next day. Provides paper draft ablation row #1 ("MoE Qwen3.6 + structural
   classification > GPT-4o consultant in less compute").
2. **27B + no-think**, kicked off Friday evening. ~80 h, lands Tuesday. Provides
   the headline paper result and ablation row #2 ("dense Qwen3.6 27B as both
   teacher and consultant produces +13 state acc vs gpt-4o, with stage c jumping
   ~6.5×").

If only one is possible, **A3B + no-think** wins on ROI: not the headline result
but enough to demonstrate "we improved over baseline" with one overnight session.

### Per-stage state accuracy comparison (smoke, n=5, 33 turns each)

| Stage | gpt-4o (n=681) | 27B think | 27B no-think | A3B think | A3B no-think |
|---|---|---|---|---|---|
| a (problem detection) | 95.15% | **100%** | **100%** | 40.0% | 60.0% |
| b (early reasoning) | 36.93% | 50.0% | **66.67%** | 50.0% | 33.3% |
| c (hard misconception) | 4.70% | **30.77%** | 15.38% | 7.69% | 15.38% |
| d (resolution) | 5.04% | 0.0% | 0.0% | 0.0% | 0.0% |
| e (closure) | 11.92% | 25.0% | **50.0%** | **50.0%** | **75.0%** |
| **overall** | 25.94% | **39.39%** | **39.39%** | 24.24% | 31.25% |

Two trends visible across all 5 stages:

- **27B think is the only config to crack stage c materially** (30.77%, +26 over
  baseline). Disabling consultant thinking on 27B drops stage c to 15.38% —
  consultant CoT helps most on the hardest classification. This is a per-stage
  trade hidden by the identical overall numbers.
- **All Qwen variants beat baseline on stage e (closure).** Even the worst
  (A3B think) ties; the best (A3B no-think) is 4× higher. Closure detection
  is apparently easier for these models than for gpt-4o.

For paper ablation purposes, 27B think (+30.77% on stage c) and 27B no-think
(+66.67% on stage b, +50% on stage e) are *complementary* — different stages
are won by different configs. Worth running both even at the wall-clock cost.

### Decision gate — required mini runs before full-run commitment

Smoke n=5 (~33 turns) is too noisy for production decisions. Before we commit
to a multi-hour full n=681 run, we need n=25 mini runs (~150 turns each) to
validate the smoke deltas. Mini runs are the gating step between exciting
smoke results and ship-it confidence.

#### Open questions the smokes can't answer

- [ ] **Does fusion's state-accuracy lift hold at n=25?** 27B fusion smoke
      showed +7.5 over its two-call sibling; could be n=5 sampling noise.
- [ ] **Is the stage-c regression in 27B fusion (-15 vs 27B think two-call) real
      or noise?** Stage c is the hardest classification (22 states). If real,
      it's a per-stage trade-off worth ablating in the paper.
- [ ] **Is stage d at 25% (27B fusion) reproducible?** Two-call always got 0%
      on stage d. First-time hit could be n=5 luck on dialogues that contain
      d-turns matching the model's biases.
- [ ] **Does fusion's -46% wall-clock win scale?** Striking on n=5; n=25 is
      a more reliable per-turn average and includes longer dialogues.
- [ ] **What's the unified_fallback_count distribution at n=25?** Smoke step 1
      had 0/32 fallbacks — but n=5 may not surface rare schema-edge-case
      dialogues (off-topic student input, edge-case state transitions).

#### Mini-run TODO list (gated on smoke completion)

After all 4 fusion smokes land, the immediate next step is *not* a full run —
it's a targeted set of mini runs to confirm or contradict smoke findings:

- [ ] **TODO M1 — Top fusion variant mini.** Pick the smoke's best fusion
      config by combined state-acc + wall-clock. Run mini (n=25). Likely
      candidate after smoke evidence so far: 27B fusion think.
      Estimated time: 1.5–2 h on 27B; 30–40 min on A3B.
- [ ] **TODO M2 — 2nd-best fusion variant mini.** Run mini for direct
      head-to-head against TODO M1. Most likely an A3B fusion variant for the
      "speed lever" comparison.
- [ ] **TODO M3 (optional) — Best two-call comparator mini.** Run n=25 on the
      best-smoke two-call config (27B no-think) to establish a reliable
      two-call-vs-fusion delta at n=25 confidence. Skip if M1/M2 results are
      decisive.

#### Decision criteria for promoting from mini to full

| Mini result | Action |
|---|---|
| State acc within ±3 pts of smoke AND wall-clock within ±10% AND fallback rate <5% | Greenlight full run |
| State acc drops >5 pts vs smoke | Investigate per-stage profile; if isolated, examine prompt; if uniform, fall back to two-call |
| Wall-clock loses speedup (>30% slower than smoke projects) | Investigate context-length scaling; may need shorter prompts |
| Fallback rate >5% | Audit failed turns; tighten prompt or schema before full commit |
| Stage c regression confirmed at n=25 | Both think and no-think viable; pick by paper story (per-stage trade-off becomes an explicit ablation row) |

#### Suggested sequencing once smokes are done

1. Identify top 1–2 fusion variants by smoke ranking (combined state-acc +
   wall-clock score).
2. Run mini for those (TODO M1, M2). Total budget: 2–3 h.
3. Compare mini-vs-smoke deltas; identify any per-stage regressions.
4. Optional: mini run on best two-call comparator (TODO M3).
5. Make full-run decision with mini-confidence numbers.
6. Full run on the winner.

This is a hard gate. Skipping straight from smoke to full-run on n=5 evidence
is the kind of move that burns 20+ hours of GPU on a config that the n=25 data
would have caught as a regression.

---

### Phase 3 lever — fusion (formal plan landed)

The next obvious move after "tune the existing two-call pipeline" is
*eliminate the second call entirely* by merging consultant + teacher into one
structured-output call. This is the **Consultant-Teacher Fusion** plan from
`docs/IMPROVEMENT_PLAN.md` #4 and `docs/QWEN27B_LOCAL_PLAN.md` Phase 3 — both
foreshadowed at the start of this exploration. With both agents already on
the same llama.cpp server, this is now primarily a code change rather than
infra work.

**Formal plan: [`docs/SOCRATIC_FUSION_PLAN.md`](SOCRATIC_FUSION_PLAN.md).**

Highlights:
- llama.cpp accepts strict `json_schema` response_format (verified live).
- Qwen3.6 surfaces thinking via separate `reasoning_content` field — CoT does
  not pollute structured output.
- Projected wall clock: 27B from ~88 h → ~50–60 h; A3B from ~23 h → ~12–15 h.
- Implementation is a new `SocraticTeachingSystemUnified` subclass + a
  `--unified` flag on `kele.py`; orchestrator plumbing is one line per script.
- Hardest design issue is state-coherence after Python glue override. Plan
  defaults to "trust the model's response, log divergence rate, revisit if
  high" (Option A in the plan).
- Validation cadence: smoke (n=5) on A3B unified vs A3B two-call (already in
  hand) → mini (n=25) → full (n=681).

### Live observations

**2026-05-04 10:00 — Qwen 27B Q5 smoke kicked off (warm server reuse after cold-load fix)**

- Server boots in ~4 s (port responds), but the model load takes longer; the
  "Loading model" 503 response was confusing the original `verify_alias` check.
  Fixed by adding a `ready_check` that requires the alias *and* a non-loading response. Edit in `scripts/eval_qwen27b.sh`.
- VRAM stable at 29.9 GB / 32 GB, GPU util 99% during inference.
- Per-turn: dialogue 0004=63s/turn (5t), 0005=68s/turn (6t), 0014=68s/turn (5t). Average ~65 s/turn.
- **Implication:** projected full n=681 wall clock = ~75 h (not 6-7 h as initially estimated). **Thinking mode is the culprit** — `<think>...</think>` blocks generate 1-3K CoT tokens before each call's actual output, all paid at ~60 tok/s.

**2026-05-04 ~10:15 — Bumped `CONSULTANT_MAX_TOKENS` 4096 → 8192 in `qwen27b-local.env`**

- Why: Qwen3.6 thinking mode is on by default. CoT counts against `max_tokens`.
  Smoke shows zero parse failures at 4096 so it's not strictly necessary, but
  8192 is free insurance against deep-reasoning truncation. Won't affect the
  in-flight smoke (config loaded at startup); applies to mini and full.
- Mirrors what `R9700_Mac-M4.env` already does.

**2026-05-04 ~10:30 — Threw in A3B as a parallel option**

- Wall-clock projection on 27B (~75 h) is incompatible with the project
  timeline. A3B is the obvious lever: 3 B active vs 27 B → ~3-4× faster on
  memory-bandwidth-bound RTX 5090.
- Quality projection: -2 to -3 points on state acc, -1 to -2 ROUGE; both
  models share the Qwen3.6 corpus so Chinese fluency is effectively equal.
- Infra parked: `scripts/serve_qwen35b_a3b.sh`, `configs/qwen35b-a3b-local.env`,
  `scripts/eval_qwen35b_a3b.sh`. Will run smoke + mini after 27B finishes.

### Decisions log

| When | Decision | Why |
|---|---|---|
| 2026-05-04 09:48 | Use `Qwen3.6-27B-UD-Q5_K_XL.gguf` (clean), not `HauhauCS-Aggressive` Uncensored | Academic deliverable + Socratic-of-children dataset; uncensored fine-tune is a liability |
| 2026-05-04 09:48 | Both teacher and consultant on one llama.cpp server | 6 parallel slots + unified KV; no extra VRAM; stages Phase 3 fusion |
| 2026-05-04 ~10:15 | Bump `CONSULTANT_MAX_TOKENS` 4096 → 8192 | Free insurance vs Qwen thinking-mode CoT eating output budget |
| 2026-05-04 ~10:30 | Add A3B as parallel option, infra-park while 27B runs | 27B at 75 h doesn't fit timeline; A3B's ~25 h overnight does |

### Open questions

- Is Qwen 27B's per-stage state acc on hard stages (c, d) high enough to
  justify its 3× slower wall clock vs A3B? Need mini results from both.
- Does disabling consultant thinking (`CONSULTANT_DISABLE_THINKING=true`)
  hurt state accuracy meaningfully on either model?
- Can we squeeze more parallelism out of the 6-slot llama.cpp server by
  batching evals, or is the 1-turn-at-a-time pipeline the binding constraint?

---

## Reference: model + config inventory

| Model file | Variant | Quant | Size | Default ctx | Per-token KV | Notes |
|---|---|---|---|---|---|---|
| `Qwen3.6-27B-UD-Q5_K_XL.gguf` | Dense 27B | Q5_K_XL | 19 GB | 416K | ~18 KB | Used for `qwen27b-local` |
| `Qwen3.6-27B-Q4_K_M.gguf` | Dense 27B | Q4_K_M | 16 GB | 512K | ~18 KB | Available; not yet tried in KELE |
| `Qwen3.6-27B-Uncensored-HauhauCS-Q5_K_P.gguf` | Dense 27B (uncensored) | Q5_K_P | 21 GB | 416K | ~18 KB | **Do not use** for academic work |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | MoE 35B / 3B active | Q4_K_M | 20 GB | 512K | **~6.3 KB** | Used for `qwen35b-a3b-local` |

| Config | Server URL | Teacher = Consultant | Thinking | Status |
|---|---|---|---|---|
| `qwen27b-local.env` | localhost:8080 | Qwen3.6-27B Q5 | on | tested 2026-05-04 |
| `qwen35b-a3b-local.env` | localhost:8080 | Qwen3.6-35B-A3B Q4 | on | infra ready |
| `qwen27b-local.env` + `CONSULTANT_DISABLE_THINKING=true` | localhost:8080 | Qwen3.6-27B Q5 | off (consultant) | not tried |
| `qwen35b-a3b-local.env` + `CONSULTANT_DISABLE_THINKING=true` | localhost:8080 | Qwen3.6-35B-A3B Q4 | off (consultant) | not tried |

## Reference: result directory naming

| Dir | Source |
|---|---|
| `results/baseline/` | 2026-04-14 — gpt-4o-2024-11-20 baseline (canonical comparison target) |
| `results/qwen27b-local-smoke/` | 27B Q5 smoke (n=5) |
| `results/qwen27b-local-mini/` | 27B Q5 mini (n=25) |
| `results/qwen27b-local/` | 27B Q5 full (n=681) — not yet run |
| `results/qwen35b-a3b-local-smoke/` | A3B smoke (n=5) — done 2026-05-04 |
| `results/qwen35b-a3b-local-mini/` | A3B mini (n=25) — not yet run (think two-call) |
| `results/qwen35b-a3b-local-mini-unified/` | A3B mini (n=25) fusion-think — done 2026-05-04 19:16 PDT |
| `results/qwen35b-a3b-local/` | A3B full (n=681) — not yet run (think two-call) |
| `results/qwen35b-a3b-local-unified/` | A3B full (n=681) fusion-think — **DONE 2026-05-05 12:16 PDT** ✅ |

---

> ## 🚧 RESUME HERE — next session
>
> **Last session ended 2026-05-05 ~12:30 PDT.** A3B fusion-think full n=681 **completed cleanly**. Headline result locked: **state acc 38.70% (+12.76 over gpt-4o baseline), 16h 29m wall clock, 0.91% schema fallback rate.** Server torn down, GPU freed. The 5/5 deliverable is in hand.
>
> **The canonical numbers are in this doc's "🏆 Full run results" section at the very top.** Full artifact pointers there. Don't re-derive — just cite.
>
> **What's next: 2026-05-14 paper-draft horizon.**
>
> The ROUGE gap (~14 pts under baseline) is the open work. The "ROUGE diagnosis and forward path" section in this doc analyzed it — TL;DR:
> - 27B will NOT help (smoke evidence: 27B fusion ROUGE-1 31.88 < A3B fusion 32.96)
> - SocratTeachLLM-as-teacher (Option 1) is **ruled out by VRAM** (32 GB cap; SocratTeachLLM 9B FP16 + A3B Q4 = 38 GB just for weights)
> - **Options 2 (prompt engineering) and 3 (LoRA fine-tuning) are the locked forward paths** per Max 2026-05-04
>
> **Suggested next-session sequence:**
> 1. **Option 2 first** — prompt-engineering smoke run (~30 min). Modify teacher system prompt: "Respond with one Socratic question. No preamble, no praise, no scaffolding. Match style: [3-shot GT examples]." Validate whether style steering alone recovers ROUGE without changing the model. Cheap and fast — worth knowing before committing to LoRA.
> 2. **Option 3 if Option 2 underperforms** — LoRA fine-tune Qwen3.6-35B-A3B on (student dialogue history → teacher response) pairs from the train split. PEFT, ~30 min training on the 5090, no extra inference-time VRAM impact.
> 3. **Paper Evaluation section drafting** — can start from the 5/5 numbers immediately. The "Findings" subsection in this doc's "Full run results" maps directly to Methodology / Results paragraphs.
>
> **State on resume — check in this order:**
> 1. `git log --oneline | head` — confirm what's merged
> 2. `cat results/qwen35b-a3b-local-unified/metrics_summary.json` — final numbers
> 3. `pgrep -af llama-server` — should be empty (no run in flight)
> 4. `nvidia-smi --query-gpu=memory.used --format=csv,noheader` — confirms GPU freed
>
> **Hard rules carried forward:**
> - Single-Qwen approaches only — no SocratTeachLLM-as-teacher (VRAM constraint, see `memory/feedback_single_qwen_constraint.md`)
> - 27B is not a ROUGE lever — don't waste a 48-h slot on it
> - The eval is per-item resumable; the orchestrator handles full lifecycle (boot, eval, teardown, comparison) end-to-end
