# Qwen Local Exploration Log

**CSEN 346 · Santa Clara University · started 2026-05-04**

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
| 3 | qwen27b + no-think (consultant) smoke | 5 | — | — | — | — | — | — | — | running |
| 4 | qwen35b-a3b + no-think (consultant) smoke | 5 | 28.36 | 10.48 | 20.12 | 4.33 | **31.25%** | 17 | 9.3 min | done — state acc +7 pts vs think-on |

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
| `results/qwen35b-a3b-local-smoke/` | A3B smoke (n=5) — not yet run |
| `results/qwen35b-a3b-local-mini/` | A3B mini (n=25) — not yet run |
| `results/qwen35b-a3b-local/` | A3B full (n=681) — not yet run |
