# csen-346

[![CI](https://github.com/ulises-c/csen-346/actions/workflows/ci.yml/badge.svg)](https://github.com/ulises-c/csen-346/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ulises-c/csen-346/graph/badge.svg)](https://codecov.io/gh/ulises-c/csen-346)

Natural Language Processing — CSEN 346, Santa Clara University.

This project reproduces and extends **KELE**, a multi-agent framework for structured Socratic teaching with LLMs, then pushes it further than the published baseline along two axes: (1) collapsing the two-agent architecture into a single open-weight backbone via a fusion structured-output call, and (2) replacing the LLM consultant with a 24M-parameter BERT classifier that routes the cognitive state with surgical precision while a stage-balanced 10-shot prompt-engineered LLM teacher handles response generation.

Our **locked, full-scale headline** as of 2026-05-18 is the **BERT + Gemma 4 31B + 10-shot integration**: **+22.21 point absolute lift** in overall state accuracy over the GPT-4o baseline on the full 681-dialogue test split (n=681, **48.15% state acc / 36.78 ROUGE-1**), a Pareto win over the prior A3B locked headline on both axes (+9.45 state, +6.15 R-1). Running entirely on a single 32 GB consumer GPU at zero per-run API cost. **Standalone Gemma 4 31B fusion at n=681 underperformed** (31.39% / 27.27 R-1, driven by a 21% schema-fallback rate vs A3B's 0.91%); the BERT-consultant integration removes the schema-fallback dependency entirely by routing state through a deterministic 24M-param classifier — see §4.6 and §4.8.1 of the paper for the methodological finding.

- **Paper we reproduce:** Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models", *Findings of EMNLP 2025* — [aclanthology.org/2025.findings-emnlp.888](https://aclanthology.org/2025.findings-emnlp.888/)
- **Original repository:** https://github.com/yuanpan1020/KELE
- **Our paper draft:** [`deliverables/overleaf/latex/acl_latex.tex`](deliverables/overleaf/latex/acl_latex.tex)
- **SocratTeachLLM (our HF mirror):** https://huggingface.co/ulises-c/SocratTeachLLM
- **SocratDataset (Chinese):** https://huggingface.co/datasets/ulises-c/SocratDataset
- **SocratDataset-EN (English):** https://huggingface.co/datasets/ulises-c/SocratDataset-EN

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

**Headline locked result (Qwen 35B-A3B fusion-think):**
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
./scripts/run_eval.sh baseline
./scripts/serve_socratteachllm.sh
./scripts/serve_consultant.sh
./scripts/serve_both.sh
```

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
