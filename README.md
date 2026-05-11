# csen-346

[![CI](https://github.com/ulises-c/csen-346/actions/workflows/ci.yml/badge.svg)](https://github.com/ulises-c/csen-346/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ulises-c/csen-346/graph/badge.svg)](https://codecov.io/gh/ulises-c/csen-346)

Natural Language Processing — CSEN 346, Santa Clara University.

This project reproduces and extends **KELE**, a multi-agent framework for structured Socratic teaching with LLMs, then pushes it further than the published baseline by collapsing the two-agent architecture into a single open-weight backbone serving both consultant and teacher roles via a fusion structured-output call. Our headline locked result is a **+12.76 point absolute lift** in overall state accuracy over the GPT-4o baseline on the full 681-dialogue test split, with **3–5× multipliers on the harder middle and closure stages**, running entirely on a single 32 GB consumer GPU at zero per-run API cost.

- **Paper we reproduce:** Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models", *Findings of EMNLP 2025* — [aclanthology.org/2025.findings-emnlp.888](https://aclanthology.org/2025.findings-emnlp.888/)
- **Original repository:** https://github.com/yuanpan1020/KELE
- **Our paper draft:** [`deliverables/overleaf/latex/acl_latex.tex`](deliverables/overleaf/latex/acl_latex.tex)
- **SocratTeachLLM (our HF mirror):** https://huggingface.co/ulises-c/SocratTeachLLM
- **SocratDataset (Chinese):** https://huggingface.co/datasets/ulises-c/SocratDataset
- **SocratDataset-EN (English):** https://huggingface.co/datasets/ulises-c/SocratDataset-EN

## Headline Results

All open-weight runs use a single RTX 5090 (32 GB VRAM) with one model serving both consultant and teacher roles via the fusion architecture (see [Architecture](#architecture)). The GPT-4o baseline uses the canonical KELE two-model stack (SocratTeachLLM teacher + GPT-4o consultant via API).

| Run | n | State acc | Δ vs GPT-4o | ROUGE-1 | BLEU-4 | Wall clock | API spend |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o + SocratTeachLLM (baseline) | 681 | 25.94% | — | **44.61** | **19.60** | 4h 34m | $17.49 |
| **Qwen 3.6 35B-A3B fusion-think (full)** | **681** | **38.70%** | **+12.76 (1.49×)** | 30.63 | 5.86 | 16h 29m | **$0** |
| Gemma 4 31B fusion-think (smoke) | 5 | 51.52% | +25.58 | 33.84 | 7.02 | 12m | $0 |
| Gemma 4 31B fusion-think (mini) | 25 | **41.89%** | **+15.95** | 30.11 | 5.47 | 55m | $0 |
| *Gemma 4 31B fusion-think (full, projected)* | *681* | *~46.71%* | *~+20.77* | *—* | *—* | *~30h* | *$0* |

**Per-stage picture for the locked A3B full run** (vs baseline):

| Stage | Baseline | A3B fusion-think | Δ | Multiplier |
|---|---:|---:|---:|---:|
| a (problem detection) | 95.15% | 91.78% | −3.37 | 0.96× |
| b (early reasoning) | 36.93% | 39.29% | +2.36 | 1.06× |
| c (22-state induction) | 4.70% | **17.57%** | **+12.87** | **3.74×** |
| d (resolution) | 5.04% | **14.78%** | **+9.74** | **2.93×** |
| e (closure) | 11.92% | **56.83%** | **+44.91** | **4.77×** |
| **Overall** | **25.94%** | **38.70%** | **+12.76** | **1.49×** |

**Where Gemma 4 31B comes in.** Gemma is our forward candidate, surpassing A3B on every smoke and mini metric we measured (state acc, every per-stage cell except stage e mini, and matching/beating on text overlap) at ~1.85× the per-turn cost. The full Gemma run is the priority experiment for the next deliverable; smoke + mini average projects ~+20.77 points over the GPT-4o baseline.

Full experimental record (all 14 runs across two-call/fusion × think/no-think × Qwen 27B/A3B/Gemma) lives in [`deliverables/overleaf/latex/acl_latex.tex`](deliverables/overleaf/latex/acl_latex.tex) Section 4 and the per-run logs in [`results/`](results/).

## Architecture

Two key design choices distinguish our extension from a vanilla KELE reproduction:

**Unified fusion call.** KELE separates consultant (state classifier) and teacher (response generator) into two LLM calls per turn. Under our 32 GB single-GPU budget, the canonical stack (SocratTeachLLM ≈ 19 GB + 14B+ consultant) does not fit. We collapse both roles into a *single* JSON-schema-constrained LLM call that emits `{evaluation, state_code, teacher_response}` on every turn, with the schema enforced server-side via `llama.cpp`'s strict `json_schema` grammar. Empirically, fusion improves state accuracy by an average of +9.9 points over two-call across the smoke matrix (within each model and thinking mode), at roughly half the per-turn wall clock. Schema-fallback rate at full scale is 0.91% (38/4171 turns).

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

**Forward candidate (Gemma 4 31B fusion):**
```bash
make serve-gemma4-31b
./scripts/eval_gemma4_31b.sh smoke --unified    # n=5
./scripts/eval_gemma4_31b.sh mini  --unified    # n=25
./scripts/eval_gemma4_31b.sh full  --unified    # n=681 (~30 h)
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
