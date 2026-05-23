# Handoff — feat/multi-dataset-training (PR #65)

**Branch:** `feat/multi-dataset-training` → `main`
**PR:** https://github.com/ulises-c/csen-346/pull/65
**Date:** 2026-05-22
**Project:** CSEN 346 · SocratTeachLLM v2 (Santa Clara University)

---

## What this branch is for

Build and ship the SFT fine-tuning pipeline for SocratTeachLLM v2.

The original paper's teacher model (GLM4-9B) was trained only on SocratDataset (Chinese, ~6,800
dialogues). This branch adds:
- A unified multi-dataset loader (`src/project/dataset.py`) covering all four HF sources
- A ready-to-run SFT training script (`scripts/train_sft.py`) using TRL SFTTrainer + LoRA/QLoRA
- A LoRA config for Qwen2.5-7B (`configs/train-sft-qwen25-7b-lora.env`)
- 10 data-shape smoke tests (`tests/test_dataset.py`) — all passing, no GPU required

**The branch is code-complete. The single remaining step is to run the actual training.**

---

## Machine roles

| Machine | Role |
|---|---|
| MacBook (this machine) | Code editing, PR management, test runs (CPU-only) |
| R9700 AI PRO (32 GB VRAM, AMD ROCm gfx1201) | All GPU training + evaluation |
| RTX 5090 (partner's PC, 32 GB VRAM, NVIDIA CUDA) | Optional: same training works there too |

**All `train_sft.py` runs must happen on the R9700 (or RTX 5090). Never on the Mac.**

---

## What is done in this branch

### src/project/dataset.py (NEW)
Unified HF dataset loader. Four sources → one normalized `messages` format.

| Source key | HF repo | Records | SocRule annotations |
|---|---|---|---|
| `socrat-zh` | `ulises-c/SocratDataset` | 6,803 | ✅ state + action prefix |
| `socrat-en` | `ulises-c/SocratDataset-EN` | 6,803 | ✅ state + action prefix |
| `socrateach-multi` | `ulises-c/SocraTeach_Multi` | ~30K dialogues | ❌ |
| `socrateach-single` | `ulises-c/SocraTeach_Single` | 20,845 | ❌ |

Key entry points:
- `load_training_data(sources, split, seed)` — concurrent load + combine
- `load_split_pair(sources, seed)` — single HF download pass, returns `(train, eval)` tuples

**Critical invariant:** The 90/10 shuffle-split uses `seed=42` with `random.Random` (not `random`
module global). The same seed and logic is used in `src/project/kele.py:load_dataset()`. Do not
change the split logic without updating both files — train/eval sets must never overlap.

Each annotated record's assistant turn is prefixed: `[State: a1] [Action: give_example]\n{teacher_text}`
At inference time the live consultant agent supplies state/action; at training time ground-truth is used.

### scripts/train_sft.py (NEW)
- Config via environment variables (load with `--config configs/train-sft-qwen25-7b-lora.env`)
- `--dry-run` flag: validates config + downloads data without loading model weights
- Delegates all device handling to accelerate (works on both ROCm and CUDA without changes)
- Loss masked to assistant turns only via `SFTConfig(assistant_only_loss=True)`
- Saves final adapter to `{TRAIN_OUTPUT_DIR}/final`

### configs/train-sft-qwen25-7b-lora.env (NEW)
Ready-to-use config for Qwen2.5-7B LoRA on R9700:
- `TRAIN_METHOD=lora`, `TRAIN_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct`
- `LORA_RANK=32`, `LORA_ALPHA=64`, all 7 linear projections
- Effective batch = 4 × 4 = 16, `lr=5e-5`, 3 epochs
- `TRAIN_SOURCES=socrat-zh,socrat-en` (Phase 2 structural fine-tuning default)
- Output: `outputs/sft-qwen25-7b-lora`

### tests/test_dataset.py (NEW)
10 smoke tests, 0 HF network calls, 0 GPU. All pass.

Coverage on `dataset.py`: **87%**. Uncovered lines are mostly the `load_split_pair()` function
body (lines 389–408) and a few edge paths (`_strip_quotes` for unquoted strings at line 46,
`split=="all"` early-return in three loaders). Easy to cover with two more tests if desired.

### src/project/config.py (MODIFIED)
`load_env_file()` now strips inline shell comments before unquoting:
```
KEY=value # this comment is now correctly ignored
```

---

## How to run training on the R9700

**Step 1 — Dry run (Mac or R9700, verifies config + downloads data):**
```bash
uv run python scripts/train_sft.py --config configs/train-sft-qwen25-7b-lora.env --dry-run
```

**Step 2 — Full Phase 2 training run (R9700 only):**
```bash
uv run python scripts/train_sft.py --config configs/train-sft-qwen25-7b-lora.env
```
Expected: ~3–4 h on R9700. Peak VRAM ~24 GB (fits in 32 GB without gradient checkpointing).
Output adapter saved to `outputs/sft-qwen25-7b-lora/final/`.

**Step 3 — Evaluate the fine-tuned model:**
The adapter is a standard PEFT LoRA adapter compatible with `serve_teacher_local.sh`.
Set `TEACHER_LOCAL_PATH=outputs/sft-qwen25-7b-lora/final` and run the existing eval pipeline:
```bash
source configs/R9700_Mac-M4.env  # or your local config
uv run python -m src.project.kele --split test --output results/sft-qwen25-7b-lora/
uv run python -m src.project.evaluate results/sft-qwen25-7b-lora/
```
Compare against the locked baseline (`bert-consultant-fewshot10-gemma-n50`: 51.1% state acc / 38.53 R-1).

**Optional — Phase 1 breadth pre-training (before Phase 2):**
Edit `TRAIN_SOURCES=socrateach-multi,socrateach-single` in the env file (or export it),
then run. Save that checkpoint, then re-run Phase 2 starting from it by setting
`TRAIN_BASE_MODEL=outputs/sft-phase1/final`. See `docs/TRAINING_PLAN.md §3.2` for rationale.

---

## Current best results (for comparison)

Top systems on the test set (`n=681` unless noted):

| Run | State acc | ROUGE-1 | Notes |
|---|---|---|---|
| `bert-consultant-fewshot10-mini` | 54.1% | 33.12 | mini scale only |
| `bert-consultant-fewshot10-a4b-mini` | 53.2% | 36.26 | mini scale only |
| `tournament-cell-1-length_budget` | 52.0% | 39.91 | prompt-eng win |
| `bert-consultant-fewshot10-gemma-n50` | **51.1%** | **38.53** | **locked headline** |
| `baseline` | 25.9% | 44.61 | original SocratTeachLLM |

The fine-tuned Qwen2.5-7B targets **>51.1% state acc** while recovering ROUGE-1 toward the baseline's 44.61.

---

## Open next steps (from docs/EXPERIMENT_TIERS.md)

### S.2 — LoRA fine-tune (this branch, next step)
Run the training as described above. This is the primary goal of this branch.

### S.3 — Hierarchical BERT state classifier
Separate task. 5-way stage head + within-stage head trained on 42K labeled turns.
Would target stage-c specifically (currently ~4–14% acc). See `EXPERIMENT_TIERS.md S.3`.

### Tier A extras (after S.2 eval)
- If 7B LoRA underperforms → upgrade to QLoRA on Qwen2.5-14B: change `TRAIN_METHOD=qlora`
  and `TRAIN_BASE_MODEL=Qwen/Qwen2.5-14B-Instruct` in the env file.
- Many-shot sweep (A.4), stage-aware exemplars (A.5) still available as prompt-eng fallbacks.

---

## Key files quick reference

```
configs/train-sft-qwen25-7b-lora.env  ← training hyperparameters
scripts/train_sft.py                  ← training entry point
src/project/dataset.py               ← multi-dataset HF loader
src/project/kele.py                  ← eval pipeline (load_dataset shares split logic)
src/project/config.py                ← load_env_file (loads .env / config files)
tests/test_dataset.py                ← dataset loader smoke tests
docs/TRAINING_PLAN.md                ← full LoRA/QLoRA analysis + curriculum rationale
docs/EXPERIMENT_TIERS.md            ← prioritized open experiment list
scripts/test_gpu_stack.sh            ← ROCm smoke test (TODO: add CUDA/RTX 5090 path)
```

---

## Before merging PR #65 to main

- [ ] Dry-run passes on R9700 (verifies ROCm + HF download path)
- [ ] Full training run completes without OOM
- [ ] Eval results recorded in `results/sft-qwen25-7b-lora/`
- [ ] `comparison.json` updated with the new run
- [ ] `docs/EXPERIMENT_LOG.md` entry added

If training fails (OOM or ROCm issue), fall back to:
1. `TRAIN_GRAD_CKPT=true` (adds ~2 GB headroom, ~20% slower)
2. `TRAIN_METHOD=qlora` (drops to ~12 GB peak, slight quality loss)
3. Use RTX 5090 — same script, CUDA auto-detected by accelerate
