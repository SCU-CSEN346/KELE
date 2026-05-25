# Handoff — Stage 2b Training Launch (2026-05-25)

## Status
Training is **NOT running**. It crashed. Read the full diagnosis below before
attempting to relaunch.

## What this session accomplished
- Fixed TRAIN_BASE_MODEL to `Qwen/Qwen3.6-27B` (no `-Instruct` suffix)
- Added `.gitignore` covering `outputs/`, `.venv/`, etc.
- Fixed `_install-torch-rocm` Makefile target: drops torchvision (ABI mismatch)
- Fixed gradient checkpointing: `use_reentrant=False` in `SFTConfig`
- **Fixed root cause of persistent torch revert**: pre-commit hook and Makefile
  now use `uv run --no-sync` so committing never reverts ROCm torch again

## Root cause of all training crashes

There is one root cause that manifested as several different errors:

**`uv run` (without `--no-sync`) triggers `uv sync` which reverts `torch+rocm7.2`
back to `torch+cu130` (CUDA build). The CUDA build has no GPU on this AMD machine.**

Every commit triggered the pre-commit hook → `uv run pyright` / `uv run pytest` →
`uv sync` → cu130 replaces rocm7.2. If this ran between `make _install-torch-rocm`
and the training process importing torch (~10s window), the CUDA build got imported.

**This is now fixed.** `hooks/pre-commit` and `Makefile` use `--no-sync` everywhere.

## Known working launch sequence

```bash
# 1. Install ROCm torch (always do this before launching training)
make _install-torch-rocm

# 2. Verify — must show: 2.11.0+rocm7.2  7.2.26015  True
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"

# 3. Launch training (both env vars required)
nohup env TORCH_USE_HIPBLASLT=0 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  uv run --no-sync python scripts/train_sft.py \
  --config configs/train-sft-stage2-socratic.env \
  > outputs/sft-stage2-socratic/train.log 2>&1 &
echo "Training PID: $!"

# 4. Confirm step 1 appears (~5 min for initial HIP kernel compile)
tail -f outputs/sft-stage2-socratic/train.log
```

## Required env vars and why

| Var | Value | Why |
|-----|-------|-----|
| `TORCH_USE_HIPBLASLT=0` | `0` | hipBLASLt causes `HIPBLAS_STATUS_INVALID_VALUE` in Qwen3.5 RoPE on gfx1201. Forces rocBLAS fallback. Side effect: ~90s/step. |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | Reduces GPU memory fragmentation. Only works with ROCm torch. |
| `uv run --no-sync` | required | Prevents uv sync reverting ROCm torch to cu130. |

## What NOT to do

- **DO NOT use `HSA_OVERRIDE_GFX_VERSION=11.0.0`** — causes immediate GPU hang
- **DO NOT run `uv run` without `--no-sync`** after `make _install-torch-rocm`
- **DO NOT install torchvision** — `make _install-torch-rocm` handles removal automatically

## Training config

File: `configs/train-sft-stage2-socratic.env`

- Base model: `Qwen/Qwen3.6-27B`
- Method: QLoRA 4-bit NF4, LoRA r=16 alpha=32
- Data: `socrat-zh` + `socrat-en` → 12,244 train / 1,362 eval, 3 epochs
- Steps: 2,298 total (~90s/step → ~58 hour wall time)
- Output: `outputs/sft-stage2-socratic/`

## On completion: push to HF and GH

```bash
# Push model (LoRA adapter) to HuggingFace
uv run --no-sync python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("SocratTeachLLM-v2-stage2b", exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path="outputs/sft-stage2-socratic",
    repo_id="ulises-c/SocratTeachLLM-v2-stage2b",
    repo_type="model",
    ignore_patterns=["*.log"],
)
PY

# Push branch
git push origin feat/stage2-sft-pipeline-design

# Comment on PR #79
LOSS=$(tr '\r' '\n' < outputs/sft-stage2-socratic/train.log | grep "{'loss'" | tail -1)
gh pr comment 79 --body "## Stage 2b training complete

**Model**: Qwen/Qwen3.6-27B QLoRA 4-bit NF4, LoRA r=16
**Data**: socrat-zh + socrat-en (12,244 train / 1,362 eval, 3 epochs)
**Final loss**: ${LOSS}
**HF model**: https://huggingface.co/ulises-c/SocratTeachLLM-v2-stage2b

Next: ablation B eval per TRAINING_PLAN.md §5.3 (bert x FineTuned on test + synthetic)."
```

## Branch and PR

- Branch: `feat/stage2-sft-pipeline-design`
- PR: #79 — https://github.com/ulises-c/csen-346/pull/79
