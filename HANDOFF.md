# Handoff — Stage 2b Training (2026-05-25, session 3)

## Status

Training is **NOT running**. The FLA Triton kernel compilation deadlocked on gfx1201.
`max_seq_len=512` is confirmed stable (70+ steps, no crash) as a fallback.

A `num_stages=1` patch for the Triton deadlock is on the remote branch (commit `d6c6333`).
**Pull first, then apply the patch, then launch at 768 tokens.**

## Recommended next action

```bash
# 0. Pull the FLA Triton patch from remote (requires network)
git fetch && git rebase origin/feat/stage2-sft-pipeline-design

# 1. Install ROCm torch (always required after any uv sync / package install)
make _install-torch-rocm

# 2. Apply the FLA num_stages=1 patch (re-run after every uv sync)
make patch-fla-rocm

# 3. Verify
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
# Must print: 2.11.0+rocm7.2  7.2.26015  True

# 4. Launch at 768 tokens
mkdir -p outputs/sft-stage2-socratic
nohup env TORCH_USE_HIPBLASLT=0 PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
  uv run --no-sync python scripts/train_sft.py \
  --config configs/train-sft-stage2-socratic.env \
  > outputs/sft-stage2-socratic/train.log 2>&1 &
echo "Training PID: $!"

# 5. Watch — first step appears within ~3 min; confirm FLA JIT clears (vs. hangs)
tail -f outputs/sft-stage2-socratic/train.log
```

**`TRAIN_MAX_SEQ_LEN=768` is set in `configs/train-sft-stage2-socratic.env`.**
If the FLA patch is not yet applied, fall back to 512 (confirmed stable, no FLA needed).

---

## Critical discoveries: AMD ROCm / gfx1201 known issues

### Model architecture

`Qwen/Qwen3.6-27B` is `Qwen3_5ForConditionalGeneration` — a multimodal VL model with:
- Vision encoder (ViT, ~550M params) — unused in text SFT, now deleted after load
- 64 LM layers: **48 `linear_attention` (SSM/delta-rule) + 16 `full_attention`**
- Requires `causal-conv1d` + `flash-linear-attention` for the efficient linear attention path

### Issue 1 — `flash-linear-attention` Triton kernels deadlock on gfx1201

`flash-linear-attention==0.5.0` installs without error and imports fine, but during
the very first training step it JIT-compiles Triton kernels. On gfx1201 (RDNA4, Navi 48)
one kernel hangs indefinitely:
- Process stays alive at 0% CPU with VRAM full
- No error is logged; progress bar never advances past step 0
- 120 kernels compile successfully; the hang occurs on a subsequent kernel

**Status**: fix on remote branch (`d6c6333` — `make patch-fla-rocm`). Root cause is
Triton 3.6.0's `tritonamdgpu-pipeline` pass having a use-after-free when `num_stages>=2`
on gfx1201. The patch rewrites `num_stages=[2-9]→1` in the installed FLA wheel.

**Workaround (no network)**: keep `max_seq_len=512`. At 512 the torch fallback
(`torch_chunk_gated_delta_rule`) does not OOM and the run is stable (70+ steps confirmed).

### Issue 2 — `expandable_segments:True` unsupported on gfx1201

`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` prints a warning and is silently
ignored. Do not use it. Only `garbage_collection_threshold:0.8` is effective.

### Issue 3 — `max_split_size_mb` blocks large allocs

`PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128` prevents the allocator from servicing
allocations larger than 128 MB even when VRAM is available. Remove it.

### Issue 4 — hipBLASLt crashes in Qwen3.5 RoPE on gfx1201

`HIPBLAS_STATUS_INVALID_VALUE` in RoPE attention. Fixed by `TORCH_USE_HIPBLASLT=0`
(forces rocBLAS fallback). Side effect: ~75–90 s/step vs ~50 s/step with hipBLASLt.

### Issue 5 — `uv run` without `--no-sync` reverts ROCm torch to CUDA

`uv run` triggers `uv sync` which reinstalls CUDA torch from PyPI, overwriting the
manually installed `torch+rocm7.2`. Always use `--no-sync`.

### Issue 6 — `causal-conv1d` build requires `--no-build-isolation` + `wheel`

```bash
uv pip install wheel
uv pip install --no-build-isolation causal-conv1d
```

Standard `uv pip install causal-conv1d` uses an isolated build env that pulls CUDA
torch and fails. Both libraries are already installed in `.venv`.

---

## Installed libraries (already in .venv)

| Library | Version | Status on gfx1201 |
|---------|---------|-------------------|
| `causal-conv1d` | 1.6.2.post1 | ✅ works — forward + backward pass verified |
| `flash-linear-attention` | 0.5.0 | ❌ Triton JIT hangs at step 0 |

---

## What NOT to do

- **DO NOT** use `HSA_OVERRIDE_GFX_VERSION=11.0.0` — GPU hang
- **DO NOT** run `uv run` without `--no-sync` after `make _install-torch-rocm`
- **DO NOT** install `torchvision` — ABI mismatch with ROCm torch
- **DO NOT** add `expandable_segments:True` — unsupported on gfx1201
- **DO NOT** add `max_split_size_mb` — blocks large allocs
- **DO NOT** set `max_seq_len` above 512 until FLA Triton hang is resolved
- **DO NOT** re-install `flash-linear-attention` and try again without first verifying
  the specific hanging kernel — it will deadlock at step 0 again

---

## Training config

File: `configs/train-sft-stage2-socratic.env`

| Setting | Value |
|---------|-------|
| Base model | `Qwen/Qwen3.6-27B` |
| Method | QLoRA 4-bit NF4, LoRA r=16 α=32 |
| Data | `socrat-zh` + `socrat-en` |
| Train/eval | 12,244 / 1,362 records, 3 epochs |
| Steps | 2,298 total |
| `TRAIN_MAX_SEQ_LEN` | **768** (requires FLA patch); fallback: 512 |
| Speed | ~75 s/step (rocBLAS); faster with FLA fast-path enabled |
| Output | `outputs/sft-stage2-socratic/` |

Token length distribution (train set):
- p50=599, p75=671, p90=753, p95=816, p99=953, p100=1279

At 512: ~50% of dialogues are truncated. The first 512 tokens cover the system prompt
+ early turns. Later turns are clipped. Quality impact is real but training completes.

---

## Changes made to scripts/train_sft.py this session

1. `attn_implementation="sdpa"` on `AutoModelForCausalLM.from_pretrained()` — for
   the 16 full-attention layers
2. Vision encoder deleted after load: `del model.visual; torch.cuda.empty_cache()`
   — saves ~1 GB VRAM
3. `prepare_model_for_kbit_training(use_gradient_checkpointing=False)` — prevents
   duplicate GC enable (SFTConfig owns gradient checkpointing)
4. `optim="paged_adamw_8bit"` removed — optimizer states only ~1 GB, not bottleneck
5. `group_by_length=True` removed — not in TRL 1.4.0 SFTConfig

---

## On completion: push to HF and GH

```bash
# Push LoRA adapter to HuggingFace
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

Next: ablation B eval per TRAINING_PLAN.md §5.3."
```

---

## Branch and PR

- Branch: `feat/stage2-sft-pipeline-design`
- PR: #79 — https://github.com/ulises-c/csen-346/pull/79
