#!/usr/bin/env bash
# Unattended: train the Qwen3.5-0.8B-Base LoRA state-classifier (with W&B), then
# publish the merged checkpoint to the HF Hub (private). Retries the full run on
# transient GPU faults — the trainer writes an epoch-level checkpoint so each retry
# resumes rather than restarting — but bails immediately on OOM (retrying won't
# help) and publishes only on a clean completion.
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Backend detection: this branch runs on two machines — the AMD ROCm R9700
# (gfx1201) and the NVIDIA CUDA RTX 4000 Ada. They need different pre-checks,
# precision, and memory strategy:
#   - gfx1201 page-faults under hipBLASLt and is 5× slower in bf16, so it trains
#     pure fp32, and OOMs in the linear-attn layers without grad-checkpointing —
#     it keeps --gradient-checkpointing.
#   - Ada has native bf16 tensor cores (--bf16-autocast) and, with the
#     flash-linear-attention GDN kernels engaged, is compute-bound at bs=8: the
#     run fits in 12.6/20 GB WITHOUT grad-checkpointing, and dropping it removes
#     ~35% recompute overhead (measured 174→235 steps/min, 4.4h→3.3h).
# Detect once and branch.
# ---------------------------------------------------------------------------
BACKEND=$(uv run --no-sync python -c \
  'import torch; print("rocm" if torch.version.hip else "cuda")')
if [[ "$BACKEND" == "rocm" ]]; then
  GPU_TEST=scripts/test_training_gpu_amd.py
  PRECISION_FLAG=()
  GRADCKPT_FLAG=(--gradient-checkpointing)
  export TORCH_USE_HIPBLASLT=0
else
  GPU_TEST=scripts/test_training_gpu_nvidia.py
  PRECISION_FLAG=(--bf16-autocast)
  GRADCKPT_FLAG=()
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
printf '[pre-check] backend=%s  verifying GPU stack via %s ...\n' "$BACKEND" "$GPU_TEST"

# Runs under --no-sync so the manually-installed ROCm/CUDA torch wheel is not
# reverted by uv.
uv run --no-sync python "$GPU_TEST"
printf '[pre-check] GPU OK\n\n'

OUT=results/state-clf-qwen3.5-0.8b-lora-wandb
REPO_ID=ulises-c/socrates-state-classifier-qwen3.5-lora
BASE_MODEL=Qwen/Qwen3.5-0.8B-Base
MAX_ATTEMPTS=3
ORCH_LOG="$OUT/orchestrate.log"

mkdir -p "$OUT"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$ORCH_LOG" >&2; }

trained_ok=false
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  # Attempt 1 starts fresh; retries resume from the last epoch checkpoint (same
  # W&B run), so a transient fault costs at most the in-progress epoch, not 13h.
  resume_flag=()
  if (( attempt > 1 )); then
    resume_flag=(--resume)
  fi
  log "training attempt ${attempt}/${MAX_ATTEMPTS} starting ${resume_flag[*]} ${PRECISION_FLAG[*]}"
  set +e
  WANDB_PROJECT=csen346-state-classifier \
  uv run --no-sync python -u scripts/train_state_classifier_34way.py \
    --model-id "$BASE_MODEL" \
    --lora --lora-r 8 --lora-alpha 16 \
    --batch_size 8 \
    "${GRADCKPT_FLAG[@]}" \
    --wandb \
    "${PRECISION_FLAG[@]}" \
    "${resume_flag[@]}" \
    --out-dir "$OUT" \
    >"$OUT/train.log" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]] && grep -q '^Done\.$' "$OUT/train.log"; then
    log "training attempt ${attempt} succeeded (clean Done.)"
    trained_ok=true
    break
  fi

  if grep -qiE "out of memory|CUDA out of memory" "$OUT/train.log"; then
    log "attempt ${attempt} failed with OOM — not retrying"
    break
  fi

  log "attempt ${attempt} failed (rc=${rc}); $(tail -n 1 "$OUT/train.log")"
  if (( attempt < MAX_ATTEMPTS )); then
    log "cooling down 60s before retry"
    sleep 60
  fi
done

if [[ "$trained_ok" != true ]]; then
  log "training did not complete after ${MAX_ATTEMPTS} attempts — NOT publishing"
  exit 1
fi

log "publishing to HF (private): ${REPO_ID}"
uv run --no-sync python scripts/publish_to_hf.py \
  --results-dir "$OUT" \
  --repo "$REPO_ID" \
  --base-model "$BASE_MODEL" 2>&1 | tee -a "$ORCH_LOG" >&2
log "publish complete → https://huggingface.co/${REPO_ID}"
