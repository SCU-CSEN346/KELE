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
# GPU pre-check: verify the device is visible and kernels execute before
# committing to the run. Runs under --no-sync so the manually-installed
# ROCm/CUDA torch wheel is not reverted by uv.
# ---------------------------------------------------------------------------
printf '[pre-check] verifying GPU stack ...\n'
uv run --no-sync python - <<'GPUCHECK'
import os, sys, time, torch

if not torch.cuda.is_available():
    print("ERROR: no GPU device visible (torch.cuda.is_available() == False)", file=sys.stderr)
    sys.exit(1)

name    = torch.cuda.get_device_name(0)
vram    = torch.cuda.get_device_properties(0).total_memory / 1e9
backend = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
print(f"  device  : {name}  ({vram:.1f} GB)")
print(f"  backend : {backend}  torch {torch.__version__}")

# fp32 + bf16 matmul (basic kernel smoke test)
x = torch.randn(4096, 4096, device="cuda")
t0 = time.perf_counter(); _ = (x @ x).sum().item(); ms = (time.perf_counter()-t0)*1000
xb = x.to(torch.bfloat16)
t0 = time.perf_counter(); _ = (xb @ xb).sum().item(); ms_bf16 = (time.perf_counter()-t0)*1000
print(f"  matmul  : fp32 {ms:.0f}ms  bf16 {ms_bf16:.0f}ms")

# Scaled-dot-product attention (exercises the same kernel path as Qwen3.5 RoPE/attention)
B, H, T, D = 2, 16, 128, 64
q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
t0 = time.perf_counter()
out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
out.sum().backward()
ms_attn = (time.perf_counter()-t0)*1000
print(f"  sdp-attn: {ms_attn:.0f}ms  backward OK")

print("  PASS")
GPUCHECK
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
  log "training attempt ${attempt}/${MAX_ATTEMPTS} starting ${resume_flag[*]}"
  set +e
  WANDB_PROJECT=csen346-state-classifier \
  uv run --no-sync python -u scripts/train_state_classifier_34way.py \
    --model-id "$BASE_MODEL" \
    --lora --lora-r 8 --lora-alpha 16 \
    --batch_size 8 \
    --gradient-checkpointing \
    --bf16-autocast \
    --wandb \
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
