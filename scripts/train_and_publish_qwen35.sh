#!/usr/bin/env bash
# Unattended: train the Qwen3.5-0.8B-Base LoRA state-classifier (with W&B), then
# publish the merged checkpoint to the HF Hub (private). Retries the full run on
# transient CUDA faults — the script has no mid-run checkpointing, so each attempt
# starts from step 0 — but bails immediately on OOM (retrying won't help) and
# publishes only on a clean completion. Mirrors the original 9126a72 invocation.
set -euo pipefail

REPO_ROOT=/home/ulises/github/csen-346
cd "$REPO_ROOT"

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
  uv run python -u scripts/train_state_classifier_34way.py \
    --model-id "$BASE_MODEL" \
    --lora --lora-r 8 --lora-alpha 16 \
    --batch_size 4 \
    --gradient-checkpointing \
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
uv run python scripts/publish_to_hf.py \
  --results-dir "$OUT" \
  --repo "$REPO_ID" \
  --base-model "$BASE_MODEL" 2>&1 | tee -a "$ORCH_LOG" >&2
log "publish complete → https://huggingface.co/${REPO_ID}"
