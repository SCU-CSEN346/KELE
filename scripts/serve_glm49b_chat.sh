#!/usr/bin/env bash
# Serve GLM-4-9B-Chat-base (THUDM/glm-4-9b-chat) via vLLM on port 8001.
# Mirror of scripts/serve_socratteachllm.sh — same architecture (ChatGLM4),
# same launch flags. Only the model path and alias differ.
#
# Why vLLM (not llama.cpp): the llama.cpp ChatGLM converter doesn't embed BPE
# merges, so llama-server fails with "cannot find tokenizer merges in model
# file" when loading any ChatGLM4 GGUF. This affects both this base model AND
# the existing SocratTeachLLM-Q8_0.gguf — they share the bug. vLLM works
# directly from HF weights via trust_remote_code.
#
# Usage: ./scripts/serve_glm49b_chat.sh
#
# NOTE — AMD gfx1201 (R9700 AI PRO): vLLM is NOT recommended on this GPU.
# Use serve_teacher_local.sh (HF Transformers) on the R9700 instead.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="${TEACHER_MODEL_NAME:-GLM-4-9B-Chat-base}"
MODEL_PATH="${TEACHER_MODEL_PATH:-${HF_HOME:-$HOME/hf_models}/glm-4-9b-chat}"
HOST="${TEACHER_HOST:-0.0.0.0}"
PORT="${TEACHER_PORT:-8001}"
MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.70}"
LOG_FILE="${TEACHER_LOG_FILE:-logs/vllm_glm49b_chat.log}"

mkdir -p logs

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download it first:"
    echo "  hf download THUDM/glm-4-9b-chat --local-dir $MODEL_PATH"
    exit 1
fi

# ── GPU architecture detection ────────────────────────────────────────────────
GPU_VENDOR="unknown"
GPU_CC=0

if command -v nvidia-smi &>/dev/null; then
    _CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -1 | tr -d '.' | tr -d ' ') || true
    if [[ -n "$_CC" ]]; then
        GPU_VENDOR="nvidia"
        GPU_CC="$_CC"
    fi
fi

if [[ "$GPU_VENDOR" == "unknown" ]] && command -v rocm-smi &>/dev/null; then
    GPU_VENDOR="amd"
fi

echo "GPU vendor: $GPU_VENDOR  (CC: ${GPU_CC:0:1}.${GPU_CC:1:-0})"

if [[ -n "${TEACHER_DTYPE:-}" ]]; then
    DTYPE="$TEACHER_DTYPE"
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    DTYPE="bfloat16"
elif [[ "$GPU_CC" -ge 80 ]]; then
    DTYPE="bfloat16"
else
    DTYPE="float16"
fi

EXTRA_ARGS=()
if [[ "$GPU_VENDOR" == "nvidia" && "$GPU_CC" -lt 80 ]]; then
    EXTRA_ARGS+=(--enforce-eager)
fi

echo "Serving $MODEL_NAME (teacher) on port $PORT..."
echo "Model path: $MODEL_PATH"
echo "Host: $HOST"
echo "DTYPE: $DTYPE"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Log: $LOG_FILE"
echo "Test: curl http://localhost:$PORT/v1/models"
echo "---"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
    echo "ERROR: vllm binary not found at $VLLM_BIN" >&2
    echo "Run 'uv sync' in the repo to install vllm, or set VLLM_BIN=/path/to/vllm" >&2
    exit 1
fi

# vLLM 0.21+ spawns subprocesses that runtime-compile CUDA kernels via ninja.
# nohup'd shells don't inherit the venv's bin on PATH, so the subprocesses
# fail with `FileNotFoundError: 'ninja'`. Prepend it explicitly.
export PATH="$REPO_ROOT/.venv/bin:$PATH"

exec "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
