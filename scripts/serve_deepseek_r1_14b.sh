#!/usr/bin/env bash
# Serve DeepSeek-R1-Distill-Qwen-14B Q5 for tournament evaluation.
#
# ~10 GB model + KV (32K ctx, q4_0) — fits comfortably on 32 GB VRAM.
# Thinking disabled server-side via --reasoning off.
#
# Boot:  bash scripts/serve_deepseek_r1_14b.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHTS_DIR="${DEEPSEEK_R1_14B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${DEEPSEEK_R1_14B_WEIGHT_FILE:-DeepSeek-R1-Distill-Qwen-14B-Q5_K_M.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "DeepSeek R1 14B" \
  -c 32768 \
  --reasoning off \
  "$@"
