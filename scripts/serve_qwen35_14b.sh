#!/usr/bin/env bash
# Serve Qwen3-14B Q4 for tournament evaluation.
#
# ~9 GB model + KV (32K ctx, q4_0) — fits comfortably on 32 GB VRAM.
# Thinking disabled server-side via --reasoning off.
#
# Boot:  bash scripts/serve_qwen35_14b.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHTS_DIR="${QWEN3_14B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${QWEN3_14B_WEIGHT_FILE:-Qwen3-14B-UD-Q4_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Qwen 14B Q4" \
  -c 32768 \
  --reasoning off \
  "$@"
