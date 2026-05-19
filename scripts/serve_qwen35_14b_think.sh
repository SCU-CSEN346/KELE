#!/usr/bin/env bash
# Serve Qwen3-14B Q4 for tournament evaluation with thinking ENABLED.
#
# Companion to serve_qwen35_14b.sh — identical except --reasoning off is NOT
# passed, so Qwen3's chat-template thinking mechanism is honored.
#
# Boot:  bash scripts/serve_qwen35_14b_think.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHTS_DIR="${QWEN3_14B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${QWEN3_14B_WEIGHT_FILE:-Qwen3-14B-UD-Q4_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Qwen 14B Q4" \
  -c 32768 \
  "$@"
