#!/usr/bin/env bash
# Serve Gemma 3 27B Q4 for tournament evaluation.
#
# ~18 GB model + KV (32K ctx, q4_0) — fits on 32 GB VRAM.
#
# Boot:  bash scripts/serve_gemma3_27b.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHTS_DIR="${GEMMA3_27B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${GEMMA3_27B_WEIGHT_FILE:-gemma-3-27b-it-Q4_K_M.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Gemma 3 27B" \
  -c 32768 \
  "$@"
