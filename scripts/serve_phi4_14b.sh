#!/usr/bin/env bash
# Serve Phi-4 14B Q5 for tournament evaluation.
#
# ~10 GB model + KV (32K ctx, q4_0) — fits on 32 GB VRAM.
#
# Boot:  bash scripts/serve_phi4_14b.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHTS_DIR="${PHI4_14B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${PHI4_14B_WEIGHT_FILE:-phi-4-Q5_K_M.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Phi 4 14B" \
  -c 32768 \
  "$@"
