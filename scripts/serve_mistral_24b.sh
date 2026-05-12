#!/usr/bin/env bash
# Serve Mistral-Small-3.2-24B-Instruct Q4 for tournament evaluation.
#
# ~15 GB model + KV (32K ctx, q4_0) — fits on 32 GB VRAM.
#
# Boot:  bash scripts/serve_mistral_24b.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHTS_DIR="${MISTRAL_24B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${MISTRAL_24B_WEIGHT_FILE:-Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Mistral Small 24B" \
  -c 32768 \
  "$@"
