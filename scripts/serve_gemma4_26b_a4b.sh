#!/usr/bin/env bash
# Serve Gemma 4 26B-A4B-it Unsloth UD-Q4_K_XL GGUF as an OpenAI-compatible API.
#
# MoE variant: ~4B active params per token, ~17 GB VRAM at Q4 — fits comfortably
# on any 32 GB device and runs 4-5x faster than the dense Gemma 4 31B.
#
# Override the weights dir with:
#   GEMMA4_26B_WEIGHTS_DIR=/some/other/path ./scripts/serve_gemma4_26b_a4b.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GEMMA4_26B_WEIGHTS_DIR="${GEMMA4_26B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${GEMMA4_26B_WEIGHT_FILE:-gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$GEMMA4_26B_WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Gemma 4 26B A4B" \
  -c 131072 \
  "$@"
