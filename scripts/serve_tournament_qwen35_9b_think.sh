#!/usr/bin/env bash
# Serve Qwen3.5-9B UD-Q4_K_XL for tournament evaluation with thinking ENABLED.
#
# Companion to serve_tournament_qwen35_9b.sh — identical except --reasoning off
# is NOT passed, so Qwen3's chat-template thinking mechanism is honored.
#
# Boot:  bash scripts/serve_tournament_qwen35_9b_think.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHT_FILE="${QWEN35_9B_WEIGHT_FILE:-$HOME/models/Qwen3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHT_FILE" \
  -a "Qwen 9B Q4" \
  -c 32768 \
  "$@"
