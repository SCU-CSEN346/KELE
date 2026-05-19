#!/usr/bin/env bash
# Serve Qwen3.6-27B Q4_K_M for tournament evaluation with thinking ENABLED.
#
# Companion to serve_qwen27b_q4_local.sh — identical except --reasoning off is NOT
# passed, so Qwen3's chat-template thinking mechanism is honored.
#
# Boot:  bash scripts/serve_qwen27b_q4_local_think.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHT_FILE="${QWEN27B_Q4_WEIGHT_FILE:-$HOME/models/Qwen3.6-27B/Qwen3.6-27B-Q4_K_M.gguf}"

exec "$SCRIPT_DIR/serve_qwen27b.sh" \
  -m "$WEIGHT_FILE" \
  -a "Qwen 27B Q4" \
  -c 131072 \
  -np 4 \
  "$@"
