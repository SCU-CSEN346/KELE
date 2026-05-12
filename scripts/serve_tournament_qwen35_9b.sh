#!/usr/bin/env bash
# Serve Qwen3.5-9B UD-Q4_K_XL on port 8080 for tournament evaluation.
#
# Note: the translation server (make start-local-tl-server) uses this same
# model on port 8000. This script serves it on 8080 so the tournament
# orchestrator can hit it alongside other llama.cpp models.
#
# Context: 32K tokens (9B model + Q4 KV fits in <14 GB, ~18 GB headroom).
# Both the KELE teacher and consultant point to this server
# (configs/tournament-qwen35-9b.env).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHT_FILE="${QWEN35_9B_WEIGHT_FILE:-$HOME/models/Qwen3.5-9B/Qwen3.5-9B-UD-Q4_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHT_FILE" \
  -a "Qwen 9B Q4" \
  -c 32768 \
  --reasoning off \
  "$@"
