#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B (MoE: 256 experts, 9 active per token, 3B active of 35B
# total) UD-Q4_K_M as an OpenAI-compatible API server on the RTX 5090.
#
# 512K context: ~20 GB model + ~3.1 GB KV (Q4_0, ~6.3 KB/token vs 27B's ~18) =
# ~23.1 GB. ~9 GB VRAM headroom. Could safely run at 1M context.
#
# Endpoint: http://localhost:8080/v1/chat/completions
#
# This shares port 8080 with serve_qwen27b_q5.sh — only one Qwen variant can
# run at a time (5090 cannot fit both: 27B Q5 26.3 GB + A3B 23.1 GB = 49.4 GB).
# The eval orchestrator's alias verification will bail if the wrong one is up.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

QWEN35B_WEIGHTS_DIR="${QWEN35B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${QWEN35B_WEIGHT_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"

exec "$SCRIPT_DIR/serve_qwen27b.sh" \
  -m "$QWEN35B_WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Qwen 35B A3B" \
  -c 524288 \
  "$@"
