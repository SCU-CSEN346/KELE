#!/usr/bin/env bash
# Serve Qwen3.6-27B Q4_K_M from ~/models/ for tournament evaluation on AMD R9700.
#
# 34 GB VRAM: ~16 GB model + ~4.5 GB KV (Q4_0, 128K ctx, 4 slots) = ~21 GB. ~13 GB headroom.
# 4 parallel slots, unified KV.
#
# Boot:  bash scripts/serve_qwen27b_q4_local.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHT_FILE="${QWEN27B_Q4_WEIGHT_FILE:-$HOME/models/Qwen3.6-27B/Qwen3.6-27B-Q4_K_M.gguf}"

exec "$SCRIPT_DIR/serve_qwen27b.sh" \
  -m "$WEIGHT_FILE" \
  -a "Qwen 27B Q4" \
  -c 131072 \
  -np 4 \
  --reasoning off \
  "$@"
