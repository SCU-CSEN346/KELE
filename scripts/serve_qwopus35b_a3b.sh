#!/usr/bin/env bash
# Serve Qwopus3.6-35B-A3B-v1 (Q4_K_M) for tournament evaluation.
#
# ~21 GB model + ~4 GB KV (Q4_0, 32K ctx unified) = ~25 GB. ~7 GB headroom on R9700.
# Qwen3.6-35B-A3B base with LoRA reasoning fine-tune.
# Uses --reasoning off: model has Qwen3 thinking mode, suppressed server-side.
#
# Boot:  bash scripts/serve_qwopus35b_a3b.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHT_FILE="${QWOPUS35B_WEIGHT_FILE:-$HOME/models/Qwopus3.6-35B-A3B-v1/Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf}"

exec "$SCRIPT_DIR/serve_qwen27b.sh" \
  -m "$WEIGHT_FILE" \
  -a "Qwopus 35B A3B" \
  -c 32768 \
  --reasoning off \
  "$@"
