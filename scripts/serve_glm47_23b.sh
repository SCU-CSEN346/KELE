#!/usr/bin/env bash
# Serve GLM-4.7-Flash REAP 23B-A3B (Unsloth UD-Q4_K_XL) for tournament evaluation.
#
# ~14 GB model + ~4 GB KV (Q4_0, 32K ctx unified) = ~18 GB. ~14 GB headroom on R9700.
# Uses --reasoning off: GLM-4.7-Flash has thinking mode, suppressed server-side.
#
# Boot:  bash scripts/serve_glm47_23b.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHT_FILE="${GLM47_23B_WEIGHT_FILE:-$HOME/models/GLM-4.7-Flash-REAP-23B-A3B/GLM-4.7-Flash-REAP-23B-A3B-UD-Q4_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHT_FILE" \
  -a "GLM 23B A3B" \
  -c 32768 \
  --reasoning off \
  "$@"
