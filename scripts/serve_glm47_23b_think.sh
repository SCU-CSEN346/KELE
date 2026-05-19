#!/usr/bin/env bash
# Serve GLM-4.7-Flash REAP 23B-A3B for tournament evaluation with thinking ENABLED.
#
# Companion to serve_glm47_23b.sh — identical except --reasoning off is NOT
# passed, so GLM-4.7's thinking mechanism is honored.
#
# Boot:  bash scripts/serve_glm47_23b_think.sh
# Stop:  kill %1

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WEIGHT_FILE="${GLM47_23B_WEIGHT_FILE:-$HOME/models/GLM-4.7-Flash-REAP-23B-A3B/GLM-4.7-Flash-REAP-23B-A3B-UD-Q4_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$WEIGHT_FILE" \
  -a "GLM 23B A3B" \
  -c 32768 \
  "$@"
