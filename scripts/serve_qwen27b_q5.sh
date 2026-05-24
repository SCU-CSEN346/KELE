#!/usr/bin/env bash
# Serve Qwen3.6-27B UD-Q5_K_XL (clean variant, NOT the HauhauCS uncensored fine-tune)
# as an OpenAI-compatible API server on the RTX 5090.
#
# 256K context (= model's native n_ctx_train; eliminates the YaRN extrapolation
# warning we had at 416K). ~19 GB model + ~4.5 GB KV (Q4_0) + ~1.3 GB compute
# ≈ 24.8 GB. ~7 GB headroom on the 32 GB 5090 — wide enough to avoid the
# NVRM Xid 8 watchdog lockups we hit at 416K on 2026-05-22.
# 6 parallel slots, unified KV — both the KELE teacher and consultant hit this
# same server (configs/qwen27b-local.env).
#
# Endpoint: http://localhost:8080/v1/chat/completions
#
# Override the weights dir with:
#   QWEN27B_WEIGHTS_DIR=/some/other/path ./scripts/serve_qwen27b_q5.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

QWEN27B_WEIGHTS_DIR="${QWEN27B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${QWEN27B_WEIGHT_FILE:-Qwen3.6-27B-UD-Q5_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_qwen27b.sh" \
  -m "$QWEN27B_WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Qwen 27B Q5" \
  -c 262144 \
  --reasoning off \
  "$@"
