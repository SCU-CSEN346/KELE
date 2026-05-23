#!/usr/bin/env bash
# Serve Gemma 4 31B-it Unsloth UD-Q5_K_XL GGUF as an OpenAI-compatible API
# server on the RTX 5090.
#
# 180K context: ~26 GB total / ~6 GB headroom on the 32 GB 5090. Reduced from
# the earlier 220K ceiling on 2026-05-23 after the bilingual probe hit CUDA
# OOM trying to load the T4 BERT consultant on top of a 220K Gemma server —
# the headroom was simply too thin for a co-resident consultant.
# Per-token KV scales: 128K→200K=20 KB/token, 200K→250K=25 KB/token (heavier
# slots above 200K). 180K stays in the cheap regime AND leaves ~6 GB for
# whatever co-resident BERT consultant the eval needs.
# Per-slot ~30K context across 6 slots — still well above KELE turn size (<10K).
#
# Both the KELE teacher and consultant hit this same server (configs/
# gemma4-31b-local.env). Shares port 8080 with the other Qwen/Gemma variants
# — only one fits at a time on the 5090's 32 GB.
#
# Endpoint: http://localhost:8080/v1/chat/completions
#
# Override the weights dir with:
#   GEMMA4_31B_WEIGHTS_DIR=/some/other/path ./scripts/serve_gemma4_31b_q5.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GEMMA4_31B_WEIGHTS_DIR="${GEMMA4_31B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${GEMMA4_31B_WEIGHT_FILE:-gemma-4-31B-it-UD-Q5_K_XL.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$GEMMA4_31B_WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Gemma 4 31B" \
  -c 184320 \
  "$@"
