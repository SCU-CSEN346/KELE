#!/usr/bin/env bash
# Serve Gemma 4 31B-it Unsloth UD-Q5_K_XL GGUF as an OpenAI-compatible API
# server on the RTX 5090.
#
# 150K context: tighter than the previous 180K ceiling. Reduced 2026-05-26
# after the t4-bert-gemma-fewshot10-n681 crash showed we were operating too
# close to the 32 GB VRAM limit — peak transients during checkpoint
# restore/erase (~250 MiB each, up to 32 per slot) were spiking into the
# headroom. 180K traces: ~26 GB resident / ~6 GB headroom (insufficient).
# Earlier history: 220K → CUDA OOM with co-resident T4 BERT consultant
# (2026-05-23). Per-token KV scales: 128K→200K=20 KB/token, 200K→250K=25
# KB/token. 150K saves ~600 MB vs 180K AND stays well within the cheap regime.
# Per-slot ~38K context across 4 slots — still well above KELE turn size (<10K).
# Slot count matches KELE_PARALLEL_WORKERS=4 (see serve_gemma4_31b.sh PARALLEL).
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
  -c 153600 \
  "$@"
