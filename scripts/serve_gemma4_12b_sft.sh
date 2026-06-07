#!/usr/bin/env bash
# Serve the merged Socratic-SFT Gemma 4 12B (Q8_0 GGUF) as an OpenAI-compatible API
# on the NVIDIA RTX 4000 Ada. Thin wrapper over scripts/serve_gemma4_31b.sh.
#
# The weight file is produced by the merge+convert pipeline (see the 12B PoC plan):
#   scripts/merge_lora_gemma4_sft.py --base google/gemma-4-12b-it \
#     --adapter outputs/sft-gemma4-12b-qlora/final --out outputs/sft-gemma4-12b-qlora/merged
#   bash scripts/convert_gemma4_12b_sft_to_gguf.sh
# The convert wrapper writes gemma-4-12B-kele-socratic-sft-Q8_0.gguf AND stages it
# into the weights dir below (matching GEMMA4_12B_SFT_WEIGHT_FILE) — no rename needed.
#
# Distinct alias "Gemma 4 12B SFT" so the eval orchestrator proves the SFT weights
# answered (gemma4-12b-sft-local.env). Stop the base server first — one model at a
# time on the 20 GB card.
#
# Endpoint: http://localhost:8080/v1/chat/completions
# Override the weights dir with GEMMA4_12B_WEIGHTS_DIR=/some/path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GEMMA4_12B_WEIGHTS_DIR="${GEMMA4_12B_WEIGHTS_DIR:-$HOME/Documents/models/weights}"
WEIGHT_FILE="${GEMMA4_12B_SFT_WEIGHT_FILE:-gemma-4-12B-kele-socratic-sft-Q8_0.gguf}"

exec "$SCRIPT_DIR/serve_gemma4_31b.sh" \
  -m "$GEMMA4_12B_WEIGHTS_DIR/$WEIGHT_FILE" \
  -a "Gemma 4 12B SFT" \
  -c "${GEMMA4_12B_CTX:-32768}" \
  "$@"
