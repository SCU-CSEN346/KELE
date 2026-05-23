#!/usr/bin/env bash
# One-shot: convert SocratTeachLLM (HF format) → GGUF (Q8_0) for llama.cpp serving.
#
# SocratTeachLLM is a LoRA fine-tune of THUDM/glm-4-9b-chat (ChatGLM4 architecture,
# ~9B params). The HF repo distributes the merged weights as 4× safetensors shards
# totaling ~18 GB in fp16. This script runs llama.cpp's convert_hf_to_gguf.py to
# produce a Q8_0 GGUF (~10 GB, nearly lossless) suitable for `serve_socratteachllm_llamacpp.sh`.
#
# Why Q8_0 (not Q5/Q4): we want minimal quantization-induced behavioral drift
# on this model specifically — its value is the fine-tune-specific Socratic-teaching
# behavior we're trying to characterize for the paper's overfit-hypothesis test.
# At ~10 GB on a 32 GB 5090, the extra memory cost vs Q5 is negligible.
#
# Usage:
#   ./scripts/convert_socratteachllm_to_gguf.sh
#
# Override paths with env vars:
#   HF_DIR=/path/to/hf  GGUF_DIR=/path/to/out  QUANT=Q5_K_M ./scripts/convert_socratteachllm_to_gguf.sh

set -euo pipefail

HF_DIR="${HF_DIR:-$HOME/hf_models/SocratTeachLLM}"
GGUF_DIR="${GGUF_DIR:-$HOME/Documents/models/weights}"
QUANT="${QUANT:-Q8_0}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/Documents/models/llama.cpp}"
CONVERT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZE="$LLAMA_CPP_DIR/build/bin/llama-quantize"

F16_GGUF="$GGUF_DIR/SocratTeachLLM-f16.gguf"
OUT_GGUF="$GGUF_DIR/SocratTeachLLM-${QUANT}.gguf"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -d "$HF_DIR" ]]; then
  echo "ERROR: HF model directory not found at $HF_DIR" >&2
  echo "Download with: hf download yuanpan/SocratTeachLLM --local-dir $HF_DIR" >&2
  echo "(or set HF_DIR=/some/other/path)" >&2
  exit 1
fi

if [[ ! -f "$CONVERT" ]]; then
  echo "ERROR: convert_hf_to_gguf.py not found at $CONVERT" >&2
  echo "Build llama.cpp first or set LLAMA_CPP_DIR=/path/to/llama.cpp" >&2
  exit 1
fi

if [[ ! -x "$QUANTIZE" ]]; then
  echo "ERROR: llama-quantize binary not found at $QUANTIZE" >&2
  echo "Build llama.cpp first: cmake --build $LLAMA_CPP_DIR/build --config Release" >&2
  exit 1
fi

mkdir -p "$GGUF_DIR"

if [[ -f "$OUT_GGUF" ]]; then
  echo "SocratTeachLLM-${QUANT}.gguf already exists at $OUT_GGUF"
  echo "Delete and re-run if you want to re-convert."
  exit 0
fi

# ── Step 1: HF → f16 GGUF (intermediate; ~18 GB) ──────────────────────────────
if [[ ! -f "$F16_GGUF" ]]; then
  echo "=== Step 1: HF → f16 GGUF ==="
  echo "Input:  $HF_DIR"
  echo "Output: $F16_GGUF"
  echo "(this takes ~5 min on CPU; no GPU needed)"
  echo
  cd "$LLAMA_CPP_DIR"
  # ChatGLM4 needs --model-name override because the HF config sets it to "ChatGLMModel"
  # The output type "f16" is the safest intermediate (preserves all precision for the
  # quantization step). For odd config issues, try --no-check (some ChatGLM forks).
  uv run python "$CONVERT" "$HF_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16
  cd - >/dev/null
fi

# ── Step 2: f16 GGUF → Q8_0 (or whatever QUANT is set to) ─────────────────────
echo
echo "=== Step 2: f16 GGUF → $QUANT ==="
echo "Input:  $F16_GGUF"
echo "Output: $OUT_GGUF"
echo "(this takes ~1 min on CPU)"
echo
"$QUANTIZE" "$F16_GGUF" "$OUT_GGUF" "$QUANT"

# ── Optional cleanup ──────────────────────────────────────────────────────────
echo
echo "=== Done ==="
ls -lh "$OUT_GGUF"
echo
echo "Intermediate f16 GGUF (~18 GB) is preserved at:"
echo "  $F16_GGUF"
echo "Delete it to reclaim disk space:"
echo "  rm $F16_GGUF"
echo
echo "Next: bash scripts/serve_socratteachllm_llamacpp.sh"
