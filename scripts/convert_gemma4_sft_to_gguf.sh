#!/usr/bin/env bash
# Convert merged Gemma-4-31B SFT (HF BF16) → f16 GGUF → Q5_K_M GGUF.
#
# Pipeline step 2+3 of 3 — assumes scripts/merge_lora_gemma4_sft.py already ran
# and produced outputs/sft-stage2-gemma4-31b/merged/.
#
# Naming convention: the KELE-tagged filename signals "this is the CSEN-346 KELE
# Socratic-SFT product" and must coexist with — never overwrite — the base
# gemma-4-31b-it-*.gguf in ~/Documents/models/weights/ that other projects use.
#
# Why Q5_K_M not Q5_K_XL: Q5_K_XL is Unsloth Dynamic's variant (per-layer
# importance-weighted) and is not natively producible by llama.cpp's
# llama-quantize tool. Q5_K_M is the standard llama.cpp Q5 quant at the same
# bit budget — ~5.33 bpw, +0.0569 ppl on Llama-3-8B (per llama-quantize --help).
# The size and quality are essentially equivalent for our serving use case.
#
# Usage:
#   bash scripts/convert_gemma4_sft_to_gguf.sh
#
# Override:
#   MERGED_DIR=... GGUF_DIR=... QUANT=Q5_K_M  bash scripts/convert_gemma4_sft_to_gguf.sh

set -euo pipefail

# Resolve relative paths against the repo root so `cd "$LLAMA_CPP_DIR"` below
# doesn't break them. realpath -m allows paths whose final component doesn't
# exist yet (the GGUF_DIR/output files are about to be created).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MERGED_DIR="$(realpath -m "${MERGED_DIR:-$REPO_ROOT/outputs/sft-stage2-gemma4-31b/merged}")"
GGUF_DIR="$(realpath -m "${GGUF_DIR:-$REPO_ROOT/outputs/sft-stage2-gemma4-31b}")"
QUANT="${QUANT:-Q5_K_M}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/Documents/models/llama.cpp}"
CONVERT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZE="$LLAMA_CPP_DIR/build/bin/llama-quantize"

# KELE-tagged filenames — must NOT collide with base gemma-4-31b-it-*.gguf.
NAME_TAG="gemma-4-31B-kele-socratic-sft"
F16_GGUF="$GGUF_DIR/${NAME_TAG}-f16.gguf"
OUT_GGUF="$GGUF_DIR/${NAME_TAG}-${QUANT}.gguf"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if [[ ! -d "$MERGED_DIR" ]]; then
  echo "ERROR: Merged HF checkpoint not found at $MERGED_DIR" >&2
  echo "Run scripts/merge_lora_gemma4_sft.py first." >&2
  exit 1
fi

if [[ ! -f "$CONVERT" ]]; then
  echo "ERROR: convert_hf_to_gguf.py not found at $CONVERT" >&2
  exit 1
fi

if [[ ! -x "$QUANTIZE" ]]; then
  echo "ERROR: llama-quantize binary not found at $QUANTIZE" >&2
  exit 1
fi

mkdir -p "$GGUF_DIR"

if [[ -f "$OUT_GGUF" ]]; then
  echo "$OUT_GGUF already exists. Delete and re-run to overwrite."
  exit 0
fi

# ── Step 1: HF (BF16 merged) → f16 GGUF ───────────────────────────────────────
if [[ ! -f "$F16_GGUF" ]]; then
  echo "=== Step 1: HF → f16 GGUF ==="
  echo "Input:  $MERGED_DIR"
  echo "Output: $F16_GGUF"
  echo "(~10-15 min on CPU, ~62 GB write)"

  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
  fi

  cd "$LLAMA_CPP_DIR"
  "$PYTHON" "$CONVERT" "$MERGED_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16
  cd - >/dev/null
fi

# ── Step 2: f16 GGUF → Q5_K_M ─────────────────────────────────────────────────
echo
echo "=== Step 2: f16 GGUF → $QUANT ==="
echo "Input:  $F16_GGUF"
echo "Output: $OUT_GGUF"
echo "(~5-10 min on CPU)"

"$QUANTIZE" "$F16_GGUF" "$OUT_GGUF" "$QUANT"

# ── Done ──────────────────────────────────────────────────────────────────────
echo
echo "=== Done ==="
ls -lh "$OUT_GGUF"
echo
echo "Intermediate f16 GGUF (~62 GB) preserved at:"
echo "  $F16_GGUF"
echo "Delete with:  rm $F16_GGUF"
echo
echo "Next: smoke-test, then copy to ~/Documents/models/weights/"
echo "      cp $OUT_GGUF \$HOME/Documents/models/weights/"
