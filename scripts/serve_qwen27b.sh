#!/usr/bin/env bash
# Generic llama.cpp OpenAI-compatible server for the Qwen 27B local experiments.
#
# Mirrors the launcher pattern from ~/Documents/models/scripts/serve.sh so a
# single llama-server build (and weight directory) can serve both the personal
# AI stack and KELE without duplication.
#
# Usage:
#   ./scripts/serve_qwen27b.sh -m <weight.gguf> [-a alias] [-c ctx] [-np slots] [-p port] [extra args...]
#
# Model-specific scripts (e.g. serve_qwen27b_q5.sh) call this with preset
# defaults; any extra args pass through to llama-server.
#
# Endpoint: http://localhost:<port>/v1/chat/completions

set -euo pipefail

if [[ -z "${LLAMA_SERVER:-}" ]]; then
  if [[ -x "$HOME/Github/llama.cpp/build/bin/llama-server" ]]; then
    LLAMA_SERVER="$HOME/Github/llama.cpp/build/bin/llama-server"
  else
    LLAMA_SERVER="$HOME/Documents/models/llama.cpp/build/bin/llama-server"
  fi
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "Error: llama-server not found at $LLAMA_SERVER" >&2
  echo "Build llama.cpp first — see ~/Documents/models/REFERENCE.md" >&2
  echo "Or override with:  LLAMA_SERVER=/path/to/llama-server $0 ..." >&2
  exit 1
fi

# Defaults (overridable by model scripts or CLI flags) — match upstream serve.sh
MODEL=""
ALIAS=""
CONTEXT=65536
KV_QUANT="q4_0"
GPU_LAYERS=99
PARALLEL=6
HOST="0.0.0.0"
PORT=8080

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)             MODEL="$2";      shift 2 ;;
    -a|--alias)     ALIAS="$2";      shift 2 ;;
    -c)             CONTEXT="$2";    shift 2 ;;
    -np|--parallel) PARALLEL="$2";   shift 2 ;;
    -ctk|-ctv)      KV_QUANT="$2";   shift 2 ;;
    -ngl)           GPU_LAYERS="$2"; shift 2 ;;
    --host)         HOST="$2";       shift 2 ;;
    -p|--port)      PORT="$2";       shift 2 ;;
    *)              EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: no model specified. Use -m <path> or a model-specific script (serve_qwen27b_q5.sh)." >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Error: weight file not found at $MODEL" >&2
  echo "Download it first:  hf download unsloth/Qwen3.6-27B-GGUF $(basename "$MODEL") --local-dir \$(dirname \"$MODEL\")" >&2
  exit 1
fi

ALIAS_ARGS=()
if [[ -n "$ALIAS" ]]; then
  ALIAS_ARGS=(-a "$ALIAS")
fi

echo "=== Qwen 27B (llama.cpp) ==="
echo "Binary:    $LLAMA_SERVER"
echo "Weights:   $MODEL"
echo "Alias:     ${ALIAS:-<none>}"
echo "Context:   $CONTEXT  (per-slot: ~$((CONTEXT / PARALLEL)))"
echo "Slots:     $PARALLEL  (unified KV, continuous batching)"
echo "KV quant:  ${KV_QUANT}"
echo "Endpoint:  http://localhost:${PORT}/v1/chat/completions"
echo "Test:      curl http://localhost:${PORT}/v1/models"
echo "---"

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  "${ALIAS_ARGS[@]}" \
  -ngl "$GPU_LAYERS" \
  -c "$CONTEXT" \
  -np "$PARALLEL" \
  --kv-unified \
  -ctk "$KV_QUANT" \
  -ctv "$KV_QUANT" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
