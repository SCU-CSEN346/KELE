#!/usr/bin/env bash
# Start the mlx_lm consultant server and verify it's reachable.
# Used when running the KELE eval from a host PC against a Mac Mini consultant.
#
# Run this on the Mac Mini before starting the eval on the host PC.
# If mlx_lm.server is already running on the expected port, this script just
# verifies connectivity and prints the endpoint — no extra process needed.
#
# Usage:
#   source configs/consultants/m4-mlx.env
#   ./scripts/serve_consultant_mlx.sh

set -euo pipefail

HOST="${CONSULTANT_HOST:-0.0.0.0}"
PORT="${CONSULTANT_PORT:-8080}"
MODEL="${CONSULTANT_MODEL_NAME:?Set CONSULTANT_MODEL_NAME to the HF repo ID (e.g. source configs/consultants/m4-mlx.env)}"
LOG_FILE="${CONSULTANT_LOG_FILE:-logs/mlx_consultant.log}"
KV_SIZE="${CONSULTANT_NUM_CTX:-16384}"

mkdir -p logs

echo "=== mlx_lm Consultant (Mac Mini) ==="
echo "Model:    $MODEL"
echo "Host:     $HOST"
echo "Port:     $PORT"
echo "KV size:  $KV_SIZE"
echo "Log:      $LOG_FILE"
echo ""

if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "mlx_lm.server is already running on port $PORT."

    MLX_PID=$(lsof -iTCP:"$PORT" -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1 {print $2}' | head -1)
    if [[ -n "$MLX_PID" ]] && command -v caffeinate &>/dev/null; then
        caffeinate -s -w "$MLX_PID" &
        echo "Sleep inhibited (caffeinate -s -w $MLX_PID)."
    fi
else
    echo "mlx_lm.server not running — starting..."

    mlx_lm.server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --max-kv-size "$KV_SIZE" \
        > "$LOG_FILE" 2>&1 &
    MLX_PID=$!
    echo "PID: $MLX_PID  Log: $LOG_FILE"

    if command -v caffeinate &>/dev/null; then
        caffeinate -s -w "$MLX_PID" &
        echo "Sleep inhibited (caffeinate -s -w $MLX_PID)."
    fi

    echo -n "Waiting for mlx_lm.server to be ready (model load may take ~30s)..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
            printf " ready! (~%ss)\n" "$i"
            break
        fi
        sleep 1
    done

    if ! curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "ERROR: mlx_lm.server failed to start. Check $LOG_FILE"
        exit 1
    fi

    echo "Warming up model..."
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
        > /dev/null
    echo "Model loaded."
fi

MAC_HOSTNAME=$(scutil --get LocalHostName 2>/dev/null || echo "")
MAC_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<mac-ip>")
MAC_ADDR="${MAC_HOSTNAME:+${MAC_HOSTNAME}.local}"
MAC_ADDR="${MAC_ADDR:-$MAC_IP}"

echo ""
echo "=== Consultant ready ==="
echo "OpenAI-compatible endpoint: http://${MAC_ADDR}:${PORT}/v1"
echo ""
echo "On the host PC, set:"
echo "  export CONSULTANT_BASE_URL=http://${MAC_ADDR}:${PORT}/v1"
echo "  export CONSULTANT_MODEL_NAME=${MODEL}"
echo "  export CONSULTANT_API_KEY=no-key"
echo ""
echo "Test from host PC:"
echo "  curl http://${MAC_ADDR}:${PORT}/v1/models"
