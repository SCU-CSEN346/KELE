#!/usr/bin/env bash
# Self-host the LOCKED HEADLINE / top-performer stack for a live demo:
#   Qwen3.5-0.8B-LoRA state-classifier consultant  ×  Gemma 4 31B teacher
#   + 10-shot stage-balanced exemplars   (unified 72.24, n=681, 2026-05-26)
#
# Target hardware: a single NVIDIA RTX 5090 (32 GB, CUDA0) — the box that
# produced the locked headline. The teacher is served by llama.cpp on
# 0.0.0.0:8080, so it is reachable over the Tailscale tailnet without any
# extra proxy. The interactive Socratic REPL then runs on this host; at the
# event you SSH into this box over Tailscale and drive the REPL here.
#
# Demo usage (run on the RTX 5090 host, ideally inside tmux so it survives
# an SSH drop on event wifi):
#
#   tmux new -s demo
#   ./scripts/serve_demo_top_performer.sh
#
# The llama-server is booted DETACHED and left running on exit, so restarting
# the conversation is instant (no 30-90s cold reload). To take it down:
#   kill "$(cat results/_demo/llama_server.pid)"   # or: pkill -f llama-server
#
# Overrides:
#   BERT_CKPT=...        classifier checkpoint dir (default: the wandb run's final/)
#   GEMMA4_31B_WEIGHTS_DIR=...   Gemma GGUF dir (passed through to serve script)
#   PORT=8080            llama-server port
#   FEW_SHOT_N=10        teacher exemplar count (headline = 10)

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

# ── Config ───────────────────────────────────────────────────────────────────
EXPERIMENT="gemma4-31b-local"          # teacher → Gemma 4 31B on localhost:8080
# The headline classifier weights live in the W&B training run's final/ dir.
# (The bare results/state-clf-qwen3.5-0.8b-lora/final referenced in the eval
# scripts only ever held test_eval.json on this box; the real safetensors are
# under the -wandb run.) HF fallback: ulises-c/socrates-state-classifier-qwen3.5-lora
BERT_CKPT="${BERT_CKPT:-results/state-clf-qwen3.5-0.8b-lora-wandb/final}"
PORT="${PORT:-8080}"
FEW_SHOT_N="${FEW_SHOT_N:-10}"
LLAMA_URL="http://localhost:${PORT}"
EXPECTED_ALIAS="Gemma 4 31B"
DEMO_DIR="results/_demo"
PID_FILE="$DEMO_DIR/llama_server.pid"

mkdir -p "$DEMO_DIR"

# ── Pre-flight ───────────────────────────────────────────────────────────────
if [[ ! -d "$BERT_CKPT" ]] || [[ ! -f "$BERT_CKPT/model.safetensors" ]]; then
  printf 'error: classifier checkpoint missing at %s\n' "$BERT_CKPT" >&2
  printf '       fetch it with:  hf download ulises-c/socrates-state-classifier-qwen3.5-lora --local-dir %s\n' "$BERT_CKPT" >&2
  printf '       or set BERT_CKPT=/path/to/final\n' >&2
  exit 1
fi

echo "=== Top-performer demo: Qwen3.5-LoRA classifier × Gemma 4 31B + ${FEW_SHOT_N}-shot ==="
echo "Host:       $(hostname)"
echo "Experiment: $EXPERIMENT  (teacher → Gemma 4 31B @ $LLAMA_URL)"
echo "Consultant: $BERT_CKPT  (34-state classifier)"
echo

# ── GPU sanity (RTX 5090) ────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "GPU VRAM free: ${VRAM_FREE} MiB"
  if [[ "$VRAM_FREE" =~ ^[0-9]+$ ]] && [[ "$VRAM_FREE" -lt 26500 ]]; then
    printf 'warn: < 26.5 GB VRAM free — Gemma 4 31B Q5 may OOM at 150K context.\n' >&2
  fi
else
  printf 'warn: nvidia-smi not found — expected an RTX 5090 (CUDA0) host.\n' >&2
fi

# ── Tailscale / remote-access info for the event ─────────────────────────────
echo
echo "--- Remote access (Tailscale) ---"
if command -v tailscale &>/dev/null; then
  TS_IP=$(tailscale ip -4 2>/dev/null | head -1)
  if [[ -n "${TS_IP:-}" ]]; then
    echo "Tailnet IP:    ${TS_IP}"
    echo "SSH from event: ssh ${USER}@${TS_IP}   (then run this script inside tmux)"
    echo "Teacher API:    http://${TS_IP}:${PORT}/v1   (OpenAI-compatible, if hitting it directly)"
  else
    printf 'warn: tailscale installed but no tailnet IP — run "tailscale up" first.\n' >&2
  fi
else
  printf 'warn: tailscale not found — install/login so the event laptop can reach this box.\n' >&2
fi
if command -v systemctl &>/dev/null && ! systemctl is-active --quiet sshd 2>/dev/null \
   && ! systemctl is-active --quiet ssh 2>/dev/null; then
  printf 'warn: sshd does not appear active — you may not be able to SSH in over the tailnet.\n' >&2
fi
echo

# ── Server lifecycle (persistent: boot once, reuse on restart) ───────────────
ready_check() {
  local resp
  resp=$(curl -s --max-time 3 "$LLAMA_URL/v1/models" 2>/dev/null) || return 1
  [[ -z "$resp" ]] && return 1
  echo "$resp" | grep -q '"Loading model"' && return 1
  echo "$resp" | grep -q "\"$EXPECTED_ALIAS\""
}

if ready_check; then
  echo "Reusing existing $EXPECTED_ALIAS server on $LLAMA_URL."
else
  TS=$(date -u +%Y-%m-%dT%H-%M-%S)
  SERVER_LOG="$DEMO_DIR/server_${TS}.log"
  echo "Booting llama-server detached (cold load: 30-90s) → $SERVER_LOG"
  # nohup + disown keeps the server alive after the REPL (and this script)
  # exit — instant demo restarts. serve_gemma4_31b_q5.sh exec-chains straight
  # to llama-server, so $! is the real server PID (survives an SSH drop).
  nohup bash "$ROOT/scripts/serve_gemma4_31b_q5.sh" \
    --port "$PORT" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  disown "$SERVER_PID" 2>/dev/null || true
  echo "$SERVER_PID" > "$PID_FILE"

  echo -n "Waiting for $EXPECTED_ALIAS to be ready "
  for i in $(seq 1 180); do
    if ready_check; then
      echo " ready (~$((i * 2))s)"
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo " FAILED — server process died."
      tail -30 "$SERVER_LOG" >&2
      exit 1
    fi
    echo -n "."
    sleep 2
  done
  if ! ready_check; then
    echo " TIMEOUT after 360s"
    exit 1
  fi
fi

# ── Interactive Socratic teaching session ────────────────────────────────────
echo
echo "=== Launching interactive session ==="
echo "Note: the dataset/headline is Chinese — the student prompt is '你:'."
echo "      Type your math question, or 'quit'/Ctrl-C to end the conversation."
echo "      The server stays up for instant restarts (kill: cat $PID_FILE)."
echo

PATH="$ROOT/.venv/bin:$PATH" \
KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N="$FEW_SHOT_N" \
  uv run python -m src.project.kele \
    --experiment "$EXPERIMENT" \
    interactive \
    --bert-consultant "$BERT_CKPT"
