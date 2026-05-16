#!/usr/bin/env bash
# Autonomous chainer for the 2026-05-16 → 2026-05-17 weekend window.
#
# Waits for the in-flight Gemma 4 31B full run to complete, then if there is
# >= 90 min remaining before the Sun 15:00 PDT deadline, runs one follow-up
# experiment: standalone Gemma 4 31B + 10-shot exemplars at n=50.
#
# This fills a gap in the level-up campaign's n=50 leaderboard — we have
# A3B+10-shot LLM-only and BERT+Gemma+10-shot, but no Gemma+10-shot LLM-only
# datapoint. Decomposes whether the headline's gains are from BERT, the
# teacher swap, or both.
#
# Crash-safe: if Gemma full crashes, this chainer will sleep forever waiting
# for its metrics_summary.json — that's the safer failure mode.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

GEMMA_METRICS="results/gemma4-31b-local-unified/metrics_summary.json"
DEADLINE_EPOCH=$(date -d "2026-05-17 15:00:00 PDT" +%s)

LOG="logs/autonomous_chain_$(date -u +%Y%m%dT%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Autonomous chainer started: $(date) ==="
echo "Deadline: Sun 2026-05-17 15:00 PDT"
echo "Waiting for Gemma full to complete: $GEMMA_METRICS"
echo

# Poll for metrics_summary.json appearance (created at end of run)
while [[ ! -f "$GEMMA_METRICS" ]]; do
  sleep 300
done

echo "Gemma full completed: $(date)"
echo
cat "$GEMMA_METRICS"
echo

NOW_EPOCH=$(date +%s)
SECONDS_REMAINING=$(( DEADLINE_EPOCH - NOW_EPOCH ))
HOURS_REMAINING=$(( SECONDS_REMAINING / 3600 ))
MIN_REMAINING=$(( (SECONDS_REMAINING % 3600) / 60 ))

echo "Time remaining before deadline: ${HOURS_REMAINING}h ${MIN_REMAINING}m"
echo

# Need 90 min minimum for the follow-up (n=50 Gemma ≈ 50-70 min + setup)
if [[ $SECONDS_REMAINING -lt 5400 ]]; then
  echo "Insufficient time for follow-up experiment (< 90 min remaining). Stopping."
  exit 0
fi

echo "=== Launching follow-up: standalone Gemma + 10-shot exemplars (n=50) ==="
echo

# Wait for the Gemma full orchestrator to fully tear down its server before
# starting a fresh server (defensive: orchestrator does its own teardown,
# but it may not be complete at the moment metrics_summary lands).
sleep 30

# Check port 8080 is free before we proceed
until ! curl -s --max-time 2 http://localhost:8080/v1/models > /dev/null 2>&1; do
  echo "Port 8080 still serving — waiting another 30 s..."
  sleep 30
done

OUT_DIR="results/gemma4-31b-local-fewshot10-n50"
mkdir -p "$OUT_DIR"

# Boot server in background, then run eval
nohup bash "$ROOT/scripts/serve_gemma4_31b_q5.sh" > "$OUT_DIR/server_$(date -u +%Y%m%dT%H%M%S).log" 2>&1 &
SRV_PID=$!

# Wait for server ready
echo -n "Waiting for Gemma 4 31B server ..."
for _ in $(seq 1 180); do
  if curl -s --max-time 3 http://localhost:8080/v1/models 2>/dev/null | grep -q '"Gemma 4 31B"'; then
    echo " ready"
    break
  fi
  echo -n "."
  sleep 2
done

# Run eval with unified + 10-shot exemplars
PATH="$ROOT/.venv/bin:$PATH" \
KELE_FEW_SHOT_TEACHER=1 KELE_FEW_SHOT_N=10 \
  uv run python -m src.project.kele \
    --experiment gemma4-31b-local \
    --unified \
    test --n 50 --output "$OUT_DIR"

EVAL_EXIT=$?

# Teardown server
kill "$SRV_PID" 2>/dev/null || true
sleep 5
kill -9 "$SRV_PID" 2>/dev/null || true

echo
echo "=== Follow-up complete: exit=$EVAL_EXIT  $(date) ==="
if [[ -f "$OUT_DIR/metrics_summary.json" ]]; then
  cat "$OUT_DIR/metrics_summary.json"
fi
