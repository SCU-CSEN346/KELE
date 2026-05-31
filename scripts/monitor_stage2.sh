#!/usr/bin/env bash
# This script builds GitHub-markdown PR comments via printf; the format strings
# are single-quoted on purpose (%s positional args + literal backticks/newlines),
# so SC2016 ("expressions don't expand in single quotes") is a false positive.
# shellcheck disable=SC2016
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE2_LOG="$REPO_DIR/outputs/sft-stage2-gemma4-31b/train.log"
FINAL_DIR="$REPO_DIR/outputs/sft-stage2-gemma4-31b/final"
SELF_LOG="$REPO_DIR/outputs/monitor_stage2.log"
PR_NUMBER=101
MAX_RETRIES=2
POLL_SECONDS=300

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SELF_LOG"
}

post_pr() {
    gh pr comment "$PR_NUMBER" --body "$1" >> "$SELF_LOG" 2>&1 || \
        log "WARNING: gh pr comment failed"
}

training_running() {
    pgrep -f "train_sft\.py" > /dev/null 2>&1
}

last_step() {
    if [[ -f "$STAGE2_LOG" ]]; then
        grep -oE "[0-9]+/[0-9]+" "$STAGE2_LOG" | tail -1 || true
    fi
}

crash_hint() {
    if [[ -f "$STAGE2_LOG" ]]; then
        grep -i "page fault\|traceback\|runtimeerror\|out of memory\|killed" "$STAGE2_LOG" \
            | tail -3 | sed 's/^/  /' || true
    fi
}

kill_and_clean() {
    log "Killing stray train_sft.py processes"
    pkill -9 -f "train_sft\.py" 2>/dev/null || true
    local waited=0
    while pgrep -f "train_sft\.py" > /dev/null 2>&1 && [[ $waited -lt 30 ]]; do
        sleep 2; (( waited += 2 ))
    done
    log "KFD settle wait (20s)"
    sleep 20
}

start_stage2() {
    log "Starting Stage 2 training"
    cd "$REPO_DIR" && make train-gemma4-31b-stage2-unsloth
    sleep 30
}

log "Stage 2 monitor starting (PR #$PR_NUMBER, poll every ${POLL_SECONDS}s, max $MAX_RETRIES retries)"
post_pr "**Stage 2 monitoring active** — watching \`outputs/sft-stage2-gemma4-31b/train.log\` for crashes. Will post on completion or failure."

sleep 30  # let it get going before first poll

retries=0

while true; do
    if training_running; then
        step="$(last_step)"
        log "Stage 2 running${step:+ — step $step}"
        sleep "$POLL_SECONDS"
        continue
    fi

    if [[ -d "$FINAL_DIR" ]]; then
        log "Stage 2 complete — adapter at $FINAL_DIR"
        total_steps="$(last_step || true)"
        post_pr "$(printf '## Stage 2 Training: COMPLETE ✓\n\nFull QLoRA fine-tune finished.\n\n- Adapter: `outputs/sft-stage2-gemma4-31b/final`\n- Last recorded step: %s\n\nNext: serving generation-primer audit before downstream eval (see handoff §2).' "${total_steps:-unknown}")"
        log "Stage 2 done — monitor exiting"
        exit 0
    fi

    hint="$(crash_hint)"
    log "CRASH DETECTED (retry $retries/$MAX_RETRIES)"
    [[ -n "$hint" ]] && log "Log tail:$hint"

    if (( retries >= MAX_RETRIES )); then
        post_pr "$(printf '## Stage 2 Training: CRASHED (max retries)\n\nExceeded %d retries. Manual intervention needed.\n\nCrash hint:\n```\n%s\n```\n\nLog: `outputs/sft-stage2-gemma4-31b/train.log`' "$MAX_RETRIES" "$hint")"
        log "Max retries exceeded — exiting"
        exit 1
    fi

    post_pr "$(printf '**Stage 2 CRASHED** (retry %d/%d) — restarting.\n\n```\n%s\n```' "$((retries+1))" "$MAX_RETRIES" "$hint")"

    retries=$(( retries + 1 ))
    kill_and_clean
    start_stage2
done
