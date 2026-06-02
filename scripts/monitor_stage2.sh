#!/usr/bin/env bash
# This script builds GitHub-markdown PR comments via printf; the format strings
# are single-quoted on purpose (%s positional args + literal backticks/newlines),
# so SC2016 ("expressions don't expand in single quotes") is a false positive.
# shellcheck disable=SC2016
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$REPO_DIR/outputs/sft-stage2-gemma4-31b"
STAGE2_LOG="$OUTPUT_DIR/train.log"
FINAL_DIR="$OUTPUT_DIR/final"
SELF_LOG="$REPO_DIR/outputs/monitor_stage2.log"
PR_NUMBER=101
# The gfx1201 fault is non-deterministic, so the run advances by crashing and
# resuming from the latest checkpoint (save_steps=10). MAX_RETRIES bounds
# CONSECUTIVE retries that make NO forward progress — a run that keeps advancing
# never exhausts it; only a genuinely stuck GPU (no new checkpoint across this
# many tries) stops the monitor. Crawl-forward needs a high bound, not 2.
MAX_RETRIES=8
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

# The make target writes train.log with `>` (truncate), so each relaunch would
# overwrite the crashed run's full traceback. Archive the whole log + a dmesg
# snapshot per crash BEFORE relaunch so an overnight crawl leaves a reviewable
# trail of every fault, not just the 3-line hint.
archive_crash_log() {
    local ckpt="$1" ts dest crashdir
    [[ -f "$STAGE2_LOG" ]] || return 0
    ts="$(date '+%Y%m%d-%H%M%S')"
    crashdir="$OUTPUT_DIR/crashlogs"
    mkdir -p "$crashdir"
    dest="$crashdir/crash-step${ckpt}-${ts}.log"
    if cp "$STAGE2_LOG" "$dest" 2>/dev/null; then
        log "Archived full crash log → $dest"
    fi
    dmesg 2>/dev/null \
        | grep -iE "amdgpu|gfxhub|VM_L2|page fault|PERMISSION_FAULTS|WALKER_ERROR" \
        | tail -40 > "$crashdir/dmesg-step${ckpt}-${ts}.log" 2>/dev/null || true
}

kill_and_clean() {
    log "Killing stray train_sft.py processes"
    pkill -9 -f "train_sft\.py" 2>/dev/null || true
    local waited=0
    while pgrep -f "train_sft\.py" > /dev/null 2>&1 && [[ $waited -lt 30 ]]; do
        sleep 2; (( waited += 2 ))
    done
    # A gfx1201 fault leaves the amdkfd dirty (orphaned HIP context + stale VRAM);
    # relaunching into that state faults early on stale PTEs (the cascade that
    # turned a single crash into a permanent loop). Verify the GPU is ACTUALLY
    # clean before relaunch instead of a blind sleep (GFX1201_RDNA4_TRAINING.md §10).
    if ! bash "$REPO_DIR/scripts/test_gpu_stack.sh" --wait-clean 180 >> "$SELF_LOG" 2>&1; then
        log "WARNING: GPU still dirty after 180s — relaunch will likely fault on stale PTEs"
    fi
}

latest_ckpt_step() {
    local d max=-1 n
    for d in "$OUTPUT_DIR"/checkpoint-*; do
        [[ -d "$d" ]] || continue
        n="${d##*checkpoint-}"
        [[ "$n" =~ ^[0-9]+$ ]] && (( n > max )) && max="$n"
    done
    printf '%s' "$max"
}

# A crash landing during a checkpoint write leaves an incomplete checkpoint-N.
# HF resume always picks the highest-numbered dir, so a partial one makes every
# resume fail to load and loop on the same bad checkpoint. trainer_state.json is
# written last in _save_checkpoint, so a valid one implies the rest is complete;
# if it's missing or unparsable, quarantine the dir so resume falls back to N-1.
quarantine_bad_checkpoint() {
    local step latest
    step="$(latest_ckpt_step)"
    [[ "$step" -lt 0 ]] && { log "No checkpoint yet — resume will start from step 0"; return 0; }
    latest="$OUTPUT_DIR/checkpoint-$step"
    if [[ -f "$latest/trainer_state.json" ]] \
        && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$latest/trainer_state.json" 2>/dev/null; then
        log "Latest checkpoint OK: checkpoint-$step (resume target)"
        return 0
    fi
    log "checkpoint-$step is INCOMPLETE (crash mid-save) — quarantining; resume falls back to prior"
    mv "$latest" "$OUTPUT_DIR/.broken-checkpoint-$step" 2>/dev/null || rm -rf "$latest"
}

start_stage2() {
    # Use a fresh data_seed each launch so resume presents different samples at each
    # step — breaking the data-order-sticky fault confirmed in run #13 (PR #101).
    # Verified (run #14 verif): TRAIN_DATA_SEED genuinely reshuffles the post-resume
    # order and step 22 (previously the sticky fault step) completed clean under seed 99.
    # Unix timestamp gives a distinct seed every resume (each cycle is ≥10 min apart).
    local data_seed
    data_seed="$(date +%s)"
    log "Starting Stage 2 training (data_seed=$data_seed, gpu-preflight gates the launch)"
    cd "$REPO_DIR" || return 0
    TRAIN_DATA_SEED="$data_seed" WANDB_PROJECT=csen346-sft make train-gemma4-31b-stage2-unsloth \
        || log "launch aborted (gpu-preflight failed?) — counts as no forward progress"
    sleep 30
}

log "Stage 2 monitor starting (PR #$PR_NUMBER, poll every ${POLL_SECONDS}s, max $MAX_RETRIES retries)"
post_pr "**Stage 2 monitoring active** — watching \`outputs/sft-stage2-gemma4-31b/train.log\` for crashes. Will post on completion or failure."

sleep 30  # let it get going before first poll

retries=0
last_progress_step="$(latest_ckpt_step)"

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

    # Forward-progress check: did a new checkpoint land since the last crash? If
    # so the crawl is advancing — reset the no-progress retry counter. Only
    # consecutive stalls (no new checkpoint) count toward MAX_RETRIES, so a run
    # that keeps inching forward across faults never gives up.
    ckpt_step="$(latest_ckpt_step)"
    if (( ckpt_step > last_progress_step )); then
        log "Progress since last crash: checkpoint $last_progress_step → $ckpt_step — resetting retry counter"
        last_progress_step="$ckpt_step"
        retries=0
    fi

    hint="$(crash_hint)"
    log "CRASH DETECTED (consecutive no-progress retries $retries/$MAX_RETRIES, latest ckpt step $ckpt_step)"
    [[ -n "$hint" ]] && log "Log tail:$hint"
    # Preserve the full traceback + dmesg BEFORE relaunch overwrites train.log.
    archive_crash_log "$ckpt_step"

    if (( retries >= MAX_RETRIES )); then
        post_pr "$(printf '## Stage 2 Training: STALLED (no progress in %d retries)\n\nLatest checkpoint step: %s. The run is not advancing across resumes — manual intervention needed (run `make diagnose-gfx1201-fault`).\n\nCrash hint:\n```\n%s\n```\n\nPer-crash full tracebacks + dmesg: `outputs/sft-stage2-gemma4-31b/crashlogs/`' "$MAX_RETRIES" "$ckpt_step" "$hint")"
        log "Stalled — no forward progress in $MAX_RETRIES retries — exiting"
        exit 1
    fi

    post_pr "$(printf '**Stage 2 CRASHED** (no-progress retry %d/%d, latest ckpt step %s) — cleaning GPU + resuming.\n\n```\n%s\n```' "$((retries+1))" "$MAX_RETRIES" "$ckpt_step" "$hint")"

    retries=$(( retries + 1 ))
    kill_and_clean
    quarantine_bad_checkpoint
    start_stage2
done
