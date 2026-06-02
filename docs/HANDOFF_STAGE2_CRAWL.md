# Handoff — Stage 2 Gemma 4 31B QLoRA crawl: training in progress

**Date:** 2026-06-02 · **Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**For:** a fresh Claude instance picking up while training is running (or has crashed and resumed).

---

## TL;DR

Stage 2 Gemma 4 31B QLoRA is running autonomously via `monitor_stage2.sh`. The gfx1201
ISA1201 Tensile kernel bug is still present but **no longer blocking** — the crawl harness
rotates `TRAIN_DATA_SEED` on every resume, breaking the data-order stickiness that caused
run #13 to stall. As of handoff: step ~96/4826, checkpoint-90, no crashes since checkpoint-22.

**Your job is to monitor and respond to the outcome**, not to debug the fault (closed) or
re-run ablation arms (all done). See `docs/GFX1201_FAULT_ABLATION_LOG.md` for the full
diagnostic record.

---

## Current state

| Item | Value |
|---|---|
| Training script | `scripts/train_sft.py` via `make train-gemma4-31b-stage2-unsloth` |
| Monitor PID | check `pgrep -f monitor_stage2` |
| Monitor log | `outputs/monitor_stage2.log` |
| Train log | `outputs/sft-stage2-gemma4-31b/train.log` |
| Latest checkpoint | `checkpoint-90` (as of handoff; will advance) |
| Total steps | 4,826 (1 epoch, 77k records, batch 1×16) |
| Per-step time | ~71 s/it |
| Estimated wall clock | ~93 h base compute from step 0 |
| W&B project | `csen346-sft` at `uchavarria-santa-clara-university` (active on next resume) |

**Quick status check:**
```bash
tail -5 outputs/monitor_stage2.log
ls -d outputs/sft-stage2-gemma4-31b/checkpoint-* | sort -V | tail -3
```

---

## How the crawl works

`monitor_stage2.sh` runs a loop: launch training → poll every 5 min → on crash: archive
log + dmesg, clean GPU (waits up to 180s for KFD to drain), quarantine any partial
checkpoint, relaunch with a **fresh `TRAIN_DATA_SEED=$(date +%s)`**.

The rotating seed is the key fix (commit `587b60e`): HF Trainer restores `rng_state.pth`
on resume, which previously pinned the same samples to the same steps and caused the same
fault every time. A new `data_seed` reshuffles the post-resume sample sequence each cycle,
converting the sticky-deterministic fault back to the probabilistic regime of run #2
(which sometimes cleared 100 steps). Verified: all 12 M values differed between seed=default
and seed=99; step 22 (previously 8/8 fault) completed clean.

**Forward-progress guard:** `MAX_RETRIES=8` consecutive no-progress retries (no new
checkpoint across 8 tries) → monitor posts `STALLED` to PR #101 and exits. A new checkpoint
resets the counter, so a crawl that keeps advancing never exhausts it.

---

## If training is running

Leave it alone. Watch the PR for crash/progress posts from the monitor. The training is
autonomous — no action needed unless:
- The monitor posts `STALLED` → see "If STALLED" below
- A crash post shows something unexpected (e.g., OOM, not a page fault)
- Loss diverges (grad_norm consistently > 50, loss climbing after step 100)

---

## If training crashed and the monitor is handling it

Normal — the monitor will clean, reseed, and relaunch. You'll see a PR comment like
`**Stage 2 CRASHED** (no-progress retry N/8, latest ckpt step X)`. No action needed
unless it STALLs.

---

## If the monitor STALLED

Monitor posted: `## Stage 2 Training: STALLED (no progress in 8 retries)`

This means 8 consecutive resumes all faulted before banking a new checkpoint. Options:

1. **Run `make gpu-preflight`** — confirm GPU is actually clean, not stuck in a bad KFD state
2. **Check the latest crashlog** in `outputs/sft-stage2-gemma4-31b/crashlogs/` — is the
   fault still a page fault, or is it something new (OOM, driver crash)?
3. **Relaunch the monitor** with a manually chosen seed far from the timestamp cluster:
   ```bash
   TRAIN_DATA_SEED=1234567 make train-gemma4-31b-stage2-unsloth  # one test run
   # if it advances past a checkpoint:
   nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &
   ```
4. If consistently stalling regardless of seed → **post to PR #101** and escalate to
   the consultant. Cloud GPU (CUDA) is the remaining option.

---

## If training completed

Monitor posted: `## Stage 2 Training: COMPLETE ✓`

Adapter is at `outputs/sft-stage2-gemma4-31b/final/`. Next steps:

1. **Verify the adapter loads:**
   ```bash
   uv run --no-sync python -c "
   from peft import PeftModel
   from transformers import AutoModelForCausalLM
   m = PeftModel.from_pretrained('outputs/sft-stage2-gemma4-31b/final', 'unsloth/gemma-4-31B-it-unsloth-bnb-4bit')
   print('OK')
   "
   ```
2. **Merge and export for serving:**
   ```bash
   uv run --no-sync python scripts/merge_lora_gemma4_sft.py
   ```
3. **Run downstream eval** (BERT consultant routing + LLM judge) — post results to PR #101
4. **Update ablation log** (`docs/GFX1201_FAULT_ABLATION_LOG.md`) with final step count
   and any fault events observed

---

## Key invariants — do not violate

- **Always `uv run --no-sync`** — bare `uv run` reinstalls CUDA torch over ROCm
- **`make gpu-preflight` before any manual launch** — dirty KFD cascades into early faults
- **Do not add `AMD_SERIALIZE_KERNEL` or `HIP_LAUNCH_BLOCKING`** — SYNC mode makes the
  fault deterministic and non-representative; all SYNC runs were diagnostic only
- **Do not restart the monitor without checking if it's already running** (`pgrep -f monitor_stage2`)

---

## What is closed — do not re-investigate

The gfx1201 ISA1201 Tensile GEMM bug is **fully root-caused and documented**. All ablation
arms are exhausted. See `docs/GFX1201_FAULT_ABLATION_LOG.md` for the complete record.
The upstream report is being filed at `ROCm/rocm-libraries` with all necessary artifacts.

Do not:
- Re-run probe scripts (probes 1–3 are done)
- Try `TORCH_USE_HIPBLASLT=1` (run #11 — no kernel for this shape, Tensile fallback)
- Try `BNB_FORCE_B_CONTIGUOUS=1` (run #12 — Python `.contiguous()` can't reach BLAS descriptor)
- Try `expandable_segments:True` (run #10 — silently ignored on gfx1201)
- Try `ignore_data_skip=True` (rejected — silent garbage adapter, no stall alarm)

---

## Commit history (this session)

```
aebcc3f  feat(monitor): pass WANDB_PROJECT to training launches
587b60e  feat(monitor): rotate TRAIN_DATA_SEED per resume to break sticky fault
30509a3  docs(diag): record run #13 — crawl STALLED, data-order sticky fault
253a749  docs(handoff): session 3 complete — all fix arms exhausted, crawl phase next
7bd3b7e  feat(train): data_seed lever + data-order dump to break the sticky gfx1201 fault
6877a95  docs(diag): record crawl viability go/no-go; hold Arm E off the baseline
```

## Key files

| File | Purpose |
|---|---|
| `docs/GFX1201_FAULT_ABLATION_LOG.md` | Canonical run log — append after every run |
| `scripts/monitor_stage2.sh` | Crawl harness — rotating seed, KFD cleanup, PR posts |
| `scripts/train_sft.py` | Training script; `TRAIN_DATA_SEED` + `TRAIN_LOG_DATA_ORDER` wired |
| `outputs/sft-stage2-gemma4-31b/crashlogs/` | Per-crash full log + dmesg archive |
| PR #101 | Full diagnostic thread + all session findings |
