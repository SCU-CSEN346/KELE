# Handoff — Stage 2 Gemma 4 31B QLoRA crawl: STALLED — GPU reset required

**Date:** 2026-06-04 · **Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**For:** a fresh Claude instance picking up after the monitor STALLed due to a wedged GPU.

---

## TL;DR

**Monitor has exited (STALLED 8/8).** Everything is dead. The GPU has a stale 24 GB KFD
context that survived `pkill -9` — it was left by a tokenization deadlock that followed the
step-1230 crash. Every subsequent monitor retry failed preflight (GPU dirty), consumed all 8
retries, and the monitor posted STALLED to PR #101 at 17:19 on 2026-06-03 and exited.

**Before restarting anything you must reset the GPU driver.** Once clean, restart the monitor
and it will resume from checkpoint-1230. See "If the monitor STALLED" below for exact steps.

Progress banked: **checkpoint-1230 / 4826 steps = ~25.5% complete.**

---

## Current state

| Item | Value |
|---|---|
| Training script | `scripts/train_sft.py` via `make train-gemma4-31b-stage2-unsloth` |
| Monitor | **DEAD** — exited after STALL at 17:19 2026-06-03 |
| Monitor log | `outputs/monitor_stage2.log` |
| Train log | `outputs/sft-stage2-gemma4-31b/train.log` |
| Latest checkpoint | `checkpoint-1230` (safe — trainer_state.json intact) |
| Total steps | 4,826 (1 epoch, 77k records, batch 1×16) |
| Per-step time | ~71 s/it |
| GPU state | **DIRTY** — 24 GB VRAM consumed, stale KFD context, needs driver reset |
| W&B project | `csen346-sft` at `uchavarria-santa-clara-university` (active on next resume) |

**Quick status check:**
```bash
tail -5 outputs/monitor_stage2.log
ls -d outputs/sft-stage2-gemma4-31b/checkpoint-* | sort -V | tail -3
rocm-smi --showmeminfo vram   # must show near-zero before restart
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

**Current situation (2026-06-04):** The 8 retries were NOT page-fault stalls — they were
preflight failures caused by a wedged GPU. The root cause:

1. Training crashed at step 1230 (normal page fault)
2. Monitor restarted; tokenization deadlocked in multiprocessing at 12868/77202 samples
3. `pkill -9` cleared the Python processes but left a stale 24 GB KFD context on the GPU
4. Every subsequent monitor retry failed preflight (VRAM 24 GB > 1024 MB threshold)
5. 8 failed preflights = 8 "no progress" retries → STALL → monitor exited

**Fix sequence:**

1. **Verify GPU is still dirty:**
   ```bash
   rocm-smi --showmeminfo vram
   # expect ~24 GB used despite no processes running
   ```

2. **Check if nvtop is holding the KFD context** (it was open on `/dev/kfd` in lsof):
   ```bash
   lsof /dev/kfd /dev/dri/renderD128 2>/dev/null | grep -v "^COMMAND"
   pkill nvtop   # if listed; recheck VRAM after ~5s
   ```

3. **If VRAM still dirty — reload the GPU driver** (this is the nuclear option but clean):
   ```bash
   sudo modprobe -r amdgpu && sudo modprobe amdgpu
   rocm-smi --showmeminfo vram   # should now show ~57 MB
   ```

4. **Confirm GPU clean, then restart monitor:**
   ```bash
   make gpu-preflight   # must PASS before continuing
   nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &
   tail -f outputs/monitor_stage2.log
   ```
   Monitor will resume from checkpoint-1230 automatically.

5. **Watch for tokenization deadlock on the first restart.** If train.log stalls at
   `Tokenizing train dataset (num_proc=12): XX%` for more than 60 seconds without
   advancing, the multiprocessing pool deadlocked again. In that case:
   ```bash
   pkill -9 -f 'train_sft\.py'
   # wait for monitor to detect + clean + retry — usually clears on next attempt
   ```
   If it deadlocks repeatedly, consider patching `train_sft.py` to use
   `dataset_num_proc=1` (slower tokenization, no deadlock risk).

6. If monitor STALLs again after a clean GPU restart → **post to PR #101** and escalate.
   Cloud GPU (CUDA) is the remaining option.

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

## Session history

**Session 4 (2026-06-02):** Crawl launched from checkpoint-90. Made forward progress to
checkpoint-1230 overnight (~50 crashes, all recovered). Tokenization deadlock after
step-1230 crash wedged the GPU; monitor STALLed 8/8 at 17:19 on 2026-06-03.

**Session 5 (2026-06-04):** Monitor and training dead, GPU dirty (24 GB wedged KFD context).
No code changes this session — GPU driver reset needed before restart.

```
21fc7ab  feat(diag): add rocblas bench capture + replay scripts for gfx1201 repro
9016f9f  fix(monitor): crash_hint matches the real gfx1201 fault signature
d4ce31d  feat(train): enable W&B in config (next-crash pickup) + pin run for continuity
623bb06  docs(handoff): session 4 complete — crawl running, rotating seed fix live
aebcc3f  feat(monitor): pass WANDB_PROJECT to training launches
587b60e  feat(monitor): rotate TRAIN_DATA_SEED per resume to break sticky fault
```

## Key files

| File | Purpose |
|---|---|
| `docs/GFX1201_FAULT_ABLATION_LOG.md` | Canonical run log — append after every run |
| `scripts/monitor_stage2.sh` | Crawl harness — rotating seed, KFD cleanup, PR posts |
| `scripts/train_sft.py` | Training script; `TRAIN_DATA_SEED` + `TRAIN_LOG_DATA_ORDER` wired |
| `outputs/sft-stage2-gemma4-31b/crashlogs/` | Per-crash full log + dmesg archive |
| PR #101 | Full diagnostic thread + all session findings |
