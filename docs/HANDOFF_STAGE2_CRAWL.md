# Handoff — Stage 2 Gemma 4 31B QLoRA crawl

**Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**For:** a fresh Claude instance picking up after session 6.

---

## TL;DR

Training is **NOT running**. GPU state unknown — assume dirty (24 GB wedged KFD context from
session 4). **Before restarting anything, reset the GPU driver** (see below).

Session 6 added: HF checkpoint backup via `HFCheckpointCallback`, raised `save_total_limit`
to 5, extracted the callback to `src/project/hf_callback.py` with 7 unit tests. **These
changes are uncommitted** — commit them before anything else.

Progress banked: **checkpoint-1230 / 4826 steps ≈ 25.5% complete.**

---

## Immediate actions

### 1. Commit session 6 code changes (not yet committed)

```bash
git add scripts/train_sft.py src/project/hf_callback.py tests/test_hf_callback.py
git commit -m "refactor(train): extract HFCheckpointCallback to src/project/hf_callback + tests"
git push origin feat/gfx1201-rdna4-qlora-fla-training
```

### 2. Check GPU state

```bash
rocm-smi --showmeminfo vram
# If >1 GB used with no processes → GPU dirty, needs driver reset
lsof /dev/kfd /dev/dri/renderD128 2>/dev/null | grep -v "^COMMAND"
pkill nvtop    # if nvtop is listed, clear it and recheck VRAM
```

### 3. If GPU still dirty — driver reset

```bash
sudo modprobe -r amdgpu && sudo modprobe amdgpu
rocm-smi --showmeminfo vram   # must show ~57 MB
```

### 4. Restart monitor

```bash
make gpu-preflight   # must PASS
nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &
tail -f outputs/monitor_stage2.log
```

Monitor resumes from checkpoint-1230 automatically.

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
| GPU state | **Likely dirty** — 24 GB KFD context from session 4 stall |
| HF backup repo | `ulises-c/SocratesLM-31B-stage2b-QLoRA` (auto-push every 50 steps) |
| W&B project | `csen346-sft` at `uchavarria-santa-clara-university` |

---

## How the crawl works

`monitor_stage2.sh` runs a loop: launch training → poll every 5 min → on crash: archive
log + dmesg, clean GPU (waits up to 180s for KFD to drain), quarantine any partial
checkpoint, relaunch with a **fresh `TRAIN_DATA_SEED=$(date +%s)`**.

The rotating seed (commit `587b60e`) reshuffles the post-resume sample sequence each cycle,
converting the sticky-deterministic gfx1201 page fault back to probabilistic. Verified:
all 12 M values differed between seed=default and seed=99.

**Forward-progress guard:** `MAX_RETRIES=8` consecutive no-progress retries → monitor posts
`STALLED` to PR #101 and exits. A new checkpoint resets the counter.

**HF auto-backup:** `HFCheckpointCallback` (via `TRAIN_HF_REPO` env var set in Makefile)
pushes each saved checkpoint to `ulises-c/SocratesLM-31B-stage2b-QLoRA` in a daemon thread.
On-disk `save_total_limit=5` keeps only the last 5 local checkpoints; HF keeps all of them.

---

## If training is running

Leave it alone. Watch PR #101 for crash/progress posts. The training is autonomous — no
action needed unless:
- The monitor posts `STALLED` → see below
- A crash post shows something unexpected (OOM, not a page fault)
- Loss diverges (grad_norm consistently > 50, loss climbing after step 100)

---

## If training crashed and the monitor is handling it

Normal — monitor will clean, reseed, and relaunch. You'll see a PR comment like
`**Stage 2 CRASHED** (no-progress retry N/8, latest ckpt step X)`. No action needed
unless it STALLs.

---

## If the monitor STALLED

Monitor posted: `## Stage 2 Training: STALLED (no progress in 8 retries)`

**Most likely cause:** wedged GPU KFD context, not actual page-fault stalls. See session 5
root cause in Session History below. Fix sequence:

1. Verify GPU dirty: `rocm-smi --showmeminfo vram` → expect ~24 GB despite no processes
2. `lsof /dev/kfd /dev/dri/renderD128` → `pkill nvtop` if listed; recheck after 5s
3. If still dirty: `sudo modprobe -r amdgpu && sudo modprobe amdgpu`
4. `make gpu-preflight` → must PASS
5. `nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &`
6. Watch for tokenization deadlock on first restart: if `train.log` stalls at
   `Tokenizing train dataset (num_proc=12): XX%` for >60s without advancing, the
   multiprocessing pool deadlocked. `pkill -9 -f 'train_sft\.py'` — monitor will retry.
   If repeated: patch `train_sft.py` to use `dataset_num_proc=1`.
7. If STALLs again after clean GPU → post to PR #101, escalate to cloud GPU.

---

## If training completed

Monitor posted: `## Stage 2 Training: COMPLETE ✓`

Adapter at `outputs/sft-stage2-gemma4-31b/final/`. Next steps:

1. **Verify adapter loads:**
   ```bash
   uv run --no-sync python -c "
   from peft import PeftModel
   m = PeftModel.from_pretrained('outputs/sft-stage2-gemma4-31b/final',
       'unsloth/gemma-4-31B-it-unsloth-bnb-4bit')
   print('OK')
   "
   ```
2. Merge and export: `uv run --no-sync python scripts/merge_lora_gemma4_sft.py`
3. Run downstream eval (BERT consultant routing + LLM judge) — post to PR #101
4. Update `docs/GFX1201_FAULT_ABLATION_LOG.md` with final step count + fault events

---

## Key invariants — do not violate

- **Always `uv run --no-sync`** — bare `uv run` reinstalls CUDA torch over ROCm
- **`make gpu-preflight` before any manual launch** — dirty KFD cascades into early faults
- **Do not add `AMD_SERIALIZE_KERNEL` or `HIP_LAUNCH_BLOCKING`** — SYNC mode makes the
  fault deterministic and non-representative; all SYNC runs were diagnostic only
- **Do not restart the monitor without checking if it's already running** (`pgrep -f monitor_stage2`)

---

## What is closed — do not re-investigate

The gfx1201 ISA1201 Tensile GEMM bug is **fully root-caused and documented**. All
ablation arms are exhausted. See `docs/GFX1201_FAULT_ABLATION_LOG.md`.

The `ROCBLAS_LAYER=2` bench logging path is a dead end through PyTorch: PyTorch routes
through `libhipblas.so → librocblas.so` internal dispatch, bypassing the C API logging
hooks. Both B-matrix encodings for `rocblas-bench gemm_ex` dispatch `ISA000` (MLIR
generic), never `ISA1201`. The `ISA1201 MT64x64x64 DTVB1 BH_Bias_HA_S_SAV_UserArgs`
kernel requires the `UserArgs/Bias` extension internal to bitsandbytes. Investigated and
confirmed closed in session 6. Env snapshot: `docs/diagnostics/gfx1201-report-env-20260605-101605.txt`.

Do not:
- Re-run probe scripts (probes 1–3 are done; `capture-rocblas-bench.sh` / `replay-rocblas-bench.sh` confirmed dead end)
- Try `TORCH_USE_HIPBLASLT=1` (run #11 — no kernel for this shape, Tensile fallback)
- Try `BNB_FORCE_B_CONTIGUOUS=1` (run #12 — Python `.contiguous()` can't reach BLAS descriptor)
- Try `expandable_segments:True` (run #10 — silently ignored on gfx1201)
- Try `ignore_data_skip=True` (rejected — silent garbage adapter, no stall alarm)

---

## Open items

- **Upstream rocBLAS reproducer** for `ROCm/rocm-libraries` issue: a ~20-line Python
  script using `bnb.nn.Linear4bit` at exact shape (M=608, N=5376→21504 NF4 dequant).
  Not yet written. This would enable AMD to reproduce without full Gemma 4 31B training.
  See `docs/diagnostics/gfx1201-report-env-20260605-101605.txt` for full env details.

---

## Session history

**Session 4 (2026-06-02):** Crawl launched from checkpoint-90. Forward progress to
checkpoint-1230 overnight (~50 crashes, all recovered). Tokenization deadlock after
step-1230 crash wedged the GPU; monitor STALLed 8/8 at 17:19 on 2026-06-03.

**Session 5 (2026-06-04):** Monitor and training dead, GPU dirty (24 GB wedged KFD).
No code changes. Commits: env snapshot script, rocBLAS probe scripts.

**Session 6 (2026-06-05):**
- Ran `capture-rocblas-bench.sh` + `replay-rocblas-bench.sh` → confirmed `ROCBLAS_LAYER`
  dead through PyTorch's hipBLAS path; `rocblas-bench gemm_ex` only dispatches `ISA000`
- Fixed venv torch build regression: `torch+cu130` (CUDA) → `torch+rocm7.2` (`make install`)
- Env snapshot collected: `docs/diagnostics/gfx1201-report-env-20260605-101605.txt`
- Created HF model repo `ulises-c/SocratesLM-31B-stage2b-QLoRA` with README + ATTRIBUTION
- `scripts/train_sft.py`: `save_total_limit` 2→5; `HFCheckpointCallback` added (async
  HF push every 50 steps via `TRAIN_HF_REPO` + `TRAIN_HF_PUSH_EVERY` env vars)
- `Makefile`: `train-gemma4-31b-stage2-unsloth` wired with `TRAIN_HF_REPO` + `TRAIN_HF_PUSH_EVERY`
- Extracted `HFCheckpointCallback` to `src/project/hf_callback.py` (module-level, testable)
- Added `tests/test_hf_callback.py` — 7 unit tests, all passing (162 pass / 2 skip total)
- **Uncommitted as of handoff** — commit `scripts/train_sft.py`, `src/project/hf_callback.py`,
  `tests/test_hf_callback.py` before restarting training

```
e6ed01a  feat(train): HF auto-push callback + raise save_total_limit to 5
21fc7ab  feat(diag): add rocblas bench capture + replay scripts for gfx1201 repro
9016f9f  fix(monitor): crash_hint matches the real gfx1201 fault signature
d4ce31d  feat(train): enable W&B in config (next-crash pickup) + pin run for continuity
587b60e  feat(monitor): rotate TRAIN_DATA_SEED per resume to break sticky fault
```

---

## Key files

| File | Purpose |
|---|---|
| `docs/GFX1201_FAULT_ABLATION_LOG.md` | Canonical run log — append after every run |
| `scripts/monitor_stage2.sh` | Crawl harness — rotating seed, KFD cleanup, PR posts |
| `scripts/train_sft.py` | Training script; HF auto-push + `TRAIN_DATA_SEED` wired |
| `src/project/hf_callback.py` | `HFCheckpointCallback` — async HF push, skip-if-in-flight |
| `tests/test_hf_callback.py` | 7 unit tests for `HFCheckpointCallback` |
| `outputs/sft-stage2-gemma4-31b/crashlogs/` | Per-crash full log + dmesg archive |
| `docs/diagnostics/gfx1201-report-env-20260605-101605.txt` | Env snapshot for ROCm upstream report |
| PR #101 | Full diagnostic thread + all session findings |
