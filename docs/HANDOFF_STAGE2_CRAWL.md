# Handoff — Stage 2 Gemma 4 31B QLoRA crawl

**Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**For:** a fresh Claude instance picking up after session 7.

---

## TL;DR

Training is **NOT running**. The monitor was not restarted in session 7 because a diagnostic
SYNC run is/was occupying the GPU to collect fresh fault data for the AMD upstream bug report.

**Before restarting the monitor:**
1. Verify the diagnostic process is dead (`pgrep -f train_sft` → should be empty)
2. Parse the diagnostic log (see Open items below)
3. `make gpu-preflight` — the diagnostic run deliberately faults the GPU

Working tree is **clean**. All session 7 changes committed and pushed at `109eb72`.

Progress banked: **checkpoint-1230 / 4826 steps ≈ 25.5% complete.**

---

## Immediate actions

### 1. Check diagnostic run status

```bash
pgrep -f train_sft   # should be empty — training crashed or completed
wc -c docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log
```

### 2. Parse the diagnostic log

```bash
# Get the fault address
grep "Memory access fault" docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log

# Get fresh operand pointers at checkpoint-1230 VA layout (last probe before fault)
grep "dequant_probe.*21504" docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log | tail -5

# Confirm ShaderName dispatched
grep -o 'MT64x64x64[^[:space:]]*ISA1201[^[:space:]]*' \
  docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log | head -3
```

Expected: fault address 2 MB-aligned, ~1.6 GB above B.end; same `MT64x64x64…ISA1201…DTVB1…`
ShaderName as run #9. If the ShaderName doesn't appear in lvl3 log, the fault happened
before the kernel launched (unlikely under SYNC) — the fault message alone is still valid.

### 3. File the AMD upstream issue

All evidence is collected. File at **https://github.com/ROCm/rocm-libraries/issues**,
component: **rocBLAS / Tensile**.

Include:
- `scripts/repro_gfx1201_rocblas.py` (standalone reproducer)
- Full ShaderName from run #9 / diagnostic log
- Operand descriptors: A=(608,21504) row-major bf16, B=(21504,5376) col-major NF4→bf16
- Fault addresses (all 2 MB-aligned, ~1.6 GB above B.end — see analysis in session 7)
- Key finding: `bias=True` required (BH_Bias_UserArgs epilogue selects this tile); col-major B
  is intrinsic to PyTorch + bitsandbytes 4bit matmul — no userspace workaround possible
- Env snapshot: `docs/diagnostics/gfx1201-report-env-20260605-101605.txt`
- Note: standalone reproducer dispatches the kernel but may not crash (sparse VA — see docstring)

### 4. GPU preflight + restart monitor

```bash
make gpu-preflight   # must PASS (diagnostic run faults GPU)
pgrep -f monitor_stage2   # confirm not already running
nohup bash scripts/monitor_stage2.sh > outputs/monitor_stage2.log 2>&1 &
tail -f outputs/monitor_stage2.log
```

---

## Current state

| Item | Value |
|---|---|
| Training script | `scripts/train_sft.py` via `make train-gemma4-31b-stage2-unsloth` |
| Monitor | **NOT RUNNING** — not restarted in session 7 (diagnostic run active) |
| Monitor log | `outputs/monitor_stage2.log` |
| Train log | `outputs/sft-stage2-gemma4-31b/train.log` |
| Latest checkpoint | `checkpoint-1230` (safe — trainer_state.json intact) |
| Total steps | 4,826 (1 epoch, 77k records, batch 1×16) |
| Per-step time | ~71 s/it (async); slower under SYNC |
| GPU state | **Dirty after diagnostic** — run `make gpu-preflight` before monitor |
| HF backup repo | `ulises-c/SocratesLM-31B-stage2b-QLoRA` (auto-push every 50 steps) |
| W&B project | `csen346-sft` at `uchavarria-santa-clara-university` |

---

## Recurring venv issue — check this first every session

**`torch+rocm7.2` gets silently overwritten by `torch+cu130` (CUDA build)** after `uv sync`
or any pip/uv install outside the Makefile. This has happened in sessions 6 and 7.

Always verify at the start of any session before touching the GPU:

```bash
uv run --no-sync python -c "import torch; print(torch.version.hip)"
# Must print e.g. "7.2.26015". If it prints "None" → ROCm torch is missing.
make install-rocm   # fix: reinstalls torch==2.11.0+rocm7.2
```

---

## How the crawl works

`monitor_stage2.sh` runs a loop: launch training → poll every 5 min → on crash: archive
log + dmesg, clean GPU (waits up to 180s for KFD to drain), quarantine any partial
checkpoint, relaunch with a **fresh `TRAIN_DATA_SEED=$(date +%s)`**.

The rotating seed (commit `587b60e`) reshuffles the post-resume sample sequence each cycle,
converting the sticky-deterministic gfx1201 page fault back to probabilistic. At a fixed seed
the fault occurs deterministically within steps 22–24 from checkpoint-20; from checkpoint-1230
the window is 0–100 steps (wider, still seed-dependent).

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

**Most likely cause:** wedged GPU KFD context. Fix sequence:

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
- **Always verify `torch.version.hip` is not None** before any GPU work (regression recurs)
- **`make gpu-preflight` before any manual launch** — dirty KFD cascades into early faults
- **Do not add `AMD_SERIALIZE_KERNEL` or `HIP_LAUNCH_BLOCKING` to the monitor** — SYNC mode
  makes the fault deterministic; all SYNC runs are diagnostic only
- **Do not restart the monitor without checking if it's already running** (`pgrep -f monitor_stage2`)

---

## What is closed — do not re-investigate

The gfx1201 ISA1201 Tensile GEMM bug is **fully root-caused and documented**. All
ablation arms are exhausted. See `docs/GFX1201_FAULT_ABLATION_LOG.md`.

The `ROCBLAS_LAYER=2` bench logging path is a dead end through PyTorch: PyTorch routes
through `libhipblas.so → librocblas.so` internal dispatch, bypassing the C API logging
hooks. Both B-matrix encodings for `rocblas-bench gemm_ex` dispatch `ISA000` (MLIR
generic), never `ISA1201`. Confirmed in session 6.

The standalone reproducer (`scripts/repro_gfx1201_rocblas.py`) dispatches the faulting
kernel but does NOT crash in a small process — the wild address lands on mapped VA because
the process VA footprint is too small. This is expected and documented in the script. Do not
re-investigate or try to make it crash in isolation.

Do not:
- Re-run probe scripts (probes 1–3 done; `capture-rocblas-bench.sh` / `replay-rocblas-bench.sh` confirmed dead end)
- Try `TORCH_USE_HIPBLASLT=1` (run #11 — no kernel for this shape, Tensile fallback)
- Try `BNB_FORCE_B_CONTIGUOUS=1` (run #12 — Python `.contiguous()` can't reach BLAS descriptor)
- Try `expandable_segments:True` (run #10 — silently ignored on gfx1201)
- Try `ignore_data_skip=True` (rejected — silent garbage adapter, no stall alarm)

---

## Open items

- **Parse session 7 diagnostic log** — `docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log`
  (SYNC+lvl3+BNB_DEQUANT_PROBE from checkpoint-1230; will have crashed within 0–100 steps).
  Extract: fault address, B operand pointer, ShaderName. Confirm 2 MB-aligned fault pattern.
  Append a row to `docs/GFX1201_FAULT_ABLATION_LOG.md` as run #13 (or next available).

- **File AMD upstream issue** — all evidence collected (see Immediate actions §3 above).
  URL: https://github.com/ROCm/rocm-libraries/issues · component: rocBLAS / Tensile.

---

## Session history

**Session 4 (2026-06-02):** Crawl launched from checkpoint-90. Forward progress to
checkpoint-1230 overnight (~50 crashes, all recovered). Tokenization deadlock after
step-1230 crash wedged the GPU; monitor STALLed 8/8 at 17:19 on 2026-06-03.

**Session 5 (2026-06-04):** Monitor and training dead, GPU dirty (24 GB wedged KFD).
No code changes. Commits: env snapshot script, rocBLAS probe scripts.

**Session 6 (2026-06-05 AM):**
- Ran `capture-rocblas-bench.sh` + `replay-rocblas-bench.sh` → confirmed `ROCBLAS_LAYER`
  dead through PyTorch's hipBLAS path; `rocblas-bench gemm_ex` only dispatches `ISA000`
- Fixed venv torch build regression (first occurrence): `torch+cu130` → `torch+rocm7.2`
- Env snapshot: `docs/diagnostics/gfx1201-report-env-20260605-101605.txt`
- Created HF model repo `ulises-c/SocratesLM-31B-stage2b-QLoRA` with README + ATTRIBUTION
- `scripts/train_sft.py`: `save_total_limit` 2→5; `HFCheckpointCallback` added
- Extracted `HFCheckpointCallback` → `src/project/hf_callback.py` + 7 unit tests
- Commits: `71d6111` (refactor HFCheckpointCallback), `e6ed01a` (HF auto-push + limit)

**Session 7 (2026-06-05 PM):**
- GPU was clean at start — no driver reset needed (24 GB wedge from session 4 was gone)
- ROCm torch regression recurred: `torch+cu130` again; fixed with `make install-rocm`
- **Wild address analysis:** all 12 production crash addresses are exactly 2 MB-aligned;
  fault consistently lands ~1.6 GB above B.end regardless of ASLR
- **Standalone reproducer written:** `scripts/repro_gfx1201_rocblas.py` — `bnb.nn.Linear4bit`
  at exact fault shape (in=21504, out=5376, bias=True, nf4, bf16); does not crash in small
  process (sparse VA); confirmed behaviour expected and documented
- **Diagnostic SYNC run launched** from checkpoint-1230: AMD_SERIALIZE_KERNEL=3,
  AMD_LOG_LEVEL=3, BNB_DEQUANT_PROBE=1, TRAIN_SAVE_STEPS=99999. Log at
  `docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log`. Will fault within
  0–100 steps. Parse this log at start of next session.
- Monitor NOT restarted — diagnostic run occupied GPU at end of session
- Commit: `109eb72` (standalone reproducer + codespell *.log skip)

```
109eb72  feat(diag): standalone gfx1201 ISA1201 Tensile GEMM reproducer
71d6111  refactor(train): extract HFCheckpointCallback to module level + tests
e6ed01a  feat(train): HF auto-push callback + raise save_total_limit to 5
587b60e  feat(monitor): rotate TRAIN_DATA_SEED per resume to break sticky fault
```

---

## Key files

| File | Purpose |
|---|---|
| `docs/GFX1201_FAULT_ABLATION_LOG.md` | Canonical run log — append after every run |
| `scripts/monitor_stage2.sh` | Crawl harness — rotating seed, KFD cleanup, PR posts |
| `scripts/train_sft.py` | Training script; HF auto-push + `TRAIN_DATA_SEED` + `BNB_DEQUANT_PROBE` wired |
| `scripts/repro_gfx1201_rocblas.py` | Standalone AMD upstream reproducer (dispatches ISA1201 kernel) |
| `src/project/hf_callback.py` | `HFCheckpointCallback` — async HF push, skip-if-in-flight |
| `tests/test_hf_callback.py` | 7 unit tests for `HFCheckpointCallback` |
| `outputs/sft-stage2-gemma4-31b/crashlogs/` | Per-crash full log + dmesg archive |
| `docs/diagnostics/diag-sync-probe-step1230-20260605-124008.log` | Session 7 diagnostic — parse first |
| `docs/diagnostics/gfx1201-report-env-20260605-101605.txt` | Env snapshot for AMD upstream report |
| PR #101 | Full diagnostic thread + all session findings |
