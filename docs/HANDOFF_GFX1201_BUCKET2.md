# Handoff — gfx1201 ISA1201 Tensile GEMM wild address: upstream bug report + mitigation

**Date:** 2026-06-01 · **Branch:** `feat/gfx1201-rdna4-qlora-fla-training` · **PR:** #101
**For:** a fresh Claude instance picking up from a confirmed root cause.

---

## TL;DR — where we are

The GPU page fault that has prevented Gemma 4 31B QLoRA from training on the R9700
(gfx1201/RDNA4) is **fully root-caused**. All other hypotheses have been eliminated.
The fault is a **wild address computed by an ISA1201 Tensile GEMM kernel** from its
column-major operand descriptor. This is an upstream rocBLAS/Tensile bug.

Training survives via checkpoint+resume (`save_steps=10`). The bug needs an upstream
report, and `expandable_segments:True` may mitigate it in the meantime.

---

## Root cause — Bucket #2 confirmed (probe-3, 2026-06-01)

### Evidence chain (do not re-investigate)

| Hypothesis | Ruled out by |
|---|---|
| hipBLASLt | Crashes with `TORCH_USE_HIPBLASLT=0` at step 14 |
| Async GPU race | Serialized run (`AMD_SERIALIZE_KERNEL=3`) still faults at step 84 |
| bnb dequant descriptor (wrong size/shape) | probe-1: fault addr ~1.5 GB from any dequant `[ptr,end)` |
| Freed/recycled operand — Bucket #1 | probe-3: both forward-GEMM operands logged, neither brackets fault |

### probe-3 result (decisive)

Faulting GEMM: **grad-checkpoint recomputed forward pass**.
- probe-2 revealed this (8 unpaired `[dequant_probe]` lines before fault → `MatMul4Bit.backward` hook blind to it)
- probe-3 added `[gemm_forward_input]` wrapper on `MatMul4Bit.forward` (commit `dad8f4a`)

Last pair before fault (lines 180753–180754, `probe-3.log`):
```
[gemm_forward_input] shape=(1, 608, 21504) stride=(13074432, 21504, 1) contig=True
                     ptr=0x7f655ece8000  end=0x7f65605d8000
[dequant_probe]      shape=(21504, 5376) stride=(1, 21504) contig=False
                     ptr=0x7f63f01a0000  end=0x7f63fde20000
FAULT: 0x7f6459a00000
```

- Fault is ~1.0 GB **below** A's ptr — NOT inside A
- Fault is ~1.2 GB **above** B's end — NOT inside B
- Full-log scan: **0 operands bracket `0x7f6459a00000`** across entire run
- Both operands are valid (A is row-major contiguous; B is column-major `stride=(1,21504)`)

**Verdict: Bucket #2.** The ISA1201 kernel computes a wild address from the
column-major B descriptor. Fault occurs in both forward (probe-3) and backward
(probe-1: `MT64x64x64 ISA1201`) directions.

---

## YOUR TWO TASKS

### Task 1 — Get the forward kernel ShaderName, then file the rocBLAS bug

A `AMD_LOG_LEVEL=3` run (no `BNB_DEQUANT_PROBE`) is **currently running** as PID 395310,
logging to `outputs/sft-stage2-gemma4-31b/kernel-name.log`. Wait for it to fault, then:

```bash
# Get the ShaderName of the faulting forward kernel:
grep -B2 "Memory access fault\|Memory Fault Error" outputs/sft-stage2-gemma4-31b/kernel-name.log \
  | grep -i "ShaderName\|Cijk" | tail -3

# Confirm fault address is consistent:
grep -i "Memory access fault" outputs/sft-stage2-gemma4-31b/kernel-name.log | tail -2
```

If the log is too large to grep quickly, use:
```bash
grep -niE "ShaderName|Cijk_" outputs/sft-stage2-gemma4-31b/kernel-name.log | tail -10
```

**Then file the upstream rocBLAS/Tensile bug** with:
- Kernel ShaderName (from above) — confirmed backward tile is `MT64x64x64 ISA1201`; forward tile TBD
- Both operand descriptors from probe-3 (A: `(1,608,21504)` contig row-major; B: `(21504,5376)` column-major `stride=(1,21504)`)
- Fault address: `0x7f6459a00000`
- Architecture: gfx1201 (RDNA4/ISA1201), ROCm 7.2
- Reproducible with `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1` (serialized → not a race)
- File at: https://github.com/ROCm/rocm-libraries/issues — component **rocBLAS** (`projects/rocblas/`) / **Tensile** (`shared/tensile/`). The standalone `ROCm/rocBLAS` and `ROCm/Tensile` repos are **retired**; rocm-libraries is the source of truth.

### Task 2 — `expandable_segments:True` mitigation test

After the kernel-name run completes (GPU clears), run:

```bash
make gpu-preflight    # must PASS

CKPT=outputs/sft-stage2-gemma4-31b
nohup env TORCH_USE_HIPBLASLT=0 \
  AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=1 \
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit TRAIN_PREQ=true \
  TRAIN_MAX_STEPS=95 TRAIN_OUTPUT_DIR="$CKPT" \
  uv run --no-sync python scripts/train_sft.py --config configs/train-sft-stage2-gemma4-31b.env \
  > "$CKPT/expandable-seg.log" 2>&1 &
```

**Asymmetric reading (important):**
- **Fault persists / moves step** → `expandable_segments` doesn't help; confirms it's a kernel stride bug, not placement
- **Fault vanishes** → NOT a diagnosis — could be masking a real kernel bug by changing memory layout. Do NOT declare victory. Replicate (N=2–3) and note it's a mitigation, not a fix.

After this run, post findings to PR #101 as an operator findings comment.

---

## Environment

- HW: AMD Radeon AI PRO R9700, gfx1201 (RDNA4), 32 GB VRAM; Ryzen 9 5900X
- SW: ROCm 7.2, torch 2.11.0+rocm7.2, bitsandbytes 0.49.2, transformers 5.9.0, peft 0.19.1, trl 1.4.0, Python 3.12
- Model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` (pre-quantized NF4, ~19 GB, cached on box)
- **Always `uv run --no-sync`** — bare `uv run` reinstalls CUDA torch over ROCm torch
- After ANY fault: `make gpu-preflight` before next launch (dirty KFD cascades)
- Training config: `configs/train-sft-stage2-gemma4-31b.env`, `save_steps=10` (survival crawl)

## Key files

| File | Purpose |
|---|---|
| `scripts/train_sft.py` | Training script; `BNB_DEQUANT_PROBE` block at line ~547 (remove when done) |
| `outputs/sft-stage2-gemma4-31b/probe-3.log` | Decisive probe-3 run; ~180k lines |
| `outputs/sft-stage2-gemma4-31b/probe-1.log` | First dequant probe (7.4 GB, backward fault) |
| `docs/GFX1201_RDNA4_TRAINING.md` | Full evidence record and ablation matrix |
| PR #101 comments | Full diagnostic thread including Consultant Reviews #1–#4 and probe findings |

## Commit history (this branch, diagnostic phase)

```
dad8f4a  feat(diag): BNB_DEQUANT_PROBE extends to MatMul4Bit.forward (probe-3)
1b752c2  feat(diag): BNB_DEQUANT_PROBE also logs grad_output (GEMM A operand)
81cebb4  feat(diag): BNB_DEQUANT_PROBE — log NF4 dequant output descriptors
8580844  diag(gfx1201): Phase 0 serialized-kernel run — fault at step 84, race ruled out
```
