# gfx1201 QLoRA Backward Page Fault — Ablation & Run Tracker (PR #101)

Canonical, append-only log of every diagnostic/ablation run for the Gemma‑4‑31B QLoRA
backward page fault on the AMD R9700 (gfx1201 / RDNA4, ROCm 7.2). One row per run.
**Update this file whenever a run completes** — it is the system of record; PR comments
and wandb are the raw sources.

> Companion: `docs/GFX1201_RDNA4_TRAINING.md §6.1` holds the narrative + original ablation
> plan. Where the two disagree, **this file wins** — §6.1 was written before the fault was
> localized and still frames it as bnb‑side (now ruled out, see below).

---

## Current root‑cause state (2026‑06‑01)

The fault is a GPU page fault ("page not present", 2 MB‑aligned host‑VA `0x7f…` address)
during the **QLoRA backward pass**, specifically the **gradient‑checkpoint forward recompute**.
The faulting kernel is named and confirmed on a clean GPU:

- **Faulting kernel:** `Cijk_Ailk_Bjlk_…_MT64x64x64_…_ISA1201` — a **rocBLAS Tensile GEMM**,
  gfx1201‑native. The **same tile succeeds elsewhere** in the same backward pass → the fault
  is specific to that call's inputs/addresses, not a categorical kernel bug.
- **bitsandbytes is NOT the faulting op.** `kDequantizeBlockwise<float>` and `<hip_bfloat16>`
  succeed in every cycle; the fault is in the GEMM that consumes the dequantized weight.

**Two live hypotheses (buckets):**
- **#1 — freed/recycled operand:** the forward activation passed to the recomputed GEMM was
  released by the allocator before the kernel read it (deterministic lifetime bug, plausibly
  the `use_reentrant=False` grad‑ckpt hook path under 89–98 % VRAM). Serialization does **not**
  rule this out — it kills races, not deterministic lifetime bugs.
- **#2 — wild address:** the `MT64x64x64 ISA1201` kernel computes an out‑of‑bounds address
  from a column‑major (`stride=(1,N)`) operand (Tensile stride bug). Fits a fault address far
  from any base buffer.

`probe-3` (commit `dad8f4a`) forks #1 vs #2 by logging the forward GEMM's activation operand.

---

## Run log

Legend: **GPU=clean** means `make gpu-preflight` PASSed immediately before launch (else a
dirty‑KFD cascade can fault early and confound the result). All runs use baseline unless noted:
`TORCH_USE_HIPBLASLT=0`, `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8`,
`grad_ckpt=use_reentrant:False`, bnb 0.49.2, SDPA, seq 1280, bs 1×16, seed 42.
`SYNC` = `AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1`. wandb IDs marked `?` need confirming on the box.

| # | Run / log | wandb | Code | Start | Key vars | GPU | Fault step | Faulting kernel | What it established |
|---|---|---|---|---|---|---|---|---|---|
| 1 | historical `train.log` | various | pre‑diag | fresh@0 | baseline, lvl1 | unknown | ~84 | not named | baseline fault exists |
| 2 | eos‑gate set | `g2df2ifl` `gv9fbjac` `5xd8qt5w` `irgdklt9` | SHAs `c752a2b4`,`eb12dbd9` | fresh@0 | baseline | unknown | 10 / 84 / clears 100 | not named | **probabilistic** — same SHA both finishes & crashes |
| 3 | `gfx1201_fault_2026-06-01.log` | `h9z6ebjd` | `8580844` | fresh@0 | **SYNC**, lvl1 | unknown | 84 | not named (lvl1) | **async race RULED OUT** (serialized still faults) |
| 4 | `diag-l3-resume.log` | `akfrsb3z` | no probe (≈Review#1 cmd) | ckpt‑10 | SYNC, **lvl3** | **dirty** (no preflight) | 22 | **`MT64x64x64 ISA1201`** | named the kernel (but cascade‑confounded) |
| 5 | `clean-repro-1.log` | `001pnijf` | no probe | ckpt‑20 | SYNC, lvl3 | **clean** | 21 | `MT64x64x64 ISA1201` | fault is **real, not a cascade**; bnb dequant succeeds |
| 6 | `probe-1.log` | `i6m2e3sx`? | probe v1 `81cebb4` | ckpt | SYNC, lvl3, `BNB_DEQUANT_PROBE` | clean | ~mid | `MT64x64x64` | fault **1.5 GB from dequant buffer** → "bnb descriptor too small" **RULED OUT** |
| 7 | `probe-2.log` | ? | probe v2 `1b752c2` | ckpt | SYNC, `BNB_DEQUANT_PROBE` (+grad_output) | clean | ~mid | `MT64x64x64` | 8 dequants, no `gemm_operandA` → fault is in **forward recompute**; A‑operand not logged → bucket inconclusive |
| 8 | `probe-3.log` | _pending_ | probe v3 `dad8f4a` | ckpt | SYNC, `BNB_DEQUANT_PROBE` (+forward input) | clean | **PENDING** | — | will fork **bucket #1 vs #2** |

> Note: `dad8f4a` was authored on the R9700 and is **not yet on origin** as of this writing —
> confirm it is pushed before relying on `git pull`.

---

## Pending ablation arms — each is a candidate **fix**, judged on two axes

**Axis A:** did it *eliminate* the fault? (→ clean ~94 h run, crawl unnecessary).
**Axis B:** did it *improve steps‑per‑resume*? (→ brute‑force crawl becomes cheap).
Protocol: `make gpu-preflight` → fixed seed → run ≥150 steps (or to fault) → record both axes.
Change exactly **one** factor from baseline.

| Arm | One change | Targets | Axis A result | Axis B result | Status |
|---|---|---|---|---|---|
| **HLT1** | `TORCH_USE_HIPBLASLT=1` | #2 (route GEMM off Tensile → hipBLASLt) | | | **pending** — verify it actually changes backend for these shapes |
| **C** | grad‑ckpt `use_reentrant=True` | #1 (activation lifetime) | | | pending — one‑line, run after probe‑3 if #1 |
| **D** | `PYTORCH_HIP_ALLOC_CONF` unset / `expandable_segments:True` | placement | | | pending — **asymmetric read:** a *disappearance* can mask a kernel bug, not diagnose it |
| **E** | `HSA_ENABLE_SDMA=0` | DMA page‑fault mitigation | | | pending — cheap, stackable |
| **B** | `TRAIN_GRAD_CKPT=false` | confirms recompute drives it | | | pending — likely **OOMs** at 98 % VRAM (diagnostic, not a fix) |
| ~~A~~ | ~~bnb source build `-DBNB_ROCM_ARCH=gfx1201`~~ | ~~bnb kernel~~ | — | — | **DROPPED** — probe‑1 ruled out bnb as the faulting op |
| TUN | `PYTORCH_TUNABLEOP_ENABLED=1` | #2 (pick different GEMM) | | | speculative fallback — selects on **speed not correctness**; may re‑pick the bad kernel or crash during tuning |

---

## Ruled out — do not re‑investigate (with the run that settled it)

| Hypothesis | Status | Evidence |
|---|---|---|
| Async / concurrency race | **ruled out** | run #3 — serialized run still faults |
| Numerical / data / bad batch | ruled out | §6.1 — smooth loss, byte‑identical per‑step losses |
| LR / optimizer magnitude | ruled out | §6.1 — lr 5e‑6 crashed *earlier* than 5e‑5 |
| Sequence length / `max_length` | ruled out | §6.1 — real token max 909 < 1024; non‑binding |
| bnb NF4 backward kernel is the faulting op | **ruled out** | runs #5–7 — dequant succeeds; fault is in the Tensile GEMM. **Supersedes §6.1's "fault is in the bitsandbytes‑NF4 path" and removes Arm A.** |
| bnb hands the GEMM a too‑small descriptor | **ruled out** | run #6 — fault 1.5 GB from the dequant buffer, not adjacent to `end` |
| hipBLASLt is the **cause** | ruled out | §6.1 — `HIPBLASLT=0` still faults. **But:** =1 was **never tried as a *fix*** (see arm HLT1) — §6.1's "hipBLASLt ruled out" conflated cause with remedy |
| Dirty‑KFD cascade is the *true* fault | ruled out | run #5 — fault reproduces on a verified‑clean GPU |

---

## How to log a run (keep this file current)

After any run completes, append a row to the **Run log** (or fill an ablation arm's Axis
A/B) with: run/log filename, wandb id, commit hash (`git rev-parse --short HEAD`), resume
point, the one changed variable, GPU‑clean status, fault step, faulting kernel, and verdict.

Standard launch (adjust the one variable under test; keep `AMD_LOG_LEVEL=1` unless you need
to re‑name the kernel — level 3 writes multi‑GB logs):

```bash
make gpu-preflight                                   # MUST pass
COMMIT=$(git rev-parse --short HEAD)
CKPT=outputs/sft-stage2-gemma4-31b
nohup env TORCH_USE_HIPBLASLT=0 \
  AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 AMD_LOG_LEVEL=1 \
  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
  TRAIN_BASE_MODEL=unsloth/gemma-4-31B-it-unsloth-bnb-4bit TRAIN_PREQ=true \
  TRAIN_MAX_STEPS=95 TRAIN_OUTPUT_DIR="$CKPT" \
  uv run --no-sync python scripts/train_sft.py --config configs/train-sft-stage2-gemma4-31b.env \
  > "$CKPT/run-$COMMIT.log" 2>&1 &
```
