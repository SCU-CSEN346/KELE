# Handoff — R9700 gfx1201 EOS Gate Run

**Branch:** `feat/gfx1201-rdna4-qlora-fla-training`  
**Date:** 2026-05-31  
**Last commit:** `860fc5f`

---

## What was attempted

100-step QLoRA EOS gate checkpoint on the R9700 (gfx1201, 32 GB) to verify that
the schema-drift fixes from PR #101 / PR #94 actually produce a model that
terminates on realistic inference prompts before the full Stage 2 run is launched.

Model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`  
Config: `configs/train-sft-stage2-gemma4-31b.env`  
Data: `socrat-zh-sft` + `socrat-en-sft` (77 202 train / 8 578 eval)

---

## Environment issues encountered and fixed

### 1 — `make test-vllm` poisoned the venv

Running `make test-vllm` before training auto-installed `vllm==0.22.0`, which
pulled in `torchvision==0.26.0` (CUDA build). The CUDA torchvision registers a
fake `torchvision::nms` op at import time that doesn't exist in the ROCm torch,
breaking `peft` and `transformers` at startup:

```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'BloomPreTrainedModel'
```

**Fix:** `uv pip uninstall vllm torchvision`

**Going forward:** Never run `make test-vllm` before or during a training session
on this machine. The test auto-installs into the shared training venv and is
destructive. Either isolate it or add a guard to the Makefile.

### 2 — `expandable_segments:True` unsupported on HIP → GPU page fault at step 16

All training Makefile targets had:
```
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,expandable_segments:True
```

HIP does not support `expandable_segments`. The runtime warned at startup
(`HIPAllocatorConfig.h:40`) and the training crashed with a GPU page fault
every time at step 16:

```
Memory access fault by GPU node-1 on address 0x7f78cbc00000.
Reason: Page not present or supervisor privilege.
```

**Fix (commit 860fc5f):** Stripped `expandable_segments:True` from all Makefile
targets. Note: the PR #101 comment that suggested this setting used
`PYTORCH_CUDA_ALLOC_CONF` (CUDA variable) — on ROCm the variable is
`PYTORCH_HIP_ALLOC_CONF` and `expandable_segments` is not a valid key.

### 3 — GPU KFD dirty state after each page fault

Every page fault on gfx1201 leaves the KFD in a dirty state; subsequent runs
fault at random steps until all Python processes are killed. Always run before
retrying:

```bash
rocm-smi --showpids | grep python && echo "GPU dirty — kill before retrying" || echo "GPU clean"
```

### 4 — `nohup` → `setsid` for clean process group management

Changed the `train-gemma4-31b-eos-gate` target to use `setsid` so the entire
process tree lives in its own process group. Kill cleanly with:
```bash
kill -9 -- -$PGID
```

### 5 — `eos_gate.py` crashed: `apply_chat_template` returns `BatchEncoding` in transformers 5.9

`tokenizer.apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`
dict in transformers 5.9, not a raw tensor. `model.generate` received the dict
and raised `AttributeError` on `.shape[0]`.

**Fix:** Extract `enc["input_ids"]` when the return value is not a tensor.

---

## Training run outcome

After all environment fixes, the 100-step checkpoint trained successfully:

- **Steps:** 100/100 ✓
- **VRAM:** alloc=21.1 GB, reserved=29.1 GB / 31.9 GB total (stable throughout)
- **Step time:** ~70 s/step
- **hipBLASLt:** enabled (`TORCH_USE_HIPBLASLT=1`) — Gemma 4 uses standard
  softmax attention; no linear-attn delta-rule kernel, so hipBLASLt is safe
- **Adapter saved:** `outputs/eos-gate-gemma4-31b/final`

---

## EOS gate result: FAILED

```
STEP 1  warm-up (trivial prompt)    → PASS   (terminates after 1 sentence)
STEP 2  EOS gate (realistic prompt) → FAILED (hits max_new_tokens, repetition loop)
```

### Failure output (truncated)

```
很好！你能想象一下，当光从空气进入水中时，它的速度会发生什么变化呢？
这对光的方向有什么影响？这对光的方向有什么影响？这对光的方向有什么
影响？[...×80+ repetitions, never emits EOS]
```

The first sentence is correct Socratic output. Repetition begins at token 2.
This is the same failure mode as the 5090 run documented in PR #94.

### Root cause (as diagnosed in PR #94 / PR #101)

**Train/serve schema mismatch.** The training data uses structured consultant
annotations (`学生处于 {state} 状态`). The EOS gate — and real inference — uses
live free-form consultant prose. The model learned to depend on the structural
markers for termination; without them it enters a repetition loop.

The PR #101 / PR #94 fixes corrected the *action/eval drift* (the teacher turn
format at training time now matches inference), but the *consultant evaluation
field* is still the irreducible residual: training stores discretized state labels,
inference sends multi-sentence prose from the live consultant.

### This is expected behavior for a 100-step gate checkpoint

100 steps is ~0.13% of one epoch over the 77K training records. The gate is not
meant to produce a working model — it is meant to catch whether the
format/schema is correct enough that termination is at least possible. A gate
FAIL at 100 steps could mean:

1. **Insufficient training** — the model hasn't seen enough data to generalize
   termination to free-form prompts (most likely at 100 steps)
2. **Residual schema drift** — the consultant evaluation field still differs
   enough between training and inference to destabilize generation

---

## What to do next

### Option A — Launch the full Stage 2 run

If the assessment is that 100 steps is just too few and the schema fixes are
correct, launch the full run:

```bash
make train-gemma4-31b-stage2-unsloth
```

After the full run, re-run the EOS gate:
```bash
make eos-gate-gemma4-31b   # swap adapter path to outputs/sft-stage2-gemma4-31b/final
```

The full run is ~hours on the R9700. Monitor with:
```bash
tail -f outputs/sft-stage2-gemma4-31b/train.log
```

### Option B — Investigate consultant field drift first

Audit what the training records actually put in the consultant evaluation slot vs
what `_capture_prompt` in `eos_gate.py` sends. If the training data has
structured `学生处于 b2 状态` but the inference path always sends 128-char prose,
that's the residual drift to fix before the full run.

Quick diagnostic: grep a few training records for the evaluation field content:
```python
from src.project.dataset import load_socrat_zh_sft
records = load_socrat_zh_sft()
# inspect records[0]["messages"][1]["content"] for the consultant eval format
```

### Option C — Report the negative result (per PR #94 §2.8 outcome matrix)

The PR #94 notes already anticipate this outcome:
> "We sit somewhere in STATUS_REPORT §2.8 outcome matrix's 'fine-tuned Gemma
> loses both' cell. Negative result is still a paper contribution: documents
> that Pattern-A SFT on this size dataset caused output collapse."

---

## Files changed this session

| File | Change |
|------|--------|
| `Makefile` | Strip `expandable_segments:True`; add EOS gate targets; `setsid`; `TORCH_USE_HIPBLASLT=1` |
| `scripts/train_sft.py` | Add `VRAMLogCallback` (logs alloc/reserved/total at each logging step) |
| `scripts/eos_gate.py` | Fix `apply_chat_template` → extract `input_ids` from `BatchEncoding` |
| `docs/HANDOFF_SFT_SCHEMA_DRIFT_FIX.md` | Pre-existing handoff from prior session |

---

## Key commands

```bash
# Check GPU state before any run
rocm-smi --showpids | grep python || echo "GPU clean"

# GPU stack smoke test (all 13 steps should pass)
make test-gpu-stack

# EOS gate full sequence
make train-gemma4-31b-eos-gate        # ~115 min (100 steps × 70 s/step)
tail -f outputs/eos-gate-gemma4-31b/train.log
make eos-gate-gemma4-31b              # ~10 min (loads model + generates)

# Full Stage 2 run (only after EOS gate PASS)
make train-gemma4-31b-stage2-unsloth
```
