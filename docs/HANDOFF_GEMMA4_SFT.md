# Handoff — Gemma 4 31B Stage 2b SFT launch

**For:** the next Claude Code session on the R9700 box  
**Branch:** `feat/stage2-sft-pipeline-design` (PR #79 still open)  
**Previous handoff:** `docs/HANDOFF_STAGE2_TRAINING.md` — Stage 2b overview + Qwen3.6-27B RDNA4 FLA investigation  
**Authoritative plan:** `docs/TRAINING_PLAN.md`

---

## TL;DR — what this session did

Pivoted to **Gemma 4 31B** as the primary SFT target on the R9700 (Track A of the two-headed strategy from the last PR comment). Qwen3.6-27B is blocked on gfx1201 FLA Triton GPU page faults at runtime even with `num_stages=1` + `num_warps=4` + `triton==3.5.1` — see the Session 4 summary in PR #79 comments.

**Work done this session:**

1. Added `download-gemma4-31b` Makefile target + help entry.
2. Fixed dry-run crash: `SFTConfig(bf16=True)` validates GPU at construction — added a `bf16=False` override scoped to the dry-run path in `train_sft.py`.
3. Added `low_cpu_mem_usage=True` to `from_pretrained` in `train_sft.py` — see §"RAM OOM during loading" below.
4. Downloaded `google/gemma-4-31b-it` (~60 GB) to HF cache:
   `~/.cache/huggingface/hub/models--google--gemma-4-31b-it/`
5. Confirmed `torch+rocm7.2` sees the R9700 (`torch.cuda.is_available() = True`, `GPU: AMD Radeon AI PRO R9700`).

**Current state:** uncommitted changes in `train_sft.py` and `Makefile`. Training was killed after stalling at shard 815/1188 due to RAM exhaustion. Fix is in place (`low_cpu_mem_usage=True`). Swap was still draining at handoff time — wait for it to clear before relaunching.

---

## First actions for the new session

```bash
# 1. Confirm branch and pull latest
git status
git pull

# 2. Check swap is clear before launching (must be < 1 GB used)
free -h
# If Swap used > 1 GB, wait. If a stale training process is running, kill it:
# ps aux | grep train_sft | grep -v grep
# kill <PID>

# 3. Confirm ROCm torch (should show 2.11.0+rocm7.2, CUDA True, R9700)
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# If it shows +cu130 / CUDA False → ROCm torch got overwritten by uv sync.
# Fix: make _install-torch-rocm

# 4. Commit the pending changes from this session, then launch training
git add scripts/train_sft.py Makefile
git commit -m "fix(train): low_cpu_mem_usage=True + dry-run bf16 override + download-gemma4-31b target"

# 5. Launch
make train-gemma4-31b-stage2
tail -f outputs/sft-stage2-gemma4-31b/train.log
```

---

## RAM OOM during loading — what happened and the fix

**Root cause:** `AutoModelForCausalLM.from_pretrained` reads each BF16 safetensors shard
(1188 shards × ~52 MB ≈ 62 GB total) into CPU RAM before quantising to 4-bit and
moving to VRAM. Python's allocator held onto freed blocks rather than returning them
to the OS, causing peak RSS to grow to ~56 GB on a 62 GB system. By shard 815 the
process was swapping at 1 shard/minute instead of ~20/second.

**Fix applied:** `low_cpu_mem_usage=True` in the `from_pretrained` call
(`scripts/train_sft.py` line ~140). This flag instructs HF to process and discard
each shard before loading the next, keeping peak RAM near the size of one shard
(~52 MB) rather than the full model.

**Expected loading time after fix:** ~2–3 minutes at 20+ shards/sec (same as the
first ~570 shards before RAM filled up in this session). Total VRAM after load:
~18–20 GB (4-bit NF4 Gemma 4 31B + LoRA overhead). ~12 GB headroom on the 32 GB R9700.

**If loading stalls again:**
- Check `free -h` — if RAM > 55 GB used, something else is holding memory.
- Check `rocm-smi --showmeminfo vram` — if VRAM is barely growing, loading
  is swapping rather than quantising.
- Kill, wait for swap to drain (`Swap used < 1 GB`), then relaunch.

---

## Why Gemma 4 31B (not Qwen3.6-27B)

| Issue | Qwen3.6-27B | Gemma 4 31B |
|---|---|---|
| Architecture | Hybrid: 48 `linear_attention` (gated delta-rule) + 16 full | Standard softmax attention (local + global sliding window) |
| FLA dependency | Required for seq > 512 | None |
| gfx1201 status | GPU page fault after all Triton patches applied | No Triton custom kernels — SDPA backend stable |
| Leaderboard rank | — | #1 local model (unified=70.08, state acc=50.72%, judge=8.17) |

Qwen3.6-27B configs (`train-sft-qwen36-27b-qlora.env`, `train-sft-stage2-socratic.env`)
are untouched and ready for an NVIDIA machine (CUDA Triton is unaffected by the gfx1201
FLA bugs). If a 5090/3090 is available, that is still Track B.

---

## Locked decisions (carry forward from PR #79 — do not relitigate)

1. **SFT format = Pattern A + long inference-matching labels** (`苏格拉底教学顾问评估结果:` /
   `苏格拉底教学顾问建议的操作:`). Chinese markers in both `socrat-zh` and `socrat-en`.
2. **`socrat-synthetic` is eval-only.** Never in `TRAIN_SOURCES`.
3. **DPO Sources 1 & 2 inert** until Stage 2b checkpoint exists.
4. **`assistant_only_loss=True`** is set in `SFTConfig` — the format fix is
   precisely what guarantees the assistant turn is clean teacher output.
5. **Do NOT run `patch-fla-rocm` before Gemma 4 31B training.** That patch
   targets FLA/Qwen only; it is irrelevant and harmless but misleading.

---

## Training config summary

```
Base model:    google/gemma-4-31b-it  (already in HF cache)
Config file:   configs/train-sft-stage2-gemma4-31b.env
Output dir:    outputs/sft-stage2-gemma4-31b/
Method:        QLoRA 4-bit NF4, r=16, alpha=32
Targets:       q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
Epochs:        3  |  lr=5e-5  |  batch 1×16=16  |  seq=1280
Sources:       socrat-zh, socrat-en  (12,244 train / 1,362 eval)
VRAM estimate: ~21–23 GB peak (~9–11 GB headroom on 32 GB R9700)
Launch env:    TORCH_USE_HIPBLASLT=0  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8
```

Full launch command (mirrors `make train-gemma4-31b-stage2`):
```bash
mkdir -p outputs/sft-stage2-gemma4-31b
nohup env TORCH_USE_HIPBLASLT=0 PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
  uv run --no-sync python scripts/train_sft.py \
  --config configs/train-sft-stage2-gemma4-31b.env \
  > outputs/sft-stage2-gemma4-31b/train.log 2>&1 &
tail -f outputs/sft-stage2-gemma4-31b/train.log
```

---

## What to watch for in the training log

**Healthy startup sequence:**
```
Loading training data  sources=['socrat-zh', 'socrat-en']
  train: 12244 records  eval: 1362 records
Loading tokenizer  model=google/gemma-4-31b-it
  QLoRA: 4-bit NF4, double-quant, bf16 compute
Loading weights: 100%|██████████| 1188/1188 [02:xx<00:00, ...]
  Dropped 'vision_tower' vision encoder to free VRAM
{'loss': <finite number>, 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
```

**Red flags:**
- Loading stalls mid-bar and `free -h` shows Swap > 2 GB → RAM issue, kill and check.
- `loss: nan` at step 1 → bf16 instability; try adding `TORCH_USE_HIPBLASLT=0` (already set) or reducing lr.
- `HIPBLAS_STATUS_INVALID_VALUE` → already suppressed by `TORCH_USE_HIPBLASLT=0`; if it still appears, the env var wasn't picked up — verify with `uv run --no-sync`.
- Process exits at step 0 with Triton hang → should NOT happen (Gemma 4 has no FLA); if it does, check whether `uv sync` reverted ROCm torch.

---

## After training completes

1. **Sanity-check the checkpoint:**
   ```bash
   uv run --no-sync python scripts/train_sft.py \
     --config configs/train-sft-stage2-gemma4-31b.env --dry-run
   # Verify output dir has adapter_model.safetensors and adapter_config.json
   ls outputs/sft-stage2-gemma4-31b/checkpoint-*/
   ```

2. **Post-SFT eval** — the existing `make eval-gemma4-31b-full` uses the GGUF/llama.cpp
   path (pre-SFT baseline). A separate eval target for the LoRA adapter is not yet
   written. Options:
   - Merge the LoRA adapter into the base model and GGUF-quantise it for llama.cpp.
   - Run eval directly via `transformers` + PEFT with the adapter loaded, bypassing
     llama.cpp. This is the path of least resistance but requires a new eval script.
   - Talk to the user about which path before writing anything.

3. **DPO Source 1** becomes unblocked once the Stage 2b checkpoint exists.
   `scripts/build_dpo_pairs.py` Source 1 scaffold is already wired; it needs the
   checkpoint path passed as `STAGE2B_CHECKPOINT`.

---

## Uncommitted changes at handoff

These were in the working tree when this handoff was written. Commit them first:

| File | Change |
|---|---|
| `scripts/train_sft.py` | `low_cpu_mem_usage=True` in `from_pretrained`; `bf16=False` override in `dry_run()` |
| `Makefile` | `download-gemma4-31b` target + `.PHONY` entry + help text |

---

## References

| File | Why |
|---|---|
| `configs/train-sft-stage2-gemma4-31b.env` | Active training config |
| `configs/train-sft-gemma4-31b-qlora.env` | Base hyperparameter reference (same as stage2 env) |
| `scripts/train_sft.py` | Training entry point — `low_cpu_mem_usage` fix is here |
| `docs/TRAINING_PLAN.md` | Authoritative plan §4 (Stage 2 ablation table) |
| `docs/HANDOFF_STAGE2_TRAINING.md` | Previous handoff — RDNA4 FLA investigation history |
| PR #79 comments | Full RDNA4 FLA debugging thread + two-headed strategy rationale |
