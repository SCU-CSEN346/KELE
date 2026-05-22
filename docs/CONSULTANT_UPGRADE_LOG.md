# Consultant-Axis Upgrade — campaign log

**Started:** 2026-05-22 (mid-morning, PDT)
**Branch:** `mk/final-project-legs`
**Goal:** Test 4 candidate backbones to potentially upgrade the locked state-classifier consultant from `bge-small-zh-v1.5` (86.55% stage / 61.64% state) to something better. Spec in [`EXPERIMENT_TIERS.md`](EXPERIMENT_TIERS.md#-locked-next-steps-queue--consultant-axis-upgrade-2026-05-22).

This doc is the **recovery log** — if the machine crashes, a fresh Claude session can read this doc + the EXPERIMENT_TIERS queue + `git log` and resume without losing context.

## ⚡ LIVE STATE — last updated 2026-05-22 ~16:32 PDT (PM session, Layer-2 mini-tests)

**Funnel is locked** (T1-T4 results, see below). Currently running **Layer-2 mini-tests** — feeding the winning T4 classifier into the full kele.py pipeline with each of the two locked open-weight teachers (Gemma 4 31B, Qwen 35B-A3B) at n=50.

### Active right now (2026-05-22 16:32 PDT)

- **T4 + Gemma 4 31B + 10-shot @ n=50** (PID found via `pgrep -f "src.project.kele evaluate"`)
  - Started: 2026-05-22 ~15:40 PDT (after auto-CPU consultant routing was added)
  - Output dir: `results/t4-bert-gemma-fewshot10-n50/`
  - Progress at this snapshot: **12/50 dialogues complete** (24%), all ✓, ~43 dlg/hr
  - ETA: ~52 min remaining → completion ~17:24 PDT
  - **CPU consultant + GPU teacher** auto-routing engaged (T4 at 1.5 GB bf16 wouldn't fit alongside Gemma's 28 GB on a 32 GB 5090; see "CPU routing for the consultant" section below)
  - Comparison anchor: locked BERT bge-small + Gemma + 10shot n=50 = **51.06% state / 38.53 R-1**

- **T4 + Qwen 35B-A3B + 10-shot @ n=50** — pending (sequential after Gemma completes; GPU contention forbids parallel runs)
  - Launch command (when Gemma test completes):
    ```bash
    mkdir -p results/t4-bert-a3b-fewshot10-n50
    OUT_DIR=results/t4-bert-a3b-fewshot10-n50 \
    BERT_CKPT=results/state-clf-qwen3.5-0.8b-lora/final \
    LIMIT=50 \
      bash scripts/eval_bert_a3b_fewshot10_full.sh \
      > results/t4-bert-a3b-fewshot10-n50/launch.log 2>&1
    ```
  - Comparison anchor: locked BERT bge-small + A3B + 10shot n=50 = **48.19% state / 35.57 R-1**

### Recovery procedure if this session crashes

A fresh Claude session can resume by:

1. **Read this doc + `docs/EXPERIMENT_TIERS.md` + `git log -20`** to recover context. Branch is `mk/post-funnel-experiments` (cut from `main` after PR #67 merged).

2. **Check if T4+Gemma finished:** `ls results/t4-bert-gemma-fewshot10-n50/metrics_summary.json`. If file exists, the run is done — read it. If not, check `pgrep -f "src.project.kele evaluate"` to see if it's still running; if no process, re-launch with the Gemma command below.

3. **Re-launch T4+Gemma (full command):**
   ```bash
   mkdir -p results/t4-bert-gemma-fewshot10-n50
   OUT_DIR=results/t4-bert-gemma-fewshot10-n50 \
   BERT_CKPT=results/state-clf-qwen3.5-0.8b-lora/final \
   LIMIT=50 \
     bash scripts/eval_bert_gemma_fewshot10_full.sh \
     > results/t4-bert-gemma-fewshot10-n50/launch.log 2>&1
   ```
   The kele.py pipeline is crash-safe (per-dialogue JSONs in `dialogues/`); restarting picks up where it left off.

4. **Then launch T4+A3B** (command above). Two sequential ~50-min runs ≈ 1.5-2 hr total wall-clock.

5. **After both land,** produce a side-by-side comparison table (Task #18) and update both `CONSULTANT_UPGRADE_LOG.md` and the paper (specifically Section 4.x "BERT-integration full-scale result" → consider updating to T4-integration headline).

### Open queue beyond this PM session

- **Task #15 — HF Hub publish (5 funnel checkpoints).** Spec in `docs/HF_PUBLISHING_PLAN.md`. Awaits Max bringing his HF account online.
- **Task #4 / C.31 — Bilingual probe.** Stage 1: cheap eval of T4 on SocratDataset-EN test split with no retraining. ~30 min. Stage 2 (bilingual co-training) only if Stage 1 shows >10 pp drop.
- **Layer-2 at n=400** (full convergent ground-truth size). Probably wait until the n=50 mini-tests show enough lift to justify the ~5 hr full eval. Requires `--sample-seed INT` patch to kele.py first (~10 lines).

---

## (Historical) Funnel landing context — last updated 2026-05-22 ~12:50 PDT

**T1 v5 landed.** Final test state accuracy: **58.32%** vs locked bge-small 61.34% → **Δ = −3.02 pp** (T1 LOSES). All stages 0-8 pp behind, biggest gap on stage d (-8.18 pp). Disconfirms the EXPERIMENT_TIERS hypothesis ("+2-5 pp expected"). Trained model saved to `results/state-clf-qwen3-emb-0.6b-frozen/final/` (2.3 GB safetensors).

**Interpretation:** frozen Qwen3-Embedding features (596M backbone, only 34K trainable head params) cannot beat fully-fine-tuned bge-small (24M params, all trainable). Embedding-pretrained features are optimized for retrieval/similarity, not 34-way classification. Trainable parameter count appears to matter more than backbone size on this task.

**T2 landed.** Final test state accuracy: **66.66%** vs locked bge-small 61.34% → **Δ = +5.32 pp** (T2 WINS). Per-stage Pareto: a 100% (tied), b 90.67% (~tied), **c 84.87% (+7.54)**, d 74.93% (~tied), e 96.48% (+1.18). The stage-c win is the architectural breakthrough — cracks the 22-way within-stage classification that's been the persistent weakness across the entire campaign. Trained model saved to `results/state-clf-qwen3-emb-0.6b-lora/final/` (2.3 GB merged-weight safetensors). Total wall-clock: ~37 min at bs=8 with manual training loop.

**T3 landed.** Final test state accuracy: **63.73%** vs locked bge-small 61.34% → **Δ = +2.39 pp** (T3 BEATS the baseline). Per-stage: a 100% / b 90.01% / c 80.51% / d 72.07% / e 94.13%. Frozen Qwen3.5 features outperform a fully fine-tuned bge-small — meaningful result even before LoRA. Trained model saved to `results/state-clf-qwen3.5-0.8b-frozen/final/`. Total wall-clock: ~35 min at bs=32.

**T4 landed (final config: v5, bs=16 + 3 epochs + gradient_checkpointing + bf16-autocast).** Final test state accuracy: **67.57%** vs locked bge-small 61.34% → **Δ = +6.23 pp — NEW FUNNEL HEADLINE.** Per-stage: a 100% / b 90.93% / **c 85.14% (+7.81 vs baseline)** / d 75.89% / e 96.77%. T4 beats T2 across EVERY stage (overall +0.91, all per-stage deltas positive). Total wall-clock: ~50 min at the optimized config. Iterations to get there: v1 bs=4 (too slow, 4.7 hr ETA), v2 bs=16 (~73 min), v3 bs=32 (~50 min), v4 bs=48 (slower than v3 due to grad-checkpoint scaling), v5 bs=16 + bf16-autocast (50 min, final).

**Funnel pattern — final:**
- Frozen → LoRA on same backbone: +5-8 pp gain (T1 58.32 → T2 66.66 = +8.34; T3 63.73 → T4 67.57 = +3.84)
- Qwen3-Embedding → Qwen3.5 in frozen regime: +5.41 pp (T1 → T3)
- Qwen3-Embedding → Qwen3.5 in LoRA regime: +0.91 pp (T2 → T4)
- The two effects DO NOT compose linearly — Qwen3.5's frozen-regime advantage shrinks dramatically under LoRA, suggesting LoRA on the smaller backbone (T2) closes most of the gap to the larger backbone (T4). LoRA is a strong leveler.

**Headline call:** T4 (67.57%) is the new paper headline. T2 (66.66%) remains the deployment-ergonomic alternative — smaller model (596M vs 752M), no grad-checkpoint or bf16-autocast complexity required, runs on any 32 GB VRAM setup. Within statistical noise the two are tied; choose by deployment cost.

**Working launch command for T2 (use this if re-launching):**
```bash
.venv/bin/python -u scripts/train_state_classifier_34way.py \
    --model-id Qwen/Qwen3-Embedding-0.6B \
    --lora --lora-r 8 --lora-alpha 16 \
    --batch_size 8 \
    --out-dir results/state-clf-qwen3-emb-0.6b-lora \
    > results/state-clf-qwen3-emb-0.6b-lora/train.log 2>&1
```

Note both `python -u` (live stdout) and `--batch_size 8` (avoid LoRA OOM).

**ETA for T2:** at bs=8 with 5 epochs, ~22916 steps total. Per-step is faster than T1's bs=32 but ~4× more steps; LoRA backward adds ~30% overhead. Expected total ~30 min.

**VRAM peak risk:** T1 v5 hit 98.2% VRAM during the train→eval transition without OOM. T2 will hit similar or slightly higher (LoRA adapter optimizer state adds ~25 MB; negligible). If OOM occurs, re-launch with `--batch_size 16` to halve the memory footprint.

### Update 12:54 PDT — T2 needed bs=8 to fit; gradient-checkpointing flag added

T2 attempts at bs=32 and bs=16 both OOM'd in the MLP SiLU activation. The accumulated forward activations across 28 transformer layers — at fp32, batch×seq×intermediate(3072) per layer — overwhelm 32 GB on the LoRA path. Activation memory drops linearly with batch size; bs=8 brings T2 to ~22 GB VRAM with comfortable headroom.

T2 v3 is now running at bs=8 (started ~12:54 PDT). New per-epoch step count is ~4583 (4× more than T1's bs=32), but per-step wall-time is correspondingly lower; total run-time should land around 30 min.

**New flag added to the trainer:** `--gradient-checkpointing` enables PyTorch's activation-recomputation. ~30% slower wall-clock but cuts activation memory ~4×. Use this for T4 (Qwen3.5 + LoRA) which has a larger backbone and would otherwise need bs=4 or smaller.

### Update 12:51 PDT — T2 OOM'd at bs=32, relaunched at bs=16

T2 (Qwen3-Embedding + LoRA, bs=32) hit `torch.OutOfMemoryError: 158 MiB requested, 167 MiB free` during the first training step. Diagnostic forward passed cleanly (logits ok, loss 5.67), so the model is healthy — the issue is purely **backward-pass activation memory**.

**Critical memory-profile insight for LoRA-path runs (T2 and T4):**
- T1's frozen-head path: backbone activations are NOT saved for backward, because gradients stop at the head's input (the pooled hidden state). Only that one tensor needs to be retained. Memory ~7 GB peak.
- T2/T4's LoRA path: backbone activations MUST be saved for backward, because LoRA adapters are inserted INSIDE the backbone (at q/k/v/o_proj on every attention block), and gradients need to flow back through every layer to reach them. With 28 layers × batch=32 × seq=512 × hidden=1024 × fp32, this is ~30+ GB of activation memory. **OOM is expected at bs=32.**

T2 v2 launched at `bs=16` (halves activation memory to ~15 GB; total VRAM should stay around 18-20 GB). Per-step time is roughly halved (batch=16 vs 32 forward), so 2× more steps per epoch but each step is faster — total epoch wall-time should be similar.

**Pre-staged adjusted commands for T3/T4** (replacing the earlier ones):

```bash
# T3 — Qwen3.5 frozen (similar memory profile to T1, bs=32 should fit)
.venv/bin/python -u scripts/train_state_classifier_34way.py \
    --model-id Qwen/Qwen3.5-0.8B-Base \
    --freeze-backbone \
    --out-dir results/state-clf-qwen3.5-0.8b-frozen \
    > results/state-clf-qwen3.5-0.8b-frozen/train.log 2>&1

# T4 — Qwen3.5 + LoRA: use bs=4 with gradient_checkpointing (larger model than T2 + LoRA)
.venv/bin/python -u scripts/train_state_classifier_34way.py \
    --model-id Qwen/Qwen3.5-0.8B-Base \
    --lora --lora-r 8 --lora-alpha 16 \
    --batch_size 4 \
    --gradient-checkpointing \
    --out-dir results/state-clf-qwen3.5-0.8b-lora \
    > results/state-clf-qwen3.5-0.8b-lora/train.log 2>&1
```

T4 uses bs=4 + `--gradient-checkpointing` because Qwen3-Embedding (596M) OOM'd at bs=16 on the LoRA path; the larger Qwen3.5 (752M) plus the hybrid Mamba+attention layers (extra in_proj_qkv/out_proj activations) needs both smaller batch AND activation recomputation. ~30% slower wall-clock but stays well within VRAM.

**If this session has crashed and you (a fresh Claude) are reading this:**

1. **Check whether T1 v5 finished** — if `results/state-clf-qwen3-emb-0.6b-frozen/test_eval.json` exists, training is done. Read it for the final number.

2. **If test_eval.json doesn't exist, check if process 33748 (or any `train_state_classifier_34way.py` process) is still running:** `ps -ef | grep train_state | grep -v grep`. If yes, wait for it to finish (the log will be silent due to buffering — that's expected, NOT a hang). If no, the training crashed before save; re-launch fresh from the command below.

3. **For ALL future training launches, add `-u` to python** for unbuffered stdout so the log is live. Example:
   ```bash
   .venv/bin/python -u scripts/train_state_classifier_34way.py [args] > train.log 2>&1
   ```
   The `-u` is critical — without it, training appears frozen for the entire run when redirected to a file.

4. **The remaining work in the funnel** (T2-T4 commands pre-staged in §"Recovery procedure" below):
   - T2: Qwen3-Embedding-0.6B + LoRA r=8 → `results/state-clf-qwen3-emb-0.6b-lora/`
   - T3: Qwen3.5-0.8B-Base + frozen → `results/state-clf-qwen3.5-0.8b-frozen/`
   - T4: Qwen3.5-0.8B-Base + LoRA r=8 → `results/state-clf-qwen3.5-0.8b-lora/`

5. **Working tree state at this update** (run `git diff --stat` to verify):
   - `M  scripts/train_state_classifier_34way.py` (+314/-52: parameterization + NaN-init fix + manual training loop)
   - `M  docs/EXPERIMENT_TIERS.md` (+12/-12: errata for Qwen3.5 + PR #66 false claim)
   - `??  docs/CONSULTANT_UPGRADE_LOG.md` (this doc; untracked)
   - `??  results/state_classifier_v1_repro/` (bge-small baseline reproduction, validated at 61.34%)
   - `??  results/state-clf-qwen3-emb-0.6b-frozen/` (T1 v5 in flight, possibly complete by recovery time)
   - All changes uncommitted. No git operations have been performed this session.

6. **The hard-won fix is in `scripts/train_state_classifier_34way.py:_force_correct_head`.** Transformers 5.8.1 initializes Qwen3ForSequenceClassification's missing `score.weight` as all-NaN. The fix detects NaN/Inf in the head post-load and Kaiming-reinits. See §6 "Qwen3ForSequenceClassification missing-head NaN init" below for the full debugging story (a long false trail through bf16/fp16/lr/Trainer integration before reaching the actual cause).

---

---

## Where we came from (entering this session)

- **Locked headline:** BERT + Gemma 4 31B + 10-shot → 48.15% / 36.78 R-1 at n=681 (`results/bert-consultant-fewshot10-gemma-full/`).
- **Locked consultant:** `bge-small-zh-v1.5` fine-tuned 34-way at `results/state_classifier_v1/final/`. Test accuracy: 61.64% state / 86.55% stage on the full n=681 split.
- **Frontier ceiling probe (Phase 3):** Opus 4.6 + top-3 narrowly beats Gemma at n=681 (within sampling noise).
- **Sample-size convergence (2026-05-22 morning):** `n=400 random dialogues` is the new canonical sample size per `CONVERGENCE_ANALYSIS.md` — ε ≤ 2pp on all four primary metrics, ~41% compute savings vs n=681.
- The 4-test queue was locked into `EXPERIMENT_TIERS.md` earlier today (commit `36d552c`) under the heading "Locked next-steps queue — consultant-axis upgrade".

## Audit findings (things that turned out to be wrong or surprising)

### 1. PR #66 LoRA wiring is NOT on this branch
The EXPERIMENT_TIERS doc claimed "the SFT pipeline merged in PR #66 already has LoRA wiring we can reuse." False on `mk/final-project-legs`:
- PR #66 introduced `scripts/train_sft.py` (TRL `SFTTrainer` for generative fine-tuning + LoRA) on commits `c90029b` / `6110084`, but those commits live on a different branch and are not reachable from `mk/final-project-legs`.
- `peft>=0.19.1` IS in `pyproject.toml`, just unused.
- **Action taken:** wrote LoRA wiring fresh against `peft.get_peft_model(TaskType.SEQ_CLS, ...)`. Doc updated.

### 2. Qwen3.5-0.8B-Base is NOT a vanilla decoder LLM
The doc described T3/T4's backbone as "decoder LLM (0.8B params, 24 layers, 1024 hidden, March 2026 release)." Reality:
- **Architecture:** hybrid Mamba+attention multimodal base. `Qwen3_5ForConditionalGeneration` at the model level, but `Qwen3_5ForSequenceClassification` is registered.
- **Layer stack:** 24 layers total, of which 18 are `linear_attention` (SSM-style) and 6 are `full_attention`, interleaved (3 linear, 1 full, repeat).
- **Multimodal:** the on-disk checkpoint contains 12-block visual encoder; loaded as `*ForSequenceClassification` discards those weights cleanly.
- **Head bug:** `Qwen3_5ForSequenceClassification` ignores `num_labels=34` and ships a 1024→2 head. Must be replaced post-load.
- **Action taken:** doc updated, custom `_force_correct_head` helper added to trainer.

### 3. kele.py drop-in contract is exactly one line
`src/project/socratic_teaching_bert_consultant.py:65`:
```python
self.bert_model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
```
**Implication:** every T1-T4 checkpoint must satisfy this. For LoRA paths, that means **merge adapters before save** (`peft_model.merge_and_unload()`) — the on-disk artifact is then a standard HF SeqClassification checkpoint, no PEFT-aware loader needed downstream.

### 4. n=400 sampling is NOT random in the existing pipeline
`kele.py:74-100` shuffles all dialogues with `seed=42`, takes the last 10% as test, then **sorts by index** before returning. `--limit N` (line 479) does `dataset[:N]` — **first N by sorted ID**.

The convergence analysis's ε≤2pp guarantee was based on B=500 random subsamples. First-N may carry hidden bias (if SocratDataset.json is ordered by chapter). For paired T1-vs-T2 comparison on the same 400, this doesn't matter. For absolute comparison vs the locked n=681 baseline, it could shift numbers slightly.

**Decision deferred:** add `--sample-seed INT` to kele.py (~10 lines) when we actually need Layer 2 eval. Not blocking Layer 1.

### 5. Qwen3 backbones load in bf16, BERT-family in fp32
The original trainer hardcoded `fp16=torch.cuda.is_available()`. fp16 AMP's grad scaler doesn't have a bf16 unscale kernel, so Qwen3-family training crashes at the first `clip_grad_norm`:
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```
**Action taken:** detect `next(model.parameters()).dtype` after load, pick `bf16=True` for bf16 backbones (Qwen-family) and `fp16=True` for fp32 backbones (BERT-family). RTX 5090 (Blackwell) does bf16 natively, so this is the right default.

### 6. Qwen3ForSequenceClassification missing-head NaN init (T1 v1–v4 failure, TRUE root cause)

T1 collapsed FIVE times in a row, each with 0.00% state acc and `train_loss=0, head_weight=NaN`. Every hypothesis I tested (bf16 AdamW underflow, fp32-trainable cast, full-model fp32, pad_token_id propagation, fp16 AMP overflow, HF Trainer integration, lr too high) **was wrong**. The model was DOA from `from_pretrained` itself.

**Actual root cause:** transformers 5.8.1's default `from_pretrained` code path initializes `Qwen3ForSequenceClassification`'s missing `score.weight` (not in the pretrained checkpoint because the embedding model wasn't trained for classification) as **all-NaN in bf16**. Direct probe:
```python
m = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-Embedding-0.6B", num_labels=35)
# m.score.weight.min() == nan, m.score.weight.max() == nan, has_nan=True
```
The `.float()` cast preserves NaN. Every downstream symptom (loss=0, grad_norm=NaN, 0% acc, single-class argmax collapse) traces back to this. The `low_cpu_mem_usage=True` load path uses accelerate's modeling utilities which DO properly Kaiming-init missing weights — that's why early dry-load probes (which used `low_cpu_mem_usage=True`) showed clean heads while real training did not.

**Action taken:** two-part fix in `_force_correct_head`:
1. **Detect NaN/Inf in head weights** after load. If present, replace the head with a fresh `nn.Linear` of the correct shape and explicit `nn.init.kaiming_uniform_(a=sqrt(5))` initialization (matches PyTorch's default Linear init).
2. **Also handle shape mismatch** (Qwen3.5's binary→34-way bug) in the same code path.

Also added `low_cpu_mem_usage=True` to from_pretrained as the cleaner load path; the head NaN-guard catches anything that slips through.

**Verification:** 1-epoch smoke at lr=1e-3 — pre-training diagnostic shows `logits[-5.39, 7.29], loss=5.32` (close to `ln(35) ≈ 3.56` for random head). Training loss 4.64 → 1.71 across 1146 steps, grad_norms 30 → 17. End-of-epoch eval **56.35%** and test acc **56.58%** after one epoch (vs locked bge-small 61.34% at 5 epochs).

**Pre-existing comment errata fixed:** the script's `ALL_STATES` list comment said "Full 34 states" but the list has 35 items (a0, a1, b2-b7, c8-c29, d30-d33, e34 = 2+6+22+4+1=35). `len(ALL_STATES)` is and was always 35; the comment was just misleading. Both BERT and Qwen3 heads are 35-way; this is not a separate bug.

**Failure signature for future debugging:**
- `train_loss: 0` and `grad_norm: nan` reported from the very first logged step
- Test accuracy collapses to exactly 0.00% (worse than uniform random ~2.86% for 35 classes)
- All predictions argmax to a single class (NaN logits behave deterministically in argmax kernels)
- Direct probe: `model.score.weight.min().item()` returns NaN (the smoking gun)
- pad_token_id config is a SECONDARY issue worth fixing but NOT the cause of NaN

**General lessons (hard-earned):**
1. **Always run a `Trainer`-bypassing single-step forward probe before launching real training**, especially for a new model family. 30 seconds, rules out the entire NaN-cascade failure mode.
2. **The earliest signal beats the cleverest hypothesis.** I chased four sophisticated numerical-precision theories before stepping back and probing directly. The probe immediately revealed `score.weight has_nan=True` after load — answer in 5 seconds, not 4 hours.
3. **Add a pre-training NaN-detect-and-reinit guard.** Cheap belt-and-suspenders; catches transformers-version bugs without requiring root-cause investigation. Inlined in `_force_correct_head` now.

### Secondary fixes (still load-bearing, applied at the same time)

- **`model.config.pad_token_id` propagation.** Qwen3*ForSequenceClassification pools the last non-pad token; without `pad_token_id` set on `model.config`, pooling is degenerate. The original guard only fired when `tokenizer.pad_token_id` was None — Qwen tokenizers have it pre-set, so the model never received it. Fix: unconditional propagation. (Was a real issue, just not the cause of NaN.)
- **Manual training loop replaces HF Trainer.** Originally introduced when I suspected Trainer/Accelerator was casting trainable params back to bf16. Turns out that wasn't happening — Trainer was fine, the head was just NaN. Manual loop is kept anyway: simpler, fewer surprise interactions with mixed-precision, and the diagnostic visibility (printing loss/grad_norm at known intervals) is what eventually let me trace this back.
- **Pure fp32 training (no AMP).** Originally introduced to bypass mixed-precision interactions. Costs ~2× wall-clock per epoch on Qwen3-Embedding but is acceptable for the funnel. Once T1-T4 lands, worth revisiting bf16 AMP for production.

## What we built

### Refactored `scripts/train_state_classifier_34way.py` (+114 / -11)

Single in-place refactor. **All existing invocations with no args still produce the locked bge-small baseline** — old wrappers don't break.

New CLI surface:
```
--model-id <hf-id>             default: BAAI/bge-small-zh-v1.5
--out-dir <path>               default: results/state_classifier_v1
--freeze-backbone              flag; trains only the head (linear probe)
--lora                         flag; PEFT LoRA + merge_and_unload before save
--lora-r <int>                 default: 8
--lora-alpha <int>             default: 16
--lora-target-modules <auto|csv>  default: auto (per-arch defaults)
--lr <float>                   default: 2e-5 (full-FT/LoRA) or 1e-3 (frozen)
```

Per-arch LoRA target registry (validated by inspecting `named_modules()` on each backbone):
```
bert:           query, value
qwen3:          q_proj, k_proj, v_proj, o_proj
qwen3_5:        q_proj, k_proj, v_proj, o_proj, in_proj_qkv, out_proj
qwen3_5_text:   same as qwen3_5 (model_type rename at runtime)
```

Helpers added:
- `_freeze_backbone(model)` — sets `requires_grad=False` on everything except `classifier`/`score` head; refuses to run if no head is found.
- `_force_correct_head(model, num_labels)` — replaces the head if `out_features != num_labels` (fixes the Qwen3.5 binary→34-way head bug).
- `_apply_lora(model, model_type, r, alpha, targets)` — wraps via `peft.get_peft_model` with `TaskType.SEQ_CLS`, `modules_to_save=["classifier", "score"]` so the head trains alongside the adapter, returns PEFT-wrapped model.
- AMP-precision auto-select (bf16 for Qwen-family, fp16 for BERT-family).
- LoRA save path: `trainer.model.merge_and_unload()` then `.save_pretrained(out_dir/final)` → kele.py-compatible artifact.

Mutual-exclusion guard: `--freeze-backbone` and `--lora` are not allowed simultaneously.

### Doc updates: `docs/EXPERIMENT_TIERS.md`
- Removed false PR #66 LoRA-wiring claim; documented actual implementation pointer.
- Updated Qwen3.5-0.8B-Base description to reflect hybrid Mamba+attention reality.
- Updated T3 expected-outcome from "even with T1 to slightly worse" to "most genuinely uncertain of the four" given the arch revision.

## Validation runs

### Dry-load (no training, ~1 min total)
5/5 configurations passed end-to-end: load → head fixup → optional LoRA wrap → merge_and_unload → save → reload via `AutoModelForSequenceClassification.from_pretrained` (kele.py contract).

| Config | Trainable | Validated |
|---|---:|---|
| bge-small full-FT | 24.0M | ✅ |
| Qwen3-Embedding-0.6B full-FT | 595.8M | ✅ |
| Qwen3-Embedding-0.6B + LoRA r=8 | 2.3M (0.39%) | ✅ |
| Qwen3.5-0.8B-Base full-FT | 752.4M | ✅ |
| Qwen3.5-0.8B-Base + LoRA r=8 | 2.05M (0.27%) | ✅ |

### Baseline reproduction (Task #3)
`scripts/train_state_classifier_34way.py --out-dir results/state_classifier_v1_repro` (all defaults).
- Result: **61.34%** state acc (vs locked **61.64%**). Δ = −0.30 pp = ~0.4 standard errors of test-set sampling noise (SE ≈ 0.74 pp on 4304 turns).
- Per-stage: a 100.00 / b 90.93 / c 77.33 / d 75.48 / e 95.30 (vs locked 100 / 90.28 / 76.16 / 76.98 / 95.45 — all within 1.5 pp).
- Train wall-clock 155s (locked was 148s). Behavior preserved.

## Two-stage funnel (current strategy)

To avoid spending ~20 GPU-hours running expensive end-to-end kele.py evaluations for non-winning candidates:

**Layer 1 (cheap, ~30 min per test, GPU-only):**
- Train classifier candidate → trainer writes `test_eval.json` with classifier-only state acc on full n=681 split.
- Comparable apples-to-apples with the locked bge-small 61.34-61.64%.
- This is the funnel pre-filter.

**Layer 2 (expensive, ~5-13 hr per test, kele.py + Gemma):**
- Drop checkpoint into `--bert-consultant <path>`, run end-to-end teaching dialogue eval.
- Only run for the **winner** identified by Layer 1.
- Will need the deferred `--sample-seed` kele.py change when we get here.

**Total Layer 1 budget:** 4 × ~30-60 min = ~2-3 hr. Way cheaper than running all 4 end-to-end.

## Status as of this writing (2026-05-22 morning)

### Running
- **T1 (Qwen3-Embedding-0.6B frozen + linear head)** — background process, ETA ~25-35 min total. Output dir: `results/state-clf-qwen3-emb-0.6b-frozen/`. Confirmed past the bf16-AMP crash point: log shows `AMP precision: bf16=True, fp16=False (backbone dtype=torch.bfloat16)`.
- Monitor armed for `Overall state accuracy` / `train_runtime` / error markers / `T1 finished`.

### Completed this session
- Trainer parameterized and validated 5/5.
- Baseline reproduced (61.34%, behavior-preserving).
- EXPERIMENT_TIERS doc errata fixed.
- This recovery doc written.

### Pending
- T2, T3, T4 trainings (commands pre-staged below)
- Layer-1 funnel comparison (after T1-T4 land)
- Layer-2 evaluation on the winner
- Commit decision: refactor + doc are uncommitted in working tree. Awaiting Max's call on whether to commit now or after T1-T4 land.

## Recovery procedure (if this session crashes)

A fresh Claude session should be able to recover by:

1. **Read this doc first**, then `docs/EXPERIMENT_TIERS.md`'s "Locked next-steps queue" section.

2. **Check what's on disk** to determine current state:
   ```bash
   git status                                  # any uncommitted state?
   git diff --stat                             # what files are dirty?
   ls -lat results/state-clf-*/test_eval.json  # which tests finished?
   ls -lat results/state-clf-*/train.log       # which tests started but didn't finish?
   nvidia-smi                                  # is something still training?
   ```

3. **Check if T1 finished** (if you don't see a `test_eval.json` in `results/state-clf-qwen3-emb-0.6b-frozen/`, it didn't complete):
   ```bash
   tail -50 results/state-clf-qwen3-emb-0.6b-frozen/train.log
   # If it shows "Done." + a state accuracy: T1 is done; pick up from Layer-1 comparison
   # If it shows a traceback: T1 crashed; debug or re-launch
   # If it just stops mid-training: machine crashed; re-launch T1 (it's not resumable —
   # the trainer's save_total_limit=1 + epoch-only saves means partial state may be
   # incomplete; simpler to re-launch from scratch)
   ```

4. **Pre-staged launch commands** (T2-T4 are independent of each other, run sequentially after T1 lands):

   ```bash
   # T1 (re-launch if it didn't finish)
   .venv/bin/python scripts/train_state_classifier_34way.py \
       --model-id Qwen/Qwen3-Embedding-0.6B \
       --freeze-backbone \
       --out-dir results/state-clf-qwen3-emb-0.6b-frozen \
       > results/state-clf-qwen3-emb-0.6b-frozen/train.log 2>&1

   # T2 (Qwen3-Embedding + LoRA)
   .venv/bin/python scripts/train_state_classifier_34way.py \
       --model-id Qwen/Qwen3-Embedding-0.6B \
       --lora --lora-r 8 --lora-alpha 16 \
       --out-dir results/state-clf-qwen3-emb-0.6b-lora \
       > results/state-clf-qwen3-emb-0.6b-lora/train.log 2>&1

   # T3 (Qwen3.5 frozen)
   .venv/bin/python scripts/train_state_classifier_34way.py \
       --model-id Qwen/Qwen3.5-0.8B-Base \
       --freeze-backbone \
       --out-dir results/state-clf-qwen3.5-0.8b-frozen \
       > results/state-clf-qwen3.5-0.8b-frozen/train.log 2>&1

   # T4 (Qwen3.5 + LoRA)
   .venv/bin/python scripts/train_state_classifier_34way.py \
       --model-id Qwen/Qwen3.5-0.8B-Base \
       --lora --lora-r 8 --lora-alpha 16 \
       --out-dir results/state-clf-qwen3.5-0.8b-lora \
       > results/state-clf-qwen3.5-0.8b-lora/train.log 2>&1
   ```

   Run each `mkdir -p <out-dir>` first if the dir doesn't exist. Each test is ~30-60 min on a 5090.

5. **Reading results:** each test produces `<out-dir>/test_eval.json` with:
   - `test_state_accuracy` — the headline number; compare to locked **0.6164** (bge-small)
   - `test_stage_accuracy_via_state_pred` — per-stage breakdown (a/b/c/d/e)
   - `test_per_state_accuracy` — 34-way per-state breakdown (interesting for stage c)
   - `n_test_turns` — should be 4304 across all tests (full n=681 split)

6. **Layer-1 winner selection:** the candidate with the highest `test_state_accuracy` is the Layer-2 candidate. Tie-break by per-stage-c performance (the persistent weakness) or by total trainable params (smaller = cheaper).

7. **Layer 2 (if/when we get there):** add `--sample-seed INT` to `kele.py` for clean random-400 sampling, then run:
   ```bash
   bash scripts/eval_bert_gemma_fewshot10_full.sh
   # but with KELE_BERT_CKPT=results/state-clf-<winner>/final and --limit 400 --sample-seed 0
   ```
   (Existing wrapper assumes the locked bge-small checkpoint at `results/state_classifier_v1/final` — will need wrapper modification or env-var override.)

## Decision log (chronological, for traceability)

| Time (PDT) | Decision | Rationale |
|---|---|---|
| Mid-morning | Parameterize trainer in place (vs forking new file) | Cleanest history; old default-arg invocations preserved |
| Mid-morning | Smoke-load Qwen IDs before refactor | Caught Qwen3.5 hybrid-arch surprise + head shape bug pre-flight |
| Mid-morning | Keep Qwen3.5-0.8B-Base for T3/T4 (vs swap to Qwen3-0.6B-Base) | Embrace the architectural delta — hybrid Mamba vs encoder transformer is a more interesting comparison than two flavors of Qwen3 |
| Mid-morning | Add `--freeze-backbone` flag (vs dropping the frozen tier) | Spec fidelity; honors original T1/T3 design intent of "frozen + linear head" |
| Mid-morning | Auto-pick AMP dtype (fp16 vs bf16) per backbone | Fixes Qwen-family training without breaking BERT-family |
| Late-morning | Cast trainable params to fp32 post-load (DEAD-END, REVERTED) | First hypothesis for T1 v1's 0% collapse — bf16 AdamW underflow. Did not fix v2. |
| Late-morning | Cast entire model to fp32 + remove AMP (DEAD-END, KEPT) | Second hypothesis — fp16 AMP overflow. Did not fix v3-v4. Kept as defensive (cheap on the 5090). |
| Late-morning | Propagate pad_token_id unconditionally (PARTIAL FIX) | Real issue, but not the root cause of NaN. Kept. |
| Late-morning | Replace HF Trainer with manual training loop (KEPT) | Suspected Trainer-Accelerator interaction was the bug. Wasn't, but the manual loop's per-step visibility is what eventually traced the actual cause. |
| Late-morning | **NaN-detect-and-reinit head in `_force_correct_head` (TRUE FIX)** | Transformers 5.8.1's default `from_pretrained` initializes Qwen3's missing `score.weight` as all-NaN. Guard catches it before training. T1 1-epoch smoke = 56.58% test acc. |
| Mid-morning | Two-stage funnel (Layer 1 → winner → Layer 2) | Saves ~15-20 GPU-hours vs running all 4 end-to-end |
| Mid-morning | Defer `--sample-seed` kele.py change | Not needed for Layer 1; ~10 line addition deferrable to Layer 2 |
| Mid-morning | Run T2-T4 sequentially after T1, not in parallel | GPU memory contention — only one of Qwen3.5 / Qwen3-Embedding fits comfortably with full-FT |

## Cross-lingual deployment — why bilingual retraining may be unnecessary

Late-afternoon discussion (2026-05-22) flagged the language question: SocratDataset is Chinese; will our T2/T4 winners be useless on English deployment? The answer is **probably no, due to a structural property of multilingual base models**, and we should probe-then-decide rather than committing to bilingual retraining upfront.

**The mechanism (papered up in `acl_latex.tex` §2.2 "Bilingual deployment without bilingual training"):** Modern multilingual base models (Qwen3-Embedding, Qwen3.5) develop language-agnostic representations in their deeper layers as a structural consequence of next-token-prediction training. Under that objective, the cheapest internal encoding of cross-lingual training data is one where semantically equivalent content (e.g., 苹果是水果 / "an apple is a fruit") maps to nearby points in latent space, with language as an approximately separable axis. This is well-documented in the multilingual-BERT/XLM-R literature (Pires et al. 2019, Wu & Dredze 2019, Conneau et al. 2020).

**Crucially: typological similarity is NOT required.** Chinese and English are radically different at the grammar level — analytic vs inflected, topic-prominent vs subject-prominent, logographic vs alphabetic, very different word-order conventions. The cross-lingual transfer works *despite* grammatical distance because the alignment happens at the semantic level, where both languages encode the same underlying meanings. (Saying "Chinese and English have similar grammar" would be flat-out wrong; the correct framing is "Chinese and English encode the same semantic content, and multilingual models learn that semantic axis.")

**Why LoRA preserves this property:** LoRA fine-tuning adapts ~0.3% of base parameters (the q/k/v/o_proj adapters of rank 8). This is too small a perturbation to reshape the latent space — the adapter mostly steers the existing representations toward the classification task, rather than overwriting the cross-lingual structure encoded during pretraining. Full fine-tuning at 100% trainable can break this; LoRA generally doesn't.

**Empirical anchor from our own work:** the cross-lingual LLM-judge experiment in PR #67 §5 showed Opus + top-3 transferring zh→en with a -0.07 judge delta (essentially free), while SocratTeachLLM-using configurations lost 14× more. Frontier-architecture configurations DID transfer; monolingually fine-tuned 9B configurations DID NOT. T2/T4 use Qwen3-(3.5-)Embedding base, which is frontier-architecture and multilingual — strong prior that the Chinese-trained classifier will transfer.

**Action implications (queued in EXPERIMENT_TIERS C.31 as "PROBE FIRST"):**
- Stage 1 (cheap, ~30 min): evaluate the funnel winner on SocratDataset-EN test split with NO retraining. Expected: within 5-10 pp of the Chinese number. If true, ship the Chinese-trained classifier with confidence and skip Stage 2 entirely.
- Stage 2 (only if Stage 1 fails): bilingual co-training on concatenated zh+en train splits. ~1-2 h additional GPU.

The honest estimate: 60-70% likely we get free transfer and save the retraining work. 30-40% we see meaningful Chinese-side degradation and need Stage 2. Both outcomes are scientifically informative — the former validates the multilingual-representation argument; the latter quantifies the LoRA-vs-cross-lingual-preservation tradeoff.

## CPU routing for the consultant when teacher saturates GPU

Added 2026-05-22 during the first T4 Layer-2 mini-test (T4 + Gemma 4 31B + 10-shot at n=50). The locked bge-small consultant (24M params, ~95 MB in fp32) fit easily alongside any teacher on the 5090. T4 (Qwen3.5-0.8B-Base, 752M params, ~3 GB in fp32) does not.

**The OOM cascade we hit:**
1. **First attempt:** OOM at `model.to(cuda).eval()` — 3 GB fp32 model didn't fit in the ~4 GB free after Gemma loaded.
2. **Second attempt (bf16 load via `dtype=torch.bfloat16, low_cpu_mem_usage=True`):** model loaded successfully at ~1.5 GB, but every per-turn forward pass OOM'd. CUDA's allocator had ~1.6 GB free but fragmented; even 16 MB contiguous allocations failed.
3. **Third attempt (auto-CPU routing):** loaded model on CPU instead. Inference happens on CPU, teacher on GPU. Works clean.

**The routing logic** (in `src/project/socratic_teaching_bert_consultant.py`):

```
env_device = os.environ.get("KELE_BERT_DEVICE", "auto").lower()
- "cpu"  → CPU always
- "cuda" → CUDA always (force; will OOM if teacher saturated)
- "auto" (default):
    if model_size_mb > 200 AND cuda_free < 3 GB:
        CPU
    else:
        CUDA
```

The 200 MB threshold lets bge-small (95 MB) stay on CUDA even when memory is tight (frequent reuse, cheap to fit). The 3 GB free threshold catches the "teacher saturated GPU" case for larger Qwen-family classifiers.

**The cost:** ~100-300 ms per turn for CPU inference on a Qwen3.5-class model (single forward, no batch). For a 50-dialogue mini-test averaging ~6 turns each = 300 forward passes × 300 ms ≈ 90 s overhead on a ~50-min run. **~3% wall-clock penalty.** Effectively free relative to the alternative of swapping the consultant for a smaller model.

**The dtype choice (bf16 on CPU vs fp32 on CPU):**
- We're currently loading in bf16 even on CPU because that's how the auto-CPU code path arose (the bf16 load was step 2 of debugging, the CPU move was step 3, and they composed).
- bf16 on CPU is often SLOWER than fp32 because most x86 CPUs (AMD, older Intel) lack native bf16 instructions and emulate via fp32 cast → fp32 matmul → bf16 cast. AMX-BF16 (Intel Sapphire Rapids+) is the exception.
- bf16 vs fp32 accuracy delta on classification argmax: < 0.1 pp empirically; effectively indistinguishable.
- **Future-tweak:** consider loading in fp32 when device=CPU. Marginal speed win on most CPUs, marginally more faithful to training-time numerics. Not worth restarting an in-flight run for; flip the default when starting a fresh series.

**Implications for the HF Hub publishing plan (Task #15):**
- Model cards should mention this routing logic in the "How to use" section: large consultant checkpoints (T2/T3/T4) need CPU placement when paired with a multi-GB teacher.
- A user with two GPUs could place teacher on GPU 0 and consultant on GPU 1 — out of scope for our single-5090 setup, but worth flagging.

## Open questions

1. **Should we commit the refactor + doc now or wait until T1-T4 land?** Working tree has uncommitted changes to `scripts/train_state_classifier_34way.py` (+114/-11) and `docs/EXPERIMENT_TIERS.md` (+12/-12, plus this doc). Awaiting Max's call.
2. **Stratified n=400 vs random n=400 for Layer 2?** The convergence analysis notes a stratified design (stage × subject × difficulty) could halve n required for the same tolerance. Worth considering when we wire `--sample-seed`.
3. **Hierarchical head follow-up?** EXPERIMENT_TIERS notes "T1+T2+T3+T4 each paired with a hierarchical 5+22 head if any single test shows strong stage-c lift potential." Holding until Layer 1 results inform.
