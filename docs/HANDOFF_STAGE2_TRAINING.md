# Handoff — Stage 2b training + n=75 baseline expansion

**For:** the next Claude Code session, likely on the R9700 training box
**Branch to start from:** `feat/stage2-sft-pipeline-design` if PR #79 is still
open, else `main` after merge. Check with
`gh pr view 79 --json mergedAt,state,headRefName` before deciding.
**Previous handoff (historical):** `.claude/HANDOFF_STAGE2_PIPELINE.md` — the
design-PR brief. Items 1-8 of its first-action checklist are now done.
**Authoritative plan:** `docs/TRAINING_PLAN.md` — §0.1 §0.2 §3 §4 are the
sections that matter for *this* session.

---

## TL;DR — what this session does

PR #79 landed the **Stage 2 pipeline infrastructure** (format fix, loaders,
configs, DPO pair builder scaffold). This next session moves from
infrastructure to *evidence and training*. Two parallel tracks:

1. **GPU track — Stage 2b QLoRA training** on Qwen3.6-27B with the new format
   fix. Produces the first fine-tuned teacher checkpoint, which then unblocks
   DPO Source 1 in the next-next PR. ~4-6 h on the R9700.
2. **Evidence track — n=75 synthetic baseline expansion** + LLM judge on the
   three existing baselines. Without this, the post-fine-tune lift can't be
   measured against a defensible floor (n=37 Wilson SE is ~9pp per-stage).
   ~$5 API + ~2-3 h GPU eval.

These tracks can run sequentially or with the API-only synthetic generation
happening in parallel with the GPU training. Pick a sequencing in §1.

---

## Project context — 5-bullet refresher

- CSEN-346 NLP class project reproducing and extending **KELE** (multi-agent
  Socratic teaching with LLMs, Peng et al. EMNLP 2025 Findings).
- Headline contribution: 24M-param BERT consultant routes pedagogical state
  to an LLM teacher. On a memorisation-resistant metric the open-weight
  teacher (Qwen3.6-27B / Gemma 4 31B) reaches head-to-head parity with
  frontier APIs (Opus / Sonnet).
- Primary metric: `unified = 0.5 × stage_balanced + 0.5 × (judge × 10)` —
  see `docs/UNIFIED_RANKING.md`. ROUGE/BLEU is diagnostic only because the
  benchmark rewards memorisation of the training set (see
  `docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md`).
- Hardware: single 32 GB AMD R9700 (gfx1201, RDNA4). Training via QLoRA;
  inference via llama.cpp + Vulkan. Cannot train and serve simultaneously.
- Master leaderboard with 131 model variants lives in `README.md`. Significant
  prior evaluation infrastructure under `scripts/` and `src/project/`.

---

## What PR #79 landed (you don't need to redo any of this)

Commit `208816f` on branch `feat/stage2-sft-pipeline-design`:

- **`src/project/dataset.py` format fix** — state/action moved from the
  assistant target (where it was being trained as literal `[State:..] [Action:..]`
  output) into the user turn with **long inference-matching labels**
  (`苏格拉底教学顾问评估结果:` / `苏格拉底教学顾问建议的操作:`). Mirrors
  `socratic_teaching_system.py:451-452` token-for-token.
- **Six new loaders:** `socrat-synthetic` (eval-only — no `action` field in
  HF schema, so no marker injection), `socraticmath` (problem hoisted into
  system prompt — train-split is 100% non-alternating per empirical check),
  `socraticmath-sol`, `openhermes-2.5`, `ultrachat_200k`, `slimorca-dedup`.
  Stage 1 loaders use `streaming=True` + keyword+length pedagogy filter.
- **Two new configs:** `configs/train-sft-stage1-general.env` (1 epoch,
  lr=2e-5), `configs/train-sft-stage2-socratic.env` (3 epochs, lr=5e-5,
  default sources `socrat-zh,socrat-en`).
- **`scripts/build_dpo_pairs.py`** — Source 3 (5 anti-pattern perturbations)
  fully functional; Sources 1 and 2 scaffolded inert with clear stderr
  gating messages. Parquet output, stage/source stratification, 5% holdout.
- **`docs/TRAINING_PLAN.md` §0.2** — replaced two-pattern framing with
  3-way table (short / long compressed body / full inference mirror) and
  locked the decision to long-label-compressed-body.
- **Tests:** 17 → 26 dataset tests; full suite 156 passed, 2 skipped.

PR URL: https://github.com/ulises-c/csen-346/pull/79

---

## Locked decisions from PR #79 — do NOT relitigate

1. **SFT format = Pattern A placement + long inference-matching labels**
   (`苏格拉底教学顾问评估结果:` / `苏格拉底教学顾问建议的操作:`). Chinese
   markers in both `socrat-zh` and `socrat-en` — BERT consultant emits
   Chinese regardless of dialogue language.
2. **`socrat-synthetic` is eval-only.** Never include in `TRAIN_SOURCES`.
3. **Stage 1 filter = keyword + length.** $35 LLM-judge is queued for
   escalation only if 100-sample eyeball audit fails on all 3 sources.
4. **DPO Sources 1 & 2 inert** until Stage 2b checkpoint exists (Source 1)
   and per-turn STL synthetic dialogue logs are persisted (Source 2).
5. **HF dataset collection** (`huggingface.co/collections/ulises-c/socratic-teaching-datasets`)
   is referenced only in `dataset_catalog.json:3`. Documentation-only —
   nothing programmatic depends on it. If you add a new HF dataset to the
   project, also link it in the collection via the HF web UI (no API).

---

## Open work — pick a sequencing

### Option A — GPU first, evidence later (fastest path to a Stage 2b checkpoint)

1. **Stage 2b training** (4-6 h GPU):
   ```bash
   uv run python scripts/train_sft.py --config configs/train-sft-stage2-socratic.env
   ```
   First run downloads Qwen3.6-27B base (~17 GB to HF cache, NOT the branch).
   Effective batch 16, peak ~22 GB VRAM with grad checkpointing — fits with
   ~9 GB headroom on the 32 GB R9700.

2. While training runs (no GPU contention from API):
   - **LLM judge on 3 existing baselines** (~$5, ~30 min):
     ```bash
     for r in qwen27b-q4 qwen27b-q4-think-4096 gemma4-31b; do
       uv run python scripts/llm_judge_eval.py results/synthetic-baseline/$r
     done
     ```
   - **Synthetic n=75 generation** (~$5, ~30 min, no GPU):
     `scripts/generate_synthetic_socrat.py` — extend the existing 37-record
     `ulises-c/SocratDataset-SYNTHETIC` to 75. Talk to the user before
     pushing the upload to HF (HF re-upload is destructive).

3. After training: **eval the Stage 2b checkpoint on the n=75 synthetic
   set** (and the test split of socrat-en for in-domain accuracy).

### Option B — Evidence first, then training

1. n=75 synthetic generation + eval (the lift-measurement floor).
2. LLM judge on the three baselines.
3. Stage 2b training with a defensible baseline ready to compare against.

### Option C — Train Stage 1 → Stage 2b A/B

Only worthwhile if you have time for two training runs (~10-14 h GPU
combined). Stage 1 → 2b is *gated* per `TRAINING_PLAN.md` §2 — if Stage 1
doesn't lift synthetic unified by ≥1.0, it gets dropped from the headline.
Run Stage 2b alone first; do Stage 1 → 2b as the A/B comparison only after
2b-alone numbers exist.

**Recommend Option A.** Stage 2b is the headline candidate; getting that
checkpoint produces the most downstream unlock (DPO Source 1, eval comparison,
ablation table baseline). Run synthetic generation + LLM judge in parallel
slots to fill non-GPU wall clock.

---

## RDNA4 / gfx1201 — FLA Triton deadlock workaround

The Qwen3.6-27B base is `Qwen3_5ForConditionalGeneration` — the linear-attention
layers (`Qwen3NextGatedDeltaNet`) call `flash-linear-attention` Triton kernels
which deadlock at step 0 on gfx1201 / Triton 3.6.0. Root cause: Triton 3.6.0's
AMD software pipelining pass (`tritonamdgpu-pipeline`) has a use-after-free when
`num_stages >= 2` on RDNA4. Same bug class as the sageattention crash on the
same stack — upstream fix there is to force `num_stages=1` when
`torch.version.hip` is set (kijai/ComfyUI-WanVideoWrapper#2007).

**Patch script:** `scripts/patch_fla_rocm.sh` (also `make patch-fla-rocm`).
Sed-rewrites `num_stages=[2-9]` → `num_stages=1` in the installed FLA wheel,
clears `~/.triton/cache` and FLA's `__pycache__`. Idempotent. Has `--dry-run`
and `--restore` modes (the latter rolls back from `.bak` files).

**Re-run after every** `uv sync` / `make install-rocm` / `make install` —
those reinstall a fresh FLA wheel and undo the patch. This is symmetric with
the existing `_install-torch-rocm` story.

**Validation sequence on the R9700:**

```bash
# 1. Confirm the patch finds the FLA install and shows a non-zero ref count.
make patch-fla-rocm-dry-run
# Expect: "num_stages>=2 references found: <N>"  where N is positive.

# 2. Apply the patch.
make patch-fla-rocm

# 3. Bump TRAIN_MAX_SEQ_LEN in configs/train-sft-stage2-socratic.env from 512
#    back to 768 (or higher — FLA is ~5-10x more memory-efficient than the
#    PyTorch fallback per the HF Qwen3.5 docs and the gated-deltanet analysis).

# 4. Relaunch training with the existing env-var set from PR #79:
nohup env TORCH_USE_HIPBLASLT=0 PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
  uv run --no-sync python scripts/train_sft.py \
  --config configs/train-sft-stage2-socratic.env \
  > outputs/sft-stage2-socratic/train.log 2>&1 &

# 5. Watch for the FLA JIT compile pass to *clear* (rather than hang) and the
#    first training step to log a non-NaN loss. Expect compile to take a few
#    minutes — Triton recompiles all FLA kernels with num_stages=1.
tail -f outputs/sft-stage2-socratic/train.log
```

**If the patch does not unblock training** — fall back, in order:

1. Pin `triton==3.5.1` (pre-pipelining-bug; reported known-good with
   ROCm 7.2 + torch 2.9.1 by RDNA4 community).
2. Also cap `num_warps=4` in FLA (`find $FLA/ops -name '*.py' -exec sed -i -E 's/num_warps=([5-9]|[1-9][0-9]+)/num_warps=4/g' {} +` — Gemini's
   shotgun, addresses RDNA4 wave-scheduling assumptions distinct from the
   pipelining bug).
3. Stay on torch fallback at `TRAIN_MAX_SEQ_LEN=512` and squeeze VRAM
   (`LORA_RANK=8`, attention-only target modules, CPU optimizer offload) to
   try to reach 768 — only if (1) and (2) both fail.

Do NOT spoof `HSA_OVERRIDE_GFX_VERSION` or use nightly ROCm — multiple gfx1201
reports attribute hangs and driver crashes to those overrides.

---

## Pitfalls — what previous sessions learned

1. **Inference vs training VRAM:** Qwen Q4_K_M inference is ~16 GB; QLoRA
   training is ~22 GB. Can't do both at once on the 32 GB R9700.
2. **Sonnet judge + Sonnet-generated synthetic = correlation risk.** When
   the judge runs on synthetic, spot-check ~10 stratified samples with Opus
   or human (§5.4 of the revised plan).
3. **`assistant_only_loss=True` is on by default** (`train_sft.py:240`).
   Whatever lands in the assistant turn is what gets supervised. The format
   fix is the entire reason this matters; don't regress it.
4. **n=37 Wilson SE is ~3pp overall / ~9pp per-stage state-acc.** Any
   per-stage claim from n=37 is noise. n=75 halves the SE; n=100 better.
5. **Working tree pre-existing noise:** `.coverage`, `.railguard/`,
   `poetry.lock` were noise at PR #79 start and are still noise. **Do not
   commit them** — they belong in a separate small gitignore-cleanup PR.
6. **HF collection is not programmatic.** `dataset_catalog.json:3` URL is
   documentation-only; no code queries it. If you publish a new dataset,
   manually add it to the collection via the HF web UI.
7. **User has two push remotes on `origin`:** `ulises-c/csen-346` and
   `SCU-CSEN346/KELE`. `git push origin` pushes to both. The user is fine
   with this; don't force-push across remotes without asking.

---

## First-action checklist for the new session

1. `gh pr view 79 --json mergedAt,state,headRefName` — confirm whether
   PR #79 is open or merged.
2. If merged: `git checkout main && git pull`. If open:
   `git fetch && git checkout feat/stage2-sft-pipeline-design && git pull`.
3. Confirm GPU presence: `rocminfo | grep -E '(Name|gfx)'` should show
   gfx1201 / R9700. If you're on the wrong machine, stop and tell the user.
4. Confirm Python env: `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
5. Skim `docs/TRAINING_PLAN.md` §3 and §4 (DPO + Stage 2 ablation table —
   you didn't need these in the previous session but you'll need them
   to interpret training results).
6. **Talk to the user before:**
   - Starting the Stage 2b training run (~17 GB HF download + 4-6 h GPU).
   - Generating n=75 synthetic dialogues (~$5 + HF re-upload, destructive).
   - Running the LLM judge on baselines (~$5).
   - Any cost > $1 or any operation that takes > 30 min wall clock.
7. Pick a sequencing (Option A recommended) and confirm with user before
   launching long-running work.
8. Once training is launched, stream logs to a file and `tail -f` so the
   user can monitor without you blocking. Save `outputs/sft-stage2-socratic/`
   to disk *and* push intermediate checkpoint manifests if useful.

---

## What success looks like for this session

- Stage 2b checkpoint at `outputs/sft-stage2-socratic/` with training loss
  curve saved and a sample eval run on socrat-en test (~10 dialogues) to
  sanity-check the format fix transferred to the trained model.
- n=75 synthetic baseline numbers in `results/synthetic-baseline/`
  (extended or new directory), plus LLM judge JSONs alongside the existing
  three baseline configs.
- Updated `README.md` leaderboard entry for the fine-tuned model under
  the bert-consultant routing.
- A short results-summary commit on a new branch:
  `feat/stage2b-training-results` (open PR after committing).

**Out of scope (don't do):**

- DPO training (Stage 3) — needs Source 1 pairs which need a Stage 2b
  checkpoint and the judge run. Multi-step; save for the next-next PR.
- Gemma 4 31B head-to-head — Phase 2 per `TRAINING_PLAN.md` §0.1.
- Re-evaluating the closed leaderboard models — not relevant to the
  fine-tune lift measurement.

---

## References

| File | Why |
|---|---|
| `docs/TRAINING_PLAN.md` | Authoritative plan, especially §3 §4 |
| `docs/UNIFIED_RANKING.md` | Primary metric definition |
| `docs/AMD_R9700_LLAMACPP_VULKAN.md` | Inference stack on the R9700 |
| `configs/train-sft-stage2-socratic.env` | Default training config |
| `scripts/train_sft.py` | Training entry point (TRL SFTTrainer) |
| `scripts/patch_fla_rocm.sh` | RDNA4 FLA Triton workaround (see §"RDNA4 / gfx1201") |
| `scripts/llm_judge_eval.py` | 4-axis judge on existing baselines |
| `scripts/generate_synthetic_socrat.py` | n=75 generation |
| `scripts/build_dpo_pairs.py` | DPO Source 3 functional; 1/2 inert |
| `src/project/dataset.py` | All loaders (Stage 1 + Stage 2) |
| `src/project/dataset_catalog.json` | Reference metadata; HF collection URL |
| `results/synthetic-baseline/` | n=37 baseline (extend to n=75 here) |
| `.claude/HANDOFF_STAGE2_PIPELINE.md` | Historical — PR #79 brief |
