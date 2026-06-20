# SFT Handoff — Gemma 4 12B Socratic QLoRA (NVIDIA PoC, phase 2)

**Written 2026-06-10.** For a fresh session picking up the SFT-uplift work. Phase 1
(base teacher baseline) is complete; **SFT training has not started yet.** This doc
is the single source of truth for what to run, in what order, and what to watch out
for. Cross-check against `docs/EXPERIMENT_LOG.md` (newest entries at top) and the
live eval log on GitHub issue #130.

Branch: `feat/gemma4-12b-sft-poc-nvidia`. All phase-1 work is committed and pushed
(HEAD `74ce6b6`).

---

## The one-sentence goal

Measure whether **1 epoch of Socratic QLoRA SFT on Gemma 4 12B** improves the KELE
eval over the **base** Gemma 4 12B teacher, with everything else held fixed. Uplift
= SFT − base on **state accuracy**.

## The number to beat

**State accuracy 50.30%** — the base teacher, MTP-on, full Chinese test set (n=681).
This is the canonical baseline.

| | state acc | rouge1 | rougeL | bleu4 | turns |
|---|---:|---:|---:|---:|---:|
| base, MTP **on** (`results/gemma4-12b-base-mtp`) | **50.30** | 28.69 | 21.24 | 5.28 | 3991 |
| base, MTP **off** (`results/gemma4-12b-base`) | 49.62 | 28.56 | 21.02 | 5.22 | 4033 |

Per-stage (MTP-on): a 100.0 · b 44.55 · c 32.85 · d 35.98 · e 60.21. The middle
states (b/c/d) are where there's headroom; a is already perfect, e partial.

**Run-to-run σ ≈ 0.7 pp on state accuracy** (the two rows above are identical-config
runs differing only as seeds do — they calibrate the noise floor). So an SFT uplift
is only real if it clears ~1.5 pp. Also: the convergence curve (06-09 log entry)
shows state accuracy stable within ±1 pp from **n ≈ 200–300**, so a partial SFT eval
at n≈300 is a valid early read if you want a fast signal before committing the full
~17 h run.

---

## What is FIXED between base and SFT (do not change — it's what isolates the delta)

- **Consultant = Qwen3.5-0.8B LoRA state classifier** (`results/state-clf-qwen3.5-0.8b-lora-wandb/final`,
  HF `ulises-c/socrates-state-classifier-qwen3.5-lora`), passed via `--bert-consultant`,
  **runs on CPU** (`KELE_BERT_DEVICE=cpu`) to free VRAM for the teacher. Despite the
  "bert" naming everywhere, this is the Qwen3.5 classifier, not a BERT. The teacher is
  the ONLY thing that changes between base and SFT.
- **Bare teacher prompt** — no `KELE_FEW_SHOT_TEACHER` / fewshot10. Both base and SFT
  evals run bare, so the delta is purely the adapter. (Note for leaderboard context:
  the `t4-bert-*` runs that out-rank the base on state acc also use fewshot10, so they
  confound prompting with the model. Don't compare SFT-bare against those directly.)
- **Dataset**: Chinese-only test split, n=681 (`ulises-c/SocratDataset`, the
  `load_dataset` default). The English default was reverted for this PoC.
- **MTP = ON for the SFT eval.** The base on/off A/B proved MTP is lossless at n=681
  (deltas within the 0.7 pp σ) and ~2× faster per stream, so the SFT eval uses MTP.

---

## The pipeline — five steps, in order

### Step 2 — Train (NOT YET RUN)

```
make train-gemma4-12b          # nvidia-preflight gate runs first, then nohup training
tail -f outputs/sft-gemma4-12b-qlora/train.log
```

- QLoRA r16/α32, 1 epoch over ~77k per-turn records ≈ **4826 steps** @ effective batch 16.
- Trains from `unsloth/gemma-4-12b-it` bnb-4bit base (`TRAIN_PREQ=true`, no BF16 CPU staging).
- Adapter saved to `outputs/sft-gemma4-12b-qlora/` every 50 steps; **auto-pushed to HF
  `ulises-c/SocratesLM-12B-QLoRA` every 50 steps**.
- W&B: project `csen346-sft`, run id `gemma4-12b-qlora-poc`, `resume=allow` (crash-resumes
  append to one continuous step axis). Config: `configs/train-sft-gemma4-12b-qlora.env`.
- `train_sft.py` **auto-resumes** from the latest `checkpoint-*` in the output dir. If you
  change `TRAIN_EPOCHS`, you MUST `rm -rf outputs/sft-gemma4-12b-qlora/checkpoint-*` first
  (resume assumes the old schedule).
- Dry run (no weights touched): `make train-gemma4-12b-dry-run`.

### Step 3 — Merge LoRA → HF BF16

```
uv run --no-sync python scripts/merge_lora_gemma4_sft.py \
  --base google/gemma-4-12b-it \
  --adapter outputs/sft-gemma4-12b-qlora/final \
  --out outputs/sft-gemma4-12b-qlora/merged
```

### Step 4 — Convert merged → Q8_0 GGUF (and auto-stage)

```
bash scripts/convert_gemma4_12b_sft_to_gguf.sh
```

Writes `gemma-4-12B-kele-socratic-sft-Q8_0.gguf` and copies it into the weights dir
where the serve wrapper looks — no rename needed. Q8_0 matches the base teacher's
bit budget so the quant delta is noise. The script pre-flights its llama.cpp deps
and fails loud if missing (see hazards).

### Step 5 — Eval (MTP on) + compare

```
MTP=1 make monitor-eval-gemma4-12b-sft        # → results/gemma4-12b-sft-mtp
# (the monitor owns serve+eval, auto-resumes on crash, logs to issue #130)
python -m src.project.evaluate --compare results/gemma4-12b-base results/gemma4-12b-sft-mtp
```

Sanity-gate first if you like: `make eval-gemma4-12b-sft-smoke` (n=5, no monitor).
W&B eval run lands in project `csen346-eval` named after the output dir basename.
Per-dialogue metric curves log every 10 dialogues (`WANDB_EVAL_LOG_EVERY`).

---

## Hazards & gotchas (read before launching)

1. **The box is KNOWN-UNSTABLE.** The RTX 4000 Ada (20 GB) took a power surge and
   crashes under load (see memory `training-host-hardware-fault`). Eval crawls proved
   stable at **85 W**. Training is compute-bound so higher power helps throughput —
   but that's exactly the regime that historically faulted. Expect crashes; rely on
   checkpoint-every-50 + auto-resume.

2. **There is NO training monitor for the 12B yet** — this is the main gap. `make
   train-gemma4-12b` is a bare `nohup`; on a GPU fault it dies and does NOT relaunch
   itself (train_sft.py auto-resumes only when re-invoked). `scripts/monitor_stage2.sh`
   exists but is **hardcoded to the 31B** (`outputs/sft-stage2-gemma4-31b`, calls
   `make train-gemma4-31b-stage2-unsloth`). Options for the new session:
   (a) babysit `train.log` and re-run `make train-gemma4-12b` on each crash (auto-resume
   makes this safe), or (b) adapt `monitor_stage2.sh` to 12B (parameterize OUTPUT_DIR +
   the make target + add a power-step-down search like `monitor_eval_gemma4_12b.sh`
   already has). (b) is the robust path if crashes are frequent.

3. **`nvidia-preflight` is a hard gate** before training (CUDA fwd/bwd smoke through the
   kernel paths). `make train-gemma4-12b` runs it automatically; if it fails, the GPU/
   driver is in a bad state — do not force past it.

4. **GGUF conversion depends on a built llama.cpp** at `~/Documents/models/llama.cpp`
   (`convert_hf_to_gguf.py` + `build/bin/llama-quantize`). The base teacher is already
   served via llama-server, so llama.cpp is present and built — but the memory note
   `gemma4-12b-nvidia-poc-stack` flagged it as needing a build (cmake/gcc16-vs-nvcc
   friction), so if step 4 errors on a missing binary, that's where to look. The convert
   script pre-flights both paths and prints the exact missing file.

5. **One model fits at a time** on the 20 GB card. Stop the base server before serving
   the SFT model (both bind port 8080). The SFT serve uses a distinct alias "Gemma 4 12B
   SFT" so the eval fails fast if it accidentally hits base weights.

6. **VRAM math for the SFT eval**: base MTP-on (drafter + f16 KV + `-np 4`) fit in ~16.4
   GB, so SFT MTP-on should fit identically. The SFT serve script's MTP path forces f16
   KV (same as base). Note the SFT serve script does NOT have the `GEMMA4_12B_KV` env
   passthrough the base one got (commit `d5565c7`) — irrelevant since the SFT eval uses
   MTP=1 which forces f16 anyway, but flag it if you ever want SFT MTP-off.

7. **Eval `run_config.json` now records `bert_consultant`** (fixed `74ce6b6` era) — older
   runs lack the field and misattribute the consultant to `consultant_model` ("Gemma 4
   12B"), which is wrong; the real consultant is the Qwen3.5 classifier.

---

## Where things live

- **Plan + live status**: GitHub issue **#130** — pinned comment has the run checklist
  (items 1a/1b base on/off are ✅; step 2 SFT train is the next unchecked box) and a
  per-event table the eval monitor appends to.
- **Experiment log**: `docs/EXPERIMENT_LOG.md` (06-09 baseline + convergence, 06-10 MTP
  A/B — newest at top).
- **Configs**: `configs/train-sft-gemma4-12b-qlora.env` (training),
  `configs/gemma4-12b-sft-local.env` (SFT eval), `configs/gemma4-12b-local.env` (base eval).
- **Scripts**: `scripts/train_sft.py`, `scripts/merge_lora_gemma4_sft.py`,
  `scripts/convert_gemma4_12b_sft_to_gguf.sh`, `scripts/serve_gemma4_12b_sft.sh`,
  `scripts/monitor_eval_gemma4_12b.sh` (eval crawl, handles base + sft phases).
- **W&B**: training → `csen346-sft`; eval → `csen346-eval` (org
  `uchavarria-santa-clara-university`).
- **Tests/commits**: a pre-commit hook runs ruff + pyright + codespell + shellcheck +
  the full pytest suite on every commit; expect ~9 s and keep it green. Use `uv run
  --no-sync` for any Python (torch is pinned outside uv.lock).
- **Relevant memories**: `training-host-hardware-fault`, `gemma4-12b-nvidia-poc-stack`,
  `use-uv-run-for-tests`, `coverage-non-blocking-research-code`.

## Definition of done

`results/gemma4-12b-sft-mtp/metrics_summary.json` exists with 681/681 valid dialogues,
the `--compare` output quantifies SFT − base state accuracy (real only if > ~1.5 pp),
issue #130's checklist items 2/3/compare are checked, and an `EXPERIMENT_LOG.md` entry
records the uplift with per-stage breakdown.
