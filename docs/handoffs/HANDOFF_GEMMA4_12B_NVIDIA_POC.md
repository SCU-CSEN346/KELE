# Handoff — Gemma 4 12B SFT-uplift PoC (+ MTP speed test) on NVIDIA RTX 4000 Ada

**For:** the next Claude Code session, on the NVIDIA RTX 4000 Ada box.
**Branch to start from:** `feat/gemma4-12b-sft-poc-nvidia` (pushed to both
`ulises-c/csen-346` and `SCU-CSEN346/KELE`). Forked off
`feat/gfx1201-rdna4-qlora-fla-training`.
**Authoritative plan:** the scaffolding was built from a gated plan; the gate
ordering (G0–G9) is reproduced in §3 below.

---

## TL;DR — what this PoC does

Answer one question on the new, smaller **Gemma 4 12B**: **does 1-epoch Socratic
QLoRA SFT give measurable eval uplift?** Establish a baseline first (Qwen3.5-LoRA
state classifier as consultant + **base** 12B as teacher), then SFT, then re-eval
the same way and compare. Separately, A/B **llama.cpp MTP** (multi-token
prediction, PR #23398) on/off on the base teacher to measure inference speedup and
confirm quality is unaffected (it is lossless speculative decoding, so the quality
check should show ≈0 delta — the payoff is tokens/sec).

This is the NVIDIA sibling of the in-flight 31B AMD Stage-2 work. The 31B's
gfx1201/ROCm plumbing (`TORCH_USE_HIPBLASLT`, `PYTORCH_HIP_ALLOC_CONF`,
`gpu-preflight`, `patch-fla`) is intentionally **dropped** here; attention stays
`sdpa`. The host is **known-unstable** (see memory: training-host-hardware-fault),
so the run leans on frequent checkpoints + `train_sft.py` auto-resume.

---

## Project context — 5-bullet refresher

- CSEN-346 NLP project reproducing/extending **KELE** (multi-agent Socratic
  teaching; Peng et al. EMNLP 2025 Findings). A small state classifier routes
  pedagogical state to an LLM teacher.
- Eval hits a **served** OpenAI-compatible endpoint (`TEACHER_BASE_URL`), not
  in-process weights — so each model must be served (llama.cpp) before its eval.
- Metrics (`src/project/metrics.py`): ROUGE-1/2/L, BLEU-4, state-classification
  accuracy (overall + per-stage), written to `metrics_summary.json`. ROUGE/BLEU
  is diagnostic; the benchmark rewards memorisation, so weight the state-accuracy
  delta. Compare runs with `python -m src.project.evaluate --compare A B`.
- The consultant is the published **Qwen3.5-LoRA** classifier
  `ulises-c/socrates-state-classifier-qwen3.5-lora`, passed via
  `--bert-consultant <dir>` (accepts any SeqClassification checkpoint).
- Single 20 GB card: **cannot serve two models at once**, and training and serving
  compete for VRAM — do them sequentially.

---

## Decisions baked into the scaffolding

- **Eval scale:** smoke-gate → full (n=5 sanity, then full n=681 on base and SFT).
- **MTP:** base only (the drafter is base-derived → highest acceptance; the uplift
  comparison itself stays MTP-off so it isn't confounded).
- **Quant:** base served at user-chosen `gemma-4-12b-it-UD-Q8_K_XL.gguf` (13.6 GB);
  SFT served at `Q8_0` (UD is not stock-llama.cpp-producible). At Q8 the
  quant-scheme delta is ~noise, so base↔SFT differ effectively only by the adapter.

---

## What this branch already landed (don't redo)

Commits `fb82188` (scaffold) + `81686c2` (convert wrapper):

- `configs/train-sft-gemma4-12b-qlora.env` — QLoRA, 1 epoch (~4826 steps @
  eff-batch 16 over 77202 records), anchored LoRA regex (ports unchanged to the
  12B's 48 decoder layers), `socrat-zh-sft,socrat-en-sft` (inference-matching
  schema), `save_steps=50`, in-training eval off.
- `configs/gemma4-12b-local.env` / `configs/gemma4-12b-sft-local.env` — dual-role
  local-serve eval configs; distinct teacher aliases (`Gemma 4 12B` vs
  `Gemma 4 12B SFT`) so the eval fails fast against the wrong loaded weights.
- `scripts/serve_gemma4_12b.sh` — serves UD-Q8_K_XL via the generic engine
  `serve_gemma4_31b.sh` (CUDA auto-detect). `MTP=1` attaches the drafter
  (`MTP/gemma-4-12B-it-MTP-Q8_0.gguf`) with `--spec-type draft-mtp
  --spec-draft-model … --spec-draft-n-max 4` and forces f16 KV.
- `scripts/serve_gemma4_12b_sft.sh` — serves the merged SFT Q8_0 GGUF.
- `scripts/convert_gemma4_12b_sft_to_gguf.sh` — merge→GGUF with the **12B**
  NAME_TAG and auto-stage into the weights dir (no manual rename; default Q8_0).
- `Makefile` — `nvidia-preflight`, `train-gemma4-12b{,-dry-run}`,
  `serve-gemma4-12b{,-mtp,-sft}`, `eval-gemma4-12b-{base,sft}-{smoke,full}`,
  `_classifier-ckpt` (downloads the Qwen3.5 classifier on first eval).

Validated off-GPU: dry-run loads 77202/8578 records + accepts the LoRA config;
shellcheck clean; `make -n` parses all targets; venv torch is `2.11.0+cu130`
(`cuda_avail=True`).

---

## §3 — Gated run order (do these on the box)

```
G1  make nvidia-preflight                 # HARD GATE: CUDA fwd/bwd must pass
G2  Assets:
      hf download unsloth/gemma-4-12b-it-GGUF gemma-4-12b-it-UD-Q8_K_XL.gguf --local-dir ~/Documents/models/weights
      hf download unsloth/gemma-4-12b-it-GGUF MTP/gemma-4-12B-it-MTP-Q8_0.gguf --local-dir ~/Documents/models/weights
      # classifier auto-downloads on first eval (target: BERT_CKPT)
G3  make serve-gemma4-12b   (bg) ; poll: curl localhost:8080/v1/models  (alias "Gemma 4 12B")
    make eval-gemma4-12b-base-smoke         # SANITY GATE: non-degenerate state_accuracy
G4  make eval-gemma4-12b-base-full          # → results/gemma4-12b-base   (CHECKPOINT)
    [MTP] make serve-gemma4-12b-mtp ; benchmark tok/s on vs off + a smoke quality-parity eval
    stop server (free VRAM)
G5  make train-gemma4-12b                   # ~4826 steps; on crash, re-run → auto-resumes (save_steps=50)
                                            # GATE: outputs/sft-gemma4-12b-qlora/final/adapter_model.safetensors
G6  scripts/merge_lora_gemma4_sft.py --base google/gemma-4-12b-it \
      --adapter outputs/sft-gemma4-12b-qlora/final --out outputs/sft-gemma4-12b-qlora/merged
    bash scripts/convert_gemma4_12b_sft_to_gguf.sh   # → Q8_0 GGUF, auto-staged to weights dir
                                            # needs ~24 GB system RAM (BF16 staging); GPU idle
G7  make serve-gemma4-12b-sft ; make eval-gemma4-12b-sft-smoke    # SANITY GATE
G8  make eval-gemma4-12b-sft-full           # → results/gemma4-12b-sft   (CHECKPOINT)
G9  python -m src.project.evaluate --compare results/gemma4-12b-base results/gemma4-12b-sft
    # uplift = SFT − base on state_accuracy (overall + per-stage), rouge*, bleu4
```

---

## Caveats / things to verify on the box (not code bugs)

1. **MTP needs a llama.cpp build ≥ PR #23398** (merged 2026-06-07; arch
   `gemma4-assistant`) — stock builds can't load the drafter. Also: the spec-flag
   names (`--spec-type draft-mtp / --spec-draft-model / --spec-draft-n-max`) are
   PR-sourced; verify against `llama-server --help` on the actual build — older
   llama.cpp speculative convention was `-md / --model-draft / --draft-max`.
2. **VRAM at Q8 is untested.** 13.6 GB weights + the engine's inherited `-b 2048`
   compute buffer + KV (+ draft model under MTP) on 20 GB. If serve OOMs, lower
   `-c` / `-b` / `-np` (all overridable; defaults `-c 32768`).
3. **System RAM ≥ ~32 GB** for the G6 BF16 merge.
4. **MTP-on-SFT** (if you extend beyond base): the base-derived drafter sees the
   SFT'd distribution → lower acceptance / smaller speedup, still lossless.

---

## Next-next (out of scope here)

If 1 epoch underfits, bump `TRAIN_EPOCHS` (clear `outputs/sft-gemma4-12b-qlora/
checkpoint-*` first). If uplift is real, the same pattern scales to a fuller
NVIDIA training run or a DPO stage. The 31B AMD track is independent and unchanged.
