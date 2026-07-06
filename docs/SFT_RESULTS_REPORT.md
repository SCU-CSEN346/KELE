# Socratic QLoRA SFT vs Base — Results Report

**Gemma 4 12B-it, NVIDIA PoC.** Branch `feat/gemma4-12b-sft-poc-nvidia`. Updated 2026-07-06.
Companion to `docs/SFT_HANDOFF.md` (pipeline/provenance) and `docs/SFT_VS_BASE_ANALYSIS_PLAN.md`
(follow-on ablations). Live tracker: GitHub issue #130.

## TL;DR

Socratic QLoRA SFT on Gemma 4 12B **improves state accuracy on every axis tested** — both
languages, in-distribution and out-of-distribution — with **no regressions**. The gain is large on
held-out in-distribution data (**+7.7 to +10.3 pp**) and smaller but consistently positive on
never-seen synthetic data (**+3.5 to +3.9 pp**), the healthy shape that shows the model learned a
**transferable Socratic skill, not corpus memorization**. Achieved with a **~0.88-epoch** adapter
recovered after a NaN divergence — i.e. a lower bound on a clean full run.

## Headline: state accuracy (the metric of record)

| eval set | distribution | base | SFT | **Δ (pp)** | n |
|---|---|---|---|---|---|
| ZH test | in-distribution (held-out 10%) | 49.62 | **59.93** | **+10.31** | 681 |
| EN test | in-distribution (held-out 10%) | 51.00 | **58.68** | **+7.68** | 681 |
| ZH synthetic | OOD (never trained) | 27.19 | **31.09** | **+3.90** | 75 |
| EN synthetic | OOD (never trained) | 27.44 | **30.97** | **+3.53** | 75 |

Run-to-run σ ≈ 0.7 pp (decoding is stochastic), so the held-out gains are ~11–15σ — unambiguous.
The MTP-on ZH base scores 50.30 (within σ of the 49.62 MTP-off base used above); SFT clears either
by a wide margin. Every run: 681/681 (or 75/75) valid, **0 errors**.

## Per-stage breakdown (state accuracy by Socratic stage a–e)

| set | a | b | c | d | e |
|---|---|---|---|---|---|
| ZH test base → SFT | 100 → 100 | 43.4 → **57.0** | 30.6 → **42.9** | 36.7 → **46.2** | 61.9 → **78.1** |
| EN test base → SFT | 100 → 100 | 48.8 → **54.5** | 33.0 → **40.9** | 37.9 → **47.2** | 57.1 → **75.4** |
| ZH synth base → SFT | 100 → 100 | 17.1 → 20.4 | 5.3 → 4.5 | 0.0 → 1.5 | 29.2 → **49.3** |
| EN synth base → SFT | 100 → 100 | 30.7 → 28.0 | 4.6 → 6.9 | 0.0 → 0.0 | 18.9 → **38.8** |

- Stage **a** is trivially 100 everywhere (the opening turn).
- The largest, most consistent gain is **stage e** (closing/summary) — +16 to +20 pp on the
  in-distribution sets and the only clear OOD mover. The SFT most improved *how the teacher closes*.
- On OOD synthetic, mid-dialogue stages **c/d nearly collapse for both** models (c≈5, d≈0) — those
  synthetic dialogues are structurally hard / off-distribution in the middle; the SFT's OOD edge
  comes from stages **b** and **e**.

## Text-overlap metrics (vs ground-truth teacher turns)

SFT roughly doubles–triples ROUGE/BLEU on every set, confirming it learned the teacher's phrasing,
not just state labels:

| set | ROUGE-1 | ROUGE-L | BLEU-4 |
|---|---|---|---|
| ZH test base → SFT | 28.6 → **48.1** | 21.0 → **40.9** | 5.2 → **20.1** |
| EN test base → SFT | 51.3 → **69.5** | 34.4 → **46.8** | 3.2 → **11.7** |
| ZH synth base → SFT | 24.3 → **33.7** | 17.7 → **26.5** | 3.3 → **8.3** |
| EN synth base → SFT | 46.4 → **62.6** | 30.5 → **37.2** | 0.9 → **1.7** |

## What this means

1. **Real, large in-distribution improvement** — +10.3 pp ZH / +7.7 pp EN on held-out test, far
   past the ~1.5 pp significance bar.
2. **Generalizes, not memorizes** — the SFT still wins +3.5–3.9 pp on synthetic dialogues it has
   *never* seen (different origin, phrasing, question pool). A pure memorizer would show ~0 OOD gain.
3. **Cross-lingual transfer** — EN (+7.7) tracks ZH (+10.3); the Socratic behavior survives the
   language boundary (the SFT trained on both ZH and EN per-turn data).
4. **Behavioral signal** — the base model is ~6× slower to evaluate (≈38 vs ≈225 dlg/hr; e.g. ZH
   base 17.4 h vs SFT 3.0 h) because it **rambles to the 2048-token cap** instead of producing
   short, terminating Socratic turns. The SFT terminates cleanly — itself evidence of better form
   (to be quantified; see analysis plan T0.1).

## Consultant ablation (T1.1): does the advantage survive removing the classifier?

Every result above gave **both** base and SFT the same external Qwen state classifier, so those
numbers don't isolate the SFT's *own* contribution. This ablation reruns the ZH-test 2×2 with the
external classifier **removed** — the served LLM self-consults (dual-role: it produces the state
assessment itself, then consumes it), the apples-to-apples "no external classifier" baseline. All
other settings pinned (Q8_0, `-np 4`, q4_0 KV, MTP off, 8 rounds). Both arms 681/681, 0 errors
(base self-consult ran ~7 days across auto-resumed crashes on the unstable box).

**The two skills decouple cleanly.** Teacher-turn *quality* is intrinsic to the SFT and survives;
*state-tracking* was entirely the external classifier and does not.

### State accuracy (overall %) — the classifier's job

| | Qwen classifier | self-consult | Δ (self − Qwen) |
|---|---:|---:|---:|
| base | 49.62 | 34.45 | −15.17 |
| SFT | 59.93 | **26.80** | **−33.13** |
| **SFT − base** | **+10.31** | **−7.65** | *advantage inverts* |

### Teacher-turn quality (SFT − base gap, Qwen → self-consult) — the SFT's real deliverable

| metric | base Qwen→self | SFT Qwen→self | **SFT − base gap** |
|---|---:|---:|---:|
| ROUGE-1 | 28.56 → 28.02 | 48.13 → 44.21 | **+19.57 → +16.19** |
| ROUGE-L | 21.02 → 20.35 | 40.94 → 37.62 | **+19.92 → +17.27** |
| BLEU-4  | 5.22 → 4.80   | 20.12 → 18.31 | **+14.90 → +13.51** |

### What each contrast says

- **SFT(self) vs SFT(Qwen)** — text metrics barely move (ROUGE-1 −3.9, BLEU −1.8), but state
  accuracy **collapses** 59.93 → 26.80. The SFT's Socratic *writing* is classifier-independent; its
  scored *state* was the classifier's.
- **SFT(self) vs base(self)** *(the key comparison — no external help on either side)* — the SFT
  still writes far better turns: **+16.2 ROUGE-1, +17.3 ROUGE-L, +13.5 BLEU-4**. But it self-tracks
  state **worse** than base (26.80 vs 34.45), losing on every hard stage (b 24.4 vs 41.7, c 5.9 vs
  16.5, e 27.1 vs 36.0).
- **Interaction** — the SFT's text advantage *holds* (shrinks only ~3 pp of a ~20 pp gap) while its
  state advantage *inverts* (+10.3 → −7.65).

### Interpretation

Consistent with the SFT's training format (`dataset.py:608–647`): the SFT was trained to **consume**
the consultant's assessment + action and emit a clean teacher turn — it **never learned to emit
state**. So `state_accuracy` in self-consult mode measures a skill the SFT never trained, and it
appears to have *regressed below base's zero-shot classification* (it learned to expect state to be
handed to it). The base model, never specialized, is the better self-classifier.

**Headline answer:** the SFT genuinely internalized how to *write* Socratic turns — that advantage
survives removing the external classifier almost entirely (+16 ROUGE-1, +13.5 BLEU-4 with zero
external help). It did **not** internalize how to *track* state — that gain was the external
classifier's, and self-tracking is worse than base. In deployment the SFT still wants the external
classifier (or an equivalent state source) for the state label; what it brings on its own is the
teacher-turn quality. Results: `results/gemma4-12b-{base,sft}-noconsult/`.

## Method (held fixed; only model + dataset vary)

- **Teacher** = the only variable under test: base `unsloth/gemma-4-12b-it` vs the merged Socratic
  SFT, both served as **Q8_0 GGUF** on llama.cpp (`-np 4`, q4_0 KV, **MTP off**, workers=4).
- **Consultant** = Qwen3.5-0.8B LoRA state classifier on CPU (same checkpoint for every run).
- 8 teaching rounds, no fewshot, no thinking budget, stochastic server-default sampling (identical
  across runs). Eval replays ground-truth student turns; the teacher generates; the classifier
  scores state. 90/10 train/test split, seed 42, dialogues kept whole.
- **Datasets** (KELE-v2 collection): `SocratDataset` (ZH) / `SocratDataset-EN` held-out **test**
  splits; `SocratDataset-SYNTHETIC` / `-EN` (75 each) run **whole** as OOD probes (never in training).

## Caveats

- **Adapter is ~0.88 epoch**, `checkpoint-4250`, recovered from HF history after a NaN divergence
  at step ~4260 (loss had plateaued since ~step 3000). Report as ~0.88 epoch; a clean full run is
  a plausible further gain (see `SFT_HANDOFF.md`).
- **Synthetic n is small** (37 ZH + 38 EN merged → 75 each; ~215/431 turns). Directional OOD
  signal, not σ-tight. State accuracy is per-turn, so steadier than the dialogue count implies.
- **Stochastic decoding** — no temperature/seed pinned, so each run carries ~0.7 pp noise. Greedy
  and multi-seed runs (analysis plan T2.1/T2.2) would tighten the point estimates.
- **Consultant-free control DONE** (see the consultant-ablation section above) — the headline
  state-accuracy numbers rely on the shared external classifier and do **not** reflect the SFT's own
  state-tracking (self-consult drops SFT to 26.8, below base). The SFT's *teacher-turn quality*,
  however, is classifier-independent (+16 ROUGE-1 self-consult vs base). Read the headline table as
  "SFT + classifier vs base + classifier," not "SFT alone."

## Artifacts

- **Models (HF, private):** adapter `ulises-c/SocratesLM-12B-QLoRA` (ckpts 3200–4250); merged BF16
  `ulises-c/SocratesLM-12B`; Q8_0 GGUF `ulises-c/SocratesLM-12B-GGUF`.
- **Datasets:** `ulises-c/SocratDataset{,-EN,-SYNTHETIC,-SYNTHETIC-EN}` (synthetic ZH completed to
  75 this PR).
- **Results:** `results/gemma4-12b-{base,sft}{,-en,-synth-zh,-synth-en}/` + `-base-mtp`.

## What this PR changed (code)

- GGUF convert: CPU-only `llama-quantize` via `QUANTIZE` override (the CUDA build segfaults `nvcc`
  on this box).
- Eval: `--hf-repo`/`--split` on `evaluate` + monitor `EVAL_HF_REPO/EVAL_SPLIT/EVAL_OUT_SUFFIX`
  (no schema adapter needed — all KELE-v2 sets share the `{student,teacher,state}` turn keys).
- Data: merged the 38-record ZH-synthetic extension into HF (37 → 75); fixed the loader that
  referenced the never-uploaded config.
- Monitor: log the GPU's **actual enforced** power, not the card max (the step-down is inert
  without passwordless sudo).
- Docs: eval-plan rationale + workers A/B, this report, and the further-analysis/ablation plan.

## Reproduce

```
# serve the model under test (base or SFT), then:
KELE_PARALLEL_WORKERS=4 EVAL_HF_REPO=<repo> EVAL_SPLIT=<test|all> EVAL_OUT_SUFFIX=<tag> \
  make monitor-eval-gemma4-12b-{base,sft}
python -m src.project.evaluate --compare results/<base-run> results/<sft-run>
```

## Next

Deeper analysis is scoped in `docs/SFT_VS_BASE_ANALYSIS_PLAN.md`. No-consultant ablation (T1.1) is
**done** (above): the SFT internalized teacher-turn *quality*, not *state-tracking*. Remaining top
picks: **LLM-judge on Socratic quality** (the text-quality win begs an absolute-quality read, not
just overlap-vs-reference), **oracle-consultant run** (feed ground-truth state → isolates
teacher-turn quality with the classifier confound removed entirely), and **multi-seed error bars**.
Also worth a **strong-consultant** cell (Claude as classifier) to bound how much a better state
source lifts each teacher.
