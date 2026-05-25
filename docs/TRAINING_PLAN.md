# Training Plan — SocratTeachLLM v2 (revised 2026-05-24)

**CSEN 346 · Santa Clara University**

Goal: fine-tune a Socratic-teaching model that generalises beyond SocratTeachLLM's
training data, using a 3-stage pipeline: **scoped** general SFT → Socratic SFT →
DPO. The contamination-control baseline is established on novel data —
see `results/synthetic-baseline/` (`SocratDataset-SYNTHETIC`, n=37).

## Revision notes (2026-05-24)

This revision incorporates evidence from the synthetic-baseline review. Four
load-bearing changes:

1. **Teacher: Qwen3.6-27B selected as Stage 2 base on velocity grounds** — the
   n=37 baseline has overlapping Wilson CIs and does not support an
   evidence-based winner. Gemma 4 31B fine-tune planned as a Phase 2
   head-to-head study (§0.1).
2. **SFT training format must match BERT-consultant inference format** — the
   current `dataset.py` format trains the model to *emit* state/action; the
   BERT-consultant integration *supplies* them in the user turn (§0.2).
3. **Primary success metric switches from ROUGE-1 to unified score** — `unified
   = 0.5 × stage_balanced + 0.5 × (judge × 10)`. Targeting ROUGE re-installs the
   memorisation-rewarding failure mode this project exists to detect (§4.1).
4. **DPO data construction reuses existing project assets** — judge-mined pairs,
   SocratTeachLLM-as-rejected on synthetic, and programmatic perturbations of
   ground-truth, drawing on the hand-crafted anti-patterns already validated as
   +0.29 composite in the prompt-engineering tournament (§3.3).

Additional: every ablation is evaluated *paired with the BERT consultant* — the
project's headline architecture — not standalone (§4.3).

---

## 0. Decisions and open items before Stage 2

### 0.1 Teacher base — Qwen3.6-27B selected (velocity-driven)

Three candidates were measured on `SocratDataset-SYNTHETIC` (n=37). Wilson 95% CIs
overlap by ~7pp:

| Config | state_acc | n_turns | 95% CI |
|---|---:|---:|---|
| Qwen3.6-27B Q4 no-think | 23.44% | 209 | [18.1, 29.7] |
| Qwen3.6-27B Q4 think-4096 | 26.92% | 208 | [21.3, 33.4] |
| Gemma 4 31B Q5_K_XL | 27.23% | 213 | [21.6, 33.6] |

The data does not support an evidence-based winner. Qwen-think vs Gemma is
0.31pp (pure noise); Qwen no-think vs Qwen-think is 3.48pp (~1 SE). Per-stage
swings (e.g. stage-e +18pp between Qwen think modes) sit on n≈30 denominators
where SE ≈ 9pp — also indistinguishable. The earlier tournament's ROUGE-1 win
for Qwen is the memorisation-driven metric this project's thesis discounts,
so it is not a valid tie-breaker either.

**Decision: Qwen3.6-27B-Instruct is the Stage 2 base.** Honest framing — this
is a velocity choice, not an evidence choice. Three reasons it's defensible
under the no-evidence-based-winner constraint:

1. **Smaller inference footprint** — ~16 GB Q4_K_M vs ~22 GB Gemma Q5_K_XL.
   Faster iteration on the 32 GB R9700 leaves more headroom for longer-context
   training data and parallel inference jobs during eval.
2. **Tournament continuity** — Qwen3.6-27B has been the teacher across the
   tournament, fewshot10, and BERT-consultant cells. Fine-tune delta is
   directly interpretable against those baselines without a model-change
   confound.
3. **Cheaper contingency** — if Qwen Stage 2b underperforms expectations,
   training a Gemma 4 31B QLoRA with the same data + hyperparameters is the
   fallback (and a planned Phase 2 study regardless). Doing Qwen first costs
   less per iteration if either model ends up being the answer.

#### Planned Phase 2 — Gemma 4 31B head-to-head fine-tune

After Qwen Stage 2b lands and is evaluated paired with the BERT consultant, run
the same SFT recipe on Gemma 4 31B — same datasets, same hyperparameters, same
DPO pair sources — and compare the two fine-tunes on test + synthetic under the
same unified-score metric. This isolates the base-model contribution and gives
the paper a clean two-model comparison instead of a one-model claim. Not a
blocker for Phase 1.

#### Deferred evidence (queued, not blocking)

Two evidence-gathering steps would either ratify the velocity choice
retroactively or motivate prioritising the Gemma head-to-head. **Both should
land before §5.3 ablation eval, not before Stage 2 training launch:**

- **Synthetic-extension to n=75 (≥100 stretch)** on the Qwen3.6-27B baseline,
  and on Gemma if the head-to-head proceeds. n=37 Wilson SE is ~3pp on overall
  state-acc and ~9pp per-stage — too loose to measure the Stage 2b
  contamination tax (`test_unified − synth_unified`) with confidence. n=75
  halves the SE; n=100 better still. **Required for the lift measurement to be
  defensible** — without it the post-fine-tune comparison sits on the same
  noise floor as the baseline.
- **LLM judge on the three baseline runs** (`scripts/llm_judge_eval.py`,
  ~$1.50 × 3 ≈ $5). Places the baselines on the unified-score metric defined
  in `docs/UNIFIED_RANKING.md`, so the Stage 2b lift can be reported on the
  project's primary metric rather than stage_balanced alone. Not started in
  this PR — scheduled as a parallel evidence task during Stage 2 training.

Both are evidence-gathering for the *comparison*, not the *decision*. Qwen is
locked in and Stage 2 training can launch without either.

### 0.2 SFT data format must match BERT-consultant inference

The current `src/project/dataset.py:103, 167` format prepends
`[State: X] [Action: Y]\n` to the **assistant content** — i.e. trains the model
to *emit* state/action as the first tokens of every reply:

```python
teacher_content = f"[State: {state}] [Action: {action}]\n" + teacher_content
messages.append({"role": "assistant", "content": teacher_content})
```

But the BERT-consultant integration feeds state/action into the **user turn**
(`src/project/socratic_teaching_system.py:445-453`):

```
历史对话记录: {formatted_history}
当前学生输入: {student_input}
苏格拉底教学顾问评估结果: {evaluation}
苏格拉底教学顾问建议的操作: {action}
```

…and there is no parser anywhere that strips the bracketed prefix off the
teacher output. Training as-is means the fine-tuned model will (a) emit literal
`[State: c12] [Action: …]` strings into user-facing responses, and (b) never
learn to condition on the consultant's text.

**Resolution.** Move state/action from the assistant target into the user input,
mirroring inference. Two equally acceptable patterns:

**Pattern A — Concise (recommended for first pass).** Append the state ID and
action to the student utterance in each user turn; the assistant target is the
clean teacher response:

```python
user_content = turn["student"]
if state and action:
    user_content += f"\n\n顾问评估: 学生处于 {state} 状态\n顾问操作: {action}"
messages.append({"role": "user", "content": user_content})
messages.append({"role": "assistant", "content": turn["teacher"]})
```

**Pattern B — Verbatim inference mirror.** Reconstruct the full Chinese
narrative from `socratic_teaching_bert_consultant.py:204-209` at training time
(dropping the classifier-confidence numbers since training has none). More
faithful but larger token cost.

Pick Pattern A for the first Stage 2 run. Verify with
`uv run python scripts/train_sft.py --config … --dry-run` printing a sample
record before launching.

---

## 1. Base model

| Field | Value | Notes |
|---|---|---|
| HF model ID | `Qwen/Qwen3.6-27B-Instruct` | Stage 2 base — see §0.1 |
| Planned head-to-head | `google/gemma-4-31b-it` | Phase 2 — same SFT recipe (§0.1) |
| GGUF (inference) | `unsloth/Qwen3.6-27B-GGUF` Q4_K_M | Gemma Q5_K_XL for Phase 2 |
| Training method | **QLoRA** (4-bit NF4 base + bf16 adapters) | Identical for 27B and 31B |
| Training VRAM | ~22-26 GB peak on R9700 (32 GB) with gradient checkpointing |
| Inference VRAM | ~16 GB Qwen Q4_K_M / ~22 GB Gemma Q5_K_XL (Phase 2) |

Same fine-tuning pipeline applies to both candidates. QLoRA hyperparameters
are identical for 27B and 31B; the Phase 2 head-to-head reuses every config in
this plan with only `TRAIN_BASE_MODEL` swapped.

---

## 2. Datasets

### Stage 1 — Scoped general SFT (not the firehose)

Stage 1 is **kept but scoped**. The dual risk: catastrophic forgetting from too
much instruction-following retraining, vs. the contamination-prone "straight to
Socratic SFT on an Instruct base" path SocratTeachLLM took (proven memorisation
artefact in `docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md`).

Target ~30-50k records of *pedagogy-adjacent* skill — long-form explanation,
hint generation, multi-step reasoning — not a 1.4M general-instruction firehose:

| Dataset | HF ID | Subsample | Filter |
|---|---|---:|---|
| OpenHermes-2.5 | `teknium/OpenHermes-2.5` | ~15k | long-form explanation, step-by-step reasoning |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` | ~10k | multi-turn pedagogy (Q&A, tutoring patterns) |
| SlimOrca-Dedup | `Open-Orca/slimorca-deduped-cleaned-corrected` | ~10k | math/science reasoning chains |

Loaders TBD in `src/project/dataset.py`. Each loader writes to the same
`messages` format as existing sources.

**Mandatory A/B before committing Stage 1**: run synthetic eval after Stage 2b
alone, and after Stage 1 → Stage 2b. If Stage 1 doesn't lift synthetic unified
score by ≥1.0, drop Stage 1 from the locked headline (keep the ablation row).

### Stage 2 — Socratic SFT

| Dataset | HF ID | Records | SocRule annotations | Language | Role |
|---|---|---:|---|---|---|
| SocratDataset (original) | `ulises-c/SocratDataset` | 6,803 dialogues | ✅ state, action, evaluation | Chinese | Structural anchor |
| SocratDataset-EN | `ulises-c/SocratDataset-EN` | 6,803 dialogues | ✅ state, action, evaluation | English | Structural anchor (English) |
| SocraTeach_Multi | `ulises-c/SocraTeach_Multi` | ~30K+ dialogues | ❌ | English | Socratic breadth — math |
| SocraTeach_Single | `ulises-c/SocraTeach_Single` | 20,845 exchanges | ❌ | English | Persona diversity |
| SocraticMATH | `ulises-c/SocraticMATH` | 6,846 conversations | ❌ | Chinese | Math Socratic dialogues |

Currently implemented sources in `dataset.py`: `socrat-zh`, `socrat-en`,
`socrateach-multi`, `socrateach-single`. SocraticMATH and `socrat-synthetic`
loaders to be added.

### Stage 3 — DPO preference pairs (three sources, no API regeneration cost)

The previous plan's "think-disabled + direct-answer prompt" approach is
**dropped**: n=37 evidence shows the think-mode toggle is ~1 SE on state
accuracy, so think-off ≠ pedagogically-worse. Conflating reasoning depth with
teaching quality is exactly the confusion Stage 3 must avoid.

Instead, three sources of *hard* negatives built from existing project assets:

**Source 1 — Judge-mined pairs (highest signal).** After Stage 2b checkpoint
exists:

- For each SocratDataset-EN train dialogue, replay through the checkpoint with
  K=3 candidates per turn (different sampling seeds, temperature 0.7).
- Score each candidate + ground-truth with `scripts/llm_judge_eval.py` (4-axis
  Sonnet rubric).
- Pair `(ground_truth, candidate)` where `judge(gt) − judge(cand) ≥ 2.0` and
  they're not lexically identical.
- Expected yield: ~6,000 dialogues × ~6 turns × ~2 valid pairs ≈ 70k pairs.

**Source 2 — SocratTeachLLM-as-rejected on synthetic.** STL scored stage_bal
32.86 on the clean synthetic probe (master leaderboard cell #31) vs. Gemma 31B at
56.13. STL responses on synthetic dialogues are memorisation-pattern hard
negatives *by construction* — the exact failure mode Stage 3 must suppress.

- Pair `(synthetic_ground_truth_teacher_turn, STL_response_on_same_turn)` per
  synthetic dialogue.
- Expected yield: ~200 pairs from n=37 baseline; scales linearly with synthetic
  expansion (n=100 ⇒ ~600 pairs).

**Source 3 — Programmatic perturbations.** The hand-crafted anti-patterns at
`src/project/tournament_utilizations.py:244-261` were validated as +0.29
composite when used as in-context negatives — they are empirically the right
catalogue. Apply each mechanically to ground-truth teacher turns:

| Perturbation | Mechanism | Provenance |
|---|---|---|
| Preamble bloat | Prepend `好的，让我想想…` to teacher turn | tournament_utilizations.py |
| Multi-question | Append a second `？` clause | "" |
| Off-topic explainer | Insert tangential paragraph before the question | "" |
| Direct-answer reveal | At b/c-stage, append `答案是: {final_answer}` | derived from SocRule |
| Stage skip | At b-stage, output what's expected at e-stage (final summary) | derived from SocRule |

Expected yield: 5 perturbations × 6,000 dialogues × ~6 turns ≈ 180k pairs;
≈ 100k after deduplication.

**Total: ~150-200k preference pairs, zero API regeneration cost.**

Filter pipeline (`scripts/build_dpo_pairs.py`, to be written):

1. Apply LLM-judge filter (`judge(preferred) − judge(rejected) ≥ 2`) to all
   sources — drops weak pairs uniformly across sources.
2. Stratify by stage to balance a/b/c/d/e (in-distribution counts are heavily
   b/c-weighted).
3. Hold out 5% as a DPO-eval set never seen during training.

---

## 3. VRAM budget — QLoRA on 27B/31B (32 GB R9700)

| Component | Estimate |
|---|---|
| Base model (4-bit NF4) | ~14 GB (27B) / ~17 GB (31B) |
| LoRA adapters (bf16, r=16) | ~0.3 GB |
| Activations (batch=1, seq=2048) | ~4-6 GB |
| Optimizer states (AdamW on adapters only) | ~1-2 GB |
| **Total peak** | **~20-23 GB (27B) / ~23-26 GB (31B)** |

Gradient checkpointing (`TRAIN_GRAD_CKPT=true`) is required for either candidate.
At batch=1 + grad_accum=16 the effective batch is 16, matching the paper.

`configs/train-sft-qwen25-7b-lora.env` is preserved as a lighter fallback.

---

## 4. Three-stage pipeline

### Stage 1 — Scoped general SFT

**Goal:** conversational fluency on pedagogy-adjacent inputs before Socratic
shaping. **Decision gated by A/B**: if Stage 1 → 2b doesn't lift synthetic
unified score by ≥1.0, drop it.

**Config:** `configs/train-sft-stage1-general.env` (to be created)

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs | 1 | Avoid catastrophic forgetting |
| LR | 2e-5 | Subtle shaping |
| Batch | 1 × 16 grad accum | Effective 16 |
| Max seq len | 2048 | Validate against sample distribution |
| LoRA rank | 16 | |
| Grad ckpt | true | |

### Stage 2 — Socratic SFT

**Goal:** state-conditioned generation — the model learns to consume the
consultant's `(state, action)` signal supplied in the user turn (see §0.2) and
produce a stage-appropriate teacher response.

**Sub-phase 2a — Breadth pre-training:** `socrateach-multi` + `socrateach-single`
(+ `socraticmath` once integrated). Plain Socratic dialogue format, no state
conditioning. 1-2 epochs.

**Sub-phase 2b — Structural fine-tuning:** 2a checkpoint → `socrat-zh` +
`socrat-en` with state/action conditioning supplied in the user turn. 3 epochs,
matching the paper.

**Config:** `configs/train-sft-stage2-socratic.env` (to be created)

| Hyperparameter | Value |
|---|---|
| Epochs (2a) | 1-2 |
| Epochs (2b) | 3 |
| LR | 5e-5 |
| Batch | 1 × 16 grad accum |
| Max seq len | 2048 |
| LoRA rank | 16 |
| Grad ckpt | true |

#### Training data format (state-conditioned, after §0.2 fix)

```
system:    "You are a Socratic teacher...\nProblem: {question}\nOptions: {options}\nHint: {hint}"
user:      "{student_turn_1}\n\n顾问评估: 学生处于 {state_1} 状态\n顾问操作: {action_1}"
assistant: "{teacher_turn_1}"
user:      "{student_turn_2}\n\n顾问评估: 学生处于 {state_2} 状态\n顾问操作: {action_2}"
assistant: "{teacher_turn_2}"
```

At inference the BERT consultant supplies `state` and `action` via the
user turn; at training, ground-truth values from SocratDataset annotations are
used. The assistant target contains **only** the teacher response — no
`[State: …]` prefix to strip.

#### Alternative — weighted mixed training

If two-phase shows no improvement over 2b-alone:

| Source | Weight |
|---|---|
| socrat-zh | 3× |
| socrat-en | 3× |
| socrateach-multi | 1× |
| socrateach-single | 1× |
| socraticmath | 1× |

### Stage 3 — DPO preference optimisation

**Goal:** teach the model *when* to hint, explain, challenge, or reveal — the
adaptive-scaffolding behaviour the in-context negative exemplars proved
(+0.29 composite) is teachable as preference signal.

**Data:** ~150-200k pairs from the three sources in §2 Stage 3.

**Script:** `scripts/train_dpo.py` (to be written — mirrors `train_sft.py`,
uses TRL `DPOTrainer`).

**Config:** `configs/train-dpo-qwen36-27b.env` (to be created — name updates
if Gemma is chosen).

| Hyperparameter | Value | Notes |
|---|---|---|
| Beta | 0.1 | KL penalty weight; standard DPO starting point |
| Epochs | 1-2 | DPO converges fast; more risks mode collapse |
| LR | 1e-5 | Lower than SFT |
| Max seq len | 2048 | |
| Reference model | Stage 2b checkpoint | Frozen |

---

## 5. Evaluation plan

### 5.1 Targets — unified score, not ROUGE

Primary metric: **`unified = 0.5 × stage_balanced + 0.5 × (judge × 10)`**
(`docs/UNIFIED_RANKING.md`).

| Run | unified target | Rationale |
|---|---:|---|
| Stage 2b on synthetic | **TBD** | Lift over best baseline; baseline unified pending §0.1 judge run |
| Stage 2b on test | **≥ 68.65** | Match the current locked headline (`bert × Gemma-31B · fewshot10 · n=681`) |
| **Generalisation criterion** | `(test_unified − synth_unified) ≤ 5` | A small gap = generalisation. Current ~24pp gap = contamination tax |
| Stage 3 (DPO) on synthetic | ≥ Stage 2b + 2 | DPO should lift pedagogical correctness specifically |

ROUGE/BLEU/IAR/PRR remain as **diagnostic** metrics. If R-1 climbs faster than
unified, that is a memorisation warning — not a win. The previous plan's
"primary target: exceed Qwen2.5-14B zero-shot 43.79 ROUGE-1" is **dropped** —
it re-installs the failure mode `docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md`
exists to detect.

### 5.2 Evaluation datasets

| Dataset | When | What it measures |
|---|---|---|
| SocratDataset-EN test (n=681) | After each stage | In-distribution performance |
| SocratDataset-SYNTHETIC (n ≥ 75) | After each stage | Generalisation (contamination control) |

Synthetic n raised from 37 to ≥75 going forward — n=37 has overlapping CIs that
cannot resolve the gaps this plan needs to measure.

### 5.3 Ablation runs — paired with the BERT consultant

The locked headline architecture is `bert × Teacher`. Every ablation runs
**with the BERT consultant**, not standalone — otherwise A/B is measuring a
different architecture than the project's contribution.

| Run | Stages | Evaluator |
|---|---|---|
| A | Stage 2b only (socrat-en) | `bert × FineTuned` |
| B | Stage 2b (socrat-zh + socrat-en) | `bert × FineTuned` |
| C | Stage 2a → 2b | `bert × FineTuned` |
| D | Stage 1 (scoped) → 2a → 2b | `bert × FineTuned` |
| E | Stage 1 → 2a → 2b → Stage 3 (DPO) | `bert × FineTuned` |

Run A before B, B before C, C before D, D before E.

**Plus one standalone-fusion ablation per stage** to measure the BERT
consultant's contribution. (Standalone Gemma 4 31B fusion underperformed at
n=681 due to a 21% schema-fallback rate; replicating that ablation for the
fine-tuned model isolates whether SFT closes the fallback gap.)

### 5.4 Judge calibration spot-check

Both `SocratDataset-SYNTHETIC` (generated by Claude Sonnet, see
`scripts/generate_synthetic_socrat.py`) and the LLM judge (`llm_judge_eval.py`)
are Sonnet-based. Before treating judge scores as ground truth on synthetic,
**spot-check calibration with Opus (or a human) on a stratified ~10-sample
subset**. Document any systematic deviation. Not fatal — the judge rates
teaching behaviour, not generation similarity — but worth verifying once.

---

## 6. Sequence and status

**Phase 1 — Qwen3.6-27B fine-tune (locked teacher per §0.1).**

| # | Task | Status |
|---|---|---|
| 1 | Establish synthetic baseline (state_acc only) | ✅ Complete — `results/synthetic-baseline/` |
| 2 | Teacher base decision (Qwen3.6-27B, velocity-driven) | ✅ Recorded — §0.1 |
| 3 | **Fix SFT format in `dataset.py`** per §0.2 | **🔥 Next — unblocks Stage 2** |
| 4 | Add Stage 1 dataset loaders (subsampled OpenHermes / UltraChat / SlimOrca) | Planned |
| 5 | Integrate SocraticMATH + `socrat-synthetic` loaders | Planned |
| 6 | Create `configs/train-sft-stage1-general.env` | Planned |
| 7 | Create `configs/train-sft-stage2-socratic.env` | Planned |
| 8 | `scripts/build_dpo_pairs.py` — three sources from §2 Stage 3 | Planned (parallel with Stage 2) |
| 9 | Run ablation A — socrat-en only, paired with BERT | Planned |
| 10 | Run ablation B — socrat-zh + socrat-en | Planned |
| 11 | Run Stage 2a (breadth) → 2b (structural) | Planned |
| 12 | A/B Stage 1 → 2b vs Stage 2b alone — decide Stage 1 inclusion | Planned |
| 13 | Build `scripts/train_dpo.py` + `configs/train-dpo-*.env` | Planned |
| 14 | Run Stage 3 DPO on Stage 2b checkpoint | Planned |

**Parallel evidence-gathering (must complete before #15, not before #11).**

| # | Task | Status | Why |
|---|---|---|---|
| E1 | Synthetic-extension to n=75 (≥100 stretch) on Qwen baseline | Planned | n=37 SE too loose to measure contamination tax with confidence — required for the lift measurement to be defensible |
| E2 | LLM judge on three baseline runs (~$5) | Planned | Place baselines on unified-score so Stage 2b lift can be reported on the project's primary metric |

**Phase 1 evaluation + publication.**

| # | Task | Status |
|---|---|---|
| 15 | Evaluate all Qwen runs — paired with BERT + Opus judge calibration check (§5.4) | Planned |
| 16 | Publish Qwen winning checkpoint to HuggingFace | Stretch |

**Phase 2 — Gemma 4 31B head-to-head (after Phase 1 lands).**

| # | Task | Status |
|---|---|---|
| 17 | Synthetic-extension n=75 on Gemma 4 31B baseline | Planned (depends on E1 method) |
| 18 | Create `configs/train-sft-gemma4-31b-qlora.env` (mirror of Qwen config, `TRAIN_BASE_MODEL` swapped) | Planned |
| 19 | Run Gemma Stages 2a → 2b → 3 with the same data + hyperparameters | Planned |
| 20 | Head-to-head eval: Qwen-FT vs Gemma-FT, paired with BERT, on test + synthetic | Planned |
| 21 | Publish whichever wins on unified score (or both, with paper noting the comparison) | Stretch |

---

## 7. Config files

| File | Purpose | Status |
|---|---|---|
| `configs/train-sft-qwen36-27b-qlora.env` | QLoRA on Qwen3.6-27B, Stages 2a/2b (Phase 1) | ✅ Created |
| `configs/train-sft-qwen25-7b-lora.env` | LoRA on Qwen2.5-7B (lighter fallback) | ✅ Exists |
| `configs/train-sft-stage1-general.env` | Stage 1 scoped general SFT | Planned |
| `configs/train-sft-stage2-socratic.env` | Stage 2 (split-config from base QLoRA) | Planned |
| `configs/train-dpo-qwen36-27b.env` | Stage 3 DPO (Phase 1) | Planned |
| `configs/train-sft-gemma4-31b-qlora.env` | Phase 2 head-to-head — mirror of Qwen config | Planned |
| `configs/train-dpo-gemma4-31b.env` | Phase 2 DPO | Planned |

---

## 8. References

- Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models," EMNLP 2025 Findings
- Liu et al., "SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models," NeurIPS 2024 (Spotlight)
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," NeurIPS 2023
- `docs/BENCHMARK_CRITIQUE_AND_PROPOSAL.md` — four-metric memorisation-resistant panel
- `docs/UNIFIED_RANKING.md` — canonical unified-score definition
- `docs/SOCRATTEACHLLM_CONTAMINATION_PROOF.md` — STL memorisation evidence (motivates synthetic baseline + Source 2 of DPO pairs)
- `docs/PROMPT_ENGINEERING_PLAN.md` — negative-exemplar utility (+0.29 composite) — Source 3 anti-pattern catalogue
- HF TRL `SFTTrainer`: https://huggingface.co/docs/trl/sft_trainer
- HF TRL `DPOTrainer`: https://huggingface.co/docs/trl/dpo_trainer
- PEFT LoRA: https://huggingface.co/docs/peft/conceptual_guides/lora
