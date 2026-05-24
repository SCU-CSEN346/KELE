# Training Plan — SocratTeachLLM v2

**CSEN 346 · Santa Clara University**

Goal: fine-tune Qwen3.6-27B into a Socratic teaching model that generalizes beyond SocratTeachLLM's
training data, using a 3-stage pipeline: general instruction fluency → Socratic SFT →
preference optimization (DPO).

The contamination-control baseline for generalization is established: see
`results/synthetic-baseline/` (SocratDataset-SYNTHETIC, n=37, unseen data).

---

## 0. Base model

| Field | Value |
|---|---|
| HF model ID | `Qwen/Qwen3.6-27B-Instruct` |
| Base repo | `Qwen/Qwen3.6-27B` |
| GGUF (inference) | `unsloth/Qwen3.6-27B-GGUF` — Q4_K_M at `~/models/Qwen3.6-27B/` |
| Training method | **QLoRA** (4-bit NF4 base + bf16 adapters) |
| Training VRAM | ~22–26 GB peak on R9700 (32 GB) with gradient checkpointing |
| Inference VRAM | ~16 GB (Q4_K_M GGUF via llama-server) |

### Why Qwen3.6-27B

- **Best generalisation** on SocratDataset-SYNTHETIC (think-mode eval in progress)
- Tournament results (Qwen 41.65 > Gemma 38.71) and synthetic baseline (Gemma 27.23 > Qwen 23.44 no-think)
  together suggest think-mode Qwen closes or reverses the gap — final numbers pending
- 27B parameter count gives meaningfully more capacity than 7B/14B for the dual
  conditioning task (generate teacher response given evaluation + action)
- Same model used throughout tournament and synthetic evals → fine-tune delta is directly
  interpretable against those baselines

---

## 1. Datasets

### Stage 1 — General SFT (to be integrated)

| Dataset | HF ID | Size | Role |
|---|---|---|---|
| OpenHermes | `teknium/openhermes` | ~1M | Instruction-following breadth |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` | 200K | Multi-turn conversational fluency |
| SlimOrca | `Open-Orca/SlimOrca` | ~500K | Reasoning + instruction diversity |

These datasets are **not yet integrated** in `src/project/dataset.py` — loaders to be added
before Stage 1 training begins.

### Stage 2 — Socratic SFT

| Dataset | HF ID | Records | SocRule annotations | Language | Role |
|---|---|---|---|---|---|
| SocratDataset (original) | `ulises-c/SocratDataset` | 6,803 dialogues | ✅ state, action, evaluation | Chinese | Structural anchor |
| SocratDataset-EN | `ulises-c/SocratDataset-EN` | 6,803 dialogues | ✅ state, action, evaluation | English | Structural anchor (English) |
| SocraTeach_Multi | `ulises-c/SocraTeach_Multi` | ~30K+ dialogues | ❌ | English | Socratic breadth — math |
| SocraTeach_Single | `ulises-c/SocraTeach_Single` | 20,845 exchanges | ❌ | English | Persona diversity |
| SocraticMATH | `github.com/ECNU-ICALK/SocraticMath` | TBD | ❌ | English | Math Socratic dialogues (to integrate) |

All currently implemented sources: `socrat-zh`, `socrat-en`, `socrateach-multi`, `socrateach-single`
(see `src/project/dataset.py`).

### Stage 3 — DPO

Preference pairs constructed from SocratDataset-EN:

| Preferred | Rejected |
|---|---|
| Teacher asks guiding question, delays answer | Teacher gives direct answer immediately |
| Adapts to student confusion (cycles c-stage) | Ignores learner state, moves on |
| Correct stage transition (follows SocRule) | Wrong stage (e.g. skips to e before d) |
| Uses recommended action from consultant | Ignores consultant action field |

Preference pair generation strategy: use ground-truth teacher turns as **preferred**;
generate **rejected** completions with thinking disabled and a prompt that encourages
direct answers. Human review or LLM-judge filter (B.5 rubric) to validate pairs.

---

## 2. VRAM budget — QLoRA on Qwen3.6-27B (32 GB R9700)

| Component | Estimate |
|---|---|
| Base model (4-bit NF4) | ~14 GB |
| LoRA adapters (bf16, r=16) | ~0.3 GB |
| Activations (batch=1, seq=2048) | ~4–6 GB |
| Optimizer states (AdamW on adapters only) | ~1–2 GB |
| **Total peak** | **~20–23 GB** |

Gradient checkpointing (`TRAIN_GRAD_CKPT=true`) is enabled by default for 27B QLoRA
to trade compute for memory. At batch=1 + grad_accum=16 the effective batch is 16,
matching the paper.

Previous analysis (Qwen2.5-7B LoRA bf16, ~24 GB peak) is preserved in
`configs/train-sft-qwen25-7b-lora.env` as a lighter fallback if 27B is unstable.

---

## 3. Three-stage pipeline

### Stage 1 — General instruction SFT

**Goal:** conversational fluency and instruction following before Socratic shaping.

**Datasets:** OpenHermes + UltraChat 200k + SlimOrca (loaders TBD)

**Format:** plain chat — system/user/assistant, no SocRule conditioning.

**Config:** `configs/train-sft-stage1-general.env` (to be created)

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs | 1 | Enough to absorb patterns; avoid catastrophic forgetting of base |
| LR | 2e-5 | Lower than Stage 2 — subtle shaping, not structural rewrite |
| Batch | 1 × 16 grad accum | Effective 16 |
| Max seq len | 2048 | |
| LoRA rank | 16 | Lighter adapter; structural knowledge comes in Stage 2 |
| Grad ckpt | true | 27B QLoRA |

Save checkpoint after Stage 1 for ablation comparison.

---

### Stage 2 — Socratic SFT

**Goal:** teach SocRule-conditioned generation (the `evaluation` + `action` conditioning signal).

**Sub-phase 2a — Breadth pre-training:**
Train on `socrateach-multi` + `socrateach-single` (+ SocraticMATH once integrated).
Plain Socratic dialogue format, no state conditioning. 1–2 epochs.

**Sub-phase 2b — Structural fine-tuning:**
Fine-tune the 2a checkpoint on `socrat-zh` + `socrat-en` with full SocRule conditioning.
3 epochs, matching the paper.

**Config:** `configs/train-sft-stage2-socratic.env` (to be created)

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs (2a) | 1–2 | Breadth |
| Epochs (2b) | 3 | Match paper |
| LR | 5e-5 | Paper's value |
| Batch | 1 × 16 grad accum | |
| Max seq len | 2048 | Longest SocratDataset-EN dialogues ~1,200 tokens |
| LoRA rank | 16 | |
| Grad ckpt | true | |

#### Training data format (state-conditioned)

```
system:    "You are a Socratic teacher...\nProblem: {question}\nOptions: {options}\nHint: {hint}"
user:      "{student_turn_1}"
assistant: "[State: {state_1}] [Action: {action_1}]\n{teacher_turn_1}"
user:      "{student_turn_2}"
assistant: "[State: {state_2}] [Action: {action_2}]\n{teacher_turn_2}"
```

At inference the consultant predicts state/action; at training, ground-truth values are used.

#### Alternative — weighted mixed training

If two-phase shows no improvement over baseline:

| Source | Weight |
|---|---|
| socrat-zh | 3× |
| socrat-en | 3× |
| socrateach-multi | 1× |
| socrateach-single | 1× |

---

### Stage 3 — DPO preference optimization

**Goal:** teach the model *when* to hint, explain, challenge, or reveal — not just *how* to ask questions.

**Datasets:** preference pairs derived from SocratDataset-EN (see Section 1, Stage 3).

**Script:** `scripts/train_dpo.py` (to be written — mirrors `scripts/train_sft.py` structure,
uses TRL `DPOTrainer`).

**Config:** `configs/train-dpo-qwen36-27b.env` (to be created)

| Hyperparameter | Value | Notes |
|---|---|---|
| Beta | 0.1 | KL penalty weight; standard DPO starting point |
| Epochs | 1–2 | DPO converges fast; more risks mode collapse |
| LR | 1e-5 | Lower than SFT |
| Max seq len | 2048 | |
| Reference model | Stage 2b checkpoint | Frozen |

The insight from the ChatGPT notes applies: models that endlessly ask questions become
frustrating. DPO teaches the model that "dumping the answer immediately" and
"refusing to help" are *both* rejected — the preferred behavior is adaptive scaffolding
guided by the SocRule stage.

---

## 4. Evaluation plan

### 4.1 Baselines to beat

From `docs/DATASET_EXPANSION_PLAN.md` Table 1:

| Model | ROUGE-1 | PRR | IAR | state_acc (synthetic) |
|---|---|---|---|---|
| SocratTeachLLM (paper) | 57.4 | 75.13 | 89.03 | — |
| Qwen2.5-14B zero-shot | 43.79 | 65.21 | 80.81 | — |
| Qwen3.6-27B Q4 no-think | — | — | — | 23.44% |
| Qwen3.6-27B Q4 think-4096 | — | — | — | pending |
| Gemma 4 31B Q5 | — | — | — | 27.23% |

Primary target: **exceed Qwen2.5-14B zero-shot** (43.79 ROUGE-1) after Stage 2b.
Contamination control: **synthetic Δ ≥ test Δ** → genuine generalisation, not memorisation.
Stretch target: approach SocratTeachLLM (57.4 ROUGE-1).

### 4.2 Evaluation datasets

| Dataset | When | What it measures |
|---|---|---|
| SocratDataset-EN test (n=681) | After each stage | In-distribution performance |
| SocratDataset-SYNTHETIC (n=37) | After each stage | Generalisation (contamination control) |

### 4.3 Ablation runs

| Run | Stages | Notes |
|---|---|---|
| A | Stage 2b only (socrat-en) | English-only baseline |
| B | Stage 2b (socrat-zh + socrat-en) | Bilingual annotated baseline |
| C | Stage 2a → 2b | Full Socratic curriculum |
| D | Stage 1 → 2a → 2b | Full 3-stage pipeline |
| E | Stage 1 → 2a → 2b → Stage 3 | With DPO |

Run A before B, B before C, C before D.

---

## 5. Sequence and status

| Step | Task | Status |
|---|---|---|
| 1 | Establish synthetic baseline (contamination control) | ✅ Complete — `results/synthetic-baseline/` |
| 2 | Think-mode Qwen3.6-27B synthetic baseline | 🔄 In progress (4096 budget, ~6h run) |
| 3 | Add Stage 1 dataset loaders (OpenHermes, UltraChat, SlimOrca) | Planned |
| 4 | Integrate SocraticMATH into dataset.py | Planned |
| 5 | Create `configs/train-sft-stage1-general.env` | Planned |
| 6 | Run ablation A — socrat-en only fine-tune (Qwen3.6-27B QLoRA) | Planned |
| 7 | Run ablation B — socrat-zh + socrat-en | Planned |
| 8 | Run Stage 2a (socrateach breadth pre-train) | Planned |
| 9 | Run Stage 2b (structural fine-tune from 2a checkpoint) | Planned |
| 10 | Build `scripts/train_dpo.py` + preference pair generator | Planned |
| 11 | Run Stage 3 DPO on Stage 2b checkpoint | Planned |
| 12 | Evaluate all runs on SocratDataset-EN test + synthetic | Planned |
| 13 | Publish winning checkpoint to HuggingFace | Stretch |

---

## 6. Config files

| File | Purpose | Status |
|---|---|---|
| `configs/train-sft-qwen36-27b-qlora.env` | QLoRA on Qwen3.6-27B, Stages 2a/2b | ✅ Created |
| `configs/train-sft-qwen25-7b-lora.env` | LoRA on Qwen2.5-7B (lighter fallback) | ✅ Exists |
| `configs/train-sft-stage1-general.env` | Stage 1 general SFT | Planned |
| `configs/train-dpo-qwen36-27b.env` | Stage 3 DPO | Planned |

---

## 7. References

- Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models," EMNLP 2025 Findings
- Liu et al., "SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models," NeurIPS 2024 (Spotlight)
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," NeurIPS 2023
- HF TRL `SFTTrainer`: https://huggingface.co/docs/trl/sft_trainer
- HF TRL `DPOTrainer`: https://huggingface.co/docs/trl/dpo_trainer
- PEFT LoRA: https://huggingface.co/docs/peft/conceptual_guides/lora
