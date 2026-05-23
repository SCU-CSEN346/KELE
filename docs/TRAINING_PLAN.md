# Training Plan — SocratTeachLLM v2

**CSEN 346 · Santa Clara University**

Goal: fine-tune a new Socratic teaching model that generalizes beyond SocratTeachLLM's training data.
Hypothesis: the original KELE model is over-fit to the ~6,800-record SocratDataset; adding two external datasets
(SocratDataset-EN + SocraTeach_Multi/Single) should produce a model that handles a wider range of topics
and student personas without losing the structural SocRule precision.

Pre-requisites: PRs 62 and 64 merged.

---

## 1. Datasets

| Dataset | HF ID | Records | SocRule annotations | Language | Role |
|---|---|---|---|---|---|
| SocratDataset (original) | `ulises-c/SocratDataset` | 6,803 dialogues | ✅ `state`, `action`, `evaluation` | Chinese | Fine-tune anchor — original paper's training set |
| SocratDataset-EN | `ulises-c/SocratDataset-EN` | 6,803 dialogues | ✅ `state`, `action`, `evaluation` | English | Fine-tune anchor — English version of same data |
| SocraTeach_Multi | `ulises-c/SocraTeach_Multi` | 10,273 problems / ~30K+ dialogues | ❌ | English | Pre-training breadth — math Socratic patterns |
| SocraTeach_Single | `ulises-c/SocraTeach_Single` | 20,845 exchanges | ❌ | English | Pre-training breadth — student misconception diversity |

Local copy of SocratDataset (original Chinese): `references/KELE/SocratDataset.json`

### 1.1 Dataset roles in detail

**SocratDataset (original Chinese)** is the dataset the original SocratTeachLLM was trained on.
Including it alongside the English translation doubles the SocRule-annotated training signal
(~13,600 total annotated dialogues) and makes the model bilingual. The `state`, `action`, and
`evaluation` fields carry the full KELE structural signal in both languages.

**SocratDataset-EN** is the English translation of the above. Including both ensures the model
learns SocRule-conditioned generation in English, which is needed for evaluation against the
existing English test pipeline (`src/project/kele.py`, `src/project/metrics.py`).

**SocraTeach_Multi** covers GSM8K + MAWPS math word problems with multi-turn Socratic dialogues.
The teacher's guiding questions (`system` field per turn) serve as pre-training signal for Socratic
question generation. No state annotations — cannot be used for SocRule-conditioned fine-tuning.
Math coverage extends beyond the science-only SocratDataset.

**SocraTeach_Single** provides 3 student personas (`incorrect`, `ask_for_hint`, `ask_for_answer`)
per problem, teaching the model to adapt to different student approaches. Stored as single-turn
(prompt/response) with a conversation history prefix. Richer in persona diversity than SocratDataset-EN
where student behavior is fixed.

---

## 2. Training approach — LoRA vs QLoRA

The choice depends on the base model and R9700 VRAM budget (32 GB).

| Approach | Base model VRAM | Peak training VRAM | Fits R9700 (32 GB) | Tradeoff |
|---|---|---|---|---|
| LoRA on Qwen2.5-7B (bf16) | ~14 GB | ~24–26 GB | ✅ comfortably | Full precision — best quality |
| LoRA on Qwen2.5-14B (bf16) | ~28 GB | >32 GB | ❌ OOM likely | Needs gradient checkpointing, risky |
| QLoRA on Qwen2.5-7B (4-bit) | ~5 GB | ~10–12 GB | ✅ large headroom | Slight quality degradation vs LoRA |
| QLoRA on Qwen2.5-14B (4-bit) | ~9 GB | ~16–18 GB | ✅ comfortably | Better capacity than 7B LoRA |
| LoRA on GLM4-9B (bf16) | ~19 GB | ~28–30 GB | ⚠️ tight | Paper's original model; Chinese-centric |

### Recommendation

**LoRA on Qwen2.5-7B** for the first training run. Rationale:
- SocratDataset-EN is already in English — Qwen2.5-7B outperforms GLM4-9B on English instruction following
- 7B fits LoRA in bf16 without quantization, avoiding QLoRA's precision loss
- Fits comfortably in R9700's 32 GB without gradient checkpointing tricks
- If 7B results are insufficient, upgrade to QLoRA on Qwen2.5-14B for the next run

**QLoRA on Qwen2.5-14B** is the fallback if 7B LoRA underperforms on the structural metrics (PRR, IAR).
The 14B model has meaningfully more capacity for the dual conditioning task
(generate teacher response given evaluation + action).

### LoRA configuration (for Qwen2.5-7B)

| Setting | Value | Notes |
|---|---|---|
| `r` (rank) | 32 | Match paper's expressiveness; 16 if VRAM tight |
| `lora_alpha` | 64 | Standard 2× r |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All linear layers |
| `lora_dropout` | 0.05 | Light regularization |
| `bias` | none | |
| Epochs | 3 | Match paper |
| Learning rate | 5e-5 | Paper's value; reduce to 2e-5 if loss spikes |
| Batch size | 4 (per device) + grad accum 4 | Effective batch 16 to match paper |
| Max sequence length | 2048 | Longest SocratDataset-EN dialogues are ~1,200 tokens |
| bf16 | true | R9700 supports bfloat16 |
| Gradient checkpointing | false | Not needed at 7B LoRA; enable if tight |

### ROCm-specific notes

The R9700 uses AMD ROCm (HF Transformers path, not vLLM). Training works via `accelerate` + HF `Trainer`/TRL's `SFTTrainer`.
Key compatibility checks before first training run:

- `torch.backends.cuda.enable_mem_efficient_sdp(False)` if FlashAttention errors
- `peft >= 0.19.1` (already in `pyproject.toml`)
- `trl >= 1.4.0` (already in `pyproject.toml`)
- Verify `accelerate config` selects the ROCm device before launching

---

## 3. Curriculum learning strategy

### 3.1 Rationale

Training directly on SocratDataset-EN alone replicates the original KELE over-fit problem.
Training on all three datasets mixed uniformly dilutes the structural SocRule signal.
Curriculum learning addresses both: first build broad Socratic dialogue competence,
then sharpen the structural conditioning.

### 3.2 Two-phase curriculum

**Phase 1 — Breadth pre-training (SocraTeach)**

Train on SocraTeach_Multi + SocraTeach_Single only. These datasets have no SocRule annotations;
the training objective is plain causal LM on (student_input → teacher_response) pairs. This
teaches the model Socratic question generation across diverse topics and student personas
without imposing the strict SocRule structure.

Datasets: `SocraTeach_Multi` + `SocraTeach_Single` (~51K total training records after dedup and split)
Training format: plain messages — no state/action conditioning
Epochs: 1–2 (enough to absorb Socratic patterns; more risks forgetting base model capabilities)
Checkpoint: save after phase 1 for comparison

**Phase 2 — Structural fine-tuning (SocratDataset-EN)**

Fine-tune the phase 1 checkpoint on SocratDataset-EN with full SocRule conditioning.
The `evaluation` and `action` fields are included in the input prompt so the model learns
to follow the 34-strategy taxonomy when conditioned on the consultant's output.

Dataset: `SocratDataset-EN` (6,803 records, 90/10 train/test split = same as paper)
Training format: state-conditioned messages — evaluation + action in system or prefix
Epochs: 3 (match paper)
Checkpoint: final model for evaluation

### 3.3 Alternative — mixed training with source weighting

If two-phase curriculum shows no improvement over baseline, try single-phase mixed training:

| Source | Records | Sampling weight |
|---|---|---|
| SocratDataset (Chinese) | 6,803 | 3× (oversample annotated data) |
| SocratDataset-EN | 6,803 | 3× (oversample annotated data) |
| SocraTeach_Multi | ~30K dialogues | 1× |
| SocraTeach_Single | ~18K train records | 1× |

Use TRL `SFTTrainer` with a custom sampler or `datasets.concatenate_datasets` with explicit
weighting. The 3× oversample on SocratDataset-EN preserves the structural signal while
using the SocraTeach data for regularization.

---

## 4. Training data format

All datasets are normalized to HF `messages` format, compatible with TRL's `SFTTrainer` and
`apply_chat_template`. See `src/project/dataset.py` for implementation.

### 4.1 SocratDataset — Chinese and English (state-conditioned)

The consultant's `evaluation` and `action` are prepended to each teacher turn so the model
learns to condition its response on the state classification.

Chinese records use a Chinese system prompt; English records use an English system prompt.
Both use the same state/action prefix format so the model learns the conditioning signal in both languages.

```
system:    "You are a Socratic teacher...\nProblem: {question}\nOptions: {options}\nHint: {hint}"
user:      "{student_turn_1}"
assistant: "[State: {state_1}] [Action: {action_1}]\n{teacher_turn_1}"
user:      "{student_turn_2}"
assistant: "[State: {state_2}] [Action: {action_2}]\n{teacher_turn_2}"
...
```

At inference time the consultant agent predicts the state/action; at training time ground-truth
values are used. This matches the KELE two-agent setup: the fine-tuned teacher model is always
called with consultant output prepended to the user turn or as a system prefix.

### 4.2 SocraTeach_Multi (plain Socratic)

Each dialogue within a problem is one training sequence. The teacher's step-guiding question
is the assistant turn; the student's (confused/incorrect) response is the user turn.

```
system:    "You are a Socratic teacher. Guide the student through this problem using questions.\nProblem: {question}"
assistant: "{teacher_question_turn_1}"
user:      "{student_response_turn_1}"
assistant: "{teacher_question_turn_2}"
user:      "{student_response_turn_2}"
...
```

Note: SocraTeach_Multi teacher turns lead (teacher asks → student responds), which differs from
SocratDataset-EN (student speaks first). This reflects different Socratic dialogue styles
and is intentionally preserved — the model should handle both entry patterns.

### 4.3 SocraTeach_Single (persona-aware)

Each record is one (prompt, response) exchange with conversation history. The `student_type`
field encodes the student persona (`incorrect`, `ask_for_hint`, `ask_for_answer`).

```
system:    "You are a Socratic teacher. Student type: {student_type}.\nContext: {history_system_message}"
user:      "{turn_from_history[0]}"
assistant: "{turn_from_history[1]}"
...
user:      "{prompt}"           (current student input)
assistant: "{response}"         (target teacher response)
```

---

## 5. Evaluation plan

### 5.1 Metrics

Use the same metrics as the KELE paper (implemented in `src/project/metrics.py`):

| Metric | Source | Notes |
|---|---|---|
| ROUGE-1, ROUGE-2 | `src/project/metrics.py` | Character-level for EN dataset |
| BLEU-4 | `src/project/metrics.py` | |
| State accuracy | `src/project/metrics.py` | Only on SocratDataset-EN test set |
| PRR / NDAR | Manual or GPT-4o judge | Binary per turn |
| SPR / IAR | Manual or GPT-4o judge | Binary per dialogue |

SocraTeach datasets cannot be used for evaluation against KELE metrics because they lack
SocRule state/action annotations. Evaluation must run on the SocratDataset-EN test split only.

### 5.2 Baselines to beat

From `docs/DATASET_EXPANSION_PLAN.md` Table 1:

| Model | ROUGE-1 | PRR | IAR |
|---|---|---|---|
| SocratTeachLLM (paper) | 57.4 | 75.13 | 89.03 |
| Qwen2.5-7B (zero-shot) | 40.95 | 59.02 | 76.45 |
| GPT-4o (zero-shot) | 38.25 | 72.13 | 87.74 |

Primary target: exceed Qwen2.5-7B zero-shot (40.95 ROUGE-1) after phase 2 fine-tuning.
Stretch target: approach SocratTeachLLM (57.4) — this is the same task but in English.

### 5.3 Ablation runs

| Run | Dataset | Notes |
|---|---|---|
| A | SocratDataset-EN only | English-only baseline; measures curriculum benefit |
| B | SocratDataset (ZH + EN) | Bilingual annotated baseline; measures Chinese data benefit |
| C | SocraTeach pre-train → SocratDataset (ZH + EN) fine-tune | Full curriculum |
| D | All four datasets mixed (weighted) | Alternative to curriculum |

Run A and B must complete before C or D to establish the annotated-data baselines.

---

## 6. Sequence and status

| Step | Task | Status |
|---|---|---|
| 1 | Upload SocraTeach_Multi + SocraTeach_Single to HuggingFace | ✅ Complete |
| 2 | Implement unified data loader (`src/project/dataset.py`) | ✅ Complete |
| 3 | Update `kele.py::load_dataset` for HF source | ✅ Complete |
| 4 | Merge PR 62 (parallel eval) + PR 64 (tournament results) | Pre-req |
| 5 | Run ablation A — SocratDataset-EN only fine-tune | Planned |
| 6 | Run ablation B — curriculum (SocraTeach → SocratDataset-EN) | Planned |
| 7 | Evaluate all runs; compare to Table 1 baselines | Planned |
| 8 | Publish winning model checkpoint to HuggingFace | Stretch |

---

## 7. References

- Peng et al., "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models," EMNLP 2025 Findings
- Liu et al., "SocraticLM: Exploring Socratic Personalized Teaching with Large Language Models," NeurIPS 2024 (Spotlight)
- HF TRL `SFTTrainer` docs: https://huggingface.co/docs/trl/sft_trainer
- PEFT LoRA docs: https://huggingface.co/docs/peft/conceptual_guides/lora
