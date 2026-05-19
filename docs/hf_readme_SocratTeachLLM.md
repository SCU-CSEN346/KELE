---
license: apache-2.0
language:
  - zh
  - en
tags:
  - education
  - socratic-teaching
  - dialogue
  - fine-tuned
  - glm4
  - kele
  - lora
base_model: THUDM/glm-4-9b-chat
---

# SocratTeachLLM

A LoRA fine-tuned [GLM4-9B-Chat](https://huggingface.co/THUDM/glm-4-9b-chat) model trained to act as a **Socratic teacher** in structured educational dialogues. It generates heuristic questions and formative feedback that guide students through a principled sequence of reasoning stages, following the [KELE framework](https://aclanthology.org/2025.findings-emnlp.888) (Peng et al., EMNLP 2025 Findings).

> **Original model:** [yuanpan/SocratTeachLLM](https://huggingface.co/yuanpan/SocratTeachLLM) — this repository is a copy with an expanded README.

---

## What It Does

SocratTeachLLM is designed for the **teacher role** in a dual-agent Socratic tutoring system. A separate **consultant agent** (e.g., GPT-4o or Qwen) selects a teaching strategy from a predefined set of 34 Socratic rules (SocRule); SocratTeachLLM then generates the actual dialogue turn implementing that strategy.

Teaching proceeds through five stages (SocRule):

| Stage | Name | State codes | Description |
|---|---|---|---|
| a | Initiation | a1 | Student poses the question; dialogue begins |
| b | Concept Probing | b2–b7 | Teacher probes prior knowledge and surfaces misconceptions |
| c | Inductive Reasoning | c8–c29 | Core teaching stage — guides the student toward generalizations; can repeat many turns |
| d | Answer Derivation | d30–d33 | Help the student arrive at the correct answer |
| e | Summary | e34 | Consolidate and reinforce learning |

The model was fine-tuned on **SocratDataset**: 6,803 multi-turn Socratic dialogues covering 42,000+ interaction turns across elementary school science topics in Chinese.

---

## Published Performance

Results from Table 1 of the KELE paper (test set: 680 dialogues, 4,245 single-turn examples):

| Model | ROUGE-1 | ROUGE-2 | BLEU-4 | PRR | NDAR | SPR | IAR | Guidance | Logicality | Flexibility |
|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | 38.25 | 22.35 | 29.93 | 72.13 | 81.19 | 85.00 | 87.74 | 4.35 | 4.50 | 4.33 |
| Qwen2.5-7B | 40.95 | 15.27 | 24.96 | 59.02 | 80.52 | 60.00 | 76.45 | 3.87 | 3.96 | 3.87 |
| Qwen2.5-14B | 43.79 | 17.06 | 26.63 | 65.21 | 78.57 | 74.00 | 80.81 | 3.99 | 4.15 | 4.03 |
| Qwen2.5-32B | 46.22 | 19.90 | 28.85 | 65.57 | 83.13 | 81.00 | 84.68 | 4.12 | 4.44 | 4.21 |
| EduChat-13B | 34.75 | 9.91 | 21.11 | 47.62 | 90.73 | 51.00 | 69.02 | 2.93 | 3.42 | 3.18 |
| SocraticLM-7B | 18.63 | 5.56 | 10.93 | 26.83 | 30.26 | 36.00 | 27.05 | 2.62 | 2.88 | 2.78 |
| **SocratTeachLLM (this model)** | **57.40** | **33.63** | **41.96** | **75.13** | **94.71** | **87.00** | **89.03** | **4.66** | **4.53** | **4.45** |

**Metric definitions:**
- **PRR** — Problem Relevance Rate: teacher question relates directly to the problem
- **NDAR** — No Direct Answer Rate: teacher avoids giving away the answer
- **SPR** — Summary Pass Rate: correct and complete final summary
- **IAR** — Instruction Adherence Rate: teacher follows the consultant's recommended strategy
- **Guidance / Logicality / Flexibility** — GPT-4o judge scores on a 1–5 scale (B.5 rubric)

SocratTeachLLM outperforms GPT-4o on every metric despite being ~40× smaller.

---

## Training Details

| Setting | Value |
|---|---|
| Base model | GLM4-9B-Chat |
| Method | LoRA |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Batch size | 16 |
| Train split | 6,123 dialogues (90%) |
| Test split | 680 dialogues (10%) |
| Hardware | 2× NVIDIA A800 80GB |
| Dataset | SocratDataset (6,803 records, Chinese) |

### Training Objective

```
P(teacher_response | dialogue_history, evaluation, action)
```

The `evaluation` (consultant's stage/state assessment) and `action` (recommended strategy) fields are required conditioning signals. At inference time, a consultant agent produces these before the teacher agent generates its response. Without the consultant outputs as conditioning, the model will underperform.

---

## Model Architecture

| Parameter | Value |
|---|---|
| Base model | GLM4-9B-Chat (`ChatGLMForConditionalGeneration`) |
| Total parameters | ~9.4B |
| Layers | 40 |
| Hidden size | 4,096 |
| Attention heads | 32 |
| FFN hidden size | 13,696 |
| KV channels | 128 |
| Vocabulary size | 151,552 |
| Max context length | 131,072 tokens (128K) |
| Storage dtype | bfloat16 |
| Attention | Multi-query (2 groups), RoPE (ratio 500) |
| Normalization | RMSNorm |
| Weight files | 4× safetensors shards (~18.8 GB total) |

**Generation defaults:** temperature 0.8, top-p 0.8.

---

## Usage

### Transformers (recommended, ~19 GB VRAM)

The model uses custom modeling code, so `trust_remote_code=True` is required.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "ulises-c/SocratTeachLLM"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "What do you think causes the seasons to change?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=512, temperature=0.8, top_p=0.8)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

### 4-bit NF4 via bitsandbytes (~6.5 GB VRAM)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

### vLLM (OpenAI-compatible endpoint)

```bash
vllm serve ulises-c/SocratTeachLLM \
  --served-model-name SocratTeachLLM \
  --dtype bfloat16 \
  --trust-remote-code
```

### Ollama

This repo includes a `Modelfile` (auto-generated by LlamaFactory) with the correct ChatGLM4 stop sequences and a 4,096-token context window.

```bash
ollama create SocratTeachLLM -f Modelfile
ollama run SocratTeachLLM
```

> **Note:** Ollama caps context at 4,096 tokens. For the full 128K context, use Transformers or vLLM.

---

## Built With This Model

**[csen-346](https://github.com/ulises-c/csen-346)** is a downstream course project (CSEN 346 NLP, Santa Clara University) that reproduces and extends the KELE framework using this model as the teacher agent.

Key integration details:
- **Teacher:** SocratTeachLLM, served via FastAPI/HF Transformers (AMD R9700 AI PRO) or vLLM (RTX 5090, L40S, V100)
- **Consultant:** GPT-4o (baseline) or Qwen3.5-9B (local variant)
- **Evaluation:** 680-dialogue test split of SocratDataset, automated with ROUGE, BLEU, and GPT-4o judge (B.5 rubric)
- **English extension:** An English translation of the training dataset is available at [ulises-c/SocratDataset-EN](https://huggingface.co/datasets/ulises-c/SocratDataset-EN)

```bash
hf download ulises-c/SocratTeachLLM --local-dir ~/hf_models/SocratTeachLLM
```

---

## Training Data

| Property | Value |
|---|---|
| Dataset | [ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset) |
| Dialogues | 6,803 |
| Turns | 42,000+ |
| Domain | Elementary school science (grades 1–6) |
| Language | Chinese (Simplified) |
| Train split | 6,123 dialogues (90%) |
| Test split | 680 dialogues (10%) |
| Strategies | 34 SocRule teaching strategies |

An English translation of the training data is available at [ulises-c/SocratDataset-EN](https://huggingface.co/datasets/ulises-c/SocratDataset-EN).

---

## Citation

If you use this model, please cite the original KELE paper:

```bibtex
@inproceedings{peng-etal-2025-kele,
  title     = {{KELE}: A Multi-Agent Framework for Structured {S}ocratic Teaching with Large Language Models},
  author    = {Peng, Yuan and others},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025},
  url       = {https://aclanthology.org/2025.findings-emnlp.888/}
}
```

---

## Related Resources

| Resource | Link |
|---|---|
| KELE paper (EMNLP 2025 Findings) | https://aclanthology.org/2025.findings-emnlp.888/ |
| KELE GitHub repository | https://github.com/yuanpan1020/KELE |
| Original model | https://huggingface.co/yuanpan/SocratTeachLLM |
| Training data (Chinese) | https://huggingface.co/datasets/ulises-c/SocratDataset |
| Training data (English translation) | https://huggingface.co/datasets/ulises-c/SocratDataset-EN |
| Evaluation + inference code | https://github.com/ulises-c/csen-346 |

---

## License

[Apache 2.0](LICENSE)
