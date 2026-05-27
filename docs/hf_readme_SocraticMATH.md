---
license: cc-by-nc-4.0
language:
  - zh
task_categories:
  - text-generation
pretty_name: SocraticMATH
tags:
  - education
  - socratic-teaching
  - dialogue
  - mathematics
  - elementary-school
  - chinese
  - socraticllm
  - cikm-2024
  - llm-training
size_categories:
  - 1K<n<10K
---

# SocraticMATH

**Chinese primary-school Socratic math tutoring dialogues.**

This is the SocraticMATH dataset from the paper *"Boosting Large Language Models with Socratic Method for Conversational Mathematics Teaching"* (CIKM '24) by Ding et al. It contains 6,846 multi-turn Socratic tutoring conversations covering 513 primary school math knowledge points.

> **Two variants available:**
> - `ulises-c/SocraticMATH` — Conversations only (this dataset)
> - `ulises-c/SocraticMATH-sol` — Conversations with solutions prepended in the first assistant turn

---

## Dataset Summary

| Property | Value |
|---|---|
| Language | Chinese (Simplified) |
| Domain | Primary school mathematics (513 knowledge points) |
| Records | 6,846 total (5,476 train / 685 val / 685 test) |
| Dialogue turns | ~5 turns per conversation, ~86 words per utterance |
| Question types | Multiple choice, fill-in-the-blank, answer questions |
| Framework | Socratic teaching (review → heuristic → rectification → summarization) |
| License | CC BY-NC 4.0 (non-commercial) |

### Comparison to Existing Datasets

| Dataset | Socratic? | Conversational? | Knowledge Tags | Math Teaching Focus |
|---|---|---|---|---|
| **SocraticMATH** | ✅ | ✅ | ✅ | ✅ |
| GSM8K | ❌ | ❌ | ❌ | ❌ |
| MathQA | ❌ | ❌ | ❌ | ❌ |
| MathDial | ❌ | ⚠️ (semi-auto) | ❌ | Limited |

> SocraticMATH is the first dataset designed explicitly for Socratic-style math tutoring.

---

## Dataset Structure

Each record contains:

| Field | Type | Description |
|---|---|---|
| `id` | int64 | Unique conversation identifier |
| `conversations` | list[dict] | Multi-turn dialogue |

Each turn in `conversations`:

| Field | Type | Description |
|---|---|---|
| `from` | string | Speaker role: `"user"` (student) or `"assistant"` (teacher) |
| `value` | string | Utterance text |

---

## Usage

```python
from datasets import load_dataset

ds = load_dataset("ulises-c/SocraticMATH", split="train")
record = ds[0]
print(f"Conversation {record['id']}: {len(record['conversations'])} turns")
for turn in record["conversations"]:
    print(f"  [{turn['from']}]: {turn['value'][:80]}")
```

### Training Objective

The paper fine-tunes SocraticLLM (based on Qwen1.5-7B with LoRA) to generate teacher responses that follow a 4-phase strategy:
1. **Review** — Clarify concepts or prior knowledge
2. **Heuristic** — Ask guiding questions to promote discovery
3. **Rectification** — Detect and correct student errors
4. **Summarization** — Reinforce learning and conclude

---

## Splits

| Split | Records |
|---|---|
| Train | 5,476 |
| Validation | 685 |
| Test | 685 |

---

## Provenance

This dataset was produced by the ECNU-ICALK lab for the paper published at CIKM 2024. The original data is hosted on [GitHub](https://github.com/ECNU-ICALK/SocraticMath). This HuggingFace upload was created by [Ulises Chavarria](https://huggingface.co/ulises-c) to make the dataset more accessible.

The source data consists of real primary school exam questions from China, manually annotated with Socratic-style tutoring dialogues. Each conversation includes original problem text, step-by-step solutions, knowledge tags, and difficulty levels.

---

## Citation

If you use this dataset, please cite the original paper:

```bibtex
@inproceedings{ding2024socratic,
  title     = {Boosting Large Language Models with {S}ocratic Method for Conversational Mathematics Teaching},
  author    = {Ding, Yuyang and Hu, Hanglei and Zhou, Jie and Chen, Qin and Jiang, Bo and He, Liang},
  booktitle = {Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '24},
  year      = {2024},
  doi       = {10.1145/3627673.3679881}
}
```

---

## Related Resources

| Resource | Link |
|---|---|
| Paper (CIKM '24) | https://doi.org/10.1145/3627673.3679881 |
| SocraticMath GitHub | https://github.com/ECNU-ICALK/SocraticMath |
| SocraticLLM model | https://huggingface.co/CogBase-USTC/SocraticLM (enhanced version) |
| With-solutions variant | https://huggingface.co/datasets/ulises-c/SocraticMATH-sol |
| Socratic Teaching collection | https://huggingface.co/collections/ulises-c/socratic-teaching-datasets |
| Upload repository | https://github.com/ulises-c/csen-346 |
