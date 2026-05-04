---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-generation
pretty_name: SocratDataset-EN
tags:
  - education
  - socratic-teaching
  - dialogue
  - science
  - elementary-school
  - english
  - kele
  - llm-training
  - translation
size_categories:
  - 1K<n<10K
---

# SocratDataset-EN

**English translation of SocratDataset — Chinese elementary-school science tutoring dialogues following the SocRule framework.**

SocratDataset-EN is a complete English translation of [ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset), the training corpus for SocratTeachLLM (KELE, EMNLP 2025 Findings). It enables English-language research and fine-tuning of Socratic teaching models without requiring access to the original Chinese data.

---

## Dataset Summary

| Property | Value |
|---|---|
| Language | English |
| Source language | Chinese (Simplified) |
| Domain | Elementary school science (grades 1–6) |
| Records | 6,803 (100% of source) |
| Dialogue turns | 5–12 per record (median: 6) |
| Question types | `multiple_choice`, `true_false` |
| Framework | SocRule (5 stages, 34 strategies) |

---

## Dataset Structure

Each record contains the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | int | Record identifier matching the source dataset (1–6803) |
| `grade` | string | Grade level and volume (e.g., `Grade 4 Vol. 1`) |
| `chapter` | string | Chapter or topic label |
| `mission` | string | `multiple_choice` or `true_false` |
| `question` | string | Translated question text |
| `options` | list[string] | Translated answer options |
| `answer` | string | Correct answer (carried over from source) |
| `newHint` | string | Translated guiding clue (does not reveal the answer) |
| `newKnowledgePoint` | string | Translated academic concept description |
| `newAnalyze` | string | Translated full analysis of the question and options |
| `dialogueRound` | int | Number of dialogue turns |
| `dialogue` | list[dict] | The translated multi-turn Socratic dialogue |
| `translation_meta` | dict | Translation provenance: `model` and `translated_at` timestamp |

Each turn in `dialogue` contains:

| Field | Type | Description |
|---|---|---|
| `student` | string | Translated student utterance |
| `evaluation` | string | Translated consultant assessment (stage + state + justification) |
| `state` | string | SocRule state code — unchanged (`a1`, `b2`–`b7`, `c8`–`c29`, `d30`–`d33`, `e34`) |
| `action` | string | Translated teaching strategy |
| `teacher` | string | Translated teacher response |

### SocRule Stages

| Stage | Code range | Description |
|---|---|---|
| a — Initiation | a1 | Dialogue starts; student poses the question |
| b — Concept Probing | b2–b7 | Teacher probes prior knowledge |
| c — Inductive Reasoning | c8–c29 | Core teaching stage; can repeat multiple turns |
| d — Answer Derivation | d30–d33 | Guide student to the correct answer |
| e — Summary | e34 | Dialogue ends; teacher summarises |

---

## Usage

```python
from datasets import load_dataset

ds = load_dataset("ulises-c/SocratDataset-EN", split="train")
record = ds[0]

print(record["question"])
print("Options:", record["options"])
for turn in record["dialogue"]:
    print(f"\n[{turn['state']}] {turn['action']}")
    print(f"  Student : {turn['student'][:80]}")
    print(f"  Teacher : {turn['teacher'][:80]}")
```

### Training Objective

The paper formulates teacher fine-tuning as:

```
P(teacher_response | dialogue_history, evaluation, action)
```

The `evaluation` and `action` fields are required conditioning signals — do not discard them when constructing training examples. At inference time a consultant agent produces these fields before the teacher agent generates its response.

---

## Translation Methodology

### Overview

The translation was produced using a local LLM server running **Qwen3.5-9B-UD-Q4_K_XL** (quantized, via llama.cpp), with a two-tier checkpointing system (local every 5 records, HuggingFace every 50 records) to make long overnight runs resumable.

The translation script is open-source and available at:
**[github.com/ulises-c/csen-346](https://github.com/ulises-c/csen-346/blob/main/src/project/translate_dataset.py)**

### What was translated

- All free-text fields: `question`, `options`, `newHint`, `newKnowledgePoint`, `newAnalyze`
- All dialogue turn fields: `student`, `evaluation`, `teacher`
- All `action` strings (translated in bulk via a shared cache to ensure consistency)
- `grade` and `mission` were mapped using deterministic lookup tables (not LLM-translated)
- `state` codes (`a1`, `b2`, …, `e34`) were passed through unchanged — they are structural labels, not natural language

### LLM translation prompt design

- A one-shot example was prepended to every record
- The system prompt enforced: preserve JSON structure, translate values only, use single quotes instead of double quotes inside values, maintain Socratic/pedagogical tone appropriate for elementary school age
- Thinking was disabled (`enable_thinking: false`) for speed — translation does not require chain-of-thought
- `temperature=0.1` for near-deterministic output

### Validation and retry logic

Each translated record was validated before being saved:
1. JSON must parse successfully (`json.JSONDecoder.raw_decode` to tolerate trailing content)
2. Turn count must match the source
3. Every turn must contain both `student` and `teacher` fields
4. No Chinese characters may remain in `student` or `teacher` fields

On failure the script retried up to 3 times, sending a targeted reminder based on the failure type (JSON parse error, missing fields, or residual Chinese).

### Post-processing

- `_merge_split_turns()` detects when the model outputs 2N alternating student-only/teacher-only turns instead of N combined turns and merges them back
- `_safe_quotes()` strips inline option arrays embedded in student text and replaces ASCII and Unicode smart double-quotes with single quotes
- `_BRACKETLESS_BOOL_RE` strips bare `"Yes"/"No"` pairs that appear in true/false questions without brackets

### Throughput

Approximately **370–390 records/hour** on a local machine with an AMD R9700 GPU running the 9B model at 4-bit quantisation.

---

## Known Limitations and Records to Review

The following 9 records were **manually translated by the dataset author** after the automated pipeline failed on all 3 retries. Readers with stronger Chinese literacy are encouraged to review them for naturalness and accuracy:

| ID | Grade | Type | Failure reason |
|---|---|---|---|
| 1328 | Grade 3 Vol. 1 | multiple_choice | Persistent missing-fields error |
| 1639 | Grade 3 Vol. 1 | true_false | Persistent missing-fields error |
| 2655 | Grade 4 Vol. 1 | true_false | Persistent missing-fields error |
| 2766 | Grade 4 Vol. 1 | true_false | Persistent missing-fields error |
| 3524 | Grade 4 Vol. 2 | multiple_choice | Model consistently produced 7 turns instead of 8 |
| 3532 | Grade 4 Vol. 2 | multiple_choice | Persistent missing-fields error |
| 4162 | Grade 5 Vol. 1 | true_false | JSON delimiter error (unescaped quotes in output) |
| 6313 | Grade 6 Vol. 2 | multiple_choice | Persistent missing-fields error |
| 6437 | Grade 6 Vol. 2 | true_false | JSON delimiter error (unescaped quotes in output) |

These records follow the same field structure and SocRule conventions as the rest of the dataset. They can be identified via `translation_meta.model = "claude-sonnet-4-6"` (all other records have `model = "Qwen3.5-9B-UD-Q4_K_XL.gguf"`).

An additional post-hoc pass fixed 19 residual Chinese characters found in other records — primarily embedded technical terms (`通电`, `归纳`, `人参`, `残缺`, `牛郎星`), untranslated app names (`形色`), and action-field lookup misses — bringing the dataset to zero Chinese characters across all 6,803 records.

---

## Provenance

The original Chinese dataset (SocratDataset) was produced by the KELE research team and distributed as a JSON file in [github.com/yuanpan1020/KELE](https://github.com/yuanpan1020/KELE). It was not published directly on HuggingFace. The Chinese dataset was uploaded to HuggingFace at [ulises-c/SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset) and this English translation was produced by **Ulises Chavarria** as part of coursework for CSEN 346 (Natural Language Processing) at Santa Clara University.

---

## Citation

If you use this dataset, please cite the original KELE paper and credit the translation:

```bibtex
@inproceedings{peng-etal-2025-kele,
  title     = {{KELE}: A Multi-Agent Framework for Structured {S}ocratic Teaching with Large Language Models},
  author    = {Peng, Yuan and others},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025},
  url       = {https://aclanthology.org/2025.findings-emnlp.888/}
}
```

English translation by Ulises Chavarria — [github.com/ulises-c/csen-346](https://github.com/ulises-c/csen-346)

---

## Related Resources

| Resource | Link |
|---|---|
| KELE paper (EMNLP 2025 Findings) | https://aclanthology.org/2025.findings-emnlp.888/ |
| KELE GitHub repository | https://github.com/yuanpan1020/KELE |
| SocratTeachLLM model | https://huggingface.co/yuanpan/SocratTeachLLM |
| Original Chinese dataset | https://huggingface.co/datasets/ulises-c/SocratDataset |
| Translation + evaluation code | https://github.com/ulises-c/csen-346 |
