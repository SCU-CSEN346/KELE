---
license: cc-by-nc-4.0
language:
  - en
task_categories:
  - text-generation
pretty_name: SocraTeach_Single
tags:
  - education
  - socratic-teaching
  - dialogue
  - mathematics
  - single-turn
  - gsm8k
  - llm-training
  - socraticlm
size_categories:
  - 10K<n<100K
---

# SocraTeach_Single

**Single-turn Socratic teacher responses organized by student response type.**

This is the single-turn split of the SocraTeach dataset, used to train SocraticLM as published in the NeurIPS 2024 Spotlight paper. It contains 20,845 examples where the teacher must respond to a specific student utterance given prior dialogue history. Examples are organized by four student response types: correct, incorrect, irrelevant, and question.

> **Multi-turn split available:** See [ulises-c/SocraTeach_Multi](https://huggingface.co/datasets/ulises-c/SocraTeach_Multi) for the 10,273 full multi-turn Socratic dialogue records.

---

## Dataset Summary

| Property | Value |
|---|---|
| Language | English |
| Domain | Math word problems (GSM8K, MAWPS) |
| Records | 20,845 |
| Student response types | 4 (correct, incorrect, irrelevant, question) |
| Format | Single-turn: history + prompt → response |

---

## Dataset Structure

Each record is keyed by `{type}#{problem_id}` (e.g., `incorrect#GSM8K_train_0_0_4@0`) and contains:

| Field | Type | Description |
|---|---|---|
| `prompt` | string | The student's current utterance (augmented) |
| `response` | string | The expected Socratic teacher reply |
| `history` | list | Prior conversation turns as context |

### Student Response Types

| Type | Description |
|---|---|
| `correct` | Student gives a correct answer — teacher confirms and advances |
| `incorrect` | Student gives a wrong answer — teacher corrects via Socratic probing |
| `irrelevant` | Student goes off-topic — teacher redirects |
| `question` | Student asks a clarifying question — teacher answers and guides |

---

## Usage

```python
from datasets import load_dataset

ds = load_dataset("ulises-c/SocraTeach_Single", split="train")
record = ds[0]

print("Student:", record["prompt"])
print("Teacher:", record["response"])
print("History turns:", len(record["history"]))
```

### Training Objective

Fine-tune the teacher model to generate `response` conditioned on `history` + `prompt`. The `history` field provides the full preceding conversation context and is required for correct teacher behavior.

---

## Provenance

This dataset was produced by the SocraticLM research team at USTC (CogBase lab) and distributed via the [SocraticLM GitHub repository](https://github.com/Ljyustc/SocraticLM). The math problems are drawn from GSM8K (Cobbe et al., 2021) and MAWPS.

This HuggingFace upload was created by [Ulises Chavarria](https://huggingface.co/ulises-c) to make the dataset more accessible.

---

## Citation

If you use this dataset, please cite the original SocraticLM paper:

```bibtex
@article{liu2024socraticlm,
  title={SocraticLM: exploring socratic personalized teaching with large language models},
  author={Liu, Jiayu and Huang, Zhenya and Xiao, Tong and Sha, Jing and Wu, Jinze and Liu, Qi and Wang, Shijin and Chen, Enhong},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={85693--85721},
  year={2024}
}
```

---

## Related Resources

| Resource | Link |
|---|---|
| SocraticLM paper (NeurIPS 2024 Spotlight) | https://proceedings.neurips.cc/paper_files/paper/2024/file/9bae399d1f34b8650351c1bd3692aeae-Paper-Conference.pdf |
| SocraticLM GitHub repository | https://github.com/Ljyustc/SocraticLM |
| SocraticLM model (CogBase-USTC) | https://huggingface.co/CogBase-USTC/SocraticLM |
| Multi-turn split (SocraTeach_Multi) | https://huggingface.co/datasets/ulises-c/SocraTeach_Multi |
| Socratic Teaching collection | https://huggingface.co/collections/ulises-c/socratic-teaching-datasets |
| Upload repository | https://github.com/ulises-c/csen-346 |
