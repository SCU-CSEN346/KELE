---
license: cc-by-nc-4.0
language:
  - en
task_categories:
  - text-generation
pretty_name: SocraTeach_Multi
tags:
  - education
  - socratic-teaching
  - dialogue
  - mathematics
  - multi-turn
  - gsm8k
  - llm-training
  - socraticlm
size_categories:
  - 10K<n<100K
---

# SocraTeach_Multi

**Multi-turn Socratic math tutoring dialogues across 6 student personas.**

This is the multi-turn split of the SocraTeach dataset, used to train SocraticLM as published in the NeurIPS 2024 Spotlight paper. It contains 10,273 math problems (drawn from GSM8K and MAWPS), each paired with multiple Socratic teaching dialogues simulating different real-world student types.

> **Single-turn split available:** See [ulises-c/SocraTeach_Single](https://huggingface.co/datasets/ulises-c/SocraTeach_Single) for the 20,845 single-turn response examples.

---

## Dataset Summary

| Property | Value |
|---|---|
| Language | English |
| Domain | Math word problems (GSM8K, MAWPS) |
| Records | 10,273 |
| Student types | 6 (simulated real-world student scenarios) |
| Dialogue turns | Variable multi-turn per record |
| Framework | Socratic personalized teaching |

---

## Dataset Structure

Each record is keyed by a problem ID (e.g., `GSM8K_train_0`) and contains:

| Field | Type | Description |
|---|---|---|
| `question` | string | The math word problem |
| `analysis` | string | Step-by-step solution analysis |
| `answer` | string | Correct answer |
| `steps` | list[string] | Guiding sub-questions that scaffold the solution |
| `dialogues` | dict | Map of dialogue ID → list of turns |

Each turn in a dialogue contains:

| Field | Type | Description |
|---|---|---|
| `system` | string | Teacher's Socratic question or response |
| `user` | string | Student's reply |
| `user_type` | string | Student persona type (e.g., `(1)` through `(6)`) |

### Student Types

The `user_type` field encodes one of six student personas simulating different real-world scenarios (e.g., confused students, students making arithmetic errors, students asking off-topic questions).

---

## Usage

```python
from datasets import load_dataset

ds = load_dataset("ulises-c/SocraTeach_Multi", split="train")
record = ds[0]

print(record["question"])
for dlg_id, turns in record["dialogues"].items():
    print(f"\nDialogue {dlg_id}:")
    for turn in turns:
        print(f"  [{turn['user_type']}] Student: {turn['user'][:60]}")
        print(f"           Teacher: {turn['system'][:60]}")
```

### Training Objective

The paper fine-tunes the teacher model to generate `system` (teacher) responses conditioned on the dialogue history, the current `steps` scaffolding, and the student's `user_type`. Do not discard `steps` or `user_type` — they are required conditioning signals.

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
| Single-turn split (SocraTeach_Single) | https://huggingface.co/datasets/ulises-c/SocraTeach_Single |
| Socratic Teaching collection | https://huggingface.co/collections/ulises-c/socratic-teaching-datasets |
| Upload repository | https://github.com/ulises-c/csen-346 |
