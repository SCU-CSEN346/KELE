# Attribution — ulises-c/SocratTeachLLM

## Original Model

This is a copy of the model published by the KELE research team:

> **Original:** [yuanpan/SocratTeachLLM](https://huggingface.co/yuanpan/SocratTeachLLM)

The model weights, LoRA fine-tuning, and all training methodology are the intellectual property of the KELE research team as described in:

> Peng, Yuan et al. "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models."
> *Findings of the Association for Computational Linguistics: EMNLP 2025.*
> https://aclanthology.org/2025.findings-emnlp.888/

## Base Model

Fine-tuned from [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) (GLM4-9B) by Zhipu AI.

## Training Data

Trained on **SocratDataset** — 6,803 multi-turn Socratic dialogues in Chinese covering elementary school science, constructed using the KELE B.3/B.4 pipeline from the CSQ dataset (Liu et al., 2025).

- Chinese dataset: https://huggingface.co/datasets/ulises-c/SocratDataset
- English translation: https://huggingface.co/datasets/ulises-c/SocratDataset-EN
- Original distribution (JSON): https://github.com/yuanpan1020/KELE

## This Copy

**Copied by:** Ulises Chavarria
**Context:** CSEN 346 (Natural Language Processing), Santa Clara University
**Reason:** The original `yuanpan/SocratTeachLLM` had a minimal README. This copy adds a detailed README with architecture specs, usage examples, the full Table 1 benchmark results from the paper, and cross-references to the associated datasets and evaluation code. No weights were modified.

Repository for downstream evaluation and extension work:
https://github.com/ulises-c/csen-346

## License

Apache 2.0 — consistent with the original model and the GLM4-9B-Chat base model.
