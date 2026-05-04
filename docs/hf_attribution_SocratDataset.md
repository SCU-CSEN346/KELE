# Attribution — SocratDataset

## Original Work

This dataset is derived from the research presented in:

> Peng, Yuan et al. "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models."
> *Findings of the Association for Computational Linguistics: EMNLP 2025.*
> https://aclanthology.org/2025.findings-emnlp.888/

The dataset construction pipeline, SocRule framework (5 stages, 34 strategies), and all dialogue generation prompts are the intellectual property of the KELE research team.

## Source Data

The raw questions come from the **CSQ dataset**:

> Liu et al., 2025 — Chinese elementary-school science questions (CSQ).

## Code and Model

- KELE pipeline and evaluation code: https://github.com/yuanpan1020/KELE
- SocratTeachLLM (fine-tuned model): https://huggingface.co/yuanpan/SocratTeachLLM

## HuggingFace Upload

The original SocratDataset was distributed as a JSON file in the KELE GitHub repository and was not published directly on HuggingFace. This HuggingFace upload was created by **Ulises Chavarria** to make the dataset more accessible for the research community.

- Uploader: Ulises Chavarria
- Context: CSEN 346 (Natural Language Processing), Santa Clara University
- Upload repository: https://github.com/ulises-c/csen-346

## Model Trained on This Dataset

- Original fine-tuned model: https://huggingface.co/yuanpan/SocratTeachLLM
- Copy with expanded README: https://huggingface.co/ulises-c/SocratTeachLLM

## English Translation

An English translation of this dataset is available at:
https://huggingface.co/datasets/ulises-c/SocratDataset-EN

## License

This dataset is shared under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license, consistent with open academic research. Use of this dataset should cite the original KELE paper.
