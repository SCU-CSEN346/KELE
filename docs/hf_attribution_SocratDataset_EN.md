# Attribution — SocratDataset-EN

## Original Work

SocratDataset-EN is an English translation of SocratDataset, which was produced by the KELE research team:

> Peng, Yuan et al. "KELE: A Multi-Agent Framework for Structured Socratic Teaching with Large Language Models."
> *Findings of the Association for Computational Linguistics: EMNLP 2025.*
> https://aclanthology.org/2025.findings-emnlp.888/

The SocRule framework, dialogue construction methodology, and all original Chinese content are the intellectual property of the KELE research team.

## Source Dataset

- Chinese original: https://huggingface.co/datasets/ulises-c/SocratDataset
- Original distribution (JSON file): https://github.com/yuanpan1020/KELE

The original dataset was NOT published directly on HuggingFace by the KELE authors. The Chinese HuggingFace upload and this English translation were both created by Ulises Chavarria.

## Translation

**Translator:** Ulises Chavarria
**Context:** CSEN 346 (Natural Language Processing), Santa Clara University
**Translation code:** https://github.com/ulises-c/csen-346/blob/main/src/project/translate_dataset.py

### Automated translation (6,794 records)
Translated using **Qwen3.5-9B-UD-Q4_K_XL** (quantized 4-bit, via llama.cpp) running locally on an AMD R9700 GPU. The translation pipeline included JSON validation, retry logic with targeted error messages, split-turn detection and merging, and a post-hoc Chinese character scan.

### Manual translation (9 records)
The following records failed automated translation after 3 retries and were translated manually by the author. Readers with stronger Chinese literacy are encouraged to review them:

IDs: **1328, 1639, 2655, 2766, 3524, 3532, 4162, 6313, 6437**

These can be identified by `translation_meta.model = "claude-sonnet-4-6"`.

## Related Resources

| Resource | Link |
|---|---|
| KELE paper | https://aclanthology.org/2025.findings-emnlp.888/ |
| KELE GitHub | https://github.com/yuanpan1020/KELE |
| SocratTeachLLM model | https://huggingface.co/yuanpan/SocratTeachLLM |
| Original Chinese dataset | https://huggingface.co/datasets/ulises-c/SocratDataset |
| Translation + evaluation code | https://github.com/ulises-c/csen-346 |

## License

Shared under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Use of this dataset must cite the original KELE paper. Credit for the translation should reference this repository and the translator.
