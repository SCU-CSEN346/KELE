# Attribution — SocraticMATH-sol

## Original Work

This dataset was produced by the ECNU-ICALK research group:

> Ding, Yuyang and Hu, Hanglei and Zhou, Jie and Chen, Qin and Jiang, Bo and He, Liang.
> "Boosting Large Language Models with Socratic Method for Conversational Mathematics Teaching."
> *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM '24)*, 2024.
> https://doi.org/10.1145/3627673.3679881

All dataset content, annotation methodology, and Socratic teaching framework are the intellectual property of the original authors. The `_sol` files with prepended solutions (`【解析】:` prefix) are part of the original ECNU-ICALK distribution.

## Source Data

The dataset is derived from real Chinese primary school exam questions, manually annotated with Socratic-style tutoring dialogues. Original data (including `_sol` variants) is distributed via:
- GitHub: https://github.com/ECNU-ICALK/SocraticMath

The source data was NOT published directly on HuggingFace by the original authors.

## HuggingFace Upload

This HuggingFace upload was created by **Ulises Chavarria** to make the dataset more accessible for the research community.

- Uploader: Ulises Chavarria
- Context: CSEN 346 (Natural Language Processing), Santa Clara University
- Upload script: https://github.com/ulises-c/csen-346/blob/main/scripts/upload_socraticmath_to_hf.py
- Upload repository: https://github.com/ulises-c/csen-346

No modifications were made to the dataset content. The upload script reshapes the original `_sol.jsonl` files into HuggingFace Dataset format with `id` and `conversations` fields.

## How to Cite

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

## Related Resources

| Resource | Link |
|---|---|
| Paper (CIKM '24) | https://doi.org/10.1145/3627673.3679881 |
| SocraticMath GitHub | https://github.com/ECNU-ICALK/SocraticMath |
| SocraticLLM model | https://huggingface.co/CogBase-USTC/SocraticLM |
| Base variant (no solutions) | https://huggingface.co/datasets/ulises-c/SocraticMATH |
| Socratic Teaching collection | https://huggingface.co/collections/ulises-c/socratic-teaching-datasets |
| Upload repository | https://github.com/ulises-c/csen-346 |

## License

This dataset is shared under **Creative Commons Attribution Non-Commercial 4.0 International (CC BY-NC 4.0)**. Non-commercial use only. Any use must cite the original CIKM 2024 paper.
