#!/usr/bin/env python3
"""Publish a trained state-classifier checkpoint to the HuggingFace Hub.

Generates a model card from the run's test_eval.json, then creates the repo and
uploads results/<dir>/final/ (the merged HF SeqClassification artifact). Adapter
weights are already merged at train time, so the upload is a standard checkpoint
that loads via AutoModelForSequenceClassification.from_pretrained(<repo>).

    uv run python scripts/publish_to_hf.py \
        --results-dir results/state-clf-qwen3.5-0.8b-lora-wandb \
        --repo ulises-c/socrates-state-classifier-qwen3.5-lora \
        --base-model Qwen/Qwen3.5-0.8B-Base \
        --dry-run            # list the file inventory without uploading

Drop --dry-run for the real push. Repos are created private by default; pass
--public to create a public repo (or flip visibility in the web UI afterwards).
Requires `hf auth login` (write scope) or an HF_TOKEN env var.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Files that live in results/<dir>/final/ but should not ship to the Hub.
IGNORE_PATTERNS = ["checkpoints/*", "*.pt", "train.log", "wandb/*", "optimizer.pt"]

# Static doc blocks shared by the model card and ATTRIBUTION.md. Kept as plain
# (non-f) strings so the bibtex braces stay literal.
_RELATED_RESOURCES = """\
## Related Resources

| Resource | Link |
|---|---|
| KELE paper (EMNLP 2025 Findings) | https://aclanthology.org/2025.findings-emnlp.888/ |
| KELE GitHub repository | https://github.com/yuanpan1020/KELE |
| Base model — Qwen3.5-0.8B-Base | https://huggingface.co/Qwen/Qwen3.5-0.8B-Base |
| SocratTeachLLM (original) | https://huggingface.co/yuanpan/SocratTeachLLM |
| Training dataset — SocratDataset | https://huggingface.co/datasets/ulises-c/SocratDataset |
| Clean-probe synthetic eval set | https://huggingface.co/datasets/ulises-c/SocratDataset-SYNTHETIC |
| Training + evaluation code | https://github.com/ulises-c/csen-346 |
"""

_CITATIONS = """\
```bibtex
@inproceedings{peng-etal-2025-kele,
  title     = {{KELE}: A Multi-Agent Framework for Structured {S}ocratic Teaching with Large Language Models},
  author    = {Peng, Yuan and others},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025},
  url       = {https://aclanthology.org/2025.findings-emnlp.888/}
}
```

```bibtex
@misc{chavarria-khan-2026-socrates-state-classifier,
  title  = {Socrates State Classifier ({Q}wen3.5-0.8{B} {L}o{RA}): A Socratic Dialogue State Classifier for the {KELE} Pipeline},
  author = {Chavarria, Ulises and Khan, Maximilian},
  year   = {2026},
  url    = {https://huggingface.co/ulises-c/socrates-state-classifier-qwen3.5-lora}
}
```
"""


def _weighted_stage_acc(eval_data: dict) -> float:
    """Turn-weighted mean of the 5 per-stage accuracies."""
    per_stage = eval_data["test_stage_accuracy_via_state_pred"]
    per_state_n = eval_data["test_per_state_n"]
    stage_n = {s: 0 for s in "abcde"}
    for state, n in per_state_n.items():
        stage_n[state[0]] += n
    total = sum(stage_n.values())
    if not total:
        return 0.0
    return sum(per_stage[s] * stage_n[s] for s in "abcde") / total


def build_model_card(repo: str, base_model: str, eval_data: dict, training_cmd: str) -> str:
    slug = repo.split("/")[-1]
    state_acc = eval_data["test_state_accuracy"] * 100
    stage_acc = _weighted_stage_acc(eval_data) * 100
    per_stage = eval_data["test_stage_accuracy_via_state_pred"]
    n_turns = eval_data["n_test_turns"]
    n_states = len(eval_data["test_per_state_accuracy"])
    n_active = sum(1 for n in eval_data["test_per_state_n"].values() if n > 0)
    stage_rows = "\n".join(f"| {s} | {per_stage[s] * 100:.2f}% |" for s in "abcde")

    head = f"""---
language: zh
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: {base_model}
datasets:
  - ulises-c/SocratDataset
tags:
  - text-classification
  - chinese
  - education
  - socratic-teaching
  - state-classification
  - lora
  - KELE
metrics:
  - accuracy
model-index:
  - name: {slug}
    results:
      - task:
          type: text-classification
          name: Socratic dialogue state classification
        dataset:
          type: ulises-c/SocratDataset
          name: SocratDataset (test split)
        metrics:
          - type: accuracy
            value: {state_acc:.2f}
            name: state_accuracy
          - type: accuracy
            value: {stage_acc:.2f}
            name: stage_accuracy
---

# {slug}

**A Chinese-text Socratic-teaching state classifier — a fast, deterministic drop-in
for the LLM-consultant call in the [KELE](https://aclanthology.org/2025.findings-emnlp.888/)
pipeline.**

Given a student/teacher dialogue turn, it predicts the active pedagogical **state**
(one of the SocRule strategies, e.g. `c16`, `d33`) that the teacher should occupy
next. In the KELE reproduction study it replaces a per-turn LLM consultant call
with a single forward pass, so a full `kele.py` evaluation runs without an external
consultant model. This is a LoRA fine-tune of
[`{base_model}`](https://huggingface.co/{base_model}) with the adapter **merged into
the base weights** — the repo is a plain `AutoModelForSequenceClassification`
checkpoint, no PEFT required to load it.

> Built as part of the KELE reproduction and extension study — CSEN 346 (Natural
> Language Processing), Santa Clara University, 2026.

---

## Model Summary

| Property | Value |
|---|---|
| Base model | [`{base_model}`](https://huggingface.co/{base_model}) (Apache-2.0) |
| Task | Single-turn dialogue state classification |
| Language | Chinese (Simplified) |
| Classes | {n_states} state labels (`a0`–`e34`); {n_active} active in the test set (`a0` is unused → 34 SocRule strategies across 5 stages) |
| Method | LoRA (r=8, α=16, auto-selected target modules), merged |
| Trainable params | ~2.05M (0.24% of 855M) |
| Precision | bf16 autocast over fp32 master weights |
| Framework | SocRule (KELE, Peng et al. 2025) |
| Training data | [SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset) train split |

---

## State Taxonomy

The label space follows the SocRule schema — 34 teaching strategies grouped into
five monotonic pedagogical stages (`a` → `b` → `c` → `d` → `e`):

| Stage | Code range | Description |
|---|---|---|
| a — Initiation | a1 | Student poses the question |
| b — Concept Probing | b2–b7 | Teacher probes prior knowledge |
| c — Inductive Reasoning | c8–c29 | Core teaching stage; may repeat |
| d — Answer Derivation | d30–d33 | Guide the student to the answer |
| e — Summary | e34 | Teacher summarises; dialogue ends |

(`a0` is a reserved slot with no examples in the corpus.)

---

## Evaluation

Held-out **test split, {n_turns} turns** (the SocratDataset test split — separate from
the 5% in-domain hold-out used during training):

- **Overall state accuracy:** {state_acc:.2f}%
- **Turn-weighted stage accuracy:** {stage_acc:.2f}%

| Stage | Stage accuracy (via state prediction) |
|---|---|
{stage_rows}

**Per-state caveat.** Accuracy is strong on well-represented states and drops
sharply on the long tail of rare states (those with < 30 test turns), which the
classifier rarely emits — a class-imbalance artifact of SocratDataset's natural
state distribution, not of the architecture. Full per-state numbers are in
`eval_results.json`. For a tail-balanced variant, see
`scripts/train_state_classifier_34way_balanced.py` in the project repo.

---

## How to Use

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForSequenceClassification.from_pretrained("{repo}")

logits = model(**tok("学生: 什么是化学键？", return_tensors="pt")).logits
pred = model.config.id2label[int(logits.argmax(-1))]   # e.g. "b4"
```

As the consultant inside a KELE evaluation:

```bash
uv run python kele.py --bert-consultant {repo}   # ... + your usual eval flags
```

---

## Training Procedure

```bash
{training_cmd}
```

- **Data:** SocratDataset train split (77,258 labeled turns), with 5% held out for
  in-domain eval (≈73.4K train / 3.9K eval); seed=42; 5 epochs.
- **Hardware / speed:** trained on a single NVIDIA RTX 4000 Ada (20 GB) in ~3.3 h.
  Qwen3.5-0.8B is 75% gated-DeltaNet linear attention, so
  [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) is
  installed to engage its Triton GDN kernels (~2.8× over the pure-PyTorch fallback);
  combined with bf16 autocast and no gradient-checkpointing (the run fits in
  ~12.6 GB) this is ~4× faster than the original fp32 config.
- The full training loop is `scripts/train_state_classifier_34way.py` in the
  [project repo](https://github.com/ulises-c/csen-346).

---

## Limitations

- **Chinese-only**, single-domain (elementary-school science / KELE SocratDataset);
  no out-of-distribution or cross-lingual validation.
- **Long-tail blind spots:** rare states are under-predicted (see the per-state
  caveat above).
- Trained and evaluated on SocratDataset, which has documented benchmark
  contamination for *generative* SocratTeachLLM; this is a discriminative
  state classifier, but the same single-corpus caveat applies.

---

## Citation

If you use this model, cite the original KELE paper and this checkpoint:

""" + _CITATIONS + "\n---\n\n" + _RELATED_RESOURCES + """
---

## License

Apache-2.0, inherited from the [`{base_model}`](https://huggingface.co/{base_model})
base. Use must also cite the KELE paper, whose SocRule schema defines the label space.
""".replace("{base_model}", base_model)

    return head


def build_attribution(repo: str, base_model: str, training_cmd: str) -> str:
    prose = f"""# Attribution — {repo.split("/")[-1]}

## Original Work

The pedagogical state schema this model predicts (the SocRule framework — 5 stages,
34 strategies) is introduced in the KELE paper:

> Peng, Yuan et al. "KELE: A Multi-Agent Framework for Structured Socratic Teaching
> with Large Language Models." *Findings of the Association for Computational
> Linguistics: EMNLP 2025.* https://aclanthology.org/2025.findings-emnlp.888/

The SocRule state-transition specification is the intellectual property of the KELE
research team. This classifier learns to predict those states; the label taxonomy is
derived directly from that specification.

## Base Model

This is a LoRA fine-tune of
[`{base_model}`](https://huggingface.co/{base_model}) (Alibaba Qwen team),
distributed under **Apache-2.0**. The adapter is merged into the base weights, so the
published artifact is a standard `AutoModelForSequenceClassification` checkpoint.

## Training Data

Trained on the **train split of [SocratDataset](https://huggingface.co/datasets/ulises-c/SocratDataset)**
(77,258 labeled dialogue turns; Chinese elementary-school science), with 5% held out
for in-domain evaluation. SocratDataset originates with the KELE project; this model
adds no new training data of its own.

## Training

- **Authors:** Ulises Chavarria, Maximilian Khan
- **Context:** CSEN 346 (Natural Language Processing), Santa Clara University, 2026
- **Code:** https://github.com/ulises-c/csen-346/blob/main/scripts/train_state_classifier_34way.py
- **Hardware:** single NVIDIA RTX 4000 Ada (20 GB), ~3.3 h
- **Command:**

```bash
{training_cmd}
```

[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) is used
to engage Qwen3.5's Triton gated-delta-rule kernels (its 18 linear-attention layers
are 75% of the model).

## Purpose

This classifier is a fast, deterministic replacement for the per-turn LLM-consultant
call in the KELE pipeline: instead of querying a large model for the next teaching
state, a `kele.py` evaluation can load this checkpoint and get the state in a single
forward pass.

## How to Cite

Cite both the KELE paper and this checkpoint:

""" + _CITATIONS + "\n" + _RELATED_RESOURCES + """
## License

Released under **Apache-2.0**, inherited from the base model. Use of this model must
cite the original KELE paper, whose SocRule schema defines the predicted label space.
"""
    return prose


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--repo", required=True, help="HF repo id, e.g. ulises-c/<slug>")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--public", action="store_true", help="Create public (default: private)")
    p.add_argument("--dry-run", action="store_true", help="List the upload inventory, don't push")
    p.add_argument(
        "--training-cmd",
        default=(
            "uv run python scripts/train_state_classifier_34way.py "
            "--model-id Qwen/Qwen3.5-0.8B-Base --lora --lora-r 8 --lora-alpha 16 "
            "--batch_size 8 --bf16-autocast"
        ),
        help="Training command echoed into the model card",
    )
    args = p.parse_args()

    final_dir = args.results_dir / "final"
    if not final_dir.is_dir():
        raise SystemExit(f"{final_dir} does not exist — training not finished?")
    eval_path = args.results_dir / "test_eval.json"
    if not eval_path.is_file():
        raise SystemExit(f"{eval_path} missing — cannot build the model card.")

    eval_data = json.loads(eval_path.read_text())
    card = build_model_card(args.repo, args.base_model, eval_data, args.training_cmd)
    (final_dir / "README.md").write_text(card, encoding="utf-8")
    (final_dir / "ATTRIBUTION.md").write_text(
        build_attribution(args.repo, args.base_model, args.training_cmd), encoding="utf-8"
    )
    (final_dir / "eval_results.json").write_text(
        json.dumps(eval_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    inventory = sorted(f.name for f in final_dir.iterdir() if f.is_file())
    print(f"Repo:      {args.repo}  ({'public' if args.public else 'private'})")
    print(f"Source:    {final_dir}")
    print(f"State acc: {eval_data['test_state_accuracy'] * 100:.2f}%")
    print("Files:")
    for name in inventory:
        print(f"  - {name}")

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=not args.public)
    api.upload_folder(
        folder_path=str(final_dir),
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"\nPublished → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
