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
    state_acc = eval_data["test_state_accuracy"] * 100
    stage_acc = _weighted_stage_acc(eval_data) * 100
    per_stage = eval_data["test_stage_accuracy_via_state_pred"]
    n_turns = eval_data["n_test_turns"]
    stage_rows = "\n".join(f"| {s} | {per_stage[s] * 100:.2f}% |" for s in "abcde")
    return f"""---
language: zh
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
base_model: {base_model}
tags:
  - text-classification
  - chinese
  - education
  - socratic-teaching
  - state-classification
  - KELE
metrics:
  - accuracy
model-index:
  - name: {repo.split("/")[-1]}
    results:
      - task:
          type: text-classification
          name: 34-state Socratic dialogue state classification
        metrics:
          - type: accuracy
            value: {state_acc:.2f}
            name: state_accuracy
          - type: accuracy
            value: {stage_acc:.2f}
            name: stage_accuracy
---

# {repo.split("/")[-1]}

A Chinese-text 34-state classifier predicting the active Socratic-teaching state of
a student turn. It is a drop-in replacement for the LLM-consultant call in the KELE
pipeline — load it via `--bert-consultant {repo}` on a `kele.py` evaluation.

LoRA fine-tune of [`{base_model}`](https://huggingface.co/{base_model}); the LoRA
adapter is merged into the base weights, so this repo is a plain HF
SeqClassification checkpoint.

## State taxonomy

34 states across 5 pedagogical stages (a/b/c/d/e), inherited from SocRule.

## Evaluation (test split, {n_turns} turns)

- **Overall state accuracy:** {state_acc:.2f}%
- **Turn-weighted stage accuracy:** {stage_acc:.2f}%

| Stage | Stage accuracy (via state pred) |
|---|---|
{stage_rows}

## Training procedure

```bash
{training_cmd}
```

Trained on the SocratDataset train split (~42K labeled turns) with a 5% in-domain
eval hold-out, seed=42. See `scripts/train_state_classifier_34way.py` in the
project repo for the full training loop.

## How to use

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForSequenceClassification.from_pretrained("{repo}")
logits = model(**tok("学生: 什么是化学键？", return_tensors="pt")).logits
pred = model.config.id2label[int(logits.argmax(-1))]
```

## Limitations

Chinese-only training data; school-chemistry domain; single dataset; no
out-of-distribution validation.
"""


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
            "--batch_size 4 --gradient-checkpointing"
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
