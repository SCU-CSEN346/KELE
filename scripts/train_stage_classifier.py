#!/usr/bin/env python3
"""Train a Chinese-BERT 5-stage classifier as a drop-in KELE consultant.

Approach:
1. Build (history_text, stage_label) pairs from SocratDataset train split
2. Fine-tune `BAAI/bge-small-zh-v1.5` (24M params, fast on 5090) with a
   5-class classification head
3. Save the checkpoint to results/stage_classifier_v1/

Why a 5-stage (not 34-state) classifier first:
- Easier to converge, faster to validate the pipeline
- Stage classification is the hardest sub-problem per the gradient analysis
  (stage c at 4-17% across baselines)
- The within-stage state can stay LLM-driven for now; this only replaces
  the stage decision

Usage:
  uv run python scripts/train_stage_classifier.py --epochs 3 --batch_size 32

Hardware: requires a GPU but NOT a serving model — must be run when no
llama-server is running.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Disable HF telemetry / progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

STAGES = ["a", "b", "c", "d", "e"]
STAGE_TO_LABEL = {s: i for i, s in enumerate(STAGES)}
MODEL_ID = "BAAI/bge-small-zh-v1.5"
OUT_DIR = Path("results/stage_classifier_v1")


def build_examples() -> list[dict]:
    """Build (text, label) examples from SocratDataset train split.

    Each turn becomes one example whose `text` is the formatted dialogue
    history up to (not including) the consultant's classification of that
    turn, and whose label is the ground-truth stage of that turn.
    """
    from src.project.kele import load_dataset

    train = load_dataset(split="train")
    examples: list[dict] = []
    for dlg in train:
        history_lines: list[str] = []
        for turn in dlg.get("dialogue", []):
            # The classifier sees: prior turns + current student input
            student = turn.get("student", "").strip()
            state = turn.get("state", "")
            if not state or state[0] not in STAGE_TO_LABEL:
                continue
            current_input = f"学生: {student}"
            history_text = "\n".join(history_lines + [current_input])
            examples.append({
                "text": history_text[-4000:],  # truncate history to last 4K chars
                "label": STAGE_TO_LABEL[state[0]],
            })
            # Append this turn to the history for the next iteration
            history_lines.append(f"学生: {student}")
            teacher = turn.get("teacher", "").strip()
            if teacher:
                history_lines.append(f"老师: {teacher}")

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--eval_split_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} tokenizer + model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(STAGES),
        id2label={i: s for s, i in STAGE_TO_LABEL.items()},
        label2id=STAGE_TO_LABEL,
    )

    print("Building training examples from SocratDataset train split ...")
    examples = build_examples()
    print(f"  Total examples: {len(examples)}")
    label_counts = {s: 0 for s in STAGES}
    for e in examples:
        label_counts[STAGES[e["label"]]] += 1
    print(f"  Label counts: {label_counts}")

    # Hold out a small eval slice
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(examples))
    n_eval = int(len(examples) * args.eval_split_frac)
    eval_examples = [examples[i] for i in indices[:n_eval]]
    train_examples = [examples[i] for i in indices[n_eval:]]
    print(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    train_ds = Dataset.from_list(train_examples).map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = Dataset.from_list(eval_examples).map(tokenize, batched=True, remove_columns=["text"])

    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        labels = pred.label_ids
        acc = float((preds == labels).mean())
        # per-stage acc
        per_stage = {}
        for i, s in enumerate(STAGES):
            mask = labels == i
            if mask.sum() > 0:
                per_stage[f"stage_{s}_acc"] = float((preds[mask] == labels[mask]).mean())
            else:
                per_stage[f"stage_{s}_acc"] = 0.0
        return {"acc": acc, **per_stage}

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        report_to=[],
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    # transformers 5.x replaced tokenizer= with processing_class=
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nTraining ...")
    trainer.train()

    print("\nFinal eval:")
    final_metrics = trainer.evaluate()
    print(json.dumps(final_metrics, indent=2))

    print(f"\nSaving to {OUT_DIR} ...")
    trainer.save_model(str(OUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUT_DIR / "final"))
    with (OUT_DIR / "training_metrics.json").open("w") as f:
        json.dump(final_metrics, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
