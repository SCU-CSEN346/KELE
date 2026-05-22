#!/usr/bin/env python3
"""34-state classifier v2: class-weighted loss + slightly more epochs.

The v1 classifier collapsed on rare within-c states (c8/c10/c11/c13-c15/
c18/c20/c23-c29 each at <11% accuracy with n<45 in test split). This
version computes inverse-frequency class weights from the training set
and applies them to a weighted cross-entropy loss.

Output: results/state_classifier_v2/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "BAAI/bge-small-zh-v1.5"
OUT_DIR = Path("results/state_classifier_v2")
ALL_STATES = (
    ["a0", "a1"]
    + [f"b{i}" for i in range(2, 8)]
    + [f"c{i}" for i in range(8, 30)]
    + [f"d{i}" for i in range(30, 34)]
    + ["e34"]
)
STATE_TO_LABEL = {s: i for i, s in enumerate(ALL_STATES)}


def build_examples(split: str) -> list[dict]:
    from src.project.kele import load_dataset

    data = load_dataset(split=split)
    examples: list[dict] = []
    for dlg in data:
        history_lines: list[str] = []
        for turn in dlg.get("dialogue", []):
            student = turn.get("student", "").strip()
            state = turn.get("state", "")
            if state not in STATE_TO_LABEL:
                continue
            current_input = f"学生: {student}"
            history_text = "\n".join(history_lines + [current_input])
            examples.append(
                {
                    "text": history_text[-4000:],
                    "label": STATE_TO_LABEL[state],
                    "state": state,
                }
            )
            history_lines.append(f"学生: {student}")
            teacher = turn.get("teacher", "").strip()
            if teacher:
                history_lines.append(f"老师: {teacher}")
    return examples


class WeightedTrainer(Trainer):
    """Subclass that applies per-class loss weighting."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=7)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--weight_temp",
        type=float,
        default=0.5,
        help="Power for inverse-freq weights (0=none, 1=fully balanced)",
    )
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(ALL_STATES),
        id2label={i: s for s, i in STATE_TO_LABEL.items()},
        label2id=STATE_TO_LABEL,
    )

    print("Building examples ...")
    train_examples = build_examples("train")
    test_examples = build_examples("test")
    print(f"  Train: {len(train_examples)}, Test: {len(test_examples)}")

    # Compute class weights from train distribution
    counts = np.zeros(len(ALL_STATES))
    for e in train_examples:
        counts[e["label"]] += 1
    # Add Laplace smoothing to avoid 1/0
    counts = counts + 1.0
    # Power-scaled inverse frequency
    inv = 1.0 / (counts**args.weight_temp)
    weights = inv / inv.mean()  # normalize so mean = 1.0
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(
        f"  Class weight range: [{weights.min():.3f}, {weights.max():.3f}]  (temp={args.weight_temp})"
    )
    print(f"  Top-weighted states: {sorted(zip(weights, ALL_STATES), reverse=True)[:5]}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_examples))
    n_eval = int(len(train_examples) * 0.05)
    eval_examples_in = [train_examples[i] for i in idx[:n_eval]]
    train_examples = [train_examples[i] for i in idx[n_eval:]]

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    train_ds = Dataset.from_list(train_examples).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    eval_ds = Dataset.from_list(eval_examples_in).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        labels = pred.label_ids
        return {"acc": float((preds == labels).mean())}

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=200,
        report_to=[],
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("\nTraining ...")
    trainer.train()

    print("\nEvaluating on test split ...")
    model.eval()
    device = next(model.parameters()).device
    correct_state = {s: 0 for s in ALL_STATES}
    total_state = {s: 0 for s in ALL_STATES}
    correct_stage = {s: 0 for s in "abcde"}
    total_stage = {s: 0 for s in "abcde"}
    overall_correct = 0
    bs = 64
    for i in range(0, len(test_examples), bs):
        batch = test_examples[i : i + bs]
        texts = [e["text"] for e in batch]
        labels = [e["label"] for e in batch]
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            preds = logits.argmax(-1).cpu().tolist()
        for p, lab in zip(preds, labels):
            gt = ALL_STATES[lab]
            pred = ALL_STATES[p]
            total_state[gt] += 1
            total_stage[gt[0]] += 1
            if p == lab:
                overall_correct += 1
                correct_state[gt] += 1
            if pred[0] == gt[0]:
                correct_stage[gt[0]] += 1

    test_overall = overall_correct / len(test_examples)
    per_state_acc = {
        s: (correct_state[s] / total_state[s] if total_state[s] else 0.0) for s in ALL_STATES
    }
    per_stage_acc = {
        s: (correct_stage[s] / total_stage[s] if total_stage[s] else 0.0) for s in "abcde"
    }
    out = {
        "weight_temp": args.weight_temp,
        "epochs": args.epochs,
        "n_test_turns": len(test_examples),
        "test_state_accuracy": test_overall,
        "test_stage_accuracy_via_state_pred": per_stage_acc,
        "test_per_state_accuracy": per_state_acc,
        "test_per_state_n": total_state,
    }
    (OUT_DIR / "test_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n=== 34-state v2 (class-weighted) on test split ({len(test_examples)} turns) ===")
    print(f"  Overall state accuracy:        {test_overall * 100:.2f}%")
    print(f"  Implied stage accuracy:")
    for s in "abcde":
        print(f"    Stage {s}: {per_stage_acc[s] * 100:.2f}% (n={total_stage[s]})")
    # Rare-state recovery — show only the rare ones from v1 with non-zero v1 n
    print("  Rare-state recovery (states v1 collapsed):")
    rare = [
        "b3",
        "b7",
        "c8",
        "c10",
        "c11",
        "c13",
        "c14",
        "c15",
        "c17",
        "c18",
        "c20",
        "c21",
        "c23",
        "c24",
        "c25",
        "c26",
        "c27",
        "c28",
        "c29",
        "d30",
        "d31",
        "d32",
    ]
    for s in rare:
        if s in total_state and total_state[s] > 0:
            print(f"    {s}: {per_state_acc[s] * 100:.2f}% (n={total_state[s]})")

    trainer.save_model(str(OUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUT_DIR / "final"))
    print(f"\nSaved to {OUT_DIR}/final")
    print("Done.")


if __name__ == "__main__":
    main()
