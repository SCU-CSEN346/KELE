#!/usr/bin/env python3
"""Train the full 34-state classifier — the original proposed Improvement #2.

This is the harder version of train_stage_classifier.py: 34 labels instead of 5.
Stage-c alone has 22 within-stage states. The flat 34-way classification is
the most direct replacement for the LLM consultant's state prediction.

Outputs: results/state_classifier_v1/
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
from datasets import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "BAAI/bge-small-zh-v1.5"
OUT_DIR = Path("results/state_classifier_v1")

# Full 34 states per SocRule
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
            examples.append({
                "text": history_text[-4000:],
                "label": STATE_TO_LABEL[state],
                "state": state,
            })
            history_lines.append(f"学生: {student}")
            teacher = turn.get("teacher", "").strip()
            if teacher:
                history_lines.append(f"老师: {teacher}")
    return examples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
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
    print(f"  Train turns: {len(train_examples)}")
    test_examples = build_examples("test")
    print(f"  Test turns:  {len(test_examples)}")

    # Hold out 5% of train as in-domain eval
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_examples))
    n_eval = int(len(train_examples) * 0.05)
    eval_examples_in = [train_examples[i] for i in idx[:n_eval]]
    train_examples = [train_examples[i] for i in idx[n_eval:]]

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    train_ds = Dataset.from_list(train_examples).map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = Dataset.from_list(eval_examples_in).map(tokenize, batched=True, remove_columns=["text"])

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

    # Evaluate on the TEST split (the real comparison)
    print("\nEvaluating on test split ...")
    test_ds = Dataset.from_list(test_examples).map(tokenize, batched=True, remove_columns=["text"])
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    # Per-state and per-stage breakdown on test
    print("Per-state inference for breakdown ...")
    model.eval()
    device = next(model.parameters()).device
    test_texts = [e["text"] for e in test_examples]  # rebuild raw text since we dropped column
    # Rebuild raw text from test_examples (we still have it)
    correct_state = {s: 0 for s in ALL_STATES}
    total_state = {s: 0 for s in ALL_STATES}
    correct_stage = {s: 0 for s in "abcde"}
    total_stage = {s: 0 for s in "abcde"}
    overall_correct = 0
    bs = 64
    for i in range(0, len(test_examples), bs):
        batch = test_examples[i : i + bs]
        # Reload raw text - we need to grab from the original dicts (still have it)
        texts = [e["text"] for e in batch]
        labels = [e["label"] for e in batch]
        # Tokenize
        enc = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            preds = logits.argmax(-1).cpu().tolist()
        for p, lab, e in zip(preds, labels, batch):
            gt_state = ALL_STATES[lab]
            pred_state = ALL_STATES[p]
            total_state[gt_state] += 1
            total_stage[gt_state[0]] += 1
            if p == lab:
                overall_correct += 1
                correct_state[gt_state] += 1
            if pred_state[0] == gt_state[0]:
                correct_stage[gt_state[0]] += 1

    test_overall = overall_correct / len(test_examples)
    per_state_acc = {s: (correct_state[s] / total_state[s] if total_state[s] else 0.0) for s in ALL_STATES}
    per_stage_acc = {s: (correct_stage[s] / total_stage[s] if total_stage[s] else 0.0) for s in "abcde"}
    out = {
        "n_test_turns": len(test_examples),
        "test_state_accuracy": test_overall,
        "test_stage_accuracy_via_state_pred": per_stage_acc,
        "test_per_state_accuracy": per_state_acc,
        "test_per_state_n": total_state,
        **{k: v for k, v in test_metrics.items() if isinstance(v, float)},
    }
    (OUT_DIR / "test_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n=== 34-state classifier on test split ({len(test_examples)} turns) ===")
    print(f"  Overall state accuracy:        {test_overall * 100:.2f}%")
    print(f"  Implied stage accuracy:")
    for s in "abcde":
        print(f"    Stage {s}: {per_stage_acc[s] * 100:.2f}% (n={total_stage[s]})")
    print(f"  Top per-state hits (rate, n) for stages c+d:")
    for s in ALL_STATES:
        if s[0] in "cd" and total_state[s] > 0:
            print(f"    {s}: {per_state_acc[s] * 100:.2f}% (n={total_state[s]})")

    trainer.save_model(str(OUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUT_DIR / "final"))
    print(f"\nSaved to {OUT_DIR}/final")
    print("Done.")


if __name__ == "__main__":
    main()
