#!/usr/bin/env python3
"""Evaluate the trained stage classifier on the actual SocratDataset test split.

The model was trained with 5% held out from the *train* split for in-distribution
eval; this script evaluates on the held-out *test* split (the same 681
dialogues used for every other system in the paper) for apples-to-apples
comparison with the LLM consultants.

Outputs:
  results/stage_classifier_v1/test_eval.json
  results/stage_classifier_v1/test_confusion.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CKPT = Path("results/stage_classifier_v1/final")
OUT = Path("results/stage_classifier_v1")
STAGES = ["a", "b", "c", "d", "e"]
STAGE_TO_LABEL = {s: i for i, s in enumerate(STAGES)}


def build_test_examples() -> list[dict]:
    from src.project.kele import load_dataset

    test = load_dataset(split="test")
    examples: list[dict] = []
    for dlg in test:
        history_lines: list[str] = []
        for turn in dlg.get("dialogue", []):
            student = turn.get("student", "").strip()
            state = turn.get("state", "")
            if not state or state[0] not in STAGE_TO_LABEL:
                continue
            current_input = f"学生: {student}"
            history_text = "\n".join(history_lines + [current_input])
            examples.append(
                {
                    "text": history_text[-4000:],
                    "stage": state[0],
                    "label": STAGE_TO_LABEL[state[0]],
                    "dlg_id": dlg.get("id"),
                    "state": state,
                }
            )
            history_lines.append(f"学生: {student}")
            teacher = turn.get("teacher", "").strip()
            if teacher:
                history_lines.append(f"老师: {teacher}")
    return examples


def main() -> None:
    print(f"Loading {CKPT} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT))
    model = AutoModelForSequenceClassification.from_pretrained(str(CKPT))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    print("Building test examples from SocratDataset test split ...")
    examples = build_test_examples()
    print(f"  {len(examples)} test turns")

    # Inference in batches
    batch_size = 64
    all_preds: list[int] = []
    all_labels = [e["label"] for e in examples]

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            texts = [e["text"] for e in batch]
            enc = tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            preds = logits.argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            if (i // batch_size) % 10 == 0:
                print(f"  {min(i + batch_size, len(examples))}/{len(examples)}")

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    overall_acc = float((preds == labels).mean())
    per_stage = {}
    for i, s in enumerate(STAGES):
        mask = labels == i
        if mask.sum() > 0:
            per_stage[s] = {
                "acc": float((preds[mask] == labels[mask]).mean()),
                "n": int(mask.sum()),
            }

    # Confusion matrix (5x5)
    cm = np.zeros((5, 5), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1

    out_metrics = {
        "n_test_turns": len(examples),
        "overall_acc": overall_acc,
        "per_stage": per_stage,
        "confusion_matrix": cm.tolist(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_eval.json").write_text(json.dumps(out_metrics, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}/test_eval.json")

    # Print summary
    print("\n=== BERT classifier on SocratDataset test split ===")
    print(f"  Overall acc: {overall_acc * 100:.2f}% ({len(examples)} turns)")
    for s in STAGES:
        ps = per_stage[s]
        print(f"  Stage {s}: {ps['acc'] * 100:.2f}% (n={ps['n']})")

    print("\nConfusion matrix (rows=gt, cols=pred):")
    print("       " + "  ".join(f"  pred_{s}" for s in STAGES))
    for i, s in enumerate(STAGES):
        row = "  ".join(f"{c:6d}" for c in cm[i])
        print(f"  gt_{s}  {row}")


if __name__ == "__main__":
    main()
