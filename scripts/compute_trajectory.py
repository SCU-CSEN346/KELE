"""Compute running metrics at every Nth dialogue checkpoint and save trajectory JSON.

Used to generate the trajectory plot for the A3B fusion-think full run.
Reads dialogue JSONs in numeric ID order and computes incremental ROUGE/BLEU/state-acc
on growing prefixes.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.project.metrics import compute_rouge, compute_bleu, _ZhCharTokenizer  # noqa


def dialog_id(path: Path) -> int:
    return int(path.stem)


def main(dialogues_dir: Path, out_path: Path, every: int) -> None:
    files = sorted(dialogues_dir.glob("*.json"), key=dialog_id)
    print(f"Found {len(files)} dialogue files")

    # Accumulators
    predictions: list[str] = []
    references: list[str] = []
    correct = 0
    total = 0
    stage_correct: dict[str, int] = {}
    stage_total: dict[str, int] = {}

    checkpoints: list[dict] = []

    for i, f in enumerate(files, start=1):
        data = json.loads(f.read_text())
        if "error" in data:
            continue
        for turn in data.get("dialogue", []):
            gt_state = turn.get("ground_truth_state", "")
            pred_state = turn.get("state", "")
            pred_text = turn.get("teacher_response", "")
            ref_text = turn.get("ground_truth_teacher", "")
            if not gt_state:
                continue
            stage = gt_state[0]
            total += 1
            stage_total[stage] = stage_total.get(stage, 0) + 1
            if pred_state == gt_state:
                correct += 1
                stage_correct[stage] = stage_correct.get(stage, 0) + 1
            if pred_text and ref_text:
                predictions.append(pred_text)
                references.append(ref_text)

        # Take a checkpoint every `every` dialogues, plus always at the very end
        if i % every == 0 or i == len(files):
            rouge = compute_rouge(predictions, references) if predictions else {}
            bleu = compute_bleu(predictions, references) if predictions else 0.0

            per_stage: dict[str, float] = {}
            for s in "abcde":
                per_stage[s] = (
                    stage_correct.get(s, 0) / stage_total[s] * 100
                    if stage_total.get(s, 0) > 0
                    else 0.0
                )

            cp = {
                "n_dialogues": i,
                "n_turns": total,
                "state_acc_overall": (correct / total * 100) if total else 0.0,
                "per_stage": per_stage,
                "rouge1": rouge.get("rouge1", 0.0),
                "rouge2": rouge.get("rouge2", 0.0),
                "rougeL": rouge.get("rougeL", 0.0),
                "bleu4": bleu,
            }
            checkpoints.append(cp)
            print(
                f"  dlg={i:>3} turns={total:>4}  "
                f"state_acc={cp['state_acc_overall']:.2f}  "
                f"ROUGE-1={cp['rouge1']:.2f}  BLEU-4={cp['bleu4']:.2f}"
            )

    out_path.write_text(json.dumps(checkpoints, indent=2))
    print(f"\nSaved {len(checkpoints)} checkpoints to {out_path}")


if __name__ == "__main__":
    DIALOGUES = Path("results/qwen35b-a3b-local-unified/dialogues")
    OUT = Path("results/qwen35b-a3b-local-unified/trajectory.json")
    EVERY = 10
    main(DIALOGUES, OUT, EVERY)
