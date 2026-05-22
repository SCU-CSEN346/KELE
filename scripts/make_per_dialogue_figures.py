#!/usr/bin/env python3
"""Per-dialogue metric distributions: BLEU/ROUGE vs dialogue length, etc.

Computes per-dialogue ROUGE-1 by extracting teacher_response and
ground_truth_teacher from each dialogue file, then scoring with the same
char-tokenized scorer as `src.project.metrics`.

Figures produced:
- bleu_vs_length.{pdf,png}    — per-dialogue R-1 scatter vs turn count
- per_dialogue_r1_dist.{pdf,png} — box plot of per-dialogue R-1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer

RESULTS = Path("results")
FIGS = Path("docs/figures")
FIGS.mkdir(parents=True, exist_ok=True)


def _char_tokenize(s: str) -> list[str]:
    return list(s)


class CharRouge:
    def __init__(self):
        # Reuse the same tokenizer pattern as metrics.py
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=False,
            tokenizer=type("T", (), {"tokenize": staticmethod(_char_tokenize)})(),
        )

    def score_dialogue(self, dialogue: list[dict]) -> dict:
        rs = {"rouge1": [], "rouge2": [], "rougeL": []}
        for turn in dialogue:
            pred = turn.get("teacher_response", "")
            ref = turn.get("ground_truth_teacher", "")
            if not pred or not ref:
                continue
            s = self.scorer.score(ref, pred)
            for k in rs:
                rs[k].append(s[k].fmeasure)
        return {k: float(np.mean(v) * 100) if v else 0.0 for k, v in rs.items()}


def per_dialogue_metrics(results_dir: str) -> list[dict]:
    p = RESULTS / results_dir / "dialogues"
    if not p.exists():
        return []
    cr = CharRouge()
    out = []
    for f in sorted(p.glob("*.json")):
        d = json.loads(f.read_text())
        dl = d.get("dialogue", [])
        n_turns = len(dl)
        rouge = cr.score_dialogue(dl)
        out.append({"id": d.get("id"), "n_turns": n_turns, **rouge})
    return out


def fig_r1_vs_length(data: list[dict], outname: str, title: str) -> None:
    if not data:
        return
    lengths = [d["n_turns"] for d in data]
    r1 = [d["rouge1"] for d in data]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lengths, r1, alpha=0.5, s=20, color="steelblue")
    # Add trend line
    z = np.polyfit(lengths, r1, 1)
    p = np.poly1d(z)
    xs = sorted(set(lengths))
    ax.plot(xs, p(xs), "--", color="red", label=f"slope = {z[0]:.2f}")
    ax.set_xlabel("Turns per dialogue")
    ax.set_ylabel("Per-dialogue ROUGE-1 (char-tokenized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / f"{outname}.pdf", bbox_inches="tight")
    plt.savefig(FIGS / f"{outname}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {outname}")


def fig_r1_distribution(datasets: dict[str, list[dict]], outname: str, title: str) -> None:
    if not datasets:
        return
    valid_names = [n for n, d in datasets.items() if d]
    r1_lists = [[x["rouge1"] for x in datasets[n]] for n in valid_names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(r1_lists, tick_labels=valid_names, showmeans=True)
    ax.set_ylabel("Per-dialogue ROUGE-1")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS / f"{outname}.pdf", bbox_inches="tight")
    plt.savefig(FIGS / f"{outname}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✓ {outname}")


def main() -> None:
    print("Computing per-dialogue ROUGE for runs ...")
    runs = {
        "A3B locked full": "qwen35b-a3b-local-unified",
        "GPT-4o baseline": "baseline",
        "A3B + 3-shot mini": "qwen35b-a3b-local-mini-unified-fewshot",
        "A3B + 10-shot mini": "qwen35b-a3b-local-mini-unified-fewshot10",
        "A3B mini (locked)": "qwen35b-a3b-local-mini-unified",
    }
    all_data: dict[str, list[dict]] = {}
    for label, dirname in runs.items():
        print(f"  {label} → {dirname}")
        data = per_dialogue_metrics(dirname)
        all_data[label] = data
        if data:
            r1s = [d["rouge1"] for d in data]
            print(
                f"    n={len(data)}, R-1 mean={np.mean(r1s):.2f}, std={np.std(r1s):.2f}, range=[{min(r1s):.1f}, {max(r1s):.1f}]"
            )

    # Scatter: R-1 vs length, on the big-n A3B locked full
    if all_data.get("A3B locked full"):
        fig_r1_vs_length(
            all_data["A3B locked full"],
            "r1_vs_length_a3b_full",
            "Per-dialogue ROUGE-1 vs turn count — A3B locked full (n=681)",
        )
    if all_data.get("GPT-4o baseline"):
        fig_r1_vs_length(
            all_data["GPT-4o baseline"],
            "r1_vs_length_baseline",
            "Per-dialogue ROUGE-1 vs turn count — GPT-4o baseline (n=681)",
        )

    # Box plot comparison
    fig_r1_distribution(
        all_data, "per_dialogue_r1_dist", "Per-dialogue ROUGE-1 distribution across systems"
    )


if __name__ == "__main__":
    main()
