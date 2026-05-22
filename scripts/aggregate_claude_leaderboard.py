"""Aggregate all Claude n=50 experiment results into a single leaderboard.

Reads metrics_summary.json from every results dir matching the Claude experiment
naming patterns + the locked Gemma headline reference, and prints a sorted
comparison table by composite score (state + 0.5 * R-1).

Run:  uv run python scripts/aggregate_claude_leaderboard.py
"""

import json
from pathlib import Path


def load_metrics(path: Path) -> dict | None:
    """Return metrics dict or None if file missing/malformed."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def composite(state_acc: float, rouge1: float) -> float:
    return state_acc + 0.5 * rouge1


def row(label: str, metrics: dict | None) -> tuple:
    if metrics is None:
        return (label, "—", "—", "—", "—", "—", "—")
    state = metrics["state_accuracy"]["overall"]
    r1 = metrics["rouge1"]
    r2 = metrics["rouge2"]
    b4 = metrics["bleu4"]
    n_turns = metrics["n_turns"]
    comp = composite(state, r1)
    return (label, f"{state:.2f}", f"{r1:.2f}", f"{r2:.2f}", f"{b4:.2f}", f"{comp:.2f}", str(n_turns))


def main() -> None:
    results = Path("results")

    # All architectures × {sonnet, opus} at n=50
    runs = [
        # Reference: locked Gemma headline at n=50 (BERT + Gemma + 10-shot, no top-3)
        ("Gemma 4 31B + BERT + 10-shot (locked, ref)", None,  # filled by hardcoded baseline below
            {"state_accuracy": {"overall": 51.06}, "rouge1": 38.53, "rouge2": 16.93, "bleu4": 9.68, "n_turns": 281}),

        # B: Claude as TEACHER, raw (no exemplars, no top-3)
        ("Sonnet 4.6 (TEACHER, raw)",
            results / "bert-claude-sonnet-raw-n50/metrics_summary.json", None),
        ("Opus 4.6 (TEACHER, raw)",
            results / "bert-claude-opus-raw-n50/metrics_summary.json", None),

        # C: Claude as TEACHER, 10-shot only (no top-3)
        ("Sonnet 4.6 (TEACHER, 10-shot)",
            results / "bert-claude-sonnet-fewshot10-n50/metrics_summary.json", None),
        ("Opus 4.6 (TEACHER, 10-shot)",
            results / "bert-claude-opus-fewshot10-n50/metrics_summary.json", None),

        # Already-completed: Claude as TEACHER, 10-shot + top-3 composed
        ("Sonnet 4.6 (TEACHER, 10-shot + top-3)",
            results / "bert-consultant-fewshot10-claude-sonnet-n50/metrics_summary.json", None),
        ("Opus 4.6 (TEACHER, 10-shot + top-3)",
            results / "bert-consultant-fewshot10-claude-opus-n50/metrics_summary.json", None),

        # A: Claude as CONSULTANT + SocratTeachLLM teacher (literal GPT-4o baseline mirror)
        ("Sonnet 4.6 (CONSULTANT, SocratTeachLLM teacher)",
            results / "claude-sonnet-consultant-socratteachllm-n50/metrics_summary.json", None),
        ("Opus 4.6 (CONSULTANT, SocratTeachLLM teacher)",
            results / "claude-opus-consultant-socratteachllm-n50/metrics_summary.json", None),
    ]

    print()
    print("=" * 110)
    print(f"  {'Configuration':<52} {'State':>7} {'R-1':>7} {'R-2':>7} {'BLEU-4':>7} {'Composite':>10} {'Turns':>6}")
    print("=" * 110)

    rows_with_comp: list[tuple] = []
    for label, path, override in runs:
        m = override if override is not None else load_metrics(path) if path else None
        r = row(label, m)
        rows_with_comp.append(r)

    # Sort by composite descending (treating "—" as last)
    def sort_key(r):
        try:
            return -float(r[5])
        except (ValueError, TypeError):
            return float("inf")
    rows_with_comp.sort(key=sort_key)

    for r in rows_with_comp:
        print(f"  {r[0]:<52} {r[1]:>7} {r[2]:>7} {r[3]:>7} {r[4]:>7} {r[5]:>10} {r[6]:>6}")
    print("=" * 110)
    print()

    # Highlight deltas vs locked headline (70.33 composite)
    BASELINE = 70.33
    print(f"Composite Δ vs locked Gemma headline ({BASELINE}):")
    for r in rows_with_comp:
        if r[5] == "—":
            continue
        delta = float(r[5]) - BASELINE
        sign = "+" if delta >= 0 else ""
        marker = "↑ ABOVE" if delta > 0 else "↓ below" if delta < 0 else "= tie"
        print(f"  {sign}{delta:.2f}  {marker:8}  {r[0]}")
    print()


if __name__ == "__main__":
    main()
