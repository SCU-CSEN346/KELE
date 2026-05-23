#!/usr/bin/env python3
"""Backtest every metrics_summary.json under results/ with stage-balanced macro.

Zero new compute. Reads existing per-stage accuracies, computes:
  - macro          state_accuracy.overall (original, frequency-weighted)
  - stage_balanced (1/5) * Σ p_s for s in {a,b,c,d,e}            (Option A)
  - pedagogical   weighted: a=.10 b=.20 c=.25 d=.20 e=.25         (Option B)
  - freq_inverse  weights ∝ 1/f_s using a/b/c/d/e turn counts     (Option C)
    (when per-stage counts are unavailable, falls back to Option A)

Writes a sortable markdown leaderboard + per-stage breakdown for the top 20.

Usage:
  uv run python scripts/backtest_stage_balanced.py [--min-turns N] [--out PATH]

Default filters smoke runs at n_turns < 50.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

STAGES = ("a", "b", "c", "d", "e")
PEDAGOGICAL_W = {"a": 0.10, "b": 0.20, "c": 0.25, "d": 0.20, "e": 0.25}


def stage_balanced(per_stage: dict[str, float]) -> float | None:
    """Option A: plain mean of per-stage accuracies."""
    vals = [per_stage.get(s) for s in STAGES]
    if any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


def pedagogical(per_stage: dict[str, float]) -> float | None:
    """Option B: hand-picked weights giving closure parity with questioning."""
    vals = [(per_stage.get(s), PEDAGOGICAL_W[s]) for s in STAGES]
    if any(v is None for v, _ in vals):
        return None
    return sum(v * w for v, w in vals)


def freq_inverse(per_stage: dict[str, float], per_stage_counts: dict[str, int] | None) -> float | None:
    """Option C: weights ∝ 1/freq, normalized."""
    if per_stage_counts is None:
        return stage_balanced(per_stage)
    if any(per_stage_counts.get(s, 0) == 0 for s in STAGES):
        return stage_balanced(per_stage)
    inv = {s: 1.0 / per_stage_counts[s] for s in STAGES}
    z = sum(inv.values())
    weights = {s: inv[s] / z for s in STAGES}
    vals = [(per_stage.get(s), weights[s]) for s in STAGES]
    if any(v is None for v, _ in vals):
        return None
    return sum(v * w for v, w in vals)


def per_stage_counts_from_dialogues(cell_dir: Path) -> dict[str, int] | None:
    """If we have dialogues/ under this cell, count turns per stage from gt."""
    ddir = cell_dir / "dialogues"
    if not ddir.is_dir():
        return None
    counts = {s: 0 for s in STAGES}
    for f in ddir.glob("*.json"):
        try:
            dlg = json.loads(f.read_text())
        except Exception:
            continue
        for t in dlg.get("dialogue", []):
            gt = t.get("ground_truth_state")
            if isinstance(gt, str) and gt and gt[0] in counts:
                counts[gt[0]] += 1
    if sum(counts.values()) == 0:
        return None
    return counts


def fmt_pct(v: float | None) -> str:
    return f"{v:5.2f}" if v is not None else "  —  "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-turns", type=int, default=50)
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--top", type=int, default=25, help="Top-N to show per-stage breakdown for")
    args = ap.parse_args()

    rows = []
    skipped = 0
    for mfile in sorted(args.results_root.rglob("metrics_summary.json")):
        try:
            m = json.loads(mfile.read_text())
        except Exception:
            skipped += 1
            continue
        sa = m.get("state_accuracy", {})
        per_stage = sa.get("per_stage", {})
        if not per_stage:
            skipped += 1
            continue
        n_turns = m.get("n_turns") or sa.get("total_turns") or 0
        if n_turns < args.min_turns:
            skipped += 1
            continue
        macro = sa.get("overall")
        if macro is None:
            skipped += 1
            continue
        counts = per_stage_counts_from_dialogues(mfile.parent)
        sb = stage_balanced(per_stage)
        ped = pedagogical(per_stage)
        finv = freq_inverse(per_stage, counts)
        # LLM-judge: read judge_summary.json sibling if present
        judge_score = None
        jfile = mfile.parent / "judge_summary.json"
        if jfile.exists():
            try:
                judge_score = json.loads(jfile.read_text()).get("overall_avg")
            except Exception:
                pass
        rows.append({
            "config": mfile.parent.relative_to(args.results_root).as_posix(),
            "n": n_turns,
            "macro": macro,
            "stage_bal": sb,
            "pedagogical": ped,
            "freq_inv": finv,
            "judge": judge_score,
            "rouge1": m.get("rouge1"),
            "per_stage": per_stage,
            "counts": counts,
        })

    rows_with_sb = [r for r in rows if r["stage_bal"] is not None]
    rows_with_sb.sort(key=lambda r: -r["stage_bal"])
    rows_by_macro = sorted(rows, key=lambda r: -(r["macro"] or 0))
    macro_rank = {r["config"]: i + 1 for i, r in enumerate(rows_by_macro)}

    if args.out is None:
        from datetime import datetime
        args.out = Path("results/_orchestrator_logs") / f"backtest_stage_balanced_{datetime.now():%Y_%m_%d}.md"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Stage-balanced backtest — {args.out.stem.split('_')[-3]}_{args.out.stem.split('_')[-2]}_{args.out.stem.split('_')[-1]}")
    lines.append("")
    lines.append(f"Recomputed from {len(rows)} configs (filtered to n_turns ≥ {args.min_turns}; {skipped} smaller files skipped).")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("- **macro** — frequency-weighted state acc (`Σ correct / Σ turns`). The published headline.")
    lines.append("- **stage_bal** — Option A: `(1/5) × Σ p_s`. ML-standard macro-F1 move; recommended new headline.")
    lines.append("- **pedagogical** — Option B: weights `a=.10 b=.20 c=.25 d=.20 e=.25` giving closure parity with questioning.")
    lines.append("- **freq_inv** — Option C: weights ∝ 1/(per-stage turn count); falls back to stage_bal when counts unavailable.")
    lines.append("- **judge** — LLM-judge `overall_avg` from `judge_summary.json` (Claude Sonnet 4.6 rubric, 0-10 scale); `—` if not judged.")
    lines.append("- **Δrank** — `macro_rank − stage_bal_rank` (positive = moved UP under stage-balanced; negative = moved DOWN).")
    lines.append("")
    n_judged = sum(1 for r in rows_with_sb if r["judge"] is not None)
    lines.append(f"## Leaderboard (sorted by stage_bal; Δrank vs macro; {n_judged}/{len(rows_with_sb)} have judge scores)")
    lines.append("")
    lines.append("| sb# | macro# | Δ | config | n | macro | stage_bal | pedagogical | freq_inv | judge | R-1 |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows_with_sb, 1):
        mrank = macro_rank.get(r["config"], 0)
        drank = mrank - i
        d_str = f"+{drank}" if drank > 0 else (f"{drank}" if drank < 0 else "·")
        judge_str = f"{r['judge']:5.2f}" if r["judge"] is not None else "  —  "
        lines.append(
            f"| {i} | {mrank} | {d_str} | `{r['config']}` | {r['n']} | "
            f"{fmt_pct(r['macro'])} | **{fmt_pct(r['stage_bal'])}** | "
            f"{fmt_pct(r['pedagogical'])} | {fmt_pct(r['freq_inv'])} | "
            f"{judge_str} | {fmt_pct(r['rouge1'])} |"
        )

    big_movers_up = [(i + 1, macro_rank[r["config"]] - (i + 1), r)
                     for i, r in enumerate(rows_with_sb)
                     if (macro_rank[r["config"]] - (i + 1)) >= 3]
    big_movers_dn = [(i + 1, macro_rank[r["config"]] - (i + 1), r)
                     for i, r in enumerate(rows_with_sb)
                     if (macro_rank[r["config"]] - (i + 1)) <= -3]

    if big_movers_up:
        lines.append("")
        lines.append("## Big movers UP (≥3 ranks under stage-balanced)")
        lines.append("")
        lines.append("| Δ | sb# | macro# | config | n | macro → stage_bal | stage_e |")
        lines.append("|---:|---:|---:|---|---:|---|---:|")
        for sb_rank, d, r in sorted(big_movers_up, key=lambda x: -x[1])[:20]:
            stage_e = r["per_stage"].get("e", 0)
            lines.append(
                f"| +{d} | {sb_rank} | {macro_rank[r['config']]} | `{r['config']}` | {r['n']} | "
                f"{r['macro']:.2f} → {r['stage_bal']:.2f} | {stage_e:.1f}% |"
            )

    if big_movers_dn:
        lines.append("")
        lines.append("## Big movers DOWN (≥3 ranks under stage-balanced)")
        lines.append("")
        lines.append("| Δ | sb# | macro# | config | n | macro → stage_bal | stage_c |")
        lines.append("|---:|---:|---:|---|---:|---|---:|")
        for sb_rank, d, r in sorted(big_movers_dn, key=lambda x: x[1])[:20]:
            stage_c = r["per_stage"].get("c", 0)
            lines.append(
                f"| {d} | {sb_rank} | {macro_rank[r['config']]} | `{r['config']}` | {r['n']} | "
                f"{r['macro']:.2f} → {r['stage_bal']:.2f} | {stage_c:.1f}% |"
            )

    lines.append("")
    lines.append(f"## Per-stage breakdown (top {args.top} by stage_bal)")
    lines.append("")
    lines.append("| sb# | config | n | a | b | c | d | e |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows_with_sb[: args.top], 1):
        ps = r["per_stage"]
        lines.append(
            f"| {i} | `{r['config']}` | {r['n']} | "
            f"{fmt_pct(ps.get('a'))} | {fmt_pct(ps.get('b'))} | {fmt_pct(ps.get('c'))} | "
            f"{fmt_pct(ps.get('d'))} | {fmt_pct(ps.get('e'))} |"
        )

    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}")
    print(f"  configs: {len(rows)}  with stage_bal: {len(rows_with_sb)}  skipped: {skipped}")
    print(f"  big movers UP   (≥+3): {len(big_movers_up)}")
    print(f"  big movers DOWN (≤−3): {len(big_movers_dn)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
