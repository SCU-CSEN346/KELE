"""Master aggregator: collect ALL metrics (surface ROUGE/BLEU, state acc,
semantic R-1, LLM-judge) across every results dir we care about and write
one consolidated leaderboard.

Reads:
  results/<dir>/metrics_summary.json   (surface metrics + state acc)
  results/<dir>/semantic_r1.json       (cosine similarity if present)
  results/<dir>/judge_summary.json     (LLM-judge scores if present)
  results/<dir>/run_config.json        (provenance)

Writes:
  results/master_leaderboard.json
  results/master_leaderboard.md        (human-readable)
"""

from __future__ import annotations

import json
from pathlib import Path

# Map of (display_name, results_dir) pairs to aggregate.
# Order matters for the markdown table — listed by approximate rank by composite.
CONFIGS = [
    # Paper reference (special — no n=50 dir, hardcoded values)
    (
        "GPT-4o + SocratTeachLLM (paper baseline)",
        None,
        {
            "state": 25.94,
            "r1": 44.61,
            "r2": 26.04,
            "rl": 38.02,
            "b4": 19.60,
            "n_turns": 4294,
            "n_dialogues": 681,
        },
    ),
    # Phase 2-Local
    ("Gemma 4 31B + top-3 stack", "results/bert-gemma-composed-top3-n50", None),
    ("Qwen 35B-A3B + top-3 stack", "results/bert-a3b-composed-top3-n50", None),
    (
        "Gemma 4 31B + 10-shot (n=50 ref)",
        None,
        {
            "state": 51.06,
            "r1": 38.53,
            "r2": 16.93,
            "rl": 29.48,
            "b4": 9.68,
            "n_turns": 284,
            "n_dialogues": 50,
        },
    ),
    ("Gemma 4 31B + 10-shot (n=681 LOCKED)", "results/bert-consultant-fewshot10-gemma-full", None),
    ("Qwen 35B-A3B + 10-shot (n=681)", "results/bert-consultant-fewshot10-a3b-full", None),
    # Phase 2-Claude (top-3)
    ("Sonnet 4.6 + BERT + top-3", "results/bert-consultant-fewshot10-claude-sonnet-n50", None),
    ("Opus 4.6 + BERT + top-3", "results/bert-consultant-fewshot10-claude-opus-n50", None),
    # Phase 2-Claude (10-shot only)
    ("Sonnet 4.6 + BERT + 10-shot only", "results/bert-claude-sonnet-fewshot10-n50", None),
    ("Opus 4.6 + BERT + 10-shot only", "results/bert-claude-opus-fewshot10-n50", None),
    # Phase 2-Claude (raw, no exemplars)
    ("Sonnet 4.6 + BERT raw (no exemplars)", "results/bert-claude-sonnet-raw-n50", None),
    ("Opus 4.6 + BERT raw (no exemplars)", "results/bert-claude-opus-raw-n50", None),
    # Phase 3 (full runs, if available)
    ("Opus 4.6 + BERT + top-3 (n=681 Phase 3)", "results/bert-claude-opus-top3-n681", None),
    ("Sonnet 4.6 + BERT + top-3 (n=681 Phase 3)", "results/bert-claude-sonnet-top3-n681", None),
    ("Gemma 4 31B + top-3 (n=200 validation)", "results/bert-gemma-composed-top3-n200", None),
    # Experiment-A: Claude consultant + SocratTeachLLM teacher (the smoking gun)
    (
        "Sonnet 4.6 consultant + SocratTeachLLM (n=50)",
        "results/claude-sonnet-consultant-socratteachllm-n50",
        None,
    ),
    (
        "Sonnet 4.6 consultant + SocratTeachLLM (n=50 clean rerun)",
        "results/claude-sonnet-consultant-socratteachllm-n50-clean",
        None,
    ),
    (
        "Opus 4.6 consultant + SocratTeachLLM (n=50)",
        "results/claude-opus-consultant-socratteachllm-n50",
        None,
    ),
    # Cross-lingual (SocratDataset-EN)
    (
        "Sonnet 4.6 consultant + SocratTeachLLM (EN)",
        "results/claude-sonnet-consultant-socratteachllm-EN-n50",
        None,
    ),
    (
        "Opus 4.6 consultant + SocratTeachLLM (EN)",
        "results/claude-opus-consultant-socratteachllm-EN-n50",
        None,
    ),
    ("Opus 4.6 + BERT + top-3 (EN)", "results/bert-claude-opus-top3-EN-n50", None),
    # ──────────────────────────────────────────────────────────────────────────
    # GLM-4-9B-Chat base-model ablation campaign (2026-05-29, mk/glm-4-test).
    # SocratTeachLLM is a LoRA fine-tune of THUDM/glm-4-9b-chat. Running the
    # base under identical conditions isolates the LoRA contribution from
    # architecture/serving/sample/consultant variance. See memory:
    #   project_glm49b_base_ablation_2026_05_29.md
    # The STL rows below are flagged [contaminated]: per
    # SOCRATTEACHLLM_CONTAMINATION_PROOF.md they include training-set
    # memorization signal that inflates ROUGE/BLEU relative to true pedagogy.
    # The GLM-4-9B-Chat-base rows are the "true STL ceiling" reference —
    # subtract them from the STL rows to get the fine-tune contribution.
    # ──────────────────────────────────────────────────────────────────────────
    (
        "bert-fixed × GLM-4-9B-Chat-base · fewshot10 · n=681 [STL base — true ceiling]",
        "results/bert-fixed-bert-glm49b-chat-fewshot10-n681-fixed",
        None,
    ),
    (
        "qwen3.5 × GLM-4-9B-Chat-base · fewshot10 · n=681 [STL base — true ceiling]",
        "results/t4-bert-glm49b-chat-fewshot10-n681-fixed",
        None,
    ),
    (
        "bert-fixed × SocratTeachLLM · fewshot10 · n=681 [contaminated]",
        "results/bert-fixed-bert-socratteachllm-fewshot10-n681-fixed",
        None,
    ),
    (
        "qwen3.5 × SocratTeachLLM · fewshot10 · n=681 [contaminated]",
        "results/t4-bert-socratteachllm-fewshot10-n681-fixed",
        None,
    ),
]


def load_metrics(results_dir: Path) -> dict | None:
    p = results_dir / "metrics_summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def load_semantic_r1(results_dir: Path) -> float | None:
    p = results_dir / "semantic_r1.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("mean_cosine")
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def load_judge(results_dir: Path) -> dict | None:
    p = results_dir / "judge_summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def collect_one(name: str, results_dir: str | None, hardcoded: dict | None) -> dict:
    row = {"name": name, "available": False}
    if hardcoded:
        row.update(
            {
                "available": True,
                "results_dir": None,
                "state": hardcoded.get("state"),
                "r1": hardcoded.get("r1"),
                "r2": hardcoded.get("r2"),
                "rl": hardcoded.get("rl"),
                "b4": hardcoded.get("b4"),
                "n_turns": hardcoded.get("n_turns"),
                "n_dialogues": hardcoded.get("n_dialogues"),
                "surface_sum": (hardcoded.get("r1") or 0)
                + (hardcoded.get("r2") or 0)
                + (hardcoded.get("b4") or 0),
                "composite_state_r1": (hardcoded.get("state") or 0)
                + 0.5 * (hardcoded.get("r1") or 0),
                "semantic_r1": None,
                "judge_overall": None,
            }
        )
        return row

    if not results_dir:
        return row
    rdir = Path(results_dir)
    m = load_metrics(rdir)
    if m is None:
        return row

    sem = load_semantic_r1(rdir)
    judge = load_judge(rdir)
    state = m["state_accuracy"]["overall"]
    r1 = m["rouge1"]
    r2 = m["rouge2"]
    rl = m.get("rougeL", 0)
    b4 = m["bleu4"]

    row.update(
        {
            "available": True,
            "results_dir": results_dir,
            "state": state,
            "r1": r1,
            "r2": r2,
            "rl": rl,
            "b4": b4,
            "n_turns": m["n_turns"],
            "n_dialogues": m.get("total_dialogues", m["n_turns"] // 6),
            "surface_sum": r1 + r2 + b4,
            "composite_state_r1": state + 0.5 * r1,
            "semantic_r1": sem,
            "judge_overall": (judge or {}).get("overall_avg"),
            "judge_axis_avgs": (judge or {}).get("axis_avgs"),
            "per_stage": m["state_accuracy"]["per_stage"],
        }
    )
    return row


def main():
    rows = [collect_one(n, d, h) for n, d, h in CONFIGS]

    out = {
        "rows": rows,
        "summary": {
            "n_configs_available": sum(1 for r in rows if r.get("available")),
            "n_configs_total": len(rows),
        },
    }

    Path("results/master_leaderboard.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )

    # Markdown table
    md = ["# Master leaderboard\n"]
    md.append(
        f"Configs available: {out['summary']['n_configs_available']}/{out['summary']['n_configs_total']}\n"
    )

    md.append("\n## Sorted by composite (state + 0.5*R-1)\n")
    md.append(
        "| Config | State | R-1 | R-2 | BLEU-4 | Semantic R-1 | LLM-judge | Composite | n_turns |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    avail = [r for r in rows if r.get("available")]
    avail.sort(key=lambda r: r["composite_state_r1"], reverse=True)
    for r in avail:
        sem = f"{r['semantic_r1']:.4f}" if r["semantic_r1"] is not None else "—"
        judge = f"{r['judge_overall']:.2f}" if r["judge_overall"] is not None else "—"
        md.append(
            f"| {r['name']} | {r['state']:.2f} | {r['r1']:.2f} | {r['r2']:.2f} | {r['b4']:.2f} | {sem} | {judge} | {r['composite_state_r1']:.2f} | {r['n_turns']} |"
        )

    md.append("\n## Sorted by surface-form sum (R-1 + R-2 + BLEU-4) — KELE-paper-style\n")
    md.append("| Config | R-1 | R-2 | BLEU-4 | Sum | State | LLM-judge | Semantic |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    avail2 = sorted(avail, key=lambda r: r["surface_sum"], reverse=True)
    for r in avail2:
        sem = f"{r['semantic_r1']:.4f}" if r["semantic_r1"] is not None else "—"
        judge = f"{r['judge_overall']:.2f}" if r["judge_overall"] is not None else "—"
        md.append(
            f"| {r['name']} | {r['r1']:.2f} | {r['r2']:.2f} | {r['b4']:.2f} | {r['surface_sum']:.2f} | {r['state']:.2f} | {judge} | {sem} |"
        )

    md.append("\n## Sorted by LLM-judge (memorization-resistant)\n")
    judged = [r for r in avail if r["judge_overall"] is not None]
    if judged:
        judged.sort(key=lambda r: r["judge_overall"], reverse=True)
        md.append("| Config | LLM-judge | State | R-1 | Semantic | Surface sum |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for r in judged:
            sem = f"{r['semantic_r1']:.4f}" if r["semantic_r1"] is not None else "—"
            md.append(
                f"| {r['name']} | {r['judge_overall']:.2f} | {r['state']:.2f} | {r['r1']:.2f} | {sem} | {r['surface_sum']:.2f} |"
            )

    # ── STL contamination ablation (GLM-4-9B-Chat-base vs SocratTeachLLM)
    # Cross-references the four n=681 rows added 2026-05-29 to expose the
    # fine-tune lift (true) and the surface-form inflation (contamination).
    md.append("\n## STL contamination ablation (GLM-4-9B-Chat-base vs SocratTeachLLM, n=681)\n")
    md.append(
        "SocratTeachLLM is a LoRA fine-tune of THUDM/glm-4-9b-chat. At matched n=681 + "
        "fewshot10 + consultant + vLLM serving, the only difference between paired rows "
        "is the LoRA adapter. The Δ row shows what the fine-tune actually contributes; the "
        "ROUGE/state-acc ratio quantifies how much of that delta is contamination."
    )
    md.append("")
    md.append("| Consultant | Cell | State acc | Stage-bal | R-1 | R-2 | BLEU-4 |")
    md.append("|---|---|---:|---:|---:|---:|---:|")

    def row_by_name(needle: str) -> dict | None:
        for r in rows:
            if r.get("available") and needle in r["name"]:
                return r
        return None

    def stage_bal(r: dict) -> float | None:
        ps = r.get("per_stage") or {}
        if not ps or len(ps) < 5:
            return None
        return sum(ps.values()) / 5.0

    for cons_label, base_needle, stl_needle in [
        ("bert-fixed", "bert-fixed × GLM-4-9B-Chat-base", "bert-fixed × SocratTeachLLM"),
        ("qwen3.5", "qwen3.5 × GLM-4-9B-Chat-base", "qwen3.5 × SocratTeachLLM"),
    ]:
        base = row_by_name(base_needle)
        stl = row_by_name(stl_needle)
        if not (base and stl):
            continue
        base_sb = stage_bal(base)
        stl_sb = stage_bal(stl)
        base_sb_str = f"{base_sb:.2f}" if base_sb is not None else "—"
        stl_sb_str = f"{stl_sb:.2f}" if stl_sb is not None else "—"
        md.append(
            f"| {cons_label} | GLM-4-9B-Chat-base | {base['state']:.2f} | "
            f"{base_sb_str} | {base['r1']:.2f} | {base['r2']:.2f} | {base['b4']:.2f} |"
        )
        md.append(
            f"| {cons_label} | SocratTeachLLM [contaminated] | {stl['state']:.2f} | "
            f"{stl_sb_str} | {stl['r1']:.2f} | {stl['r2']:.2f} | {stl['b4']:.2f} |"
        )
        if base_sb is not None and stl_sb is not None:
            d_state = stl["state"] - base["state"]
            d_sb = stl_sb - base_sb
            d_r1 = stl["r1"] - base["r1"]
            d_r2 = stl["r2"] - base["r2"]
            d_b4 = stl["b4"] - base["b4"]
            ratio = d_r1 / d_state if d_state else float("nan")
            md.append(
                f"| {cons_label} | **Δ (STL − base)** | **+{d_state:.2f}** | **+{d_sb:.2f}** | "
                f"**+{d_r1:.2f}** | **+{d_r2:.2f}** | **+{d_b4:.2f}**  ·  ROUGE-1/state ratio = {ratio:.2f}× |"
            )
    md.append("")
    md.append(
        "**Reading:** the fine-tune adds ~+8 pp pedagogical state accuracy but ~+15 pp "
        "ROUGE-1. ROUGE-1 inflation is ~1.8× the pedagogical gain at matched everything — "
        "the cleanest contamination signature in the experimental record, complementing "
        "the four prior diagnostics (surface-form inversion, n-gram-length scaling, "
        "cross-lingual translation, synthetic clean-probe collapse). STL retains a real "
        "fine-tune lift on state accuracy; the surface-form portion of its advantage is "
        "training-data memorization."
    )

    md.append("\n## Configs awaiting data\n")
    md.append("\n".join(f"- {r['name']}" for r in rows if not r.get("available")) or "(none)")

    Path("results/master_leaderboard.md").write_text("\n".join(md))
    print("Wrote results/master_leaderboard.json + .md")
    print(f"Available: {out['summary']['n_configs_available']}/{out['summary']['n_configs_total']}")


if __name__ == "__main__":
    main()
