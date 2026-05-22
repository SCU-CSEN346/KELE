"""LLM-judge evaluation over saved dialogue logs.

Runs Claude Sonnet 4.6 as a single judge with a 4-axis rubric:
1. Socratic validity (0-3): Valid teaching move at the current stage?
2. Advancement (0-3): Does the response advance student reasoning?
3. Age-appropriateness (0-2): Vocabulary suitable for elementary?
4. Question-form fidelity (0-2): Single question, not over-leading.

Total per turn: 0-10. Aggregated per dialogue + per config.

Cost (Sonnet 4.6 at $3 in / $15 out):
  ~1000 input tokens + ~150 output tokens per turn judged.
  At n=50 (~280 turns) → ~$1.50 per config.
  At n=681 (~4000 turns) → ~$21 per config.

Usage:
  uv run python scripts/llm_judge_eval.py <results_dir> [--model claude-sonnet-4-6] [--workers 10]

Output: <results_dir>/judge_summary.json with per-turn + per-dialogue + overall scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai

ALL_STATES = (
    ["a0", "a1"]
    + [f"b{i}" for i in range(2, 8)]
    + [f"c{i}" for i in range(8, 30)]
    + [f"d{i}" for i in range(30, 34)]
    + ["e34"]
)

JUDGE_SYSTEM_PROMPT = """You are an expert pedagogy evaluator scoring an AI teacher's Socratic response to an elementary school student. You assess responses on four axes, given the conversation history and the ground-truth pedagogical state.

The SocRule framework has 5 stages:
- a (questioning): student poses a question
- b (concept probing): probe student concepts
- c (inductive reasoning): help student reason through misconceptions
- d (rule construction): help student build correct rule and apply it
- e (closure): summarize and confirm understanding

You score on four axes. Respond ONLY with a JSON object:

{
  "socratic_validity": 0-3,    // Is this a valid Socratic teaching move at the current stage? 3=textbook, 2=acceptable, 1=marginal, 0=wrong stage/move
  "advancement": 0-3,           // Does this response advance the student's reasoning toward the answer? 3=clearly advances, 2=neutral but valid, 1=stalls, 0=confuses/regresses
  "age_appropriateness": 0-2,   // Is vocabulary and complexity right for elementary school? 2=well-calibrated, 1=borderline, 0=too complex or too simple
  "question_form": 0-2,         // Is the response a single Socratic question? 2=clean single question, 1=multiple questions or hint, 0=statement only or over-leading
  "comment": "<brief one-line rationale>"
}

Total possible: 10 points (3+3+2+2). Be calibrated — a perfect 10 should be rare.
"""


def build_judge_user_prompt(dialogue: dict, turn_idx: int) -> str:
    """Build the per-turn judge prompt with conversation history."""
    turns = dialogue.get("dialogue", [])
    if turn_idx >= len(turns):
        return ""
    target = turns[turn_idx]

    # History (previous turns)
    history_lines = []
    for t in turns[:turn_idx]:
        s = t.get("student", "")
        tr = t.get("teacher_response", t.get("teacher", ""))
        if s:
            history_lines.append(f"STUDENT: {s}")
        if tr:
            history_lines.append(f"TEACHER: {tr}")
    history = "\n".join(history_lines) if history_lines else "(start of dialogue)"

    # Current turn
    student = target.get("student", "")
    teacher_response = target.get("teacher_response", target.get("teacher", ""))
    gt_state = target.get("ground_truth_state", target.get("state", "?"))
    gt_teacher = target.get("ground_truth_teacher", "")

    return f"""Conversation history:
{history}

Current turn:
STUDENT: {student}
GROUND-TRUTH STATE (for reference, do not penalize differences in routing): {gt_state}
GROUND-TRUTH REFERENCE RESPONSE (one valid Socratic move, but not the only valid one): {gt_teacher}

AI TEACHER RESPONSE TO SCORE: {teacher_response}

Score the AI teacher response."""


def score_turn(
    client: openai.Client, model: str, dialogue: dict, turn_idx: int, max_retries: int = 6
) -> dict | None:
    """Score a single turn via the judge. Returns dict with axis scores or None on failure."""
    user_prompt = build_judge_user_prompt(dialogue, turn_idx)
    if not user_prompt:
        return None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0,
            )
            content = resp.choices[0].message.content or ""
            # Trim markdown json fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            scores = json.loads(content)
            # Validate
            for k in ("socratic_validity", "advancement", "age_appropriateness", "question_form"):
                if k not in scores:
                    return None
                scores[k] = int(scores[k])
            scores["total"] = (
                scores["socratic_validity"]
                + scores["advancement"]
                + scores["age_appropriateness"]
                + scores["question_form"]
            )
            scores["usage_in"] = resp.usage.prompt_tokens
            scores["usage_out"] = resp.usage.completion_tokens
            return scores
        except openai.RateLimitError:
            wait = min(2**attempt, 30)
            time.sleep(wait)
        except (openai.APIStatusError, openai.APIConnectionError):
            wait = min(2**attempt, 30)
            time.sleep(wait)
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"  judge error (attempt {attempt + 1}): {e}", file=sys.stderr)
            time.sleep(2)

    return None


def judge_dialogue(client: openai.Client, model: str, dialogue: dict) -> list[dict]:
    """Judge every turn in a dialogue. Returns list of scored turns (skipping failures)."""
    out = []
    turns = dialogue.get("dialogue", [])
    for i in range(len(turns)):
        score = score_turn(client, model, dialogue, i)
        if score is not None:
            score["turn_idx"] = i
            score["ground_truth_state"] = turns[i].get("ground_truth_state", "?")
            out.append(score)
    return out


def judge_results_dir(results_dir: Path, client: openai.Client, model: str, workers: int) -> dict:
    """Run judge over all dialogues in results_dir. Returns summary dict."""
    dialogues_dir = results_dir / "dialogues"
    if not dialogues_dir.is_dir():
        raise FileNotFoundError(f"No dialogues/ in {results_dir}")

    dialogue_files = sorted(dialogues_dir.glob("*.json"))
    print(f"\n=== Judging {len(dialogue_files)} dialogues in {results_dir} ===")
    print(f"Judge model: {model}, workers: {workers}")

    started = time.time()
    all_turn_scores = []
    per_dialogue_summaries = []
    total_in_tokens = 0
    total_out_tokens = 0

    def _judge_one(dfile: Path):
        d = json.loads(dfile.read_text())
        scores = judge_dialogue(client, model, d)
        return dfile.name, d.get("id", -1), scores

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_judge_one, f) for f in dialogue_files]
        for i, fut in enumerate(as_completed(futures)):
            fname, did, scores = fut.result()
            if scores:
                dlg_total = sum(s["total"] for s in scores)
                dlg_n = len(scores)
                avg = dlg_total / dlg_n if dlg_n else 0
                in_tokens = sum(s.get("usage_in", 0) for s in scores)
                out_tokens = sum(s.get("usage_out", 0) for s in scores)
                total_in_tokens += in_tokens
                total_out_tokens += out_tokens
                per_dialogue_summaries.append(
                    {
                        "file": fname,
                        "id": did,
                        "n_turns_judged": dlg_n,
                        "avg_total": avg,
                        "avg_validity": sum(s["socratic_validity"] for s in scores) / dlg_n,
                        "avg_advancement": sum(s["advancement"] for s in scores) / dlg_n,
                        "avg_age": sum(s["age_appropriateness"] for s in scores) / dlg_n,
                        "avg_qform": sum(s["question_form"] for s in scores) / dlg_n,
                    }
                )
                all_turn_scores.extend(scores)
            elapsed = time.time() - started
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(dialogue_files) - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i + 1}/{len(dialogue_files)}  {fname}  "
                f"({dlg_n if scores else 0} turns, avg {avg:.2f}/10) "
                f"{elapsed:.0f}s elapsed, {remaining:.0f}s remaining",
                flush=True,
            )

    elapsed = time.time() - started

    # Per-stage aggregation
    per_stage = {s: {"n": 0, "total": 0.0} for s in "abcde"}
    for s in all_turn_scores:
        gt = s.get("ground_truth_state", "?")
        if gt and gt[0] in per_stage:
            per_stage[gt[0]]["n"] += 1
            per_stage[gt[0]]["total"] += s["total"]
    for k in per_stage:
        n = per_stage[k]["n"]
        per_stage[k]["avg"] = per_stage[k]["total"] / n if n else 0.0

    n_turns = len(all_turn_scores)
    summary = {
        "results_dir": str(results_dir),
        "judge_model": model,
        "n_dialogues_judged": len(per_dialogue_summaries),
        "n_turns_judged": n_turns,
        "wall_clock_seconds": elapsed,
        "overall_avg": sum(s["total"] for s in all_turn_scores) / n_turns if n_turns else 0,
        "axis_avgs": {
            "socratic_validity": sum(s["socratic_validity"] for s in all_turn_scores) / n_turns
            if n_turns
            else 0,
            "advancement": sum(s["advancement"] for s in all_turn_scores) / n_turns
            if n_turns
            else 0,
            "age_appropriateness": sum(s["age_appropriateness"] for s in all_turn_scores) / n_turns
            if n_turns
            else 0,
            "question_form": sum(s["question_form"] for s in all_turn_scores) / n_turns
            if n_turns
            else 0,
        },
        "per_stage": per_stage,
        "tokens_in": total_in_tokens,
        "tokens_out": total_out_tokens,
        # Cost estimate at Sonnet 4.6 rates
        "cost_usd": round(total_in_tokens / 1e6 * 3.0 + total_out_tokens / 1e6 * 15.0, 4),
        "per_dialogue": per_dialogue_summaries,
    }

    summary_path = results_dir / "judge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary written to {summary_path}")
    print(
        f"Overall: {summary['overall_avg']:.2f}/10  ({n_turns} turns, "
        f"${summary['cost_usd']:.3f} cost, {elapsed:.0f}s wall clock)"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM-judge eval over saved dialogues")
    parser.add_argument("results_dir", type=Path, help="Path to results dir containing dialogues/")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Judge model ID")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent judge calls")
    parser.add_argument("--api-key", default=None, help="Override ANTHROPIC_API_KEY")
    parser.add_argument(
        "--base-url",
        default="https://api.anthropic.com/v1/",
        help="OpenAI-compat endpoint base URL",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # try .env
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = openai.Client(api_key=api_key, base_url=args.base_url)
    judge_results_dir(args.results_dir, client, args.model, args.workers)


if __name__ == "__main__":
    main()
