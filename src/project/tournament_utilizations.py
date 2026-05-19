"""Prompt-engineering utilizations for the Phase 1 tournament.

Each utilization is gated by an environment variable (KELE_* prefix) and
modifies the BERT + Gemma + 10-shot baseline either by:
  - Pre-call: rewriting the teacher's system prompt / user input / exemplar block
  - Post-call: wrapping the LLM call (retries, N-best, two-pass CoT)

The dispatch entry point `apply_pre_call(...)` and `call_teacher_wrapped(...)`
are invoked from `socratic_teaching_system.socrates_teacher`. Default
(all env vars unset) reproduces the Phase 0 locked behavior exactly.

Tournament cells (see docs/PROMPT_ENGINEERING_PLAN.md §3):
  #1  KELE_STAGE_LENGTH_BUDGET=1     — per-stage char-length budget
  #2  KELE_LEXICAL_PRIORS=1          — top-10 opener 4-grams primer
  #3  KELE_STYLE_MATCHED_EXEMPLARS=1 — similarity-weighted exemplar selection
  #4  KELE_PER_STATE_EXEMPLARS=1     — BERT-state-conditional exemplars
  #5  KELE_NEGATIVE_EXEMPLARS=1      — paired positive/anti-example exemplars
  #6  KELE_FORMAT_RETRY=1            — post-hoc format validation + 1 retry
  #7  KELE_TEACHER_COT=1             — two-pass: hidden reasoning then output
  #8  KELE_NBEST_RERANK=<N>          — generate N candidates, pick best by style critic
  #9  KELE_TEACHER_PERSONA=1         — 苏老师 pedagogical persona anchor
  #10 KELE_COMPRESSED_HISTORY=1      — summarize last k student turns to 1 sentence

Mutex: at most one of #6, #7, #8 should be enabled per cell.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

# ─── Stage length budget (§3.2 in plan) ──────────────────────────────────────
# Per-stage char distribution measured from SocratDataset.json (43,892 teacher
# turns, see docs/PROMPT_ENGINEERING_PLAN.md table). Values are (p10, median, p90).
_STAGE_LENGTH_BUDGET: dict[str, tuple[int, int, int]] = {
    "a": (14, 21, 31),
    "b": (18, 26, 42),
    "c": (19, 28, 43),
    "d": (24, 37, 56),
    "e": (31, 42, 56),
}

# ─── Lexical priors (§3.2 in plan) ───────────────────────────────────────────
# Top-10 most common 4-character openers in the SocratDataset teacher corpus.
_TOP_OPENER_4GRAMS: list[str] = [
    "很好！那",  # Great! So…
    "非常好！",  # Excellent!
    "很好！所",  # Great! So…
    "你能告诉",  # Can you tell me…
    "太棒了！",  # Awesome!
    "很好！你",  # Great! You…
    "你有没有",  # Do you have…
    "完全正确",  # Completely correct
    "很好，那",  # Great, so…
    "那你觉得",  # So what do you think…
]


# ─── Module-level caches built lazily on first use ───────────────────────────
_TRAIN_TURNS_BY_STATE: dict[str, list[dict]] | None = None
_TRAIN_TURNS_BY_STAGE: dict[str, list[dict]] | None = None


def _resources_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "references" / "KELE"


def _load_train_turns_indexed() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Load train-split teacher turns, indexed by full state and by stage prefix.

    Train split is dialogues NOT in the test split (the test split is a fixed
    90/10 random sample with seed=42, computed in kele.load_dataset).
    """
    global _TRAIN_TURNS_BY_STATE, _TRAIN_TURNS_BY_STAGE
    if _TRAIN_TURNS_BY_STATE is not None and _TRAIN_TURNS_BY_STAGE is not None:
        return _TRAIN_TURNS_BY_STATE, _TRAIN_TURNS_BY_STAGE

    import random

    src = _resources_dir() / "SocratDataset.json"
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    # Replicate kele.load_dataset(split="train", seed=42) selection
    rng = random.Random(42)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    split_point = int(len(data) * 0.9)
    train = [data[i] for i in sorted(indices[:split_point])]

    by_state: dict[str, list[dict]] = {}
    by_stage: dict[str, list[dict]] = {"a": [], "b": [], "c": [], "d": [], "e": []}
    for item in train:
        for turn in item.get("dialogue", []):
            state = turn.get("state", "")
            student = (turn.get("student") or "").strip()
            teacher = (turn.get("teacher") or "").strip()
            if not state or not student or not teacher:
                continue
            stage = state[0]
            ex = {"state": state, "student": student, "teacher": teacher}
            by_state.setdefault(state, []).append(ex)
            if stage in by_stage:
                by_stage[stage].append(ex)

    _TRAIN_TURNS_BY_STATE = by_state
    _TRAIN_TURNS_BY_STAGE = by_stage
    return by_state, by_stage


# ─── Utilization #1: Per-stage length budget ─────────────────────────────────
def _apply_length_budget(system_prompt: str, predicted_state: str) -> str:
    stage = predicted_state[0] if predicted_state else "a"
    if stage not in _STAGE_LENGTH_BUDGET:
        return system_prompt
    p10, median, p90 = _STAGE_LENGTH_BUDGET[stage]
    addition = (
        f"\n本轮处于阶段 {stage}。回答长度应在 {p10}–{p90} 字之间"
        f"（典型 {median} 字）。只问一个问题，以\"？\"结束。"
    )
    return system_prompt + addition


# ─── Utilization #2: Lexical priors ──────────────────────────────────────────
def _apply_lexical_priors(system_prompt: str) -> str:
    openers = " / ".join(_TOP_OPENER_4GRAMS)
    addition = f"\n优秀教师的开头通常是：{openers}\n请采用其中一种风格开始你的问题。"
    return system_prompt + addition


# ─── Utilization #9: Persona anchor ──────────────────────────────────────────
_PERSONA_BLOCK = (
    "你是苏老师（Teacher Su），一位有20年经验的苏格拉底式儿童教师。"
    "你的特点是：每次只问一个能让学生顿悟的问题，从不解释，从不说废话。"
    "你说话简短、温和、循序渐进。\n"
)


def _apply_persona(system_prompt: str) -> str:
    return _PERSONA_BLOCK + system_prompt


# ─── Utilization #4: Per-state few-shot routing (BERT-conditional) ───────────
def _per_state_exemplar_block(predicted_state: str, k: int = 3) -> str | None:
    """Return an exemplar block of k turns drawn from the same state in the
    train split. Falls back to the same stage if fewer than k state-matched
    turns are available, and to None if neither pool reaches k.

    The caller decides whether to use this block (return value None = abstain;
    leave the default block intact).
    """
    by_state, by_stage = _load_train_turns_indexed()
    pool = by_state.get(predicted_state, [])
    stage = predicted_state[0] if predicted_state else ""

    if len(pool) < k:
        pool = pool + by_stage.get(stage, [])
    if len(pool) < k:
        return None

    chosen = pool[:k]
    lines = ["---", "", "# 第四部分：教师风格示例（按当前预测状态检索）",
             f"以下是来自训练集、与当前状态 {predicted_state} 一致的教师回复示例：", ""]
    for i, ex in enumerate(chosen, 1):
        lines.append(f"示例{i}（state={ex['state']}）:")
        lines.append(f"学生: {ex['student']}")
        lines.append(f"老师: {ex['teacher']}")
        lines.append("")
    lines.append("请严格遵循这种风格：开头不要长篇铺垫，只问一个有针对性的问题，语气亲切。")
    return "\n".join(lines) + "\n"


# ─── Utilization #3: Style-matched exemplars ─────────────────────────────────
def _char_ngram_counter(text: str, n: int = 3) -> Counter[str]:
    return Counter(text[i : i + n] for i in range(max(0, len(text) - n + 1)))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    inter = set(a.keys()) & set(b.keys())
    num = sum(a[t] * b[t] for t in inter)
    da = sum(v * v for v in a.values()) ** 0.5
    db = sum(v * v for v in b.values()) ** 0.5
    return num / (da * db) if da and db else 0.0


def _style_matched_exemplar_block(predicted_state: str, k: int = 10) -> str | None:
    """Select k exemplars whose teacher response has highest char-3-gram
    cosine similarity to the typical (median-length, target-stage) response.

    The "query" is the per-stage median teacher length archetype: a centroid
    built from all stage-matched train turns. We re-rank the stage pool by
    similarity to that centroid, picking the top-k.
    """
    _, by_stage = _load_train_turns_indexed()
    stage = predicted_state[0] if predicted_state else "a"
    pool = by_stage.get(stage, [])
    if len(pool) < k:
        return None

    # Build the stage centroid from the median-length k*5 candidates' teacher text
    sorted_by_len_distance = sorted(
        pool, key=lambda ex: abs(len(ex["teacher"]) - _STAGE_LENGTH_BUDGET.get(stage, (0, 30, 0))[1])
    )
    centroid_pool = sorted_by_len_distance[: min(k * 5, len(sorted_by_len_distance))]
    centroid_text = "".join(ex["teacher"] for ex in centroid_pool)
    centroid = _char_ngram_counter(centroid_text)

    ranked = sorted(
        pool, key=lambda ex: _cosine(_char_ngram_counter(ex["teacher"]), centroid), reverse=True
    )
    chosen = ranked[:k]

    lines = ["---", "", "# 第四部分：教师风格示例（按风格相似度检索）",
             "以下是与当前阶段典型风格最匹配的教师回复示例：", ""]
    for i, ex in enumerate(chosen, 1):
        lines.append(f"示例{i}（state={ex['state']}）:")
        lines.append(f"学生: {ex['student']}")
        lines.append(f"老师: {ex['teacher']}")
        lines.append("")
    lines.append("请严格遵循这种风格：开头不要长篇铺垫，只问一个有针对性的问题，语气亲切。")
    return "\n".join(lines) + "\n"


# ─── Utilization #5: Negative-exemplar contrast ──────────────────────────────
# Hand-crafted anti-examples paired with positive exemplars. Each positive is
# from _EXEMPLARS_BY_STAGE; the anti-example shows what to avoid (preamble,
# multiple questions, off-topic).
_NEGATIVE_PAIRS: list[tuple[str, str, str]] = [
    # (stage, positive_teacher, anti_example)
    (
        "b",
        "嗯，叶子在变绿的时候，它在做什么呢？",
        "嗯，这是一个很好的问题。让我想想... 叶子变绿是因为叶绿素。那么你觉得叶子的颜色和它的功能有什么关系？另外你还想了解什么吗？",
    ),
    (
        "c",
        "实际上，青苔没有真正的根，只有假根用于固定。你觉得为什么之前没有想到这一点呢？",
        "嗯，让我详细解释一下。青苔是一种苔藓植物，与高等植物不同，它没有真正的根、茎、叶分化。所以它怎么固定呢？青苔是怎么吸收水分的？",
    ),
    (
        "d",
        "很好！现在我们可以总结一下：植物的需水量会受到天气和土壤湿度的影响。你能根据这个总结回答题目吗？",
        "好的，让我们来回顾一下我们讨论过的内容。植物需要水，水来自土壤，土壤的湿度受天气影响。综合来看，这是一个复杂的系统。那么具体到这道题应该怎么回答呢？还有其他因素需要考虑吗？",
    ),
]


def _apply_negative_exemplars(system_prompt: str) -> str:
    lines = ["", "---", "", "# 风格对比：好示例 vs 不推荐",
             "以下展示了哪些回复风格应该避免：", ""]
    for stage, positive, anti in _NEGATIVE_PAIRS:
        lines.append(f"阶段 {stage}：")
        lines.append(f"  好的示例：{positive}")
        lines.append(f"  不推荐：{anti}")
        lines.append("")
    lines.append("请避免：长篇铺垫、多个问题、偏题、解释过多。")
    return system_prompt + "\n".join(lines) + "\n"


# ─── Utilization #6: Output-format validation + retry ────────────────────────
_PREAMBLE_MARKERS = ("好的，", "让我", "嗯，这是", "首先", "我来", "我们来")


def _validate_teacher_output(text: str, predicted_state: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Checks ends-with-？, single-？, length p10..p90×1.5,
    no preamble markers."""
    if not text:
        return False, "empty output"
    text = text.strip()
    if not text.endswith("？") and not text.endswith("?"):
        return False, "does not end with ？"
    q_count = text.count("？") + text.count("?")
    stage = predicted_state[0] if predicted_state else "a"
    p10, _, p90 = _STAGE_LENGTH_BUDGET.get(stage, (10, 30, 60))
    # Stage e is closure (summary then check) — allow longer + permit 0 questions
    if stage != "e":
        if q_count == 0:
            return False, "no question"
        if q_count > 1:
            return False, "multiple questions"
        # Hard cap at 2× p90 to catch runaway preamble
        if len(text) > 2 * p90:
            return False, f"too long ({len(text)} > {2*p90})"
    if any(text.startswith(m) for m in _PREAMBLE_MARKERS):
        return False, "starts with preamble marker"
    return True, "ok"


# ─── Utilization #8 style critic (also used by #6 re-validate) ───────────────
def _style_score(text: str, predicted_state: str) -> int:
    if not text:
        return 0
    text = text.strip()
    score = 0
    stage = predicted_state[0] if predicted_state else "a"
    p10, _, p90 = _STAGE_LENGTH_BUDGET.get(stage, (10, 30, 60))
    if p10 <= len(text) <= p90:
        score += 1
    if any(text.startswith(opener) for opener in _TOP_OPENER_4GRAMS):
        score += 1
    q_count = text.count("？") + text.count("?")
    if q_count == 1 and not any(text.startswith(m) for m in _PREAMBLE_MARKERS):
        score += 1
    return score


# ─── Utilization #10: Compressed dialogue history ────────────────────────────
def _compress_history(teacher_client: Any, model_name: str, formatted_history: str) -> str:
    """One small LLM call to summarize the dialogue history to 1 sentence."""
    if not formatted_history.strip():
        return formatted_history
    try:
        response = teacher_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "用一句中文总结学生最近的回答和理解状态，不超过30字。",
                },
                {"role": "user", "content": formatted_history},
            ],
            max_tokens=80,
        )
        summary = (response.choices[0].message.content or "").strip()
        return f"对话摘要：{summary}" if summary else formatted_history
    except Exception:
        return formatted_history  # fall back to verbatim on any failure


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch entry points called from socratic_teaching_system.socrates_teacher
# ═══════════════════════════════════════════════════════════════════════════════


def apply_pre_call(
    system_prompt: str,
    user_input: str,
    predicted_state: str,
    teacher_client: Any,
    teacher_model_name: str,
    formatted_history: str,
) -> tuple[str, str]:
    """Apply env-gated pre-call utilizations to the system prompt and user input.

    Order of application matters for prompt-overlay utilizations: persona
    prepends (#9 is the strongest stylistic anchor); length budget and lexical
    priors append (additive guidance); exemplar swaps (#3, #4) replace the
    existing few-shot block; negative-exemplar contrast (#5) appends.

    Returns the (possibly modified) system_prompt and user_input.
    """
    if os.environ.get("KELE_TEACHER_PERSONA") == "1":
        system_prompt = _apply_persona(system_prompt)
    if os.environ.get("KELE_STAGE_LENGTH_BUDGET") == "1":
        system_prompt = _apply_length_budget(system_prompt, predicted_state)
    if os.environ.get("KELE_LEXICAL_PRIORS") == "1":
        system_prompt = _apply_lexical_priors(system_prompt)
    if os.environ.get("KELE_NEGATIVE_EXEMPLARS") == "1":
        system_prompt = _apply_negative_exemplars(system_prompt)

    # Exemplar swaps: #4 takes precedence over #3 if both set (per-state is
    # the more targeted retrieval). The swap replaces the existing few-shot
    # block by appending a fresh one — the LLM weights the more-recent block.
    if os.environ.get("KELE_PER_STATE_EXEMPLARS") == "1":
        block = _per_state_exemplar_block(predicted_state, k=3)
        if block:
            system_prompt = system_prompt + "\n" + block
    elif os.environ.get("KELE_STYLE_MATCHED_EXEMPLARS") == "1":
        block = _style_matched_exemplar_block(predicted_state, k=10)
        if block:
            system_prompt = system_prompt + "\n" + block

    # Compressed history: replace the verbatim history in user_input
    if os.environ.get("KELE_COMPRESSED_HISTORY") == "1":
        summary = _compress_history(teacher_client, teacher_model_name, formatted_history)
        # Substitute the verbatim history block with the summary
        user_input = user_input.replace(formatted_history, summary, 1)

    return system_prompt, user_input


def call_teacher_wrapped(
    teacher_client: Any,
    teacher_model_name: str,
    system_prompt: str,
    user_input: str,
    predicted_state: str,
) -> str:
    """Make the teacher LLM call, optionally wrapped by post-call utilizations.

    Mutex: at most one of {#6 format-retry, #7 CoT, #8 N-best} may be set.
    If multiple are set, precedence is #8 > #7 > #6 (most-elaborate wins).
    """
    nbest_str = os.environ.get("KELE_NBEST_RERANK")

    if nbest_str:
        try:
            n = max(1, min(5, int(nbest_str)))
        except ValueError:
            n = 3
        return _call_nbest_rerank(
            teacher_client, teacher_model_name, system_prompt, user_input, predicted_state, n
        )

    if os.environ.get("KELE_TEACHER_COT") == "1":
        return _call_cot_two_pass(
            teacher_client, teacher_model_name, system_prompt, user_input, predicted_state
        )

    if os.environ.get("KELE_FORMAT_RETRY") == "1":
        return _call_with_format_retry(
            teacher_client, teacher_model_name, system_prompt, user_input, predicted_state
        )

    # Default: single call (unchanged behavior)
    response = teacher_client.chat.completions.create(
        model=teacher_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content or ""


def _call_with_format_retry(
    client: Any, model: str, system_prompt: str, user_input: str, predicted_state: str
) -> str:
    """Utilization #6: single call, validate, retry once with error feedback if invalid."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    text = response.choices[0].message.content or ""
    valid, reason = _validate_teacher_output(text, predicted_state)
    if valid:
        return text

    # One retry with error appended to user input
    retry_prompt = user_input + f"\n\n（上一次回复不合格：{reason}。请按格式重写一句话，以\"？\"结尾，无前言。）"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_prompt},
            ],
        )
        retry_text = response.choices[0].message.content or ""
        # Accept the retry even if it still fails validation (best-effort)
        return retry_text if retry_text.strip() else text
    except Exception:
        return text  # fall back to the original (failed-validation) output


def _call_cot_two_pass(
    client: Any, model: str, system_prompt: str, user_input: str, predicted_state: str
) -> str:
    """Utilization #7: hidden reasoning pass, then final question pass.

    Two LLM calls per turn (2× inference cost).
    """
    # Pass 1: internal reasoning
    reasoning_prompt = (
        system_prompt
        + "\n\n本轮特殊指令：先内部分析当前学生输入的核心误解和应当问的最小苏格拉底问题。"
        + "只输出分析过程，不要输出问题本身。"
    )
    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": reasoning_prompt},
                {"role": "user", "content": user_input},
            ],
            max_tokens=200,
        )
        reasoning = (r1.choices[0].message.content or "").strip()
    except Exception:
        reasoning = ""

    # Pass 2: final output, given the reasoning
    final_user = user_input + (
        f"\n\n你的内部分析：{reasoning}\n根据上述分析，输出一句话作为最终的苏格拉底问题，以\"？\"结尾。"
        if reasoning
        else ""
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user},
        ],
    )
    return response.choices[0].message.content or ""


def _call_nbest_rerank(
    client: Any,
    model: str,
    system_prompt: str,
    user_input: str,
    predicted_state: str,
    n: int,
) -> str:
    """Utilization #8: generate N candidates at temperature 0.8, pick by style critic."""
    candidates: list[str] = []
    for _ in range(n):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.8,
            )
            text = response.choices[0].message.content or ""
            if text.strip():
                candidates.append(text)
        except Exception:
            continue

    if not candidates:
        # Fallback to one default-temperature call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        return response.choices[0].message.content or ""

    # Score and pick the highest; tiebreaker = first generated
    best = max(candidates, key=lambda c: _style_score(c, predicted_state))
    return best
