"""Generate a synthetic Chinese Socratic teaching dataset for clean-probe testing.

Why this exists:
    SocratTeachLLM is fine-tuned on SocratDataset (see KELE paper §4.3), so
    every evaluation against SocratDataset measures memorization mixed with
    generalization. To isolate generalization we need test items STL cannot
    have seen during training. This script asks Claude Sonnet to generate a
    small dataset of fresh Chinese elementary-science multiple-choice
    questions paired with realistic student trajectories, in the exact
    schema expected by ``src/project/kele.py:run_single_dialogue``.

Output schema (matches references/KELE/SocratDataset.json):
    [
      {
        "id": <int>,
        "question": "<student-facing problem statement, often MCQ>",
        "answer": "<correct option / answer phrase>",
        "dialogue": [
          {"student": "<student utterance>", "teacher": "<reference teacher response>", "state": "<SocRule state, a1..e34>"},
          ...
        ]
      },
      ...
    ]

The reference teacher / state fields are used by the eval orchestrator for
ROUGE / state-acc baselines. For our memorization probe the ground-truth
teacher text is NOT the load-bearing signal — the LLM-judge score on STL's
generated responses is. We still emit them so the dataset slots in without
schema changes.

Usage:
    uv run python scripts/generate_synthetic_socrat.py \
        --n-dialogues 50 \
        --output /tmp/SocratDataset_SYNTHETIC.json \
        --model claude-sonnet-4-6

Cost: ~$0.10 per dialogue at Sonnet 4.6 rates (~3K input + ~1K output tokens
per dialogue, batched 5 per call). 50 dialogues ≈ $5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Use the Anthropic SDK directly to avoid pulling in the eval orchestrator's
# config layer for a script that only needs network IO.
import anthropic

PROMPT_TEMPLATE = """你将生成 {batch_size} 个全新的中国小学科学课苏格拉底式教学对话样本。这些样本将用于测试一个中国苏格拉底教学语言模型的真实泛化能力（与训练数据无关），所以**绝对不能**抄袭或改写任何已发布的 SocratDataset 题目。题目主题、题干和正确答案都必须是你新创作的。

要求：
1. **题目** (`question`)：每个样本必须包含一道中国小学（3-6 年级）科学课的选择题或判断题，题干必须以中文呈现，并在末尾给出 2-4 个候选答案，**用中文全角括号包围、用中文顿号分隔**（不要使用方括号或英文逗号或英文引号——它们会破坏 JSON 解析）。例如：
   - "下列哪种动物属于哺乳动物？（鲸鱼、鳄鱼、海龟）"
   - "光在真空中传播的速度大约是多少？（30 万千米每秒、3 万千米每秒、300 万千米每秒）"
   主题应涵盖物理、化学、生物、地理、天文中的任意一项。

2. **答案** (`answer`)：给出正确选项的简短文字描述。

3. **对话** (`dialogue`)：5-7 轮 student/teacher 交替的教学对话，严格遵循以下 SocRule 五阶段的状态码（state）：
   - a1: 学生最初提出问题（每个对话第一轮必须是 a1）
   - b2-b7: 概念探查（探查学生已有概念，澄清误解）
   - c8-c29: 归纳推理（通过反例、引导性提问帮助学生推理）
   - d30-d33: 规则建构（帮助学生用自己的话总结规律）
   - e34: 教师总结（在学生答对后给出最终总结）
   每轮的 `state` 字段必须从上面这些代码中选一个，且应按 a → b → c → d → e 的方向推进（允许在同一阶段内停留 1-2 轮）。

4. **学生发言** (`student`)：第一轮是学生原始提问，之后每轮是学生对老师上一轮提问的真实回答（可以答对、答错、或半对半错——这会让对话更真实）。学生的回答风格应像一个 8-12 岁的中国小学生。

5. **老师发言** (`teacher`)：作为一个理想的苏格拉底式老师，每轮通过提问引导学生思考（不要直接给答案），最终在 e34 阶段给出简短总结。每轮 1-3 句话，不要超过 80 个汉字。

6. **多样性**：5 个样本之间的主题、难度、状态轨迹、学生回答模式都要有差异。

**重要**：题目和老师对话**不能与 SocratDataset 中任何已有题目相似**。你生成的所有题目都必须是全新的、未在公开数据集中出现过的。

请以以下 JSON 数组格式输出（不要任何解释或代码块标记，直接输出 JSON）：

[
  {{
    "id": <从 {start_id} 开始递增的整数>,
    "question": "...",
    "answer": "...",
    "dialogue": [
      {{"student": "...", "teacher": "...", "state": "a1"}},
      {{"student": "...", "teacher": "...", "state": "b4"}},
      ...
      {{"student": "...", "teacher": "...", "state": "e34"}}
    ]
  }},
  ...
]
"""


def generate_batch(client: anthropic.Anthropic, model: str, batch_size: int, start_id: int) -> list[dict]:
    """Ask Claude for one batch of synthetic dialogues."""
    prompt = PROMPT_TEMPLATE.format(batch_size=batch_size, start_id=start_id)
    resp = client.messages.create(
        model=model,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip()
    # Strip markdown fences just in case the model adds them despite instructions.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("\n", 1)[0]
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        # Fallback: try to recover individual dialogue objects via brace matching.
        # The model occasionally puts unescaped single quotes or special chars
        # in Chinese strings that break strict JSON parsing.
        print(f"  WARN: strict parse failed ({e}); trying object-by-object recovery", file=sys.stderr)
        items = []
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = raw[start : i + 1]
                    try:
                        items.append(json.loads(chunk))
                    except json.JSONDecodeError:
                        pass
                    start = None
        if not items:
            print(f"  WARN: recovery also failed; first 200 chars: {raw[:200]}", file=sys.stderr)
            return []
        print(f"  recovered {len(items)} of likely 5 objects", file=sys.stderr)
    if not isinstance(items, list):
        print(f"  WARN: batch did not return a list", file=sys.stderr)
        return []
    return items


def validate_item(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    for k in ("id", "question", "answer", "dialogue"):
        if k not in item:
            return False
    if not isinstance(item["dialogue"], list) or len(item["dialogue"]) < 3:
        return False
    for turn in item["dialogue"]:
        if not all(k in turn for k in ("student", "teacher", "state")):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-dialogues", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=5, help="Dialogues per Claude call (5 fits in 16K output)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--start-id", type=int, default=100001, help="ID for first dialogue (avoid collision with SocratDataset)")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    items: list[dict] = []
    next_id = args.start_id
    n_batches = (args.n_dialogues + args.batch_size - 1) // args.batch_size

    t0 = time.time()
    for b in range(n_batches):
        want = min(args.batch_size, args.n_dialogues - len(items))
        print(f"Batch {b + 1}/{n_batches}: requesting {want} dialogues starting at id={next_id}", file=sys.stderr)
        batch = generate_batch(client, args.model, want, next_id)
        valid = [x for x in batch if validate_item(x)]
        print(f"  got {len(batch)} ({len(valid)} valid)", file=sys.stderr)
        for x in valid[:want]:
            x["id"] = next_id
            items.append(x)
            next_id += 1
        if len(items) >= args.n_dialogues:
            break
        # No need to sleep between calls; anthropic sdk handles backoff if needed.

    elapsed = time.time() - t0
    print(f"\nGenerated {len(items)} dialogues in {elapsed:.1f}s", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
