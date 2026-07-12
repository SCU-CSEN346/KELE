"""Unit tests for the LLM-judge rubric parser (scripts/llm_judge_eval.py).

Only _parse_rubric is exercised — it's the fragile seam that turns free-form
judge output into validated axis scores. The backends themselves (openai client /
`claude -p` subprocess) are integration-only.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "llm_judge_eval", Path(__file__).resolve().parents[1] / "scripts" / "llm_judge_eval.py"
)
assert _spec and _spec.loader
llm_judge_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm_judge_eval)
_parse_rubric = llm_judge_eval._parse_rubric


def test_parse_rubric_clean_json():
    scores = _parse_rubric(
        '{"socratic_validity": 3, "advancement": 2, "age_appropriateness": 2, '
        '"question_form": 1, "comment": "ok"}'
    )
    assert scores is not None
    assert scores["total"] == 8
    assert scores["socratic_validity"] == 3


def test_parse_rubric_strips_markdown_fence():
    scores = _parse_rubric(
        '```json\n{"socratic_validity": 1, "advancement": 1, '
        '"age_appropriateness": 0, "question_form": 0}\n```'
    )
    assert scores is not None
    assert scores["total"] == 2


def test_parse_rubric_missing_axis_returns_none():
    assert _parse_rubric('{"socratic_validity": 3, "advancement": 2}') is None


def test_parse_rubric_non_json_returns_none():
    assert _parse_rubric("I cannot score this.") is None


def test_parse_rubric_coerces_stringified_ints():
    scores = _parse_rubric(
        '{"socratic_validity": "3", "advancement": "0", '
        '"age_appropriateness": "1", "question_form": "2"}'
    )
    assert scores is not None
    assert scores["total"] == 6
