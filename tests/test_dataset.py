"""Data-shape smoke tests for src/project/dataset.py.

Each test mocks `datasets.load_dataset` so no HF network call is made.
The tests verify that each loader converts raw HF records into the
expected `messages` format without requiring GPU or LLM server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock data builders — minimal valid records for each HF dataset
# ---------------------------------------------------------------------------


def _mock_hf(records: list[dict]) -> MagicMock:
    """Return a mock object that iterates over `records` when passed to list()."""
    m = MagicMock()
    m.__iter__ = MagicMock(side_effect=lambda: iter(records))
    return m


def _socrat_record(id_: int) -> dict:
    """Minimal SocratDataset / SocratDataset-EN record."""
    return {
        "id": id_,
        "question": f"Q{id_}",
        "options": ["opt1", "opt2"],
        "newHint": "hint",
        "newKnowledgePoint": "kp",
        "dialogue": [
            {
                "student": "student turn 1",
                "teacher": "teacher turn 1",
                "state": "a1",
                "action": "give_example",
                "evaluation": "Student is at a1",
            },
            {
                "student": "student turn 2",
                "teacher": "teacher turn 2",
                "state": "b2",
                "action": "heuristic_question",
                "evaluation": "Student progressed to b2",
            },
        ],
    }


def _socrat_synthetic_record(id_: int) -> dict:
    """Minimal SocratDataset-SYNTHETIC record (no `action`, no top-level meta)."""
    return {
        "id": id_,
        "question": f"Synthetic Q{id_}",
        "answer": "synthetic answer",
        "dialogue": [
            {"student": "synthetic student 1", "teacher": "synthetic teacher 1", "state": "a1"},
            {"student": "synthetic student 2", "teacher": "synthetic teacher 2", "state": "b3"},
        ],
    }


def _socraticmath_record(id_: int) -> dict:
    """Minimal SocraticMATH HF record.

    Empirical schema: conversations[0] = problem (from=assistant), [1] = teacher
    opening (from=assistant), [2..] alternating user/assistant.
    """
    return {
        "id": id_,
        "conversations": [
            {"from": "assistant", "value": f"Problem {id_}: solve for x."},
            {"from": "assistant", "value": "What is the first step you would take?"},
            {"from": "user", "value": "I'd isolate x."},
            {"from": "assistant", "value": "Good — show me how."},
            {"from": "user", "value": "Subtract 3 from both sides."},
        ],
    }


def _socrateach_multi_record() -> dict:
    """Minimal SocraTeach_Multi record (one problem with one dialogue)."""
    return {
        "question": "What is 2+2?",
        "dialogues": [
            {
                "dialogue_id": "dlg_001",
                "turns": [
                    {"system": "Can you think of what 2+2 could be?", "user": "4?"},
                    {"system": "Correct! How did you get there?", "user": "I counted."},
                ],
            }
        ],
    }


def _socrateach_single_record() -> dict:
    """Minimal SocraTeach_Single record."""
    return {
        "id": "single_001",
        "student_type": "incorrect",
        "history": [
            ["user question 1", "teacher answer 1"],
            ["user question 2", "teacher answer 2"],
        ],
        "prompt": "current student question",
        "response": "current teacher response",
    }


# ---------------------------------------------------------------------------
# Shared schema assertions
# ---------------------------------------------------------------------------


def _assert_record_schema(record: dict, expected_source: str) -> None:
    assert "id" in record, f"missing 'id' in {record}"
    assert "source" in record, f"missing 'source' in {record}"
    assert "messages" in record, f"missing 'messages' in {record}"
    assert "ground_truth_states" in record, f"missing 'ground_truth_states' in {record}"

    assert record["source"] == expected_source, (
        f"expected source={expected_source!r}, got {record['source']!r}"
    )

    messages = record["messages"]
    assert isinstance(messages, list), f"'messages' must be a list, got {type(messages)}"
    assert len(messages) >= 1, "record must have at least one message"

    valid_roles = {"system", "user", "assistant"}
    for msg in messages:
        assert "role" in msg, f"message missing 'role': {msg}"
        assert "content" in msg, f"message missing 'content': {msg}"
        assert msg["role"] in valid_roles, f"unexpected role {msg['role']!r}"
        assert isinstance(msg["content"], str), f"'content' must be str, got {type(msg['content'])}"


# ---------------------------------------------------------------------------
# socrat-en
# ---------------------------------------------------------------------------


def test_load_socrat_en_schema():
    from src.project.dataset import load_socrat_en

    raw = [_socrat_record(i) for i in range(10)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = load_socrat_en(split="all")

    assert len(records) == 10
    for r in records:
        _assert_record_schema(r, "socrat-en")

    # Annotated dataset must carry ground_truth_states
    assert records[0]["ground_truth_states"] is not None
    assert records[0]["ground_truth_states"] == ["a1", "b2"]


def test_load_socrat_en_user_turn_contains_consultant_markers():
    """Per TRAINING_PLAN §0.2: state/action live on the user turn with the
    inference-matching long labels (苏格拉底教学顾问评估结果 / 苏格拉底教学顾问建议的操作),
    never as a bracketed prefix on the assistant target."""
    from src.project.dataset import load_socrat_en

    raw = [_socrat_record(1)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = load_socrat_en(split="all")

    user_turns = [m for m in records[0]["messages"] if m["role"] == "user"]
    assert len(user_turns) == 2
    assert "苏格拉底教学顾问评估结果: 学生处于 a1 状态" in user_turns[0]["content"]
    assert "苏格拉底教学顾问建议的操作: give_example" in user_turns[0]["content"]
    assert "苏格拉底教学顾问评估结果: 学生处于 b2 状态" in user_turns[1]["content"]
    assert "苏格拉底教学顾问建议的操作: heuristic_question" in user_turns[1]["content"]
    # Student utterance is preserved verbatim, with the marker appended after it.
    assert user_turns[0]["content"].startswith("student turn 1")
    assert user_turns[1]["content"].startswith("student turn 2")


def test_load_socrat_en_assistant_turn_has_no_state_prefix():
    """Assistant target is the clean teacher response — never the bracketed
    prefix that was the original bug. Loss only fires on assistant turns
    (SFTConfig.assistant_only_loss=True), so any leakage here would teach
    the model to emit literal [State:..] strings."""
    from src.project.dataset import load_socrat_en

    raw = [_socrat_record(1)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = load_socrat_en(split="all")

    assistant_turns = [m for m in records[0]["messages"] if m["role"] == "assistant"]
    assert len(assistant_turns) == 2
    for at in assistant_turns:
        assert not at["content"].startswith("[State:"), f"leaked: {at['content']!r}"
        assert "[Action:" not in at["content"], f"leaked: {at['content']!r}"
    assert assistant_turns[0]["content"] == "teacher turn 1"
    assert assistant_turns[1]["content"] == "teacher turn 2"


def test_load_socrat_zh_user_turn_contains_consultant_markers():
    """Mirror coverage for socrat-zh — same long labels, since the BERT
    consultant emits Chinese markers regardless of dialogue language."""
    from src.project.dataset import load_socrat_zh

    raw = [_socrat_record(1)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = load_socrat_zh(split="all")

    user_turns = [m for m in records[0]["messages"] if m["role"] == "user"]
    assistant_turns = [m for m in records[0]["messages"] if m["role"] == "assistant"]
    assert "苏格拉底教学顾问评估结果: 学生处于 a1 状态" in user_turns[0]["content"]
    assert "苏格拉底教学顾问建议的操作: give_example" in user_turns[0]["content"]
    for at in assistant_turns:
        assert not at["content"].startswith("[State:"), f"leaked: {at['content']!r}"


def test_load_socrat_en_train_test_split():
    from src.project.dataset import load_socrat_en

    raw = [_socrat_record(i) for i in range(20)]
    # Return a fresh mock on each call so the iterator isn't exhausted between splits.
    with patch("datasets.load_dataset", side_effect=lambda *a, **kw: _mock_hf(raw)):
        train = load_socrat_en(split="train", seed=42)
        test = load_socrat_en(split="test", seed=42)

    assert len(train) + len(test) == 20
    train_ids = {r["id"] for r in train}
    test_ids = {r["id"] for r in test}
    assert train_ids.isdisjoint(test_ids), "train and test must not overlap"


# ---------------------------------------------------------------------------
# socrat-zh
# ---------------------------------------------------------------------------


def test_load_socrat_zh_schema():
    from src.project.dataset import load_socrat_zh

    raw = [_socrat_record(i) for i in range(5)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = load_socrat_zh(split="all")

    assert len(records) == 5
    for r in records:
        _assert_record_schema(r, "socrat-zh")
    assert records[0]["ground_truth_states"] is not None


# ---------------------------------------------------------------------------
# socrat-synthetic
# ---------------------------------------------------------------------------


def test_load_socrat_synthetic_no_consultant_marker():
    """Synthetic records have no `action` field, so the Pattern A marker
    cannot be constructed. User turns are clean student utterances; the
    BERT consultant supplies markers at inference."""
    from src.project.dataset import load_socrat_synthetic

    raw = [_socrat_synthetic_record(1)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = load_socrat_synthetic(split="all")

    user_turns = [m for m in records[0]["messages"] if m["role"] == "user"]
    assert user_turns[0]["content"] == "synthetic student 1"
    assert "苏格拉底教学顾问" not in user_turns[0]["content"]
    assert "苏格拉底教学顾问" not in user_turns[1]["content"]


# ---------------------------------------------------------------------------
# socrateach-multi
# ---------------------------------------------------------------------------


def test_load_socrateach_multi_schema():
    from src.project.dataset import load_socrateach_multi

    raw = [_socrateach_multi_record() for _ in range(5)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = load_socrateach_multi(split="all")

    # Each problem has 1 dialogue → 5 records
    assert len(records) == 5
    for r in records:
        _assert_record_schema(r, "socrateach-multi")
        assert r["ground_truth_states"] is None, "socrateach-multi has no SocRule annotations"


def test_load_socrateach_multi_teacher_leads():
    """In SocraTeach_Multi, teacher turn comes first (assistant, then user)."""
    from src.project.dataset import load_socrateach_multi

    with patch("datasets.load_dataset", return_value=_mock_hf([_socrateach_multi_record()])):
        records = load_socrateach_multi(split="all")

    non_system = [m for m in records[0]["messages"] if m["role"] != "system"]
    assert non_system[0]["role"] == "assistant", (
        "First non-system turn must be assistant (teacher asks first in SocraTeach_Multi)"
    )


# ---------------------------------------------------------------------------
# socrateach-single
# ---------------------------------------------------------------------------


def test_load_socrateach_single_schema():
    from src.project.dataset import load_socrateach_single

    raw = [_socrateach_single_record() for _ in range(5)]
    with patch("datasets.load_dataset", return_value=_mock_hf(raw)):
        records = load_socrateach_single(split="all")

    assert len(records) == 5
    for r in records:
        _assert_record_schema(r, "socrateach-single")
        assert r["ground_truth_states"] is None


def test_load_socrateach_single_ends_with_assistant():
    """Final message in each record must be the target teacher response (assistant)."""
    from src.project.dataset import load_socrateach_single

    with patch("datasets.load_dataset", return_value=_mock_hf([_socrateach_single_record()])):
        records = load_socrateach_single(split="all")

    last_msg = records[0]["messages"][-1]
    assert last_msg["role"] == "assistant"
    assert last_msg["content"] == "current teacher response"


# ---------------------------------------------------------------------------
# load_training_data (unified entry point)
# ---------------------------------------------------------------------------


def test_load_training_data_combines_sources():
    from src.project.dataset import load_training_data

    en_raw = [_socrat_record(i) for i in range(6)]
    zh_raw = [_socrat_record(i) for i in range(6)]

    def mock_hf(repo, split="train", **_kwargs):
        return _mock_hf(en_raw if "EN" in repo else zh_raw)

    with patch("datasets.load_dataset", side_effect=mock_hf):
        records = load_training_data(sources=["socrat-en", "socrat-zh"], split="all")

    assert len(records) == 12
    sources = {r["source"] for r in records}
    assert sources == {"socrat-en", "socrat-zh"}


def test_load_training_data_rejects_unknown_source():
    import pytest

    from src.project.dataset import load_training_data

    with pytest.raises(ValueError, match="Unknown sources"):
        load_training_data(sources=["nonexistent-source"])
