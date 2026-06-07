"""Unit tests for the env-gated prompt-engineering utilizations.

tournament_utilizations is pure-stdlib (no torch/transformers), so it imports
directly. The teacher LLM is injected as a client object, so the post-call
wrappers are exercised with a scripted fake instead of a live model.
"""

import json

import pytest

from src.project import tournament_utilizations as tu

_KELE_ENV_VARS = [
    "KELE_STAGE_LENGTH_BUDGET",
    "KELE_LEXICAL_PRIORS",
    "KELE_STYLE_MATCHED_EXEMPLARS",
    "KELE_PER_STATE_EXEMPLARS",
    "KELE_NEGATIVE_EXEMPLARS",
    "KELE_FORMAT_RETRY",
    "KELE_TEACHER_COT",
    "KELE_NBEST_RERANK",
    "KELE_TEACHER_PERSONA",
    "KELE_COMPRESSED_HISTORY",
]


@pytest.fixture(autouse=True)
def _clean_kele_env(monkeypatch):
    """Every utilization is off by default so tests start from Phase-0 behavior."""
    for var in _KELE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ─── Fake OpenAI-style client ────────────────────────────────────────────────


class _Resp:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})})]


class _Completions:
    def __init__(self, parent):
        self._parent = parent

    def create(self, **kwargs):
        self._parent.calls.append(kwargs)
        item = self._parent._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return _Resp(item)


class FakeClient:
    """Returns scripted contents (or raises scripted exceptions) in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = type("Chat", (), {"completions": _Completions(self)})


# ─── #1 Stage length budget ──────────────────────────────────────────────────


def test_length_budget_appends_stage_bounds():
    out = tu._apply_length_budget("SYS", "b2")
    p10, median, p90 = tu._STAGE_LENGTH_BUDGET["b"]
    assert out.startswith("SYS")
    assert f"{p10}–{p90}" in out
    assert str(median) in out


def test_length_budget_unknown_stage_is_noop():
    assert tu._apply_length_budget("SYS", "z9") == "SYS"


def test_length_budget_empty_state_defaults_to_stage_a():
    out = tu._apply_length_budget("SYS", "")
    p10, _, p90 = tu._STAGE_LENGTH_BUDGET["a"]
    assert f"{p10}–{p90}" in out


# ─── #2 Lexical priors / #9 Persona / #5 Negative exemplars ──────────────────


def test_lexical_priors_appends_all_openers():
    out = tu._apply_lexical_priors("SYS")
    assert out.startswith("SYS")
    for opener in tu._TOP_OPENER_4GRAMS:
        assert opener in out


def test_persona_prepends_block():
    out = tu._apply_persona("SYS")
    assert out.startswith(tu._PERSONA_BLOCK)
    assert out.endswith("SYS")


def test_negative_exemplars_appends_each_pair():
    out = tu._apply_negative_exemplars("SYS")
    assert out.startswith("SYS")
    for stage, positive, anti in tu._NEGATIVE_PAIRS:
        assert positive in out
        assert anti in out


# ─── #3 ngram / cosine primitives ────────────────────────────────────────────


def test_char_ngram_counter_counts_overlapping_trigrams():
    assert tu._char_ngram_counter("abcd") == {"abc": 1, "bcd": 1}


def test_char_ngram_counter_shorter_than_n_is_empty():
    assert tu._char_ngram_counter("ab", n=3) == {}


def test_cosine_identical_is_one():
    c = tu._char_ngram_counter("今天天气很好")
    assert tu._cosine(c, c) == pytest.approx(1.0)


def test_cosine_disjoint_is_zero():
    a = tu._char_ngram_counter("abcdef")
    b = tu._char_ngram_counter("uvwxyz")
    assert tu._cosine(a, b) == 0.0


def test_cosine_empty_is_zero():
    assert tu._cosine(tu._char_ngram_counter(""), tu._char_ngram_counter("abc")) == 0.0


# ─── #4 / #3 exemplar block routing ──────────────────────────────────────────


def _turn(state, student="问题", teacher="为什么呢？"):
    return {"state": state, "student": student, "teacher": teacher}


def test_per_state_block_returns_none_when_pool_too_small(monkeypatch):
    monkeypatch.setattr(tu, "_load_train_turns_indexed", lambda: ({}, {}))
    assert tu._per_state_exemplar_block("b2", k=3) is None


def test_per_state_block_uses_state_matched_turns(monkeypatch):
    by_state = {"b2": [_turn("b2") for _ in range(3)]}
    monkeypatch.setattr(tu, "_load_train_turns_indexed", lambda: (by_state, {}))
    block = tu._per_state_exemplar_block("b2", k=3)
    assert block is not None
    assert "state=b2" in block
    assert block.count("（state=") == 3


def test_per_state_block_falls_back_to_stage_pool(monkeypatch):
    by_state = {"b2": [_turn("b2")]}
    by_stage = {"b": [_turn("b1") for _ in range(3)]}
    monkeypatch.setattr(tu, "_load_train_turns_indexed", lambda: (by_state, by_stage))
    block = tu._per_state_exemplar_block("b2", k=3)
    assert block is not None
    assert block.count("（state=") == 3


def test_load_train_turns_indexed_replicates_split(monkeypatch, tmp_path):
    dataset = [
        {
            "dialogue": [
                {"state": "b2", "student": f"问题{i}", "teacher": f"回答{i}？"},
                {"state": "", "student": "skip", "teacher": "missing state"},
                {"state": "c1", "student": "", "teacher": "missing student"},
            ]
        }
        for i in range(20)
    ]
    src = tmp_path / "SocratDataset.json"
    src.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(tu, "_resources_dir", lambda: tmp_path)
    monkeypatch.setattr(tu, "_TRAIN_TURNS_BY_STATE", None)
    monkeypatch.setattr(tu, "_TRAIN_TURNS_BY_STAGE", None)

    by_state, by_stage = tu._load_train_turns_indexed()

    assert len(by_state["b2"]) == 18  # 90% train split of 20 dialogues
    assert len(by_stage["b"]) == 18
    assert "" not in by_state  # turns missing state/student/teacher are dropped
    assert "c1" not in by_state

    # Second call returns the cached value without re-reading the file.
    src.unlink()
    again_state, _ = tu._load_train_turns_indexed()
    assert again_state is by_state


def test_style_matched_block_returns_none_when_pool_too_small(monkeypatch):
    monkeypatch.setattr(tu, "_load_train_turns_indexed", lambda: ({}, {"a": []}))
    assert tu._style_matched_exemplar_block("a1", k=10) is None


def test_style_matched_block_picks_k_exemplars(monkeypatch):
    pool = [_turn("a1", teacher=f"问题{i}？") for i in range(12)]
    monkeypatch.setattr(tu, "_load_train_turns_indexed", lambda: ({}, {"a": pool}))
    block = tu._style_matched_exemplar_block("a1", k=10)
    assert block is not None
    assert block.count("（state=") == 10


# ─── #6 output validation ────────────────────────────────────────────────────


def test_validate_rejects_empty():
    assert tu._validate_teacher_output("", "b2") == (False, "empty output")


def test_validate_rejects_missing_question_mark():
    valid, reason = tu._validate_teacher_output("叶子在做什么呢", "b2")
    assert not valid
    assert "？" in reason


def test_validate_accepts_clean_single_question():
    valid, reason = tu._validate_teacher_output("叶子在变绿时做什么呢？", "b2")
    assert valid
    assert reason == "ok"


def test_validate_rejects_multiple_questions():
    valid, reason = tu._validate_teacher_output("这是什么？那又是什么？", "b2")
    assert not valid
    assert reason == "multiple questions"


def test_validate_rejects_preamble_marker():
    valid, reason = tu._validate_teacher_output("好的，叶子在做什么呢？", "b2")
    assert not valid
    assert reason == "starts with preamble marker"


def test_validate_rejects_runaway_length():
    p90 = tu._STAGE_LENGTH_BUDGET["b"][2]
    text = "啊" * (2 * p90 + 1) + "？"
    valid, reason = tu._validate_teacher_output(text, "b2")
    assert not valid
    assert "too long" in reason


def test_validate_stage_e_allows_long_summary():
    p90 = tu._STAGE_LENGTH_BUDGET["e"][2]
    text = "今天讨论的全部内容总结如下" + "好" * (2 * p90) + "，你可以回答题目了吗？"
    valid, _ = tu._validate_teacher_output(text, "e1")
    assert valid


# ─── #8 style critic ─────────────────────────────────────────────────────────


def test_style_score_empty_is_zero():
    assert tu._style_score("", "a1") == 0


def test_style_score_perfect_response_scores_three():
    opener = tu._TOP_OPENER_4GRAMS[0]
    p10, _, p90 = tu._STAGE_LENGTH_BUDGET["a"]
    text = opener + "么" * max(0, p10 - len(opener)) + "？"
    assert p10 <= len(text) <= p90
    assert tu._style_score(text, "a1") == 3


# ─── #10 compressed history ──────────────────────────────────────────────────


def test_compress_history_empty_returns_input():
    client = FakeClient([])
    assert tu._compress_history(client, "m", "   ") == "   "
    assert client.calls == []


def test_compress_history_summarizes():
    client = FakeClient(["学生理解了光合作用"])
    out = tu._compress_history(client, "m", "学生: 我不懂\n老师: 为什么？")
    assert out == "对话摘要：学生理解了光合作用"
    assert len(client.calls) == 1


def test_compress_history_falls_back_on_exception():
    client = FakeClient([RuntimeError("boom")])
    history = "学生: 我不懂"
    assert tu._compress_history(client, "m", history) == history


def test_compress_history_falls_back_on_empty_summary():
    client = FakeClient([""])
    history = "学生: 我不懂"
    assert tu._compress_history(client, "m", history) == history


# ─── apply_pre_call dispatch ─────────────────────────────────────────────────


def test_apply_pre_call_noop_by_default():
    sp, ui = tu.apply_pre_call("SYS", "USER", "b2", FakeClient([]), "m", "HIST")
    assert (sp, ui) == ("SYS", "USER")


def test_apply_pre_call_persona_then_budget(monkeypatch):
    monkeypatch.setenv("KELE_TEACHER_PERSONA", "1")
    monkeypatch.setenv("KELE_STAGE_LENGTH_BUDGET", "1")
    sp, _ = tu.apply_pre_call("SYS", "USER", "b2", FakeClient([]), "m", "HIST")
    assert sp.startswith(tu._PERSONA_BLOCK)
    assert "字之间" in sp


def test_apply_pre_call_compressed_history_rewrites_user(monkeypatch):
    monkeypatch.setenv("KELE_COMPRESSED_HISTORY", "1")
    client = FakeClient(["简短摘要"])
    sp, ui = tu.apply_pre_call("SYS", "前缀 HIST 后缀", "b2", client, "m", "HIST")
    assert ui == "前缀 对话摘要：简短摘要 后缀"


def test_apply_pre_call_per_state_appends_block(monkeypatch):
    monkeypatch.setenv("KELE_PER_STATE_EXEMPLARS", "1")
    by_state = {"b2": [_turn("b2") for _ in range(3)]}
    monkeypatch.setattr(tu, "_load_train_turns_indexed", lambda: (by_state, {}))
    sp, _ = tu.apply_pre_call("SYS", "USER", "b2", FakeClient([]), "m", "HIST")
    assert "state=b2" in sp


def test_apply_pre_call_style_matched_appends_block(monkeypatch):
    monkeypatch.setenv("KELE_STYLE_MATCHED_EXEMPLARS", "1")
    pool = [_turn("a1", teacher=f"问题{i}？") for i in range(12)]
    monkeypatch.setattr(tu, "_load_train_turns_indexed", lambda: ({}, {"a": pool}))
    sp, _ = tu.apply_pre_call("SYS", "USER", "a1", FakeClient([]), "m", "HIST")
    assert "风格相似度" in sp


# ─── call_teacher_wrapped dispatch ───────────────────────────────────────────


def test_call_teacher_default_single_call():
    client = FakeClient(["回答？"])
    out = tu.call_teacher_wrapped(client, "m", "SYS", "USER", "b2")
    assert out == "回答？"
    assert len(client.calls) == 1


def test_call_teacher_format_retry_accepts_valid_first(monkeypatch):
    monkeypatch.setenv("KELE_FORMAT_RETRY", "1")
    client = FakeClient(["叶子在做什么呢？"])
    out = tu.call_teacher_wrapped(client, "m", "SYS", "USER", "b2")
    assert out == "叶子在做什么呢？"
    assert len(client.calls) == 1


def test_call_teacher_format_retry_retries_invalid(monkeypatch):
    monkeypatch.setenv("KELE_FORMAT_RETRY", "1")
    client = FakeClient(["好的，让我解释一下。", "叶子在做什么呢？"])
    out = tu.call_teacher_wrapped(client, "m", "SYS", "USER", "b2")
    assert out == "叶子在做什么呢？"
    assert len(client.calls) == 2


def test_format_retry_falls_back_to_original_on_retry_exception(monkeypatch):
    monkeypatch.setenv("KELE_FORMAT_RETRY", "1")
    client = FakeClient(["好的，让我解释一下。", RuntimeError("boom")])
    out = tu.call_teacher_wrapped(client, "m", "SYS", "USER", "b2")
    assert out == "好的，让我解释一下。"


def test_call_teacher_cot_makes_two_calls(monkeypatch):
    monkeypatch.setenv("KELE_TEACHER_COT", "1")
    client = FakeClient(["内部分析", "最终问题？"])
    out = tu.call_teacher_wrapped(client, "m", "SYS", "USER", "b2")
    assert out == "最终问题？"
    assert len(client.calls) == 2


def test_call_teacher_nbest_picks_highest_style_score(monkeypatch):
    monkeypatch.setenv("KELE_NBEST_RERANK", "3")
    opener = tu._TOP_OPENER_4GRAMS[0]
    p10 = tu._STAGE_LENGTH_BUDGET["a"][0]
    best = opener + "么" * max(0, p10 - len(opener)) + "？"
    client = FakeClient(["短？", best, "好的，前言太多了？"])
    out = tu.call_teacher_wrapped(client, "m", "SYS", "USER", "a1")
    assert out == best
    assert len(client.calls) == 3


def test_call_teacher_nbest_invalid_n_defaults_to_three(monkeypatch):
    monkeypatch.setenv("KELE_NBEST_RERANK", "notanumber")
    client = FakeClient(["a？", "b？", "c？"])
    tu.call_teacher_wrapped(client, "m", "SYS", "USER", "a1")
    assert len(client.calls) == 3


def test_nbest_falls_back_when_all_candidates_fail(monkeypatch):
    monkeypatch.setenv("KELE_NBEST_RERANK", "2")
    client = FakeClient([RuntimeError("x"), RuntimeError("y"), "兜底回答？"])
    out = tu.call_teacher_wrapped(client, "m", "SYS", "USER", "a1")
    assert out == "兜底回答？"
    assert len(client.calls) == 3
