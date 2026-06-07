"""Unit tests for the SocraticTeachingSystem history/state bookkeeping.

The constructor only builds openai.Client objects (no network), so the system
instantiates cheaply with dummy credentials. These tests exercise the pure
session/history helpers, not the LLM-calling methods.
"""

import pytest

pytest.importorskip("openai")

import httpx
import openai

import src.project.socratic_teaching_system as stss
import src.project.tournament_utilizations as tu
from src.project.socratic_teaching_system import SocraticTeachingSystem


def make_system(**overrides):
    kwargs = dict(
        consultant_api_key="x",
        consultant_base_url="http://localhost:1/v1",
        consultant_model_name="consultant",
        teacher_api_key="x",
        teacher_base_url="http://localhost:2/v1",
        teacher_model_name="teacher",
    )
    kwargs.update(overrides)
    return SocraticTeachingSystem(**kwargs)


def test_reset_session_initial_state():
    sys = make_system()
    assert sys.current_state == "a0"
    assert sys.conversation_history == []
    assert sys.consultant_history == []
    assert sys.teaching_rounds == 0


def test_get_action_for_state_known_and_default():
    sys = make_system()
    assert sys.get_action_for_state("a0") == "引导学生提出问题"
    assert sys.get_action_for_state("nonexistent") == "继续提问"


def test_add_to_history_appends_role_and_content():
    sys = make_system()
    sys.add_to_history("student", "为什么天是蓝色的？")
    assert sys.conversation_history == [{"role": "student", "content": "为什么天是蓝色的？"}]


def test_get_formatted_history_pairs_and_drops_trailing_student():
    sys = make_system()
    sys.add_to_history("student", "问题一")
    sys.add_to_history("teacher", "回答一？")
    sys.add_to_history("student", "未配对的输入")

    out = sys.get_formatted_history()

    assert out == "学生: 问题一\n老师: 回答一？"
    assert "未配对" not in out


def test_get_full_formatted_history_includes_consultant_analysis():
    sys = make_system(max_teaching_rounds=5)
    sys.add_to_history("student", "问题一")
    sys.add_to_history("teacher", "回答一？")
    sys.teaching_rounds = 2
    sys.add_to_consultant_history("学生有误解", "b2", "换个角度提问")

    out = sys.get_full_formatted_history()

    assert "学生: 问题一" in out
    assert "[顾问分析]" in out
    assert "状态: b2" in out
    assert "行动: 换个角度提问" in out
    assert "教学阶段轮数: 2/5" in out


def test_get_full_formatted_history_omits_rounds_when_zero():
    sys = make_system()
    sys.add_to_history("student", "问题一")
    sys.add_to_history("teacher", "回答一？")
    sys.add_to_consultant_history("评估", "a1", "提问")  # teaching_rounds defaults to 0

    out = sys.get_full_formatted_history()

    assert "教学阶段轮数" not in out


def test_add_to_consultant_history_records_teaching_rounds():
    sys = make_system()
    sys.teaching_rounds = 3
    sys.add_to_consultant_history("评估", "c8", "提供反例")
    assert sys.consultant_history == [
        {
            "evaluation": "评估",
            "state": "c8",
            "action": "提供反例",
            "teaching_rounds": 3,
        }
    ]


def _rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


@pytest.fixture
def patched_teacher(monkeypatch):
    """Patch socrates_teacher's collaborators: no-op pre-call, no real sleep.

    Returns a dict the test fills with `call` (the call_teacher_wrapped stub)
    and reads `sleeps` from (the recorded backoff durations).
    """
    sleeps: list[float] = []
    monkeypatch.setattr(stss.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(tu, "apply_pre_call", lambda sp, ui, **kw: (sp, ui))
    state: dict = {"sleeps": sleeps}

    def install(fn):
        monkeypatch.setattr(tu, "call_teacher_wrapped", fn)

    state["install"] = install
    return state


def test_socrates_teacher_retries_then_succeeds(patched_teacher):
    calls = {"n": 0}

    def fake_call(client, model, system_prompt, user_input, predicted_state=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _rate_limit_error()
        return "TEACHER_OK"

    patched_teacher["install"](fake_call)

    sys = make_system()
    result = sys.socrates_teacher("hi", "eval", "ask")

    assert result == "TEACHER_OK"
    assert calls["n"] == 3
    assert patched_teacher["sleeps"] == [5, 10]  # exponential backoff: 5*2**attempt


def test_socrates_teacher_honors_retry_after_header(patched_teacher):
    calls = {"n": 0}

    def fake_call(client, model, system_prompt, user_input, predicted_state=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limit_error(retry_after="2")
        return "OK"

    patched_teacher["install"](fake_call)

    sys = make_system()
    result = sys.socrates_teacher("hi", "eval", "ask")

    assert result == "OK"
    assert patched_teacher["sleeps"] == [2.0]  # header wins over exponential backoff


def test_socrates_teacher_exhausts_retries_returns_fallback(patched_teacher):
    def fake_call(client, model, system_prompt, user_input, predicted_state=None):
        raise _rate_limit_error()

    patched_teacher["install"](fake_call)

    sys = make_system()
    result = sys.socrates_teacher("hi", "eval", "ask")

    assert result == "I need a moment to think. Please try again shortly."
    assert patched_teacher["sleeps"] == [5, 10, 20]  # 3 backoffs, 4th attempt re-raises


def test_socrates_teacher_language_override_appends_instruction(patched_teacher, monkeypatch):
    captured: dict = {}

    def fake_call(client, model, system_prompt, user_input, predicted_state=None):
        captured["sp"] = system_prompt
        return "ok"

    patched_teacher["install"](fake_call)
    sys = make_system()

    monkeypatch.setenv("KELE_TEACHER_LANG", "auto")
    sys.socrates_teacher("Why is the sky blue?", "eval", "ask")
    assert "Always respond in the same language" in captured["sp"]

    monkeypatch.delenv("KELE_TEACHER_LANG", raising=False)
    captured.clear()
    sys.socrates_teacher("Why is the sky blue?", "eval", "ask")
    assert "Always respond in the same language" not in captured["sp"]
