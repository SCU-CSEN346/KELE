"""Unit tests for src/project/validate_translation.py.

Covers JSON-repair, structural checks (the Phase-1 gate), eval-result type
validation, Chinese-character detection, sampling, checkpoint I/O, and the
retry behaviour of eval_record (with a stubbed OpenAI client).

No network. All tests use synthetic records.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.project import validate_translation as vt

# ── Helpers ──────────────────────────────────────────────────────────────────


def _zh_record(rid: int, *, options=None, dialogue=None, mission="选择题", grade="g") -> dict:
    return {
        "id": rid,
        "mission": mission,
        "grade": grade,
        "question": f"Q{rid}",
        "options": options if options is not None else ["A", "B"],
        "newHint": "h",
        "newKnowledgePoint": "k",
        "newAnalyze": "a",
        "dialogue": dialogue
        if dialogue is not None
        else [
            {
                "student": "s",
                "teacher": "t",
                "evaluation": "e",
                "state": "a1",
                "action": "ask",
            }
        ],
    }


def _en_record(rid: int, *, options=None, dialogue=None) -> dict:
    return {
        "id": rid,
        "question": f"Q{rid} EN",
        "options": options if options is not None else ["A", "B"],
        "newHint": "h",
        "newKnowledgePoint": "k",
        "newAnalyze": "a",
        "dialogue": dialogue
        if dialogue is not None
        else [
            {
                "student": "S",
                "teacher": "T",
                "evaluation": "E",
                "state": "a1",
                "action": "ask",
            }
        ],
    }


# ── _extra ───────────────────────────────────────────────────────────────────


def test_extra_returns_enable_thinking_false_when_budget_zero(monkeypatch):
    monkeypatch.setattr(vt, "THINKING_BUDGET", 0)
    assert vt._extra() == {"chat_template_kwargs": {"enable_thinking": False}}


def test_extra_returns_thinking_budget_when_positive(monkeypatch):
    monkeypatch.setattr(vt, "THINKING_BUDGET", 1024)
    assert vt._extra() == {"chat_template_kwargs": {"thinking_budget": 1024}}


# ── _strip_fences / _parse_json ──────────────────────────────────────────────


def test_strip_fences_removes_json_block():
    assert vt._strip_fences('```json\n{"a":1}\n```') == '{"a":1}'


def test_strip_fences_removes_bare_fence():
    assert vt._strip_fences('```\n{"a":1}\n```') == '{"a":1}'


def test_parse_json_clean_payload():
    assert vt._parse_json('{"overall_score": 4}') == {"overall_score": 4}


def test_parse_json_with_trailing_commentary_uses_raw_decode():
    """raw_decode should accept the longest valid JSON prefix."""
    out = vt._parse_json('{"a": 1, "b": 2} and then some commentary')
    assert out == {"a": 1, "b": 2}


def test_parse_json_strips_markdown_fences():
    assert vt._parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_repairs_missing_closing_brace():
    """Suffix '}' should close a truncated-before-brace payload."""
    assert vt._parse_json('{"a": 1, "flags": []') == {"a": 1, "flags": []}


def test_parse_json_repairs_missing_array_and_brace():
    """Suffix ']}' should close a payload truncated after an array's last item."""
    assert vt._parse_json('{"a": 1, "flags": ["x"') == {"a": 1, "flags": ["x"]}


def test_parse_json_repairs_truncated_mid_string_in_array():
    """Suffix '\"]}' was the typo-fix case — close unfinished string + array + obj."""
    assert vt._parse_json('{"a": 1, "flags": ["bad') == {"a": 1, "flags": ["bad"]}


def test_parse_json_unrepairable_raises():
    with pytest.raises(json.JSONDecodeError):
        vt._parse_json("not json at all {")


# ── _validate_eval_result ────────────────────────────────────────────────────


def _good_eval() -> dict:
    return {
        "overall_score": 4,
        "meaning_preserved": True,
        "socratic_tone_preserved": False,
        "fluency": 5,
        "flags": ["minor"],
    }


def test_validate_eval_result_accepts_good_payload():
    payload = _good_eval()
    assert vt._validate_eval_result(payload) is payload


def test_validate_eval_result_rejects_missing_keys():
    payload = _good_eval()
    del payload["fluency"]
    with pytest.raises(ValueError, match="missing keys"):
        vt._validate_eval_result(payload)


def test_validate_eval_result_rejects_string_score():
    payload = _good_eval()
    payload["overall_score"] = "5"
    with pytest.raises(ValueError, match="overall_score"):
        vt._validate_eval_result(payload)


def test_validate_eval_result_rejects_bool_as_int_score():
    """bool is a subclass of int — must be explicitly excluded."""
    payload = _good_eval()
    payload["overall_score"] = True
    with pytest.raises(ValueError, match="overall_score"):
        vt._validate_eval_result(payload)


def test_validate_eval_result_rejects_out_of_range_score():
    payload = _good_eval()
    payload["overall_score"] = 6
    with pytest.raises(ValueError, match="overall_score"):
        vt._validate_eval_result(payload)


def test_validate_eval_result_rejects_string_meaning_preserved():
    payload = _good_eval()
    payload["meaning_preserved"] = "true"
    with pytest.raises(ValueError, match="meaning_preserved"):
        vt._validate_eval_result(payload)


def test_validate_eval_result_rejects_non_list_flags():
    payload = _good_eval()
    payload["flags"] = "none"
    with pytest.raises(ValueError, match="flags"):
        vt._validate_eval_result(payload)


def test_validate_eval_result_rejects_float_fluency():
    payload = _good_eval()
    payload["fluency"] = 4.5
    with pytest.raises(ValueError, match="fluency"):
        vt._validate_eval_result(payload)


def test_validate_eval_result_rejects_non_bool_socratic_tone():
    payload = _good_eval()
    payload["socratic_tone_preserved"] = 1
    with pytest.raises(ValueError, match="socratic_tone_preserved"):
        vt._validate_eval_result(payload)


# ── _is_flagged ──────────────────────────────────────────────────────────────


def test_is_flagged_low_score():
    assert vt._is_flagged(
        {"overall_score": 2, "meaning_preserved": True, "socratic_tone_preserved": True}
    )


def test_is_flagged_meaning_lost():
    assert vt._is_flagged(
        {"overall_score": 5, "meaning_preserved": False, "socratic_tone_preserved": True}
    )


def test_is_flagged_tone_lost():
    assert vt._is_flagged(
        {"overall_score": 5, "meaning_preserved": True, "socratic_tone_preserved": False}
    )


def test_is_flagged_clean_passes():
    assert not vt._is_flagged(
        {"overall_score": 4, "meaning_preserved": True, "socratic_tone_preserved": True}
    )


# ── _has_chinese_in_en ───────────────────────────────────────────────────────


def test_has_chinese_in_en_clean_returns_false():
    assert vt._has_chinese_in_en(_en_record(1)) is False


def test_has_chinese_in_en_detects_row_field():
    rec = _en_record(1)
    rec["newHint"] = "Think 思考 about this"
    assert vt._has_chinese_in_en(rec) is True


def test_has_chinese_in_en_detects_options():
    rec = _en_record(1, options=["A", "选项"])
    assert vt._has_chinese_in_en(rec) is True


def test_has_chinese_in_en_detects_dialogue_teacher():
    rec = _en_record(1)
    rec["dialogue"][0]["teacher"] = "Hello 同学"
    assert vt._has_chinese_in_en(rec) is True


def test_has_chinese_in_en_detects_action():
    """Bug fix: action is translated via cache, must be checked too."""
    rec = _en_record(1)
    rec["dialogue"][0]["action"] = "生成一个问题"
    assert vt._has_chinese_in_en(rec) is True


# ── run_structural_checks ────────────────────────────────────────────────────


def test_structural_happy_path():
    zh = [_zh_record(1), _zh_record(2)]
    en = {1: _en_record(1), 2: _en_record(2)}
    res = vt.run_structural_checks(zh, en)
    assert res.passed
    assert res.failures == {}


def test_structural_id_coverage_distinguishes_missing_and_extra():
    zh = [_zh_record(1), _zh_record(2)]
    en = {1: _en_record(1), 3: _en_record(3)}  # 2 missing in EN, 3 extra in EN
    res = vt.run_structural_checks(zh, en)
    assert not res.passed
    assert res.failures["id_coverage"] == {"missing_in_en": [2], "extra_in_en": [3]}


def test_structural_field_completeness_catches_empty_row_field():
    zh = [_zh_record(1)]
    en_rec = _en_record(1)
    en_rec["newAnalyze"] = ""
    res = vt.run_structural_checks(zh, {1: en_rec})
    assert res.failures.get("field_completeness") == [1]


def test_structural_field_completeness_catches_empty_turn_field():
    zh = [_zh_record(1)]
    en_rec = _en_record(1)
    en_rec["dialogue"][0]["teacher"] = ""
    res = vt.run_structural_checks(zh, {1: en_rec})
    assert res.failures.get("field_completeness") == [1]


def test_structural_field_completeness_single_append_per_record():
    """for/else guard: a record failing the row check should not be re-appended for turns."""
    zh = [_zh_record(1)]
    en_rec = _en_record(1)
    en_rec["newHint"] = ""
    en_rec["dialogue"][0]["teacher"] = ""  # would-also-fail the turn check
    res = vt.run_structural_checks(zh, {1: en_rec})
    assert res.failures.get("field_completeness") == [1]  # not [1, 1]


def test_structural_state_preservation_mismatch():
    zh = [_zh_record(1)]
    en_rec = _en_record(1)
    en_rec["dialogue"][0]["state"] = "b2"  # ZH was "a1"
    res = vt.run_structural_checks(zh, {1: en_rec})
    assert res.failures.get("state_preservation") == [1]


def test_structural_option_count_mismatch():
    """Reproduces the real-world record 5240 bug — EN options dropped one entry."""
    zh = [_zh_record(5240, options=["①②③", "①②", "③④"])]
    en = {5240: _en_record(5240, options=["①②③", "①②"])}
    res = vt.run_structural_checks(zh, en)
    assert res.failures.get("option_count") == [5240]


def test_structural_dialogue_round_count_mismatch():
    zh = [_zh_record(1)]
    en_rec = _en_record(1)
    en_rec["dialogue"].append(
        {"student": "S2", "teacher": "T2", "evaluation": "E2", "state": "b2", "action": "ask"}
    )
    res = vt.run_structural_checks(zh, {1: en_rec})
    assert res.failures.get("dialogue_round_count") == [1]


def test_structural_no_chinese_in_en_catches_action():
    zh = [_zh_record(1)]
    en_rec = _en_record(1)
    en_rec["dialogue"][0]["action"] = "生成问题"
    res = vt.run_structural_checks(zh, {1: en_rec})
    assert res.failures.get("no_chinese_in_en") == [1]


# ── StructuralResult._fail_count / report ────────────────────────────────────


def test_fail_count_zero_when_check_absent():
    r = vt.StructuralResult(passed=True, failures={})
    assert r._fail_count("id_coverage") == 0


def test_fail_count_for_list_of_ids():
    r = vt.StructuralResult(passed=False, failures={"state_preservation": [1, 2, 3]})
    assert r._fail_count("state_preservation") == 3


def test_fail_count_for_id_coverage_dict():
    r = vt.StructuralResult(
        passed=False,
        failures={"id_coverage": {"missing_in_en": [1, 2], "extra_in_en": [9]}},
    )
    assert r._fail_count("id_coverage") == 3


def test_report_prints_all_pass_when_clean(capsys):
    vt.StructuralResult(passed=True, failures={}).report()
    out = capsys.readouterr().out
    assert "All checks passed." in out
    assert "FAIL" not in out


def test_report_prints_per_check_fail_counts(capsys):
    vt.StructuralResult(
        passed=False,
        failures={
            "id_coverage": {"missing_in_en": [2], "extra_in_en": [9]},
            "state_preservation": [5, 7],
        },
    ).report()
    out = capsys.readouterr().out
    assert "id_coverage" in out and "FAIL (2 records)" in out
    assert "state_preservation" in out and "FAIL (2 records)" in out
    assert "option_count" in out and "PASS" in out


# ── stratified_sample ────────────────────────────────────────────────────────


def test_stratified_sample_odd_shard_filters_to_odd_ids():
    records = [_zh_record(i) for i in range(1, 11)]
    sample = vt.stratified_sample(records, 1.0, seed=42, shard="odd")
    assert all(r["id"] % 2 == 1 for r in sample)
    assert {r["id"] for r in sample} == {1, 3, 5, 7, 9}


def test_stratified_sample_even_shard_filters_to_even_ids():
    records = [_zh_record(i) for i in range(1, 11)]
    sample = vt.stratified_sample(records, 1.0, seed=42, shard="even")
    assert all(r["id"] % 2 == 0 for r in sample)


def test_stratified_sample_all_shard_keeps_everything():
    records = [_zh_record(i) for i in range(1, 11)]
    sample = vt.stratified_sample(records, 1.0, seed=42, shard="all")
    assert len(sample) == 10


def test_stratified_sample_proportional_per_stratum():
    records = [_zh_record(i, mission="m1") for i in range(1, 21)] + [
        _zh_record(i, mission="m2") for i in range(21, 41)
    ]
    sample = vt.stratified_sample(records, 0.5, seed=42, shard="all")
    m1 = sum(1 for r in sample if r["mission"] == "m1")
    m2 = sum(1 for r in sample if r["mission"] == "m2")
    assert m1 == 10 and m2 == 10


def test_stratified_sample_min_one_per_stratum():
    """Tiny strata are guaranteed at least one record."""
    records = [_zh_record(1, mission="m1"), _zh_record(2, mission="m2")]
    sample = vt.stratified_sample(records, 0.01, seed=42, shard="all")
    assert len(sample) == 2


def test_stratified_sample_deterministic_under_seed():
    records = [_zh_record(i, grade=f"g{i % 3}") for i in range(1, 51)]
    a = vt.stratified_sample(records, 0.3, seed=42, shard="all")
    b = vt.stratified_sample(records, 0.3, seed=42, shard="all")
    assert [r["id"] for r in a] == [r["id"] for r in b]


# ── Checkpoint I/O ───────────────────────────────────────────────────────────


def test_load_checkpoint_returns_empty_when_missing(tmp_path: Path):
    processed, results = vt._load_checkpoint(tmp_path / "nope.json")
    assert processed == set()
    assert results == []


def test_checkpoint_roundtrip(tmp_path: Path):
    p = tmp_path / "ck.json"
    vt._save_checkpoint(p, {1, 2, 3}, [{"id": 1, "score": {}}])
    processed, results = vt._load_checkpoint(p)
    assert processed == {1, 2, 3}
    assert results == [{"id": 1, "score": {}}]


# ── merge_shards ─────────────────────────────────────────────────────────────


def test_merge_shards_writes_sorted_combined_and_flagged(tmp_path: Path, capsys):
    odd = tmp_path / "odd.json"
    even = tmp_path / "even.json"
    odd.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "score": {
                        "overall_score": 5,
                        "meaning_preserved": True,
                        "socratic_tone_preserved": True,
                        "fluency": 5,
                    },
                },
                {
                    "id": 3,
                    "score": {
                        "overall_score": 2,
                        "meaning_preserved": True,
                        "socratic_tone_preserved": True,
                        "fluency": 3,
                    },
                },
            ]
        )
    )
    even.write_text(
        json.dumps(
            [
                {
                    "id": 2,
                    "score": {
                        "overall_score": 4,
                        "meaning_preserved": False,
                        "socratic_tone_preserved": True,
                        "fluency": 4,
                    },
                },
            ]
        )
    )
    vt.merge_shards([str(odd), str(even)], tmp_path)
    combined = json.loads((tmp_path / "validate_llm_scores_all.json").read_text())
    flagged = json.loads((tmp_path / "validate_llm_flagged_all.json").read_text())
    assert [r["id"] for r in combined] == [1, 2, 3]
    assert sorted(r["id"] for r in flagged) == [2, 3]
    capsys.readouterr()  # drain summary stdout


# ── _print_summary ───────────────────────────────────────────────────────────


def _score_row(rid: int, **kwargs) -> dict:
    base = {
        "overall_score": 4,
        "meaning_preserved": True,
        "socratic_tone_preserved": True,
        "fluency": 4,
    }
    base.update(kwargs)
    return {"id": rid, "score": base}


def test_print_summary_handles_empty(capsys):
    vt._print_summary([], [])
    assert "No scores to summarise." in capsys.readouterr().out


def test_print_summary_renders_distribution_and_flag_rate(capsys):
    scores = [
        _score_row(1, overall_score=5),
        _score_row(2, overall_score=4),
        _score_row(3, overall_score=2),  # flagged: low score
    ]
    flagged = [s for s in scores if vt._is_flagged(s["score"])]
    vt._print_summary(scores, flagged)
    out = capsys.readouterr().out
    assert "Phase 2 summary (3 records)" in out
    assert "avg=" in out
    assert "flag_rate" in out
    assert "above threshold" in out  # 1/3 > 10%


# ── _ckpt_path ───────────────────────────────────────────────────────────────


def test_ckpt_path_includes_shard(tmp_path: Path):
    assert vt._ckpt_path(tmp_path, "odd").name == "validate_llm_checkpoint_odd.json"
    assert vt._ckpt_path(tmp_path, "all").name == "validate_llm_checkpoint_all.json"


# ── eval_record (with stubbed client) ────────────────────────────────────────


def _make_client(*responses) -> MagicMock:
    """Stub OpenAI client where chat.completions.create yields each response in order.

    Each `response` is either a JSON string (returned as the message content) or
    an Exception instance (raised when called).
    """
    client = MagicMock()
    calls: list = list(responses)

    def _create(**_kwargs):
        item = calls.pop(0)
        if isinstance(item, Exception):
            raise item
        msg = MagicMock()
        msg.content = item
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    client.chat.completions.create.side_effect = _create
    return client


def test_eval_record_success_on_first_attempt(monkeypatch):
    monkeypatch.setattr(vt.time, "sleep", lambda _s: None)
    payload = json.dumps(_good_eval())
    client = _make_client(payload)
    out = vt.eval_record(client, "m", _zh_record(1), _en_record(1))
    assert out["overall_score"] == 4


def test_eval_record_retries_after_malformed_then_succeeds(monkeypatch):
    monkeypatch.setattr(vt.time, "sleep", lambda _s: None)
    client = _make_client("not json", json.dumps(_good_eval()))
    out = vt.eval_record(client, "m", _zh_record(1), _en_record(1))
    assert out["overall_score"] == 4
    assert client.chat.completions.create.call_count == 2


def test_eval_record_retries_after_type_failure_then_succeeds(monkeypatch):
    """A schema-valid-keys but type-invalid response should drive a retry."""
    monkeypatch.setattr(vt.time, "sleep", lambda _s: None)
    bad = dict(_good_eval(), overall_score="4")
    client = _make_client(json.dumps(bad), json.dumps(_good_eval()))
    out = vt.eval_record(client, "m", _zh_record(1), _en_record(1))
    assert out["overall_score"] == 4
    assert client.chat.completions.create.call_count == 2


def test_eval_record_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(vt.time, "sleep", lambda _s: None)
    client = _make_client("garbage", "still garbage", "nope")
    with pytest.raises(RuntimeError, match="eval failed after 3 attempts"):
        vt.eval_record(client, "m", _zh_record(1), _en_record(1))
