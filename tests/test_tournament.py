"""Tests for the tournament controller (pure-Python parts only).

Server lifecycle, subprocess calls, and hardware-dependent functions are
excluded from coverage via # pragma: no cover in tournament.py.
"""

import json
from argparse import Namespace
from io import StringIO

import pytest

from src.project.tournament import (
    MODEL_BY_ID,
    MODEL_REGISTRY,
    TournamentState,
    _build_report_markdown,
    _server_has_alias,
    cmd_add,
    cmd_download,
    cmd_eliminate,
    cmd_reset,
    cmd_status,
    load_state,
    print_leaderboard,
    save_state,
)

# ── Registry ──────────────────────────────────────────────────────────────────


def test_registry_has_13_models():
    assert len(MODEL_REGISTRY) == 13


def test_registry_has_13_on_disk_models():
    on_disk = [m for m in MODEL_REGISTRY if m.on_disk]
    assert len(on_disk) == 13
    assert {m.id for m in on_disk} == {
        "qwen35-9b",
        "qwen27b",
        "qwen27b-q4",
        "qwen35b-a3b",
        "gemma4-31b",
        "gemma4-26b-a4b",
        "glm47-23b",
        "qwopus35b-a3b",
        "deepseek-r1-14b",
        "phi4-14b",
        "gemma3-27b",
        "mistral-24b",
        "qwen3-14b",
    }


def test_registry_has_0_downloadable_models():
    downloadable = [m for m in MODEL_REGISTRY if not m.on_disk]
    assert len(downloadable) == 0


def test_model_by_id_lookup():
    assert MODEL_BY_ID["qwen27b"].alias == "Qwen 27B Q5"
    assert MODEL_BY_ID["gemma4-31b"].alias == "Gemma 4 31B"


def test_all_models_have_hf_repo_and_file():
    for m in MODEL_REGISTRY:
        assert m.hf_repo, f"{m.id} missing hf_repo"
        assert m.hf_file, f"{m.id} missing hf_file"


# ── Server alias probe ────────────────────────────────────────────────────────


def test_server_has_alias_match():
    response = '{"data": [{"id": "Qwen 27B Q5"}]}'
    assert _server_has_alias(response, "Qwen 27B Q5") is True


def test_server_has_alias_no_match():
    response = '{"data": [{"id": "Gemma 4 31B"}]}'
    assert _server_has_alias(response, "Qwen 27B Q5") is False


def test_server_has_alias_partial_no_match():
    # Partial name should not match
    response = '{"data": [{"id": "Qwen 27B"}]}'
    assert _server_has_alias(response, "Qwen 27B Q5") is False


# ── State persistence ─────────────────────────────────────────────────────────


def test_load_state_default_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = load_state()
    assert state.round == 0
    assert set(state.models_active) == {
        "qwen35-9b",
        "qwen27b",
        "qwen27b-q4",
        "qwen35b-a3b",
        "gemma4-31b",
        "gemma4-26b-a4b",
        "glm47-23b",
        "qwopus35b-a3b",
        "deepseek-r1-14b",
        "phi4-14b",
        "gemma3-27b",
        "mistral-24b",
        "qwen3-14b",
    }
    assert state.models_eliminated == []


def test_save_and_load_state_roundtrip(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("src.project.tournament.STATE_FILE", state_file)

    original = TournamentState(
        round=2,
        n_per_round=50,
        models_active=["qwen27b", "gemma4-31b"],
        models_eliminated=["qwen35-9b"],
        scores={"qwen27b": {"round1": 0.85, "round2": 0.88}, "gemma4-31b": {"round1": 0.80}},
        round_complete=[],
        final_scores={},
    )
    save_state(original)

    loaded = load_state()
    assert loaded.round == 2
    assert loaded.models_active == ["qwen27b", "gemma4-31b"]
    assert loaded.models_eliminated == ["qwen35-9b"]
    assert loaded.scores["qwen27b"]["round2"] == pytest.approx(0.88)


def test_save_state_creates_parent_dir(tmp_path, monkeypatch):
    state_file = tmp_path / "nested" / "dir" / "state.json"
    monkeypatch.setattr("src.project.tournament.STATE_FILE", state_file)
    save_state(load_state())
    assert state_file.exists()


# ── Leaderboard ───────────────────────────────────────────────────────────────


def test_print_leaderboard_no_scores(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = load_state()
    print_leaderboard(state)
    out = capsys.readouterr().out
    assert "Qwen3.5-9B Q4" in out
    assert "active" in out


def test_print_leaderboard_with_scores_and_eliminated(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = TournamentState(
        round=1,
        n_per_round=50,
        models_active=["qwen27b", "gemma4-31b"],
        models_eliminated=["qwen35-9b"],
        scores={
            "qwen27b": {"round1": 0.85},
            "gemma4-31b": {"round1": 0.80},
            "qwen35-9b": {"round1": 0.65},
        },
        round_complete=[],
        final_scores={},
    )
    print_leaderboard(state)
    out = capsys.readouterr().out
    assert "0.850" in out
    assert "ELIMINATED" in out
    # Higher score should appear before lower score in active section
    assert out.index("0.850") < out.index("0.800")


# ── cmd_status ────────────────────────────────────────────────────────────────


def test_cmd_status_no_data_prints_message(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    # Force empty scores to trigger the "no data" branch
    state_file = tmp_path / "state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(
            {
                "round": 0,
                "n_per_round": 50,
                "models_active": [],
                "models_eliminated": [],
                "scores": {},
                "round_complete": [],
                "final_scores": {},
            }
        )
    )
    cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "No tournament data" in out


def test_cmd_status_with_default_state(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "active" in out


# ── cmd_eliminate ─────────────────────────────────────────────────────────────


def _make_scored_state(tmp_path, monkeypatch) -> TournamentState:
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = TournamentState(
        round=1,
        n_per_round=50,
        models_active=["qwen35-9b", "qwen27b", "qwen35b-a3b", "gemma4-31b"],
        models_eliminated=[],
        scores={
            "qwen35-9b": {"round1": 0.60},
            "qwen27b": {"round1": 0.85},
            "qwen35b-a3b": {"round1": 0.78},
            "gemma4-31b": {"round1": 0.82},
        },
        round_complete=[],
        final_scores={},
    )
    save_state(state)
    return state


def test_cmd_eliminate_drops_worst(capsys, tmp_path, monkeypatch):
    _make_scored_state(tmp_path, monkeypatch)
    cmd_eliminate(Namespace(n=1))
    state = load_state()
    assert "qwen35-9b" not in state.models_active
    assert "qwen35-9b" in state.models_eliminated
    assert len(state.models_active) == 3


def test_cmd_eliminate_respects_floor_of_3(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = TournamentState(
        round=1,
        n_per_round=50,
        models_active=["qwen27b", "gemma4-31b", "qwen35b-a3b"],
        models_eliminated=["qwen35-9b"],
        scores={
            "qwen27b": {"round1": 0.85},
            "gemma4-31b": {"round1": 0.82},
            "qwen35b-a3b": {"round1": 0.78},
        },
        round_complete=[],
        final_scores={},
    )
    save_state(state)
    cmd_eliminate(Namespace(n=1))
    out = capsys.readouterr().out
    assert "Only 3" in out
    # Nothing should be eliminated
    state = load_state()
    assert len(state.models_active) == 3


def test_cmd_eliminate_warns_when_unscored(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = TournamentState(
        round=1,
        n_per_round=50,
        models_active=["qwen27b", "gemma4-31b", "qwen35b-a3b", "qwen35-9b"],
        models_eliminated=[],
        scores={"qwen27b": {"round1": 0.85}},  # others have no score
        round_complete=[],
        final_scores={},
    )
    save_state(state)
    cmd_eliminate(Namespace(n=1))
    out = capsys.readouterr().out
    assert "WARNING" in out


def test_cmd_eliminate_drop_2_caps_at_floor(capsys, tmp_path, monkeypatch):
    # 4 models, n=2: floor of 3 means only 1 gets eliminated (min(2, 4-3) = 1)
    _make_scored_state(tmp_path, monkeypatch)
    cmd_eliminate(Namespace(n=2))
    state = load_state()
    assert len(state.models_active) == 3
    assert len(state.models_eliminated) == 1


# ── cmd_add ───────────────────────────────────────────────────────────────────


def test_cmd_add_single_model(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = TournamentState(
        round=1,
        n_per_round=5,
        models_active=["qwen35-9b"],
        models_eliminated=[],
        scores={},
        round_complete=[],
        final_scores={},
    )
    save_state(state)
    cmd_add(Namespace(model_id="qwen27b-q4", all=False))
    state = load_state()
    assert "qwen27b-q4" in state.models_active
    assert len(state.models_active) == 2


def test_cmd_add_all_adds_on_disk_models(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = TournamentState(
        round=1,
        n_per_round=5,
        models_active=["qwen35-9b"],
        models_eliminated=["qwen27b"],
        scores={},
        round_complete=[],
        final_scores={},
    )
    save_state(state)
    cmd_add(Namespace(model_id=None, all=True))
    state = load_state()
    # All on_disk models except already-active and eliminated should be added
    on_disk_ids = {m.id for m in MODEL_REGISTRY if m.on_disk}
    expected = on_disk_ids - {"qwen35-9b", "qwen27b"}
    assert expected.issubset(set(state.models_active))
    assert "qwen27b" not in state.models_active  # eliminated, not re-added


def test_cmd_add_unknown_model_prints_error(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    cmd_add(Namespace(model_id="does-not-exist", all=False))
    out = capsys.readouterr().out
    assert "Unknown" in out


def test_cmd_add_already_active_skips(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = TournamentState(
        round=1,
        n_per_round=5,
        models_active=["qwen35-9b"],
        models_eliminated=[],
        scores={},
        round_complete=[],
        final_scores={},
    )
    save_state(state)
    cmd_add(Namespace(model_id="qwen35-9b", all=False))
    out = capsys.readouterr().out
    assert "already active" in out
    assert len(load_state().models_active) == 1


def test_cmd_add_eliminated_cannot_readd(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    state = TournamentState(
        round=1,
        n_per_round=5,
        models_active=["qwen35-9b"],
        models_eliminated=["qwen27b"],
        scores={},
        round_complete=[],
        final_scores={},
    )
    save_state(state)
    cmd_add(Namespace(model_id="qwen27b", all=False))
    out = capsys.readouterr().out
    assert "eliminated" in out
    assert "qwen27b" not in load_state().models_active


# ── cmd_reset ─────────────────────────────────────────────────────────────────


def test_cmd_reset_without_confirm_prints_warning(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    cmd_reset(Namespace(confirm=False))
    out = capsys.readouterr().out
    assert "--confirm" in out


def test_cmd_reset_with_confirm_removes_state_file(tmp_path, monkeypatch, capsys):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr("src.project.tournament.STATE_FILE", state_file)
    save_state(load_state())
    assert state_file.exists()
    cmd_reset(Namespace(confirm=True))
    assert not state_file.exists()


def test_cmd_reset_with_confirm_no_file_is_noop(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.project.tournament.STATE_FILE", tmp_path / "state.json")
    cmd_reset(Namespace(confirm=True))  # file doesn't exist — should not raise


# ── cmd_download ──────────────────────────────────────────────────────────────


def _fake_registry(tmp_path):
    """Two-entry registry: one file present, one missing."""
    from src.project.tournament import ModelSpec

    present = ModelSpec(
        id="fake-present",
        name="Fake Present",
        alias="FP",
        weight_path=str(tmp_path / "present.gguf"),
        serve_script="scripts/serve_fake.sh",
        config_name="fake",
        hf_repo="test/fake-present",
        hf_file="present.gguf",
        on_disk=True,
    )
    missing = ModelSpec(
        id="fake-missing",
        name="Fake Missing",
        alias="FM",
        weight_path=str(tmp_path / "missing.gguf"),
        serve_script="scripts/serve_fake.sh",
        config_name="fake",
        hf_repo="test/fake-missing",
        hf_file="missing.gguf",
        on_disk=False,
    )
    (tmp_path / "present.gguf").touch()
    return [present, missing]


def test_cmd_download_generates_command_for_missing_file(capsys, tmp_path, monkeypatch):
    registry = _fake_registry(tmp_path)
    monkeypatch.setattr("src.project.tournament.MODEL_REGISTRY", registry)
    cmd_download(Namespace(dry_run=True))
    out = capsys.readouterr().out
    assert "test/fake-missing" in out
    assert "hf download" in out


def test_cmd_download_skips_present_file(capsys, tmp_path, monkeypatch):
    registry = _fake_registry(tmp_path)
    monkeypatch.setattr("src.project.tournament.MODEL_REGISTRY", registry)
    cmd_download(Namespace(dry_run=True))
    out = capsys.readouterr().out
    assert "[fake-present] already present" in out
    assert "test/fake-present" not in out


def test_cmd_download_all_present_prints_message(capsys, tmp_path, monkeypatch):
    from src.project.tournament import ModelSpec

    present = ModelSpec(
        id="only-present",
        name="Only Present",
        alias="OP",
        weight_path=str(tmp_path / "only.gguf"),
        serve_script="scripts/serve_fake.sh",
        config_name="fake",
        hf_repo="test/only-present",
        hf_file="only.gguf",
        on_disk=True,
    )
    (tmp_path / "only.gguf").touch()
    monkeypatch.setattr("src.project.tournament.MODEL_REGISTRY", [present])
    cmd_download(Namespace(dry_run=True))
    out = capsys.readouterr().out
    assert "present locally" in out


# ── _build_report_markdown ────────────────────────────────────────────────────


def _make_metrics(overall: float, n_turns: int, per_stage: dict) -> dict:
    return {
        "n_turns": n_turns,
        "rouge1": 30.0,
        "rouge2": 12.0,
        "rougeL": 22.0,
        "bleu4": 6.0,
        "state_accuracy": {"overall": overall, "per_stage": per_stage, "total_turns": n_turns},
    }


def test_build_report_markdown_cumulative_math():
    stage = {"a": 90.0, "b": 40.0, "c": 15.0, "d": 25.0, "e": 60.0}
    state = TournamentState(
        round=2,
        n_per_round=50,
        models_active=["qwen27b-q4", "gemma4-31b"],
        models_eliminated=[],
        scores={
            "qwen27b-q4": {"round1": 43.0, "round2": 42.0},
            "gemma4-31b": {"round1": 38.0, "round2": 40.0},
        },
        round_complete=[],
        final_scores={},
        metrics={
            "qwen27b-q4": {
                "round1": _make_metrics(43.0, 300, stage),
                "round2": _make_metrics(42.0, 300, stage),
            },
            "gemma4-31b": {
                "round1": _make_metrics(38.0, 300, stage),
                "round2": _make_metrics(40.0, 300, stage),
            },
        },
    )
    md = _build_report_markdown(state)
    # cumulative at n=100: qwen=(43*300+42*300)/600=42.50, gemma=(38*300+40*300)/600=39.00
    assert "42.50" in md
    assert "39.00" in md
    # per-round scores present
    assert "43.00" in md
    assert "40.00" in md
    # delta column: qwen round2 delta = 42.50-43.00 = -0.50
    assert "-0.50" in md
    # gap at n=100: qwen leads by 3.50
    assert "+3.50" in md
