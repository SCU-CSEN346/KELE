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
    _server_has_alias,
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


def test_registry_has_6_on_disk_models():
    on_disk = [m for m in MODEL_REGISTRY if m.on_disk]
    assert len(on_disk) == 6
    assert {m.id for m in on_disk} == {
        "qwen35-9b",
        "qwen27b",
        "qwen27b-q4",
        "qwen35b-a3b",
        "gemma4-31b",
        "glm47-23b",
    }


def test_registry_has_7_downloadable_models():
    downloadable = [m for m in MODEL_REGISTRY if not m.on_disk]
    assert len(downloadable) == 7


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
        "glm47-23b",
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


def test_cmd_download_prints_7_commands(capsys):
    cmd_download(Namespace(dry_run=True))
    out = capsys.readouterr().out
    # One hf download command per downloadable model
    assert out.count("hf download") == 7


def test_cmd_download_contains_all_model_repos(capsys):
    cmd_download(Namespace(dry_run=True))
    out = capsys.readouterr().out
    downloadable = [m for m in MODEL_REGISTRY if not m.on_disk]
    for m in downloadable:
        assert m.hf_repo in out
        assert m.hf_file in out
