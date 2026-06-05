from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.project.hf_callback import HFCheckpointCallback


def _state(step: int, epoch: float = 0.3, loss: float = 0.65, acc: float = 0.80):
    log = {"loss": loss, "mean_token_accuracy": acc}
    return SimpleNamespace(global_step=step, epoch=epoch, log_history=[log])


def _args(tmp_path, step: int):
    (tmp_path / f"checkpoint-{step}").mkdir()
    return SimpleNamespace(output_dir=str(tmp_path))


_ctrl = SimpleNamespace()


def test_skips_non_multiple_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    args = _args(tmp_path, 30)
    cb.on_save(args, _state(30), _ctrl)
    assert cb._thread is None


def test_pushes_at_multiple_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    args = _args(tmp_path, 50)
    with patch.object(cb, "_push") as mock_push:
        cb.on_save(args, _state(50), _ctrl)
        cb._thread.join()
        mock_push.assert_called_once()
        _, step, commit_msg = mock_push.call_args[0]
        assert step == 50
        assert "loss 0.6500" in commit_msg
        assert "acc 0.8000" in commit_msg


def test_skips_when_thread_alive(tmp_path, capsys):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    alive_thread = MagicMock()
    alive_thread.is_alive.return_value = True
    cb._thread = alive_thread
    with patch.object(cb, "_push") as mock_push:
        cb.on_save(_args(tmp_path, 50), _state(50), _ctrl)
        mock_push.assert_not_called()
    assert "skipping" in capsys.readouterr().out


def test_skips_when_checkpoint_dir_missing(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    args = SimpleNamespace(output_dir=str(tmp_path))
    cb.on_save(args, _state(50), _ctrl)
    assert cb._thread is None


def test_commit_msg_without_log(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    empty_state = SimpleNamespace(global_step=100, epoch=0.5, log_history=[])
    with patch.object(cb, "_push") as mock_push:
        cb.on_save(_args(tmp_path, 100), empty_state, _ctrl)
        cb._thread.join()
        _, step, commit_msg = mock_push.call_args[0]
        assert "step 100" in commit_msg
        assert "loss" not in commit_msg


def test_train_end_forces_push_at_non_multiple_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    with patch.object(cb, "_push") as mock_push:
        cb.on_train_end(_args(tmp_path, 37), _state(37), _ctrl)
        cb._thread.join()
        mock_push.assert_called_once()
        _, step, _ = mock_push.call_args[0]
        assert step == 37


def test_train_end_joins_alive_thread(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    cb._thread = mock_thread
    # no checkpoint dir → _maybe_push returns before spawning a new thread
    # but on_train_end still joins the existing alive thread
    args = SimpleNamespace(output_dir=str(tmp_path))
    cb.on_train_end(args, _state(37), _ctrl)
    mock_thread.join.assert_called_once()
