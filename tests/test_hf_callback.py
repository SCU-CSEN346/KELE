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


# ---------------------------------------------------------------------------
# Persistence — .hf_last_push file
# ---------------------------------------------------------------------------


def test_writes_and_reads_last_pushed(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    cb._output_dir = str(tmp_path)
    cb._write_last_pushed(100)
    assert cb._read_last_pushed() == 100
    cb._write_last_pushed(200)
    assert cb._read_last_pushed() == 200


def test_last_pushed_defaults_to_neg_one(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    assert cb._read_last_pushed() == -1


def test_writes_last_pushed_on_successful_push(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    cb._output_dir = str(tmp_path)
    with patch("huggingface_hub.HfApi.upload_folder"):
        cb._push(tmp_path, 50, "test")
    assert (tmp_path / ".hf_last_push").read_text() == "50"


# ---------------------------------------------------------------------------
# Step-skip guard — skip if already pushed
# ---------------------------------------------------------------------------


def test_skips_already_pushed_step(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    cb._output_dir = str(tmp_path)
    (tmp_path / ".hf_last_push").write_text("50")
    args = _args(tmp_path, 50)
    with patch.object(cb, "_push") as mock_push:
        cb.on_save(args, _state(50), _ctrl)
        mock_push.assert_not_called()


def test_force_push_bypasses_last_pushed(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    cb._output_dir = str(tmp_path)
    (tmp_path / ".hf_last_push").write_text("50")
    with patch.object(cb, "_push") as mock_push:
        cb.on_train_end(_args(tmp_path, 50), _state(50), _ctrl)
        mock_push.assert_called_once()


# ---------------------------------------------------------------------------
# Launch check — on_init
# ---------------------------------------------------------------------------


def test_on_init_pushes_latest_checkpoint(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    (tmp_path / "checkpoint-1000").mkdir()
    (tmp_path / "checkpoint-1200").mkdir()
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init(args, SimpleNamespace(), _ctrl)
        mock_push.assert_called_once()
        ckpt_dir, step, commit_msg = mock_push.call_args[0]
        assert step == 1200
        assert "resume push" in commit_msg


def test_on_init_skips_if_already_pushed(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    (tmp_path / "checkpoint-1200").mkdir()
    (tmp_path / ".hf_last_push").write_text("1200")
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init(args, SimpleNamespace(), _ctrl)
        mock_push.assert_not_called()


def test_on_init_skips_if_no_checkpoints(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init(args, SimpleNamespace(), _ctrl)
        mock_push.assert_not_called()


def test_on_init_picks_highest_step_numerically(tmp_path):
    cb = HFCheckpointCallback("repo/id", push_every=50)
    (tmp_path / "checkpoint-90").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    args = SimpleNamespace(output_dir=str(tmp_path))
    with patch.object(cb, "_push") as mock_push:
        cb.on_init(args, SimpleNamespace(), _ctrl)
        mock_push.assert_called_once()
        _, step, _ = mock_push.call_args[0]
        assert step == 100
