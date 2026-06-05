from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


class HFCheckpointCallback(TrainerCallback):
    """Push checkpoint-N to a HF model repo after every N-th save.

    Activated by TRAIN_HF_REPO=<repo_id>.  Cadence controlled by
    TRAIN_HF_PUSH_EVERY (default 50 steps).  Runs in a daemon thread so it
    never blocks the training loop; if the previous push is still in-flight
    when the next one fires, the new push is skipped and logged.  Also fires
    unconditionally at on_train_end to capture the final state.
    """

    def __init__(self, repo_id: str, push_every: int) -> None:
        self._repo_id = repo_id
        self._push_every = push_every
        self._thread: threading.Thread | None = None

    def _push(self, ckpt_dir: Path, step: int, commit_msg: str) -> None:
        from huggingface_hub import HfApi

        try:
            HfApi().upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"checkpoint-{step}",
                repo_id=self._repo_id,
                repo_type="model",
                commit_message=commit_msg,
            )
            print(f"  [HF] pushed checkpoint-{step} → {self._repo_id}", flush=True)
        except Exception as exc:
            print(f"  [HF] push failed (step {step}): {exc}", flush=True)

    def _maybe_push(
        self,
        args: TrainingArguments,
        state: TrainerState,
        force: bool = False,
    ) -> None:
        step = state.global_step
        if not force and step % self._push_every != 0:
            return
        output_dir = args.output_dir
        if not output_dir:
            return
        ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
        if not ckpt_dir.exists():
            return
        if self._thread and self._thread.is_alive():
            print(
                f"  [HF] previous push still running, skipping step {step}",
                flush=True,
            )
            return
        log = state.log_history[-1] if state.log_history else {}
        loss = log.get("loss", log.get("train_loss"))
        acc = log.get("mean_token_accuracy")
        epoch = state.epoch or 0.0
        if isinstance(loss, float) and isinstance(acc, float):
            commit_msg = (
                f"checkpoint-{step} (step {step}, epoch {epoch:.3f}, "
                f"loss {loss:.4f}, acc {acc:.4f})"
            )
        else:
            commit_msg = f"checkpoint-{step} (step {step}, epoch {epoch:.3f})"
        self._thread = threading.Thread(
            target=self._push, args=(ckpt_dir, step, commit_msg), daemon=True
        )
        self._thread.start()
        print(f"  [HF] pushing checkpoint-{step} in background...", flush=True)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._maybe_push(args, state)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self._maybe_push(args, state, force=True)
        if self._thread and self._thread.is_alive():
            print("  [HF] waiting for final push to complete...", flush=True)
            self._thread.join()
