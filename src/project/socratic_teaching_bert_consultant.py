"""BERT-routed consultant variant of the KELE system.

Subclasses SocraticTeachingSystem and overrides socratic_teaching_consultant
to use the trained 34-state Chinese-BERT classifier instead of an LLM call.
The teacher half (socrates_teacher) remains LLM-driven so this isolates the
contribution of the consultant component.

Usage from kele.py:
  Use `--bert-consultant <path>` to load a checkpoint, then run as usual.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.project.socratic_teaching_system import SocraticTeachingSystem

DEFAULT_CKPT = "results/state_classifier_v1/final"

# Same state-label mapping as the training script
ALL_STATES = (
    ["a0", "a1"]
    + [f"b{i}" for i in range(2, 8)]
    + [f"c{i}" for i in range(8, 30)]
    + [f"d{i}" for i in range(30, 34)]
    + ["e34"]
)

# Friendly Chinese names for each SocRule stage.
_STAGE_NAMES = {
    "a": "学生提问",
    "b": "概念探查",
    "c": "归纳推理",
    "d": "规则建构",
    "e": "教师总结",
}


class SocraticTeachingSystemBertConsultant(SocraticTeachingSystem):
    """Drop-in consultant replacement using a fine-tuned Chinese BERT.

    The BERT classifier is loaded at __init__ time. Each turn, the consultant
    call formats the current dialogue history the same way the training data
    was formatted ("学生: ..." / "老师: ..." lines, ending with the new
    student input), tokenizes, and runs the classifier to produce a state
    prediction. The evaluation field is synthesized as a placeholder since
    the BERT classifier doesn't produce free-text justification.
    """

    def __init__(self, *args: Any, bert_ckpt: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        ckpt_path = bert_ckpt or os.environ.get("KELE_BERT_CKPT", DEFAULT_CKPT)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"BERT classifier checkpoint not found at {ckpt_path}. "
                f"Train via `uv run python scripts/train_state_classifier_34way.py`."
            )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        # Load fp32 first, post-cast to bf16 after device transfer. The naive
        # `from_pretrained(..., dtype=torch.bfloat16, low_cpu_mem_usage=True)`
        # call has a thread race on transformers 5.x + torch 2.11: the meta-
        # tensor materialization path leaks fp32 sub-buffers into nominally-
        # bf16 modules, and concurrent loads in different threads land with
        # partially fp32 weights despite reporting bf16 in named_parameters().
        # Inference then explodes with `mat1/mat2 must have the same dtype,
        # got BFloat16 and Float`. Discovered 2026-05-23 while bringing up
        # the 4-cell STL bilingual probe at KELE_PARALLEL_WORKERS=4 (3 of 4
        # worker threads failed identically; the v3 load-then-cast pattern
        # tested 4/4 OK).
        # attn_implementation=eager — transformers 5.8.1's default SDPA path
        # for Qwen3.5 MLA crashes with "cannot reshape tensor of 0 elements"
        # on first forward (modeling_qwen3_5.py:450). Eager attention works
        # cleanly. BERT-class models silently ignore the kwarg, so passing
        # it unconditionally is safe across all consultant backbones.
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_path,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        # Device selection: default to CPU when KELE_BERT_DEVICE=cpu (or 'auto'
        # when the teacher already occupies most VRAM). Otherwise CUDA.
        # A 24M bge-small fits anywhere; Qwen3*-Embedding (~600M-750M params,
        # 1.2-1.5 GB in bf16) plus a 28 GB teacher (Gemma 4 31B Q5) on a 32 GB
        # 5090 leaves only ~2 GB of free + fragmented VRAM, where even 16 MB
        # contiguous allocations fail. CPU inference is ~100-300ms per turn on
        # a Qwen3.5-class model — ~3% overhead on a 50-min Gemma eval. Tiny
        # cost; massive reliability win.
        env_device = os.environ.get("KELE_BERT_DEVICE", "auto").lower()
        if env_device == "cpu":
            self.bert_device = "cpu"
        elif env_device == "cuda":
            self.bert_device = "cuda"
        else:  # auto: pick CPU when the BERT model is >100 MB AND CUDA is busy
            backbone_params_mb = sum(p.numel() for p in self.bert_model.parameters()) * 2 / 1e6
            cuda_busy = False
            if torch.cuda.is_available():
                free_bytes, _total = torch.cuda.mem_get_info()
                # If less than 3 GB free, fall back to CPU regardless of model size
                cuda_busy = free_bytes < 3 * 1024**3
            if backbone_params_mb > 200 and cuda_busy:
                self.bert_device = "cpu"
                print(
                    f"  [bert-consultant] {backbone_params_mb:.0f} MB model + busy CUDA "
                    f"({free_bytes / 1024**3:.1f} GB free) -> CPU inference"
                )
            else:
                self.bert_device = "cuda" if torch.cuda.is_available() else "cpu"
        # Force bf16 AFTER device transfer (see comment above) — this is the
        # cast that actually sticks under multi-thread loads.
        self.bert_model.to(self.bert_device).to(dtype=torch.bfloat16).eval()
        self.bert_max_length = 512
        # Ensure pad_token_id is set on the model config (Qwen3* classifiers
        # need this for last-non-pad-token pooling; tokenizer ships with one
        # already, but model.config may not).
        if self.bert_model.config.pad_token_id is None:
            self.bert_model.config.pad_token_id = self.bert_tokenizer.pad_token_id
        # id2label: trust the saved config but fall back to ALL_STATES order
        try:
            self.bert_id2label = {int(k): v for k, v in self.bert_model.config.id2label.items()}
        except Exception:
            self.bert_id2label = dict(enumerate(ALL_STATES))

    def _format_history_for_bert(self, current_input: str) -> str:
        """Mirror the training-time format: ``学生: ...\n老师: ...\n学生: <current>``.

        IMPORTANT: by the time this method is called, ``current_input`` has
        already been appended to ``self.conversation_history`` by the parent
        ``SocraticTeachingSystem.process_student_input`` (line 486:
        ``self.add_to_history("student", student_input)`` runs BEFORE the
        consultant call on line 489). So we MUST NOT append ``current_input``
        again here — that would duplicate the current student utterance and
        give the classifier ``学生: X\n学生: X`` as input, which never appears
        in the trainer's ``build_examples`` distribution.

        The duplication bug was found 2026-05-22 PM while diagnosing T4 +
        Gemma integration mini-test underperformance. T4 (Qwen3.5) was
        catastrophically affected (~45 pp drop from standalone 67.57% to
        integration ~22%). bge-small was also affected but less severely
        (~10 pp drop from standalone 61.34% to integration 51.06%), which is
        why the bug went undetected in the paper's locked headline.
        Re-running with the fix on a matched n=50 set is the apples-to-apples
        comparison.
        """
        lines: list[str] = []
        for entry in self.conversation_history:
            role = entry.get("role", "")
            content = entry.get("content", "").strip()
            if role == "student":
                lines.append(f"学生: {content}")
            elif role == "teacher":
                lines.append(f"老师: {content}")
        # The current student turn is already in conversation_history — do NOT
        # append `current_input` again. (Keeping the parameter to preserve the
        # method signature; the value matches the last student entry already.)
        _ = current_input  # silence unused-arg lint
        full = "\n".join(lines)
        # Mirror training truncation: last 4000 chars
        return full[-4000:]

    def socratic_teaching_consultant(self, student_input: str) -> dict[str, Any]:  # type: ignore[override]
        """Predict the state via BERT. Returns the dict shape kele expects.

        The `evaluation` field is synthesized into a richer Chinese narrative
        (state code + stage name + action description + top-2 BERT probabilities)
        rather than a placeholder, because the LLM teacher uses this field as
        context for response generation. See KELE_BERT_PLAIN_EVAL=1 to disable
        and fall back to a minimal placeholder (ablation hook).
        """
        text = self._format_history_for_bert(student_input)
        enc = self.bert_tokenizer(
            text,
            truncation=True,
            max_length=self.bert_max_length,
            return_tensors="pt",
        ).to(self.bert_device)
        with torch.no_grad():
            logits = self.bert_model(**enc).logits
            probs = F.softmax(logits, dim=-1)[0]
            top2 = probs.topk(k=min(2, probs.shape[0]))
            top_idxs = top2.indices.cpu().tolist()
            top_probs = top2.values.cpu().tolist()
            pred_idx = top_idxs[0]
        pred_state = self.bert_id2label.get(
            pred_idx, ALL_STATES[pred_idx] if 0 <= pred_idx < len(ALL_STATES) else "a0"
        )
        action = self.get_action_for_state(pred_state)

        if os.environ.get("KELE_BERT_PLAIN_EVAL") == "1":
            evaluation = f"BERT classifier predicted state: {pred_state}"
        else:
            stage = pred_state[0] if pred_state else "a"
            stage_name = _STAGE_NAMES.get(stage, "")
            top_p = top_probs[0]
            alt_state = self.bert_id2label.get(
                top_idxs[1], ALL_STATES[top_idxs[1]] if len(top_idxs) > 1 else pred_state
            )
            alt_p = top_probs[1] if len(top_probs) > 1 else 0.0
            evaluation = (
                f"根据当前对话内容，学生处于 {stage}({stage_name}) 阶段的 {pred_state} 状态"
                f"（分类器置信度 {top_p:.2f}；次选状态 {alt_state} 置信度 {alt_p:.2f}）。"
                f"按照苏格拉底教学法，应采取的操作是：{action}。"
                f"请基于该状态和操作，针对学生当前的表现，给出合适的教学回复。"
            )
        return {
            "state": pred_state,
            "action": action,
            "evaluation": evaluation,
        }
