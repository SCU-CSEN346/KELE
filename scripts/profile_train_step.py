#!/usr/bin/env python
"""De-risk the FA2 bet: profile where a real Stage 2 training step spends time.

The gfx1201 ~70 s/step is ~9× above the raw-TFLOPS floor. The handoff attributes
the gap to "SDPA attention is the bottleneck, not GEMM" — but that was *inferred*
from hipBLASLt=1 showing no gain, never measured directly. This script settles it:
it runs a handful of REAL training steps (same model load, LoRA, collator,
assistant_only_loss, grad-ckpt as scripts/train_sft.py) under torch.profiler and
prints the kernel self-time breakdown.

Decision it informs:
  - If SDPA / attention kernels dominate  → patching the flash-attn Triton AMD
    backward (bwd_prefill_split.py block sizes → fit 64 KB LDS) could collapse the
    step time. FA2 is worth the spike.
  - If bitsandbytes NF4 dequant / GEMM kernels dominate → FA2 buys little even if
    the backward is made to fit. Don't sink time into the kernel patch.

Run on the R9700 (this is a HIP/ROCm profile; ProfilerActivity.CUDA captures HIP
kernels on a ROCm torch build):

    make profile-gemma4-31b              # wrapper with the right env
    # or directly:
    TORCH_USE_HIPBLASLT=1 PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
      uv run --no-sync python scripts/profile_train_step.py \
      --config configs/train-sft-stage2-gemma4-31b.env

Profiles 6 optimizer steps (schedule wait=1/warmup=2/active=3) → ~6 min at 70 s/step.
Writes a Chrome trace to outputs/profile-gemma4-31b/trace.json for offline viewing.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_sft import (  # noqa: E402
    build_hf_datasets,
    build_lora_config,
    build_model_and_tokenizer,
    build_sft_config,
)

# Op-name keywords for a coarse attention-vs-gemm/dequant split. Fuzzy on
# purpose — the authoritative answer is the raw table below; this bucket summary
# is the at-a-glance decision aid.
_ATTN_KEYS = (
    "attention",
    "sdpa",
    "scaled_dot_product",
    "softmax",
    "flash",
    "efficient_attention",
    "mem_eff",
    "fmha",
)
_GEMM_KEYS = (
    "gemm",
    "matmul",
    "addmm",
    "hipblaslt",
    "rocblas",
    "cijk",  # rocBLAS tensile kernel names
    "dequant",
    "kgemm",
    "4bit",
    "nf4",
    "bnb",
    "int4",
    "cutlass",
)


def _bucket(name: str) -> str:
    low = name.lower()
    if any(k in low for k in _ATTN_KEYS):
        return "attention"
    if any(k in low for k in _GEMM_KEYS):
        return "gemm/dequant"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a real Stage 2 training step")
    parser.add_argument("--config", metavar="PATH", required=True, help="env config file")
    parser.add_argument("--steps", type=int, default=6, help="optimizer steps to run")
    args = parser.parse_args()

    from src.project.config import load_env_file

    load_env_file(Path(args.config))

    # Cap the run to a few steps and keep eval off, regardless of the config.
    os.environ["TRAIN_MAX_STEPS"] = str(args.steps)
    os.environ["TRAIN_EVAL_STRATEGY"] = "no"

    import torch
    from peft import get_peft_model
    from torch.profiler import ProfilerActivity, profile, schedule
    from transformers import TrainerCallback
    from trl import SFTTrainer

    out_dir = "outputs/profile-gemma4-31b"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds = build_hf_datasets()
    model, tokenizer = build_model_and_tokenizer()
    lora_cfg = build_lora_config()
    with tempfile.TemporaryDirectory() as tmp:
        sft_cfg = build_sft_config(output_dir=tmp, use_wandb=False)

        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        class ProfStep(TrainerCallback):
            def __init__(self, prof):
                self.prof = prof

            def on_step_end(self, *a, **kw):
                self.prof.step()

        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
        )

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        print(f"\nProfiling {args.steps} optimizer steps (schedule wait=1/warmup=2/active=3)...")
        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=2, active=3),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            trainer.add_callback(ProfStep(prof))
            trainer.train()

    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
    print("\n" + "=" * 100)
    print(f"TOP 40 OPS BY {sort_key}")
    print("=" * 100)
    print(table)

    # Coarse bucket summary over the device self-time.
    buckets: dict[str, float] = {"attention": 0.0, "gemm/dequant": 0.0, "other": 0.0}
    attr = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    for evt in prof.key_averages():
        buckets[_bucket(evt.key)] += float(getattr(evt, attr, 0.0))
    total = sum(buckets.values()) or 1.0
    print("\n" + "=" * 100)
    print(f"COARSE BUCKET SUMMARY ({attr}) — heuristic by op name, see raw table for ground truth")
    print("=" * 100)
    for name, val in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<14} {val / total * 100:5.1f}%   ({val / 1e3:.1f} ms self-time)")
    print(
        "\nDecision:\n"
        "  attention dominant    → FA2 backward-kernel patch is worth the spike\n"
        "  gemm/dequant dominant → FA2 buys little; the cost is NF4 dequant, not attention"
    )

    trace_path = f"{out_dir}/trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace: {trace_path}  (open in chrome://tracing or perfetto.dev)")


if __name__ == "__main__":
    main()
