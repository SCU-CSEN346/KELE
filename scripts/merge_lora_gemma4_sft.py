#!/usr/bin/env python3
"""Merge the Stage 2b LoRA adapter into Gemma-4-31B-IT base, save as BF16 HF checkpoint.

Pipeline step 1 of 3 for converting the SFT adapter to a llama.cpp-servable GGUF:
  (1) THIS SCRIPT: peft.merge_and_unload() → outputs/sft-stage2-gemma4-31b/merged/
  (2) scripts/convert_gemma4_sft_to_gguf.sh → f16 GGUF → Q5_K_M GGUF
  (3) Copy verified Q5_K_M into ~/Documents/models/weights/ with the KELE-tagged name

Resource budget: ~62 GB RAM (BF16 base on CPU, no GPU). ~10-15 min on CPU.

Usage:
    ./.venv/bin/python scripts/merge_lora_gemma4_sft.py
    ./.venv/bin/python scripts/merge_lora_gemma4_sft.py --adapter /path/to/checkpoint-2298
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    p = argparse.ArgumentParser(description="Merge Gemma-4-31B SFT LoRA → BF16 HF checkpoint")
    p.add_argument(
        "--base",
        default="google/gemma-4-31b-it",
        help="HF hub id or local path to the base Gemma 4 31B IT model.",
    )
    p.add_argument(
        "--adapter",
        default="outputs/sft-stage2-gemma4-31b/final",
        help="Path to the trained LoRA adapter directory.",
    )
    p.add_argument(
        "--out",
        default="outputs/sft-stage2-gemma4-31b/merged",
        help="Output dir for the merged BF16 HF checkpoint.",
    )
    args = p.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: Adapter dir not found: {adapter_path}", file=sys.stderr)
        return 1
    if not (adapter_path / "adapter_model.safetensors").exists():
        print(f"ERROR: No adapter_model.safetensors in {adapter_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    if out_path.exists() and any(out_path.iterdir()):
        print(f"ERROR: Output dir is non-empty: {out_path}", file=sys.stderr)
        print(f"Delete it and re-run: rm -rf {out_path}", file=sys.stderr)
        return 1

    print(f"Loading base in BF16 on CPU  model={args.base}")
    print("  (~62 GB RAM, ~5 min)")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter  path={args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging LoRA into base weights  (peft.merge_and_unload)")
    print("  (~3-5 min)")
    model = model.merge_and_unload()

    print(f"Saving merged BF16 checkpoint  out={args.out}")
    print("  (~5 min, ~62 GB write)")
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)

    print(f"Saving tokenizer  out={args.out}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.save_pretrained(args.out)

    size_gb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e9
    print(f"\nDone. Merged checkpoint: {size_gb:.1f} GB at {out_path}")
    print("Next: bash scripts/convert_gemma4_sft_to_gguf.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
