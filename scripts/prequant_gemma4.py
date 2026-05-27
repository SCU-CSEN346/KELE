#!/usr/bin/env python3
"""Pre-quantize Gemma 4 31B-IT to NF4 on a high-RAM CUDA machine.

Run on a machine with 96GB+ RAM and a CUDA GPU (e.g. NVIDIA L40S).
Saves ~16 GB of NF4 weights; reload on the R9700 with TRAIN_PREQ=true,
skipping the 62 GB BF16 staging that causes OOM on 62 GB RAM systems.

Usage:
    python scripts/prequant_gemma4.py
    python scripts/prequant_gemma4.py --output /path/to/output
    python scripts/prequant_gemma4.py --model google/gemma-4-31b-it --output ./gemma-4-31b-nf4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-quantize Gemma 4 31B to NF4")
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--output", default="gemma-4-31b-nf4")
    args = parser.parse_args()

    output_path = Path(args.output)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading tokenizer  model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading and quantizing model  output={output_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving NF4 checkpoint to {output_path} ...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    size_gb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
    print(f"Done. Checkpoint size: {size_gb:.1f} GB")
    print(
        f"Transfer with: rsync -avP {output_path}/ user@r9700:~/Github/csen-346/models/gemma-4-31b-nf4/"
    )


if __name__ == "__main__":
    main()
