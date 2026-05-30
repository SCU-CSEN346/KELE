#!/usr/bin/env python3
"""Training GPU smoke test — AMD ROCm (gfx1201 / R9700 AI PRO).

Validates the GPU compute stack required for Qwen3.5-0.8B LoRA training on
the AMD ROCm box before committing to a multi-hour run. All tests run
forward+backward through the same kernel paths the actual training uses.
For NVIDIA/CUDA hosts use scripts/test_training_gpu_nvidia.py instead.

Usage:
    uv run --no-sync python scripts/test_training_gpu_amd.py
    uv run --no-sync python scripts/test_training_gpu_amd.py --hipblaslt   # test with hipBLASLt on

Exit code: 0 = all pass, 1 = any failure.
"""

from __future__ import annotations

import argparse
import sys
import time

GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BOLD = "\033[1m"
NC = "\033[0m"

_step_n = 0
_failures: list[str] = []


def step(label: str) -> None:
    global _step_n
    _step_n += 1
    print(f"\n{BOLD}[{_step_n}] {label}{NC}")


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{NC}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{NC}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{NC}  {msg}")
    _failures.append(msg)


def _ms(t0: float) -> str:
    return f"{(time.perf_counter() - t0) * 1000:.0f}ms"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hipblaslt",
        action="store_true",
        help="Test with hipBLASLt enabled (off by default — gfx1201 page-fault risk)",
    )
    args = p.parse_args()

    import os

    if not args.hipblaslt:
        os.environ.setdefault("TORCH_USE_HIPBLASLT", "0")
    hipblas_val = os.environ.get("TORCH_USE_HIPBLASLT", "1")

    # ── 1. import + device ───────────────────────────────────────────────────────
    step("torch import + GPU visibility")
    try:
        import torch
    except ImportError:
        fail(
            "torch not installed — run: uv pip install --index-url https://download.pytorch.org/whl/rocm7.2 torch==2.11.0+rocm7.2"
        )
        sys.exit(1)

    backend = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
    ok(f"torch {torch.__version__}  backend: {backend}")

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() returned False — no GPU visible")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"device: {name}  ({vram:.1f} GB)")
    ok(f"TORCH_USE_HIPBLASLT={hipblas_val}")

    # ── 2. fp32 matmul + backward ────────────────────────────────────────────────
    step("fp32 matmul + backward  (baseline kernel path)")
    try:
        x = torch.randn(4096, 4096, device="cuda", requires_grad=True)
        t0 = time.perf_counter()
        (x @ x.T).sum().backward()
        torch.cuda.synchronize()
        ok(f"4096×4096 fp32: {_ms(t0)}")
    except Exception as e:
        fail(f"fp32 matmul: {e}")

    # ── 3. bf16 matmul + backward ────────────────────────────────────────────────
    step("bf16 matmul + backward  (autocast forward path)")
    try:
        xb = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        t0 = time.perf_counter()
        (xb @ xb.T).sum().backward()
        torch.cuda.synchronize()
        ok(f"4096×4096 bf16: {_ms(t0)}")
    except Exception as e:
        fail(f"bf16 matmul: {e}")

    # ── 4. scaled-dot-product attention ─────────────────────────────────────────
    step("scaled-dot-product attention (Qwen3.5 full-attention layers)")
    try:
        B, H, T, D = 4, 16, 512, 128
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
        t0 = time.perf_counter()
        torch.nn.functional.scaled_dot_product_attention(q, k, v).sum().backward()
        torch.cuda.synchronize()
        ok(f"sdp-attn B{B} H{H} T{T} D{D} bf16: {_ms(t0)}")
    except Exception as e:
        fail(f"sdp-attn: {e}")

    # ── 5. linear-attention proxy ────────────────────────────────────────────────
    # Qwen3.5-0.8B has 18 linear-attention (chunk_gated_delta_rule) layers that use
    # batched outer-products and cumulative-sum recurrence. This proxy exercises the
    # same bmm + cumsum kernel path at training batch size. GPU page faults with
    # hipBLASLt ON show up here before wasting a full run.
    # NOTE: bf16 bmm for this recurrent shape is 5× SLOWER than fp32 on gfx1201
    # rocBLAS — do not use --bf16-autocast with Qwen3.5 on this GPU.
    step("linear-attention proxy (Qwen3.5 chunk_gated_delta_rule kernel path)")
    try:
        B, T, D = 8, 512, 1024
        x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(D, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        t0 = time.perf_counter()
        state = torch.zeros(B, D, D, device="cuda", dtype=torch.bfloat16)
        for i in range(4):
            chunk = x[:, i * 128 : (i + 1) * 128, :]
            state = state + torch.bmm(chunk.transpose(1, 2), chunk @ w)
        state.sum().backward()
        torch.cuda.synchronize()
        ok(f"linear-attn proxy B{B} T{T} D{D} bf16 ×4 chunks: {_ms(t0)}")
    except Exception as e:
        fail(f"linear-attn proxy: {e}")

    # ── 6. gradient checkpointing + autocast compatibility ──────────────────────
    # grad-ckpt recomputes the forward during backward; if autocast context is not
    # preserved the recomputed dtypes mismatch and cause GPU faults on ROCm.
    step("gradient checkpointing + bf16 autocast compatibility")
    try:
        from torch.utils.checkpoint import checkpoint

        def fwd(x: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return (x @ x.T).sum(dim=-1, keepdim=True)

        xc = torch.randn(128, 512, device="cuda", requires_grad=True)
        t0 = time.perf_counter()
        out = checkpoint(fwd, xc, use_reentrant=False)
        out.sum().backward()
        torch.cuda.synchronize()
        ok(f"grad-ckpt + bf16 autocast: {_ms(t0)}")
    except Exception as e:
        fail(f"grad-ckpt + autocast: {e}")

    # ── 7. VRAM headroom ─────────────────────────────────────────────────────────
    step("VRAM headroom check")
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    if free < 4.0:
        warn(f"only {free:.1f} GB free — training may OOM at batch_size=4")
    else:
        ok(f"{free:.1f} GB free of {vram:.1f} GB")

    # ── summary ──────────────────────────────────────────────────────────────────
    print()
    if _failures:
        print(f"{RED}{BOLD}FAILED — {len(_failures)} check(s) did not pass:{NC}")
        for f in _failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print(f"{GREEN}{BOLD}ALL CHECKS PASSED ({_step_n} steps){NC}")


if __name__ == "__main__":
    main()
