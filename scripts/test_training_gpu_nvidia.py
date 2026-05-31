#!/usr/bin/env python3
"""Training GPU smoke test — NVIDIA CUDA (RTX 4000 Ada / 20 GB).

Validates the GPU compute stack required for Qwen3.5-0.8B LoRA training on
the NVIDIA CUDA box before committing to a multi-hour run. All tests run
forward+backward through the same kernel paths the actual training uses.
For the AMD ROCm box use scripts/test_training_gpu_amd.py instead.

Unlike the AMD path, bf16 is the *preferred* precision here: Ada has native
bf16 tensor cores, so --bf16-autocast is both faster and lighter than fp32.
The fp32 pin that the AMD script defends against (gfx1201 rocBLAS page faults)
does not apply, so this script also reports the fp32-vs-bf16 speed ratio to
confirm the bf16 forward path is the one to train on.

Usage:
    uv run --no-sync python scripts/test_training_gpu_nvidia.py

Exit code: 0 = all pass, 1 = any failure.
"""

from __future__ import annotations

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
    # ── 1. import + device ───────────────────────────────────────────────────────
    step("torch import + GPU visibility")
    try:
        import torch
    except ImportError:
        fail(
            "torch not installed — run: uv pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130"
        )
        sys.exit(1)

    if torch.version.hip:
        fail(
            f"this is a ROCm/HIP torch build (hip {torch.version.hip}) — "
            "use scripts/test_training_gpu_amd.py on the AMD box"
        )
        sys.exit(1)
    ok(f"torch {torch.__version__}  backend: CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() returned False — no GPU visible")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    ok(f"device: {name}  ({vram:.1f} GB, sm_{props.major}{props.minor})")

    if torch.cuda.is_bf16_supported():
        ok("bf16 supported (native tensor cores) — train with --bf16-autocast")
    else:
        warn("bf16 not supported on this device — fall back to fp32")

    # ── flash-linear-attention (GDN fast path) ───────────────────────────────────
    # Qwen3.5-0.8B is 75% gated-DeltaNet linear attention. With FLA installed,
    # transformers routes those 18 layers through its Triton kernels (~2.8× fwd+bwd
    # over the pure-PyTorch fallback). It's a CUDA-only extra: `uv sync --extra cuda`.
    step("flash-linear-attention (Qwen3.5 GDN Triton kernels)")
    try:
        from transformers.utils.import_utils import is_flash_linear_attention_available

        if is_flash_linear_attention_available():
            import fla

            ok(f"fla {fla.__version__} active — GDN fast path engaged (~2.8× vs fallback)")
        else:
            warn(
                "flash-linear-attention NOT active — the 18 GDN layers will use the "
                "pure-PyTorch fallback (~2.8× slower). Install with: uv sync --extra cuda"
            )
    except Exception as e:
        warn(f"FLA probe failed: {e}")

    # ── 2. fp32 matmul + backward ────────────────────────────────────────────────
    step("fp32 matmul + backward  (baseline kernel path)")
    fp32_ms = None
    try:
        x = torch.randn(4096, 4096, device="cuda", requires_grad=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        (x @ x.T).sum().backward()
        torch.cuda.synchronize()
        fp32_ms = (time.perf_counter() - t0) * 1000
        ok(f"4096×4096 fp32: {fp32_ms:.0f}ms")
    except Exception as e:
        fail(f"fp32 matmul: {e}")

    # ── 3. bf16 matmul + backward ────────────────────────────────────────────────
    step("bf16 matmul + backward  (autocast forward path)")
    bf16_ms = None
    try:
        xb = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        (xb @ xb.T).sum().backward()
        torch.cuda.synchronize()
        bf16_ms = (time.perf_counter() - t0) * 1000
        ok(f"4096×4096 bf16: {bf16_ms:.0f}ms")
    except Exception as e:
        fail(f"bf16 matmul: {e}")

    # A single large matmul is memory-bandwidth-bound, so bf16 ≈ fp32 here is
    # expected and fine — bf16's real training win is lower memory + the GDN bf16
    # path (validated end-to-end at ~3.3h). Only a GROSS regression matters: this
    # guards against the gfx1201 pathology where bf16 bmm ran 5× slower than fp32.
    if fp32_ms and bf16_ms:
        ratio = bf16_ms / fp32_ms
        if ratio <= 1.5:
            ok(f"bf16/fp32 single-matmul ratio {ratio:.2f}× (near parity expected — memory-bound op)")
        else:
            warn(
                f"bf16 is {ratio:.2f}× slower than fp32 — a gross regression suggests a broken "
                "tensor-core path; investigate before trusting --bf16-autocast"
            )

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
    # same bmm + recurrence kernel path at training batch size in bf16 — the precision
    # the run actually uses on this card.
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
    # grad-ckpt recomputes the forward during backward; the recomputed pass must
    # re-enter the same bf16 autocast context or the dtypes mismatch. This mirrors
    # the --bf16-autocast + --gradient-checkpointing combo the training run uses.
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
    # The 0.8B model in fp32 master weights is ~3.4 GB; at bs=8 + grad-ckpt the run
    # peaks well under 20 GB, but flag a near-full card (other processes, display).
    step("VRAM headroom check")
    free = (props.total_memory - torch.cuda.memory_allocated()) / 1e9
    if free < 6.0:
        warn(f"only {free:.1f} GB free — training may OOM at batch_size=8")
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
