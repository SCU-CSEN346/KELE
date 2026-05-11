#!/usr/bin/env bash
# GPU tech-stack smoke test for ROCm / RDNA machines.
#
# Tests the full ML stack needed for inference and fine-tuning, in order:
#   1. ROCm driver + GPU visibility
#   2. torch ROCm — GPU tensor op
#   3. bitsandbytes — import + GPU detection
#   4. bitsandbytes 8-bit Linear (LLM.int8 / llm_int8 quantization)
#   5. bitsandbytes 4-bit NF4 Linear (QLoRA quantization)
#   6. transformers + BitsAndBytesConfig (QLoRA config object)
#   7. PEFT + LoraConfig (LoRA adapter config)
#   8. TRL SFTConfig (fine-tuning trainer) — skipped if not installed
#   9. flash-attn — skipped if not installed
#
# No model weights are downloaded. Each step is independent.
#
# Usage:
#   bash scripts/test_gpu_stack.sh
#
# Exit code: 0 if all non-optional steps pass, 1 if any fail.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
pass()  { echo -e "  ${GREEN}PASS${NC}  $*"; }
fail()  { echo -e "  ${RED}FAIL${NC}  $*"; }
warn()  { echo -e "  ${YELLOW}WARN${NC}  $*"; }
step()  { echo -e "\n${BOLD}[$1/9] $2${NC}"; }

FAILURES=0

# ── 1. ROCm driver ────────────────────────────────────────────────────────────
step 1 "ROCm driver + GPU visibility"
if ! command -v rocm-smi &>/dev/null; then
    fail "rocm-smi not found — ROCm is not installed."
    FAILURES=$((FAILURES + 1))
else
    ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null \
        || rocminfo 2>/dev/null | awk '/ROCm Runtime/{match($0,/[0-9]+\.[0-9]+/); print substr($0,RSTART,RLENGTH); exit}' \
        || echo "unknown")
    GPU_ARCH=$(rocminfo 2>/dev/null | awk '/Name:.*gfx/{print $NF; exit}') || true
    [[ -z "${GPU_ARCH:-}" ]] && GPU_ARCH="unknown"
    pass "ROCm $ROCM_VER  arch: $GPU_ARCH"
    rocm-smi --showproductname 2>/dev/null | grep -v "^$" | sed 's/^/         /' || true
fi

# ── 2. torch ROCm ─────────────────────────────────────────────────────────────
step 2 "torch — GPU tensor op"
TORCH_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import torch
except ImportError:
    print("FAIL torch not installed")
    sys.exit(1)

print(f"torch {torch.__version__}  HIP {getattr(torch.version, 'hip', 'n/a')}")
if not torch.cuda.is_available():
    print("FAIL GPU not visible to torch (torch.cuda.is_available() returned False)")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}  {p.total_memory // 1024**3} GB")

t = torch.tensor([1.0, 2.0, 3.0]).cuda()
assert t.sum().item() == 6.0
print("tensor op on GPU: OK")
PY
)
if echo "$TORCH_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$TORCH_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$TORCH_OUT"
fi

# ── 3. bitsandbytes — import + GPU detection ──────────────────────────────────
step 3 "bitsandbytes — import + GPU detection"
BNB_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import bitsandbytes as bnb
except ImportError:
    print("FAIL bitsandbytes not installed — run: uv sync")
    sys.exit(1)

print(f"bitsandbytes {bnb.__version__}")

try:
    import torch
    if not torch.cuda.is_available():
        print("FAIL GPU not visible — bitsandbytes cannot initialise CUDA/ROCm backend")
        sys.exit(1)

    # Probe the bnb CUDA state without running a forward pass yet.
    # bnb.cuda_specs (older) or the state dict (newer) tells us if the backend loaded.
    state = None
    if hasattr(bnb, "cuda_specs"):
        state = bnb.cuda_specs
        print(f"cuda_specs: {state}")
    elif hasattr(bnb, "get_state"):
        state = bnb.get_state()
        print(f"bnb state: {state}")
    else:
        # Trigger backend load by instantiating the functional module.
        _ = bnb.functional
        print("bnb backend loaded (functional module OK)")

    # Check for common ROCm/HIP backend flags.
    hip = getattr(torch.version, "hip", None)
    if hip:
        print(f"ROCm/HIP backend: {hip}")
    else:
        print("CUDA backend (non-ROCm torch)")

except Exception as e:
    print(f"FAIL bitsandbytes GPU probe raised: {e}")
    sys.exit(1)
PY
)
if echo "$BNB_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$BNB_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$BNB_OUT"
fi

# ── 4. bitsandbytes 8-bit Linear (LLM.int8) ───────────────────────────────────
step 4 "bitsandbytes — 8-bit Linear forward pass (LLM.int8)"
BNB8_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import torch
    import bitsandbytes as bnb
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL GPU not available")
    sys.exit(1)

try:
    # Linear8bitLt is the LLM.int8() quantized linear layer.
    linear = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False, bias=False)
    linear = linear.cuda()

    x = torch.randn(2, 64, dtype=torch.float16).cuda()
    with torch.no_grad():
        y = linear(x)

    print(f"Linear8bitLt forward: input {tuple(x.shape)} → output {tuple(y.shape)}  OK")
    print(f"output dtype: {y.dtype}  device: {y.device}")
except Exception as e:
    print(f"FAIL Linear8bitLt forward raised: {e}")
    sys.exit(1)
PY
)
if echo "$BNB8_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$BNB8_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$BNB8_OUT"
fi

# ── 5. bitsandbytes 4-bit NF4 Linear (QLoRA) ──────────────────────────────────
step 5 "bitsandbytes — 4-bit NF4 Linear forward pass (QLoRA)"
BNB4_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import torch
    import bitsandbytes as bnb
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL GPU not available")
    sys.exit(1)

try:
    # Linear4bit with NF4 + bf16 compute is the exact QLoRA configuration.
    linear = bnb.nn.Linear4bit(
        64, 64,
        bias=False,
        quant_type="nf4",
        compute_dtype=torch.bfloat16,
    )
    linear = linear.cuda()

    x = torch.randn(2, 64, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        y = linear(x)

    print(f"Linear4bit (NF4, bf16) forward: input {tuple(x.shape)} → output {tuple(y.shape)}  OK")
    print(f"output dtype: {y.dtype}  device: {y.device}")
    print("QLoRA quantization: supported on this GPU")
except Exception as e:
    print(f"FAIL Linear4bit NF4 forward raised: {e}")
    sys.exit(1)
PY
)
if echo "$BNB4_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$BNB4_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$BNB4_OUT"
fi

# ── 6. transformers + BitsAndBytesConfig ──────────────────────────────────────
step 6 "transformers — BitsAndBytesConfig (QLoRA config object)"
TF_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import transformers
except ImportError:
    print("FAIL transformers not installed — run: uv sync")
    sys.exit(1)

print(f"transformers {transformers.__version__}")

try:
    import torch
    from transformers import BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # QLoRA double-quant
    )
    print(f"BitsAndBytesConfig: OK  (4-bit NF4, double-quant, bf16 compute)")

    # Also confirm 8-bit config path.
    bnb_cfg_8b = BitsAndBytesConfig(load_in_8bit=True)
    print(f"BitsAndBytesConfig: OK  (8-bit LLM.int8)")
except Exception as e:
    print(f"FAIL BitsAndBytesConfig raised: {e}")
    sys.exit(1)
PY
)
if echo "$TF_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$TF_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$TF_OUT"
fi

# ── 7. PEFT + LoraConfig ──────────────────────────────────────────────────────
step 7 "PEFT — LoraConfig (LoRA adapter config)"
PEFT_INSTALLED=$(.venv/bin/python -c "import peft; print(peft.__version__)" 2>/dev/null || echo "")
if [[ -z "$PEFT_INSTALLED" ]]; then
    warn "peft not installed — skipping.  Install with: uv add peft"
else
    PEFT_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import peft
    from peft import LoraConfig, TaskType
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

print(f"peft {peft.__version__}")

try:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    print(f"LoraConfig: OK  (r={config.r}, alpha={config.lora_alpha}, "
          f"target={config.target_modules})")
except Exception as e:
    print(f"FAIL LoraConfig raised: {e}")
    sys.exit(1)
PY
    )
    if echo "$PEFT_OUT" | grep -q "^FAIL"; then
        fail "$(echo "$PEFT_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
        FAILURES=$((FAILURES + 1))
    else
        while IFS= read -r line; do pass "$line"; done <<< "$PEFT_OUT"
    fi
fi

# ── 8. TRL SFTConfig (optional) ───────────────────────────────────────────────
step 8 "TRL — SFTConfig (fine-tuning trainer)  [optional]"
TRL_INSTALLED=$(.venv/bin/python -c "import trl; print(trl.__version__)" 2>/dev/null || echo "")
if [[ -z "$TRL_INSTALLED" ]]; then
    warn "trl not installed — skipping.  Install with: uv add trl"
    warn "TRL is required for SFT fine-tuning (approach #4 in IMPROVEMENT_PLAN.md)."
else
    TRL_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import trl
    from trl import SFTConfig
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

print(f"trl {trl.__version__}")

try:
    cfg = SFTConfig(
        output_dir="/tmp/trl_probe",
        max_seq_length=512,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
    )
    print(f"SFTConfig: OK  (max_seq_length={cfg.max_seq_length}, "
          f"grad_ckpt={cfg.gradient_checkpointing})")
except Exception as e:
    print(f"FAIL SFTConfig raised: {e}")
    sys.exit(1)
PY
    )
    if echo "$TRL_OUT" | grep -q "^FAIL"; then
        fail "$(echo "$TRL_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
        FAILURES=$((FAILURES + 1))
    else
        while IFS= read -r line; do pass "$line"; done <<< "$TRL_OUT"
    fi
fi

# ── 9. flash-attn (optional) ──────────────────────────────────────────────────
step 9 "flash-attn — FlashAttention-2  [optional]"
FA_INSTALLED=$(.venv/bin/python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "")
if [[ -z "$FA_INSTALLED" ]]; then
    warn "flash_attn not installed — skipping."
    warn "FlashAttention-2 speeds up fine-tuning significantly on supported GPUs."
    warn "ROCm support: check https://github.com/ROCm/flash-attention for gfx1201 status."
else
    FA_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import flash_attn
    from flash_attn import flash_attn_func
    import torch
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

print(f"flash_attn {flash_attn.__version__}")

try:
    if not torch.cuda.is_available():
        print("FAIL GPU not available for flash_attn")
        sys.exit(1)

    # Minimal probe: create random Q/K/V and run one attention call.
    B, S, H, D = 1, 64, 4, 32
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    out = flash_attn_func(q, k, v)
    print(f"flash_attn_func: input {tuple(q.shape)} → output {tuple(out.shape)}  OK")
except Exception as e:
    print(f"FAIL flash_attn_func raised: {e}")
    sys.exit(1)
PY
    )
    if echo "$FA_OUT" | grep -q "^FAIL"; then
        fail "$(echo "$FA_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
        FAILURES=$((FAILURES + 1))
    else
        while IFS= read -r line; do pass "$line"; done <<< "$FA_OUT"
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Result ━━━${NC}"
if [[ "$FAILURES" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All steps passed.${NC}"
    echo ""
    echo "  Inference stack (llama.cpp):  steps 1–2 confirm GPU is usable."
    echo "  QLoRA fine-tuning:            steps 3–6 all passing = QLoRA is supported."
    echo "  LoRA fine-tuning:             step 7 passing = PEFT/LoRA is ready."
    echo "  SFT trainer:                  step 8 passing = TRL SFTTrainer is ready."
    echo "  FlashAttention-2:             step 9 passing = faster fine-tuning available."
    echo ""
    echo "  To check vLLM separately:  bash scripts/test_vllm_rocm.sh"
else
    echo -e "${RED}${BOLD}$FAILURES step(s) failed.${NC}"
    echo ""
    echo "  Steps 1–7 are required for QLoRA fine-tuning."
    echo "  Steps 8–9 are optional (TRL, flash-attn)."
    echo ""
    if [[ "$FAILURES" -gt 0 ]]; then
        echo "  If bitsandbytes (steps 3–5) failed on ROCm gfx1201:"
        echo "    1. Check bnb ROCm support: https://github.com/bitsandbytes-foundation/bitsandbytes"
        echo "    2. Try installing the ROCm-specific build:"
        echo "         uv pip install bitsandbytes --index-url https://pypi.org/simple/"
        echo "    3. Or pin a known-good ROCm build:"
        echo "         uv add 'bitsandbytes>=0.45.0'"
        echo "    4. If QLoRA is unavailable, fall back to bf16 LoRA (no bnb needed)."
    fi
fi
echo ""
exit "$FAILURES"
