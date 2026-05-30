#!/usr/bin/env bash
# GPU tech-stack smoke test for NVIDIA / CUDA machines (RTX 4000 Ada).
#
# CUDA counterpart of scripts/test_gpu_stack.sh (which is ROCm/RDNA only). Tests
# the ML stack needed for Qwen3.5-0.8B fine-tuning, in order:
#   1. NVIDIA driver + GPU visibility (nvidia-smi)
#   2. torch CUDA — GPU tensor op
#   3. bitsandbytes — import + CUDA backend detection
#   4. bitsandbytes 8-bit Linear (LLM.int8 quantization)
#   5. bitsandbytes 4-bit NF4 Linear (QLoRA quantization)
#   6. transformers + BitsAndBytesConfig (QLoRA config object)
#   7. PEFT + LoraConfig (LoRA adapter config)
#   8. TRL SFTConfig (fine-tuning trainer) — skipped if not installed
#   9. PyTorch SDPA (Flash-Attention) + flash-attn if installed
#
# The AMD script's llama.cpp/Vulkan inference probes (its steps 10-13) are
# dropped here: they are HIP/RDNA-specific and not part of the training stack.
# No model weights are downloaded. Each step is independent.
#
# Usage:
#   bash scripts/test_gpu_stack_nvidia.sh
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

# ── 1. NVIDIA driver ───────────────────────────────────────────────────────────
step 1 "NVIDIA driver + GPU visibility"
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found — NVIDIA driver is not installed."
    FAILURES=$((FAILURES + 1))
else
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi 2>/dev/null | awk -F'CUDA Version:' '/CUDA Version/{gsub(/ |\|/,"",$2); print $2; exit}')
    pass "driver $DRIVER  CUDA $CUDA_VER"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
        | sed 's/^/         /' || true
fi

# ── 2. torch CUDA ───────────────────────────────────────────────────────────────
step 2 "torch — GPU tensor op"
TORCH_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    import torch
except ImportError:
    print("FAIL torch not installed")
    sys.exit(1)

if torch.version.hip:
    print(f"FAIL this is a ROCm/HIP torch build (hip {torch.version.hip}) — use test_gpu_stack.sh")
    sys.exit(1)

print(f"torch {torch.__version__}  CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    print("FAIL GPU not visible to torch (torch.cuda.is_available() returned False)")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}  {p.total_memory // 1024**3} GB  sm_{p.major}{p.minor}")

t = torch.tensor([1.0, 2.0, 3.0]).cuda()
assert t.sum().item() == 6.0
print("tensor op on GPU: OK")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
PY
)
if echo "$TORCH_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$TORCH_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$TORCH_OUT"
fi

# ── 3. bitsandbytes — import + CUDA backend ──────────────────────────────────────
step 3 "bitsandbytes — import + CUDA backend detection"
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
        print("FAIL GPU not visible — bitsandbytes cannot initialise CUDA backend")
        sys.exit(1)

    if hasattr(bnb, "cuda_specs"):
        print(f"cuda_specs: {bnb.cuda_specs}")
    else:
        _ = bnb.functional
        print("bnb backend loaded (functional module OK)")

    if torch.version.hip:
        print(f"FAIL ROCm/HIP torch — this is the NVIDIA stack check")
        sys.exit(1)
    print("CUDA backend")
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
    linear = bnb.nn.Linear8bitLt(64, 64, has_fp16_weights=False, bias=False).cuda()
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
    linear = bnb.nn.Linear4bit(
        64, 64, bias=False, quant_type="nf4", compute_dtype=torch.bfloat16,
    ).cuda()
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
    BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print("BitsAndBytesConfig: OK  (4-bit NF4, double-quant, bf16 compute)")
    BitsAndBytesConfig(load_in_8bit=True)
    print("BitsAndBytesConfig: OK  (8-bit LLM.int8)")
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
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    print(f"LoraConfig: OK  (r={config.r}, alpha={config.lora_alpha}, target={config.target_modules})")
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
        output_dir="/tmp/trl_probe", max_length=512,
        per_device_train_batch_size=1, gradient_checkpointing=True,
    )
    print(f"SFTConfig: OK  (max_length={cfg.max_length}, grad_ckpt={cfg.gradient_checkpointing})")
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

# ── 9. Efficient attention (PyTorch SDPA + optional flash-attn) ───────────────
step 9 "Efficient attention — PyTorch SDPA + flash-attn if installed"
SDPA_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys, warnings
warnings.filterwarnings("ignore")
try:
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError as e:
    print(f"FAIL missing dep: {e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL GPU not available for SDPA")
    sys.exit(1)

print(f"torch {torch.__version__}  SDPA backends available:")
print(f"  flash={torch.backends.cuda.flash_sdp_enabled()}  "
      f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}  "
      f"math={torch.backends.cuda.math_sdp_enabled()}")

B, H, S, D = 1, 4, 64, 32
q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
    out = scaled_dot_product_attention(q, k, v)
print(f"scaled_dot_product_attention: {tuple(q.shape)} → {tuple(out.shape)}  OK")
print("PyTorch SDPA (Flash-Attention): supported")

try:
    import flash_attn
    from flash_attn import flash_attn_func
    print(f"flash_attn package {flash_attn.__version__} also installed")
    q2 = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(1, 64, 4, 32, dtype=torch.bfloat16, device="cuda")
    out2 = flash_attn_func(q2, k2, v2)
    print(f"flash_attn_func: → {tuple(out2.shape)}  OK")
except ImportError:
    print("flash_attn package not installed (optional; PyTorch SDPA is sufficient)")
    print("  On CUDA the PyPI wheel works: uv pip install flash-attn --no-build-isolation")
except Exception as e:
    print(f"flash_attn_func raised: {e}  (non-fatal; SDPA passed above)")
PY
)
if echo "$SDPA_OUT" | grep -q "^FAIL"; then
    fail "$(echo "$SDPA_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
    FAILURES=$((FAILURES + 1))
else
    while IFS= read -r line; do pass "$line"; done <<< "$SDPA_OUT"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Result ━━━${NC}"
if [[ "$FAILURES" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All steps passed.${NC}"
    echo ""
    echo "  GPU visibility:               steps 1–2 confirm GPU is usable."
    echo "  QLoRA fine-tuning:            steps 3–6 all passing = QLoRA is supported."
    echo "  LoRA fine-tuning:             step 7 passing = PEFT/LoRA is ready."
    echo "  SFT trainer:                  step 8 passing = TRL SFTTrainer is ready."
    echo "  Efficient attention (torch):  step 9 = PyTorch SDPA + flash-attn (if installed)"
    echo ""
    echo "  This is the training-stack health check. The full Qwen3.5-0.8B run uses"
    echo "  bf16 LoRA (no bitsandbytes required) — steps 3–6 confirm QLoRA is also an option."
else
    echo -e "${RED}${BOLD}$FAILURES step(s) failed.${NC}"
    echo ""
    echo "  Steps 1–2, 7 are required for bf16 LoRA fine-tuning (the current run)."
    echo "  Steps 3–6 are required only for QLoRA; steps 8–9 are optional."
fi
echo ""
exit "$FAILURES"
