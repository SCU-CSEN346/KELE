#!/usr/bin/env bash
# Smoke test: can vLLM run on this machine's ROCm GPU?
#
# Does NOT load a full model. Tests in order:
#   1. ROCm driver + GPU visibility
#   2. torch ROCm — GPU accessible from Python
#   3. vLLM import — installs it temporarily if absent
#   4. vLLM device probe — does vLLM recognise the GPU?
#   5. vLLM engine init — can it start an engine (no model weights loaded)?
#
# Each step is independent. A later step failing does not skip earlier results.
#
# Usage:
#   bash scripts/test_vllm_rocm.sh
#
# Exit code: 0 if all steps pass, 1 if any step fails.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
pass()  { echo -e "  ${GREEN}PASS${NC}  $*"; }
fail()  { echo -e "  ${RED}FAIL${NC}  $*"; }
warn()  { echo -e "  ${YELLOW}WARN${NC}  $*"; }
step()  { echo -e "\n${BOLD}[$1/5] $2${NC}"; }

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

# ── 2. torch + ROCm ───────────────────────────────────────────────────────────
step 2 "torch — GPU accessible from Python"
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

# Simple tensor op on GPU to confirm it's not just a metadata stub
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

# ── 3. vLLM import ────────────────────────────────────────────────────────────
step 3 "vLLM — import"

# Install into the venv if not already present (install-only, no model load)
VLLM_INSTALLED=$(.venv/bin/python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
if [[ -z "$VLLM_INSTALLED" ]]; then
    warn "vLLM not installed — installing now (this may take a minute)..."
    if ! uv pip install --quiet "vllm>=0.7" 2>&1 | tail -3; then
        fail "vLLM pip install failed — likely no pre-built ROCm wheel for this arch."
        FAILURES=$((FAILURES + 1))
        VLLM_INSTALLED=""
    else
        VLLM_INSTALLED=$(.venv/bin/python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
    fi
fi

if [[ -n "$VLLM_INSTALLED" ]]; then
    IMPORT_OUT=$(.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)" 2>&1)
    if echo "$IMPORT_OUT" | grep -qi "error\|traceback\|exception"; then
        fail "vLLM import raised an error:"
        while IFS= read -r line; do echo "         $line"; done <<< "$IMPORT_OUT"
        FAILURES=$((FAILURES + 1))
    else
        pass "$IMPORT_OUT"
    fi
fi

# ── 4. vLLM device probe ──────────────────────────────────────────────────────
step 4 "vLLM — device probe (does vLLM see the GPU?)"
if [[ -z "$VLLM_INSTALLED" ]]; then
    warn "Skipped — vLLM not available."
else
    PROBE_OUT=$(.venv/bin/python - 2>&1 <<'PY'
import sys
try:
    from vllm.platforms import current_platform
    print(f"platform: {current_platform.get_device_name()} / {current_platform.__class__.__name__}")
except Exception as e:
    # Older vLLM versions don't have current_platform
    try:
        import vllm.utils as vu
        name = vu.get_device_name() if hasattr(vu, "get_device_name") else "unknown"
        print(f"platform (legacy): {name}")
    except Exception as e2:
        print(f"FAIL vLLM device probe raised: {e2}")
        sys.exit(1)
PY
    )
    if echo "$PROBE_OUT" | grep -q "^FAIL"; then
        fail "$(echo "$PROBE_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
        FAILURES=$((FAILURES + 1))
    else
        while IFS= read -r line; do pass "$line"; done <<< "$PROBE_OUT"
    fi
fi

# ── 5. vLLM engine init (no weights) ─────────────────────────────────────────
step 5 "vLLM — engine init without loading model weights"
if [[ -z "$VLLM_INSTALLED" ]]; then
    warn "Skipped — vLLM not available."
else
    # Use a tiny placeholder to probe engine startup without downloading anything.
    # The goal is to see whether the worker/executor initialises on the GPU —
    # we expect it to fail at the weight-loading stage, not before.
    ENGINE_OUT=$(timeout 60 .venv/bin/python - 2>&1 <<'PY' || true
import sys, os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    from vllm import LLM
    # facebook/opt-125m is tiny and well-known; we won't actually download it.
    # enforce_eager=True avoids CUDA-graph compilation, which is unsupported on ROCm.
    llm = LLM(
        model="facebook/opt-125m",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
        download_dir="/tmp/vllm_probe",
    )
    print("ENGINE_INIT_OK")
except Exception as e:
    err = str(e)
    # Weight-loading errors mean the engine started fine — GPU init worked.
    if any(k in err for k in ("weight", "checkpoint", "model files", "HTTPError", "not found", "download")):
        print(f"ENGINE_STARTED_OK (failed at weight load, as expected in a no-model probe)")
    else:
        print(f"FAIL engine init failed before weight load: {e}")
        sys.exit(1)
PY
    )
    if echo "$ENGINE_OUT" | grep -q "^FAIL"; then
        fail "$(echo "$ENGINE_OUT" | grep "^FAIL" | sed 's/^FAIL //')"
        echo "$ENGINE_OUT" | grep -v "^FAIL" | tail -10 | sed 's/^/         /'
        FAILURES=$((FAILURES + 1))
    elif echo "$ENGINE_OUT" | grep -qE "ENGINE_INIT_OK|ENGINE_STARTED_OK"; then
        pass "$(echo "$ENGINE_OUT" | grep -E "ENGINE_INIT_OK|ENGINE_STARTED_OK")"
    else
        warn "Inconclusive — inspect output:"
        echo "$ENGINE_OUT" | tail -15 | sed 's/^/         /'
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Result ━━━${NC}"
if [[ "$FAILURES" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All steps passed.${NC} vLLM appears usable on this ROCm GPU."
    echo ""
    echo "  If you want to use vLLM for the consultant, update amd_r9700_setup.sh"
    echo "  to install it and switch serve_consultant.sh as the serve script."
else
    echo -e "${RED}${BOLD}$FAILURES step(s) failed.${NC}"
    echo ""
    echo "  vLLM is likely not usable on this GPU yet."
    echo "  Check upstream support: https://github.com/vllm-project/vllm/issues"
    echo "  and search for your GPU arch (e.g. gfx1201)."
fi
echo ""
exit "$FAILURES"
