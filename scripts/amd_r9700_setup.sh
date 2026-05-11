#!/usr/bin/env bash
# One-time setup for a local AMD GPU machine (tested on AMD R9700 AI PRO, gfx1201).
# Works on any Linux box with ROCm drivers and uv installed.
#
# NOTE: vLLM does not have stable upstream support for gfx1201 (RDNA 4) as of
# April 2026. The teacher model is served via HF Transformers (serve_teacher_local.sh).
# For the consultant model, use an online API or a separate NVIDIA machine.
#
# Usage:
#   bash scripts/amd_r9700_setup.sh              # Install deps only
#   bash scripts/amd_r9700_setup.sh --models     # Install deps + download models (~19 GB)

set -euo pipefail
cd "$(dirname "$0")/.."

# Strip pyenv shims from PATH entirely. pyenv shims exit 127 when pyenv doesn't
# manage the Python version in use, and uv also calls bare `python` through
# PATH — so shims must be gone before any uv invocation.
if [[ -n "${PYENV_ROOT:-}" ]] || echo "$PATH" | grep -q '\.pyenv'; then
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '\.pyenv' | tr '\n' ':' | sed 's/:$//')"
    export PATH
    unset PYENV_VERSION PYENV_ROOT 2>/dev/null || true
fi

DOWNLOAD_MODELS=false
for arg in "$@"; do
    [[ "$arg" == "--models" ]] && DOWNLOAD_MODELS=true
done

# ── Colours & helpers ────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}── $* ──${NC}"; }

# ── Logging ───────────────────────────────────────────────────────────────────
mkdir -p logs
LOG_FILE="logs/amd_r9700_setup.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Log: $(pwd)/$LOG_FILE"

echo ""
echo -e "${BOLD}KELE / AMD R9700 Setup${NC}"
echo "=============================="
date
hostname
echo ""

# ── 1. ROCm driver check ──────────────────────────────────────────────────────
step "Checking ROCm driver"
command -v rocm-smi &>/dev/null || die "rocm-smi not found — install ROCm drivers first: https://rocm.docs.amd.com/"
rocm-smi --showproductname --showmeminfo vram 2>/dev/null || rocm-smi 2>/dev/null || true

GPU_COUNT=$(rocm-smi --showproductname --format=csv,noheader 2>/dev/null | grep -c "." || true)
[[ -z "${GPU_COUNT:-}" || "${GPU_COUNT}" == "0" ]] && GPU_COUNT=1
info "GPU(s) detected: $GPU_COUNT"

# ── 2. ROCm version → PyTorch wheel selection ─────────────────────────────────
step "Detecting ROCm version for PyTorch wheel"

_rocm_version() {
    # Prefer the canonical version file; fall back to rocminfo
    if [[ -f /opt/rocm/.info/version ]]; then
        cat /opt/rocm/.info/version
    elif command -v rocminfo &>/dev/null; then
        rocminfo 2>/dev/null \
            | awk '/ROCm Runtime/{match($0,/[0-9]+\.[0-9]+/); print substr($0,RSTART,RLENGTH); exit}'
    fi
}

ROCM_VER="${ROCM_VERSION:-$(_rocm_version)}"
ROCM_MAJOR="${ROCM_VER%%.*}"
ROCM_MINOR="${ROCM_VER#*.}"; ROCM_MINOR="${ROCM_MINOR%%.*}"

# Map ROCm version to pytorch.org wheel index.
# Only rocm7.2 is available as of May 2026 — extend this as new wheels ship.
if   [[ "$ROCM_MAJOR" -gt 7 ]];                              then TORCH_INDEX="rocm7.2"
elif [[ "$ROCM_MAJOR" -eq 7 && "$ROCM_MINOR" -ge 2 ]];      then TORCH_INDEX="rocm7.2"
else
    warn "ROCm $ROCM_VER — no matching wheel found; defaulting to rocm7.2."
    warn "If torch fails to load, check: https://pytorch.org/get-started/locally/"
    TORCH_INDEX="rocm7.2"
fi

TORCH_INDEX="${TORCH_INDEX:-rocm7.2}"   # env var TORCH_INDEX override wins
info "ROCm $ROCM_VER detected → using pytorch.org/whl/$TORCH_INDEX"

# ── 3. GPU architecture check (informational) ─────────────────────────────────
step "Checking GPU architecture"
if command -v rocminfo &>/dev/null; then
    GPU_ARCH=$(rocminfo 2>/dev/null | awk '/Name:.*gfx/{print $NF; exit}') || true
    if [[ -n "$GPU_ARCH" ]]; then
        info "GPU arch: $GPU_ARCH"
        if [[ "$GPU_ARCH" != "gfx1201" ]]; then
            warn "Expected gfx1201 (R9700 AI PRO) but found $GPU_ARCH — proceeding anyway."
        fi
    fi
fi

# ── 4. Python ≥ 3.12 ──────────────────────────────────────────────────────────
step "Checking Python"

_has_python312() {
    for cmd in python3.12 python3 python; do
        command -v "$cmd" &>/dev/null || continue
        ver=$("$cmd" --version 2>&1 | awk '{print $2}')
        major="${ver%%.*}"; rest="${ver#*.}"; minor="${rest%%.*}"
        [[ "$major" -ge 3 && "$minor" -ge 12 ]] && return 0
    done
    return 1
}

if ! _has_python312; then
    if command -v pyenv &>/dev/null; then
        info "Python >=3.12 not in PATH — trying pyenv..."
        pyenv install --skip-existing 3.12
        PATH="$(pyenv root)/versions/3.12.*/bin:$PATH"
        export PATH
    else
        die "Python >=3.12 not found. Install it via pyenv, your distro's package manager, or from python.org."
    fi
fi

PYTHON=""
for cmd in python3.12 python3 python; do
    command -v "$cmd" &>/dev/null || continue
    ver=$("$cmd" --version 2>&1 | awk '{print $2}')
    major="${ver%%.*}"; rest="${ver#*.}"; minor="${rest%%.*}"
    if [[ "$major" -ge 3 && "$minor" -ge 12 ]]; then
        PYTHON="$cmd"; break
    fi
done
[[ -n "$PYTHON" ]] || die "Python >=3.12 still not found after pyenv. Check your shell PATH."
info "Using $PYTHON — $("$PYTHON" --version)"

# ── 5. uv ─────────────────────────────────────────────────────────────────────
step "Checking uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    info "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    info "Add the following to your ~/.bashrc or ~/.zshrc:"
    # shellcheck disable=SC2016
    echo '    export PATH="$HOME/.local/bin:$PATH"'
fi
uv --version

# ── 6. Project dependencies ───────────────────────────────────────────────────
step "Installing project deps (uv sync)"
uv sync --group dev --python "$PYTHON"

# ── 7. PyTorch (ROCm) ─────────────────────────────────────────────────────────
step "Installing PyTorch ($TORCH_INDEX)"
uv pip install --force-reinstall \
    torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
info "PyTorch installed."

# ── 8. Sanity check ───────────────────────────────────────────────────────────
step "Sanity check"
.venv/bin/python - <<'PY'
import torch, sys

print(f"  torch  {torch.__version__}")

# ROCm builds reuse the CUDA Python API; is_available() returns True for ROCm GPUs.
print(f"  ROCm/CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    hip_ver = getattr(torch.version, "hip", None)
    if hip_ver:
        print(f"  HIP    {hip_ver}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB")
else:
    print("  ERROR: GPU not available — check ROCm drivers and torch wheel", file=sys.stderr)
    sys.exit(1)
PY
info "All imports OK."

# ── 9. .env ───────────────────────────────────────────────────────────────────
step "Environment file"
if [[ ! -f .env ]]; then
    cp .env.example .env
    info "Created .env from .env.example."
    info "For a fully local run (teacher only), edit .env and set:"
    echo "    CONSULTANT_API_KEY=<your-openai-or-remote-key>"
    echo "    CONSULTANT_BASE_URL=<remote-consultant-url>"
    echo "    CONSULTANT_MODEL_NAME=<remote-model-name>"
else
    info ".env already exists — skipping."
fi

# ── 10. Model downloads (optional) ────────────────────────────────────────────
if [[ "$DOWNLOAD_MODELS" == true ]]; then
    step "Downloading teacher model (~19 GB)"
    HF_HOME="${HF_HOME:-$HOME/hf_models}"
    mkdir -p "$HF_HOME"
    info "Model destination: $HF_HOME"

    if ! uv run hf auth whoami &>/dev/null; then
        warn "Not logged in to Hugging Face."
        warn "Run: uv run hf auth login"
        warn "Then re-run with --models to download."
        exit 1
    fi

    info "Downloading SocratTeachLLM (~19 GB)..."
    uv run hf download ulises-c/SocratTeachLLM \
        --local-dir "$HF_HOME/SocratTeachLLM"

    info "Download complete."
    echo ""
    du -sh "$HF_HOME/SocratTeachLLM" 2>/dev/null || true
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Setup complete!${NC}"
echo ""
if [[ "$DOWNLOAD_MODELS" == false ]]; then
    echo "Next steps:"
    echo "  1. Download the teacher model (needs internet, ~19 GB):"
    echo "       bash scripts/amd_r9700_setup.sh --models"
    echo ""
    echo "  2. Start the teacher server:"
    echo "       bash scripts/serve_teacher_local.sh"
    echo ""
    echo "  3. Run evaluation (in a second terminal once server is ready):"
    echo "       uv run kele-eval"
else
    echo "Next steps:"
    echo "  1. Start the teacher server:"
    echo "       bash scripts/serve_teacher_local.sh"
    echo ""
    echo "  2. Run evaluation (in a second terminal once server is ready):"
    echo "       uv run kele-eval"
fi
echo ""
echo "  NOTE: vLLM is not installed — gfx1201 (RDNA 4) lacks stable vLLM support"
echo "  as of May 2026. The consultant model must be served remotely or via Ollama."
echo "  Check upstream: https://github.com/vllm-project/vllm/issues"
echo ""
echo "  Monitor GPU usage:"
echo "       watch -n 2 rocm-smi"
echo ""
echo "  Log from this run: $LOG_FILE"
