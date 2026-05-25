#!/usr/bin/env bash
#
# Workaround for Triton 3.6.0 + RDNA4 (gfx1201) software-pipelining use-after-free.
#
# Forces num_stages=1 in flash-linear-attention Triton autotune configs.
# Mirrors the upstream sageattention fix pattern (set num_stages=1 when
# torch.version.hip is non-None) which targets the same Triton compiler bug:
# the AMD `tritonamdgpu-pipeline` pass corrupts IR when num_stages >= 2 on
# gfx1201, producing either `'tt.load' op operation destroyed but still has
# uses` (sageattn manifestation) or a silent first-step deadlock (FLA
# manifestation — 120 kernels compile, one hangs indefinitely).
#
# Re-run after every `uv sync` because uv reinstalls a fresh FLA wheel that
# resets the patched files. Symmetric with the existing `_install-torch-rocm`
# story documented in PR #79.
#
# Refs:
#   - kijai/ComfyUI-WanVideoWrapper#2007 (sageattn num_stages=1 fix on ROCm)
#   - HF Transformers docs: Qwen3.5 DeltaNet falls back to slower PyTorch ops
#     when fla/causal-conv1d kernels are unavailable.
#   - PR #79 comment "AMD ROCm / gfx1201 known issues" + Gemini/ChatGPT thread.
#
# Usage:
#   bash scripts/patch_fla_rocm.sh             # apply
#   bash scripts/patch_fla_rocm.sh --dry-run   # count refs, no changes
#   bash scripts/patch_fla_rocm.sh --restore   # roll back from .bak files

set -euo pipefail

mode='apply'
for arg in "$@"; do
  case "$arg" in
    --dry-run) mode='dry-run' ;;
    --restore) mode='restore' ;;
    -h|--help)
      sed -n '1,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      printf 'error: unknown flag %s\n' "$arg" >&2
      exit 2
      ;;
  esac
done

# Confirm ROCm — this patch is a no-op (and a regression) on CUDA.
if ! uv run --no-sync python -c 'import torch, sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null; then
  printf 'error: torch.version.hip is not set — this is the CUDA build of torch.\n' >&2
  printf '  This patch is only meant for ROCm/HIP. Aborting.\n' >&2
  exit 1
fi

# Resolve FLA install path inside the project venv.
fla_path="$(uv run --no-sync python -c 'import fla, pathlib; print(pathlib.Path(fla.__file__).parent)' 2>/dev/null || true)"
if [[ -z "${fla_path}" || ! -d "${fla_path}/ops" ]]; then
  printf 'error: FLA ops dir not found (path=%s)\n' "${fla_path:-<unset>}" >&2
  printf '  install with:  uv run pip install flash-linear-attention\n' >&2
  exit 1
fi
printf '→ FLA ops dir: %s\n' "${fla_path}/ops"

# ── restore mode ──────────────────────────────────────────────────────────────
if [[ "$mode" == 'restore' ]]; then
  restored=0
  while IFS= read -r -d '' bak; do
    mv "$bak" "${bak%.bak}"
    restored=$((restored + 1))
  done < <(find "${fla_path}/ops" -type f -name '*.py.bak' -print0)
  if [[ "$restored" -eq 0 ]]; then
    printf '  no .bak files found — nothing to restore\n'
  else
    printf '✓ restored %d files from .bak\n' "$restored"
  fi
  exit 0
fi

# ── count current num_stages >= 2 references ─────────────────────────────────
# Pattern: `num_stages=<2..9>` or `num_stages = <2..99>` (autotune configs)
before=$(grep -rhEc 'num_stages[[:space:]]*=[[:space:]]*([2-9]|[1-9][0-9]+)' \
  --include='*.py' "${fla_path}/ops" 2>/dev/null \
  | awk '{s+=$1} END {print s+0}')
printf '  num_stages>=2 references found: %d\n' "$before"

if [[ "$mode" == 'dry-run' ]]; then
  printf '(dry-run — no changes)\n'
  exit 0
fi

if [[ "$before" -eq 0 ]]; then
  printf '  (already patched or no autotune configs — nothing to do)\n'
  exit 0
fi

# ── apply patch ──────────────────────────────────────────────────────────────
# sed -i.bak retains a .bak alongside each modified file for --restore.
find "${fla_path}/ops" -type f -name '*.py' -exec sed -i.bak -E \
  's/(num_stages[[:space:]]*=[[:space:]]*)([2-9]|[1-9][0-9]+)/\11/g' {} +

after=$(grep -rhEc 'num_stages[[:space:]]*=[[:space:]]*([2-9]|[1-9][0-9]+)' \
  --include='*.py' "${fla_path}/ops" 2>/dev/null \
  | awk '{s+=$1} END {print s+0}')
printf '  num_stages>=2 references remaining: %d\n' "$after"

# Clear Triton kernel cache so previously-cached broken kernels are not reloaded.
triton_cache="${HOME}/.triton/cache"
if [[ -d "$triton_cache" ]]; then
  printf '→ clearing %s\n' "$triton_cache"
  rm -rf "$triton_cache"
fi

# Clear FLA bytecode cache so patched sources take effect on next import.
find "${fla_path}/ops" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

printf '✓ FLA ROCm patch applied (changed %d → %d).\n' "$before" "$after"
printf '  to revert:  bash scripts/patch_fla_rocm.sh --restore\n'
# shellcheck disable=SC2016  # backticks are intentional markdown-style command markers, not command substitution
printf '  re-run after any `uv sync` or `make install-rocm`.\n'
