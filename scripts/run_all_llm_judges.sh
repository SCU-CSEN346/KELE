#!/usr/bin/env bash
# Run LLM-judge sequentially across all configs we want scored.
# Sequential rather than parallel to avoid rate-limit thrash with
# the in-flight Phase 3 + experiment-A runs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CONFIGS=(
  # Phase 2 winners (top-3 stack)
  results/bert-gemma-composed-top3-n50
  results/bert-consultant-fewshot10-claude-opus-n50
  results/bert-consultant-fewshot10-claude-sonnet-n50
  results/bert-a3b-composed-top3-n50

  # Mid-tier (10-shot only, no top-3)
  results/bert-claude-sonnet-fewshot10-n50
  results/bert-claude-opus-fewshot10-n50

  # Raw Claude (no prompt engineering)
  results/bert-claude-sonnet-raw-n50
  results/bert-claude-opus-raw-n50

  # Experiment-A (Claude consultant + SocratTeachLLM teacher) — the smoking gun
  results/claude-opus-consultant-socratteachllm-n50
  results/claude-sonnet-consultant-socratteachllm-n50-clean   # Workers=1 clean rerun

  # Cross-lingual / SocratDataset-EN runs (overfit transfer test)
  results/claude-opus-consultant-socratteachllm-EN-n50
  results/claude-sonnet-consultant-socratteachllm-EN-n50
  results/bert-claude-opus-top3-EN-n50

  # Locked headlines + Phase 3 full runs for reference
  results/bert-consultant-fewshot10-gemma-full   # The paper-locked headline at n=681
  results/bert-consultant-fewshot10-a3b-full     # Phase 0.5 teacher ablation at n=681
  results/bert-claude-opus-top3-n681             # Phase 3-Opus (when complete)
  results/bert-claude-sonnet-top3-n681           # Phase 3-Sonnet (when complete)
)

ANTHROPIC_API_KEY=$(grep "^ANTHROPIC_API_KEY=" .env | cut -d= -f2- | tr -d "'\"")
export ANTHROPIC_API_KEY

for CFG in "${CONFIGS[@]}"; do
  if [[ ! -d "$CFG/dialogues" ]]; then
    echo "SKIP (no dialogues/): $CFG"
    continue
  fi
  if [[ -f "$CFG/judge_summary.json" ]]; then
    echo "SKIP (already judged): $CFG"
    continue
  fi
  echo ""
  echo "============================================================"
  echo "  JUDGING: $CFG"
  echo "============================================================"
  PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/llm_judge_eval.py "$CFG" --workers 16
done

echo ""
echo "=== All judging done. ==="
