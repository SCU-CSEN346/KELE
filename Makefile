.PHONY: help run install-hooks pre-commit sync-mirror setup setup-repo \
		slurm \
        post-eval-shutdown run-eval \
        eval-qwen27b-smoke eval-qwen27b-mini eval-qwen27b-full \
        eval-qwen27b-fusion-smoke eval-qwen27b-fusion-nothink-smoke \
        eval-qwen35b-a3b-smoke eval-qwen35b-a3b-mini eval-qwen35b-a3b-full \
        eval-qwen35b-a3b-fusion-smoke eval-qwen35b-a3b-fusion-nothink-smoke \
        eval-gemma4-31b-smoke eval-gemma4-31b-mini eval-gemma4-31b-full \
        eval-gemma4-31b-fusion-smoke \
        serve-both serve-dual-gpu serve-consultant serve-gemma4 \
        serve-gemma4-31b serve-gemma4-26b-a4b \
        serve-qwen27b serve-qwen27b-q4 serve-qwen35b-a3b \
        serve-glm47-23b serve-qwopus35b-a3b \
        serve-socratteachllm serve-teacher-online \
        setup-l40s start-local-tl-server \
        test-gpu-stack test-vllm \
        tournament tournament-think tournament-warmup tournament-warmup-think tournament-status tournament-eliminate \
        tournament-finalize tournament-archive tournament-restore tournament-reset \
        tournament-download tournament-help

# Default target
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  run                   Show how to launch the project via uv"
	@echo "  install-hooks         Install git hooks from hooks/ into .git/hooks/"
	@echo ""
	@echo "  GPU stack tests:"
	@echo "  test-gpu-stack        Full ML stack: ROCm, torch, bitsandbytes 8/4-bit, transformers, PEFT, TRL, flash-attn"
	@echo "  test-vllm             vLLM ROCm engine probe (no model weights)"
	@echo ""
	@echo "  Scripts (scripts/):"
	@echo "  post-eval-shutdown    Run scripts/post_eval_shutdown.sh"
	@echo "  run-eval              Run scripts/run_eval.sh  (GPU=<config>, default: baseline)"
	@echo "                          Dual-GPU configs: GPU=l40s  GPU=3090ti"
	@echo "                          Other configs:    GPU=baseline  GPU=gemma4"
	@echo "                          Tested hardware:  RTX 5090, RTX 3090 Ti, AMD R9700, NVIDIA L40S, V100 32GB"
	@echo "  setup-l40s            Run scripts/l40s_setup.sh (one-time setup for dual L40S machine)"
	@echo "  serve-both            Run scripts/serve_both.sh (single GPU, shared VRAM)"
	@echo "  serve-dual-gpu        Run scripts/serve_dual_gpu.sh (2 GPUs, teacher→GPU0 consultant→GPU1)"
	@echo "  serve-consultant      Run scripts/serve_consultant.sh"
	@echo "  serve-gemma4          Run scripts/serve_gemma4.sh (vLLM + Gemma-4-31B-IT-NVFP4, multi-server)"
	@echo "  serve-gemma4-31b      Run scripts/serve_gemma4_31b_q5.sh (Gemma 4 31B Q5 GGUF, dual-role on llama.cpp)"
	@echo "  serve-qwen27b         Run scripts/serve_qwen27b_q5.sh (Qwen3.6-27B Q5, dual-role teacher+consultant)"
	@echo "  serve-qwen27b-q4      Run scripts/serve_qwen27b_q4_local.sh (Qwen3.6-27B Q4, ~/models/, AMD R9700)"
	@echo "  serve-qwen35b-a3b     Run scripts/serve_qwen35b_a3b.sh (Qwen3.6-35B-A3B MoE, ~3x faster than 27B)"
	@echo "  serve-glm47-23b       Run scripts/serve_glm47_23b.sh (GLM-4.7-Flash REAP 23B-A3B, 14 GB, AMD R9700)"
	@echo "  serve-qwopus35b-a3b   Run scripts/serve_qwopus35b_a3b.sh (Qwopus 35B-A3B LoRA fine-tune, 21 GB)"
	@echo "  serve-socratteachllm  Run scripts/serve_socratteachllm.sh"
	@echo "  serve-teacher-online  Run scripts/serve_teacher_online.sh"
	@echo "  start-local-tl-server  Start local llama.cpp server for dataset translation (Qwen3.5-9B)"
	@echo "  eval-qwen27b-smoke    Run scripts/eval_qwen27b.sh smoke (n=5,   ~5 min)"
	@echo "  eval-qwen27b-mini     Run scripts/eval_qwen27b.sh mini  (n=25,  ~15 min)"
	@echo "  eval-qwen27b-full     Run scripts/eval_qwen27b.sh full  (n=681, ~75 h — measured)"
	@echo "  eval-qwen35b-a3b-smoke Run scripts/eval_qwen35b_a3b.sh smoke (n=5,   ~2 min projected)"
	@echo "  eval-qwen35b-a3b-mini  Run scripts/eval_qwen35b_a3b.sh mini  (n=25,  ~5 min projected)"
	@echo "  eval-qwen35b-a3b-full  Run scripts/eval_qwen35b_a3b.sh full  (n=681, ~20-30 h projected)"
	@echo "  eval-gemma4-31b-smoke  Run scripts/eval_gemma4_31b.sh smoke  (n=5)"
	@echo "  eval-gemma4-31b-mini   Run scripts/eval_gemma4_31b.sh mini   (n=25)"
	@echo "  eval-gemma4-31b-full   Run scripts/eval_gemma4_31b.sh full   (n=681)"
	@echo ""
	@echo "  Fusion smoke targets (single-call architecture, see SOCRATIC_FUSION_PLAN.md):"
	@echo "  eval-qwen27b-fusion-smoke           27B + unified (think on)"
	@echo "  eval-qwen27b-fusion-nothink-smoke   27B + unified + no-think"
	@echo "  eval-qwen35b-a3b-fusion-smoke         A3B + unified (think on)"
	@echo "  eval-qwen35b-a3b-fusion-nothink-smoke A3B + unified + no-think"
	@echo "  eval-gemma4-31b-fusion-smoke          Gemma 4 31B + unified (Gemma has no thinking-mode)"
	@echo ""
	@echo "  WAVE HPC (SLURM):"
	@echo "  slurm                 git pull + sbatch wave_eval.slurm + print status"
	@echo ""
	@echo "  Tournament (multi-model elimination):"
	@echo "  tournament-help       Full tournament command reference"
	@echo "  tournament            Run one round (n=50, fusion, no-think)"
	@echo "  tournament-think      Run one round (n=50, fusion, thinking budget=8192)"
	@echo "  tournament-warmup     Warmup (n=5, thinking OFF) — verifies models load; do NOT eliminate after"
	@echo "  tournament-warmup-think  Warmup (n=5, thinking budget=8192) — verifies thinking tokens are generated"
	@echo "  tournament-status     Print leaderboard"
	@echo "  tournament-archive    Save current run to archive/<run_id>/ and reset state"
	@echo "  tournament-restore    List archives  (use ID=<id> to restore one)"
	@echo "  tournament-eliminate  Drop worst model  (N=2 to drop two, etc.)"
	@echo "  tournament-finalize   Run survivors to n=681 (fusion mode)"
	@echo "  tournament-reset      Wipe all tournament state  (add CONFIRM=1 to skip prompt)"
	@echo "  tournament-download   Download all pending model weights via hf CLI"


# ── Setup ─────────────────────────────────────────────────────────────────────

setup: setup-repo install-hooks
	@echo "This project uses uv for dependency management. Install uv from https://docs.astral.sh/uv/ if not already installed."
	@echo "Setting up the project via uv:"
	uv sync --group dev

setup-repo:
	@echo "Configuring dual-push remotes..."
	# Set the fetch URL
	git remote set-url origin git@github.com:ulises-c/csen-346.git
	# Replace push URL list (--push without --add resets to a single entry)
	git remote set-url --push origin git@github.com:ulises-c/csen-346.git
	# Add the second push URL (now idempotent: list was just reset above)
	git remote set-url --add --push origin git@github.com:SCU-CSEN346/KELE.git
	@echo "Repository setup complete. Verify with 'git remote -v'."

# ── Dual remote synchronization ────────────────────────────────────────────────────

sync-mirror:
	@CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$CURRENT_BRANCH" = "main" ]; then \
		echo "Already on main. Performing standard pull/push sync..."; \
		git pull origin main && git push origin main --tags; \
	else \
		echo "On $$CURRENT_BRANCH. Performing background sync for main..."; \
		git fetch origin main:main && git push origin main:main --tags; \
	fi
	@echo "Mirror sync successful."

# ── Entry point ──────────────────────────────────────────────────────────────

run:
	@echo "Run the project via uv:"
	@echo ""
	@echo "  uv run kele            # main KELE entry point"
	@echo "  uv run kele-eval       # run evaluation"
	@echo "  uv run serve-teacher   # start teacher server"
	@echo ""
	@echo "  uv run test            # run tests (or: make test)"
	@echo "  uv run lint            # lint source  (or: make lint)"
	@echo ""
	@echo "  make pre-commit        # run format + lint + tests (mirrors git pre-commit hook)"

# ── Code quality ─────────────────────────────────────────────────────────────

pre-commit:
	uvx ruff format .
	uvx ruff check --fix .
	uv run pytest -rs

# ── Developer setup ──────────────────────────────────────────────────────────

install-hooks:
	@echo "Installing git hooks from hooks/ → .git/hooks/ …"
	@for hook in hooks/*; do \
	  name=$$(basename $$hook); \
	  cp "$$hook" ".git/hooks/$$name"; \
	  chmod +x ".git/hooks/$$name"; \
	  echo "  installed $$name"; \
	done
	@echo "Done. Hooks will run automatically on git operations."

# ── scripts/ targets ─────────────────────────────────────────────────────────

setup-l40s:
	bash scripts/l40s_setup.sh

post-eval-shutdown:
	bash scripts/post_eval_shutdown.sh

# TODO: auto-detect GPU config from hardware — query nvidia-smi for compute
# capability and total VRAM per device, then select the appropriate configs/
# file automatically (e.g. 2×24GB CC≥8.6 → 3090ti, 2×48GB CC≥8.9 → l40s,
# single GPU → serve-both, V100/CC<8.0 → float16 + enforce-eager, etc.).
# Planned: make run-eval with no GPU= arg runs detection and picks the config.
GPU ?= baseline

run-eval:
	bash scripts/run_eval.sh $(GPU)

serve-both:
	bash scripts/serve_both.sh

serve-dual-gpu:
	bash scripts/serve_dual_gpu.sh

serve-consultant:
	bash scripts/serve_consultant.sh

serve-gemma4:
	bash scripts/serve_gemma4.sh

serve-gemma4-31b:
	bash scripts/serve_gemma4_31b_q5.sh

serve-gemma4-26b-a4b:
	bash scripts/serve_gemma4_26b_a4b.sh

serve-qwen27b:
	bash scripts/serve_qwen27b_q5.sh

serve-qwen27b-q4:
	bash scripts/serve_qwen27b_q4_local.sh

serve-socratteachllm:
	bash scripts/serve_socratteachllm.sh

serve-teacher-online:
	bash scripts/serve_teacher_online.sh

start-local-tl-server:
	bash scripts/start_tl_server.sh

test-gpu-stack:
	bash scripts/test_gpu_stack.sh

test-vllm:
	bash scripts/test_vllm_rocm.sh

eval-qwen27b-smoke:
	bash scripts/eval_qwen27b.sh smoke

eval-qwen27b-mini:
	bash scripts/eval_qwen27b.sh mini

eval-qwen27b-full:
	bash scripts/eval_qwen27b.sh full

serve-qwen35b-a3b:
	bash scripts/serve_qwen35b_a3b.sh

serve-glm47-23b:
	bash scripts/serve_glm47_23b.sh

serve-qwopus35b-a3b:
	bash scripts/serve_qwopus35b_a3b.sh

eval-qwen35b-a3b-smoke:
	bash scripts/eval_qwen35b_a3b.sh smoke

eval-qwen35b-a3b-mini:
	bash scripts/eval_qwen35b_a3b.sh mini

eval-qwen35b-a3b-full:
	bash scripts/eval_qwen35b_a3b.sh full

eval-gemma4-31b-smoke:
	bash scripts/eval_gemma4_31b.sh smoke

eval-gemma4-31b-mini:
	bash scripts/eval_gemma4_31b.sh mini

eval-gemma4-31b-full:
	bash scripts/eval_gemma4_31b.sh full

# ── Fusion smoke targets (single-call architecture) ──────────────────────────
# See docs/SOCRATIC_FUSION_PLAN.md. Each writes to a distinct results/ dir
# so all four can coexist alongside the existing two-call smoke results.

eval-qwen27b-fusion-smoke:
	bash scripts/eval_qwen27b.sh smoke --unified

eval-qwen27b-fusion-nothink-smoke:
	bash scripts/eval_qwen27b.sh smoke --unified --nothink

eval-qwen35b-a3b-fusion-smoke:
	bash scripts/eval_qwen35b_a3b.sh smoke --unified

eval-qwen35b-a3b-fusion-nothink-smoke:
	bash scripts/eval_qwen35b_a3b.sh smoke --unified --nothink

# Gemma 4 has no thinking-mode equivalent, so only the --unified variant exists.
eval-gemma4-31b-fusion-smoke:
	bash scripts/eval_gemma4_31b.sh smoke --unified

# ── Tournament ────────────────────────────────────────────────────────────────

tournament-help:
	@echo ""
	@echo "Tournament — multi-model elimination benchmark"
	@echo "=============================================="
	@echo ""
	@echo "RUNNING"
	@echo "  make tournament                    Run one round (n=50, fusion, thinking OFF)"
	@echo "  make tournament-think              Run one round (n=50, fusion, thinking budget=8192)"
	@echo "  make tournament-warmup             Smoke test — n=5, thinking OFF"
	@echo "  make tournament-warmup-think       Smoke test — n=5, thinking budget=8192 (verify thinking_content in dialogues)"
	@echo "  make tournament N=<n>              Custom dialogue count, e.g. make tournament N=20"
	@echo "  make tournament-finalize           Run the 3 survivors to full n=681"
	@echo ""
	@echo "LEADERBOARD"
	@echo "  make tournament-status             Print leaderboard + detailed metrics table"
	@echo ""
	@echo "ELIMINATION"
	@echo "  make tournament-eliminate          Drop 1 worst-scoring model"
	@echo "  make tournament-eliminate N=2      Drop 2 worst (floor: 3 finalists always remain)"
	@echo ""
	@echo "ARCHIVE / RESTORE"
	@echo "  make tournament-archive            Save current run to archive/<run_id>/ and reset"
	@echo "                                     Re-archiving the same run ID overwrites (appends rounds)"
	@echo "  make tournament-restore            List all archived runs with ID, date, round, TB"
	@echo "  make tournament-restore ID=<id>    Restore run <id>; auto-archives current run first"
	@echo ""
	@echo "SETUP"
	@echo "  make tournament-download           Download any missing model weights via hf CLI"
	@echo "  make tournament-reset CONFIRM=1    Wipe everything (state + round dirs) — no undo"
	@echo ""
	@echo "TYPICAL WORKFLOW"
	@echo "  1. make tournament-warmup          # verify all models boot"
	@echo "  2. make tournament                 # run no-think round 1"
	@echo "  3. make tournament-status          # review results"
	@echo "  4. make tournament-archive         # save no-think run (gets a run_id)"
	@echo "  5. make tournament-think           # run thinking=8192 round 1"
	@echo "  6. make tournament-status"
	@echo "  7. make tournament-archive"
	@echo "  8. make tournament-restore ID=<no-think-id>  # switch back if needed"
	@echo ""

tournament-warmup:
	uv run tournament run --n 5 --unified

tournament-warmup-think:
	uv run tournament run --n 5 --unified --thinking-budget 8192

tournament:
	uv run tournament run --unified $(if $(N),--n $(N),)

tournament-think:
	uv run tournament run --unified --thinking-budget 8192 $(if $(N),--n $(N),)

tournament-status:
	uv run tournament status

tournament-eliminate:
	uv run tournament eliminate $(N)

tournament-finalize:
	uv run tournament finalize --unified

tournament-archive:
	uv run tournament archive

tournament-restore:
	@if [ -z "$(ID)" ]; then \
	  uv run tournament restore; \
	else \
	  uv run tournament restore $(ID); \
	fi

tournament-reset:
	@if [ -z "$(CONFIRM)" ]; then \
	  echo "This wipes results/tournament/ entirely. Re-run with CONFIRM=1."; \
	else \
	  uv run tournament reset --confirm; \
	fi

tournament-download:
	uv run tournament download

# ── WAVE HPC ──────────────────────────────────────────────────────────────────

slurm:
	bash scripts/slurm/submit_wave.sh
