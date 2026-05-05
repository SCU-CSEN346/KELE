.PHONY: help run install-hooks pre-commit sync-mirror setup setup-repo \
		slurm \
        post-eval-shutdown run-eval \
        eval-qwen27b-smoke eval-qwen27b-mini eval-qwen27b-full \
        eval-qwen27b-fusion-smoke eval-qwen27b-fusion-nothink-smoke \
        eval-qwen35b-a3b-smoke eval-qwen35b-a3b-mini eval-qwen35b-a3b-full \
        eval-qwen35b-a3b-fusion-smoke eval-qwen35b-a3b-fusion-nothink-smoke \
        serve-both serve-dual-gpu serve-consultant serve-gemma4 \
        serve-qwen27b serve-qwen35b-a3b \
        serve-socratteachllm serve-teacher-online \
        setup-l40s start-local-tl-server

# Default target
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  run                   Show how to launch the project via poetry"
	@echo "  install-hooks         Install git hooks from hooks/ into .git/hooks/"
	@echo ""
	@echo "  Scripts (scripts/):"
	@echo "  post-eval-shutdown    Run scripts/post_eval_shutdown.sh"
	@echo "  run-eval              Run scripts/run_eval.sh  (GPU=<config>, default: baseline)
                          Dual-GPU configs: GPU=l40s  GPU=3090ti
                          Other configs:    GPU=baseline  GPU=gemma4
                          Tested hardware:  RTX 5090, RTX 3090 Ti, AMD R9700, NVIDIA L40S, V100 32GB"
	@echo "  setup-l40s            Run scripts/l40s_setup.sh (one-time setup for dual L40S machine)"
	@echo "  serve-both            Run scripts/serve_both.sh (single GPU, shared VRAM)"
	@echo "  serve-dual-gpu        Run scripts/serve_dual_gpu.sh (2 GPUs, teacher→GPU0 consultant→GPU1)"
	@echo "  serve-consultant      Run scripts/serve_consultant.sh"
	@echo "  serve-gemma4          Run scripts/serve_gemma4.sh"
	@echo "  serve-qwen27b         Run scripts/serve_qwen27b_q5.sh (Qwen3.6-27B Q5, dual-role teacher+consultant)"
	@echo "  serve-qwen35b-a3b     Run scripts/serve_qwen35b_a3b.sh (Qwen3.6-35B-A3B MoE, ~3x faster than 27B)"
	@echo "  serve-socratteachllm  Run scripts/serve_socratteachllm.sh"
	@echo "  serve-teacher-online  Run scripts/serve_teacher_online.sh"
	@echo "  start-local-tl-server  Start local llama.cpp server for dataset translation (Qwen3.5-9B)"
	@echo "  eval-qwen27b-smoke    Run scripts/eval_qwen27b.sh smoke (n=5,   ~5 min)"
	@echo "  eval-qwen27b-mini     Run scripts/eval_qwen27b.sh mini  (n=25,  ~15 min)"
	@echo "  eval-qwen27b-full     Run scripts/eval_qwen27b.sh full  (n=681, ~75 h — measured)"
	@echo "  eval-qwen35b-a3b-smoke Run scripts/eval_qwen35b_a3b.sh smoke (n=5,   ~2 min projected)"
	@echo "  eval-qwen35b-a3b-mini  Run scripts/eval_qwen35b_a3b.sh mini  (n=25,  ~5 min projected)"
	@echo "  eval-qwen35b-a3b-full  Run scripts/eval_qwen35b_a3b.sh full  (n=681, ~20-30 h projected)"
	@echo ""
	@echo "  Fusion smoke targets (single-call architecture, see SOCRATIC_FUSION_PLAN.md):"
	@echo "  eval-qwen27b-fusion-smoke           27B + unified (think on)"
	@echo "  eval-qwen27b-fusion-nothink-smoke   27B + unified + no-think"
	@echo "  eval-qwen35b-a3b-fusion-smoke         A3B + unified (think on)"
	@echo "  eval-qwen35b-a3b-fusion-nothink-smoke A3B + unified + no-think"
	@echo ""
	@echo "  WAVE HPC (SLURM):"
	@echo "  slurm                 git pull + sbatch wave_eval.slurm + print status"


# ── Setup ─────────────────────────────────────────────────────────────────────

setup: setup-repo install-hooks
	@echo "This project requires Poetry for dependency management. Install Poetry from https://python-poetry.org/ and ensure it's on your PATH."
	@echo "Setting up the project via poetry:"
	poetry install
	@echo ""
	@echo "Then run 'make install-hooks' to set up git hooks for code quality checks on commit."

setup-repo:
	@echo "Configuring dual-push remotes..."
	# Set the primary fetch/push URL
	git remote set-url origin git@github.com:ulises-c/csen-346.git
	# Add the secondary push URL (ignore error if it already exists)
	git remote set-url --add --push origin git@github.com:ulises-c/csen-346.git 2>/dev/null || true
	git remote set-url --add --push origin git@github.com:SCU-CSEN346/KELE.git 2>/dev/null || true
	@echo "Ensuring local main branch exists and is tracked..."
	git checkout main || git checkout -b main
	git push -u origin main
	@echo "Repository setup complete. Verify with 'git remote -v'."

# ── Dual remote synchronization ────────────────────────────────────────────────────

sync-mirror:
	@echo "Syncing remote main to local main (Background Sync)..."
	# Fetches from primary and updates local main without a checkout
	git fetch origin main:main
	@echo "Mirroring local main to all push remotes..."
	# Pushes local main to both primary and SCU org
	git push origin main:main --tags
	@echo "Mirror sync successful."

# ── Entry point ──────────────────────────────────────────────────────────────

run:
	@echo "Run the project via poetry:"
	@echo ""
	@echo "  poetry run kele            # main KELE entry point"
	@echo "  poetry run kele-eval       # run evaluation"
	@echo "  poetry run serve-teacher   # start teacher server"
	@echo ""
	@echo "  poetry run test            # run tests (or: make test)"
	@echo "  poetry run lint            # lint source  (or: make lint)"
	@echo ""
	@echo "  make pre-commit            # run format + lint + tests (mirrors git pre-commit hook)"

# ── Code quality ─────────────────────────────────────────────────────────────

pre-commit:
	poetry run ruff format .
	poetry run ruff check --fix .
	poetry run pytest -rs

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

serve-qwen27b:
	bash scripts/serve_qwen27b_q5.sh

serve-socratteachllm:
	bash scripts/serve_socratteachllm.sh

serve-teacher-online:
	bash scripts/serve_teacher_online.sh

start-local-tl-server:
	bash scripts/start_tl_server.sh

eval-qwen27b-smoke:
	bash scripts/eval_qwen27b.sh smoke

eval-qwen27b-mini:
	bash scripts/eval_qwen27b.sh mini

eval-qwen27b-full:
	bash scripts/eval_qwen27b.sh full

serve-qwen35b-a3b:
	bash scripts/serve_qwen35b_a3b.sh

eval-qwen35b-a3b-smoke:
	bash scripts/eval_qwen35b_a3b.sh smoke

eval-qwen35b-a3b-mini:
	bash scripts/eval_qwen35b_a3b.sh mini

eval-qwen35b-a3b-full:
	bash scripts/eval_qwen35b_a3b.sh full

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

# ── WAVE HPC ──────────────────────────────────────────────────────────────────

slurm:
	bash scripts/slurm/submit_wave.sh
