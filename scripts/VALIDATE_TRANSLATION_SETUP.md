# Mac Mini Setup — Translation Validation

Setup guide for running `src/project/validate_translation.py` on an M4 Mac Mini.
Two Mac Minis run in parallel — each serves its own MLX model and processes its own shard.

---

## 1. Install Homebrew

Skip if already installed (`brew --version` succeeds).

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After install, follow the printed instructions to add Homebrew to your PATH (the `eval "$(/opt/homebrew/bin/brew shellenv)"` line).

---

## 2. Install dependencies

```bash
brew install git uv mlx-lm
```

Verify:

```bash
uv --version
mlx_lm.server --help
```

---

## 3. Clone the repo

```bash
git clone https://github.com/ulises-c/csen-346.git
cd csen-346
uv sync
```

---

## 4. Download the model

Skip if `~/.models/mlx-community--Qwen3.5-9B-4bit` already exists.

```bash
uv run huggingface-cli download \
    mlx-community/Qwen3.5-9B-4bit \
    --local-dir ~/.models/mlx-community--Qwen3.5-9B-4bit
```

No HF login required — the repo is public. Set `HF_TOKEN` if you hit rate limits.

---

## 5. Set your shard

Each Mac Mini handles half the dataset. Assign once and it persists across reboots.

**Mac Mini 1:**

```bash
echo 'export VALIDATE_SHARD=odd' >> ~/.zshrc && source ~/.zshrc
```

**Mac Mini 2:**

```bash
echo 'export VALIDATE_SHARD=even' >> ~/.zshrc && source ~/.zshrc
```

---

## 6. Start the MLX server (each session)

Run this on the Mac Mini before starting the validation:

```bash
cd ~/path/to/csen-346
set -a && source configs/consultants/m4-mlx.env && set +a
./scripts/serve_consultant_mlx.sh
```

The script starts `mlx_lm.server` on port 8080, warms up the model, and inhibits sleep with `caffeinate`. If the server is already running it just confirms connectivity.

Verify it's up:

```bash
curl http://localhost:8080/v1/models
```

---

## 7. Run the validation

**Phase 1 — structural checks (no LLM, runs in seconds):**

```bash
uv run python -m src.project.validate_translation --structural-only
```

**Phase 2 — LLM quality eval (requires server from step 6):**

```bash
uv run python -m src.project.validate_translation \
    --shard "$VALIDATE_SHARD" \
    --base-url http://localhost:8080/v1
```

Run this on both Mac Minis simultaneously. Each machine processes its assigned shard (~170 records, ~12 min for the default 5% sample).

**Merge results** (run on either machine after both finish):

```bash
uv run python -m src.project.validate_translation \
    --merge data/validate_llm_scores_odd.json data/validate_llm_scores_even.json
```

---

## Notes

- `validate_translation.py` is not written yet — steps 1–6 prepare the machine. Step 7 runs once the script exists.
- Thinking is disabled for all LLM calls (`enable_thinking: False` via `extra_body`). See `docs/VALIDATE_TRANSLATION.md` → Configuration for the pattern.
- The 27B model (`~/.models/mlx-community--Qwen3.6-27B-4bit`) is also available on both machines for a higher-quality single-node run at the cost of ~3× wall clock time.
