# Translation Validation — SocratDataset-EN

**CSEN 346 · Santa Clara University**

Goal: validate that `ulises-c/SocratDataset-EN` faithfully translates `ulises-c/SocratDataset` (ZH) across all 6,803 records, using structural checks first and LLM-judged quality scoring on a sample.

Script: [`src/project/validate_translation.py`](../src/project/validate_translation.py).

---

## Architecture

Two-phase design across three machines:

| Machine | Role | What it does |
|---|---|---|
| **Host** (Linux workstation) | Coordinator | Phase 1 structural gate · pushes HF dataset fixes · merges shard results |
| **Mac Mini 1** (M4, 16 GB) | Phase 2 worker — **odd** shard | Runs `mlx_lm.server` (9B Qwen3.5) + evaluates odd-id records |
| **Mac Mini 2** (M4, 16 GB) | Phase 2 worker — **even** shard | Same as Mac Mini 1 for even-id records |

Structural checks (no LLM, runs in seconds) gate the LLM quality eval. The R9700 host GPU stays free for SFT training while validation runs on the Minis.

---

## Phase 1 — Structural checks (no LLM)

Runs on the host. Any failure is a systematic bug worth fixing before spending GPU time on Phase 2.

| Check | Method | Pass criterion |
|---|---|---|
| **ID coverage** | Set diff of ZH ids vs EN ids | Every ZH id has exactly one matching EN id |
| **Field completeness** | Assert every LLM-translated field is non-empty | 0 empty fields across all 6,803 records |
| **State label preservation** | Assert `dialogue[].state` unchanged between ZH and EN | 100% match |
| **Option count preservation** | Assert `len(options_zh) == len(options_en)` per record | 100% match |
| **Dialogue round count** | Assert `len(dialogue)` unchanged between ZH and EN | 100% match |
| **No Chinese in EN fields** | Regex scan all translated fields + `dialogue[].action` for `[一-鿿]` | 0 hits |

Output: pass/fail table to stdout. On failure, `data/validate_structural_failures.json` lists offending record IDs — for `id_coverage`, split into `{"missing_in_en": [...], "extra_in_en": [...]}`.

---

## Phase 2 — LLM quality eval

Runs on the Mac Minis (one shard each, in parallel). Only proceeds if Phase 1 passes.

### Model and hardware options

| Config | Model | Tok/s | 5% sample (~340 rec) | Full 6,803 | Notes |
|---|---|---|---|---|---|
| **27B single node** | `mlx-community/Qwen3.6-27B-4bit` | 5.9 | ~78 min | ~26 h | Highest eval quality; matches translation model family |
| **9B single node** | `mlx-community/Qwen3.5-9B-4bit` | 19.5 | ~24 min | ~8 h | Same model family as translator; good fit |
| **9B two nodes (parallel)** ⭐ | `mlx-community/Qwen3.5-9B-4bit` × 2 | ~39 | ~12 min | ~4 h | Odds → Mac Mini 1, evens → Mac Mini 2; results merged on host |

**Recommendation:** 9B two-node parallel for the 5% sample (12 min, frees the Minis quickly). 27B single-node for a full-dataset run if quality signal matters more than wall clock.

#### Trade-offs

| Config | Pros | Cons |
|---|---|---|
| 27B single | Strongest reasoning; more likely to catch subtle meaning drift and tone errors | 5.9 tok/s — full run is 26 h; ties up one machine |
| 9B single | Same model family as Qwen3.5-9B translator — eval and translator share the same stylistic prior, reducing false flags from style divergence | Half the parameter count; may miss nuanced Socratic tone issues |
| 9B two-node | Fastest wall clock; ID-parity split is dead simple; each node is independent — one crash doesn't lose the other half | Requires merging two result JSONs; 9B quality ceiling applies to both |

### Parallelization design (9B two-node)

Split by record ID parity: Mac Mini 1 processes odd IDs, Mac Mini 2 processes even IDs. Each Mini runs the same script with a `--shard` flag against its own local MLX server. Each shard writes `data/validate_llm_scores_{odd,even}.json` and `data/validate_llm_flagged_{odd,even}.json`. The host pulls both, runs `--merge`, gets the combined summary.

Stratification is applied **within each shard** independently so grade×mission coverage stays balanced even if one shard is run alone.

### Per-record eval call

Each call sends the ZH record and its matched EN record. The model returns:

```json
{
  "overall_score": 1-5,
  "meaning_preserved": true|false,
  "socratic_tone_preserved": true|false,
  "fluency": 1-5,
  "flags": ["any issues noted"]
}
```

The script validates types (rejects e.g. string scores or non-bool flags) and retries up to 3× with exponential backoff. Fields scored: `question`, `options`, `newHint`, `newKnowledgePoint`, `newAnalyze`, `dialogue[].student`, `dialogue[].teacher`, `dialogue[].evaluation`.

### Thresholds

| Metric | Flag threshold |
|---|---|
| `overall_score` | < 3 → flag record |
| `meaning_preserved` | false → flag record |
| `socratic_tone_preserved` | false → flag record |
| Flag rate across sample | > 10% → surface for review |

### Output (per shard, on the Mac Mini that ran it)

- `data/validate_llm_scores_{shard}.json` — per-record scores
- `data/validate_llm_flagged_{shard}.json` — flagged record IDs with reasons
- `data/validate_llm_checkpoint_{shard}.json` — incremental checkpoint (every 10 records); enables resume after interruption

After merge on the host:

- `data/validate_llm_scores_all.json` — combined, sorted by id
- `data/validate_llm_flagged_all.json` — combined flagged list
- Summary table to stdout: score distribution, flag rate

---

## One-time setup

### Host (Linux workstation)

```bash
git clone https://github.com/ulises-c/csen-346.git
cd csen-346
uv sync
```

Authenticate to HuggingFace **only if** you anticipate having to push a fix to the EN dataset (Phase 1 failures usually mean a translator bug — pushing the corrected dataset back is how you unblock Phase 2):

```bash
uv run hf auth login   # needs a write token for ulises-c/SocratDataset-EN
```

### Mac Mini 1 and Mac Mini 2 (do once per machine)

These steps are identical on both Minis except for the `VALIDATE_SHARD` value.

**1. Install Homebrew** (skip if `brew --version` works):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Then follow the printed PATH instructions (eval "$(/opt/homebrew/bin/brew shellenv)").
```

**2. Install dependencies:**

```bash
brew install git uv mlx-lm
```

**3. Clone the repo:**

```bash
git clone https://github.com/ulises-c/csen-346.git
cd csen-346
uv sync
```

**4. Download the model** (skip if `~/.models/mlx-community--Qwen3.5-9B-4bit` already exists):

```bash
uv run hf download \
    mlx-community/Qwen3.5-9B-4bit \
    --local-dir ~/.models/mlx-community--Qwen3.5-9B-4bit
```

Repo is public — no HF login needed. Set `HF_TOKEN` if you hit rate limits.

**5. Set this Mini's shard assignment** (one-shot, persists across reboots):

```bash
# Mac Mini 1:
echo 'export VALIDATE_SHARD=odd' >> ~/.zshrc && source ~/.zshrc

# Mac Mini 2:
echo 'export VALIDATE_SHARD=even' >> ~/.zshrc && source ~/.zshrc
```

---

## Running the validation

### Step 1 — Host: Phase 1 gate

```bash
cd ~/Github/csen-346
uv run python -m src.project.validate_translation --structural-only
```

Takes seconds. If everything passes, proceed to Step 2.

**If a check fails:** the failing record IDs are in `data/validate_structural_failures.json`. Common failure → fix flow:

| Failure | Fix |
|---|---|
| `option_count` mismatch (single record) | Patch the EN record's `options` array in place and push the dataset back to HF (the symbol options like `①②③` need no translation — just restore the missing entry). Re-run Phase 1. |
| `id_coverage` missing | Re-translate the missing ids: `uv run python -m src.project.translate_dataset --ids <id1>,<id2>` against a running LLM server, then push. |
| `state_preservation` | Translator regressed — re-translate the affected ids; state should be preserved verbatim. |
| `no_chinese_in_en` | Re-translate the affected ids with stronger reminders (the script in `translate_dataset.py` has retry reminders for this case). |
| `field_completeness` | Re-translate the affected ids. |

Push pattern (single-record patch):

```python
from datasets import Dataset, load_dataset
en = load_dataset("ulises-c/SocratDataset-EN", split="train").to_list()
# ... edit the record by id ...
Dataset.from_list(en).push_to_hub(
    "ulises-c/SocratDataset-EN",
    commit_message="fix(record N): <what>",
)
```

### Step 2 — Mac Mini 1 and Mac Mini 2: Phase 2 shard

Do this on **both** Minis in parallel. Two terminals per Mini.

**Terminal A — start the MLX server:**

```bash
cd ~/csen-346
set -a && source configs/consultants/m4-mlx.env && set +a
./scripts/serve_consultant_mlx.sh
```

The script starts `mlx_lm.server` on port 8080, warms up the model, and inhibits sleep with `caffeinate`. If it's already running it just confirms connectivity. Verify with:

```bash
curl http://localhost:8080/v1/models
```

**Terminal B — run this Mini's shard:**

```bash
cd ~/csen-346
uv run python -m src.project.validate_translation \
    --shard "$VALIDATE_SHARD" \
    --base-url http://localhost:8080/v1
```

Progress logs print every record with ETA. The script checkpoints to `data/validate_llm_checkpoint_$VALIDATE_SHARD.json` every 10 records — interrupt anytime; re-run picks up where it left off.

Expected wall clock for the default 5% sample: ~12 minutes per Mini.

### Step 3 — Host: collect shard results + merge

After **both** Minis report completion:

```bash
# Pull results from each Mini onto the host
scp mac-mini-1.local:~/csen-346/data/validate_llm_scores_odd.json   data/
scp mac-mini-2.local:~/csen-346/data/validate_llm_scores_even.json  data/

# Merge and print summary
uv run python -m src.project.validate_translation \
    --merge data/validate_llm_scores_odd.json data/validate_llm_scores_even.json
```

Outputs `data/validate_llm_scores_all.json`, `data/validate_llm_flagged_all.json`, and a summary table.

### Step 4 — Interpret

- **Flag rate < 10%:** dataset is validated. Update status in `TRANSLATION_PLAN.md` and proceed to use the EN dataset for downstream training/eval.
- **Flag rate ≥ 10%:** inspect `validate_llm_flagged_all.json`. Look for clustering by grade or mission type — re-translation of a stratum is usually cheaper than chasing individual records.

---

## Configuration (script defaults)

Tunable via CLI flags or by editing constants at the top of [`validate_translation.py`](../src/project/validate_translation.py):

```python
BASE_URL: str = "http://localhost:8080/v1"
MODEL: str = "~/.models/mlx-community--Qwen3.5-9B-4bit"   # local model path
SAMPLE_SIZE: float = 0.05         # 1.0 = full run
SAMPLE_SEED: int = 42
THINKING_BUDGET: int = 0          # 0 = thinking off (translation eval doesn't need CoT)
ZH_HF_REPO: str = "ulises-c/SocratDataset"
EN_HF_REPO: str = "ulises-c/SocratDataset-EN"
OUTPUT_DIR: str = "data"
CHECKPOINT_EVERY: int = 10
SCORE_FLAG_THRESHOLD: int = 3
FLAG_RATE_THRESHOLD: float = 0.10
```

Thinking-disable pattern (matches `translate_dataset.py`): every API call sends `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. Qwen3's chat template ignores `thinking_budget` — `enable_thinking` is the real toggle. For llama.cpp targets, swap to `{"thinking_budget": N}` if `THINKING_BUDGET > 0`.

---

## CLI reference

```bash
# Phase 1 only (host):
uv run python -m src.project.validate_translation --structural-only

# Phase 1 + Phase 2, default 5% sample, single node:
uv run python -m src.project.validate_translation --base-url http://<node>:8080/v1

# Two-node parallel (run on each Mini):
uv run python -m src.project.validate_translation --shard odd  --base-url http://localhost:8080/v1
uv run python -m src.project.validate_translation --shard even --base-url http://localhost:8080/v1

# Merge shard results (host, after both Minis finish):
uv run python -m src.project.validate_translation --merge \
    data/validate_llm_scores_odd.json data/validate_llm_scores_even.json

# Full-dataset LLM eval (override default 5% sample):
uv run python -m src.project.validate_translation --sample 1.0 --base-url http://<node>:8080/v1
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl http://localhost:8080/v1/models` → connection refused | mlx_lm.server didn't start; check `logs/mlx_consultant.log` | Re-run `./scripts/serve_consultant_mlx.sh`; verify model path exists at `~/.models/mlx-community--Qwen3.5-9B-4bit` |
| Phase 2 client returns 404 on `chat/completions` | Model name in `--model` doesn't match what the server registered (it registers by the path it was loaded with) | Pass `--model "$(printf %s ~/.models/mlx-community--Qwen3.5-9B-4bit)"` or override `MODEL` in the script |
| HF anonymous load 429s during Phase 1 | Rate limited | `export HF_TOKEN=...` on the host |
| Mini sleeps mid-run | `caffeinate` didn't attach | Re-run `serve_consultant_mlx.sh`; it re-attaches `caffeinate` if the server is already up |
| Resume produces wildly low ETA | Pre-existing checkpoint mixing with new start (cosmetic only — script tracks `initial_done`) | Ignore for the first record after resume; subsequent ETA is accurate |

---

## Sequencing summary

1. **Host:** `uv sync`. Mac Minis: one-time bootstrap (Homebrew → deps → repo → model → `VALIDATE_SHARD`).
2. **Host:** run Phase 1 gate. Iterate fixes + HF pushes until it passes.
3. **Mac Mini 1 + Mac Mini 2:** start MLX server (Terminal A), run shard (Terminal B). ~12 min for the default 5% sample.
4. **Host:** scp both shard JSONs, run `--merge`, inspect summary.
5. If flag rate < 10%, the EN dataset is validated.
