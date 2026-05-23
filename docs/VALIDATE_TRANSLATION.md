# Translation Validation Plan — SocratDataset-EN

**CSEN 346 · Santa Clara University**

Goal: validate that `ulises-c/SocratDataset-EN` faithfully translates `ulises-c/SocratDataset` (ZH) across all 6,803 records, using structural checks first and LLM-judged quality scoring on a sample.

---

## Script: `src/project/validate_translation.py`

Two-phase design: structural checks (no LLM, runs in seconds) gate the LLM quality eval (sample-based, exo cluster).

---

## Phase 1 — Structural checks (no LLM)

Run first. Any failure here is a systematic problem worth fixing before spending GPU time.

| Check | Method | Pass criterion |
|---|---|---|
| **ID coverage** | Set diff of ZH ids vs EN ids | Every ZH id has exactly one matching EN id |
| **Field completeness** | Assert every LLM-translated field is non-empty | 0 empty fields across all 6,803 records |
| **State label preservation** | Assert `dialogue[].state` unchanged between ZH and EN | 100% match |
| **Option count preservation** | Assert `len(options_zh) == len(options_en)` per record | 100% match |
| **Dialogue round count** | Assert `dialogueRound` and `len(dialogue)` unchanged | 100% match |
| **No Chinese characters in EN fields** | Regex scan all translated fields for `[一-鿿]` | 0 hits |

Output: a pass/fail table to stdout + a list of failing record IDs (if any) saved to `data/validate_structural_failures.json`.

---

## Phase 2 — LLM quality eval

Only runs if Phase 1 passes.

### Model and hardware options

| Config | Model | Tok/s | 5% sample (~340 rec) | Full 6,803 | Notes |
|---|---|---|---|---|---|
| **27B single node** | `mlx-community/Qwen3.6-27B-4bit` | 5.9 | ~78 min | ~26 h | Highest eval quality; matches translation model family |
| **9B single node** | `mlx-community/Qwen3.5-9B-4bit` | 19.5 | ~24 min | ~8 h | Same model family as original translator; good fit |
| **9B two nodes (parallel)** | `mlx-community/Qwen3.5-9B-4bit` × 2 | ~39 | ~12 min | ~4 h | Odds to node 1, evens to node 2; results merged |

**Recommendation:** 9B two-node parallel for the 5% sample (12 min, frees the machines quickly). 27B single-node for a full-dataset run if quality signal matters more than wall clock.

#### Pros and cons

| Config | Pros | Cons |
|---|---|---|
| 27B single | Strongest reasoning; more likely to catch subtle meaning drift and tone errors | 5.9 tok/s — slowest; full run is 26 h; ties up one machine |
| 9B single | Same model family as original Qwen3.5-9B translator — eval and translator share the same stylistic prior, reducing false flags from style divergence | Half the parameter count; may miss nuanced Socratic tone issues |
| 9B two-node | Fastest wall clock; simple to implement (ID parity split); each node is independent — one crash doesn't lose the other half | Requires merging two result JSONs; slight coordination overhead; both machines occupied; 9B quality ceiling applies to both |

#### Parallelization design (9B two-node)

Split by record ID parity: node 1 processes odd IDs, node 2 processes even IDs. Each node runs the same script with a `--shard` flag:

```bash
# Node 1 (odd IDs):
uv run python -m project.validate_translation --shard odd --base-url http://<node1>:8080/v1

# Node 2 (even IDs):
uv run python -m project.validate_translation --shard even --base-url http://<node2>:8080/v1
```

After both finish, merge results:

```bash
uv run python -m project.validate_translation --merge data/validate_llm_scores_odd.json data/validate_llm_scores_even.json
```

Each shard writes its own `validate_llm_scores_{odd,even}.json` and `validate_llm_flagged_{odd,even}.json`. The merge step concatenates and re-computes the summary table.

### Sample design

| Mode | Records | Recommended config |
|---|---|---|
| Default (5% stratified) | ~340 | 9B two-node (~12 min) |
| Full dataset | 6,803 | 9B two-node (~4 h) or 27B single (~26 h) |

Stratification: sample proportionally across `mission` type (multiple_choice / true_false) and `grade` level so all 12 grade×volume combinations are represented. When sharding, stratification is applied within each shard independently to avoid grade imbalance.

### Per-record eval call

Each call sends the ZH record and its matched EN record. The model returns a JSON object:

```json
{
  "overall_score": 1-5,
  "meaning_preserved": true/false,
  "socratic_tone_preserved": true/false,
  "fluency": 1-5,
  "flags": ["any issues noted"]
}
```

Fields scored:
- `question`, `options`, `newHint`, `newKnowledgePoint`, `newAnalyze`
- `dialogue[].student`, `dialogue[].teacher`, `dialogue[].evaluation`

### Thresholds

| Metric | Flag threshold |
|---|---|
| `overall_score` | < 3 → flag record |
| `meaning_preserved` | false → flag record |
| `socratic_tone_preserved` | false → flag record |
| Flag rate across sample | > 10% → surface for review |

### Output

- `data/validate_llm_scores.json` — per-record scores
- `data/validate_llm_flagged.json` — flagged record IDs with reasons
- Summary table to stdout: score distribution, flag rate, per-grade breakdown

---

## Configuration

```python
# LLM endpoint — point at whichever node is running the eval model
BASE_URL: str = "http://<node>:8080/v1"
MODEL: str = "~/.models/mlx-community--Qwen3.5-9B-4bit"   # local path; same on both Mac Minis

SAMPLE_SIZE: float = 0.05         # fraction of dataset; set 1.0 for full run
SAMPLE_SEED: int = 42
THINKING_BUDGET: int = 0          # 0 = off; translation eval doesn't need CoT

SHARD: str = "all"                # "all" | "odd" | "even" — for two-node parallel runs

ZH_HF_REPO: str = "ulises-c/SocratDataset"
EN_HF_REPO: str = "ulises-c/SocratDataset-EN"

OUTPUT_DIR: str = "data/"
```

---

## Mac Mini setup

### Phase 1 (structural — no LLM)

Only needs Python + `datasets`. No model download, no server.

```bash
uv sync
uv run python -m project.validate_translation --structural-only
```

> Both HF repos are public — no login required. Anonymous downloads are rate-limited; set `HF_TOKEN` if you hit 429s.

### Phase 2 (LLM quality eval — mlx-lm)

Install `mlx-lm` (one-time; model is already at `~/.models/`):

```bash
brew install mlx-lm
```

Each Mac Mini knows its own shard via `VALIDATE_SHARD` — set this once in `~/.zshrc` (or `~/.bash_profile`):

```bash
# Mac Mini 1:
echo 'export VALIDATE_SHARD=odd'  >> ~/.zshrc

# Mac Mini 2:
echo 'export VALIDATE_SHARD=even' >> ~/.zshrc
```

When it's time to run (each Mac Mini does this independently):

```bash
# Terminal 1 — start the LLM server
set -a && source configs/consultants/m4-mlx.env && set +a
./scripts/serve_consultant_mlx.sh

# Terminal 2 — run this machine's shard
python -m project.validate_translation --shard "$VALIDATE_SHARD" --base-url http://localhost:8080/v1
```

---

## CLI

```bash
# Phase 1 only (structural, no LLM):
uv run python -m project.validate_translation --structural-only

# Phase 1 + Phase 2 on default 5% sample, single node:
uv run python -m project.validate_translation --base-url http://<node>:8080/v1

# Two-node parallel (run on each machine simultaneously):
uv run python -m project.validate_translation --shard odd  --base-url http://<node1>:8080/v1
uv run python -m project.validate_translation --shard even --base-url http://<node2>:8080/v1

# Merge shard results after both finish:
uv run python -m project.validate_translation --merge data/validate_llm_scores_odd.json data/validate_llm_scores_even.json

# Full dataset LLM eval (background run):
uv run python -m project.validate_translation --sample 1.0 --base-url http://<node>:8080/v1
```

---

## Hardware routing

| Phase | Runs on | Why |
|---|---|---|
| Phase 1 (structural) | Mac (local) | No GPU needed; pure Python on loaded JSON |
| Phase 2 (LLM eval, 5% sample) | 9B two-node parallel (2× M4 Mini) | 12 min; frees both machines quickly; R9700 stays free for SFT training |
| Phase 2 (LLM eval, full dataset) | 9B two-node (~4 h) or 27B single (~26 h) | Choose based on quality vs time budget |

The R9700 should stay free for SFT training (feat/multi-dataset-training).

---

## Sequencing

1. Pull both datasets from HF (or use local copies if available)
2. Run Phase 1 — fix any structural failures before proceeding
3. Run Phase 2 on 5% sample as a background task on exo
4. If flag rate < 10%: dataset is validated; update TRANSLATION_PLAN.md status
5. If flag rate ≥ 10%: inspect flagged records; determine if re-translation is needed for specific grade levels or field types
