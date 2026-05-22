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

## Phase 2 — LLM quality eval (exo cluster)

Only runs if Phase 1 passes. Scored by Qwen3-27B-4bit via the exo cluster's OpenAI-compatible endpoint.

### Sample design

| Mode | Records | Estimated time at 5.8 tok/s |
|---|---|---|
| Default (5% stratified) | ~340 | ~78 min |
| Full dataset | 6,803 | ~26 h |

Stratification: sample proportionally across `mission` type (multiple_choice / true_false) and `grade` level so all 12 grade×volume combinations are represented.

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
# Exo cluster endpoint (Qwen3-27B-4bit MLX, 2× M4 Mac Mini via TB3)
BASE_URL: str = "http://<exo-host>:8080/v1"
MODEL: str = "qwen3-27b"          # exo model name — confirm with `exo ps`

SAMPLE_SIZE: float = 0.05         # fraction of dataset; set 1.0 for full run
SAMPLE_SEED: int = 42
THINKING_BUDGET: int = 0          # 0 = off; translation eval doesn't need CoT

ZH_HF_REPO: str = "ulises-c/SocratDataset"
EN_HF_REPO: str = "ulises-c/SocratDataset-EN"

OUTPUT_DIR: str = "data/"
```

---

## CLI

```bash
# Phase 1 only (structural, no LLM):
uv run python -m project.validate_translation --structural-only

# Phase 1 + Phase 2 on default 5% sample (exo cluster):
uv run python -m project.validate_translation --base-url http://<exo-host>:8080/v1

# Full dataset LLM eval (background run, ~26 h):
uv run python -m project.validate_translation --sample 1.0 --base-url http://<exo-host>:8080/v1

# Override sample size:
uv run python -m project.validate_translation --sample 0.10
```

---

## Hardware routing

| Phase | Runs on | Why |
|---|---|---|
| Phase 1 (structural) | Mac (local) | No GPU needed; pure Python on loaded JSON |
| Phase 2 (LLM eval) | Exo cluster (2× M4 Mini 16 GB, TB3) | R9700 free for training; 5.8 tok/s sufficient for ~340-record sample |

The R9700 should stay free for SFT training (feat/multi-dataset-training). The exo cluster is the right target for the background LLM eval.

---

## Sequencing

1. Pull both datasets from HF (or use local copies if available)
2. Run Phase 1 — fix any structural failures before proceeding
3. Run Phase 2 on 5% sample as a background task on exo
4. If flag rate < 10%: dataset is validated; update TRANSLATION_PLAN.md status
5. If flag rate ≥ 10%: inspect flagged records; determine if re-translation is needed for specific grade levels or field types
