# Level-up experiments log (branch `mk/level-up-experiments`)

Started 2026-05-15 09:53 PDT, off `mk/8h-autonomous-extensions` (PR #52).

## Scope

Running the 8-hour bundle proposed in the post-night-run review:
1. (2h) Many-shot prompt-eng sweep — 5-shot, 7-shot, 10-shot at n=50
2. (2h) Stage-aware exemplars at smoke + mini + n=50
3. (1h) LoRA fine-tune on A3B teacher (training only)
4. (1.5h) Chinese-BERT classifier training + smoke eval
5. (1.5h) Visualization sprint

## Plumbing changes (committed before evals)

- **Variable-N few-shot patch** in `socratic_teaching_unified.py`
  - 10-exemplar pool, 2 per stage (a/b/c/d/e), drawn from train dialogues 1–12 (no test overlap)
  - `KELE_FEW_SHOT_N=<int>` env var selects N stage-balanced exemplars
  - Legacy 3-shot preserved when `KELE_FEW_SHOT_N` unset
- **5 paper figures** generated from existing results (no new runs needed):
  - `stage_confusion_a3b_full.{pdf,png}`, `stage_confusion_baseline.{pdf,png}`
  - `per_stage_accuracy.{pdf,png}` (5 systems × 5 stages)
  - `turn_index_accuracy_a3b_full.{pdf,png}`
  - `system_radar.{pdf,png}` (4 systems × 5 metrics)
- **4 additional figures** from dataset/results:
  - `state_frequency.{pdf,png}` (34 states sorted by count)
  - `dialogue_length_hist.{pdf,png}` (μ=6.30 turns, range 5–12)
  - `stage_turn_count.{pdf,png}` (c is 2.1× any other stage)
  - `turn_count_a3b_full.{pdf,png}`
- **Schema-fallback analysis** (`docs/figures/schema_fallback_analysis.md`):
  - All 12 evaluated runs cross-checked
  - **Qwopus 35B-A3B mini at 10.3% fallback rate** — over the 5% gate; only run that hits it
  - A3B locked full at 0.91% (38/4171) — well under
- **Chinese-BERT classifier training script** (`scripts/train_stage_classifier.py`):
  - Ready to run when GPU frees; targets stage-c via 5-stage classification
  - Base: `BAAI/bge-small-zh-v1.5` (24M params)

## Many-shot sweep results

A3B fusion-think on the 5090, mini scale (n=25 dialogues):

| N | Selection | n_turns | State acc | R-1 | Per-stage (a/b/c/d/e) |
|---:|---|---:|---:|---:|---|
| 0 | locked (no exemplars) | 145 | 35.17% | 30.51 | 88.0 / 32.14 / 10.64 / 13.04 / 54.55 |
| 3 | legacy b/c/d | 148 | 43.24% | 33.49 | 88.0 / **53.57** / **27.66** / 16.0 / 43.48 |
| 5 | stage-balanced (1/stage) | 146 | 38.36% | 34.60 | 96.0 / 32.14 / 14.89 / 16.67 / 54.55 |
| 7 | stage-balanced + b/c extras | 148 | **45.27%** | **35.69** | **100.0** / 53.57 / 19.15 / **24.0** / 52.17 |
| 10 | full pool | _running_ | _pending_ | _pending_ | _pending_ |

### Observations so far (subject to n=50 verification)

1. **Exemplar SELECTION dominates COUNT.** 3-shot focused on b/c/d outperforms 5-shot stage-balanced (43.24 vs 38.36). The c gain (+17 vs +4) is the dominant signal.
2. **7-shot is best at mini.** Combines stage coverage (a/d/e lift) with b/c emphasis. R-1 also highest at 35.69.
3. **5-shot loses to 3-shot.** The two added exemplars (stage a and e) don't help and may slightly hurt b/c by diluting the prompt.
4. **n=50 verification is essential.** The 3-shot mini's +8.07 collapsed to -0.55 at n=50 (per the overnight run); we expect similar regression for 7-shot. The robust signal across the 3-shot data was R-1 lift ~+1.5 pts at neutral state cost. Whether 7-shot's +10-pt mini lift survives at n=50 is open.

_The full results table will be regenerated once 10-shot completes; best variant goes to n=50 verification._
