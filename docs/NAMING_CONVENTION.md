# Naming convention for KELE experiment cells

**Status:** Active 2026-05-23. Applied by `scripts/backtest_stage_balanced.py:display_name()` to every leaderboard the project produces. On-disk dir names (`results/<dir>/`) are unchanged — this is a display-layer transform.

## The format

Every experiment cell is identified by four slots:

```
<consultant> × <teacher> [· <variant>...] · n=<N>
```

| Slot | Required? | What it is | Examples |
|---|---|---|---|
| **`<consultant>`** | yes | The state-classifier (or LLM acting as one) that routes turns into pedagogical stages | `bert`, `bert-fixed`, `qwen3.5`, `Claude-Opus` |
| **`<teacher>`** | yes | The LLM that produces the actual teacher response | `Gemma-31B`, `A3B-35B`, `Qwen-27B`, `Claude-Sonnet`, `SocratTeachLLM` |
| **`<variant>`** | optional (0+) | Modifiers describing the specific configuration; one variant per `·` slot | `think`, `no-think`, `top3`, `composed`, `raw`, `EN`, `fewshot10` |
| **`n=<N>`** | yes (at end) | Dialogue count for this evaluation | `n=50`, `n=100`, `n=200`, `n=400`, `n=681` |

Separators are load-bearing: `×` between consultant and teacher signals their relationship (consultant routes, teacher responds); `·` (middle dot) separates everything else. The format is the same in markdown tables, paper text, and conversation.

## Vocabulary

### Consultants

| Label | Underlying model | When used | Dir prefix |
|---|---|---|---|
| `bert` | BAAI bge-small-zh / state_classifier_v1 (model_type=bert) | Legacy downstream runs **before** 2026-05-22 input-format duplication fix (commit 3d68d4a) | `bert-` |
| `bert-fixed` | Same model, post-fix | Downstream runs **after** the 2026-05-22 fix | `bge-small-bert-…-fixed` |
| `qwen3` | Qwen3-Embedding-0.6B (T1 frozen / T2 LoRA from consultant-upgrade funnel) | Layer-1 classifier training only — no downstream eval cells yet | `state-clf-qwen3-emb-0.6b-` |
| `qwen3.5` | Qwen3.5-0.8B-Base (T3 frozen / T4 LoRA). T4 is the funnel winner; all downstream cells use T4 | Downstream runs using the upgraded classifier | `t4-bert-` (where "T4" = the funnel-tier label, **not** the NVIDIA Tesla T4 GPU — see `docs/CONSULTANT_UPGRADE_LOG.md`) |
| `Claude-Opus` | Claude Opus 4.6 used as an LLM-as-consultant | SocratTeachLLM-overfit experiments only | `claude-opus-consultant-` |
| `Claude-Sonnet` | Claude Sonnet 4.6 used as an LLM-as-consultant | SocratTeachLLM-overfit experiments only | `claude-sonnet-consultant-` |

Anything starting with `state-clf-…` (without a teacher infix) is a classifier-training-only run, not an end-to-end pipeline eval — it doesn't get a teacher slot.

### Teachers

| Label | Underlying model | Local? |
|---|---|---|
| `Gemma-31B` | Gemma 4 31B-it Q5_K_XL | local llama.cpp (Q5 quant) |
| `A3B-35B` | Qwen 35B-A3B (MoE, sparse) | local llama.cpp (Q4_K_M) |
| `Qwen-27B` | Qwen 3.6 27B (dense) Q5_K_XL | local llama.cpp (Q5 quant) |
| `Claude-Opus` | Claude Opus 4.6 | API |
| `Claude-Sonnet` | Claude Sonnet 4.6 | API |
| `SocratTeachLLM` | GLM4-9B fine-tune from the original KELE paper | local |

### Variants

Each appears as its own `· slot` in the format. If the residue between teacher and `n=N` is a single multi-word concept (e.g., `composed-top3`), it stays as one variant; if it's multiple recognized tokens, they split.

| Variant | Meaning |
|---|---|
| `think` | Qwen 27B with reasoning mode ON (implicit; only Qwen-27B has this slot) |
| `no-think` | Qwen 27B with `--reasoning off`; faster but less closure quality |
| `top3` | Top-3 prompt-engineering stack from Phase 2 (length-budget + persona + negative-exemplars) |
| `composed` | Multi-utilization composition (typically Phase 2 cells) |
| `composed-top3` | The full Phase 2 stack |
| `raw` | No prompt engineering at all (frontier-LLM baseline) |
| `clean` | Re-run with workers=1 to eliminate concurrency artifacts |
| `EN` | Evaluated against SocratDataset-EN (English translation) instead of Chinese |
| `ZH` | Explicit Chinese (rarely needed; ZH is the default) |
| `fewshot10` | 10-shot stage-balanced exemplars in teacher prompt |
| `fewshot7` | 7-shot variant |
| `mini` | Smoke/mini sample size (~100-150 turns) |
| `smoke` | Tiny smoke test |
| `full` | n=681 (the full SocratDataset test split). Gets rewritten to `n=681`. |

### `n=N`

The dialogue count. Common values:

| `n=N` | Meaning | Where it comes from |
|---|---|---|
| `n=50` | First-50 by sorted ID (legacy mini tier) or random-50 with seed | `--limit 50` |
| `n=100` | Random sample, seed=42 | `--limit 100 --sample-seed 42` |
| `n=200` | Random sample, seed=42 (canonical screening tier per `CONVERGENCE_ANALYSIS.md`) | `--limit 200 --sample-seed 42` |
| `n=400` | Random sample, seed=42 (canonical paper sample size — ε≤2pp on all 4 metrics) | `--limit 400 --sample-seed 42` |
| `n=681` | Full test split | no `--limit` flag (or `--limit 681`) |

## Reading the format — annotated examples

```
bert × Gemma-31B · fewshot10 · n=681
└──┬─┘ └─────┬──┘   └───┬───┘   └─┬─┘
   │         │          │        └── full test split (681 dialogues)
   │         │          └─────────── 10-shot stage-balanced exemplars in teacher prompt
   │         └────────────────────── Gemma 4 31B Q5 local llama.cpp as teacher
   └──────────────────────────────── BERT classifier as consultant (pre-fix; locked headline)
```

```
qwen3.5 × Qwen-27B · think · fewshot10 · n=50
└──┬──┘   └───┬──┘   └─┬─┘   └───┬───┘   └─┬─┘
   │          │        │         │         └── 50 dialogues
   │          │        │         └──────────── 10-shot exemplars
   │          │        └────────────────────── reasoning mode ON (Qwen 27B think variant)
   │          └─────────────────────────────── Qwen 3.6 27B Q5 local llama.cpp as teacher
   └────────────────────────────────────────── T4 (Qwen3.5-0.8B-LoRA) classifier as consultant
```

```
Claude-Sonnet × SocratTeachLLM · EN · n=50
└──────┬────┘   └──────┬─────┘   └┬┘   └─┬─┘
       │               │          │      └── 50 dialogues
       │               │          └───────── evaluated on SocratDataset-EN translation
       │               └──────────────────── GLM4-9B fine-tune from original KELE paper as teacher
       └──────────────────────────────────── Claude Sonnet 4.6 used as LLM-as-consultant
```

## On-disk dir name ↔ display name mapping

The dir naming is historical and inconsistent. The display layer normalizes it:

| Dir name pattern | Display |
|---|---|
| `bert-X-Y` | `bert × ...` (everything else parsed from rest) |
| `bge-small-bert-X-Y-fixed` | `bert-fixed × ...` |
| `t4-bert-X-Y[-fixed]` | `qwen3.5 × ...` |
| `claude-opus-consultant-X-Y` | `Claude-Opus × ...` |
| `claude-sonnet-consultant-X-Y` | `Claude-Sonnet × ...` |
| `state-clf-qwen3-emb-0.6b-X` | `qwen3-classifier-X` (Layer-1 only — no teacher slot) |
| `state-clf-qwen3.5-0.8b-X` | `qwen3.5-classifier-X` (Layer-1 only — no teacher slot) |

If you need to find a cell on disk, the parser is reversible enough:
- consultant `bert` → `bert-*`
- consultant `bert-fixed` → `bge-small-bert-*-fixed`
- consultant `qwen3.5` → `t4-bert-*`
- variant `think` (only Qwen-27B) is implicit — look for `*-qwen27b-*` without `-nothink-`

## When to use the format

- **Always in tables and leaderboards.** The backtest script writes it automatically.
- **In paper text.** Beats raw dir names — readers don't need to learn a second nomenclature.
- **In commit messages and EXPERIMENT_LOG entries.** Makes diffs and notes scannable.
- **In conversation.** Beats every alternative (`t4-bert-…-fixed`, etc.).

## When NOT to use the format

- **`scripts/`, `configs/`, and `Makefile` references** — use the raw dir name. Bash and Python can't parse `×` and `·` without escaping.
- **Filesystem operations** — the raw dir name IS the filesystem path. No translation.
- **`results/<dir>/run_config.json`** — stores the raw dir name as the run identifier.

The display format is for humans reading reports. The dir name is for machines reading filesystems.

## How to add a new cell type

1. Pick a dir name that follows existing conventions (e.g., `bge-small-bert-newteacher-fewshot10-n50-fixed`).
2. If the consultant prefix is new, add it to `_CONSULTANT_MAP` in `scripts/backtest_stage_balanced.py`.
3. If the teacher token is new, add it to `_TEACHER_MAP` with the display label and any implicit variant (e.g., `think`/`no-think` for reasoning-capable models).
4. If you introduce a new variant (e.g., `dpo-tuned`), add it to `_KNOWN_VARIANTS` so the parser splits it correctly.
5. Run the script and verify the output for a sample dir.

Total effort to extend: ~5 lines + a test case.
