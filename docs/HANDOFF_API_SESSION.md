# Handoff — API-only Claude session (no GPU, no R9700)

**For:** another Claude Code session running on a machine *without* the R9700
training box. Pure API + filesystem work — no GPU inference, no training.
**Branch to start from:** `feat/stage2-sft-pipeline-design` (PR #79) if still
open, else `main` after merge.
**Sibling handoffs:**
- `docs/HANDOFF_STAGE2_TRAINING.md` — the R9700 session (GPU work).
- `.claude/HANDOFF_STAGE2_PIPELINE.md` — historical, PR #79 design brief.

---

## TL;DR — what this session does

The R9700 box is (or will be) running Stage 2b training and synthetic eval —
those are GPU-bound and not available here. This API-only session fills the
parallel slot with **synthetic n=75 generation**, which the training session
needs as the lift-measurement floor and which is pure Sonnet-API work.

If time remains, two text-only fillers are queued: the gitignore cleanup PR
and an audit of the Stage 1 pedagogy filter on a small HF sample.

---

## Primary task — synthetic dataset n=75 expansion

### Why this matters

`TRAINING_PLAN.md` §0.1 "Deferred evidence": the current n=37 baseline has a
Wilson 95% SE of ~3pp on overall state-acc and ~9pp per-stage — too loose to
measure Stage 2b lift (target effect size ~5-10pp) defensibly. n=75 halves
the SE on the per-stage cells; n=100 better still. Required before §5.3
ablation eval lands.

The R9700 session can run the *eval* on n=75 once the data exists, but it
can't cheaply do the *generation* (Sonnet API call). This session does the
generation half.

### What exists today

- `ulises-c/SocratDataset-SYNTHETIC` on HF has **37 records** (ids 100001..100037).
  Verified via `datasets.load_dataset("ulises-c/SocratDataset-SYNTHETIC", split="train")`.
- `scripts/generate_synthetic_socrat.py` is the generator. CLI:
  `--n-dialogues N --output PATH --start-id ID --model claude-sonnet-4-6 --batch-size 5`.
  Cost ~$0.10/dialogue at Sonnet 4.6 (~3K in / ~1K out, batched 5/call).
  38 new dialogues → ~$4 + ~10-15 min wall clock.
- Schema produced matches `references/KELE/SocratDataset.json` — the same
  shape `src/project/dataset.py` loads. So the new records slot in without
  schema work.

### Concrete steps

1. **Verify the current count and max id** (sanity check before spending):
   ```bash
   uv run python -c "
   from datasets import load_dataset
   ds = load_dataset('ulises-c/SocratDataset-SYNTHETIC', split='train')
   ids = [int(dict(r)['id']) for r in ds]
   print(f'count={len(ds)}  min_id={min(ids)}  max_id={max(ids)}')
   "
   ```
   Expected: `count=37  min_id=100001  max_id=100037`. If different, stop and
   surface to the user — the assumption about extension count changes.

2. **Confirm with the user before spending Sonnet API**. Quote the cost (~$4)
   and ask for go/no-go. Per the original handoff pitfall #11: talk to the
   user before any cost > $1.

3. **Generate 38 new dialogues to a local JSON**:
   ```bash
   mkdir -p data/synthetic_extension
   uv run python scripts/generate_synthetic_socrat.py \
       --n-dialogues 38 \
       --start-id 100038 \
       --output data/synthetic_extension/socrat_synthetic_38.json \
       --model claude-sonnet-4-6
   ```
   The `--start-id 100038` is critical — it picks up where the existing set
   ends and avoids ID collisions on merge.

4. **Validate the output** before considering an HF push:
   - Eyeball the first 3 dialogues for plausibility (correct schema, Chinese
     content, realistic state progression a→b→c→d→e).
   - Count: `jq 'length' data/synthetic_extension/socrat_synthetic_38.json`
     should print `38`.
   - Spot-check state distribution: dialogues should cover all 5 stages, not
     concentrate on b/c. If state coverage is poor, the generator may need a
     prompt tweak (out of scope — flag back to the user).

5. **Do NOT upload to HF without explicit user approval.** HF dataset
   re-upload is *destructive* — overwrites the existing 37-record version
   and breaks reproducibility of prior runs (PR #77 / commit `291af4c`'s
   results were measured on the 37-record baseline). Two safer options to
   propose:
   - **Option A:** Upload as a *new* HF dataset (`ulises-c/SocratDataset-SYNTHETIC-n75`)
     and update `dataset_catalog.json` + the synthetic loader to point at it
     for new runs. Existing n=37 results stay reproducible.
   - **Option B:** Append the 38 new records into the existing repo as a
     `train_extension` config or a separate parquet split. More complex; HF
     dataset configs are a moving API.

   **Recommend Option A.** Cleaner reproducibility, single-commit catalog
   update. Surface this to the user with the cost-completed status and let
   them pick before any HF push.

6. **Commit** the local `data/synthetic_extension/socrat_synthetic_38.json`
   to the branch (a tracked artifact for the eval session to pick up if HF
   upload is deferred). Single commit, short message.

### Done criteria for this task

- 38 new dialogues in `data/synthetic_extension/socrat_synthetic_38.json`,
  validated.
- Cost recorded (`$X.YY actual vs $4 estimated`) in a brief PR-style commit
  message.
- User briefed on Option A vs B for HF upload, decision logged in their
  reply or in a follow-up commit.

---

## Secondary task — gitignore cleanup PR (filler)

Per `.claude/HANDOFF_STAGE2_PIPELINE.md` pitfall #8: working tree at session
start has `.coverage`, `.railguard/`, `poetry.lock` as ignored-but-untracked
noise. The previous handoff explicitly flagged these for a separate
one-commit PR. No GPU, no API, ~5 min.

Steps:

1. Branch off main: `git checkout main && git pull && git checkout -b chore/gitignore-cleanup`.
2. Edit `.gitignore` to cover: `.coverage`, `.railguard/`, `poetry.lock`,
   `outputs/`, `data/dpo_pairs/`, `*.parquet` (the DPO builder will produce
   these post-merge).
3. Single commit: `chore: gitignore .coverage, .railguard/, poetry.lock, outputs/`.
4. Push and open a small PR titled `chore: gitignore session-artifact noise`.

Don't combine this with the synthetic-extension work — separate concerns,
separate PR.

---

## Tertiary task — Stage 1 filter sanity check (if time permits)

The Stage 1 pedagogy filter in `src/project/dataset.py` was unit-tested but
not eyeball-audited on real HF data. The handoff §3.2 protocol calls for a
100-sample audit before committing Stage 1 training. This API session can do
the audit *up to* the filter pass — the actual training is GPU-bound.

Steps:

1. Run a small streaming sample with the filter applied:
   ```bash
   TRAIN_STAGE1_OPENHERMES_N=100 uv run python -c "
   from src.project.dataset import load_openhermes
   records = load_openhermes(split='all', streaming=True)
   for r in records[:20]:
       user = next(m for m in r['messages'] if m['role'] == 'user')
       asst = next((m for m in r['messages'] if m['role'] == 'assistant'), None)
       print('---')
       print('USER:', user['content'][:200])
       print('ASST:', (asst['content'][:200] if asst else '(none)'))
   "
   ```
   **Important:** this triggers a partial download of OpenHermes-2.5 (~2 GB
   total but streaming = small slice). Confirm with the user before running.

2. Eyeball-classify each record as `pedagogy | factoid | off-topic` per
   handoff §3.2 protocol. Accept the filter if ≥60% pedagogy AND <10%
   off-topic on the sample.

3. Save the audit log to `results/stage1_filter_audit/openhermes.jsonl`
   and a short summary commit. If the filter fails the audit, do NOT iterate
   on keywords here — surface to the user. Filter design changes are
   out-of-scope for this session.

Repeat for `ultrachat_200k` and `slimorca-dedup` only if time and user
approval permit (each is another ~1-2 GB partial stream).

---

## What's BLOCKED for this session — don't waste time on these

1. **`scripts/llm_judge_eval.py` on existing baselines.** Requires a
   `dialogues/` subdirectory under `results/synthetic-baseline/<config>/`.
   Verified `2026-05-25`: those subdirs don't exist anywhere in `results/`.
   The current baselines only persist `metrics_summary.json` and
   `run_config.json`. The R9700 session has to re-run eval with dialogue
   persistence on before the judge can score them. This is a sequencing
   issue, not a code issue.
2. **Stage 2b QLoRA training.** Needs GPU. R9700 session only.
3. **n=75 synthetic *eval*.** Generation (this session) is decoupled from
   eval (R9700 session). Don't try to run BERT consultant + teacher LLM
   inference here.
4. **DPO Source 1 implementation** in `scripts/build_dpo_pairs.py`.
   Scaffolded inert in PR #79; needs a Stage 2b checkpoint that doesn't
   exist yet. Don't fill in the body until the R9700 session has trained.
5. **DPO Source 2 implementation.** Needs per-turn STL synthetic dialogue
   logs that current STL synthetic runs only persist as aggregated
   summaries. Same gating issue as task #1 above.

---

## First-action checklist

1. Confirm you're NOT on the R9700 box: `rocminfo 2>/dev/null | grep gfx`
   should print *nothing* (or fail with "command not found"). If it shows
   `gfx1201`, switch to the R9700 handoff (`docs/HANDOFF_STAGE2_TRAINING.md`).
2. Confirm `ANTHROPIC_API_KEY` is set: `printenv ANTHROPIC_API_KEY | wc -c`
   should print > 50 (don't print the key itself).
3. `git fetch && git checkout feat/stage2-sft-pipeline-design && git pull`.
   If PR #79 already merged: `git checkout main && git pull` instead.
4. `gh pr view 79 --json mergedAt,state,reviewDecision,statusCheckRollup`
   to know whether to expect more changes on the branch.
5. Skim this handoff's "Primary task" section.
6. **Talk to the user before** running synthetic generation (~$4) or any
   HF dataset upload.

---

## Pitfalls to remember

1. **HF re-upload is destructive.** Don't overwrite `ulises-c/SocratDataset-SYNTHETIC`
   without explicit approval — it breaks reproducibility of the n=37
   baseline that PR #77's results already measured.
2. **`origin` pushes to both remotes** (`ulises-c/csen-346` and
   `SCU-CSEN346/KELE`). The user is fine with this; don't force-push.
3. **HF collection is documentation-only.** If you create
   `SocratDataset-SYNTHETIC-n75`, manually add it to
   `huggingface.co/collections/ulises-c/socratic-teaching-datasets` via the
   HF web UI. No API for collection management.
4. **Don't run Stage 1 dataset full downloads here.** OpenHermes is ~2 GB,
   UltraChat ~1.5 GB, SlimOrca ~700 MB. The audit task above only needs a
   small streamed slice; let `streaming=True` do its job and break early.
5. **Cost discipline:** ANY API spend > $1 or wall-clock > 30 min needs
   explicit user approval before launch. Quote the estimate in the question.

---

## Coordinating with the R9700 session

If both sessions are running concurrently:

- **No file-overlap conflicts expected.** R9700 writes to
  `outputs/sft-stage2-socratic/` and `results/synthetic-baseline/<config>/dialogues/`;
  this session writes to `data/synthetic_extension/` and `.gitignore`.
- **Coordinate on PR base branches.** If the R9700 session is running on
  the same `feat/stage2-sft-pipeline-design` branch, push any commits from
  this session promptly so the training session can `git pull` and pick
  them up. If the R9700 session is on its own results branch, just merge
  upward later.
- **Don't both push HF dataset uploads simultaneously.** If you're about to
  push `SocratDataset-SYNTHETIC-n75`, the R9700 session needs to know so
  the loader catalog stays consistent.

---

## References

| File | Why |
|---|---|
| `scripts/generate_synthetic_socrat.py` | The generator. Has `--start-id` for ID-collision avoidance |
| `scripts/llm_judge_eval.py` | Judge — blocked until dialogues exist. Don't run yet |
| `src/project/dataset_catalog.json` | Where to add a new HF dataset ID after upload |
| `src/project/dataset.py` | `load_socrat_synthetic` reads from HF. Update the `hf_repo=` default if you publish a -n75 variant |
| `docs/TRAINING_PLAN.md` | §0.1 deferred-evidence rationale |
| `docs/HANDOFF_STAGE2_TRAINING.md` | What the R9700 session is doing — read for context |
