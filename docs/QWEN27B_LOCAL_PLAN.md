# Local Qwen3.6-27B Deployment Plan

**CSEN 346 · Santa Clara University**

**Status:** Plan + Phase 2 implementation landed (2026-05-04). Phase 3 (unified
teacher-consultant fusion) deferred — see "Future state" below.

Goal: bring the local llama.cpp + Qwen3.6-27B serving setup from
`~/Documents/models` into csen-346 as a first-class experiment, with both the
teacher and consultant agents pointing at a single llama.cpp server. This
collapses the two-server topology onto one model and stages the architecture
for a future fused single-call teacher-consultant.

---

## 1. Source — `~/Documents/models` (external project)

Self-contained local AI stack tuned for the RTX 5090. Built around llama.cpp
with CUDA (Blackwell SM 120). Files we depend on:

| Path | Purpose |
|---|---|
| `~/Documents/models/llama.cpp/build/bin/llama-server` | OpenAI-compatible server binary |
| `~/Documents/models/weights/Qwen3.6-27B-UD-Q5_K_XL.gguf` | 19 GB clean Q5_K_XL weight file (chosen variant) |
| `~/Documents/models/scripts/serve.sh` | Reference for the two-tier launcher pattern we mirror |
| `~/Documents/models/scripts/serve-qwen-27b-q5.sh` | Reference for model-specific wrapper |

Files we explicitly do **not** depend on: MCP server, MCPO proxy, Open WebUI,
SearXNG, agent runner, eval harness, soul-file system prompts. Those are
personal-stack scaffolding around the model and have no role in KELE.

### Why Q5_K_XL (clean), not Q5_K_P (Uncensored HauhauCS)

The upstream stack defaults to `Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P`
for personal use. For KELE we use the clean `Qwen3.6-27B-UD-Q5_K_XL` because:

1. The dataset is Socratic teaching of children — uncensored fine-tuning's
   safety-removal artifacts are an active liability here.
2. UD-Q5_K_XL is higher fidelity than Q4_K_M and fits comfortably in VRAM at
   416K context (~26 GB total).
3. Academic deliverable — provenance and reproducibility matter, the clean
   `unsloth/Qwen3.6-27B-GGUF` source meets that bar.

### Why one model serves both roles

llama.cpp's server runs **6 parallel slots with unified KV cache and continuous
batching**. The teacher and consultant in our pipeline are two roles
distinguished only by system prompt, fired in sequence per turn — they fit on
one server with no contention and no extra VRAM. This collapses our existing
two-server topology (teacher 8001 + consultant 8002) onto one process.

It is also the natural staging ground for the eventual fusion: once both roles
already use the same model, replacing them with a single structured-output
call is a code-only change.

---

## 2. Phase 2 — Bring the deployment over (this commit)

### Files added

| File | Purpose |
|---|---|
| `scripts/serve_qwen27b.sh` | Generic OpenAI server launcher (port-of upstream `serve.sh`). Resolves binary and weights via env vars; passes through any extra flag. |
| `scripts/serve_qwen27b_q5.sh` | Thin wrapper: picks `Qwen3.6-27B-UD-Q5_K_XL.gguf`, alias `"Qwen 27B Q5"`, context 425984 (416K). |
| `configs/qwen27b-local.env` | Both `TEACHER_BASE_URL` and `CONSULTANT_BASE_URL` point at `localhost:8080/v1`; both model names match the server alias. |
| `Makefile` | New `serve-qwen27b` target. |

### Key choices

- **Binary and weights stay external.** The repo references
  `~/Documents/models/llama.cpp/build/bin/llama-server` and
  `~/Documents/models/weights/` via env-var-defaulted paths
  (`LLAMA_SERVER`, `QWEN27B_WEIGHTS_DIR`). Copying invites version drift; one
  llama.cpp build serves both projects.
- **Dual-role single server.** Both `TEACHER_*` and `CONSULTANT_*` env vars
  point at port 8080. The OpenAI client in `socratic_teaching_system.py`
  doesn't care that both clients hit the same URL — they just send different
  system prompts.
- **No `CONSULTANT_NUM_CTX`.** That env var is wired through as Ollama's
  `options.num_ctx` extra-body field (see `socratic_teaching_system.py:314-316`).
  llama.cpp ignores it; the server's `-c 425984` is the real context budget.
  Leaving it unset avoids a confusing knob.
- **No `CONSULTANT_DISABLE_THINKING`.** Qwen3.6 supports thinking mode;
  letting the consultant think on state-classification turns is desirable.
- **Q4_0 KV quant** stays on (matches upstream default). Cuts KV cache ~4×
  vs FP16; no observed quality degradation in upstream eval.

### Usage

```bash
# Terminal 1: start the server
make serve-qwen27b
# (or with overrides:  bash scripts/serve_qwen27b_q5.sh -np 4 -p 8081)

# Terminal 2: smoke test
./scripts/run_eval.sh qwen27b-local --limit 3

# Terminal 2: full eval
./scripts/run_eval.sh qwen27b-local

# Compare against baseline
poetry run kele-eval --compare results/baseline results/qwen27b-local
```

### VRAM and throughput expectations

| | Value |
|---|---|
| Weights | 19 GB |
| KV cache @ 416K (Q4_0) | ~7.3 GB |
| Total | ~26.3 GB on 32 GB VRAM (~6 GB headroom after OS) |
| Per-slot context | 416K / 6 ≈ 69K (KELE turns are <10K — comfortable) |
| Throughput | ~50–70 tok/s expected (vs ~80–120 tok/s for SocratTeachLLM 9B vLLM) |

### Risks / things to watch

- **Wall-clock per eval run** likely ~1.5× longer than SocratTeachLLM. Worth
  it for the quality bump and the unification. The 6-slot parallelism partially
  offsets this — the old setup didn't batch teacher+consultant either.
- **Slot contention during high parallelism:** monitor `curl localhost:8080/slots`
  during long runs. If we ever spawn parallel evals, `-np` may need lifting.
- **Server boot is slow** (~30–60 s to load 19 GB GGUF). Use `make serve-qwen27b`
  in a persistent terminal and leave it up across runs.

---

## 3. Phase 3 — All-in-one teacher-consultant (FUTURE — not in this commit)

Once Phase 2 is validated end-to-end, the next move is consolidating the two
sequential calls into one. This maps directly to **Approach #4
(Consultant-Teacher Fusion)** in `docs/IMPROVEMENT_PLAN.md`.

### Architecture

**Today (two calls per turn):**
```
student_input
   → consultant call  → {evaluation, state}
   → state_to_action lookup → action
   → teacher call      → teacher_response
```

**After fusion (one call per turn):**
```
student_input + history
   → unified call  → {state, action, teacher_response}
```

### Implementation sketch

- **New module:** `src/project/socratic_teaching_unified.py`
- **New eval entry:** `kele.py evaluate --unified` switches between
  `run_single_dialogue` (current 2-agent) and `run_single_dialogue_unified`.
  Both functions write identical-shape `dialogue[]` outputs so `metrics.py`
  and `kele-eval --compare` work unchanged.
- **Structured output:** llama.cpp supports JSON-schema-constrained decoding
  via `response_format`. The schema enforces
  `{state: enum[34 values], action: string, teacher_response: string}`.
- **State→action lookup** stays as a post-call sanity check: if the model's
  emitted action disagrees with the canonical map for the predicted state,
  prefer the canonical action. (Or: drop `action` from the schema and always
  derive it from `state` — simpler, less to go wrong.)

### Why this is now a small change

- Both agents already point at the same model — no inter-server orchestration.
- Grammar-constrained decoding makes the structured contract enforceable
  without retry loops.
- The `state_to_action` dict in `socratic_teaching_system.py:50-89` is a pure
  lookup; promoting it from prompt-internal to post-processing is trivial.

### Expected benefits (per IMPROVEMENT_PLAN.md #4)

- +5–10 ROUGE-1 (joint training signal — state predictions that help
  generation emerge naturally)
- ~½× per-turn latency
- Simpler context flow, no inter-agent context loss
- One less knob to tune (no consultant temperature/prompt drift independent
  of teacher)

### Order of operations

1. Land Phase 2 (this commit). Validate sane numbers on n=25 and n=681 against
   the existing 2-call architecture with Qwen 27B.
2. Implement `socratic_teaching_unified.py` + `--unified` flag.
3. Run on n=25 smoke. Compare metrics + visually inspect dialogues.
4. If unified beats 2-call: make it the default for `qwen27b-local` config.
   Keep 2-call mode for paper-faithful baseline comparisons.

### Risk

The unified model could collapse into degenerate behavior (e.g. always
predicting `e34` to short-circuit dialogues, or memorizing one good
teacher-style and ignoring the state). Mitigations:

- Schema-enforced state output (can't skip the field).
- Per-stage accuracy in our existing metrics catches state-collapse immediately.
- Keep 2-call mode as an A/B toggle — never delete it.

---

## 4. References

- `docs/IMPROVEMENT_PLAN.md` — full 10-approach improvement landscape; #4 is
  this fusion path.
- `docs/PLAN.md` — overall project phasing.
- `~/Documents/models/REFERENCE.md` — upstream local-AI-stack technical
  reference (VRAM tables, build instructions, parallel-slot details).
- `~/Documents/models/scripts/serve.sh` — upstream launcher pattern we mirror.
