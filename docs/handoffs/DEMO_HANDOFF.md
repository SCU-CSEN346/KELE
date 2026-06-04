# Handoff — MELE live demo (Chainlit web UI + OpenRouter / llama.cpp)

**For:** the next Claude Code session on any machine (Mac laptop, Mac mini, or RTX 5090 box)
**Branch:** `feat/demo-2026-Jun-04` — not yet merged to main as of 2026-06-04
**Purpose:** run the MELE Socratic-teaching demo in a browser at port 8000, with the
Qwen3.5-0.8B-LoRA state classifier running locally and either a hosted teacher
(laptop — no GPU needed) or a local llama.cpp teacher (RTX 5090 box).

---

## TL;DR — what was built this session

The demo stack went from a terminal REPL to a polished Chainlit web UI. Key changes:

| Commit | What |
|---|---|
| `4ea60e4` | `ONLINE=1` topology added to `serve_demo_top_performer.sh` — hosted teacher via OpenRouter |
| `3c4acc7` | `configs/gemma4-31b-online.env` pinned to OpenRouter `:free` tier |
| `e886f4c` | `make online-demo` / `make local-demo`, English UI strings, 429 retry, MELE rename |
| `289efa4` | `src/project/demo_web_ui.py` — isolated Chainlit app; no core logic polluted |
| `58316d0` | `.gitignore` covers `.DS_Store` everywhere |
| `120b9c4` | `local-demo` explicitly sets `WEBUI=1` to match `online-demo` |

---

## How to launch the demo

### Laptop (no GPU) — online mode

```bash
# One-time: put the OpenRouter key in the gitignored .env
echo 'TEACHER_API_KEY=sk-or-v1-...' >> .env

# Launch — downloads classifier from HF if missing, installs chainlit, opens :8000
make online-demo
```

Open `http://localhost:8000` in a browser. Chat with MELE.

### RTX 5090 box — local mode

```bash
make local-demo
```

Opens `:8000` same as online mode. The llama.cpp server boots detached in the
background and stays alive for instant restarts. To kill it:
`kill "$(cat results/_demo/llama_server.pid)"`.

### Terminal fallback (both modes)

```bash
# Online, no web UI
WEBUI=0 make online-demo

# Local, no web UI
WEBUI=0 make local-demo
```

---

## Key files

| File | Role |
|---|---|
| `src/project/demo_web_ui.py` | **UI-only** Chainlit app — `@cl.on_chat_start`, `@cl.on_message`. No core logic here |
| `scripts/serve_demo_top_performer.sh` | Shell launcher: boots llama.cpp (LOCAL) or skips it (ONLINE), then execs `chainlit run` |
| `configs/gemma4-31b-online.env` | Experiment config for the online topology. All teacher/consultant fields; **no** `TEACHER_API_KEY` (must be in `.env` or exported) |
| `configs/gemma4-31b-local.env` | Experiment config for the local llama.cpp topology |
| `.env` (gitignored) | Secrets — `TEACHER_API_KEY`, `TEACHER_BASE_URL`, `TEACHER_MODEL_NAME` |
| `src/project/kele.py:create_system` | Factory: loads config, builds `SocraticTeachingSystemBertConsultant` |
| `src/project/socratic_teaching_system.py` | Core teaching loop — `process_student_input()` is the entry point; all UI strings are English; name is MELE |
| `src/project/socratic_teaching_bert_consultant.py` | BERT variant — Qwen3.5-0.8B-LoRA predicts 34-state, LLM generates teacher response |

---

## Environment variables

| Variable | Default | Set by |
|---|---|---|
| `ONLINE` | `0` | `make online-demo` sets `1`; shell script |
| `EXPERIMENT` | `gemma4-31b-local` or `-online` depending on `ONLINE` | script or Make override |
| `BERT_CKPT` | `results/state-clf-qwen3.5-0.8b-lora-wandb/final` | Makefile `BERT_CKPT ?= ...` |
| `KELE_BERT_DEVICE` | `auto` | `make online-demo` / `make local-demo` set `cpu` |
| `WEBUI` | `1` | Both Make targets explicitly set `1`; `0` for terminal fallback |
| `KELE_FEW_SHOT_TEACHER` | (unset) | Set to `1` by the shell script to enable 10-shot exemplar injection |
| `KELE_FEW_SHOT_N` | `10` | Shell script |
| `KELE_TEACHER_LANG` | (unset) | `gemma4-31b-online.env` sets `auto` — mirrors student language in the system prompt |
| `TEACHER_API_KEY` | — | **Must be in `.env` for online mode.** Key: `sk-or-v1-8b1379...` (see `.env`) |

---

## The `.env` file (gitignored — never commit)

The file lives at the repo root and is never tracked. For the online topology it must contain:

```
CONSULTANT_API_KEY=not-needed
TEACHER_API_KEY=sk-or-v1-<your-openrouter-key>
TEACHER_BASE_URL=https://openrouter.ai/api/v1
TEACHER_MODEL_NAME=google/gemma-4-31b-it:free
DEBUG_MODE=false
MAX_TEACHING_ROUNDS=8
```

`config.load_env_file()` uses `setdefault()` — the experiment `.env` file sets everything
except `TEACHER_API_KEY`, so `.env` only needs to supply the key and optionally override
`TEACHER_MODEL_NAME` (e.g., drop `:free` for paid inference).

---

## Classifier checkpoint

The BERT classifier must exist at `$BERT_CKPT` (default:
`results/state-clf-qwen3.5-0.8b-lora-wandb/final`). The preflight step in
`make online-demo` / `make local-demo` downloads it automatically if missing:

```bash
hf download ulises-c/socrates-state-classifier-qwen3.5-lora \
    --local-dir results/state-clf-qwen3.5-0.8b-lora-wandb/final
```

Or override: `BERT_CKPT=/path/to/other/final make online-demo`.

---

## Rate limiting (online mode)

The OpenRouter `:free` tier throttles hard. The retry logic in
`socratic_teaching_system.py` handles `openai.RateLimitError` with up to 6
attempts for teacher calls and 4 for consultant calls. Wait is
`min(2^attempt, 30)` seconds (honors `Retry-After` header when present). The
user sees a countdown in the terminal. If the free tier is too slow for a live
booth, drop the `:free` suffix:

```bash
TEACHER_MODEL_NAME=google/gemma-4-31b-it ONLINE=1 bash scripts/serve_demo_top_performer.sh
```

---

## Chainlit setup

`chainlit` is an optional dependency — install via:

```bash
uv sync --extra demo --inexact
```

`pyproject.toml` declares `demo = ["chainlit>=2.4,<3"]`. The `--inexact` flag
prevents uv from removing other installed packages (e.g., torch). The Makefile
`_demo-preflight` target runs this automatically before either demo target.

Chainlit generates two files on first launch that are **not committed**:
- `.chainlit/` — runtime config dir
- `chainlit.md` — welcome page override

If you want to customize the welcome page, edit `chainlit.md`. It is currently
untracked; add it and commit if you want the customization to persist.

---

## Architecture of the demo stack

```
Browser (port 8000)
    ↕  Chainlit WebSocket
demo_web_ui.py
    asyncio.to_thread(system.process_student_input, user_msg)
        ↓
SocraticTeachingSystemBertConsultant.process_student_input()
    ├── BERT Qwen3.5-0.8B-LoRA  (local, CPU)  →  34-state classification
    └── Teacher LLM call (OpenRouter or llama.cpp)  →  Socratic response
        ↑
configs/gemma4-31b-online.env  +  .env (TEACHER_API_KEY)
```

The UI file is intentionally thin — it only wires Chainlit events to
`process_student_input()`. All pedagogy logic, state machine, retry, and
language-mirror are in the core files.

---

## Known issues / future work

| Issue | Status |
|---|---|
| `post-test-runner.sh` hook fails on macOS (`date +%s%3N`, `timeout` not found) | Filed: https://github.com/ulises-c/Computer-Setup/issues/33 — ignore for now |
| `chainlit.md` welcome page is untracked | Add and commit if a custom welcome screen is wanted for the demo booth |
| OpenRouter free tier rate limits mid-demo | Switch to paid tier or use local llama.cpp (`make local-demo`) for heavy live use |
| `.chainlit/` dir is untracked | Fine to leave untracked; add to `.gitignore` if it becomes noisy |
| `debug=True` in `demo_web_ui.py` → raw classifier output goes to terminal, not browser | Intentional: terminal = raw debug, browser = polished MELE conversation |

---

## Pre-demo checklist

- [ ] `.env` exists and has a valid `TEACHER_API_KEY` (online mode only)
- [ ] Classifier at `results/state-clf-qwen3.5-0.8b-lora-wandb/final/model.safetensors`  
      (or `BERT_CKPT` override set; `make online-demo` auto-downloads if missing)
- [ ] `uv sync --extra demo --inexact` done (Make preflight handles this)
- [ ] For local mode: llama.cpp compiled with CUDA/Metal, Gemma 4 31B Q5 GGUF in place,
      `scripts/serve_gemma4_31b_q5.sh` present
- [ ] Open `http://localhost:8000` after launch; type a math question to verify the full loop

---

## References

| File | Why |
|---|---|
| `scripts/serve_demo_top_performer.sh` | Launcher — read for topology logic and all overrides |
| `src/project/demo_web_ui.py` | Chainlit UI entry point |
| `configs/gemma4-31b-online.env` | Active config for laptop demo |
| `src/project/kele.py` | `create_system()` factory |
| `src/project/socratic_teaching_system.py` | `process_student_input()`, retry, MELE name, English UI strings |
| `src/project/socratic_teaching_bert_consultant.py` | BERT classifier wrapper |
| `Makefile` (lines 290–311) | `BERT_CKPT` default, `_demo-preflight`, `online-demo`, `local-demo` |
| `pyproject.toml` | `demo = ["chainlit>=2.4,<3"]` optional dep |
