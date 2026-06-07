# MELE Demo Setup

MELE is a Socratic math-tutoring assistant. The demo runs a Chainlit web UI on port 8000,
backed by a local Qwen3.5-0.8B-LoRA state classifier and a Gemma 4 31B teacher served
either remotely (OpenRouter) or locally (llama.cpp on an RTX 5090).

Three topologies are supported:

| Topology | Teacher | Where to run | GPU needed? |
|---|---|---|---|
| [Online](#1-online-demo--laptop-no-gpu) | OpenRouter (hosted) | Your laptop | No |
| [Local](#2-local-demo--rtx-5090-full-local) | llama.cpp on-box | RTX 5090 box | Yes — RTX 5090 (32 GB) |
| [Quasi-local via Tailscale](#3-quasi-local-demo--rtx-5090-over-tailscale) | llama.cpp on-box | RTX 5090 box (remote) | Yes — RTX 5090 (32 GB) |

---

## Replicate from scratch (online demo, no GPU)

Fastest path for someone who only has a laptop. The classifier runs on CPU and the
teacher is hosted by OpenRouter, so no GPU, CUDA, or llama.cpp setup is required.

```bash
# 1. Install uv (the project's package manager) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repo
git clone git@github.com:SCU-CSEN346/KELE.git
cd KELE

# 3. Get a free OpenRouter key at https://openrouter.ai, then create a
#    gitignored .env at the repo root (never commit this file):
printf 'TEACHER_API_KEY=sk-or-v1-<your-key>\n' > .env

# 4. Launch. This downloads the classifier checkpoint, runs `uv sync --extra demo`,
#    starts Chainlit, and opens http://localhost:8000 in your browser.
make online-demo
```

That's the whole loop — no other config needs editing (the teacher model and base URL
live in `configs/gemma4-31b-online.env`). If OpenRouter's free tier rate-limits you during
a heavy session, see [Rate limiting](#rate-limiting) below.

---

## Shared prerequisites

**Classifier checkpoint** — downloaded automatically by `make online-demo` / `make local-demo`
if missing. To pre-fetch manually:

```bash
hf download ulises-c/socrates-state-classifier-qwen3.5-lora \
    --local-dir results/state-clf-qwen3.5-0.8b-lora-wandb/final
```

**Python dependencies** — handled by the Make preflight step:

```bash
uv sync --extra demo --inexact   # installs chainlit; --inexact keeps torch intact
```

---

## 1. Online demo — laptop, no GPU

The classifier runs locally on CPU. The teacher is served by OpenRouter's free tier — no
local GPU or llama.cpp setup needed.

### Setup

1. Get a free OpenRouter key at https://openrouter.ai and create a gitignored `.env` at the
   repo root:

   ```
   TEACHER_API_KEY=sk-or-v1-<your-key>
   ```

   The rest of the config is in `configs/gemma4-31b-online.env` and does not need to be
   touched.

2. Launch:

   ```bash
   make online-demo
   ```

   The classifier downloads automatically if missing, dependencies sync, then Chainlit
   starts and the browser opens to `http://localhost:8000`.

### Rate limiting

The OpenRouter `:free` tier is capped (requests/day, queuing under load). For a heavy live
booth drop the `:free` suffix:

```bash
# one-off override — no config edit needed
TEACHER_MODEL_NAME=google/gemma-4-31b-it make online-demo
```

Or set `TEACHER_MODEL_NAME=google/gemma-4-31b-it` in your `.env`.

---

## 2. Local demo — RTX 5090, full local

Everything stays on the box. The teacher is Gemma 4 31B-it (Unsloth UD-Q5_K_XL GGUF) on
llama.cpp at port 8080; the classifier and web UI run in the same process.

### Prerequisites

- **llama.cpp** built with CUDA support. The server binary is expected at
  `~/Github/llama.cpp/build/bin/llama-server` or
  `~/Documents/models/llama.cpp/build/bin/llama-server`.
  Override with `LLAMA_SERVER=/path/to/llama-server`.

- **Model weights** — `gemma-4-31B-it-UD-Q5_K_XL.gguf` in
  `~/Documents/models/weights/`. Override with `GEMMA4_31B_WEIGHTS_DIR=/other/path`.
  About 22 GB on disk, ~30 GB VRAM at 150K context on the RTX 5090.

### Launch

```bash
make local-demo
```

On first run llama-server cold-loads the model (~30–90 s). It is booted **detached** and
stays alive after the session ends — restart the REPL instantly without reloading weights:

```bash
make local-demo   # reuses the running server if it is already up
```

To take the server down:

```bash
kill "$(cat results/_demo/llama_server.pid)"
```

---

## 3. Quasi-local demo — RTX 5090 over Tailscale

Run the full local stack on the RTX 5090 box at home/office and present from a laptop at
the event. Your laptop only needs a browser and Tailscale; all inference stays local.

### Prerequisites

- Tailscale installed and running on **both** the RTX 5090 box and the presenter laptop.
  Both must be on the same tailnet.
- SSH access to the box over Tailscale.
- The local-demo prerequisites above (llama.cpp + GGUF weights) satisfied on the box.

### Setup on the RTX 5090 box

```bash
# 1. Confirm Tailscale is up and get the tailnet IP
tailscale ip -4

# 2. Start sshd if not already running
sudo systemctl start ssh

# 3. Open a persistent tmux session
tmux new -s demo

# 4. Launch the demo — Chainlit binds to 0.0.0.0:8000
make local-demo
```

The terminal will print:

```
Open in browser:  http://<tailscale-ip>:8000
(or http://localhost:8000 on this machine)
```

### Connect from the presenter laptop

Open `http://<tailscale-ip>:8000` in a browser on the laptop. No port-forwarding or SSH
tunnel needed — Tailscale routes traffic directly between devices on the tailnet.

If you want to keep the tab ready before going on stage, open it after the server is up
(once `make local-demo` has printed "ready"). The server stays alive across demo restarts
so reloading the browser tab reconnects instantly.

### Detaching without stopping the demo

Press `Ctrl-B D` inside tmux to detach. The server and web UI keep running. Re-attach
with `tmux attach -t demo`.

---

## Terminal fallback (all modes)

Pass `WEBUI=0` to drop into an interactive REPL instead of the browser UI:

```bash
WEBUI=0 make online-demo
WEBUI=0 make local-demo
```

---

## Quick reference

| What | Command / path |
|---|---|
| Launch online (hosted teacher) | `make online-demo` |
| Launch local (llama.cpp teacher) | `make local-demo` |
| Force CPU for classifier | `KELE_BERT_DEVICE=cpu make ...` |
| Use paid OpenRouter tier | `TEACHER_MODEL_NAME=google/gemma-4-31b-it make online-demo` |
| Stop local llama-server | `kill "$(cat results/_demo/llama_server.pid)"` |
| Follow server log | `tail -f results/_demo/server_<timestamp>.log` |
| Classifier checkpoint (default) | `results/state-clf-qwen3.5-0.8b-lora-wandb/final/` |
| OpenRouter key | `.env` → `TEACHER_API_KEY=sk-or-v1-...` (gitignored, never commit) |
| Web UI entry point | `src/project/demo/demo_web_ui.py` |
| Shell launcher | `scripts/serve_demo_top_performer.sh` |
| Online teacher config | `configs/gemma4-31b-online.env` |
| Local teacher config | `configs/gemma4-31b-local.env` |
