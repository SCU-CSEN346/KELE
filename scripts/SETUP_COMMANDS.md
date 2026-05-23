# Setup Commands — RTX 5090 Rig

> **Status (2026-05-22).** This file documents setup for the *original 2026-04 baseline reproduction* (SocratTeachLLM teacher + Qwen3.5-9B/2B consultant on vLLM). The **current locked headline** uses llama.cpp-served Gemma 4 31B + BERT consultant — see [`RTX5090_SETUP.md`](RTX5090_SETUP.md) for the current rig setup.

## 1. Install system packages (Arch Linux)
```bash
sudo pacman -Syu --noconfirm
sudo pacman -S --noconfirm python-pip pyenv github-cli
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Set up Python 3.12 (system Python 3.14 is too new for vLLM)
```bash
pyenv install 3.12
cd ~/Documents/scu/CSEN-346/csen-346
```

## 3. Install Python dependencies
```bash
uv sync --group dev --python ~/.pyenv/versions/3.12.*/bin/python
```

## 4. Download models
```bash
# Teacher: SocratTeachLLM (~19GB)
hf download yuanpan/SocratTeachLLM --local-dir ~/hf_models/SocratTeachLLM

# Consultant: Qwen3.5-9B (~17GB)
hf download Qwen/Qwen3.5-9B --local-dir ~/hf_models/Qwen3.5-9B
```

## 5. Configure environment
```bash
cp .env.example .env
# .env is pre-configured for local models — no API keys needed
```

## 6. Start both model servers (~24GB VRAM total)
```bash
./scripts/serve_both.sh
# Or in separate terminals:
# ./scripts/serve_socratteachllm.sh   (port 8001)
# ./scripts/serve_consultant.sh       (port 8002)
```

## 7. Smoke test (3 dialogues)
```bash
python3 -m src.project.kele test --n 3 --output results/test
```

## 8. Full overnight evaluation (680 dialogues)
```bash
./scripts/run_eval.sh
```
