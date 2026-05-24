# AMD R9700 — llama.cpp Vulkan/HIP Stack

GPU tech stack upgrade completed 2026-05-24. This supersedes the HF Transformers
serving path for the R9700 noted in `GPU_SUPPORT.md`.

---

## What changed

The R9700 now runs GGUF models via **llama-server** (llama.cpp) instead of HF
Transformers. This unlocks quantized inference (Q4/Q5) and two competing GPU
backends in a single binary.

| | Before | After |
|---|---|---|
| Server | HF Transformers (FastAPI) | llama-server (llama.cpp) |
| Models | bfloat16 full weights | GGUF Q4/Q5 |
| GPU backends | ROCm HIP only | ROCm HIP **+** Vulkan |
| VRAM for Gemma 31B | ~62 GB (doesn't fit) | ~22 GB (Q5\_K\_XL) |
| Flash attention | ✗ | ✅ (`-fa auto`) |

---

## Build details

**Binary:** `~/Github/llama.cpp/build/bin/llama-server`
**Built:** 2026-05-24 from commit `5d246a792`

CMake flags used:
```
-DGGML_HIP=ON
-DGGML_VULKAN=ON
-DAMDGPU_TARGETS=gfx1201
-DGGML_CUDA_FA=ON
-DGGML_HIP_GRAPHS=ON
-DGGML_HIP_MMQ_MFMA=ON
-DCMAKE_BUILD_TYPE=Release
```

Both backends confirmed linked:
```
libggml-hip.so.0    → ROCm / HIP
libggml-vulkan.so.0 → Vulkan (RADV)
```

At runtime, llama-server auto-selects the backend. Override with `-dev`:
```bash
# Force Vulkan
EXTRA_ARGS="-dev Vulkan0" ./scripts/serve_gemma4_31b_q5.sh -m <weights>

# Force ROCm/HIP
EXTRA_ARGS="-dev rocm0" ./scripts/serve_gemma4_31b_q5.sh -m <weights>
```

---

## Lemonade

[Lemonade](https://github.com/lemonade-sdk/lemonade) v10.6.0 is installed
alongside llama-server as a higher-level option.

| Binary | Role |
|---|---|
| `/usr/bin/lemond` | Server daemon (port 8080 by default) |
| `/usr/bin/lemonade` | CLI client |
| `/usr/bin/lemonade-desktop` | Desktop GUI |

**They do not conflict** with llama-server — they're independent processes.
Never run both on port 8080 simultaneously; pass `--port` to one if needed.

Lemonade can wrap your own llama-server binary:
```bash
lemond &
lemonade run --llamacpp-device Vulkan0 --llamacpp-args "-ngl 99 --kv-unified" <model>
```

Lemonade also publishes pre-built ROCm 7 binaries for `gfx120X` (R9700):
https://github.com/lemonade-sdk/llamacpp-rocm/releases

---

## Models in use

| Model | Script | Context | VRAM est. |
|---|---|---|---|
| Gemma 4 31B Q5\_K\_XL | `scripts/serve_gemma4_31b_q5.sh` | 184K (6 slots) | ~22 GB |
| Qwen3.6-27B Q4 | `scripts/serve_qwen27b_q4_local.sh` | varies | ~16 GB |

The serve scripts pick up `LLAMA_SERVER` from the environment; default is
`~/Github/llama.cpp/build/bin/llama-server`. Use this to swap binaries without
editing scripts:
```bash
LLAMA_SERVER=/path/to/other/llama-server ./scripts/serve_gemma4_31b_q5.sh -m <weights>
```

---

## Things to test

### 1. Vulkan vs HIP head-to-head benchmark

Use `llama-bench` with the same model and both devices. Run at minimum with
`-p 512 -n 128` (prompt-processing + token-generation):

```bash
# Vulkan
~/Github/llama.cpp/build/bin/llama-bench \
  -m <weights.gguf> -dev Vulkan0 -p 512 -n 128 -fa 1

# HIP
~/Github/llama.cpp/build/bin/llama-bench \
  -m <weights.gguf> -dev rocm0 -p 512 -n 128 -fa 1
```

Expected from community benchmarks (RX 9070 XT / gfx1201):
- Vulkan: ~62 t/s TG
- HIP (ROCm 7.x): narrowing gap — WMMA now in ROCm 7.1+, may be close

Report numbers in TG (tokens/sec) and PP (prompt tokens/sec).

### 2. Flash attention — confirm it activates under Vulkan

Check startup logs for:
```
llm_load_print_meta: flash attn = 1
```
If `0`, flash attention is not active on the Vulkan path despite `-fa auto`. This
matters for long-context KELE runs (184K for Gemma).

### 3. Lemonade backend detection

Start lemond and confirm it detects the R9700:
```bash
lemond &
lemonade status
lemonade backends
```

Test loading Qwen3.6-27B Q4 via Lemonade and compare TG to llama-server direct.

### 4. lemonade-sdk prebuilt ROCm 7 binary

Download the `gfx120X` build from
https://github.com/lemonade-sdk/llamacpp-rocm/releases and benchmark it
against the hand-built binary. The prebuilt ships ROCm 7.13/7.14 bundled and
may have newer HIP kernels than the system ROCm.

```bash
LLAMA_SERVER=/path/to/prebuilt/llama-server \
  ./scripts/serve_gemma4_31b_q5.sh -m <weights>
```

### 5. Long-context KV stability

Run a KELE eval at full 184K context under Vulkan. Monitor for OOM or KV
corruption that doesn't appear under HIP. The Q4 KV quant (`-ctk q4_0 -ctv
q4_0`) combined with Vulkan is untested at this context length on gfx1201.

```bash
EXTRA_ARGS="-dev Vulkan0" ./scripts/serve_gemma4_31b_q5.sh -m <weights>
# Then run a short KELE eval (n=10) and confirm state_acc is in range
```

### 6. Dual-backend auto-selection (default)

Without `-dev`, llama-server enumerates all backends. Confirm the startup log
shows both devices and that it picks the expected one:
```
ggml_hip: using ROCm ...
ggml_vulkan: Using device ...
```
Note which backend is chosen by default — this is what all serve scripts use
unless `-dev` is explicitly passed.

### 7. vLLM on RDNA4 — re-test when upstream lands

vLLM gfx1201 support is tracked at:
https://github.com/vllm-project/vllm/issues/28649

When native RDNA4 FP8 kernels merge, re-run `scripts/test_vllm_rocm.sh` and
compare to llama-server Vulkan. Until then, vLLM is not viable on R9700
(falls back to FP32, ~50% throughput loss).

---

## Background — why Vulkan was added

Community discussion [#19890](https://github.com/ggml-org/llama.cpp/discussions/19890)
(RTX 5090 vs R9700 llama-bench) showed Vulkan matching or beating HIP on RDNA4
for token generation, while ROCm's WMMA kernels for gfx1201 were still maturing.
ROCm 7.1+ has since improved HIP for RDNA4 — the gap is narrower now, which is
why a direct benchmark (item 1 above) is important before committing to one path.
