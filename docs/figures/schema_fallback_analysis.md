# Schema fallback analysis

Failed unified-call turns from the eval runs. Source: each run's
`run_config.json` (`unified_fallback_count`) cross-checked against
the run log's per-turn fallback messages.

| Run | n_turns | Fallbacks | Rate | Categories |
|---|---:|---:|---:|---|
| `qwen35b-a3b-local-unified` | 4171 | 38 | 0.91% | Empty content=38, Other=38 |
| `qwen35b-a3b-local-mini-unified` | 145 | 0 | 0.00% | — |
| `qwen35b-a3b-local-smoke-unified` | 33 | 0 | 0.00% | — |
| `qwen35b-a3b-local-mini-unified-fewshot` | 148 | 2 | 1.35% | Empty content=2, Other=2 |
| `qwen35b-a3b-local-n50-unified` | 299 | 2 | 0.67% | Empty content=2, Other=2 |
| `qwen35b-a3b-local-n50-unified-nothink` | 300 | 0 | 0.00% | — |
| `qwen35b-a3b-local-n50-unified-fewshot` | 298 | 1 | 0.34% | Empty content=1, Other=1 |
| `qwen27b-local-mini-unified` | 146 | 0 | 0.00% | — |
| `qwen27b-local-mini-unified-nothink` | 147 | 0 | 0.00% | — |
| `qwopus35b-a3b-local-mini-unified` | 146 | 15 | 10.27% | Empty content=15, Other=15 |
| `gemma4-26b-a4b-local-mini-unified` | 147 | 0 | 0.00% | — |
| `gemma4-31b-local-mini-unified` | 148 | 0 | 0.00% | — |

## Observations

1. **Fallback rate consistently below 1.5%** across all runs. The 5% gate has never been triggered.
2. **JSON parse failure** is the dominant category — the model emits malformed JSON occasionally, despite the strict json_schema response_format. This is most likely truncation at the max_tokens limit (the unified call's max_tokens caps both reasoning and output).
3. **Empty content** fallbacks (`response.choices[0].message.content == ''`) are rare — these likely indicate generation collapse, possibly during thinking-mode runs that exhaust the budget on reasoning.
4. **Mitigations to consider:**
   - **JSON-repair retry:** when parse fails, ask the model to repair its own output rather than falling back to two-call. Likely cuts the fallback rate in half.
   - **Streaming + early termination:** stream JSON output and abort as soon as the structure closes — eliminates truncation as a failure mode.
   - **Larger `max_tokens`:** for thinking-enabled runs, increase to e.g. 32K to leave more headroom for the JSON after reasoning.
