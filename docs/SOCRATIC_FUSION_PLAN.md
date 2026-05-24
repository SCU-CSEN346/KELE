# Socratic Teaching Fusion — Single-Call Teacher + Consultant

**CSEN 346 · Santa Clara University · 2026-05-04**

> **Status (2026-05-22).** Implemented as `src/project/socratic_teaching_unified.py` and validated through the A3B/Gemma fusion campaigns. **Superseded for the locked headline** by the BERT-consultant integration (`src/project/socratic_teaching_bert_consultant.py`), which removes the JSON-schema dependency that drove the 21% schema-fallback collapse of standalone-Gemma at full scale. Retained as the **design record** for the shipped fusion architecture; not a current plan.

Goal: collapse the existing two-call per-turn architecture (consultant → teacher)
into a **single LLM call** that emits state classification, evaluation reasoning,
and teacher response together, structured-output-enforced.

This is the formal specification of `IMPROVEMENT_PLAN.md` #4 (Consultant-Teacher
Fusion) and `archive/QWEN27B_LOCAL_PLAN.md` Phase 3, now that we have empirical evidence
the per-turn wall clock is dominated by *count of LLM calls* and *teacher
generation*, not by the consultant alone.

---

## 1. Why now

From `QWEN_LOCAL_EXPLORATION_LOG.md`:

| Config | s/turn | Notes |
|---|---|---|
| 27B think | 72 | Two calls: consultant + teacher both think |
| 27B no-think (consultant) | ~90+ | Consultant CoT removed; **no faster** — teacher still thinks |
| A3B think | 19 | Both thinking on, fast model |
| A3B no-think (consultant) | 17 | Marginal -10% |

The consultant-only `/no_think` lever is a dead end for throughput. The two
remaining levers for full-run viability are:

1. **Disable teacher thinking too** (~half the gain at best, requires code change)
2. **Eliminate one of the two calls entirely** (this plan)

Fusion is the structurally bigger win. Empirically, both agents already run
on the same llama.cpp server (single weight load, 6 parallel slots), so this
is now a code-only change, not infra.

### Expected wall-clock impact

Per-turn budget today (27B): consultant thinking + JSON (~30s) + teacher
thinking + reply (~35s) + RTT/overhead (~5s) ≈ 70s per turn.

Per-turn after fusion: single call generates [thinking + JSON{evaluation, state,
teacher_response}] in one round-trip. Eliminates one warm-up of CoT, one
RTT, and one set of system-prompt-tokens to ingest.

| Model | Today | Projected after fusion | Δ |
|---|---|---|---|
| 27B Q5 | ~72 s/turn → ~88 h full | ~40-50 s/turn → **~50-60 h full** | -33 to -40% |
| A3B Q4 | ~19 s/turn → ~23 h full | ~10-12 s/turn → **~12-15 h full** | -33 to -40% |

A3B + fusion drops the full-run wall clock to under a single overnight session
with margin for restarts. 27B + fusion drops to a long weekend, still not
overnight but recoverable in one weekend.

### Empirical feasibility check (just performed against the live server)

```
curl -X POST localhost:8080/v1/chat/completions \
  -d '{"response_format":{"type":"json_schema","json_schema":{...,"strict":true}},...}'
```

→ llama.cpp accepts strict `json_schema` response_format. Schema-constrained
decoding works on this stack.

Bonus: Qwen3.6 emits thinking content into a *separate* `reasoning_content`
field on the response message. Thinking doesn't pollute the JSON output, only
the token budget. We can keep thinking enabled for the unified call (where it's
genuinely productive — multi-step reasoning across both classification and
generation) without parser hacks.

---

## 2. Current per-turn dataflow (the thing we're collapsing)

`process_student_input()` in `src/project/socratic_teaching_system.py:433`:

```
student_input arrives
  ├─→ Call 1: socratic_teaching_consultant(student_input)
  │     ├─ system: ~2.5K-token prompt (5 stages, 34 states, transition rules)
  │     ├─ user:   full history (with prior consultant analyses) + student_input
  │     └─ output: {"evaluation": str, "state": str}
  │
  ├─→ Glue logic (Python, no LLM):
  │     - Phase regression prevention (don't let state go backwards)
  │     - Teaching round counter increment
  │     - Forced d33 / e34 transitions when max_rounds reached
  │     - state_to_action[state]  → action string
  │
  ├─→ Call 2: socrates_teacher(student_input, evaluation, action)
  │     ├─ system: ~150-token prompt (Socratic teacher persona, 6 rules)
  │     ├─ user:   history (no analyses) + student + evaluation + action
  │     └─ output: free-text teacher_response
  │
  └─→ Append to histories, return teacher_response
```

Critical observations:

1. The two calls **share** history and student_input. The unified call already
   knows everything both calls need.
2. The `action` is a **deterministic lookup** from `state_to_action` (line
   50-86). No LLM needs to generate it.
3. The glue logic (regression prevention, round counter, forced transitions)
   is **Python-side, post-classification**. It lives between the two calls
   today, but in the fusion world it lives *after* the unified call, before
   the teacher_response is committed.

That last point matters: in the fused world, the model emits a tentative
`(state, teacher_response)` pair. We then run the same Python glue logic to
*possibly override* the state. If the state is overridden (regression
prevention, forced d33), the `teacher_response` may no longer match the
overridden action. This is the central architectural risk — see §6.

---

## 3. Fused per-turn dataflow

```
student_input arrives
  ├─→ Single call: unified_socratic_step(student_input)
  │     ├─ system: ~3K-token prompt = consultant prompt + state-action map
  │     │           + teacher rules + structured output schema
  │     ├─ user:   full history + student_input
  │     ├─ response_format: json_schema (strict) on
  │     │            {evaluation, state, teacher_response}
  │     └─ output: structured JSON, all three fields
  │
  ├─→ Glue logic (Python — unchanged from today):
  │     - Phase regression prevention → may override state
  │     - Teaching round counter
  │     - Forced d33 / e34 transitions
  │
  ├─→ Coherence reconciliation (NEW):
  │     - If state was overridden by glue, decide whether to re-call the
  │       model with the corrected action, or accept the original
  │       teacher_response (see §6 for tradeoffs)
  │
  └─→ Append to histories, return teacher_response
```

The single call replaces the consultant + state→action lookup + teacher chain.

---

## 4. JSON schema design

```python
SOCRATIC_STEP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evaluation", "state", "teacher_response"],
    "properties": {
        "evaluation": {
            "type": "string",
            "description": (
                "判断当前对话所处的阶段（a/b/c/d/e）和学生状态及原因，"
                "格式与现有顾问输出一致"
            ),
            "minLength": 1,
        },
        "state": {
            "type": "string",
            "enum": [
                "a0", "a1",
                "b2", "b3", "b4", "b5", "b6", "b7",
                "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15",
                "c16", "c17", "c18", "c19", "c20", "c21", "c22", "c23",
                "c24", "c25", "c26", "c27", "c28", "c29",
                "d30", "d31", "d32", "d33",
                "e34",
            ],
        },
        "teacher_response": {
            "type": "string",
            "description": (
                "苏格拉底式问题或总结，符合所判断的state对应的教学操作。"
                "每次仅一个问题，亲切语气，符合小学水平，与解题直接相关。"
                "如果state=e34，则给出题目总结而非提问。"
            ),
            "minLength": 1,
        },
    },
}
```

State enum is the source of truth — derived directly from the keys of
`state_to_action`. Schema enforcement guarantees we never get garbage state
codes.

The schema is passed to llama.cpp via:

```python
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "socratic_step",
        "schema": SOCRATIC_STEP_SCHEMA,
        "strict": True,
    },
}
```

llama.cpp uses GBNF-style grammar-constrained decoding internally — sampling
is restricted at each token to only those that can complete a valid schema
match. Failure modes:

- **Schema-impossible decoding**: the model can't produce a valid output (rare
  with reasonable max_tokens). Surface as an exception → fall back to
  the two-call path for that turn.
- **Truncation**: `max_tokens` exhausted mid-emission. Same fallback.

---

## 5. Unified system prompt

The unified prompt assembles three blocks:

```
# 角色指令 (Role)
你同时担任**苏格拉底教学顾问**和**苏格拉底教师**两个角色。
对每一轮学生输入，你需要：
  1. 作为顾问，分析当前对话状态并选定阶段和状态编号
  2. 作为教师，根据所选状态对应的教学操作生成回复

最终以单一的JSON输出同时呈现两个角色的工作。

---

# 第一部分：顾问职责
[unchanged — full text from socratic_teaching_consultant's system prompt:
 5-stage rules, state tables, transition rules, ~2500 tokens]

---

# 第二部分：状态-操作映射
（你判断的state决定了你作为教师必须执行的教学操作）

a0: 引导学生提出问题
a1: 生成一个与解题相关的子问题
b2: 从不同角度生成问题
b3: 更改问题
... [all 34 entries from state_to_action, ~600 tokens]
e34: 对题目进行总结

---

# 第三部分：教师职责
[from socrates_teacher's system prompt:
 - 每次只能提出一个问题
 - 必须与解题直接相关
 - 符合小学知识水平
 - 语气亲切、有鼓励性
 - 除非操作要求，否则不给明显提示
 - 如果操作是"对题目进行总结"，则总结题目且不再提问
 ~150 tokens]

---

# 输出格式
仅输出一个JSON对象，包含以下字段：
{
  "evaluation": ...（顾问分析）,
  "state": ...（34个状态编号之一）,
  "teacher_response": ...（教师对学生的回复，符合state对应的操作）
}
```

The user-message stays simple:

```
历史对话记录：
{full_formatted_history}

当前学生输入：{student_input}
```

Total system prompt budget: ~3.2K tokens (vs current consultant 2.5K + teacher
0.15K = 2.65K combined). Slight increase for the explicit map and bridging
language; well within context limits.

---

## 6. The state-coherence problem

The hardest design issue is what happens when Python glue logic overrides the
model's emitted state, but the model already emitted a `teacher_response` based
on the *original* state.

Concrete example: model emits `(state=a0, teacher_response="老师再说说这道题...")`
because it thinks the student didn't ask a question. Glue rules then decide
"actually we forced d33 because we hit max_rounds". The teacher_response is now
out of phase with the action `d33` would imply (`建立一个普遍定义并要求学生给出题目答案`).

Three options, ranked by complexity:

### Option A: Trust the model's response (simplest, default)
After glue overrides state, **keep the original `teacher_response`**. Log the
divergence for analysis. The state recorded in history is the corrected one;
the teacher said something nominally inconsistent, but the dialogue continues.

- **Pro:** Single LLM call, easy to implement, matches "ship it" energy.
- **Con:** Logical incoherence in some turns (when glue overrides). Likely a
  small fraction of turns (regression prevention only fires on bad model
  predictions; max_rounds forcing only fires near end of long dialogues).

### Option B: Re-call on override (medium)
If glue overrides state, fire a second call with `socrates_teacher(student,
evaluation, corrected_action)` to regenerate `teacher_response`. Falls back
to the existing two-call path *only on override*.

- **Pro:** Coherence preserved. Override is the rare case → small wall-clock
  cost (maybe 5–10% of turns trigger it).
- **Con:** Two-tier code path; harder to reason about; loses determinism.

### Option C: Glue inside the prompt (hardest, riskiest)
Embed regression-prevention and round-counter rules directly in the prompt so
the model emits a glue-compatible state from the start.

- **Pro:** No override needed, full coherence.
- **Con:** Pushes deterministic logic into the LLM; loses guardrails; the
  current Python glue is a reliability feature, not a workaround.

**Recommendation: Option A for v1, instrument the divergence rate, decide on
Option B if it matters.** Option C is a non-starter — the glue exists because
the LLM is unreliable; we don't want to take that protection away.

---

## 7. Implementation plan

### File-by-file

**New: `src/project/socratic_teaching_unified.py`**

Defines `SocraticTeachingSystemUnified`. Subclass of `SocraticTeachingSystem`
to inherit:

- `__init__` (same env config) — ensures CLI compatibility
- `reset_session`, `add_to_history`, `add_to_consultant_history`
- `get_formatted_history`, `get_full_formatted_history`
- `state_to_action`
- `start_conversation` (interactive REPL)

**Override:**
- `process_student_input(student_input) -> str` — calls the unified method
  instead of the two separate calls.

**New methods:**
- `unified_socratic_step(student_input) -> dict[str, Any]` — single LLM call,
  returns `{"evaluation": str, "state": str, "teacher_response": str}`.
- `_build_unified_system_prompt() -> str` — assembles the three-section
  prompt from existing strings (DRY).

**JSON schema:** module-level constant `SOCRATIC_STEP_SCHEMA`.

**Failure handling:**
- Schema parse failure → fall back to two-call (re-use parent's
  `socratic_teaching_consultant` + `socrates_teacher`). Log incident.
- LLM API failure → same fallback to two-call.
- Empty `teacher_response` → fall back.

**Modified: `src/project/kele.py`**

`create_system()` (line 18) gets a `unified: bool = False` parameter. When
True, instantiates `SocraticTeachingSystemUnified` instead of
`SocraticTeachingSystem`. Reads from `os.environ["UNIFIED_MODE"]` as a
default if not passed.

Add CLI flag `--unified` to both `evaluate` and `test` subcommands. Plumbs
through to `run_batch_evaluation` and into `create_system`.

`run_single_dialogue` requires **no changes** — it calls
`system.process_student_input()` polymorphically.

`run_config.json` written by `run_batch_evaluation` adds `"unified": true` when
the flag was set, so post-hoc analysis can distinguish runs.

**Modified: `configs/qwen27b-local.env`, `configs/qwen35b-a3b-local.env` and their `-nothink` siblings**

No env-var changes required if we pass `--unified` on the CLI. Optionally add
`UNIFIED_MODE=true` to `qwen27b-local-unified.env` and similar variants if we
want config-file selection.

**New: orchestrator support**

The two `eval_qwen*.sh` orchestrators get `--unified` passed through to
`kele.py`. Output dir gets `-unified` suffix (parallel to `-nothink`). Same
flag-suffix logic as the existing `--nothink` plumbing.

So after this lands, the matrix is:

```
qwen{27b,35b-a3b}-local  ×  {think, no-think}  ×  {two-call, unified}
= 2 models × 2 think modes × 2 architectures = 8 combinations
```

Each combination gets a distinct output dir and is comparable through
`kele-eval --compare`.

### Tests

- `tests/test_socratic_teaching_unified.py` (new):
  - Mock the OpenAI client to return a known JSON.
  - Verify `unified_socratic_step` parses correctly.
  - Verify schema enforcement: deliberately return non-conforming JSON,
    confirm fallback fires.
  - Verify `process_student_input` glue logic still applies post-call
    (regression prevention, round counter).
- Update `tests/test_kele.py` (if exists) to confirm `--unified` flag plumbs through.

### Validation

The "smoke → mini → full" cadence applies just like before. Specifically:

1. **Smoke (n=5) on A3B + unified (no-think).** Compare to existing A3B
   no-think two-call smoke (already in `results/qwen35b-a3b-local-smoke-nothink/`).
   Pass criteria: state acc within ±5 pts; teacher response reads coherent on
   visual inspection of a couple of dialogues; no schema parse failures; wall
   clock confirms ≥25% speedup.
2. **Mini (n=25) on A3B + unified.** Same criteria, tighter signal on metrics.
3. **Full (n=681) on A3B + unified** if smoke + mini pass.
4. *(Optional)* same path for 27B + unified, if its 27B-quality lift transfers.

### Sequencing relative to other work

- This plan does **not block** the immediate "ship A3B no-think two-call full
  run" decision. The two-call infrastructure stays in place; unified is
  additive.
- It **does block** any decision to invest in `TEACHER_DISABLE_THINKING` —
  fusion subsumes it (fusion's single thinking pass replaces both).
- Compose-able with `--gt-consultant` mode from `QWEN_EVAL_FIX_PLAN.md`?
  Yes, but only in two-call mode (you can't replace just the consultant when
  they're fused). Keep `--gt-consultant` as a two-call-only flag.

---

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| State-response incoherence after Python glue override | Medium | Low (rare turns, dialogue still flows) | Option A in §6 + logging; revisit with Option B if rate >5% of turns |
| Mode collapse (model picks one safe state for everything) | Low | High (eval becomes meaningless) | Per-stage state acc check on smoke catches this immediately |
| Schema-rejection retry loop hangs eval | Low | Medium (lost dialogues) | Hard cap retries at 1 → fall back to two-call path → log + continue |
| Teacher-response degrades vs two-call (joint optimization is harder) | Medium | Medium | Smoke + mini compare ROUGE/BLEU before full run |
| max_tokens too tight for thinking + 3-field JSON | Low | Medium | Budget 16K (raise from 8K); thinking goes to `reasoning_content` so the JSON has full budget |
| Wall-clock not actually faster than two-call | Low | Low (we just don't ship it) | Smoke timing tells us in 10 min; no commitment |

---

## 9. Decision criteria (when to ship vs when to bail)

After A3B + unified smoke (n=5):

| Smoke result | Action |
|---|---|
| State acc within ±5 pts of two-call AND speedup ≥25% AND no parse failures | Promote to mini |
| State acc drops >5 pts | Examine dialogues; iterate prompt; re-smoke |
| Speedup <15% | Bail — fusion isn't helping enough to justify the architectural complexity |
| Schema parse failures >2/5 dialogues | Bail or harden schema; re-smoke |
| Teacher responses visibly worse on inspection | Bail |

After mini (n=25):

| Mini result | Action |
|---|---|
| Metrics tied or better than two-call AND speedup holds | Greenlight full run |
| Metrics worse | Compare per-stage; if isolated to specific stages, investigate prompt; otherwise stick with two-call |

---

## 10. Stretch goal — embedding the unified call in the eval orchestrators

Once `--unified` is plumbed through `kele.py`, exposing it through the existing
orchestrators (`eval_qwen27b.sh`, `eval_qwen35b_a3b.sh`) is a one-line change
per script. After fusion is the proven default, we can promote it to
default-on for the qwen27b-local/qwen35b-a3b-local configs and remove the
flag.

---

## 11. References

- `src/project/socratic_teaching_system.py` — the two-call system we're collapsing.
  - Consultant call: lines 169–382
  - State→action map: lines 50–86
  - Teacher call: lines 388–431
  - Glue logic: lines 433–521 (`process_student_input`)
- `src/project/kele.py` — eval pipeline; CLI; `create_system` factory.
- `docs/IMPROVEMENT_PLAN.md` #4 — original concept.
- `docs/QWEN27B_LOCAL_PLAN.md` Phase 3 — earlier preview of fusion.
- `docs/QWEN_LOCAL_EXPLORATION_LOG.md` — empirical motivation for this plan.
- `docs/QWEN_EVAL_FIX_PLAN.md` — `--gt-consultant` planning, intersects in §7.
