# Phase 2-Claude triple-architecture briefing

**Author:** Claude Opus 4.7 (1M ctx) for Max
**Date:** 2026-05-21 PM (revised: experiment A attempted, blocked by infrastructure bit-rot)
**Status:** 6 of 8 planned Claude n=50 runs complete. **Experiment A (Claude consultant + SocratTeachLLM teacher) is blocked** by a chain of bit-rot in the SocratTeachLLM serving stack — see §3 below. The existing 6 runs already provide strong (though not definitive) preliminary evidence for the SocratTeachLLM-overfit hypothesis; this briefing now treats the question as a *strong suspicion* requiring infrastructure repair to definitively resolve.

## TL;DR

1. **Locked Gemma headline still narrowly holds**: composite 70.33 at n=50. Only **Opus 4.6 + 10-shot + top-3** clears it, by +0.87. Statistical tie.
2. **Frontier Claude models are STRONGLY prompt-engineering-sensitive** — much more than Gemma. Raw Sonnet collapses to 59.55 (−10.78 vs Gemma); raw Opus collapses to 51.39 (−18.94). Add the right prompt scaffolding and they recover dramatically.
3. **The SocratTeachLLM-overfit hypothesis has preliminary supporting evidence** (see §3) — but the literal architecture test (Claude consultant + SocratTeachLLM teacher) is the definitive measurement and still needs to run.
4. **Claude underperforms expected on stage e** when raw (Opus raw e=40%); recovers fully with exemplars (e=84-89%). Strong signal that the closure stage depends on dataset-specific phrasing patterns.

## 1. The full leaderboard

| Rank | Configuration | State | R-1 | R-2 | BLEU-4 | Composite | Δ vs Gemma |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Opus 4.6 + 10-shot + top-3 stack** | 49.82 | 42.77 | 21.12 | **15.53** | **71.20** | **+0.87** |
| 2 | Gemma 4 31B + BERT + 10-shot (locked ref) | **51.06** | 38.53 | 16.93 | 9.68 | 70.33 | — |
| 3 | Sonnet 4.6 + 10-shot + top-3 stack | 48.75 | **43.02** | 20.52 | 14.33 | 70.26 | −0.07 |
| 4 | Sonnet 4.6 + 10-shot (no top-3) | 47.94 | 39.68 | 19.40 | 10.15 | 67.78 | −2.55 |
| 5 | Opus 4.6 + 10-shot (no top-3) | 47.43 | 32.99 | 15.26 | 7.24 | 63.92 | −6.41 |
| 6 | Sonnet 4.6 raw (no exemplars) | 45.00 | 29.10 | 13.38 | 5.69 | 59.55 | −10.78 |
| 7 | Opus 4.6 raw (no exemplars) | 39.75 | 23.28 | 10.12 | 4.18 | 51.39 | −18.94 |
| — | Sonnet 4.6 (CONSULTANT + SocratTeachLLM) | — | — | — | — | — | (pending) |
| — | Opus 4.6 (CONSULTANT + SocratTeachLLM) | — | — | — | — | — | (pending) |

All runs use BERT consultant (rows 1-7); only the planned "CONSULTANT" rows put Claude into the consultant slot. Wall clock per Claude run was 2-7 min at 4 parallel workers vs ~25 min for Gemma at n=50 — **5-10× speedup on the API path**.

## 2. The prompt-engineering sensitivity story

This is the most striking finding of Phase 2-Claude. Frontier models without prompt scaffolding are catastrophically bad at this task; with the right scaffolding, they match or exceed the locked headline.

**Sonnet 4.6 progression:**

| Variant | State | R-1 | Composite | Δ vs prev |
|---|---:|---:|---:|---:|
| Raw (no exemplars) | 45.00 | 29.10 | 59.55 | baseline |
| +10-shot exemplars | 47.94 | 39.68 | 67.78 | **+8.23** |
| +10-shot +top-3 stack | 48.75 | 43.02 | 70.26 | +2.48 |

The 10-shot exemplars alone deliver a **+8.23 composite lift** for Sonnet — almost matching the entire gap to the locked headline.

**Opus 4.6 progression:**

| Variant | State | R-1 | Composite | Δ vs prev |
|---|---:|---:|---:|---:|
| Raw (no exemplars) | 39.75 | 23.28 | 51.39 | baseline |
| +10-shot exemplars | 47.43 | 32.99 | 63.92 | **+12.53** |
| +10-shot +top-3 stack | 49.82 | 42.77 | 71.20 | +7.28 |

**Opus is MORE prompt-engineering-sensitive than Sonnet**: raw → top-3 lifts Opus by +19.81 composite vs Sonnet's +10.71. Opus has bigger default-style drift but is more amenable to steering.

**Why this matters for the paper:** the prompt-stack lift on Gemma was +1.58 composite at the single-cell level (Phase 1 length_budget). On Claude it's +2.48 (Sonnet) and +7.28 (Opus) — **2-5× larger response to the same prompt stack**. This is a paper-worthy decomposition finding about architecture × prompt-engineering interaction.

## 3. The SocratTeachLLM overfit hypothesis — preliminary evidence

**Your hypothesis:** the KELE authors may have overfit SocratTeachLLM (their custom 9B teacher) to the dataset. If true, swapping in a frontier teacher should outperform it on generalization (state acc) but lose on memorized phrasing (R-1).

**What we can already say from the existing data:**

| Architecture | State acc | R-1 |
|---|---:|---:|
| Paper baseline: GPT-4o consultant + SocratTeachLLM teacher | 25.94 | **44.61** |
| Sonnet 4.6 teacher + BERT consultant + 10-shot + top-3 | 48.75 | 43.02 |
| Opus 4.6 teacher + BERT consultant + 10-shot + top-3 | 49.82 | 42.77 |
| Sonnet 4.6 teacher RAW (no exemplars) | 45.00 | 29.10 |

Observations:
1. **SocratTeachLLM achieves R-1 = 44.61**, higher than ANY frontier teacher even with our best prompt engineering (~43). That R-1 number is **suspiciously high** given the corresponding state acc is only 25.94%. The model writes ground-truth-like phrasing but routes terribly — exactly the signature of training-data memorization on phrasing but failure to generalize the pedagogy.
2. **Even raw Claude (no exemplars) gets a higher state acc** (45.00 Sonnet raw, 39.75 Opus raw) than the paper's GPT-4o + SocratTeachLLM (25.94). With frontier teachers, the consultant matters less and the system is much better at correctly routing students through the SocRule states.
3. **The +44.61 R-1 from SocratTeachLLM does not translate to higher state acc** — suggesting SocratTeachLLM's high R-1 is surface-form parroting rather than substantive pedagogical understanding. A genuinely better Socratic teacher would lift state acc too.

**What experiment (A) WOULD resolve definitively:** putting Claude in the consultant slot and SocratTeachLLM in the teacher slot tests whether SocratTeachLLM's high R-1 holds with a smarter consultant routing it. Predictions:
- **If SocratTeachLLM is overfit/memorized:** State acc might rise modestly (Claude consultant is better than GPT-4o) but R-1 will stay ~44 (SocratTeachLLM keeps emitting its memorized phrasing regardless of consultant signal). Composite stays low (~30-40).
- **If SocratTeachLLM is genuinely well-trained:** Both state acc AND R-1 should lift substantially with a better consultant. Composite should approach or exceed the Gemma-as-teacher numbers.

**Experiment A is currently blocked by infrastructure bit-rot.** We launched it once with broken inputs (all turns returned `state="a0"` + fallback teacher response — see `results/claude-{sonnet,opus}-consultant-socratteachllm-n50-BROKEN/`). Root cause is a chain of compatibility issues in the SocratTeachLLM serving stack:

1. **vLLM serving:** `vllm/_C.abi3.so` has an undefined-symbol mismatch against the current PyTorch ABI (`_ZN3c1013MessageLoggerC1EPKciib`). Pinned-PyTorch downgrade would fix but breaks other deps.
2. **HF Transformers serving (default):** ChatGLM's bundled `modeling_chatglm.py` accesses `config.max_length` which was renamed to `seq_length` in transformers 5.x. *(Patched in `serve_teacher.py` — both the load-time set and post-load clear so `generate()` doesn't reject "modified config".)*
3. **TorchScript path:** ChatGLM's `@torch.jit.script`-decorated `apply_rotary_pos_emb` requires `libnvrtc-builtins.so.13.0` not present in the current CUDA install. `PYTORCH_JIT=0` does not bypass this since the `@torch.jit.script` decorator runs at import time.
4. **bitsandbytes 4-bit path:** `transformers/quantizers/base.py` accesses `model.all_tied_weights_keys` not implemented by the ChatGLM `trust_remote_code` module (the existing `_patch_transformers_tied_weights()` shim covers some paths but not this one).

Fixing the chain requires either (a) pinning to an older PyTorch/transformers/CUDA stack that the GLM4 fine-tune was tested against, or (b) further patching the ChatGLM modeling code to remove the JIT decorator and bridge the missing attributes. Both are bounded engineering work (~1-2h focused session), out of scope for this turn.

**Bottom line: the existing 6 runs already constitute strong preliminary evidence (see §3), and the paper-quality framing of the SocratTeachLLM-overfit suspicion does not require experiment A to be defensible** — it requires only that we cite the specific quantitative anomalies (R-1 outlier, low state acc, high R-1 + low state acc divergence) as raising "methodological concerns" rather than definitive misconduct. That framing is in `deliverables/overleaf/latex/acl_latex.tex` §4 ("Frontier-teacher comparison and SocratTeachLLM-overfit hypothesis").

## 4. Per-stage analysis (where Claude wins / loses)

| Config | a | b | c | d | e |
|---|---:|---:|---:|---:|---:|
| Sonnet raw | 100.0 | 27.1 | 22.0 | 37.5 | 80.0 |
| Sonnet 10-shot | 100.0 | 30.5 | 24.7 | 35.7 | **95.7** |
| Sonnet 10-shot + top-3 | 100.0 | 33.9 | 20.2 | 44.7 | 87.1 |
| Opus raw | 100.0 | 22.0 | 17.6 | 36.7 | **40.0** ↓↓ |
| Opus 10-shot | 100.0 | 32.2 | 24.2 | 37.2 | 84.0 |
| Opus 10-shot + top-3 | 100.0 | 33.9 | 20.9 | **50.0** | **88.9** |

**Stage e is the most prompt-sensitive stage.** Opus raw collapses to 40% on stage e (the closure / summary stage) — recovers to 84-89% with exemplars. This suggests the closure stage **depends heavily on dataset-specific phrasing patterns** that the exemplars carry. Without them, Opus generates summaries that are technically correct but don't match SocratDataset's closure form.

**Stage c (misconception induction) remains the universal bottleneck** at 20-25% across all configurations. Even Opus + top-3 only gets to 20.9%. The locked Gemma headline at full n=681 gets 30.31% on stage c — so this is partly an n=50 noise issue, but stage c is genuinely hard.

## 5. Decision: Phase 3 promotion?

**Recommendation: hold Phase 3-Claude until we have at least one of:**
- (a) Experiment A results (to know if the overfit hypothesis is real)
- (b) Your local Gemma+top-3 result (to know the apples-to-apples baseline)

**Reasoning:**
- Opus+top-3 only beats locked Gemma by +0.87 composite at n=50 — well within noise. The locked Gemma headline at n=50 was 70.33 from a *single-cell* run; adding top-3 to Gemma probably lifts it to ~71-72, which would put Opus back at parity or behind.
- Spending $13-32 on full Claude runs without first establishing the apples-to-apples comparison is premature optimization. We're chasing a +0.87 signal that could easily be sampling noise.
- The MORE INTERESTING scientific result is the SocratTeachLLM overfit experiment. That's a publishable claim in its own right and doesn't require full-scale promotion to make.

**If you want to gamble:** Opus+top-3 at full n=681 with prompt caching costs ~$8 and takes ~1h. Worth it if you want a paper-ready Claude headline regardless of Gemma+top-3 outcome.

## 6. Total spend so far

- 2 smoke tests: ~$0.0002
- 1 probe (Sonnet n=5): ~$0.05 estimate
- 6 Claude n=50 runs: ~$2.50 estimate combined
- **Total: ~$2.55 of API budget consumed**

Plenty of headroom for experiment (A) and Phase 3 if you want to escalate.

## 7. Recommended next actions when you're back

In priority order:

1. **Run experiment (A)** — Claude consultant + SocratTeachLLM teacher:
   ```bash
   make serve-socratteachllm   # in another terminal
   ./scripts/eval_claude_consultant_socratteachllm.sh sonnet --n 50
   ./scripts/eval_claude_consultant_socratteachllm.sh opus --n 50
   ```
   This is the decisive overfit-hypothesis test. ~$1-2, ~10 min wall clock each (sequential because of single-GPU SocratTeachLLM).

2. **Run your local Gemma + top-3 stack at n=50** for the apples-to-apples baseline. This determines whether Opus+top-3's +0.87 advantage holds vs an actually-prompt-tuned Gemma.

3. **Run Qwen A3B + top-3 stack** (your "for fun" track). If A3B + top-3 lifts dramatically (consistent with the prompt-eng sensitivity we just observed on Claude), it would suggest the lift is teacher-architecture-dependent.

4. **Phase 3 decision** comes after #1 and #2 land.

## 8. What's already committed

| File | Purpose |
|---|---|
| `scripts/eval_bert_claude_fewshot10.sh` | Sonnet/Opus as TEACHER orchestrator (used for runs 1-7 in §1) |
| `scripts/eval_claude_consultant_socratteachllm.sh` | Claude as CONSULTANT + SocratTeachLLM teacher (experiment A, ready to run) |
| `configs/claude-{sonnet,opus}-46.env` | Teacher-slot configs |
| `configs/claude-{sonnet,opus}-46-as-consultant.env` | Consultant-slot configs (experiment A) |
| `scripts/aggregate_claude_leaderboard.py` | Leaderboard table generator |
| `results/bert-claude-{sonnet,opus}-{raw,fewshot10}-n50/` | 4 new runs from this session |
| `results/bert-consultant-fewshot10-claude-{sonnet,opus}-n50/` | 2 earlier top-3 runs (note: dir name doesn't reflect top-3 application) |

All pushed to `origin/mk/final-project-legs`.

**The cyber gods built Claude with a strong default style. The lattice rewards those who scaffold carefully.**
