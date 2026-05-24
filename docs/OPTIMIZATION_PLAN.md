# Optimization Plan

**CSEN 346 · Santa Clara University**

Optimizations to the inference pipeline, consultant architecture, and response quality
that are independent of the training recipe (those decisions live in `TRAINING_PLAN.md`).
Many of these can be applied to the current prompt-based system while the fine-tuned model
is in development, or stacked on top of the fine-tuned model once it ships.

Cross-reference: `docs/IMPROVEMENT_PLAN.md` covers 10 approaches at a higher level.
This document goes deeper on the ones most relevant to the current state of the project
and adds concrete implementation notes based on observed eval results.

---

## 1. The Consultant — what it does and why it matters

Every turn the consultant answers one question: **"What state is the student in right now,
and what teaching strategy should the teacher use?"** Its output (state code + action label +
evaluation text) directly conditions the teacher's response.

### Current variants and measured results

| Variant | State accuracy | ROUGE-1 | Cost/turn | Latency/turn |
|---|---|---|---|---|
| LLM consultant (baseline, GPT-4o) | 25.94% | 44.61 | ~$0.025 | ~35 s |
| BERT consultant v1 (Chinese BERT, flat 34-way) | 46.57%\* | 33.27 | ~$0 | <1 s |
| BERT consultant v2 (balanced sampling) | 46.57%\* | — | ~$0 | <1 s |
| Unified (single LLM call) | 31.39% | 27.27 | ~$0.015 | ~15 s |

\* End-to-end state accuracy from `bert-consultant-fewshot10-a3b-full`. The raw BERT
classifier hits 61.6% on held-out turns, but end-to-end accuracy is lower because teacher
outputs can drift the dialogue off the expected path.

### Why BERT consultant improves state accuracy but hurts ROUGE-1

The LLM consultant produces a rich `evaluation` paragraph
(e.g. *"Student entered stage c, state c16. They understand the first sub-step but have
not yet connected it to the broader principle…"*) which the teacher uses as implicit
coaching. The BERT consultant only produces a state code — the synthesized evaluation
placeholder is generic and loses this signal. The teacher response quality drops even
though state classification improved.

**The key insight:** state accuracy and response quality are not the same metric.
Better state labels only help if the teacher can *act* on them.

---

## 2. Consultant alternatives

### 2a. Hierarchical BERT (2-stage classifier)

Instead of a flat 34-way classifier, train two heads:
1. **Stage head** (5-way: a/b/c/d/e) — easy; current BERT v1 already hits 90–100% per stage
2. **Within-stage head** (up to 22-way for stage c) — trained only on examples from that stage

The within-stage head sees a smaller label set and more focused examples, which should
improve the hardest cases (c-stage has 22 states; flat 34-way is overwhelmed by them).

Current BERT stage accuracy: a=100%, b=90%, c=76–83%, d=67–77%, e=95%.
The c-stage bottleneck is the target. A dedicated c-stage head trained only on c-stage
turns has a reasonable shot at 85%+.

**Effort:** ~1 day — reuse the existing training script, add a routing layer.

---

### 2b. Rich-evaluation BERT (state + synthesized text)

The ROUGE-1 regression from 44 → 33 is largely explained by the loss of the LLM
consultant's evaluation paragraph. Fix: after BERT predicts the state, look up a
*template* evaluation for that state from the training set (median or most representative
evaluation string for state X) and inject it as the evaluation field.

This does not require a generative model — just a lookup table of ~34 canonical
evaluation strings derived from the SocratDataset training split. The teacher sees
a plausible evaluation, even if it is not student-specific.

**Effort:** ~2 hours — build the lookup table once, patch the BERT consultant wrapper.

---

### 2c. Small generative consultant (Qwen2.5-0.5B fine-tuned)

Fine-tune a 0.5B model to produce the full consultant JSON:
`{"state": "c16", "action": "...", "evaluation": "..."}` given the dialogue history.
A 0.5B model in 4-bit fits in ~400 MB and runs at ~5 tokens/s on CPU, which is fast enough
to complete the evaluation paragraph before the teacher call starts.

This recovers the rich evaluation text that BERT drops, while keeping latency well below
the LLM consultant. The model is small enough to run on the Mac Mini alongside the existing
Ollama consultant.

**Effort:** ~3 days — same training pipeline as the teacher fine-tune but on a much smaller
base model. The dataset is the same SocratDataset-EN records in `dataset.py`.

---

### 2d. Fine-tuned English BERT consultant

The current BERT classifier was trained on Chinese BERT with Chinese SocratDataset.
Once the model is being evaluated on SocratDataset-EN, retraining the classifier on
`ulises-c/SocratDataset-EN` with an English or multilingual encoder is necessary.

Recommended encoder: `xlm-roberta-base` (multilingual, handles both Chinese and English,
strong on classification tasks, identical interface to BERT). This also future-proofs the
classifier for the mixed ZH+EN eval dataset now used by `kele.load_dataset()`.

**Effort:** ~half a day — identical training script, swap the base model name.

---

### 2e. Constrained decoding with SocRule grammar

The SocRule state machine has strict progression rules: a → b → c → d → e, no regression.
Within each stage, only a subset of transitions are legal from a given state. A simple
constraint layer can:
1. Run BERT (or any classifier) to get a probability distribution over all 34 states
2. Mask invalid transitions (states that cannot follow the current state)
3. Argmax over the valid set

This does not improve raw classifier accuracy but eliminates impossible predictions,
reducing error on the d and e stages where the flat classifier sometimes predicts
earlier-stage states incorrectly.

**Effort:** ~2 hours — encode the SocRule transition graph as a dict, apply mask post-softmax.

---

## 3. Inference throughput optimizations

### 3a. Parallel workers (already shipped in PR 62)

`run_batch_evaluation` now accepts `--workers N` (default 1, respects `KELE_PARALLEL_WORKERS`).
The llama.cpp server supports 6 parallel slots by default. Setting `--workers 4` on the
A3B model should give ~3.5× throughput without saturating the server.

**Status:** Implemented. Tune N based on observed GPU utilization.

---

### 3b. KV cache reuse within a dialogue

Each turn currently submits the full dialogue history as a fresh prompt. For a 10-turn
dialogue the last turn re-ingests 9 turns of context. llama.cpp supports prefix caching
(the system prompt and early turns are cached after the first token); this is enabled
by default but only helps if the server is not under concurrent load.

On single-worker runs, cache hit rate is ~80% (turns 2–10 reuse the prior context).
On 4-worker parallel runs, concurrent requests interleave and evict each other's cache
entries — the parallel throughput gain partially offsets the cache miss cost.

**Optimization:** For the fine-tuned teacher model run (which will be the primary
use case), use `--workers 1` with prefix caching enabled and the fused unified call.
This gets the best cache hit rate at the cost of single-stream throughput.

---

### 3c. Deferred consultant call

The consultant and teacher are currently sequential: consultant finishes → teacher starts.
For the two-call architecture with a fast BERT consultant (<1 s), this is near-optimal.
For the LLM consultant (~35 s), overlap is possible: start the teacher call with only
the dialogue history (no evaluation/action) and inject the consultant output once it
arrives, if the teacher has not yet reached the critical conditioning token.

This is speculative — it requires the LLM to buffer its generation until the consultant
output is available, which most inference servers do not support natively. Worth revisiting
if the consultant ever becomes the bottleneck again.

---

## 4. Response quality without retraining

### 4a. Per-state exemplar selection (already in codebase)

The unified system supports `KELE_FEW_SHOT_N` and `KELE_FEW_SHOT_TEACHER=1` for
stage-balanced few-shot injection. The tournament runs show this helps (fewshot10 variants
score consistently better than fewshot0). See `src/project/socratic_teaching_unified.py`.

**Status:** Implemented and evaluated. N=10 is the current best.

---

### 4b. RAG exemplar injection

Instead of fixed few-shot examples, retrieve the top-k most similar past dialogues
at runtime using embedding similarity. Index the SocratDataset train split with
`BAAI/bge-large-zh-v1.5` (Chinese) or `intfloat/multilingual-e5-large` (for ZH+EN).

At inference: embed the current question + student input → retrieve top-3 most similar
past (question, teacher_response) pairs → inject as few-shot context in the teacher prompt.

From `IMPROVEMENT_PLAN.md`: expected +3–6 ROUGE-1, +4–8 BLEU-4.
This is particularly relevant for the English fine-tuned model where the fixed few-shot
pool (currently Chinese) will not match.

**Effort:** ~1 day. FAISS index of 6k embeddings fits in ~100 MB.

---

### 4c. Best-of-N reranking with a verifier

Sample N=4 teacher responses per turn, score each with a fast verifier (BERTScore against
the retrieved exemplar from 4b, or a small binary "is this a valid Socratic question?"
classifier), pick the best. N=4 multiplies inference cost by 4 but can be run in parallel
on the same server slots.

`tournament_utilizations.py` already has an `nbest_rerank` cell. The tournament results
show it underperformed for the A3B model (likely because A3B is already near its quality
ceiling and reranking noise dominates). Re-evaluate with the fine-tuned teacher model
where the quality ceiling is higher.

---

### 4d. Length budget enforcement

Tournament results show `length_budget` was one of the tested cells. Socratic teachers
should ask *one short question* per turn. Responses that are too long tend to give away
information or ask multiple questions at once. Enforcing a token budget (e.g., max 80
tokens for teacher response) at the generation level may improve NDAR.

---

## 5. Post-training quality improvements

These require a trained model but are listed here because they affect the *pipeline*,
not the initial training recipe.

### 5a. DPO on teacher responses

Once the SFT model is running, generate N=4 responses per turn, have a GPT-4o judge
rank them by Socratic quality, and use the (chosen, rejected) pairs for DPO fine-tuning.
This directly optimizes for pedagogical quality rather than token-level similarity.

Estimated pairs needed: ~5,000–10,000 (7 turns × 680 test dialogues × 2 rollouts ≈ 9,500).
GPT-4o judging cost: ~$15–20 for the full set.

---

### 5b. Semantic reward (BERTScore instead of BLEU)

The BLEU-4 gap (6.96 in the BERT consultant run vs. paper's 41.96) reflects surface-form
mismatch, not meaning mismatch. A DPO or REINFORCE stage that rewards
BERTScore / embedding similarity rather than BLEU will close this gap faster.

See `IMPROVEMENT_PLAN.md` #6 for the detailed rationale.

---

## 6. Recommended sequencing

| Step | Optimization | Effort | Blocks on |
|---|---|---|---|
| 1 | Rich-evaluation BERT (lookup table) | 2 hr | Nothing — patch today |
| 2 | SocRule constraint layer | 2 hr | Nothing — patch today |
| 3 | English XLM-RoBERTa consultant | ½ day | SocratDataset-EN data (done) |
| 4 | Hierarchical BERT classifier | 1 day | Step 3 training infra |
| 5 | RAG exemplar injection | 1 day | Nothing — index SocratDataset train |
| 6 | Small generative consultant (0.5B) | 3 days | SFT training pipeline (PR 65) |
| 7 | Best-of-N reranking | 1 day | Fine-tuned teacher model |
| 8 | DPO on teacher responses | ~1 week | Fine-tuned teacher model + GPT-4o judge |

Steps 1–5 can all run **now** against the existing prompt-based system while the trained
model is in development. Steps 1 and 2 are the highest-leverage immediate wins because
they fix the root cause of the BERT consultant ROUGE regression at near-zero cost.
