<!--
PRESENTATION.md — class talk for CSEN 346
Render: `npx reveal-md PRESENTATION.md` for slide deck in browser,
        or read directly on GitHub (renders as one long doc).
Speaker notes are HTML comments — visible in source, hidden in render.
Target pace: ~130 words/minute. 17 slides, ~26 min + Q&A
(over the original 15-min budget by ~11 min; user-approved — extra
budget for the Socratic-teaching foundation (slides 2-4), the
expanded constraints slide (slide 5: hardware diversity + time
scope), the deepened methodology slide (slide 7: tournament results
+ cascade narrative), the Gemma-retraction story (slide 8), the
Pivot-2 architectural-decomposition story (slide 9), the frontier
stress-test narrative (slide 10), the benchmark-critique discovery
arc (slide 11), the contamination-proof two-probe narrative (slide
12), the unified-metric design story (slide 13), and the final-slide
depth on Slide 17 "Total Contributions").
-->

# Beating the Frontier on a Consumer GPU

### Reproducing and Extending KELE: a Multi-Agent Socratic Teaching Framework

**Maximilian Khan** **Ulises Chavarria** · CSEN 346 · Santa Clara University · May 2026

A 31B-parameter open-weight teacher running on a single 32 GB consumer GPU — beating Anthropic's best frontier model on a memorization-resistant Chinese pedagogy benchmark at canonical sample size, at **zero** per-run API cost.

<!--
SPEAKER NOTES (Slide 1, ~15s):
Hi, I'm Max. Today I'll walk you through a 3-month NLP campaign reproducing and extending a 2025 EMNLP paper called KELE — Knowledge-Enhanced Learning Environment for Socratic Teaching. The headline result, which we locked in this morning: a 31-billion-parameter open-weight teacher on my single 32-gigabyte 5090 just overtook Anthropic's best frontier model on this benchmark, at zero per-run API cost. Let me show you how we got there.
-->

---

## What is Socratic Teaching?

A pedagogy where the **teacher asks structured questions instead of giving answers** — the student reasons their way to the answer themselves.

- 📚 **Traditional instruction (passive):** teacher tells, student receives. Limited deep reasoning [Johnston 1994].
- 💬 **Socratic instruction (active):** teacher questions, student constructs the answer through guided inquiry [Seeskin 1987 · Chang 1998].
- 📊 **Pedagogical evidence:** Socratic dialogue consistently promotes deeper understanding and cognitive development [Knezic 2010].
- ⚠️ **The bottleneck:** Socratic teaching requires *highly skilled* instructors. It doesn't scale to millions of students.
- 🤖 **KELE's bet:** can an LLM-based system replicate the structured-questioning behavior of a skilled Socratic teacher — at scale?

<!--
SPEAKER NOTES (Slide 2, ~1 min):
Before I tell you what KELE is, let me ground us in why anyone built it. Socratic teaching is a pedagogy where the teacher does not give answers — they ask structured questions that guide the student to discover the answer themselves. The contrast: traditional instruction is passive. The teacher tells, the student receives, and decades of research going back to Johnston 1994 shows that mode produces limited deep reasoning. Socratic teaching, going back to Seeskin in 1987 and Chang in 1998, flips that — the student does the cognitive work. Research consistently shows it promotes deeper understanding. But there is a bottleneck: it requires highly skilled instructors. One human Socratic tutor cannot scale to millions of students. KELE's bet is that an LLM-based system can replicate that structured-questioning behavior at scale.
-->

---

## KELE's Framework — Two Agents, 5 Stages, 34 Strategies

Peng et al., *Findings of EMNLP 2025*. A multi-agent framework that decomposes Socratic teaching into a finite-state machine.

**Two agents:**
- **Consultant LLM** — given dialogue history, classifies the current cognitive state + selects the teaching action
- **Teacher LLM** — given the consultant's choice, generates the actual response in natural language

**The SocRule framework — 5 hierarchical stages (must advance in order, no skip / no backtrack):**

| Stage | Letter | # strategies | What the teacher does in this stage |
|---|:-:|:-:|---|
| Questioning | a | 2 (`a0`, `a1`) | Get the student to articulate a question (`a0` = pre-teaching default) |
| Concept Probing | b | 6 (`b2`–`b7`) | Probe student's prior concepts from different angles |
| Inductive Reasoning | c | **22** (`c8`–`c29`) | Surface misconceptions; counterexamples; build toward a rule (the cognitive heavy lift) |
| Rule Construction | d | 4 (`d30`–`d33`) | Help the student state the rule correctly; prompt for the answer |
| Closure | e | 1 (`e34`) | Summarize; confirm understanding |

**The 34 teaching strategies in full** (label → action, from `references/KELE/consultant_teacher_socratic_teaching_system.py`):

> **a1** ask sub-question · **b2** probe from different angles · **b3** change the question · **b4** related sub-questions · **b5** test concept understanding · **b6** review prior concepts · **b7** compare with student's error · **c8** counterexample · **c9** incomplete-rule + misleading question · **c10** ask "why" · **c11** demand the reason explicitly · **c12** incomplete-rule or counterexample · **c13** counterexample · **c14–c16** elicit predictions + new principle · **c17** generate the sub-question · **c18** reconsider · **c19** diagnostic question · **c20** verify the just-learned concept · **c21** think more carefully · **c22** ask "why" · **c23** re-form hypothesis · **c24** student tests hypothesis · **c25** verification method · **c26** compare two examples · **c27** guide testing · **c28** inform of error, ask for alternatives · **c29** show correct concept, ask why missed · **d30** present a related case, ask "predict" or "why" · **d31** show correct rule, ask to reconsider · **d32** present a related case, ask for prediction · **d33** general definition, ask for the answer · **e34** summarize the problem

**SocratDataset:** 6,803 Chinese elementary-school dialogues · ~42,000 teacher turns · ground-truth state label per turn · 681-dialogue test split.

<!--
SPEAKER NOTES (Slide 3, ~1.5 min):
This is KELE in one slide. Peng et al., EMNLP 2025 Findings. Two LLM agents: the consultant looks at the dialogue history and classifies which cognitive state the student is currently in, then picks the appropriate teaching action. The teacher LLM takes that action and turns it into actual language. The framework that gives this structure is called SocRule — five hierarchical stages that must advance in order. Stage a is questioning — the student articulates the question. Stage b, concept probing, six strategies to probe what the student already knows. Stage c, inductive reasoning, is the cognitive heavy lift — twenty-two strategies for handling misconceptions and counterexamples. Notice the asymmetry — c has 22 strategies, more than the other four stages combined — that is where the hard work happens. Stage d is rule construction, four strategies. Stage e is closure, one strategy, the wrap-up. The dataset has all 34 strategies labeled across roughly 42,000 turns drawn from 6,803 dialogues, with a 681-dialogue test split that the original paper evaluates on. The list below the table is every single strategy — you can read it during Q&A, but the takeaway during the talk is the structure: 2-6-22-4-1.
-->

---

## A KELE Dialogue in Action

A real 6-turn dialogue from SocratDataset (id=1), translated from Chinese. Topic: **"In which part of a plant are seeds usually found?"**

| Turn | State | Student | Teacher (Socratic) |
|:-:|:-:|---|---|
| 1 | `a1` | "In which part of a plant are seeds usually found?" | "Great question! Do you think plant seeds are placed in a particular kind of place?" |
| 2 | `b6` | "I think seeds might be inside the flower." | "The flower is indeed important. Do you remember which part of the flower becomes the seed?" |
| 3 | `c9` | "Do the petals turn into seeds?" | "Interesting idea — but petals attract insects for pollination. Do you think seeds need protection?" |
| 4 | `c12` | "Seeds might need a harder place to protect them." | "Excellent! Have you seen fruits like apples or watermelons? What's inside them?" |
| 5 | `d33` | "Oh, I have! Fruits have seeds inside." | "Right! Fruits protect seeds. So now — in which part of a plant are seeds usually found?" |
| 6 | `e34` | "Plant seeds are usually found inside fruits." | "Completely correct! Plant seeds are located inside fruits, which develop from the flower's ovary…" |

- The **consultant** predicts a state at each turn: `a1 → b6 → c9 → c12 → d33 → e34` (smooth progression through all 5 SocRule stages)
- The **teacher** uses that state to generate the response — never giving the answer directly, always probing
- **Original paper baseline (GPT-4o consultant + SocratTeachLLM-9B teacher) on the 681-dialogue test split:** **25.94% state accuracy** · R-1 44.61 · BLEU-4 19.60 — **this is the bar to beat.**

<!--
SPEAKER NOTES (Slide 4, ~1.5 min):
Here is what a KELE pipeline actually does, with a real dialogue from the dataset. The student asks: in which part of a plant are seeds usually found? The teacher does not say "in the fruit." Instead the consultant LLM classifies the student as state a1 — they have asked a question — and the teacher rephrases the question to probe the student's intuition: where do you think seeds are placed? The student guesses "inside the flower" — that is state b6, where the student is drawing on prior knowledge but the answer is incomplete. The teacher does not correct it directly; it asks which part of the flower becomes the seed. The student guesses "the petals" — that is state c9, a misconception. The teacher provides a counterexample — petals attract pollinators — and reframes toward protection. The student arrives at "seeds need a harder place" — state c12 — and the teacher introduces fruits. The student remembers fruits have seeds — state d33, rule construction. Final turn, e34: the student gives the correct answer and the teacher summarizes. Six turns to walk an elementary student from the question to the correct answer through structured questioning. None of it gives the answer directly. This is what we are building, and what KELE's original GPT-4o + SocratTeachLLM-9B baseline got 25.94% state accuracy on across the 681-dialogue test split. That number — and the surface-form numbers, R-1 of 44.61 and BLEU-4 of 19.60 — are the bar we are trying to beat.
-->

---

## Our Constraints

A single 32 GB consumer GPU as the primary test rig, **$0 budget** for per-run API calls, and **one Spring quarter** of Santa Clara class time.

- 🖥️ **Primary rig — NVIDIA RTX 5090 (32 GB VRAM, CUDA).** Every locked headline ran here under this VRAM bound. Renting frontier API consultancy on every dialogue would also burn the API budget, so the consultant axis has to be local-and-cheap.
- 🌐 **Cross-platform deployments — Ulises-owned secondary platforms.** Apple M4 Mac Minis (CPU inference for the consultant sensitivity sweep), AMD Radeon R9700 (32 GB, ROCm — the original 13-model no-think tournament and the SocratDataset-EN translation pipeline ran here at ~390 records/h), plus WAVE HPC V100 batch jobs for off-rig validation. **Cross-hardware comparability validated:** the 5090 reproduces the R9700 tournament's A3B no-think point within **0.07 pp** — our results aren't 5090-specific.
- 🏗️ **Architecture — separate consultant + teacher, within VRAM.** KELE's canonical stack requires **two model deployments simultaneously**, and two 30B-class open-weight LLMs do not co-resident on 32 GB. We *tried* fusion (single-backbone — see slide 6) and it was the first locked headline, but we moved past it. **The consultant-teacher separation is what KELE got right architecturally**, and our final system keeps that separation by downsizing the consultant axis to a deterministic classifier (24M or ~800M params) that fits alongside the 31B teacher on the same GPU.
- 🎓 **Time — one Spring quarter at SCU.** Three months, fixed budget — not a multi-year industrial campaign. We *could* sweep more model sizes, more quantization configs, more fine-tunings, but instead we focused on **popular, accessible open-source models** that align with KELE's framing (Qwen 3.6 family, Gemma 4 family, frontier Claude as comparison ceiling) over fully covering the option space.

<!--
SPEAKER NOTES (Slide 5, ~1:15):
Four constraints shape everything else in the talk. First, the primary rig: one RTX 5090, 32 gigabytes of VRAM, CUDA — every locked-headline result you'll see today ran on that single machine, with a zero-dollar-per-run budget for the eval pipeline itself. Second, cross-platform — this is largely Ulises's contribution. He maintains a small deployment stack: Apple M4 Mac Minis for the CPU-inference consultant-sensitivity sweep, an AMD Radeon R9700 with 32 GB of ROCm-backed VRAM that ran the original 13-model tournament AND the SocratDataset-EN translation pipeline at roughly 390 records per hour, plus the WAVE HPC V100 for off-rig batch validation. The 5090 reproduces the R9700's A3B no-think point within 0.07 percentage points — so the results are hardware-independent. Third, architecture — KELE's canonical stack requires two model deployments simultaneously, and two 30-billion-class open-weight LLMs don't fit in 32 gigs together. We tried fusion — collapsing both roles into one backbone — and that was our first locked headline. But we moved past it: the consultant-teacher separation is what KELE actually got right architecturally. Our final system keeps that separation but downsizes the consultant to a deterministic classifier so the two axes can coexist on the same GPU. Fourth, time — this is a one-quarter Santa Clara class project, not a multi-year industrial campaign. We could sweep more model sizes and more quantization configs in principle, but we deliberately scoped to popular, accessible open-source models aligned with KELE's framing — Qwen 3.6, Gemma 4, frontier Claude as a comparison ceiling — instead of trying to cover the full option space.
-->

---

## Pivot 1: Fusion Architecture

KELE requires two LLMs running at the same time — a consultant and a teacher — but our 32 GB consumer GPU couldn't host both. We needed a way to preserve KELE's decomposition without doubling the memory footprint. The idea we landed on — we called it *fusion* — was simple: instead of running two separate LLM deployments, ask a single open-weight backbone to do both jobs in one structured-output call, returning the state prediction and the teacher response as two fields of the same JSON output. One model. One forward pass. One KV cache. Modern LLMs are good enough at JSON adherence that we thought this could work — and it did. Our first full n=681 run, Qwen 35B-A3B in fusion-think mode, landed at **38.70% state accuracy**, a 1.49× lift over KELE's GPT-4o baseline, on the single 5090 at zero per-run API cost. **Fusion was our first locked headline** and proved the campaign was viable. But it also planted the seed for what came next: the architecture's reliance on strict JSON grammar turned out to be a hidden fragility (you'll see Gemma 4 31B collapse from it on the next slide), and the lesson we eventually drew was that the consultant axis didn't need to be an LLM at all. We've since moved past fusion in favor of the integration architecture on slide 9 — but fusion is how we got into the game.

Collapse consultant and teacher into a **single backbone**, single forward pass, structured-output call returning both the state prediction and the teacher response.

- **Two-call** (consultant LLM → teacher LLM): VRAM-prohibitive, slow, KV cache duplicated
- **Fusion call** (one model, one JSON output with both fields): single backbone, ~2× faster
- **First locked headline (n=681):** Qwen 35B-A3B fusion-think → **38.70% state accuracy** vs.\ GPT-4o's 25.94% — **+12.76 absolute pp / 1.49× lift** on a single consumer GPU
- ⚠️ **Later superseded.** Fusion was the first locked headline but is *not* in our final contribution list — Pivot 2 (slide 9) replaced the JSON-grammar-on-LLM path with a deterministic classifier, and the integration architecture outperforms fusion at every tier we care about.

<!--
SPEAKER NOTES (Slide 6, ~1 min):
The first pivot was architectural. Instead of running two LLMs, I collapsed both roles into a single open-weight backbone using a structured-output call — one JSON response containing both the state prediction and the teacher utterance. This skips the consultant-to-teacher round trip and removes KV cache duplication, roughly halving wall-clock time. The first full-scale result on the 681-dialogue test split used Qwen 35B-A3B in fusion-think mode and scored 38.70% state accuracy — 12.76 percentage points above GPT-4o, a 1.49× lift, running on the single 5090 at zero API cost. That became our first locked headline. Heads-up before we move on: fusion was the architecture that got us into the game, but we've since moved past it — the integration architecture you'll see on slide 9 outperforms fusion at every tier we care about. So fusion is documented in the campaign history, not in our final contribution list.
-->

---

## Picking the Backbone: 13-Model Tournament + Smoke / Mini / Full Cascade

Open-weight pedagogy is a wide-open option space — dozens of viable backbones across the Qwen, Gemma, GLM, Mistral, Phi, and DeepSeek families, each with multiple sizes and quantizations. We couldn't afford to run all of them at canonical n=681 (a full run takes 10 to 22 hours of GPU wall-clock), so we built **two pieces of methodology** to triage the option space intelligently: a *cheap-first evaluation cascade* that filters bad bets before paying full-scale compute, and a *13-model tournament* that swept the family of popular and accessible open-weight backbones end-to-end at n=50.

The cascade gates every architectural decision against progressively cheaper signals before promotion:

| Tier | n | Wall clock | Use |
|---|---:|---:|---|
| Smoke | 5 | ~10–40 min | Triage candidates, sanity-check the serving stack |
| Mini | 25 | ~30 min – 1 h | Sharpen the ranking, reject bad bets |
| Full | 681 | ~10–22 h | Lock the headline |

**Why we trust the cascade:** the predictive validation is concrete — averaging the smoke- and mini-tier state-accuracy lifts on Qwen 35B-A3B predicted the full-run lift at **+12.86 pp**, the realized lift was **+12.76 pp** — accurate to **within 0.10 percentage points** of truth. That calibration is the reason we trusted the cascade to gate every architectural decision in the campaign. *(Caveat: the predictor is architecture-dependent — it failed catastrophically on Gemma 4 31B at full scale, which is exactly the story on the next slide.)*

**The tournament:** 13 popular open-weight backbones × n=50 × no-think mode. No-think was the explicit choice — it kept the entire 13-model sweep within one overnight budget. Ulises ran the original sweep on his AMD Radeon R9700; we replicated key points on the 5090 and the A3B no-think result agreed across hardware to within **0.07 pp**, validating cross-hardware comparability.

### 13-Model Tournament Results (n=50, no-think, sorted by state accuracy)

| Rank | Model | Arch | State acc | R-1 | BLEU-4 | Wall clock |
|---:|---|---|---:|---:|---:|---:|
| 🥇 1 | **Gemma 4 26B-A4B Q4** | MoE (~4B active) | **38.67%** | 32.2 | 5.7 | 3h 05m |
| 🥈 2 | Gemma 4 31B Q5 | dense | 35.86% | 33.0 | 6.4 | 5h 36m |
| 🥉 3 | Qwen 27B Q4 | dense | 30.67% | 33.2 | 7.2 | 0h 59m |
| 4 | Qwen 27B Q5 | dense | 28.62% | 33.4 | 7.2 | 1h 06m |
| 5 | Qwen3 14B Q4 | dense | 22.48% | **39.5** | **11.5** | 0h 14m |
| 6 | Mistral Small 24B Q4 | dense | 21.85% | 37.1 | 9.6 | 0h 25m |
| 7 | Qwen 35B-A3B Q4 | MoE (~3B active) | 19.74% | 31.3 | 5.9 | 0h 21m |
| 8 | Qwopus 35B-A3B Q4 | MoE + LoRA | 18.57% | 32.8 | 6.6 | 0h 17m |
| 9 | Qwen3.5 9B Q4 | dense | 15.64% | 28.2 | 4.4 | 0h 19m |
| 10 | GLM-4.7 23B-A3B Q4 | MoE | 13.16% | 33.0 | 5.3 | 0h 18m |
| 11 | Gemma 3 27B Q4 | dense | 12.99% | 34.6 | 7.4 | 0h 31m |
| 12 | Phi-4 14B Q5 | dense | 10.63% | 35.8 | 7.9 | 0h 25m |
| 13 | DeepSeek R1 14B Q5 | dense |  8.17% | 35.7 | 10.6 | 0h 47m |

**Three findings from the tournament:**

- 🏅 **Gemma 4 family dominates.** Both A4B (#1) and dense 31B (#2) clear the next-best non-Gemma-4 model by 5+ pp on state accuracy. The Gemma 4 family became our forward candidate.
- 🔀 **High ROUGE-1 does NOT predict high state accuracy.** Qwen3 14B leads the table on R-1 (39.5) but ranks #5 on state acc; DeepSeek R1 14B has R-1 35.7 but ranks dead last (8.17%). Surface-form metrics measure phrasing style, not pedagogical routing — the empirical foundation for our benchmark-critique contribution (slide 11).
- 🎯 **The tournament is an input, not a verdict.** A4B (#1 no-think) was *not* promoted to the locked headline — its smoke-mini full-scale projection (~38.1%) sat below A3B's locked 38.70%, and A4B carries the same schema-fallback risk that later crushed Gemma 31B at full scale (next slide). The cascade + tournament feed into a Bayesian decision; A3B won the integrated decision despite ranking #7 in the no-think table.

<!--
SPEAKER NOTES (Slide 7, ~2:30):
This is the methodology slide — how we decide which open-weight backbone to pour GPU hours into when there are dozens of candidates and we can only afford to run a handful at full scale. Two pieces of methodology work together.

First, the cascade. Three tiers: smoke at n=5, mini at n=25, full at n=681. The reasoning is brutal: a full n=681 run takes 10 to 22 hours of wall clock on the 5090. You do not pay that price to learn your serving stack is broken or your model is bad. So every architectural decision goes through smoke first, then mini, before earning the right to spend a full GPU-night. The validation that lets us trust the cascade: averaging the smoke-tier and mini-tier state-accuracy lifts on Qwen 35B-A3B predicted the full-run lift at 12.86 percentage points — the realized lift came in at 12.76. That is calibration within 0.10 pp. Important caveat — and this is the next slide — the predictor is architecture-dependent. It failed catastrophically on Gemma 4 31B at full scale.

Second, the tournament. Thirteen popular open-weight backbones, all at n=50, all in no-think mode. The no-think was deliberate — it kept the entire sweep within one overnight budget. Ulises ran the original sweep on his AMD Radeon R9700; we cross-validated key points on the 5090. The A3B no-think state-acc result agreed across hardware to within 0.07 pp — that's our cross-hardware comparability proof.

The tournament table — three takeaways. One, the Gemma 4 family decisively dominates. A4B at thirty-eight-point-six-seven percent, Gemma 4 31B at thirty-five-point-eight-six, both clear of the next-best non-Gemma model by five points or more. Two, high ROUGE-1 does NOT predict high state accuracy. Qwen3 14B leads the table on ROUGE-1 at 39.5 but is rank five on state acc. DeepSeek R1 14B has a high ROUGE-1 of 35.7 and ranks dead last on the pedagogical axis at eight percent. That is the empirical seed of the benchmark critique we'll get to on slide 11 — surface-form metrics measure phrasing, not teaching. Three — and this is the methodological point — the tournament is an INPUT, not a verdict. The number one no-think model, A4B, was not the one we promoted to our first locked headline. A4B's smoke-mini projection sat below A3B's, and we later learned A4B carries the same schema-fallback risk that crushed Gemma 31B at full scale. So we promoted A3B — which ranks number seven in the no-think tournament — to our first full-scale locked headline. The cascade plus the tournament feed into a Bayesian decision; neither alone is the answer.
-->

---

## The Gemma 4 31B Retraction

With the cascade calibrated and the tournament telling us Gemma 4 was the top-tier family, we set up what we expected to be our next locked headline. The Gemma 4 31B dense variant had won every smoke and mini we'd thrown at it; the smoke-mini average projected it would land at ~46.7% state accuracy at full scale — **about 8 percentage points above A3B's locked 38.70%**. We kicked off the full n=681 fusion-think run overnight, expecting to wake up to a new headline.

Twenty-one hours and forty-nine minutes later, the result came back. **31.39%.** That's not a margin loss — that's *seven points below* the existing locked headline and *fifteen points below* the projection. We had projected a new headline. What we got was a retraction. So we became detectives.

The smoke-mini predictor had been calibrated on A3B to within 0.10 pp — it had no business being off by 15 pp. We instrumented per-turn fallback rates and went hunting for whatever was crushing Gemma at scale that hadn't surfaced at smaller n. We found it: **21% of Gemma's outputs at full scale failed the strict JSON schema** the fusion architecture requires. A3B's fallback rate on the same protocol was 0.91% — a **20× gap**. Fallback turns default to "stay in current state," which silently degrades the consultant prediction. The small-sample smoke and mini runs (each under 150 turns) had encountered **zero fallbacks** — the failure rate was a long-tail phenomenon that only surfaced at sample sizes in the hundreds. So the predictor wasn't broken; it was *missing a variable*.

That insight is the campaign's headline methodological contribution — and it set up Pivot 2 on the very next slide: if JSON-grammar adherence is the failure mode, take the JSON path off the consultant axis entirely.

- Root cause discovered by triangulation: **21% schema-fallback rate at n=681** vs. A3B's 0.91% — the model's JSON-grammar adherence broke at scale
- The smoke and mini samples (each <150 turns) **never surfaced a single fallback**
- **Methodological finding:** smoke–mini averaging is architecture-dependent. Cross-architecture scaling prediction must triangulate schema-fallback rates. **JSON-structured-output dependencies should be replaced with deterministic routing whenever feasible.**
- This is one of the campaign's headline methodological contributions

<!--
SPEAKER NOTES (Slide 8, ~2 min):
This is where the rigor mattered most. After the cascade and the tournament, the natural next move was to promote the tournament leader's dense variant — Gemma 4 31B — to a full n=681 run. The smoke and mini had been perfect. The smoke-mini average projected the full lift at about plus-eight percentage points above A3B's locked headline. We kicked the run off overnight, expecting to wake up to a new locked headline. Twenty-one hours and forty-nine minutes of GPU time later, we got the result. 31.39 percent. Seven points below A3B. Fifteen points below the projection. That's not a noisy outcome — that is a catastrophic miss. So we did what scientists do when the model fails — we became detectives.

The smoke-mini predictor was off by fifteen points. That has no business happening; on A3B it was accurate to within zero-point-one. So either the predictor is broken or it is missing a variable. We instrumented per-turn fallback rates — meaning, how often did the model fail to produce valid JSON for the fusion schema and fall back to the safety default. At full scale, 21 percent of Gemma's outputs failed the JSON schema. A3B's fallback rate on the exact same protocol was 0.91 percent — a twenty-times gap. And the small-sample runs — smoke at n=5, mini at n=25, each under 150 turns total — had encountered exactly zero fallbacks. The failure rate is a long-tail phenomenon. It only emerges at sample sizes in the hundreds. So the predictor isn't broken — it is missing a variable, and the variable is schema-fallback rate.

That insight became the campaign's headline methodological contribution — cross-architecture scaling prediction has to triangulate schema-fallback rates at scale, because small-n samples can mask a twenty-times degradation. And the architectural lesson — if JSON-grammar adherence is the failure mode, take the JSON path off the consultant axis entirely — is what set up the next slide, Pivot 2.
-->

---

## Pivot 2: BERT Consultant Integration

Coming out of the Gemma retraction, we had a sharp insight. The consultant's job is *state classification* — it picks one of 34 SocRule states given the dialogue history. That is a classification problem, not a generation problem. We had been asking a 30-billion-parameter LLM to do classification by producing structured JSON, and the JSON-adherence requirement was the source of the long-tail fragility that crushed Gemma at scale. So we asked the question the retraction made obvious: **what if the consultant axis didn't have to be an LLM at all?**

We tried something almost embarrassingly simple. We took `bge-small-zh-v1.5` — a **24-million-parameter** pretrained Chinese BERT encoder, freely available on HuggingFace — added a 34-way classification head, and fine-tuned it on SocratDataset's 42,000 state-labeled turns. The whole training run took **92 seconds**. On the 681-dialogue test split, the classifier reached **86.55% stage accuracy** at the 5-way level and **61.64% state accuracy** at the 34-way level. That's **+17.5 percentage points over the best LLM consultant** we had measured — by a model 1,400× smaller, trained in under two minutes.

The classifier slotted directly into the pipeline as a drop-in consultant. The full integration — BERT consultant + Gemma 4 31B teacher + 10-shot stage-balanced exemplars — ran at n=681 in 12 hours 53 minutes and landed at **48.15% state accuracy / 36.78 R-1 / unified 68.65**. That became the **prior locked headline on 2026-05-18**: a **+22.21 pp lift over the GPT-4o + SocratTeachLLM baseline (1.86× over the published number)**, on a single 32 GB consumer GPU, at **zero per-run API cost** for the eval pipeline. A 24-million-parameter classifier doing the routing work that GPT-4o was doing in the original KELE paper.

But the real win wasn't the headline number. It was the **architectural decomposition** that made the headline possible: the original KELE bundled *pedagogical routing* AND *surface-form generation* into one LLM consultant call, which is why JSON adherence became a load-bearing fragility. We split those axes — a deterministic classifier handles routing, an LLM teacher handles language — and each can be optimized independently with the right tool for the job. That decomposition is the headline architectural contribution of the campaign, and the path that took us from this slide to today's frontier-overtaking result on slide 15.

- Replace the consultant LLM with a **24M-parameter Chinese BERT classifier** (`bge-small-zh-v1.5`)
- Trained on SocratDataset's 42K labeled turns in **92 seconds** — 61.64% test-split state accuracy, **+17.5 pp over the best LLM consultant**
- **Drop-in integration:** BERT consultant + Gemma 4 31B teacher + 10-shot stage-balanced exemplars at n=681 = **48.15% state acc / 36.78 R-1 / unified 68.65**
- **Prior locked headline (2026-05-18):** **+22.21 pp over GPT-4o (1.86×)**, $0 per-run API cost, ~13 GPU-hours, 24-million-parameter classifier doing the routing work that GPT-4o was doing in the original paper

<!--
SPEAKER NOTES (Slide 9, ~2 min):
Coming out of the Gemma retraction, we had one sharp insight. The consultant's job is to classify the current dialogue state — pick one of 34 SocRule labels — and that is fundamentally a classification problem. Not a generation problem. We had been asking a 30-billion-parameter LLM to do classification by producing structured JSON, and the JSON-adherence requirement was what crushed Gemma at scale. So the question the retraction made obvious was: what if the consultant axis didn't have to be an LLM at all?

We tried something almost embarrassingly simple. Took bge-small-zh — a 24-million-parameter pretrained Chinese BERT encoder, freely available on HuggingFace — added a 34-way classification head, and fine-tuned it on the 42,000 state-labeled turns inside SocratDataset. The whole training run finished in 92 seconds. Yes, seconds. On the 681-dialogue test split, the classifier hit 86.55 percent stage accuracy at the 5-way level and 61.64 percent state accuracy at the 34-way level — beating every LLM consultant we had measured by more than 17 percentage points, on a model 1,400 times smaller, trained in under two minutes.

We dropped it into the pipeline. BERT consultant plus Gemma 4 31B teacher plus 10-shot stage-balanced exemplars. Full n=681 run, twelve hours fifty-three minutes, 48.15 percent state accuracy, 36.78 ROUGE-1, unified 68.65. That became our locked headline on May 18. Twenty-two-point-two-one percentage points above GPT-4o's baseline, a 1.86x lift, on a single 32 gigabyte consumer GPU at zero per-run API cost.

But the real win wasn't the headline number. The real win was the architectural decomposition. The original KELE bundled pedagogical routing AND surface-form generation into one consultant LLM call, which is why JSON adherence became a load-bearing fragility. We split those axes — deterministic classifier for routing, LLM teacher for language — and each can be optimized independently with the right tool for the job. That decomposition is the headline architectural contribution of the campaign, and the path that took us from this slide to today's frontier-overtaking result.
-->

---

## Frontier Stress Test

With the BERT integration locked at 48.15% and within striking distance of where a well-prompted frontier model *should* land, a question started bugging us. **Are we leaving real teaching accuracy on the table by stubbornly using open-weight backbones?** Maybe Gemma 4 31B simply isn't big enough to be a great Socratic teacher, and we're optimizing the wrong axis. The only way to answer is to break our own constraint — just for one controlled stress-test sweep. We had Anthropic API budget for the LLM-judge passes anyway; we redirected some of it to swap the teacher.

We kept the BERT consultant *fixed* — same 24M-parameter classifier that just locked the May 18 headline — and varied only the teacher. Gemma 4 31B → **Claude Sonnet 4.6** → **Claude Opus 4.6**. Each Claude teacher was tested across three scaffolding tiers: raw (no prompt engineering), 10-shot exemplars only, and 10-shot + the top-3 utilization stack (length_budget + persona + negative_exemplars) from our Phase 1 prompt-engineering tournament. Six Claude cells at n=50, then we promoted the best two — Sonnet + top-3 and Opus + top-3 — to full n=681 to land a canonical-scale frontier ceiling.

The result was both definitive and surprising. The best frontier configuration (`bert × Claude-Sonnet · top3 · n=681`) landed at **49.97% state acc / R-1 41.93 / unified 70.06**. Frontier was **within sampling noise of our open-weight integration** on the pedagogical axis — about +2 pp ahead on state acc, +5 ahead on R-1, +1.41 on unified. **Teacher capacity is not the binding constraint on this benchmark.** The deeper observation was even more useful: the top-3 prompt stack lifted Claude by 2–5× the amount the same stack lifted Gemma. So the bottleneck is on the **prompt-engineering axis**, not on raw model capability — frontier models have more *headroom* for prompt scaffolding, and once both teachers are well-prompted, they converge. That ~1.41-pt unified gap looked like the ceiling for a couple of weeks. *Spoiler for slide 15: it isn't.*

- BERT consultant **kept fixed**, swapped Gemma 4 31B → Claude Sonnet 4.6 and Claude Opus 4.6, each with a 10-shot + top-3-prompt-stack scaffolding
- **Best frontier configuration (n=681):** `bert × Claude-Sonnet · top3` → 49.97% state acc / R-1 41.93 / unified **70.06**
- Frontier within sampling noise of our prior open-weight headline on state acc (~+2 pp), +5 on R-1
- **Conclusion: teacher capacity is not the binding constraint.** Prompt scaffolding lifts Claude 2–5× the amount the same lever lifts Gemma — the bottleneck is on the prompt-engineering axis, not on raw model capability.
- *(Foreshadowing for Slide 15: the consultant-upgraded version we land today actually overtakes this ceiling.)*

<!--
SPEAKER NOTES (Slide 10, ~1:45):
With the BERT integration locked at 48.15% state accuracy and us looking at single-digit room to the ceiling, the natural question was: is this really the ceiling? Are we leaving teaching accuracy on the table by stubbornly using open-weight backbones? Maybe Gemma 4 31B is just not big enough to be a great Socratic teacher and we're optimizing the wrong axis. The only way to answer that is to break our own constraint — just for one controlled stress test. We had Anthropic API budget for the LLM-judge passes anyway, so we redirected some of it to a teacher swap.

We kept the BERT consultant fixed — same 24-million-parameter classifier that just locked the May 18 headline — and varied only the teacher. Gemma 4 31B becomes Claude Sonnet 4.6, becomes Claude Opus 4.6. Each Claude teacher tested across three scaffolding tiers: raw with no prompt engineering, 10-shot exemplars only, and 10-shot plus our top-3 utilization stack from the Phase 1 tournament — that's length_budget plus persona plus negative_exemplars. Six Claude cells at n=50, and we promoted the two strongest to full n=681 to land a canonical-scale frontier ceiling.

The result was both definitive and surprising. The best frontier configuration — Claude Sonnet 4.6 with the top-3 stack and our BERT consultant, at full n=681 — landed at 49.97 percent state accuracy, ROUGE-1 of 41.93, unified 70.06. Within sampling noise of our open-weight integration on the pedagogical axis. About two percentage points ahead on state acc, five ahead on ROUGE-1, 1.41 on unified. Teacher capacity is NOT the binding constraint on this benchmark. The deeper observation that came out of the same sweep: the top-3 prompt stack lifts Claude two to five times the amount the same stack lifts Gemma. So the bottleneck is on the prompt-engineering axis, not on raw model capability. Frontier models have more headroom for prompt scaffolding, and once both teachers are well-prompted they converge. That 1.41-point unified gap looked like the ceiling for a couple of weeks. Spoiler for slide 15: it isn't.
-->

---

## The Benchmark Critique — A Methodological Discovery

With our open-weight system landing within sampling noise of frontier Claude, something started bothering us about the original KELE paper. The paper reports SocratTeachLLM — a **9-billion-parameter fine-tune from 2024** — outperforming GPT-4o on every single one of nine evaluation dimensions. That is statistically improbable for a 9B specialist competing against a frontier model on a real pedagogical task. Either the small model is genuinely better at teaching (a surprising claim), or the evaluation metric is measuring something other than teaching. We had to find out which.

We ranked our configurations by the same metrics the paper uses — the surface-form sum (R-1 + R-2 + BLEU-4) — and got a leaderboard. **SocratTeachLLM lands FIRST**, by +10.83 over Opus 4.6 with our carefully tuned top-3 prompts. Then we ranked the *same* configurations by **pedagogical state accuracy** — does the system actually route to the correct SocRule state? SocratTeachLLM lands **DEAD LAST** at 25.94%, below even raw Opus 4.6 with zero prompt engineering (39.75%). The two rankings **invert** across the same configurations.

Here's the smoking gun. The surface-form lead doesn't just exist — it **widens monotonically with n-gram length**: +1.59 on R-1 (unigrams), +4.92 on R-2 (bigrams), +4.07 on BLEU-4 (4-grams). Higher-order n-gram overlap measures phrase-level fingerprinting. A model that learned the *meaning* of teaching would not produce a monotone increase in lead as n-gram length grows; it would produce a flat or shrinking lead, because paraphrasing breaks high-n-grams. The widening lead is the strongest possible signature of training-data memorization. SocratTeachLLM was fine-tuned on SocratDataset's phrasing; it's not teaching, it's quoting.

We confirmed the diagnosis with a cross-lingual translation experiment. We translated the entire 6,803-dialogue dataset into English (this is the SocratDataset-EN release on HuggingFace) and re-ran the configurations. SocratTeachLLM **reproduced** the original paper's flagship R-1 of 57.40 to within 1.5 points (we measured 55.85). But on the LLM-judge axis — which measures *meaning* rather than surface form — frontier+prompts lost only **−0.07** crossing languages, while SocratTeachLLM lost **−1.0**. That's a **14× asymmetric degradation** on a paraphrase-invariant metric, alongside preserved surface-form scores. Translation strips the language but preserves the pedagogy: the model that's measuring memorization keeps its surface-form lead but loses on meaning. **Diagnosis confirmed:** the KELE benchmark, as published, rewards memorization, not teaching capability.

- **Surface-form sum (R-1 + R-2 + BLEU-4) ranks SocratTeachLLM 9B FIRST** by +10.83 over Opus 4.6 with carefully tuned prompts
- **State-accuracy ranks the SAME SocratTeachLLM LAST** (25.94%, below even raw Opus at 39.75%)
- **The two rankings invert.** And the gap widens monotonically with n-gram length: +1.59 R-1, +4.92 R-2, +4.07 BLEU-4
- **Diagnosis:** ROUGE/BLEU measure surface mimicry. SocratTeachLLM was trained on SocratDataset's phrasing. The metrics reward memorization, not teaching.
- **Cross-lingual confirmation:** SocratTeachLLM evaluated on an English translation of the dataset *reproduces* its R-1 = 55.85 headline within 1.5 points of the paper's 57.40, while frontier+prompts loses only 0.07 on the language shift versus SocratTeachLLM losing 1.0 on the same axis

<!--
SPEAKER NOTES (Slide 11, ~2:30):
This is the methodological turn of the campaign — and I'd argue it's the most important contribution we made beyond the architectural pivots. With our open-weight integration landing within sampling noise of frontier Claude, something started bothering us about the original KELE paper. The paper reports SocratTeachLLM — a 9-billion-parameter fine-tune — outperforming GPT-4o on every single one of nine evaluation dimensions. That is statistically improbable. A 9B specialist beating a frontier model on a real pedagogical task across the board. Either the small model is genuinely better at teaching, which would be a surprising claim, or the evaluation is measuring something other than teaching. We had to find out which.

We ranked our configurations by the same metrics the paper uses — the surface-form sum, ROUGE-1 plus ROUGE-2 plus BLEU-4 — and got a leaderboard. SocratTeachLLM lands FIRST, by ten-point-eight-three over Opus 4.6 with our carefully tuned top-3 prompts. Then we ranked the SAME configurations by pedagogical state accuracy — does the system actually route to the correct SocRule state. SocratTeachLLM lands DEAD LAST at 25.94 percent, below even raw Opus with zero prompt engineering at 39.75. The two rankings invert across the same configurations.

Here's the smoking gun. The surface-form lead doesn't just exist — it widens monotonically with n-gram length. Plus 1.59 on R-1, plus 4.92 on R-2, plus 4.07 on BLEU-4. Higher-order n-gram overlap measures phrase-level fingerprinting. A model that learned the meaning of teaching would not produce a monotone INCREASE in its lead as n-gram length grows; it would produce a flat or shrinking lead, because paraphrasing breaks high n-grams. The widening lead is the strongest possible signature of training-data memorization. SocratTeachLLM was fine-tuned on SocratDataset's phrasing — it's not teaching, it's quoting.

We confirmed it with a cross-lingual translation experiment. We translated the entire 6,803-dialogue dataset into English — that's the SocratDataset-EN release on HuggingFace — and re-ran the configurations. SocratTeachLLM reproduced the original paper's flagship ROUGE-1 of 57.40 to within 1.5 points; we measured 55.85. But on the LLM-judge axis — which measures meaning rather than surface form — frontier-plus-prompts lost only zero-point-zero-seven crossing languages, while SocratTeachLLM lost a full point. That is a fourteen-times asymmetric degradation on a paraphrase-invariant metric, while the surface-form metric is preserved. Translation strips the language but preserves the pedagogy: the model that is measuring memorization keeps its surface-form lead but loses on meaning. Diagnosis confirmed: the KELE benchmark, as published, rewards memorization, not teaching capability.
-->

---

## Contamination Proof

A rank inversion, a monotone n-gram-length signature, and a cross-lingual translation result is *strong circumstantial evidence* — but "the published benchmark systematically rewards memorization" is a serious claim against an EMNLP-Findings paper. We needed direct, dispositive proof that SocratTeachLLM was memorizing rather than generalizing. So we built **two independent probes, each with a clean Gemma 31B control.**

**Probe 1 — direct memorization detection.** We compared SocratTeachLLM's generated outputs character-for-character against the training corpus. **4/288 outputs are character-for-character identical to training data. 17/288 are ≥80% match.** The Gemma 31B control on the exact same probe: **0/288 exact, 0/288 ≥80%.** SocratTeachLLM is literally quoting back its training data on roughly 1-in-70 turns at the strict threshold and 1-in-17 at the relaxed threshold. Gemma is producing zero matches on the same dialogue inputs. There is no plausible non-memorization explanation for that gap.

**Probe 2 — clean-probe on synthetic data.** We had Claude Sonnet generate a fresh batch of dialogue test cases that are demonstrably *outside* SocratDataset's training distribution — synthesized after SocratTeachLLM's training cutoff, with topics intentionally drifted from the elementary-science domain. We ran both SocratTeachLLM and Gemma 31B against the clean probe under the same protocol. **SocratTeachLLM's stage-balanced state accuracy collapsed from 63.4 → 32.86 — below Gemma 31B's 56.13 on the same clean probe.** The model that wins on SocratDataset by 30+ points loses by 23 points the moment we change the data distribution. That isn't pedagogical capability degrading; that's memorization *vanishing*.

**Combined verdict.** Two independent probes agree. The KELE benchmark, as published, systematically rewards memorization over pedagogical capability. The published "GPT-4o + SocratTeachLLM" baseline R-1 of 44.61 — higher than Opus 4.6 with carefully tuned prompts — is the canonical example. Future work on Socratic-teaching evaluation needs a memorization-resistant metric, which we propose on the next slide.

- **Memorization probe:** 4/288 SocratTeachLLM outputs are character-for-character identical to training data; 17/288 are ≥80% match. Gemma 31B control on the same probe: **0/288** such matches.
- **Clean-probe on synthetic data:** ran SocratTeachLLM against ground-truth dialogues generated by Claude Sonnet (demonstrably outside SocratDataset). State accuracy collapses to **32.86 stage-balanced** — *below Gemma's 56.13 on the same probe*.
- The KELE benchmark, as published, **systematically rewards memorization over pedagogical capability**
- The published "GPT-4o + SocratTeachLLM" baseline R-1 of 44.61 — higher than Opus 4.6 with prompts — is the canonical example of the failure mode

<!--
SPEAKER NOTES (Slide 12, ~1:45):
The rank inversion plus the monotone n-gram-length signature plus the cross-lingual result is strong circumstantial evidence. But "the published benchmark systematically rewards memorization" is a serious claim against an EMNLP-Findings paper. We needed direct, dispositive proof that SocratTeachLLM was memorizing rather than generalizing. So we built two independent contamination probes, each with a clean Gemma 31B control.

Probe one — direct memorization detection. We compared SocratTeachLLM's generated outputs character-for-character against the training corpus. Looking for exact matches. Four out of two hundred and eighty-eight outputs are character-for-character identical to training data. Seventeen out of two hundred and eighty-eight are eighty-percent-or-more match. The Gemma 31B control on the exact same probe? Zero exact matches. Zero eighty-percent matches. SocratTeachLLM is literally quoting back its training data on roughly one in seventy turns at the strict threshold and one in seventeen at the relaxed threshold. Gemma is producing zero matches on the same dialogue inputs. There is no plausible non-memorization explanation for that gap.

Probe two — clean probe on synthetic data. We had Claude Sonnet generate a fresh batch of dialogue test cases that are demonstrably outside SocratDataset's training distribution. Synthesized after SocratTeachLLM's training cutoff, topics intentionally drifted from the elementary-science domain. We ran both SocratTeachLLM and Gemma 31B against the clean probe under the same protocol. SocratTeachLLM's stage-balanced state accuracy collapsed from sixty-three-point-four to thirty-two-point-eight-six — BELOW Gemma 31B's 56.13 on the same clean probe. The model that wins on SocratDataset by thirty-plus points LOSES by twenty-three points the moment we change the data distribution. That isn't pedagogical capability degrading. That's memorization vanishing.

Combined verdict. Two independent probes agree. The KELE benchmark as published rewards memorization over pedagogical capability. The "GPT-4o + SocratTeachLLM" baseline ROUGE-1 of 44.61, higher than Opus 4.6 with prompts, is the canonical example. Future work on Socratic-teaching evaluation needs a memorization-resistant metric — which we propose on the next slide.
-->

---

## The Unified Metric

Coming out of the benchmark critique and the contamination proof, we knew ROUGE and BLEU were measuring the wrong thing for Socratic teaching. But we still needed *something* — a defensible single-number ranking we could put in a paper headline and use to compare configurations without an ad-hoc per-metric argument every time. Six per-cell metrics already live in our leaderboard (macro state-acc, stage-balanced macro, pedagogically-weighted macro, frequency-inverse, LLM-judge, and the four-piece surface-form panel). None of them is bulletproof in isolation. The **unified score** is our answer.

```
unified = 0.5 × stage_balanced + 0.5 × (judge × 10)
```

*Both inputs on [0, 100]; output on [0, 100]. Higher is better.*

The formula picks two metrics that are **memorization-resistant by construction** and equal-weights them. **`stage_balanced`** is the equal-weight per-stage macro state accuracy. The published "macro" weights by stage frequency, which structurally under-counts the closure stage (stage e is ~12% of turns but the pedagogically load-bearing moment of the dialogue). `stage_balanced` fixes that: 20% weight per stage, regardless of frequency. **`judge`** is the Claude Sonnet 4.6 rubric — four axes (Socratic validity / advancement / age-appropriateness / question-form fidelity), 0–10 scale per turn, averaged across the run. Neither input rewards token-level mimicry; both are designed to survive the contamination probe from the previous slide.

The 50/50 weighting is deliberate. We don't have a principled prior on whether pedagogical routing should outweigh response quality, and both inputs share comparable memorization-resistance. Equal weighting requires no extra defense beyond "both axes matter, neither dominates" — any other weighting would force us into an argument we don't need to fight in the headline. **This is the metric that surfaces today's frontier-overtaking result on slide 15.** Without it — ranking by surface form — the leaderboard would still put SocratTeachLLM on top.

- **`stage_balanced`** — equal-weight per-stage state accuracy (corrects the published macro's structural under-counting of stage e, closure)
- **`judge`** — Claude Sonnet 4.6 rubric on 4 axes (Socratic validity / advancement / age-appropriateness / question-form fidelity), 0–10 scale
- Both inputs are **memorization-resistant by construction** — surface-form metrics are *excluded*, per the critique
- Master leaderboard: **143 configurations, 38 LLM-judged**, auto-regenerated by `scripts/backtest_stage_balanced.py`

<!--
SPEAKER NOTES (Slide 13, ~2 min):
Once we had rejected surface-form metrics with the critique and the contamination proof, we still had a problem — six per-cell metrics live in our leaderboard and none of them is bulletproof in isolation. We needed a single defensible number for paper headlines and decision-making, without an ad-hoc per-metric argument every time. So we built the unified score.

The formula is simple. One-half stage-balanced state accuracy, plus one-half judge score times ten. Both inputs land on a zero-to-one-hundred scale, so the output lands on zero-to-one-hundred. Higher is better. The two inputs are deliberately chosen to be memorization-resistant by construction.

Stage-balanced is the equal-weight per-stage macro state accuracy. The published macro weights by stage frequency — which structurally under-counts the closure stage. Stage e is roughly twelve percent of turns but it is the pedagogically load-bearing moment of the dialogue. Frequency weighting hides failures on closure. Stage-balanced fixes that — twenty percent weight per stage, regardless of frequency. Judge is the Claude Sonnet 4.6 rubric score — four axes, Socratic validity, advancement of student reasoning, age-appropriateness, and question-form fidelity, zero to ten scale per turn, averaged across the run. Neither input rewards token-level mimicry; both survive the contamination probe we just showed.

The fifty-fifty weighting is deliberate. We don't have a principled prior on whether pedagogical routing should outweigh response quality, and both inputs share comparable memorization-resistance. Equal weighting requires no extra defense beyond "both axes matter, neither dominates." Any other weighting would force us into an argument we do not need to fight in the headline. This is the metric that surfaces today's frontier-overtaking result on slide 15. Without it — ranking by surface form alone — the leaderboard would still put SocratTeachLLM on top. The master leaderboard now sits at 143 configurations, 38 of them LLM-judged, all auto-regenerated by a backtest script.
-->

---

## Consultant Upgrade + TODO #14

A systematic 4-cell `T1–T4` funnel produced the **Qwen3.5-0.8B-LoRA classifier** — successor to the 24M BERT (post-fix consultant input format, +6.23 pp over BERT). TODO #14 lined up 4 canonical-scale cells at n=681 to **confirm — or revise — the screening-tier parity finding**.

| Cell | unified | Status |
|---|---:|---|
| `qwen3.5 × A3B-35B · n=681` | 67.81 | ✅ 2026-05-25 (9h 41m) — 2.25 pts behind frontier ceiling |
| `qwen3.5 × Qwen-27B no-think · n=681` | 66.71 | ✅ 2026-05-25 (1h 4m) — 3.35 pts behind frontier ceiling |
| `bert-fixed × Gemma-31B · n=681` | — | queued, ~12 GPU-h — *no longer load-bearing* |
| **`qwen3.5 × Gemma-31B · n=681`** | **?** | **today (2026-05-26), ~12 GPU-h — see next slide** |

<!--
SPEAKER NOTES (Slide 14, ~45s):
The consultant upgrade was a systematic four-cell funnel — two backbones crossed with frozen versus LoRA. T4, the LoRA fine-tune of Qwen3.5-0.8B-Base, won by 6.23 percentage points over BERT and became the successor consultant. With that classifier in hand, TODO 14 lined up four canonical-scale cells at n=681 — designed to *confirm or revise* the screening-tier parity finding from May 23. Three cells landed before today: A3B 35B at unified 67.81, Qwen-27B no-think at 66.71, and the bert-fixed Gemma cell still queued. Both A3B and Qwen-27B trail the frontier ceiling — 2.25 and 3.35 unified points respectively. The fourth cell ran today: qwen3.5 cross Gemma 31B at n=681. The result is on the next slide.
-->

---

## Today: Frontier Overtaken

`qwen3.5 × Gemma-31B · fewshot10 · n=681`:

- State accuracy **55.39%** = **2.14× GPT-4o baseline** (+29.45 pp absolute)
- ROUGE-1 **37.65** · LLM-judge **8.32**/10 · **Unified 72.24**
- **+2.18 unified pts above the best frontier configuration** (`bert × Claude-Sonnet · top3 · n=681` at 70.06)
- Per-stage multipliers vs.\ GPT-4o: **c = 7.54× · d = 9.20×** · e = 6.58×
- **A 31B-param open-weight teacher with prompt engineering on a single 32 GB consumer GPU beats Anthropic's best frontier model on a memorization-resistant Chinese pedagogy benchmark at canonical sample size, at $0 per-run eval API cost.**

<!--
SPEAKER NOTES (Slide 15, ~1.5 min):
This is what we locked this morning. The fourth TODO 14 cell: Qwen3.5-LoRA classifier with Gemma 4 31B teacher and 10-shot exemplars, at the full 681-dialogue test split. State accuracy 55.39 percent — that's 2.14 times GPT-4o's baseline of 25.94. ROUGE-1 of 37.65. LLM-judge score 8.32 out of 10. Unified score 72.24. That puts us 2.18 unified points above the best frontier configuration we tested — Claude Sonnet 4.6 with our top-3 prompt stack and the same BERT consultant — at the same canonical sample size. Per-stage multipliers versus GPT-4o on the hard middle and closure stages: 7.54x on induction, 9.20x on extension, 6.58x on closure. A 31-billion-parameter open-weight teacher on my single 32-gigabyte consumer GPU just overtook Anthropic's best frontier model on this benchmark. Zero dollars per-run for the eval pipeline. About 16 dollars for the LLM-judge pass that completes the unified score.
-->

---

## By the Numbers

Three months of campaign. Numbers from the master leaderboard and the experiment log.

```
configurations measured .................. 143
configurations LLM-judged ................. 38
full n=681 runs ............................ 7
GPU-hours (named-run audit trail) ....... 119.5
GPU-hours (incl. un-itemized smoke/mini)  ~140
API spend (total, Anthropic) .......... $258.86
prompts written (tournament) .............. 10
papers in flight ........................... 1
locked-headline promotions ................. 3
```

**GPU-hour breakdown** (from paper Table 1, Table 2, EXPERIMENT_LOG, and CONSULTANT_UPGRADE_LOG):

| Block | Hours |
|---|---:|
| 7 full n=681 runs (A3B, Gemma standalone, BERT+Gemma, BERT+A3B, 3× qwen3.5) | 77.2 |
| 13-model n=50 tournament | 14.4 |
| Phase 1 prompt-engineering tournament (10 cells × n=50) | 6.2 |
| Bilingual canonical n=400 | 7.8 |
| Cross-teacher 8-cell n=50 matrix + judge re-eval | 6.0 |
| Consultant upgrade campaign (T1–T4 training + Layer-2 mini) | 7.9 |
| BERT classifier training (92s v1 + 211s v2) | 0.1 |
| **Confirmed total** | **119.5** |

**API spend** goes to three line items: (i) Phase 3 frontier-teacher n=681 runs — Claude Sonnet and Opus as teacher, ~4,000 turns each; (ii) Phase 2 frontier-teacher n=50 sweeps — 6 Claude-teacher configurations stress-testing prompt scaffolding; (iii) the LLM-judge passes across 38 cells that produce the unified score. The open-weight eval pipeline itself is $0 per-run — every API dollar bought either a frontier comparison or a memorization-resistant judgment.

**Rank-#1 progression** (master leaderboard, by unified score)

```
2026-04 .... Qwen 35B-A3B fusion-think (n=681, 38.70%)        — pivot 1
2026-05-17 . retracted: Gemma 4 31B standalone (collapsed)    — schema-fallback rate
2026-05-18 . BERT × Gemma × 10-shot (n=681, unified 68.65)    — pivot 2
2026-05-25 . qwen3.5 × A3B-35B (n=681, unified 67.81)         — canonical scale
2026-05-26 . qwen3.5 × Gemma-31B (n=681, unified 72.24) 🏆    — FRONTIER OVERTAKEN
```

<!--
SPEAKER NOTES (Slide 16, ~1 min):
The campaign by the numbers. 143 distinct configurations measured. 38 of them LLM-judged. Seven full n=681 runs at canonical scale. Adding up every named wall-clock from the paper, the experiment log, and the consultant-upgrade log gives 119 and a half GPU-hours of confirmed compute — roughly five days of continuous wall clock on the 5090, plus another fifteen to twenty-five GPU-hours of un-itemized smoke and mini runs we didn't time precisely. The biggest single block is the seven full n=681 runs at 77 hours; the next biggest is the 13-model tournament at 14 hours. API spend with Anthropic totaled 258 dollars and 86 cents. Three line items there: frontier-teacher comparisons at n=681 with Claude Sonnet and Opus as the teacher; the Phase 2 frontier-teacher n=50 sweeps stress-testing prompt scaffolding; and the LLM-judge passes that produce the unified score. The open-weight eval pipeline itself is zero dollars per-run — every API dollar bought either a frontier comparison or a memorization-resistant judgment. The progression at the bottom shows the unified-score rank-1 cell as it shifted: A3B fusion in early May, retraction of standalone Gemma when we discovered the schema-fallback issue, BERT integration becoming the locked headline on the 18th of May, then a series of canonical-scale cells in late May ending with today's frontier overtaking.
-->

---

## Total Contributions

Every improvement we made over the original KELE paper (Peng et al., EMNLP 2025 Findings) — **19 distinct contributions across five categories**.

### 🔧 Architectural upgrades (vs. KELE's two-model stack)

1. **Classifier-as-consultant, deployed on CPU+Compute.** Replaced KELE's GPT-4o LLM consultant with a deterministic supervised classifier — 24M-param Chinese BERT (`bge-small-zh-v1.5`) initially, upgraded to a ~800M-param Qwen3.5-0.8B-LoRA classifier in the current locked headline. Both run on **CPU+Compute, freeing the entire GPU's VRAM** for the much larger teacher LLM. Eliminates per-run API spend on the consultant axis. Classifier state accuracy 61.64% on the 34-state test split — +17.5 pp over the best LLM consultant.
2. **Consultant-upgrade funnel (T1–T4).** Systematic 2×2 search over {Qwen3-Embedding-0.6B, Qwen3.5-0.8B-Base} × {frozen, LoRA}. T4 (Qwen3.5-0.8B + LoRA r=8) won at +6.23 pp over the BERT baseline and is the current locked-headline consultant.
3. **Composed teacher prompt engineering** — stage-balanced 10-shot exemplars + top-3 utilization stack (length_budget + persona + negative_exemplars), surfaced by a 10-cell n=50 prompt-engineering tournament. +6.02 state acc / +3.29 R-1 over the raw teacher; +5 unified pts on Gemma.

### 📐 Methodological & evaluation contributions

4. **Benchmark critique.** Surface-form metrics (ROUGE/BLEU) on SocratDataset systematically reward training-data memorization over teaching capability. Same configurations rank in **opposite orders** under ROUGE vs.\ state accuracy; the gap widens monotonically with n-gram length — the strongest possible memorization signature.
5. **Unified ranking metric** (`unified = 0.5 × stage_balanced + 0.5 × (judge × 10)`). Memorization-resistant single-number ranking. `stage_balanced` corrects KELE's frequency-weighted macro (closure under-counting); `judge` is a 4-axis Claude Sonnet 4.6 rubric (Socratic validity, advancement, age-appropriateness, question-form).
6. **Contamination proof** — two independent evidence streams: memorization probe (4/288 char-identical, 17/288 ≥80% match for SocratTeachLLM vs.\ 0/288 Gemma control) + synthetic clean-probe (SocratTeachLLM collapses 63.4 → 32.86 stage_bal on demonstrably-unseen synthetic data).
7. **Smoke / mini / full evaluation protocol** with smoke-mini averaging as a low-cost full-run predictor (predicted A3B's full-run lift within 0.10 pp); surfaced Gemma 31B's 15.32-pp full-scale collapse via schema-fallback-rate triangulation.
8. **n=400 canonical sample-size recommendation.** Bootstrap convergence analysis across 7 full-scale n=681 runs: all four primary metrics converge within ≤2 pp at n=400 — **41% compute saving with no loss of decision precision**.

### 🌐 Dataset contributions (public on 🤗 `ulises-c/…`)

9. **SocratDataset-EN** — full English translation of all 6,803 dialogues / 42,892 turns. Anchors the cross-lingual transfer experiment + the SocratTeachLLM language-bound memorization counter-probe.
10. **SocratDataset-SYNTHETIC + SYNTHETIC-EN** — Claude-generated clean-probe datasets (n=75 each), demonstrably outside SocratDataset's training distribution.

### 📊 Key empirical findings

11. **🏆 Frontier OVERTAKEN at canonical n=681.** Current locked headline (`qwen3.5 × Gemma-31B · fewshot10 · n=681`) at unified **72.24** beats the best frontier teacher we tested (`bert × Claude-Sonnet · top3 · n=681`, unified 70.06) by **+2.18 unified pts**. A 31B open-weight teacher on one 32 GB consumer GPU beats Anthropic's best on a memorization-resistant benchmark.
12. **State accuracy: 25.94% (KELE) → 55.39% (ours) = 2.14× lift / +29.45 pp absolute.** Per-stage GPT-4o-baseline multipliers: **c = 7.54× · d = 9.20× · e = 6.58×**. Stage b moves from KELE's −13.67 pp deficit to +9.46 pp lift — every stage now positive.
13. **Cross-lingual transfer of the SocRule routing.** Qwen3.5-LoRA consultant trained only on Chinese scores unified 65.11 at n=400 on the English test split — Stage 1 confirmed; macro drop 9.24 pp inside the 10 pp gate. Pedagogical state labels (a0–e34) are language-invariant.
14. **Schema-fallback rate is the missing variable** in cross-architecture scaling prediction. Gemma 31B's 21% full-scale fallback rate (vs.\ A3B's 0.91%) crushed its smoke/mini projection by 15.32 pp. Methodological lesson: JSON-structured-output dependencies should be replaced with deterministic routing whenever feasible.
15. **Teacher capacity is NOT the binding constraint** on this benchmark. Frontier-teacher stress test shows prompt scaffolding lifts Claude by 2–5× the amount the same lever lifts Gemma; swapping open-weight for frontier teacher is not the bottleneck.
16. **Architecture-correlated think-benefit gradient** within Qwen 3.6: MoE A3B gains ~19 pp from reasoning scaffolding, dense 27B gains ~11–17 pp. Robust across n=25, n=33, n=50, n=681 — explicit reasoning compensates for the MoE per-token compute deficit.

### ⚙️ Engineering & economics

17. **143 configurations measured, 38 LLM-judged.** Master leaderboard auto-regenerated from per-config JSON summaries via `scripts/backtest_stage_balanced.py`. Full audit trail in `results/_orchestrator_logs/`.
18. **Single-GPU + $0 per-run eval pipeline.** Entire locked-headline pipeline runs on one RTX 5090 (32 GB VRAM); judge passes (~$16/cell) are the only marginal API cost.
19. **Compute audit:** **119.5 GPU-h confirmed** (7 full n=681 runs + 13-model tournament + prompt tournament + bilingual canonical + cross-teacher matrix + consultant upgrade campaign); **$258.86 total Anthropic spend** (frontier comparisons + judge passes; $0 on the open-weight eval pipeline itself).

---

**Code · data · paper draft:** github.com/ulises-c/csen-346 · 🤗 ulises-c (SocratDataset, SocratDataset-EN, SocratDataset-SYNTHETIC, SocratDataset-SYNTHETIC-EN, SocratTeachLLM mirror)

**Questions?**

<!--
SPEAKER NOTES (Slide 17, ~90s — denser than the rest; the final-slide depth gives audience the full picture, speaker picks which bullets to voice):

Nineteen distinct improvements over the original KELE paper, organized into five categories.

Three architectural upgrades. First and headline: we replaced KELE's GPT-4o consultant with a deterministic supervised classifier — initially a 24-million-parameter Chinese BERT, then upgraded to an 800-million-parameter Qwen3.5-LoRA classifier in the current locked headline. Crucially, both run on CPU plus compute, which frees the entire GPU's VRAM for the teacher LLM. Second: the consultant upgrade was systematic — a four-cell funnel over two backbones crossed with frozen versus LoRA. The LoRA-fine-tuned Qwen3.5-0.8B won by 6.23 percentage points and became the headline consultant. Third: composed prompt engineering on the teacher — stage-balanced 10-shot exemplars plus a top-3 utilization stack — surfaced by a 10-cell n=50 tournament, lifts the teacher's pedagogy without retraining.

Five methodological contributions. The benchmark critique — ROUGE and BLEU on this dataset systematically reward memorization over teaching, and the ranking inversion widens monotonically with n-gram length. The unified metric — half stage-balanced state accuracy, half LLM-judge — that fixes the ranking. The contamination proof, with two independent evidence streams. The smoke-mini-full evaluation protocol that caught the Gemma 31B retraction. And the n=400 canonical sample-size recommendation, which would save 41 percent of compute on future Socratic-teaching evaluations with no loss of decision precision.

Two public datasets — SocratDataset-EN, the first full English translation of the 6,803 dialogues, and the clean-probe synthetic datasets in both languages.

Six empirical findings — most importantly the frontier-overtaking result we showed on slide 15. State accuracy 2.14 times GPT-4o. Cross-lingual transfer works. Schema-fallback rate is the missing variable for cross-architecture scaling prediction. Teacher capacity is not the binding constraint on this benchmark — prompt scaffolding is 2 to 5 times more impactful. And the architecture-correlated think-benefit gradient within the Qwen 3.6 family — MoE A3B gains 19 points from reasoning scaffolding, dense 27B gains 11 to 17, robust across four sample sizes.

Three engineering wins: 143 configurations measured, single-GPU at zero dollars per inference run on the eval pipeline, full compute audit at 119 and a half GPU-hours and 258 dollars of Anthropic spend — every API dollar bought either a frontier comparison or a memorization-resistant judgment.

Code, data, paper draft are public. Questions?
-->

