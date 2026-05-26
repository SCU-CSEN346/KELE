<!--
PRESENTATION.md — 15-minute class talk for CSEN 346
Render: `npx reveal-md PRESENTATION.md` for slide deck in browser,
        or read directly on GitHub (renders as one long doc).
Speaker notes are HTML comments — visible in source, hidden in render.
Target pace: ~130 words/minute. 15 slides, ~14:45 + Q&A.
-->

# Beating the Frontier on a Consumer GPU

### Reproducing and Extending KELE: a Multi-Agent Socratic Teaching Framework

**Maximilian Khan** · CSEN 346 · Santa Clara University · May 2026

A 31B-parameter open-weight teacher running on a single 32 GB consumer GPU — beating Anthropic's best frontier model on a memorization-resistant Chinese pedagogy benchmark at canonical sample size, at **zero** per-run API cost.

<!--
SPEAKER NOTES (Slide 1, ~15s):
Hi, I'm Max. Today I'll walk you through a 3-month NLP campaign reproducing and extending a 2025 EMNLP paper called KELE — Knowledge-Enhanced Learning Environment for Socratic Teaching. The headline result, which we locked in this morning: a 31-billion-parameter open-weight teacher on my single 32-gigabyte 5090 just overtook Anthropic's best frontier model on this benchmark, at zero per-run API cost. Let me show you how we got there.
-->

---

## What is KELE?

Peng et al., *Findings of EMNLP 2025* — a two-model framework for **Socratic teaching** of elementary-school students, in Chinese.

- **Consultant LLM** classifies the student's cognitive state (34 fine-grained pedagogical states across 5 SocRule stages: questioning → anchoring → induction → extension → closure)
- **Teacher LLM** generates a response conditioned on the predicted state
- **SocratDataset:** 681 dialogues, ~4,300 teacher turns, ground-truth state-by-state annotations
- **Original paper baseline:** GPT-4o (consultant) + SocratTeachLLM-9B (teacher) → **25.94% state accuracy**, R-1 44.61, BLEU-4 19.60

<!--
SPEAKER NOTES (Slide 2, ~1 min):
KELE is a 2025 EMNLP-Findings paper from Peng et al. It's a two-model pipeline for Socratic teaching: one LLM acts as a "consultant" that classifies which of 34 cognitive states the student is in — these states are grouped into 5 pedagogical stages they call SocRule: questioning, anchoring, induction, extension, closure. A second LLM acts as the "teacher" generating the actual response conditioned on the consultant's state prediction. The dataset is 681 Chinese dialogues, ~4300 teacher turns. The original paper's baseline used GPT-4o as the consultant and a fine-tuned 9B model called SocratTeachLLM as the teacher. They reported 25.94% state accuracy and R-1 44.61. Hold onto those numbers — they're the bar to beat.
-->

---

## Our Constraint

A single 32 GB consumer GPU (NVIDIA RTX 5090) and **$0 budget** for per-run API calls.

- KELE's canonical stack requires **two model deployments simultaneously** — one for the consultant, one for the teacher
- Two 30B-class open-weight models will not co-resident on 32 GB
- Renting frontier API consultancy = burning the API budget on every dialogue
- **The challenge:** reproduce + improve on harder hardware than the original paper

<!--
SPEAKER NOTES (Slide 3, ~45s):
Here's the constraint that shapes everything else. I have one RTX 5090 with 32 gigabytes of VRAM, and a budget of zero dollars for per-run API calls. KELE's original architecture requires two LLMs running simultaneously — a consultant and a teacher. You cannot fit two 30-billion-parameter open-weight models in 32 gigabytes of VRAM. And renting frontier API calls for the consultant role would burn budget on every single dialogue. So the campaign was: reproduce KELE plus improve on it, on harder hardware than the original paper used.
-->

---

## Pivot 1: Fusion Architecture

Collapse consultant and teacher into a **single backbone**, single forward pass, structured-output call returning both the state prediction and the teacher response.

- **Two-call** (consultant LLM → teacher LLM): VRAM-prohibitive, slow, KV cache duplicated
- **Fusion call** (one model, one JSON output with both fields): single backbone, ~2× faster
- **First locked headline (n=681):** Qwen 35B-A3B fusion-think → **38.70% state accuracy** vs.\ GPT-4o's 25.94% — **+12.76 absolute pp / 1.49× lift** on a single consumer GPU

<!--
SPEAKER NOTES (Slide 4, ~1 min):
The first pivot was architectural. Instead of running two LLMs, I collapsed both roles into a single open-weight backbone using a structured-output call — one JSON response containing both the state prediction and the teacher utterance. This skips the consultant-to-teacher round trip and removes KV cache duplication, roughly halving wall-clock time. The first full-scale result on the 681-dialogue test split used Qwen 35B-A3B in fusion-think mode and scored 38.70% state accuracy — 12.76 percentage points above GPT-4o, a 1.49× lift, running on the single 5090 at zero API cost. That became our first locked headline.
-->

---

## Methodological Rigor: Smoke / Mini / Full

Cheap-first evaluation cascade. **Never pay full-scale compute until cheaper signals agree.**

| Tier | n | Wall clock | Use |
|---|---:|---:|---|
| Smoke | 5 | ~10–40 min | Triage candidates, sanity-check serving stack |
| Mini | 25 | ~30 min – 1 h | Sharpen ranking, reject bad bets |
| Full | 681 | ~10–22 h | Lock the headline |

- **Smoke + mini average predicted A3B's full-run state accuracy within 0.10 percentage points** (predicted +12.86, realized +12.76)
- Every architectural decision gated on a cheap signal first

<!--
SPEAKER NOTES (Slide 5, ~1 min):
This is methodological — how we decide what to spend GPU hours on. Three tiers: smoke at n=5, mini at n=25, full at n=681. Cheap signals first. The reason: a full n=681 run takes 10 to 22 hours of wall clock. You don't want to pay that price to learn your serving stack is broken or your model is bad. For the A3B model specifically, averaging the smoke and mini state-accuracy lifts predicted the full-run lift within 0.10 percentage points. That's tight. It's a methodological tool we used to gate every architectural decision.
-->

---

## The Gemma 4 31B Retraction

Smoke and mini both suggested Gemma 4 31B would beat A3B at full scale by ~+8 pp. **Full run collapsed: 31.39%** — 15.32 pp below projection.

- Root cause discovered by triangulation: **21% schema-fallback rate at n=681** vs. A3B's 0.91% — the model's JSON-grammar adherence broke at scale
- The smoke and mini samples (each <150 turns) **never surfaced a single fallback**
- **Methodological finding:** smoke–mini averaging is architecture-dependent. Cross-architecture scaling prediction must triangulate schema-fallback rates. **JSON-structured-output dependencies should be replaced with deterministic routing whenever feasible.**
- This is one of the campaign's headline methodological contributions

<!--
SPEAKER NOTES (Slide 6, ~1.5 min):
Here's where the rigor mattered. Gemma 4 31B looked like a slam dunk on smoke and mini — both tiers projected it would beat A3B at full scale by about 8 percentage points. We ran the full eval and it collapsed to 31.39% — fifteen points below projection, and seven points below A3B. We had to triangulate to find the root cause: at full scale, 21 percent of Gemma's outputs failed to match the strict JSON schema we required for the structured-output call. A3B's fallback rate was under 1 percent. The smoke and mini samples — each under 150 turns — surfaced zero fallbacks. So a methodology that worked for A3B failed for Gemma. The lesson generalized: cross-architecture scaling prediction has to triangulate schema-fallback rates, and JSON-structured-output dependencies should be replaced with deterministic routing whenever you can do it. That insight set up the next pivot.
-->

---

## Pivot 2: BERT Consultant Integration

If JSON-grammar adherence is the failure mode, **remove the JSON path** from the consultant.

- Replace the consultant LLM with a **24M-parameter Chinese BERT classifier** (`bge-small-zh-v1.5`)
- Trained on SocratDataset's 42K labeled turns in **92 seconds** — 61.64% test-split state accuracy, **+17.5 pp over the best LLM consultant**
- **Drop-in integration:** BERT consultant + Gemma 4 31B teacher + 10-shot stage-balanced exemplars at n=681 = **48.15% state acc / 36.78 R-1 / unified 68.65**
- **Prior locked headline (2026-05-18):** **+22.21 pp over GPT-4o (1.86×)**, $0 per-run API cost, ~13 GPU-hours, 24-million-parameter classifier doing the routing work that GPT-4o was doing in the original paper

<!--
SPEAKER NOTES (Slide 7, ~1.5 min):
Pivot 2 was the architectural payoff. If JSON-grammar adherence is the failure mode, take JSON off the critical path. I replaced the consultant LLM entirely with a 24-million-parameter Chinese BERT classifier — that's bge-small-zh — trained on the 42,000 labeled turns inside SocratDataset. Training took 92 seconds. The classifier hit 61.64% state accuracy on the test split, beating every LLM consultant we'd measured by more than 17 percentage points. The full integration — BERT consultant plus Gemma 4 31B teacher plus 10-shot stage-balanced exemplars — landed at 48.15% state accuracy on the full 681-dialogue split. That's 22.21 percentage points above GPT-4o, a 1.86x lift, at zero per-run API cost, in 13 GPU-hours of local compute. A 24-million-parameter classifier doing the routing work that GPT-4o does in the original paper. That became the 2026-05-18 locked headline.
-->

---

## Frontier Stress Test

Is teacher capacity the binding constraint? Swap the Gemma teacher for Anthropic's best.

- BERT consultant **kept fixed**, swapped Gemma 4 31B → Claude Sonnet 4.6 and Claude Opus 4.6, each with a 10-shot + top-3-prompt-stack scaffolding
- **Best frontier configuration (n=681):** `bert × Claude-Sonnet · top3` → 49.97% state acc / R-1 41.93 / unified **70.06**
- Frontier within sampling noise of our open-weight on state acc (~+2 pp), +5 on R-1
- **Conclusion: teacher capacity is not the binding constraint.** A well-prompted open-weight teacher matches frontier on the pedagogical axis

<!--
SPEAKER NOTES (Slide 8, ~1 min):
A natural question: is teacher capacity the limit? Are we leaving accuracy on the table by using Gemma instead of Claude? I tested this directly. Kept the BERT consultant fixed and swapped the Gemma teacher for Claude Sonnet 4.6 and Opus 4.6, both with carefully tuned prompt engineering. The best frontier configuration scored 49.97% state accuracy at full scale — within roughly two points of our open-weight integration. Frontier wins on ROUGE-1 by about five points but ties on the pedagogical-routing axis. Conclusion: teacher capacity is not the binding constraint on this benchmark. A well-prompted open-weight teacher matches the frontier on the thing that actually matters for teaching.
-->

---

## The Benchmark Critique — A Methodological Discovery

Comparing surface-form rankings to state-accuracy rankings revealed something striking.

- **Surface-form sum (R-1 + R-2 + BLEU-4) ranks SocratTeachLLM 9B FIRST** by +10.83 over Opus 4.6 with carefully tuned prompts
- **State-accuracy ranks the SAME SocratTeachLLM LAST** (25.94%, below even raw Opus at 39.75%)
- **The two rankings invert.** And the gap widens monotonically with n-gram length: +1.59 R-1, +4.92 R-2, +4.07 BLEU-4
- **Diagnosis:** ROUGE/BLEU measure surface mimicry. SocratTeachLLM was trained on SocratDataset's phrasing. The metrics reward memorization, not teaching.
- **Cross-lingual confirmation:** SocratTeachLLM evaluated on an English translation of the dataset *reproduces* its R-1 = 55.85 headline within 1.5 points of the paper's 57.40, while frontier+prompts loses only 0.07 on the language shift versus SocratTeachLLM losing 1.0 on the same axis

<!--
SPEAKER NOTES (Slide 9, ~2 min):
This is the methodological turn of the campaign. We started ranking configurations by surface-form metrics — ROUGE-1, ROUGE-2, BLEU-4 — and noticed something that didn't add up. SocratTeachLLM, the 9-billion-parameter fine-tune from the original paper, beats Anthropic Opus 4.6 with carefully tuned prompts by over 10 points on surface-form. But the same SocratTeachLLM is dead last on state accuracy — worse than raw Opus with zero prompt engineering. The two rankings invert. That's not just noise: the gap WIDENS monotonically as the n-gram length increases. Plus-1.59 on R-1, plus-4.92 on R-2, plus-4.07 on BLEU-4. That's the strongest possible memorization signature — higher-order n-gram match measures phrase-level fingerprinting. We confirmed the diagnosis with a cross-lingual translation experiment: when we ran SocratTeachLLM against an English translation of the same dataset, it reproduced the original paper's flagship R-1 of 57.40 within 1.5 points. The frontier-plus-prompts configurations only lost 0.07 on the LLM-judge axis crossing languages, while SocratTeachLLM lost a full point. The benchmark is rewarding memorization, not pedagogy.
-->

---

## Contamination Proof

Two independent evidence streams converge.

- **Memorization probe:** 4/288 SocratTeachLLM outputs are character-for-character identical to training data; 17/288 are ≥80% match. Gemma 31B control on the same probe: **0/288** such matches.
- **Clean-probe on synthetic data:** ran SocratTeachLLM against ground-truth dialogues generated by Claude Sonnet (demonstrably outside SocratDataset). State accuracy collapses to **32.86 stage-balanced** — *below Gemma's 56.13 on the same probe*.
- The KELE benchmark, as published, **systematically rewards memorization over pedagogical capability**
- The published "GPT-4o + SocratTeachLLM" baseline R-1 of 44.61 — higher than Opus 4.6 with prompts — is the canonical example of the failure mode

<!--
SPEAKER NOTES (Slide 10, ~1 min):
We didn't stop at suspicion — we ran two independent contamination probes. First: a memorization probe comparing generated outputs to training data. Four of 288 SocratTeachLLM outputs were character-for-character identical to training data. Seventeen of 288 were 80%-or-more matches. Gemma's control on the same probe: zero. Second probe: I had Claude Sonnet generate fresh synthetic dialogues that were demonstrably outside SocratDataset's training distribution. SocratTeachLLM's state accuracy collapsed to 32.86 stage-balanced on the clean probe — that's BELOW Gemma's 56.13 on the exact same probe. The benchmark, as published, rewards memorization. The fact that GPT-4o-plus-SocratTeachLLM beats every frontier configuration on ROUGE-1 in their paper isn't a finding about teaching capability — it's a benchmark artifact.
-->

---

## The Unified Metric

Six per-cell metrics, no single defensible headline. We collapsed them.

$$
\text{unified} = 0.5 \cdot \text{stage\_balanced} + 0.5 \cdot (\text{judge} \cdot 10)
$$

- **`stage_balanced`** — equal-weight per-stage state accuracy (corrects the published macro's structural under-counting of stage e, closure)
- **`judge`** — Claude Sonnet 4.6 rubric on 4 axes (Socratic validity / advancement / age-appropriateness / question-form fidelity), 0–10 scale
- Both inputs are **memorization-resistant by construction** — surface-form metrics are *excluded*, per the critique
- Master leaderboard: **143 configurations, 38 LLM-judged**, auto-regenerated by `scripts/backtest_stage_balanced.py`

<!--
SPEAKER NOTES (Slide 11, ~1.5 min):
Once we'd rejected surface-form metrics, we still had a problem: six per-cell metrics, no single defensible single-number headline. So we built one. Unified equals one-half stage-balanced state accuracy, plus one-half judge score times ten. Stage-balanced is the equal-weight per-stage state accuracy — that corrects the published macro's structural under-counting of the closure stage, which is rare in the test split but pedagogically load-bearing. Judge is the LLM-judge composite — Claude Sonnet 4.6 scoring our outputs on four axes: socratic validity, advancement of student reasoning, age-appropriateness, and question-form fidelity. Both inputs are memorization-resistant by construction; surface-form metrics are excluded per the critique. The master leaderboard now sits at 143 configurations, 38 of them LLM-judged, all auto-regenerated by a backtest script.
-->

---

## Consultant Upgrade + TODO #14

The consultant upgrade campaign produced a **Qwen3.5-0.8B-LoRA classifier** — methodological successor to the 24M BERT — and TODO #14 lined up a 4-cell canonical-scale sub-leaderboard at n=681 to confirm screening-tier parity.

| Cell | unified | Status |
|---|---:|---|
| `qwen3.5 × A3B-35B · n=681` | 67.81 | ✅ 2026-05-25 (9h 41m) |
| `qwen3.5 × Qwen-27B no-think · n=681` | 66.71 | ✅ 2026-05-25 (1h 4m) |
| `bert-fixed × Gemma-31B · n=681` | — | queued, ~12 GPU-h |
| **`qwen3.5 × Gemma-31B · n=681`** | **?** | **today, ~12 GPU-h** |

<!--
SPEAKER NOTES (Slide 12, ~45s):
The consultant upgrade campaign produced a successor to the BERT classifier — a LoRA fine-tune of Qwen3.5-0.8B-Base. Same methodological idea, larger backbone, post-fix consultant input format. TODO 14 was a four-cell canonical-scale sub-leaderboard at n=681, designed to confirm the screening-tier parity finding at full sample size. Three cells were done before today: A3B at 67.81, Qwen-27B no-think at 66.71, plus one queued. Today's run was the fourth: qwen3.5 cross Gemma-31B at n=681.
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
SPEAKER NOTES (Slide 13, ~1.5 min):
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
SPEAKER NOTES (Slide 14, ~1 min):
The campaign by the numbers. 143 distinct configurations measured. 38 of them LLM-judged. Seven full n=681 runs at canonical scale. Adding up every named wall-clock from the paper, the experiment log, and the consultant-upgrade log gives 119 and a half GPU-hours of confirmed compute — roughly five days of continuous wall clock on the 5090, plus another fifteen to twenty-five GPU-hours of un-itemized smoke and mini runs we didn't time precisely. The biggest single block is the seven full n=681 runs at 77 hours; the next biggest is the 13-model tournament at 14 hours. API spend with Anthropic totaled 258 dollars and 86 cents. Three line items there: frontier-teacher comparisons at n=681 with Claude Sonnet and Opus as the teacher; the Phase 2 frontier-teacher n=50 sweeps stress-testing prompt scaffolding; and the LLM-judge passes that produce the unified score. The open-weight eval pipeline itself is zero dollars per-run — every API dollar bought either a frontier comparison or a memorization-resistant judgment. The progression at the bottom shows the unified-score rank-1 cell as it shifted: A3B fusion in early May, retraction of standalone Gemma when we discovered the schema-fallback issue, BERT integration becoming the locked headline on the 18th of May, then a series of canonical-scale cells in late May ending with today's frontier overtaking.
-->

---

## Three Contributions

1. **Fusion architecture** — practical primitive for KELE-style multi-agent pipelines on consumer hardware. Single backbone, structured-output call, ~2× faster than two-call.
2. **Classifier-as-consultant integration** — deterministic state routing × LLM-driven response generation, two independently-optimizable axes. Replaces fragile JSON-on-LLM with a 24M-param BERT (or 800M-param Qwen3.5-LoRA) classifier that runs at zero marginal VRAM cost.
3. **Benchmark critique + unified metric** — surface-form metrics on SocratDataset reward memorization, not teaching. The unified metric (`0.5 × stage_balanced + 0.5 × judge × 10`) is what surfaces the frontier-overtaking result.

**Code, data, paper draft:** github.com/ulises-c/csen-346 · 🤗 ulises-c/SocratDataset · 🤗 ulises-c/SocratDataset-EN

**Questions?**

<!--
SPEAKER NOTES (Slide 15, ~30s):
Three takeaways from the campaign. First, the fusion architecture as a practical primitive for KELE-style multi-agent pipelines on consumer hardware. Second, the classifier-as-consultant decomposition — deterministic routing and LLM generation as two independently-optimizable axes — that's the architectural contribution. Third, the methodological reframing: surface-form metrics reward memorization on this benchmark; the unified metric, by construction memorization-resistant, is what made the frontier-overtaking visible. The code, data, and paper draft are public. Happy to take questions.
-->
