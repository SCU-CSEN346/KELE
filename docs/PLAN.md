# Project Plan — KELE Reproduction & Extension

**CSEN 346 · Santa Clara University**

This doc holds the **course-deliverable schedule and final-submission checklist**. The original Phase-1-through-Phase-5 body (2026-04-14) is superseded by reality and was removed in the 2026-05-22 doc audit — see git history for the original if you need it.

For the **current state of the work** read these first:

- [`../README.md`](../README.md) — headline results, leaderboards, architecture
- [`../deliverables/overleaf/latex/acl_latex.tex`](../deliverables/overleaf/latex/acl_latex.tex) — the paper
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — dated engineering ledger
- [`PROMPT_ENGINEERING_PLAN.md`](PROMPT_ENGINEERING_PLAN.md) and [`CLAUDE_API_TEACHER_PLAN.md`](CLAUDE_API_TEACHER_PLAN.md) — active plans-of-record
- [`BENCHMARK_CRITIQUE_AND_PROPOSAL.md`](BENCHMARK_CRITIQUE_AND_PROPOSAL.md) — the methodological contribution

---

## Course Deadlines

| Date               | Deliverable                                                                    |
| ------------------ | ------------------------------------------------------------------------------ |
| **Apr 14** | 1st documented GitHub commit + Paper: Intro & Related Work                          |
| **Apr 23** | 2nd documented GitHub commit + Paper: Dataset & Methodology                         |
| **May 5**  | 3rd documented GitHub commit + Paper: Evaluation & Results                          |
| **May 14** | 4th documented GitHub commit + Paper: Results, Intro, Conclusion, Limitations, Ethics |
| **May 26** | Demo & Present (~3 groups)                                                          |
| **May 28** | Demo & Present (~3 groups)                                                          |
| **Jun 2**  | Demo & Present (~3 groups)                                                          |
| **Jun 4**  | Final paper + final code + HuggingFace data + poster                                |

---

## Final-submission checklist

### Paper (4–6 pages, ACL template)

- [x] Introduction & Related Work — due Apr 14
- [x] Dataset & Methodology — due Apr 23
- [x] Evaluation & Results — due May 5
- [x] Conclusion, Limitations, Ethics — due May 14
- [ ] Final polish — due Jun 4
- [ ] Run through [Agentic Reviewer by Andrew Ng](https://www.agentic-reviewer.com/) before submitting

### Code submission checklist

- [x] Docstrings on all functions and classes
- [x] `.env.example` with required API key names documented
- [x] README: model description, installation, usage, expected output, member contributions
- [x] HuggingFace dataset card for any data artifacts

### Demo

- [ ] Record a short walkthrough of the system handling a full 5-stage dialogue
- [ ] Host on HuggingFace Spaces, YouTube, or Google Drive
- [ ] Reference the demo link in the paper

### Poster

- [ ] Follow the guideline: more images, minimal text
- [ ] Key panels: problem, KELE architecture, our improvement, results comparison table

---

## Division of Work (suggested)

| Member   | Primary Area                                       |
| -------- | -------------------------------------------------- |
| Member 1 | Baseline setup, serving SocratTeachLLM, API config |
| Member 2 | Evaluation pipeline (ROUGE/BLEU + LLM-as-judge)    |
| Member 3 | Improvement (classifier consultant), paper writing |

All members contribute to paper writing and final presentation.
