# CI / Automation — Future Work

Items that would improve the pipeline but were deferred for scope or complexity reasons.

---

## 1. Coverage threshold (`--cov-fail-under`)

**What:** Add a minimum coverage floor to `pytest --cov` so CI fails if overall coverage drops
below a set percentage.

**Why deferred:** Current overall coverage is ~48%, dragged down by modules that require live
GPU/LLM calls (`socratic_teaching_system.py`, `socratic_teaching_unified.py`,
`translate_dataset.py`) and are not realistically unit-testable in CI without extensive mocking.
Setting a global floor at the current baseline would be fragile; it would either be set too low to
be useful or high enough to block any new untested code.

**Recommended approach when ready:**
- Add per-module coverage exclusions for LLM-call-heavy files (via `# pragma: no cover` on
  the entry-point functions, or by omitting them in `[tool.coverage.run] omit`).
- Set a floor on the *testable* surface only (`config`, `metrics`, `evaluate`, `kele`,
  `serve_teacher`), which currently runs 74–96%.
- Use `--cov-fail-under=70` as a starting point once the untestable modules are excluded.

```toml
# pyproject.toml
[tool.coverage.run]
omit = [
    "src/project/translate_dataset.py",
    "src/project/socratic_teaching_system.py",
    "src/project/socratic_teaching_unified.py",
]
```

```yaml
# ci.yml
- name: Test
  run: uv run pytest --cov --cov-report=xml --cov-report=term-missing --cov-fail-under=70
```

---

## 2. Branch protection rules

**What:** Require both `Lint` and `Test` CI jobs to pass before any PR can be merged into `main`.

**Why deferred:** This is a GitHub repository settings change (not a file in the repo), so it
cannot be implemented via a commit. Requires a maintainer to configure it in the UI.

**Steps to enable:**
1. Go to **Settings → Branches → Add branch ruleset** (or classic branch protection).
2. Set the branch pattern to `main`.
3. Under **Require status checks to pass before merging**, add:
   - `Lint`
   - `Test`
4. Enable **Require branches to be up to date before merging** to prevent stale-branch merges.
5. Optionally enable **Require linear history** to keep `git log` clean.

---

## 3. `SECURITY.md`

**What:** A minimal security policy that tells contributors (and GitHub's vulnerability scanner)
where to report security issues privately.

**Why deferred:** Low urgency for a course research project; primarily relevant once model weights
and API key handling are in wider use.

**Recommended content** (create `.github/SECURITY.md` or `SECURITY.md` at repo root):

```markdown
# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately via GitHub's [private vulnerability reporting](https://github.com/ulises-c/csen-346/security/advisories/new)
or email ulises.engineer@gmail.com.

## Scope

Key areas to audit:
- API key handling in `src/project/config.py` (keys are read from `.env`, never logged)
- `serve_teacher.py` authentication (`TEACHER_SERVER_API_KEY` env var)
- Any shell scripts that interpolate user-controlled input
```

GitHub will surface a **"Set up a security policy"** prompt on the repository's Security tab
until this file is added.
