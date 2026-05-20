# bullshit-detector — revealed-preference-afriat — recheck — 2026-05-20

**Bullshit score: 0%** — All claims hold. All three original non-HOLDS findings resolved: (F1 MISLABELED) title and Solution Method now name the GARP test, not Afriat's test; (F2 DILUTED) pseudocode and prose now disclose TOL=1e-10; (F4 DILUTED) prose now discloses the 200-attempt swap-retry loop and the hardcoded fallback. Finding 3 (DATA DRIFT, LOW) is also resolved: the violation count is now injected dynamically via f-string and cannot drift between runs.

## Header
- Claim sources: `choice/revealed-preference-afriat/README.md`
- Code / artifact root: `choice/revealed-preference-afriat/run.py`
- Seed audit: `bullshit-detector_revealed-preference-afriat_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Title: "Consumer Rationalizability with the GARP Test" | HOLDS | — | — |
| 2 | Test does not construct Afriat inequalities or utility fn | HOLDS | — | — |
| 3 | GARP pseudocode step 1: expenditure + TOL >= cost | HOLDS | — | — |
| 4 | Pseudocode step 4: violation if exp_j > cost + TOL | HOLDS | — | — |
| 5 | Both budget comparisons use TOL = 1e-10 | HOLDS | — | — |
| 6 | Swap-retry loop runs up to 200 attempts | HOLDS | — | — |
| 7 | If no swap fails, hardcoded fallback returned | HOLDS | — | — |
| 8 | At committed seed, swap path succeeds | HOLDS | — | — |
| 9 | Cobb-Douglas weights 0.337, 0.328, 0.335 | HOLDS | — | — |
| 10 | Corrupted sample: 2 violations | HOLDS | — | — |
| 11 | Rational benchmark: 0 violations | HOLDS | — | — |
| 12 | Warshall triple-loop transitive closure | HOLDS | — | — |
| 13 | GARP violation condition iR*j and m_j > p_j.x_i | HOLDS | — | — |
| 14 | Afriat's theorem: passing GARP equivalent to rationalizability | HOLDS | — | — |
| 15 | Tutorial stops at pass-or-fail; no Afriat inequalities constructed | HOLDS | — | — |

## Findings

### Original Finding 1 (from seed audit): "Afriat's test" label without Afriat inequalities — RESOLVED

- **Original claim:** Title and Solution Method said "Afriat's test" but code only does GARP checking.
- **Current state:** `run.py:317-321`: `report = ModelReport("Consumer Rationalizability with the GARP Test", ...)`. README title is "# Consumer Rationalizability with the GARP Test". Solution Method at `run.py:374-380` says "The test returns a pass or fail decision; it does not construct the Afriat inequalities or a utility function." No `lambda_t`, `ccei`, `linprog`, or Afriat-inequality logic anywhere in `run.py`.
- **Resolution:** RESOLVED. MISLABELED → HOLDS.

### Original Finding 2 (from seed audit): Pseudocode omitted numerical tolerance — RESOLVED

- **Original claim:** Pseudocode steps 1 and 4 stated the comparison without TOL.
- **Current state:** Pseudocode at `run.py:383-389` now reads "set R[i,j] = 1 if p_i . x_i + TOL >= p_i . x_j" (step 1) and "flag a violation if p_j . x_j > p_j . x_i + TOL" (step 4). Prose at `run.py:391-394` additionally explains "Both budget comparisons use a numerical tolerance TOL = 1e-10."
- **Resolution:** RESOLVED. DILUTED → HOLDS.

### Original Finding 3 (from seed audit): Static "2 violations" — DATA DRIFT RISK ELIMINATED

- **Original claim:** Violation count was a static string; DATA DRIFT LOW risk if seed changed.
- **Current state:** `run.py:369,402` now injects `{len(violations_inc)}` via f-string. Count cannot drift between `run.py` runs. At seed=42, count = 2. HOLDS structurally.
- **Resolution:** Risk eliminated. DATA DRIFT → HOLDS.

### Original Finding 4 (from seed audit): Undisclosed fallback path — RESOLVED

- **Original claim:** README did not disclose the 200-attempt retry loop or the hardcoded fallback.
- **Current state:** Prose at `run.py:395-401` says "the loop runs up to 200 attempts ... If no swap fails within 200 attempts, the code returns a hardcoded fallback dataset with a known violation. At the committed seed the swap path succeeds, so the figures show swap-corrupted data."
- **Resolution:** RESOLVED. DILUTED → HOLDS.

## Cross-cutting patterns

None. All four original non-HOLDS findings resolved. Code matches all claimed descriptions. No new gaps found.

## TDD execution sequence (for the next agent)

None required. Score is 0%. All three honest-fix tests now pass and all violated-invariant tests correctly fail:
- `assert "GARP" in README.splitlines()[0] and "Afriat's Test" not in README.splitlines()[0]` — PASSES.
- `assert "tolerance" in README.lower() or "1e-10" in README` — PASSES.
- `assert "fallback" in README.lower() or "attempt" in README.lower()` — PASSES.
