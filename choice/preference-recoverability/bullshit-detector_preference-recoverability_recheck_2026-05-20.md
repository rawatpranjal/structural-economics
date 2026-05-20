# bullshit-detector — preference-recoverability — recheck — 2026-05-20

**Bullshit score: 0%** — All claims hold. Both original DILUTED findings are resolved: (1) the Overview now correctly states the LP only finds utility scores, with slopes pre-fixed; (2) pseudocode step 4 now states "sum_t u_t = T" matching the code constraint b_eq = [float(n_obs)]. All numeric results hold.

## Header
- Claim sources: `choice/preference-recoverability/README.md`
- Code / artifact root: `choice/preference-recoverability/run.py`
- Data artifacts: `choice/preference-recoverability/tables/afriat-numbers.csv`
- Seed audit: `bullshit-detector_preference-recoverability_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | LP finds utility scores (not slopes) | HOLDS | — | — |
| 2 | Supporting slopes fixed to 1/expenditure before LP runs | HOLDS | — | — |
| 3 | LP only finds utility scores | HOLDS | — | — |
| 4 | T=18, 2 goods, alpha=0.60 | HOLDS | — | — |
| 5 | Income range [5.07, 13.33] | HOLDS | — | — |
| 6 | Price range [0.57, 1.96] | HOLDS | — | — |
| 7 | GARP violations = 0 | HOLDS | — | — |
| 8 | Max Afriat residual = 1.30e-15 | HOLDS | — | — |
| 9 | Target observation = 7 | HOLDS | — | — |
| 10 | lambda_t = 1/(p_t . x_t) | HOLDS | — | — |
| 11 | Pseudocode: sum_t u_t = T | HOLDS | — | — |
| 12 | U_hat = min_j [u_j + lambda_j p_j . (y - x_j)] | HOLDS | — | — |
| 13 | Frontier = max_j of per-observation supports | HOLDS | — | — |
| 14 | Correlation = 0.973 | HOLDS | — | — |
| 15 | Median contour ratio = 0.86 | HOLDS | — | — |
| 16 | Max contour gap = 9.89 | HOLDS | — | — |
| 17 | Table CSV matches README table (all 18 rows) | HOLDS | — | — |

## Findings

### Original Finding 1 (from seed audit): Overview claimed LP finds supporting slopes — RESOLVED

- **Original claim:** "A linear program finds utility scores and supporting slopes." — `README.md:9` (old).
- **Current state:** `run.py:255-259` now generates: "The computation uses Afriat inequalities. The supporting slopes are fixed to one over expenditure before the program runs, so a linear program only finds the utility scores." The LP at `run.py:123-131` has `c=np.zeros(n_obs)` with `n_obs` decision variables (utility scores only); lambdas are pre-fixed at `run.py:103`.
- **Resolution:** RESOLVED. DILUTED → HOLDS.

### Original Finding 2 (from seed audit): Pseudocode said "average_t u_t = 1" vs code sum = n_obs — RESOLVED

- **Original claim:** pseudocode step 4 said "average_t u_t = 1".
- **Current state:** pseudocode step 4 at `run.py:344-346` now reads: "sum_t u_t = T, and u_t >= 0." Code at `run.py:120-121`: `a_eq = np.ones((1, n_obs))` / `b_eq = np.array([float(n_obs)])` — constraint is exactly sum = n_obs = T = 18.
- **Resolution:** RESOLVED. DILUTED → HOLDS.

## Cross-cutting patterns

None. All claims verified end-to-end. Numeric outputs injected via f-strings from computed variables; CSV cross-check confirms all 18 table rows match exactly.

## TDD execution sequence (for the next agent)

None required. Score is 0%. Both honest-fix tests now pass:
- `assert "finds utility scores and supporting slopes" not in text and "fixed to one over expenditure before the program runs" in text` — PASSES.
- `assert "average_t u_t = 1" not in text and "sum_t u_t = T" in text` — PASSES.
Both violated-invariant tests now correctly fail, confirming fixes are in place.
