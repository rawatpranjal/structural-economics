# bullshit-detector - projection-methods - recheck - 2026-05-20

**Bullshit score: 0%** - all claims HOLD against code and CSV; the sole original finding (prose "between collocation nodes") resolved to "over the full approximation interval" at run.py:390.

## Header
- Claim sources: `computational-methods/projection-methods/README.md`
- Code / artifact root: `computational-methods/projection-methods/run.py`
- Data artifacts: `computational-methods/projection-methods/tables/projection-accuracy.csv`
- Seed audit: `computational-methods/projection-methods/bullshit-detector_projection-methods_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "dense grid over the full approximation interval" (prior Finding 1, fixed) | HOLDS | - | - |
| 2 | steady-state capital = 0.1870 | HOLDS | - | - |
| 3 | interval [0.0468, 0.3273] | HOLDS | - | - |
| 4 | exact policy g*(k) = alpha*beta*A*k^alpha | HOLDS | - | - |
| 5 | Euler residual R_i formula | HOLDS | - | - |
| 6 | log-linear Chebyshev parameterization sum_{j=0}^{n-1} theta_j T_j | HOLDS | - | - |
| 7 | table values all 4 basis sizes (n=2,3,5,8) match CSV | HOLDS | - | - |
| 8 | prose "max Euler error 5.57e-04" | HOLDS | - | - |
| 9 | pseudocode steps 1-7 vs code | HOLDS | - | - |
| 10 | full-depreciation Cobb-Douglas technology | HOLDS | - | - |

## Findings

None. All claims verified against run.py and projection-accuracy.csv.

**Prior Finding 1 resolved.** `run.py:390` now reads: `"The table evaluates errors on a dense grid over the full approximation interval."` The prior phrase "between collocation nodes" is gone. `README.md:102` matches. The eval grid `np.linspace(lower, upper, 320)` at `run.py:191` spans [0.0468, 0.3273] inclusive - consistent with corrected prose.

## Cross-cutting patterns

- No systematic mislabeling or drift. All intermediate values (k_ss, interval endpoints, table numerics) generated live from code constants - no hardcoded drift risk.
- All four table rows match committed CSV to all significant figures printed.
- Log Euler residual in code (`run.py:94-95`) matches displayed equation (`README.md:43-48`) symbol-for-symbol.
- Chebyshev basis indexing (`chebvander(x, n_basis - 1)` at `run.py:61`) consistent with `sum_{j=0}^{n-1}` notation in README.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%. No action required.**
1. Original violated-invariant finding resolved. Prose now says "over the full approximation interval."
2. Existing tests `tests/test_projection-methods.py` pass 2/2. No new tests needed.
3. No sim re-runs needed; no data artifacts changed by this fix.
