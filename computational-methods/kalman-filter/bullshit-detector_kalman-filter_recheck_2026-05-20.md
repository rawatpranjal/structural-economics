# bullshit-detector — kalman-filter — recheck — 2026-05-20

**Bullshit score: 0%** — both original DILUTED findings are now resolved; all equations, parameters, and numeric results hold against code and CSV; no new gaps found.

## Header
- Claim sources: `computational-methods/kalman-filter/README.md`
- Code / artifact root: `computational-methods/kalman-filter/run.py`
- Data artifacts: `computational-methods/kalman-filter/tables/filter-diagnostics.csv`
- Seed audit (if any): `computational-methods/kalman-filter/bullshit-detector_kalman-filter_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | P_{0|0} = 0 disclosed in Model Setup | HOLDS | — | — |
| 2 | Symmetrization disclosed in pseudocode and prose | HOLDS | — | — |
| 3 | All equations (prediction, update, gain, likelihood) | HOLDS | — | — |
| 4 | All parameters (PSI, PHI, stds, periods) | HOLDS | — | — |
| 5 | All numeric results (RMSE, MAE, std, coverage, loglike) | HOLDS | — | — |

## Findings

### Finding 1 (original): P_{0|0} = 0 undisclosed — NOW HOLDS

- **Claim source (verbatim):** `"| Initial state covariance $P_{0|0}$ | $0_{2\times 2}$ (zero matrix) |"` — `README.md:67`
- **Code evidence (verbatim):**
  ```python
  cov = np.zeros((state_dim, state_dim), dtype=float) if initial_cov is None else np.asarray(initial_cov, dtype=float)
  ```
  `run.py:75`
- **Category:** HOLDS — the Model Setup table now explicitly names `P_{0|0}` and gives its value as `0_{2×2}`. The disclosure also appears in Solution Method prose at `README.md:92`: "The filter starts from $P_{0|0} = 0$, treating the initial state as known with certainty, so the early-period posteriors understate uncertainty relative to a diffuse prior."
- **Original finding resolved:** yes.

### Finding 2 (original): Symmetrization undocumented — NOW HOLDS

- **Claim source (verbatim):** `"    symmetrize:         P_{t|t} <- 0.5 (P_{t|t} + P_{t|t}')"` — `README.md:88`; `"the explicit symmetrization step replaces $P_{t|t}$ with $0.5(P_{t|t} + P_{t|t}')$ to cancel floating-point drift"` — `README.md:92`
- **Code evidence (verbatim):**
  ```python
  cov = pred_cov - np.outer(gain, PSI @ pred_cov)
  cov = 0.5 * (cov + cov.T)
  ```
  `run.py:86-87`
- **Category:** HOLDS — the pseudocode now contains an explicit `symmetrize` step at line 88 of the README, and the following prose paragraph explains both what the step does and why (floating-point drift). Code and documentation are consistent.
- **Original finding resolved:** yes.

### Numeric cross-check

All values in `tables/filter-diagnostics.csv` match `README.md:112-116` exactly:

| State | RMSE | Mean abs error | Mean posterior std | 90% band coverage |
|-------|------|----------------|--------------------|-------------------|
| s1 | 0.2297 | 0.1754 | 0.2114 | 0.86 |
| s2 | 0.2447 | 0.2106 | 0.2286 | 0.92 |
| log likelihood | — | — | — | -25.73 |

CSV rows match: `s1,0.2297,0.1754,0.2114,0.860` and `s2,0.2447,0.2106,0.2286,0.920` and `log likelihood,,,,-25.73` — `tables/filter-diagnostics.csv:2-4`. No drift.

## Cross-cutting patterns

- Both original findings were documentation gaps, not implementation errors. The fix added one Model Setup table row and one pseudocode line plus an explanatory prose sentence. No code changed; no numbers changed.
- The recheck found no new gaps. The equations (prediction, update, gain formula, likelihood increment), all six parameter values, and all nine numeric cells in the results table remain faithful to code.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required. Both honest-fix tests now pass; no violated-invariant tests remain meaningful (the invariants they tested no longer hold in the code). Ship.
1. The test suite at `tests/test_kalman-filter.py` confirms both fixes:
   - `test_f1_honest_fix_readme_discloses_initial_covariance` PASSES (README contains `P_{0|0}`).
   - `test_f2_honest_fix_readme_documents_symmetrization` PASSES (README contains `symmetr`).
2. No further action needed for this tutorial.
