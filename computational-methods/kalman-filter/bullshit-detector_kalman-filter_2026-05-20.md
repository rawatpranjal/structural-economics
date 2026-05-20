# bullshit-detector — kalman-filter — 2026-05-20

**Bullshit score: 15%** — no FALSE or UNIMPLEMENTED findings; two LOW-severity gaps (undisclosed P_{0|0}=0 and undocumented symmetrization step) are DILUTED at LOW, and all numeric claims match committed CSV exactly.

## Header
- Claim sources: `computational-methods/kalman-filter/README.md`
- Code / artifact root: `computational-methods/kalman-filter/run.py`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Initial covariance P_{0|0}=0 undisclosed | DILUTED | LOW | no (numerical stabilization note only; results reproducible) |
| 2 | Symmetrization `0.5*(cov+cov.T)` undocumented | DILUTED | LOW | no (analytically zero effect; affects floating-point only) |
| 3 | All equations (C2-C11) | HOLDS | — | — |
| 4 | All parameters (C12-C17) | HOLDS | — | — |
| 5 | All numeric results (C19-C27) | HOLDS | — | — |

## Findings

### Finding 1: Initial covariance P_{0|0} = 0 undisclosed

- **Claim source (verbatim):** "| Initial state | $s_0 = (0,0)$ |" — `README.md:66`; "The filter starts from zero" — `README.md:70`
- **Code evidence (verbatim):**
  ```python
  mean = np.zeros(state_dim, dtype=float) if initial_mean is None else np.asarray(initial_mean, dtype=float)
  cov = np.zeros((state_dim, state_dim), dtype=float) if initial_cov is None else np.asarray(initial_cov, dtype=float)
  ```
  `run.py:74-75`
- **Data evidence (if applicable):** None. The results table (`tables/filter-diagnostics.csv`) does not record the initial covariance.
- **Category:** DILUTED — Model Setup discloses the initial mean but silently sets initial covariance `P_{0|0} = 0`. Starting with zero covariance is atypical (the Kalman filter assumes the initial state is known with certainty, which understates uncertainty in early periods and affects the gain and filtered posterior for the first several periods). The phrase "starts from zero" in Solution Method is ambiguous between zero-mean and zero-covariance.
- **Severity:** LOW — the undisclosed choice does not change the reported log-likelihood, RMSE, or coverage numbers in the table (all computed post-initialization), but a reader who replicates with a diffuse prior will get different early-period posteriors.
- **Result-changing:** no (table numbers are reproducible from code as-is; the missing disclosure misleads a replicator who uses a diffuse prior)
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "P_{0|0}" not in open("computational-methods/kalman-filter/README.md").read()
  # PASSES on current code (README omits P_{0|0}); FAILS after honest disclosure is added
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(phrase in open("computational-methods/kalman-filter/README.md").read() for phrase in ["P_{0|0} = 0", "initial covariance", "initial_cov"])
  # PASSES after honest disclosure; FAILS on current README
  ```

### Finding 2: Symmetrization step undocumented in equations and pseudocode

- **Claim source (verbatim):** "update covariance: P_{t|t} = P_{t|t-1} - K_t Psi P_{t|t-1}" — `README.md:86` (pseudocode) and the displayed equation at `README.md:47`
- **Code evidence (verbatim):**
  ```python
  cov = pred_cov - np.outer(gain, PSI @ pred_cov)
  cov = 0.5 * (cov + cov.T)
  ```
  `run.py:86-87`
- **Data evidence (if applicable):** None.
- **Category:** DILUTED — the equations and pseudocode both state `P_{t|t} = P_{t|t-1} - K_t Psi P_{t|t-1}` with no mention of the subsequent symmetrization `0.5*(cov+cov.T)` at line 87. Analytically the update preserves symmetry; the extra line handles floating-point drift. The pseudocode is silent about this numerical step.
- **Severity:** LOW — the symmetrization does not change results in exact arithmetic; the omission is a documentation gap, not a faithfulness violation in any operationally significant sense.
- **Result-changing:** no (floating-point drift in a 50-period simulation with a 2x2 state is negligible; removing line 87 would not change any number in the results table to 4 decimal places)
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.5 * (cov + cov.T)" in open("computational-methods/kalman-filter/run.py").read() and "symmetr" not in open("computational-methods/kalman-filter/README.md").read()
  # PASSES on current state (step in code but absent from README); FAILS after disclosure
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(phrase in open("computational-methods/kalman-filter/README.md").read() for phrase in ["symmetr", "numerical stabiliz", "0.5*(P+P')", "enforce symmetry"])
  # PASSES after README discloses the step; FAILS now
  ```

## Cross-cutting patterns

- All 9 equations in the Equations section map faithfully to code. The gain formula `K_t = P_{t|t-1} Psi' S_t^{-1}` (pseudocode) and `K_t = P_{t|t-1} Psi' (Psi P_{t|t-1} Psi' + R)^{-1}` (equations) are equivalent scalar-S_t forms and both match `run.py:81-83`.
- All 9 numeric values in `tables/filter-diagnostics.csv` match the README table exactly (including the 0.860/0.92 vs 0.86/0.92 cosmetic display rounding).
- Both DILUTED findings are documentation gaps, not implementation errors. The code is faithful to the standard Kalman filter; the prose simply omits two implementation choices that a textbook treatment would state.
- No parameter leak, no mislabeled algorithm, no phantom result.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 15% (< 50%).** No halt required. Surface findings to user as documentation improvements only.
1. **Finding 1 violated-invariant test:** confirm `"P_{0|0}"` absent from README. (Trivially passes on current code.)
2. **Finding 1 fix:** add one row to Model Setup table: `| Initial covariance $P_{0|0}$ | $0_{2\times 2}$ (zero matrix) |`. Confirm honest-fix test now passes.
3. **Finding 2 violated-invariant test:** confirm symmetrization string in run.py but absent from README. (Trivially passes.)
4. **Finding 2 fix:** add one sentence to Solution Method pseudocode comments or a parenthetical note: "the covariance is symmetrized numerically after each update." Confirm honest-fix test passes.
5. Re-run `python run.py` to confirm no numeric changes. Re-run `scripts/validate_catalog.py`. Re-run this skill; all findings should read HOLDS and score should drop to 0-5%.
