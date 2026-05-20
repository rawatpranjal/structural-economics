# bullshit-detector — perturbation-linearization — recheck — 2026-05-20

**Bullshit score: 0%** — the original MISLABELED finding is resolved: algorithm step 1 now reads "Read the Taylor coefficients of F at x_bar straight from the known parameters" (`run.py:195`), which accurately describes `perturbation_transition`'s direct use of RHO, GAMMA, ETA; all equations, parameters, and all 24 numeric table cells hold against code and CSV.

## Header
- Claim sources: `computational-methods/perturbation-linearization/README.md`
- Code / artifact root: `computational-methods/perturbation-linearization/run.py`
- Data artifacts: `computational-methods/perturbation-linearization/tables/approximation-errors.csv`
- Seed audit (if any): `computational-methods/perturbation-linearization/bullshit-detector_perturbation-linearization_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Algorithm step 1 describes reading coefficients directly, not differentiation | HOLDS | — | — |
| 2 | Exact transition F(x) = rho*x + gamma*x^2 - eta*x^3 + kappa*x^4 | HOLDS | — | — |
| 3 | F_1 = rho*x; F_2 adds gamma*x^2; F_3 adds -eta*x^3 | HOLDS | — | — |
| 4 | rho=0.82, gamma=0.45, eta=0.80, kappa=0.35 | HOLDS | — | — |
| 5 | Shock size 0.18 in either direction; IRF periods 28 | HOLDS | — | — |
| 6 | kappa term left out by third-order perturbation | HOLDS | — | — |
| 7 | local domain abs(x) <= 0.20; wide domain abs(x) <= 0.60 | HOLDS | — | — |
| 8 | All 24 README table cells match tables/approximation-errors.csv | HOLDS | — | — |
| 9 | First-order map is symmetric; positive and negative responses cancel | HOLDS | — | — |
| 10 | Asymmetry = positive path + negative path; nonzero sum measures nonlinearity | HOLDS | — | — |
| 11 | Higher orders lower error near steady state | HOLDS | — | — |

## Findings

### Finding 1 (original): Algorithm step 1 claims "Differentiate F" — NOW HOLDS

- **Claim source (verbatim):** `"1. Read the Taylor coefficients of F at x_bar straight from the known parameters"` — `README.md:63` (and `run.py:195`)
- **Code evidence (verbatim):**
  ```python
  def perturbation_transition(x: np.ndarray | float, order: int) -> np.ndarray | float:
      """Taylor approximation to the transition rule around x=0."""
      z = np.asarray(x)
      out = RHO * z
      if order >= 2:
          out = out + GAMMA * z**2
      if order >= 3:
          out = out - ETA * z**3
      return out
  ```
  `run.py:34-42`
- **Category:** HOLDS — the pseudocode step 1 now says "Read the Taylor coefficients of F at x_bar straight from the known parameters." This matches the implementation: `perturbation_transition` reads `RHO`, `GAMMA`, `ETA` directly from module-level constants (`run.py:21-23`) without any call to a differentiation function. No symbolic differentiation, no `np.gradient`, no `scipy.misc.derivative`, no `jax.grad` appears anywhere in the function. The original MISLABELED finding ("Differentiate F at x_bar and keep terms through order n") is gone from both `run.py` (the source string) and `README.md` (the generated output).
- **Original finding resolved:** yes.

### Numeric cross-check: all 24 cells verified

`tables/approximation-errors.csv` vs `README.md:97-102`:

| Order | Domain | Max map error | Median map error | +IRF RMSE | -IRF RMSE |
|-------|--------|--------------|------------------|-----------|-----------|
| 1 | local | 2.45e-02 / 0.0245 ✓ | 4.33e-03 / 0.00433 ✓ | 1.16e-02 / 0.0116 ✓ | 1.17e-02 / 0.0117 ✓ |
| 1 | wide | 3.80e-01 / 0.38 ✓ | 2.89e-02 / 0.0289 ✓ | 1.16e-02 / 0.0116 ✓ | 1.17e-02 / 0.0117 ✓ |
| 2 | local | 6.79e-03 / 0.00679 ✓ | 7.93e-04 / 0.000793 ✓ | 5.01e-03 / 0.00501 ✓ | 2.03e-03 / 0.00203 ✓ |
| 2 | wide | 2.18e-01 / 0.218 ✓ | 2.14e-02 / 0.0214 ✓ | 5.01e-03 / 0.00501 ✓ | 2.03e-03 / 0.00203 ✓ |
| 3 | local | 5.42e-04 / 0.000542 ✓ | 3.47e-05 / 3.47e-05 ✓ | 3.15e-04 / 0.000315 ✓ | 1.25e-04 / 0.000125 ✓ |
| 3 | wide | 4.54e-02 / 0.0454 ✓ | 2.86e-03 / 0.00286 ✓ | 3.15e-04 / 0.000315 ✓ | 1.25e-04 / 0.000125 ✓ |

No drift on any cell. CSV values and README values are identical to display precision.

## Cross-cutting patterns

- The fix was a documentation-only change to a single string in `run.py:195`. No code logic changed; no numbers changed. The original finding was a MISLABELED description, not an implementation error.
- No new gaps found. The Taylor formulas in Equations, the parameter values, the domain bounds, and all 24 numeric cells in the results table are faithful to the code.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required.
1. `test_code_has_no_differentiation_step` (violated-invariant) — PASSES on current code because `perturbation_transition` contains none of `["np.gradient", "sympy", "scipy.misc.derivative", "jax.grad"]`. This remains a true description of the code; it is not a sign of regression.
2. `test_pseudocode_does_not_claim_differentiation` (honest-fix) — PASSES on current code because `"Differentiate F at x_bar"` does not appear in `run.py`. The green state is confirmed.
3. No further action needed for this tutorial.
