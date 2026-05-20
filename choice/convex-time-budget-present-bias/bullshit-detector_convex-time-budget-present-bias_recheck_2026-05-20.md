# bullshit-detector - convex-time-budget-present-bias - recheck - 2026-05-20

**Bullshit score: 20%** - Both original findings (Finding 1 FALSE, Finding 2 DILUTED) are resolved. One new LOW-severity DILUTED finding: "corner choices contribute zero residual" is technically wrong (corner residual = boundary value minus prediction, not zero), but the economic substance (no censoring term) is correct.

## Header
- Claim sources: `choice/convex-time-budget-present-bias/README.md`
- Code / artifact root: `choice/convex-time-budget-present-bias/run.py`
- Data artifacts: `choice/convex-time-budget-present-bias/tables/parameter-recovery.csv`, `choice/convex-time-budget-present-bias/tables/design-comparison.csv`
- Seed audit: `choice/convex-time-budget-present-bias/bullshit-detector_convex-time-budget-present-bias_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | NLS Takeaway: Stone-Geary minima jointly estimated (original Finding 1) | HOLDS | - | resolved: prose now correctly states minima fixed at zero, column-1 spec not run |
| 2 | Profile figure labeled as Tobit LL without disclosure (original Finding 2) | HOLDS | - | resolved: README explicitly discloses Gaussian sum-of-squares criterion |
| 3 | "corner choices contribute zero residual" | DILUTED | LOW | no - economic substance (no censoring term) correct; "zero residual" is numerically wrong |
| 4 | All numeric table values (beta, delta, alpha, SE) | HOLDS | - | - |
| 5 | Budget constraint, tangency derivation, QH utility, Tobit LL | HOLDS | - | - |
| 6 | Bootstrap clustering across 45 cells | HOLDS | - | - |
| 7 | Annual discount rate approx 30.1% | HOLDS | - | - |
| 8 | Profile uses Gaussian criterion (new disclosure in README) | HOLDS | - | - |
| 9 | Both methods fix Stone-Geary at zero, column-3 spec (Takeaway) | HOLDS | - | - |

## Findings

### Original Finding 1 (RESOLVED): NLS Takeaway claimed Stone-Geary minima jointly estimated

- **Original claim (verbatim):** "NLS on the demand function works at corners and lets the Stone-Geary minima be jointly estimated." - `README.md:184` (original version)

- **Current README (verbatim):** "Both methods here fix the Stone-Geary minima at zero, matching column 3 of Andreoni and Sprenger Table 2; the column-1 specification that estimates those minima jointly is a natural extension this tutorial does not run." - `README.md:184`

- **Code ground truth (verbatim):**
  ```python
  def fit_nls(df: pd.DataFrame, theta0: np.ndarray) -> tuple:
      """Fit (beta, delta, alpha) by NLS on the demand function."""
      bounds_lo = np.array([0.50, 0.95, 0.10])
      bounds_hi = np.array([1.50, 1.00, 0.99])
      result = least_squares(
          nls_residuals, theta0, bounds=(bounds_lo, bounds_hi),
          args=(df,), method="trf", xtol=1e-10, ftol=1e-10,
      )
      return result.x, result
  ```
  `run.py:92-100`

- **Category:** HOLDS - prose now accurately states the implementation. `fit_nls` estimates three parameters; omega is not estimated. The corrected Takeaway (`run.py:675-678`) now matches.

- **Test verdict:** `test_finding1_violated_invariant` FAILS (expected post-fix behavior - the buggy string is gone). `test_finding1_honest_fix` PASSES. Resolution confirmed.

---

### Original Finding 2 (RESOLVED): Profile figure labeled as Tobit LL without disclosure

- **Original claim:** Figure captioned "profile log-likelihood" with no qualification, implying Tobit criterion. README had no disclosure of the Gaussian simplification.

- **Current README (verbatim):** "This identification figure profiles a Gaussian sum-of-squares criterion over $\beta$, without the Tobit censoring terms used by the Method 2 point estimator. The censoring share is modest at this noise level, so the criterion is a faithful identification diagnostic; the sharp-versus-flat contrast is the lesson, not the absolute likelihood level." - `README.md:151`

- **run.py source (verbatim):**
  ```python
  "This identification figure profiles a Gaussian sum-of-squares criterion over $\\beta$, without the Tobit censoring terms used by the Method 2 point estimator. "
  "The censoring share is modest at this noise level, so the criterion is a faithful identification diagnostic; the sharp-versus-flat contrast is the lesson, not the absolute likelihood level."
  ```
  `run.py:561-562`

- **Code ground truth remains:** `neg_ll_concentrated` uses `np.sum(resid**2)`, no `logcdf`/`logsf`. `run.py:294-296`. The code has not changed; the disclosure was added to the prose. This is the correct fix path (caption fix) per the original audit.

- **Category:** HOLDS - prose now correctly discloses the Gaussian criterion.

- **Test verdict:** `test_finding2_violated_invariant` FAILS (expected post-fix behavior - the disclosure strings are now present). `test_finding2_honest_fix` PASSES. Resolution confirmed.

---

### Finding 3 (NEW): "corner choices contribute zero residual"

- **Claim source (verbatim):** "NLS does not need an interior assumption and the residual is in dollar units, but corner choices contribute zero residual rather than a censoring term." - `README.md:73`

- **Code evidence (verbatim):**
  ```python
  def nls_residuals(theta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
      """Residuals of observed sooner earnings against eq-5 prediction."""
      beta, delta, alpha = theta
      pred = demand_c_t(beta, delta, alpha,
                        df["one_plus_r"].to_numpy(),
                        df["k"].to_numpy(),
                        df["t"].to_numpy(),
                        df["m"].to_numpy())
      return df["c_t"].to_numpy() - pred
  ```
  `run.py:81-89`

  ```python
  c_t_obs = n_t_clipped * expanded["a_early"].to_numpy()
  ```
  `run.py:200`

- **Analysis:** `c_t_obs` at a lower corner = `0 * a_early = 0`. NLS residual = `0 - pred` where `pred` is the model's interior prediction. That residual is not zero - it equals `-pred`, the negative of the predicted interior allocation. At an upper corner, `c_t_obs = 100 * a_early = m/(1+r)` and residual = `m/(1+r) - pred`. Neither is zero. The prose says "zero residual" - this is numerically wrong. The correct statement is that corner observations contribute an interior-style residual (boundary value minus prediction) rather than a censoring term. The economic distinction between NLS and Tobit is correct; only the "zero residual" characterization is wrong.

- **Category:** DILUTED - the load-bearing distinction (no censoring term in NLS) holds; the specific claim that the residual is zero does not.

- **Severity:** LOW - does not change any result, does not change the qualitative lesson, is a common pedagogical shorthand. The reader does not rely on "zero" to draw any conclusion.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "contribute zero residual" in Path("README.md").read_text()
  # PASSES on current code (wrong claim still present); FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "contribute zero residual" not in Path("README.md").read_text() or "boundary" in Path("README.md").read_text().split("corner choices")[1][:80]
  # PASSES on honest fix (either phrase removed or corrected); FAILS currently
  ```

---

## Cross-cutting patterns

- Both original findings were prose-only gaps; the code and all published numbers were correct throughout. The fix correctly left the code untouched and corrected only the claim language.
- `test_finding1_violated_invariant` and `test_finding2_violated_invariant` both FAIL on the fixed code as designed - these are violated-invariant tests whose docstrings explicitly state they should fail post-fix. This is the correct post-fix behavior, not a regression.
- `test_finding1_honest_fix` and `test_finding2_honest_fix` both PASS on the fixed code as designed.
- The new Finding 3 ("zero residual") is a minor prose inaccuracy in the `Equations` section, not the `Takeaway`. It requires a one-word correction ("nonzero" or "boundary-value") rather than a structural prose fix.
- No numeric outputs changed across any fix. All CSV values match code-generated values exactly.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20%.** Original score was 35%; both findings resolved. One new LOW-severity finding remains. Below the 25% ship threshold if the new finding is addressed; currently just above it.

1. **Finding 3 fix path (one-sentence correction):**
   - In `run.py`, locate the `add_equations` string that contains "corner choices contribute zero residual" (search `run.py` for that phrase - it appears in the equations string, not in `add_solution_method`).
   - Replace "corner choices contribute zero residual rather than a censoring term" with "corner choices contribute a boundary-value residual rather than a censoring term."
   - Regenerate `README.md` from `run.py` and confirm the phrase "zero residual" no longer appears in the `Equations` section.
   - Run `scripts/validate_catalog.py`.

2. After fix: re-run this skill. Expected new score: 0-10% (all HOLDS).
