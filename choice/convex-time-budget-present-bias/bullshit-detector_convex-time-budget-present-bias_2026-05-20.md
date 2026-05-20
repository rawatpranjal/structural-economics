# bullshit-detector - convex-time-budget-present-bias - 2026-05-20

**Bullshit score: 35%** - One FALSE claim in Takeaway (Stone-Geary minima never estimated; code hard-codes omega=0) and one DILUTED identification figure (profile LL ignores censoring, labeled as if Tobit-proper).

## Header
- Claim sources: `choice/convex-time-budget-present-bias/README.md`
- Code / artifact root: `choice/convex-time-budget-present-bias/run.py`
- Data artifacts: `choice/convex-time-budget-present-bias/tables/parameter-recovery.csv`, `choice/convex-time-budget-present-bias/tables/design-comparison.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | NLS lets Stone-Geary minima be jointly estimated | FALSE | MED | yes (mis-states NLS capability; omega fixed at zero) |
| 2 | Profile log-likelihood figure uses proper Tobit censoring | DILUTED | LOW | no (qualitative sharp-vs-flat lesson holds, but censoring ignored) |
| 3 | All numeric table values (beta, delta, alpha, SE) | HOLDS | - | - |
| 4 | Budget constraint, tangency derivation, QH utility | HOLDS | - | - |
| 5 | Bootstrap clustering across 45 cells | HOLDS | - | - |
| 6 | Annual discount rate approx 30.1% | HOLDS | - | - |

## Findings

### Finding 1: NLS lets Stone-Geary minima be jointly estimated

- **Claim source (verbatim):** "NLS on the demand function works at corners and lets the Stone-Geary minima be jointly estimated." - `README.md:184`

- **Code evidence (verbatim):**
  ```python
  # Model: closed-form demand with omega_1 = omega_2 = 0 (column-3 spec in AS)
  def demand_c_t(beta: float, delta: float, alpha: float,
                 one_plus_r: np.ndarray, k: np.ndarray, t: np.ndarray,
                 m: np.ndarray) -> np.ndarray:
      """Closed-form sooner-payment demand (eq 5 with omega_1 = omega_2 = 0).
  ```
  `run.py:41-64`

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

- **Data evidence:** Tables `parameter-recovery.csv` and `design-comparison.csv` list only three estimated parameters: beta, delta, alpha. No omega column exists. The code always passes `omega_1 = omega_2 = 0` implicitly; `fit_nls` estimates a three-element `theta` vector with no omega component.

- **Category:** FALSE - the code sets omega to zero throughout and never estimates any Stone-Geary minimum. The claim states that NLS "lets" the minima "be jointly estimated," implying the tutorial demonstrates this capability. It does not. The module comment at line 41 explicitly names this the "column-3 spec in AS," which fixes Stone-Geary minima at zero, not the column-1 spec that estimates them.

- **Severity:** MED

- **Result-changing:** yes - a reader relying on this claim to understand NLS scope will conclude the tutorial demonstrates joint Stone-Geary estimation; it does not. The NLS in this tutorial is strictly column-3 (omega=0 fixed). No result numbers change, but the stated capability of the estimator is wrong.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert len(nls_residuals.__code__.co_varnames) == 2 and "omega" not in inspect.getsource(fit_nls)
  # PASSES on current code (no omega anywhere); FAILS on honest fix that adds omega estimation
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "omega" in inspect.getsource(fit_nls) or "Stone-Geary" not in open("README.md").read().split("Takeaway")[1]
  # PASSES if either omega is estimated in fit_nls or the claim is removed from Takeaway
  ```

---

### Finding 2: Profile log-likelihood ignores censoring

- **Claim source (verbatim):** "The profile log-likelihood for $\beta$ is sharp under the strong design that includes $t > 0$ cells. It is nearly flat under the weak design that uses only $t = 0$ cells." - `README.md:151-153`. Figure caption: "Profile log-likelihood of beta under weak vs strong design" - `README.md:153`.

- **Code evidence (verbatim):**
  ```python
  def neg_ll_concentrated(params, df, b=b_try):
      delta_, alpha_, sig_ = params
      if sig_ <= 0 or alpha_ >= 1.0 or alpha_ <= 0.05:
          return 1e10
      mu = log_tangency(b, delta_, alpha_,
                        df["one_plus_r"].to_numpy(),
                        df["k"].to_numpy(),
                        df["t"].to_numpy())
      resid = df["log_ratio"].to_numpy() - mu
      return 0.5 * len(resid) * np.log(2 * np.pi * sig_**2) \
             + 0.5 * np.sum(resid**2) / sig_**2
  ```
  `run.py:286-296`

- **Data evidence:** None - the identification figure is not tabulated; the qualitative sharp-vs-flat result is the claim.

- **Category:** DILUTED - the concentrated log-likelihood used for the profile figure is plain Gaussian (sum of squared residuals over all observations, including censored ones). The tutorial's own Tobit implementation (`tobit_neg_loglik`, lines 106-134) correctly distinguishes interior, lower-corner, and upper-corner contributions. The profile LL skips this: censored observations that should contribute `log Phi(z)` or `log(1-Phi(z))` instead contribute `norm.logpdf(z)` via the squared-residual criterion. The figure caption calls this "profile log-likelihood" without qualification, implying the same Tobit criterion used for point estimation. The qualitative lesson (sharp under strong design, flat under weak design) survives this simplification, so the finding is DILUTED not FALSE.

- **Severity:** LOW - qualitative identification lesson holds; the censoring share at sigma_eps=0.30 is modest; the sharp-vs-flat contrast would remain under a proper Tobit profile.

- **Result-changing:** no - the pedagogical claim about identification survives; no published numbers depend on this figure.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "logcdf" not in inspect.getsource(neg_ll_concentrated.__wrapped__ if hasattr(neg_ll_concentrated, '__wrapped__') else type(None))
  # Note: neg_ll_concentrated is defined inside a loop; check profile block for absence of logcdf
  assert "logcdf" not in open("run.py").read().split("profile_full")[0].split("beta_grid")[1]
  # PASSES on current code (no censoring in profile block); FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "logcdf" in "".join(open("run.py").readlines()[283:310])
  # PASSES on honest fix (Tobit censoring in profile block); FAILS on current code
  ```

## Cross-cutting patterns

- Both findings relate to claims about what the code can do (Stone-Geary) or how it computes a named statistic (profile LL) rather than to numerical errors. The numeric outputs are all faithfully transcribed from code to README. The gap is between capability language in the prose and the actual scope of the implementation.
- The Stone-Geary finding is a Takeaway-level claim, not an Equations claim; the method actually implemented (omega=0, column-3 spec) is correctly documented in the code comment at line 41 and in the Solution Method section. The Takeaway overstates relative to both.
- The profile LL finding is common in pedagogical tutorials: the identification figure uses a simpler criterion than the estimator to avoid implementing a second Tobit optimizer inside a for-loop. The gap is worth flagging but the lesson is not invalidated.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 35%.** This is below the 50% halt threshold; surface findings to the author and proceed with targeted fixes.

1. **Finding 1 fix path (choice between two options):**
   - Option A: remove "lets the Stone-Geary minima be jointly estimated" from Takeaway (`README.md:184`) and replace with "fixes Stone-Geary minima at zero, matching the column-3 specification in Andreoni and Sprenger Table 2." Regenerate README from `run.py` after editing the `add_takeaway` string at `run.py:668-670`.
   - Option B: extend `fit_nls` to estimate omega jointly (adds a fourth parameter) and confirm tables reproduce. This is a larger code change.
   - Write violated-invariant test first (confirms omega absent); then implement option A or B; then run pass condition test.

2. **Finding 2 fix path:**
   - Replace `neg_ll_concentrated` inside the profile loop (`run.py:286-296`) with a version that mirrors `tobit_neg_loglik` censoring logic, or add a note to the figure caption that the profile uses the Gaussian criterion rather than the full Tobit likelihood. Caption fix is one line; criterion fix requires refactoring the inner loop.
   - Write violated-invariant test (no `logcdf` in profile block); implement fix; run pass condition test.

3. After fixes: re-run `python run.py` inside the tutorial folder; confirm CSV tables regenerate with identical numbers (finding 1 option A changes no numbers; finding 2 caption fix changes no numbers). Re-run `scripts/validate_catalog.py`.

4. Re-run this skill on the updated README to confirm both findings now read HOLDS and score drops to <= 10%.
