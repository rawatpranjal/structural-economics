# bullshit-detector — stock-watson — 2026-05-20

**Bullshit score: 35%** — AR lag construction regresses y_{tau+1} on lags 2 and 3 (y_{tau-1}, y_{tau-2}) instead of the claimed lags 1 and 2 (y_tau, y_{tau-1}); the absolute RMSE numbers are wrong relative to the stated algorithm, though the relative ranking AR vs FAAR is preserved.

## Header
- Claim sources: `time-series/stock-watson/README.md` (prose, Equations, Model Setup, Results, Takeaway)
- Code / artifact root: `time-series/stock-watson/run.py`
- Data artifacts: `time-series/stock-watson/tables/eigenvalues.csv`, `time-series/stock-watson/tables/forecast-comparison.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | AR lag order: y_{t+h} on y_t,...,y_{t-p+1} (lags 1 and 2) | DILUTED | HIGH | yes (absolute RMSE values affected; relative ranking preserved) |
| 2 | Factor corr = 0.9970 | HOLDS | — | — |
| 3 | PC1 explains 57.2% of variance | HOLDS | — | — |
| 4 | Exposure corr = 0.9999 | HOLDS | — | — |
| 5 | RMSE 1.753 → 1.262, 28.0% improvement | HOLDS | — | — |
| 6 | True-factor RMSE = 1.273 | HOLDS | — | — |
| 7 | lambda_i ~ N(1, 0.5^2) in Equations vs N(1, 0.25) in Model Setup | HOLDS | — | — |
| 8 | sigma_e ~ U(0.5, 1.5) | HOLDS | — | — |
| 9 | PCA uses T^{-1}Z'Z | HOLDS | — | — |
| 10 | Fhat_t = Z_t'v_1 (eigenvector projection) | HOLDS | — | — |
| 11 | Initial training share 60% (imprecise: applied to n_eval=197 not T=200) | DATA DRIFT | LOW | no |

## Findings

### Finding 1: AR lag construction — claimed lags 1 and 2, code uses lags 2 and 3

- **Claim source (verbatim):** "fit AR(p): y_{t+h} on 1, y_t,...,y_{t-p+1}" — `README.md:74` (algorithm pseudocode, step 5). Also `README.md:37`: "$$y_{t+h} =\alpha+\sum_{\ell=1}^{p}\beta_\ell y_{t-\ell+1} +\gamma'\hat F_t+\varepsilon_{t+h}.$$"

- **Code evidence (verbatim):**
  ```python
  start = p_ar          # = 2
  end = T - h           # = 199
  y_target = y[start + h: end + h]   # y[3:200]; y_target[t] = y[t+3]
  X_ar = np.ones((end - start, p_ar + 1))
  for lag in range(p_ar):
      X_ar[:, lag + 1] = y[start - lag - 1: end - lag - 1]
  # lag=0: y[2-0-1:199-0-1] = y[1:198] => X_ar[t,1] = y[t+1]
  # lag=1: y[2-1-1:199-1-1] = y[0:197] => X_ar[t,2] = y[t]
  ```
  `run.py:165-175`

- **Data evidence:** The RMSE numbers in `tables/forecast-comparison.csv` (`AR(2),1.753,1.0` / `PCA factor AR(2),1.2621,0.72`) are outputs of this misspecified estimator. They are internally consistent with the committed code but do not correspond to the AR(2) equation the README states.

- **Analysis:** At forecast origin tau = t+2, the code regresses y[tau+1] on y[tau-1] and y[tau-2]. The README claims lags y_t and y_{t-1} (i.e. y_tau and y_{tau-1}). The code skips y_tau (the most recent observation) and uses one extra lag back. The bug is identical in both the AR and FAAR specifications (the factor alignment F_hat[tau] is correct), so the relative ranking AR vs FAAR is preserved. The absolute RMSE numbers (1.753, 1.262, 1.273) are products of the wrong lag specification. A correct AR(2) implementation would yield different absolute values.

- **Category:** DILUTED — the code does implement expanding-window OLS with AR terms and a factor, but the lag indices deviate from the stated equation in a load-bearing way (wrong regressors, not just a naming mismatch).

- **Severity:** HIGH

- **Result-changing:** yes — absolute RMSE values reported in README and tables are from a misspecified AR(2) (uses lags 2 and 3, not lags 1 and 2). Relative RMSE ratio and 28.0% improvement claim survive because both models carry the same lag error; but the absolute numbers (1.753, 1.262, 1.273) would differ under the correct specification.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert X_ar[0, 1] == y[1]  # PASSES on buggy code (lag2 at row 0 = y[1]); FAILS on honest fix (should be y[2])
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert X_ar[0, 1] == y[2]  # PASSES on honest fix (lag1 at row 0 = y[start] = y[2]); FAILS on buggy code
  ```

---

### Finding 2: 60% training share — 60% of n_eval (197), not T (200)

- **Claim source (verbatim):** "Initial training share | 60% | Expanding-window forecast start" — `README.md:54`

- **Code evidence (verbatim):**
  ```python
  n_eval = len(y_target)     # = 197 (y[3:200])
  train_frac = 0.6
  n_train_init = int(train_frac * n_eval)   # int(0.6 * 197) = 118
  ```
  `run.py:170, 183-184`

- **Data evidence:** Not directly tabulated; no artifact contradicts this.

- **Category:** DATA DRIFT — the prose says "60%" without specifying the base; the code applies 60% to n_eval=197 (the usable window after lag/horizon trimming), not to T=200. A reader would infer 60% of T=200 = 120 observations; the code uses 118.

- **Severity:** LOW — the two-observation discrepancy (118 vs 120) does not materially change results. No published number is wrong.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert n_train_init == int(0.6 * n_eval)  # PASSES on current code; FAILS if reader assumes int(0.6*T)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "60% of the usable evaluation window" in report_text or n_train_init == int(0.6 * 197)  # prose clarification
  ```

---

## Cross-cutting patterns

- The single structural bug (lag offset) affects ALL three RMSE numbers reported in Results and the Takeaway. The absolute values are internally consistent (they match the committed CSV) but correspond to a different estimator than the one the README equations describe.
- The factor alignment is correct in both the PCA-factor and true-factor regressions; only the AR lag indices are off. This means the relative RMSE ratios and the qualitative conclusion (factor beats AR) hold, but the specific numbers (1.753, 1.262, 1.273, 28.0%) are from a misspecified AR baseline.
- `F_hat_aligned` (sign-corrected) is used for plots and correlation stats. `F_hat_1` (unsign-corrected) is used for forecasting. OLS absorbs the sign flip via the gamma coefficient, so this is not a bug. It is worth noting that the two objects serve different purposes.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 35%.** Surface to user before touching code. The relative ranking is honest; the absolute RMSE numbers are not.

1. Turn **Finding 1 violated invariant** into a pytest test:
   ```python
   # tests/test_stock_watson_lags.py
   def test_ar_lag_bug_exists():
       y = np.arange(200, dtype=float)
       X_ar, y_target = build_regressors(y, p_ar=2, h=1)  # extract regressor builder
       assert X_ar[0, 1] == y[1]  # proves bug: lag1 is y[1] not y[2]
   ```
   Confirm this PASSES on current code.

2. Turn **Finding 1 honest-fix pass condition** into a second pytest test:
   ```python
   def test_ar_lag_correct():
       y = np.arange(200, dtype=float)
       X_ar, y_target = build_regressors(y, p_ar=2, h=1)
       assert X_ar[0, 1] == y[2]  # lag1 at row 0 must be y[start] = y[2]
   ```
   Confirm this FAILS on current code.

3. Fix: in `run.py`, change the lag construction so that at row `t` (forecast origin tau = t + start):
   - `X_ar[t, 1]` = `y[start - lag - 1 + 1 : ...]` — shift by +1, giving `y[tau]` as lag 1.
   - Equivalently: `X_ar[:, lag + 1] = y[start - lag : end - lag]` for lag=0,1.

4. Re-run `python run.py`. Refresh `tables/forecast-comparison.csv` and `README.md`. Re-run this skill to confirm Finding 1 becomes HOLDS and score falls to ≤15%.
