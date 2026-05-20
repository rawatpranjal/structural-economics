# bullshit-detector — stock-watson — recheck — 2026-05-20

**Bullshit score: 0%** — all 13 claims hold; the AR lag bug (Finding 1 of the original audit) is fixed; RMSE numbers are updated and consistent with committed CSVs; 60% prose now correctly says "usable evaluation window".

## Header
- Claim sources: `time-series/stock-watson/README.md`
- Code / artifact root: `time-series/stock-watson/run.py`
- Data artifacts: `time-series/stock-watson/tables/forecast-comparison.csv`, `time-series/stock-watson/tables/eigenvalues.csv`
- Seed audit: `time-series/stock-watson/bullshit-detector_stock-watson_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | AR lag-1 col = y_tau (lags 1 and 2 at row 0) | HOLDS | - | - |
| 2 | Factor corr = 0.9970 | HOLDS | - | - |
| 3 | PC1 explains 57.2% of variance | HOLDS | - | - |
| 4 | Exposure corr = 0.9999 | HOLDS | - | - |
| 5 | AR(2) RMSE = 1.4186 | HOLDS | - | - |
| 6 | PCA factor RMSE = 1.2572, relative = 0.8862 | HOLDS | - | - |
| 7 | True factor RMSE = 1.2649, relative = 0.8917 | HOLDS | - | - |
| 8 | 11.4% RMSE improvement (PCA vs AR) | HOLDS | - | - |
| 9 | 60% of the usable evaluation window | HOLDS | - | - |
| 10 | lambda_i ~ N(1, 0.5^2) = N(1, 0.25) | HOLDS | - | - |
| 11 | sigma_e ~ U(0.5, 1.5) | HOLDS | - | - |
| 12 | PCA uses T^{-1} Z'Z | HOLDS | - | - |
| 13 | F_hat_t = Z_t'v_1 (eigenvector projection) | HOLDS | - | - |

## Findings

### Finding 1 (original): AR lag construction — RESOLVED

- **Original claim:** README.md:74 algorithm step 5: "fit AR(p): y_{t+h} on 1, y_t,...,y_{t-p+1}". Original code used `y[start-lag-1:end-lag-1]`, which at row 0 gave `X_ar[0,1] = y[1]` (y_{tau-1}) instead of y[2] (y_tau). Bug caused absolute RMSE values to be wrong (old: 1.753/1.262/1.273).

- **Current code (verbatim):**
  ```python
  for lag in range(p_ar):
      X_ar[:, lag + 1] = y[start - lag: end - lag]
  ```
  `run.py:177-178`

- **Verification:** With `start=2`, `end=199`, `lag=0`: `y[2:199]` → `X_ar[0,1] = y[2]` = y_tau. With `lag=1`: `y[1:198]` → `X_ar[0,2] = y[1]` = y_{tau-1}. Matches the stated equation y_{tau+h} = alpha + beta_1 y_tau + beta_2 y_{tau-1} + gamma'F + eps exactly.

- **Category:** HOLDS (was DILUTED HIGH in original audit)

- **New RMSE values:** CSV `forecast-comparison.csv` rows: `AR(2),1.4186,1.0` / `PCA factor AR(2),1.2572,0.8862` / `True factor AR(2),1.2649,0.8917`. All three match README table at README.md:117-119 exactly.

### Finding 2 (original): 60% training share — RESOLVED

- **Original claim:** "Initial training share | 60% | Expanding-window forecast start" — old README implied 60% of T=200.

- **Current README (verbatim):** "| Initial training share | 60% of the usable evaluation window | Expanding-window forecast start |" — `README.md:54`

- **Code (verbatim):**
  ```python
  train_frac = 0.6
  n_train_init = int(train_frac * n_eval)
  ```
  `run.py:186-187`

- **Verification:** `n_eval = len(y[start+h:end+h]) = len(y[3:200]) = 197`. `int(0.6 × 197) = 118`. Prose now correctly says "usable evaluation window" rather than implying 60% of T=200=120.

- **Category:** HOLDS (was DATA DRIFT LOW in original audit)

## Cross-cutting patterns

- The single load-bearing fix (AR lag index shift from `y[start-lag-1:end-lag-1]` to `y[start-lag:end-lag]`) propagated correctly through all three RMSE numbers. The new values (1.4186, 1.2572, 1.2649) replace the old misspecified values (1.753, 1.262, 1.273) in both the README table and the committed CSV.
- All narrative claims (corr, scree, exposure) are embedded via f-strings from live computed values; no stale literals remain.
- The 11.4% improvement figure is consistent with the corrected RMSE numbers: (1 - 1.2572/1.4186) x 100 = 11.38%, which rounds to 11.4%.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 0%.** Ship. No further action required on this tutorial.
1. The violated-invariant test (`test_violated_invariant_ar_lag_skips_most_recent_obs`) now FAILS — confirming the bug is gone.
2. The honest-fix tests (`test_honest_fix_ar_lag1_is_most_recent_obs`, `test_honest_fix_ar_lag2_is_one_step_back`, `test_honest_fix_readme_discloses_training_share_base`) now PASS — confirming the fix is correct.
3. No further code changes warranted.
