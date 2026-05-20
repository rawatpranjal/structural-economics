# bullshit-detector — simulation-based-estimation — recheck — 2026-05-20

**Bullshit score: 0%** — Both prior findings verified fixed. All numeric, algorithmic, and formula claims HOLD against code and all five CSV artifacts.

## Header
- Claim sources: `computational-methods/simulation-based-estimation/README.md`
- Code / artifact root: `computational-methods/simulation-based-estimation/run.py`
- Data artifacts: `tables/parameter-recovery.csv`, `tables/method-comparison.csv`, `tables/msm-residuals.csv`, `tables/indirect-inference-residuals.csv`, `tables/abc-summary.csv`
- Seed audit: `computational-methods/simulation-based-estimation/bullshit-detector_simulation-based-estimation_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Structural model log wage + logistic acceptance | HOLDS | - | - |
| 2 | MSM 5 moments (acceptance rate, offer mean, offer SD, accepted mean, accepted SD) | HOLDS | - | - |
| 3 | MSM weight W_m = diag(1/max(|m_obs|, 0.1))^2 | HOLDS | - | - |
| 4 | II 6 auxiliary stats (LPM intercept, slope, offer mean, offer SD, acceptance rate, accepted mean) | HOLDS | - | - |
| 5 | II weight W_b same scaling form on target_aux | HOLDS | - | - |
| 6 | ABC distance rho = sqrt(Q_MSM), same 5 moments | HOLDS | - | - |
| 7 | ABC prior U(2.4,3.6) x U(0.2,0.8) x U(2.5,3.8) | HOLDS | - | - |
| 8 | ABC-SMC alpha-quantile adaptive tolerance, alpha=0.5 | HOLDS | - | - |
| 9 | Round-0 samples ceil(N/alpha) from prior | HOLDS | - | - |
| 10 | Kernel covariance K_t = N(theta', 2*Cov_{t-1}) | HOLDS | - | - |
| 11 | Uniform prior simplification in importance weights | HOLDS | - | - |
| 12 | MSM and ABC criteria directly comparable (same 5 moments, same scale vector) | HOLDS | - | - |
| 13 | II criterion not on same scale (6 aux stats, different denominators) | HOLDS | - | - |
| 14 | II residuals table call: residual_table(..., msm, ii).query("Indirect inference") (prior Finding 1, fixed) | HOLDS | - | - |
| 15 | All numeric parameter recovery values match CSV | HOLDS | - | - |
| 16 | MSM moment residuals match msm-residuals.csv | HOLDS | - | - |
| 17 | II auxiliary residuals match indirect-inference-residuals.csv | HOLDS | - | - |
| 18 | Method-comparison table matches method-comparison.csv | HOLDS | - | - |
| 19 | Per-round ABC diagnostics match abc-summary.csv | HOLDS | - | - |

## Findings

None.

**Prior Finding 1 resolved (MISLABELED: ii table call).** `run.py:829-838` now calls `residual_table(..., msm, ii).query("Estimator == 'Indirect inference'")`. The previous form passed `(ii, ii)` and queried `"MSM"` to extract II data via the mislabeled first loop iteration. The fix passes the correct distinct arguments and queries the correct estimator label. The committed `tables/indirect-inference-residuals.csv` contains II residuals generated from `ii["simulated_stats"]`, consistent with the fixed call.

**Prior Finding 2 resolved (DILUTED: "same scale" claim).** `run.py:840-851` now reads: "The MSM and ABC criteria are directly comparable, because both sum squared scaled residuals over the same 5 economic moments with the same scale vector. The II criterion is not on the same scale: II uses a different set of 6 auxiliary statistics with their own scale denominators, so its value should not be read as a sharper fit than MSM or ABC." `README.md:247` matches this corrected prose. The prior tripartite "same scale" claim is gone; the disclaimer is explicit.

## Cross-cutting patterns

- Both DILUTED/MISLABELED findings from the original audit are resolved. The reporting layer now correctly distinguishes MSM-scale from II-scale and passes distinct estimator results to distinct table calls.
- No parametric leaks. `recover_pseudo_values` / `estimate_by_simulation` receive only the observable target statistics and simulation draws.
- All five CSV artifacts are internally consistent with README prose and code logic.
- The MSM residuals table (run.py:818-827) uses `(msm, msm)` with `.query("MSM")` -- this is a deliberate and correct pattern: it extracts the MSM row from a table where only MSM is meaningful. The II table now correctly uses `(msm, ii)` with `.query("Indirect inference")`.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%. No action required.**
1. Both original findings resolved. Tests at `tests/test_simulation-based-estimation.py` show 3 pass / 2 expected-fail (violated-invariant tests whose docstrings say they should fail post-fix). That is the correct green state.
2. No sim re-runs or data artifact changes needed.
