# bullshit-detector — risk-aversion-monotone-choice — recheck — 2026-05-20

**Bullshit score: 0%** — original Finding 1 (DATA DRIFT, threshold inconsistency) resolved by introducing a shared `MONOTONICITY_TOL` constant; all other claims verified HOLDS against current code and CSV artifacts.

## Header
- Claim sources: `choice/risk-aversion-monotone-choice/README.md`
- Code / artifact root: `choice/risk-aversion-monotone-choice/run.py`
- Data artifacts: `choice/risk-aversion-monotone-choice/tables/model-comparison.csv`, `choice/risk-aversion-monotone-choice/tables/row-fits.csv`
- Seed audit: `bullshit-detector_risk-aversion-monotone-choice_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Monotonicity violations" column uses mixed thresholds | HOLDS (fixed) | — | no |
| 2 | Payoffs A=(2.00,1.60), B=(3.85,0.10) | HOLDS | — | no |
| 3 | CRRA utility formula exact | HOLDS | — | no |
| 4 | Fixed-scale logit Pr=lambda+(1-2lambda)*sigma(s*DeltaEU) | HOLDS | — | no |
| 5 | Monotone constraint alpha_{j+1}>=alpha_j via SLSQP ineq | HOLDS | — | no |
| 6 | DGP: rho=0.45, scale=5.00, lapse=0.02, 10 rows, 80 trials | HOLDS | — | no |
| 7 | Estimated rho=0.451 (README); 0.45074 (CSV) | HOLDS | — | no |
| 8 | LL: unconstrained=-215.051, CRRA=-221.83, monotone=-216.044 | HOLDS | — | no |
| 9 | LL loss monotone vs saturated = 0.99 | HOLDS | — | no |
| 10 | Monotone fit pools rows 2-3 (both 0.0313) | HOLDS | — | no |

## Findings

### Finding 1 (original): DATA DRIFT — threshold inconsistency — RESOLVED

- **Original claim:** Two helpers fed the same "Monotonicity violations" summary column with different sign thresholds (`< -1e-12` in `estimate_unconstrained_logits` and `< -1e-10` in `estimate_monotone_logits`).

- **Current code evidence:**
  ```python
  # run.py:17-20
  # Shared sign threshold for counting monotonicity violations. One constant
  # keeps the "Monotonicity violations" summary column to a single measurement
  # definition across all three estimators.
  MONOTONICITY_TOL = 1e-10
  ```
  ```python
  # run.py:64
  "violations": float(np.sum(np.diff(shares) < -MONOTONICITY_TOL)),
  # run.py:88
  "violations": float(np.sum(np.diff(probabilities) < -MONOTONICITY_TOL)),
  # run.py:115
  "violations": float(np.sum(np.diff(probabilities) < -MONOTONICITY_TOL)),
  ```

- **Resolution:** All three estimator helpers now use `< -MONOTONICITY_TOL`. No raw `-1e-12` literal remains in the violation-count comparisons. The mixed-threshold inconsistency is gone. Finding is fully resolved.

- **Category:** HOLDS (post-fix)

## Cross-cutting patterns

- Original Finding 1 was the only non-HOLDS finding. It is resolved. No new patterns identified.
- All formula-level claims (CRRA utility, fixed-scale logit, monotone constraint, DGP parameters, estimated rho, log likelihoods, LL loss, row pooling) verified against current `run.py` and CSV artifacts. All HOLD.
- The `MONOTONICITY_TOL` constant at `run.py:20` is used in exactly three places (`run.py:64`, `88`, `115`), corresponding to all three estimator helpers. No fourth usage or stray literal exists.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings read HOLDS. No further fixes required.
1. Test suite status: `test_f1_violated_invariant_mixed_literal_thresholds` FAILS (correct — fix applied, buggy literals gone); `test_f1_honest_fix_single_named_threshold` PASSES (correct — `MONOTONICITY_TOL` present, no raw literals remain, all three violation counts use it).
2. No math fix required. No data artifact update required. Published numbers unaffected by threshold unification.
</content>
</invoke>