# bullshit-detector — maximum-score-binary-choice — 2026-05-20

**Bullshit score: 0%** — Every prose claim, equation, numeric, and table value grounds exactly in code or committed CSV artifacts; hostile reviewer reads twice and finds no hole.

## Header
- Claim sources: `choice/maximum-score-binary-choice/README.md`
- Code / artifact root: `choice/maximum-score-binary-choice/run.py`
- Data artifacts: `choice/maximum-score-binary-choice/tables/estimator-comparison.csv`, `choice/maximum-score-binary-choice/tables/estimator-diagnostics.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | DGP participation rule y_i = 1{x1 + beta*x2 + eps >= 0} | HOLDS | none | no |
| 2 | Error heteroskedastic logistic, median zero | HOLDS | none | no |
| 3 | n = 2500 | HOLDS | none | no |
| 4 | True beta = -0.85 | HOLDS | none | no |
| 5 | Grid points = 501 | HOLDS | none | no |
| 6 | Bandwidth h = 0.25 | HOLDS | none | no |
| 7 | Bootstrap draws = 80 | HOLDS | none | no |
| 8 | Smoothed estimate -0.831 (rounded from -0.83084) | HOLDS | none | no |
| 9 | Bootstrap CI [-0.957, -0.679] | HOLDS | none | no |
| 10 | Benefit coeff normalized to 1 | HOLDS | none | no |
| 11 | Manski score formula | HOLDS | none | no |
| 12 | Smoothed score formula S_h(b) | HOLDS | none | no |
| 13 | Grid beta = -0.88 | HOLDS | none | no |
| 14 | Smoothed beta = -0.83084 | HOLDS | none | no |
| 15 | Logit ratio = -0.66013 | HOLDS | none | no |
| 16 | Logit misspecified for this DGP | HOLDS | none | no |
| 17 | Diagnostics table CI values | HOLDS | none | no |
| 18 | Bootstrap mean = -0.826309 | HOLDS | none | no |
| 19 | Smoothed score = 0.678026 | HOLDS | none | no |
| 20 | Error for True index = 0 | HOLDS | none | no |

## Findings

None. All 20 claims ground exactly in code and/or committed CSV artifacts.

**Evidence for key claims (supporting citations):**

- DGP claim 1-2: `run.py:22-25` — `x2 = 0.40*x1 + rng.normal(...)`, `scale = 0.55 + 0.65*np.abs(x2)`, `error = scale * rng.logistic(...)`, `latent = x1 + beta_true*x2 + error`; `y = (latent >= 0.0).astype(int)`.
- Manski formula claim 11: `run.py:31-32` — `predicted = (x1 + beta * x2 >= 0.0).astype(int)` then `np.mean(predicted == y)` is algebraically equivalent to `(1/n) sum [y_i*1{>=0} + (1-y_i)*1{<0}]`.
- Smoothed score formula claim 12: `run.py:37-38` — `probability = norm.cdf((x1 + beta*x2)/bandwidth)` then `np.mean(y*probability + (1.0-y)*(1.0-probability))` matches the stated S_h(b) exactly.
- "-0.831" in prose (README.md:64) vs -0.83084 in table: `run.py:243` uses `:.3f` format on `smooth_est['beta']`; -0.83084 rounds to -0.831. No inconsistency.
- All numeric entries in both CSV files match README table and diagnostics table values to the precision shown.

## Cross-cutting patterns

None. Audit found no pattern of drift, mislabeling, or missing implementation across the 20 claims.

## TDD execution sequence (for the next agent)

**Bullshit score is 0%. No remediation required.**

No non-HOLDS findings exist. No failing tests to write. No fixes to dispatch.

If a future run.py edit changes parameters or estimates, re-run this skill on the regenerated README.md and updated CSVs before merging.
