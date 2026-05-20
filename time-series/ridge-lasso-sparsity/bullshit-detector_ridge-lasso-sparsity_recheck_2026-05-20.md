# bullshit-detector — ridge-lasso-sparsity — recheck — 2026-05-20

**Bullshit score: 0%** — original Finding 1 (false-inclusion count presented as a lasso metric, concealing that it is a DGP tautology) resolved: `README.md:107` now discloses "always zero by DGP construction" and warns readers the zero reflects the DGP, not lasso selectivity. All algorithmic and numeric claims verified HOLDS.

## Header
- Claim sources: `time-series/ridge-lasso-sparsity/README.md`, `time-series/ridge-lasso-sparsity/run.py`
- Code / artifact root: `time-series/ridge-lasso-sparsity/run.py`
- Data artifacts: `time-series/ridge-lasso-sparsity/tables/forecast_metrics.csv`, `time-series/ridge-lasso-sparsity/tables/selection_summary.csv`
- Seed audit: `bullshit-detector_ridge-lasso-sparsity_2026-05-20.md` (one DILUTED/MED finding)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | False inclusions = 0 disclosed as DGP tautology, not lasso metric | HOLDS | — | — |
| 2 | Ridge formula: argmin (1/n)sum(...)^2 + lambda*sum(b_j^2) | HOLDS | — | — |
| 3 | Lasso formula: argmin (1/n)sum(...)^2 + lambda*sum(|b_j|) | HOLDS | — | — |
| 4 | "intercept is never penalized" | HOLDS | — | — |
| 5 | "Lasso uses cyclic coordinate descent" | HOLDS | — | — |
| 6 | "Ridge has closed-form solution" | HOLDS | — | — |
| 7 | Time-block splits (train=125, valid=55, test=79) | HOLDS | — | — |
| 8 | Ridge lambda=0.0381, Lasso lambda=0.0079 | HOLDS | — | — |
| 9 | Ridge RMSE=0.2699, Lag RMSE=0.5989 | HOLDS | — | — |
| 10 | Ridge shock correlation=0.773 | HOLDS | — | — |
| 11 | Lasso selects 56 indicators | HOLDS | — | — |
| 12 | True nonzero policy indicators=120 | HOLDS | — | — |
| 13 | Dense-signal share missed=0.510 (51.0%) | HOLDS | — | — |
| 14 | Ridge coef corr with truth=0.644 | HOLDS | — | — |
| 15 | Lasso coef corr with truth=0.740 | HOLDS | — | — |
| 16 | All tabular numbers in README match tables/*.csv | HOLDS | — | — |

## Findings

### Finding 1 (original DILUTED/MED): RESOLVED

- **Original issue:** "False inclusions by lasso = 0" was presented in the selection table without disclosing that it is structurally zero by DGP construction; prose framing invited readers to treat it as a lasso precision measurement.
- **Current README evidence (verbatim):** "Note that the false-inclusion count is always zero by DGP construction: every one of the 120 indicators has a nonzero true coefficient, so any indicator lasso selects is true by construction. The zero reflects the DGP, not lasso precision, and should not be read as a measurement of lasso selectivity." — `README.md:107`
- **Code evidence (verbatim, structural invariant):**
  ```python
  weak = concept_signs[concept] * rng.uniform(0.006, 0.018, indicators_per_concept)
  ```
  `run.py:180`
  ```python
  int(np.sum(lasso_selected & ~true_nonzero)),
  ```
  `run.py:370`
- **Category:** HOLDS — the disclosure is present, explicit, names both the DGP reason and the consequence for interpretation, and warns against misreading the zero as a lasso metric.

### Honest-fix pass conditions

- `test_honest_fix_readme_discloses_dgp_tautology` PASSES: "always zero by DGP construction" is in README.
- `test_honest_fix_disclosure_warns_against_misreading` PASSES: disclosure contains "not", mentions lasso within 400 chars.
- `test_violated_invariant_dgp_makes_all_indicators_nonzero` PASSES: structural fact unchanged (DGP is intentionally dense; test proves the structural tautology, not a bug).

### HOLDS block (all numeric claims unchanged from original audit)

All 15 numeric and algorithmic claims verified HOLDS in original audit. DGP code, CSV, and README interpolation architecture unchanged. No re-verification needed beyond the disclosure fix.

## Cross-cutting patterns

- None. The single finding is fully resolved by an inline prose disclosure. The DGP design remains intentionally dense; the tutorial teaches that "lasso selects a compact subset" and now correctly annotates why the false-inclusion metric is uninformative in this design.

## TDD execution sequence

0. **Bullshit score: 0%.** No issues. No action required.
1. `test_honest_fix_readme_discloses_dgp_tautology` PASSES (confirmed 2026-05-20).
2. `test_honest_fix_disclosure_warns_against_misreading` PASSES (confirmed 2026-05-20).
3. `test_violated_invariant_dgp_makes_all_indicators_nonzero` PASSES — structurally correct, DGP intentionally dense, no fix needed there.
