# bullshit-detector -- logit-supply-side -- recheck -- 2026-05-20

**Bullshit score: 5%** -- both original findings resolved; docstring now says `ds_k/dp_j` matching README convention; cost-recovery table committed as a separate CSV making the MAE 0.455 independently verifiable.

## Header
- Claim sources: `industrial-organization/logit-supply-side/README.md`
- Code / artifact root: `industrial-organization/logit-supply-side/run.py`
- Data artifacts: `tables/estimation-results.csv`, `tables/cost-recovery-market0.csv`
- Seed audit: `bullshit-detector_logit-supply-side_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Omega_jk = -O_jk * ds_k/dp_j; docstring matches README convention | HOLDS | none | no |
| 2 | MAE 0.455 dollars verifiable from committed cost-recovery CSV | HOLDS | none | no |
| 3 | OLS alpha = 1.009, IV alpha = 1.465 | HOLDS | none | no |
| 4 | Berry inversion: ln(s_j) - ln(s_0) = delta_j | HOLDS | none | no |
| 5 | Logit derivative: own = -alpha*s_j*(1-s_j), cross = alpha*s_k*s_j | HOLDS | none | no |
| 6 | Firms 1 and 2 own 2 products; 3 firms total | HOLDS | none | no |
| 7 | cost_shifter excluded from demand; used as instrument | HOLDS | none | no |
| 8 | All README table values match both CSVs | HOLDS | none | no |

## Findings

### Finding 1 (RESOLVED): Omega derivative subscript notation mismatch

**Original finding:** `compute_share_derivatives` docstring said `ds_j/dp_k` (row=share-changing, col=price-changing), while README used `ds_k/dp_j` (row=price-changing, col=share-changing). These are transposes, harmless under logit symmetry but dangerous if extended to asymmetric demand systems.

**Recheck evidence (verbatim):**
- `run.py:275`: `"""ds_k/dp_j matrix.  Own: -alpha*s_j*(1-s_j).  Cross: alpha*s_j*s_k."""`
- `README.md:33`: `$$\Omega_{jk}=-O_{jk}\frac{\partial s_k}{\partial p_j}.$$`

Docstring now says `ds_k/dp_j`, matching the README equation convention (row j is the price-changing product, column k is the share-changing product). The matrix entries are consistent: `D[j,k]` own (`j==k`): `-alpha*shares[j]*(1-shares[j])` = `-alpha*s_j*(1-s_j)` = `ds_j/dp_j`; cross (`j!=k`): `alpha*shares[j]*shares[k]` = `alpha*s_j*s_k` = `ds_k/dp_j` for k the share product. All conventions now aligned.

- **Category:** HOLDS
- **Verdict:** RESOLVED

---

### Finding 2 (RESOLVED): MAE 0.455 dollars not in any committed artifact

**Original finding:** README.md:74 stated "In market 0, recovered marginal costs have mean absolute error 0.455 dollars" but no committed CSV backed the number.

**Recheck evidence (verbatim):**
- `tables/cost-recovery-market0.csv` (new artifact): last row `Mean,,,,,0.455`
- `README.md:110`: `| Mean | | | | | 0.455 |`
- `run.py:586-606`: adds `tables/cost-recovery-market0.csv` via `report.add_table(...)`

The cost-recovery table is now a committed artifact. The MAE 0.455 appears in both the README and the CSV, independently verifiable. All five product rows and the Mean row match between README and CSV.

- **Category:** HOLDS
- **Verdict:** RESOLVED

---

## Cross-cutting patterns

None. Both original findings resolved. No new findings identified.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5%.** No action required.
1. `test_finding1_violated_invariant_docstring_uses_old_convention` FAILS -- correct; designed to fail after fix (docstring updated).
2. `test_finding1_honest_fix_docstring_matches_readme_convention` PASSES -- confirms docstring now says `ds_k/dp_j`.
3. `test_finding2_violated_invariant_mae_absent_from_estimation_csv` PASSES -- MAE not in estimation-results.csv (unchanged; correct).
4. `test_finding2_honest_fix_mae_present_in_cost_recovery_csv` PASSES -- cost-recovery-market0.csv now exists with MAE 0.455.
5. No further work needed on this tutorial.
