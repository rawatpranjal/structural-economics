# bullshit-detector -- merger-simulation -- recheck -- 2026-05-20

**Bullshit score: 15%** -- original DILUTED/HIGH finding fully resolved; logit alpha formula now includes the price factor and the posted result changed from 11.15% to 12.79% as expected; one new DATA DRIFT/LOW finding: six-product calibration FOC residuals cited in prose (2.8e-17 each) but not backed by any committed CSV, while the CSV column contains post-merger residuals (3.4e-10, 5.6e-17, 3.0e-12), different quantities.

## Header
- Claim sources: `industrial-organization/merger-simulation/README.md`
- Code / artifact root: `industrial-organization/merger-simulation/run.py`
- Data artifacts: `tables/merger-effects.csv`, `tables/four-product-results.csv`, `tables/market-hhi.csv`
- Seed audit: `bullshit-detector_merger-simulation_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | All 3 demand systems calibration (logit limitation correctly disclosed) | HOLDS | none | no |
| 2 | Logit avg price increase = 12.79% (corrected from 11.15%) | HOLDS | none | no |
| 3 | alpha = -2.7556 for six-product logit calibration | HOLDS | none | no |
| 4 | Six-product calibration FOC residuals 2.8e-17 each not in any committed CSV | DATA DRIFT | LOW | no |
| 5 | Four-product pre-merger FOC residual 5.6e-17 matches CSV | HOLDS | none | no |
| 6 | HHI, delta-HHI, effective-N formulas correct | HOLDS | none | no |
| 7 | Logit Jacobian and diversion ratio formulas correct | HOLDS | none | no |
| 8 | CSV table values match README tables (all three CSVs) | HOLDS | none | no |
| 9 | Screen Gap = Avg Actual - Avg GUPPI | HOLDS | none | no |

## Findings

### Finding 1 (RESOLVED): logit alpha formula omitted price factor

**Original finding:** `calibrate_logit` computed `alpha_estimates = -1.0 / (margins_obs * (1.0 - shares_obs))`, dropping the `prices_obs` factor, making the "matches margins by construction" claim false for logit. Posted result was logit avg price increase = 11.15% (wrong).

**Recheck evidence (verbatim):**
- `run.py:221-225`:
  ```python
  # Single-product logit FOC: 0 = s_j + (p_j - c_j) * alpha * s_j * (1 - s_j),
  # with p_j - c_j = m_j * p_j, gives alpha = -1 / (m_j * p_j * (1 - s_j)).
  # The price factor is load-bearing whenever observed prices are not unity.
  alpha_estimates = -1.0 / (margins_obs * prices_obs * (1.0 - shares_obs))
  alpha = float(np.mean(alpha_estimates))
  ```

Price factor `prices_obs` now included. Formula matches the single-product FOC derivation in the inline comment.

- `README.md:82-89` (rewritten): "Logit demand has a single price coefficient alpha. We pin it down from the average single-product margin condition and recover marginal costs from the pre-merger FOC, so the logit FOC residual is zero at calibration while its implied margins track the observed margins only as closely as one coefficient allows."

README now accurately distinguishes logit (FOC exact, margin approximate) from linear and log-linear (FOC exact, margin exact). No longer claims logit "matches margins by construction."

- `tables/merger-effects.csv:2`: `Logit,12.79,...` — posted result now 12.79%, matching the corrected formula output.

- **Category:** HOLDS
- **Verdict:** RESOLVED

---

### Finding 2 (NEW): Six-product calibration FOC residuals not in any committed CSV

- **Claim source (verbatim):** "FOC residuals after calibration: four-product baseline 5.6e-17; six-product extended (logit 2.8e-17, linear 2.8e-17, log-linear 2.8e-17)." -- `README.md:247`

- **Code evidence (verbatim):**
  ```python
  foc_check_logit = foc_logit(prices_obs, cal_logit["mc"], cal_logit["alpha"],
                              cal_logit["xi"], omega_pre)
  foc_check_linear = foc_linear(prices_obs, cal_linear["mc"], cal_linear["a"],
                                cal_linear["B"], omega_pre)
  foc_check_loglinear = foc_loglinear(prices_obs, cal_loglinear["mc"], cal_loglinear["a_ll"],
                                      cal_loglinear["E"], omega_pre)
  ```
  `run.py:485-490`

  These calibration residuals are printed to stdout at `run.py:574-576` but are not written to any CSV.

- **Data evidence (verbatim):** `tables/merger-effects.csv` column `Post FOC Residual`: `Logit: 3.4e-10`, `Linear: 5.6e-17`, `Log-linear: 3.0e-12`. These are POST-MERGER FOC residuals, not calibration residuals. A reader seeing `2.8e-17` in the prose and `3.4e-10` for logit in the CSV will conclude the numbers disagree. They measure different things; the README does not clarify this.

- **Category:** DATA DRIFT -- two sets of FOC residuals (calibration vs post-merger) for the same demand systems disagree, and neither the README nor the column header distinguishes them.

- **Severity:** LOW -- the code is correct; calibration works; the gap is documentation only.

- **Result-changing:** no -- the structural conclusions are unaffected.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "2.8e-17" in open("README.md").read() and not any("2.8e-17" in row for row in open("tables/merger-effects.csv"))
  # PASSES on current state (calibration residuals in prose only, not in CSV)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any("calibration" in col.lower() for col in pd.read_csv("tables/merger-effects.csv").columns)
  # PASSES if a calibration-residual column is added to merger-effects.csv or a separate CSV is committed
  ```

---

## Cross-cutting patterns

- The original root cause (alpha formula omitting price factor) is fixed. The corrected code produces 12.79% avg logit price increase; the README and CSV now agree and the logit vs linear vs log-linear comparison is honest.
- The new DATA DRIFT finding (calibration vs post-merger FOC residuals conflated in prose) is a documentation precision gap, not a code error. The fix is either to add a calibration-residual column to merger-effects.csv or to add a sentence in the prose distinguishing the two sets of residuals.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 15%.** Original HIGH finding resolved. One new LOW finding (DATA DRIFT). No halt required.
1. `test_finding1_violated_invariant_alpha_formula_drops_price_factor` FAILS -- correct; designed to fail after fix (price factor now present).
2. `test_finding1_honest_fix_alpha_formula_keeps_price_factor` PASSES -- confirms `prices_obs` in source.
3. `test_finding1_honest_fix_logit_avg_price_increase` PASSES -- confirms logit avg price increase >= 12% (the corrected value).
4. Finding 2 (DATA DRIFT/LOW): if precision matters, add calibration-residual columns to merger-effects.csv or note in README that the "Post FOC Residual" column is post-merger, not calibration.
5. No further work required on this tutorial unless the calibration residual documentation is a concern.
