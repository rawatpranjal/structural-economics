# bullshit-detector — merger-simulation — 2026-05-20

**Bullshit score: 40%** — one DILUTED/HIGH finding: `calibrate_logit` omits the price factor from the alpha formula, so the logit demand system does not match observed margins despite the explicit claim that all three systems do "by construction"; the reported logit avg price increase (11.15%) shifts to ~12.79% under the correct formula, a 14.7% relative change in the lead result.

## Header
- Claim sources: `industrial-organization/merger-simulation/README.md` (Equations section, Model Setup table, Results prose, Results tables)
- Code / artifact root: `industrial-organization/merger-simulation/run.py`; `industrial-organization/merger-simulation/tables/*.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | All 3 demand systems match observed margins "by construction" | DILUTED | HIGH | yes (logit avg price increase 11.15% vs ~12.79% correct, +14.7% relative) |
| 2 | HHI, delta-HHI, effective-N formulas correct | HOLDS | none | no |
| 3 | alpha = -2.3529 for 4-product logit calibration | HOLDS | none | no |
| 4 | alpha = -3.1611 for 6-product logit calibration | HOLDS | none | no |
| 5 | Bertrand-Nash FOC vector form correct | HOLDS | none | no |
| 6 | Logit Jacobian formula and diversion ratio formula correct | HOLDS | none | no |
| 7 | CSV table values match README tables | HOLDS | none | no |
| 8 | FOC residuals at calibration (four-product 5.6e-17, logit 1.4e-17, linear 2.8e-17, log-linear 2.8e-17) | HOLDS | none | no |
| 9 | Linear and log-linear demand systems match observed margins by construction | HOLDS | none | no |
| 10 | Efficiency frontier averages over products 1-4 ([:4] slice) | HOLDS | none | no |
| 11 | Screen Gap = Avg Actual - Avg GUPPI | HOLDS | none | no |

## Findings

### Finding 1: logit 6-product calibration omits price factor from alpha formula

- **Claim source (verbatim):** "We calibrate three demand systems. Each one matches the observed shares, prices, and margins by construction." — `README.md`, Equations section C, paragraph preceding the logit formula block (line 82 of README)

- **Code evidence (verbatim):**
  ```python
  def calibrate_logit(shares_obs: np.ndarray, prices_obs: np.ndarray,
                      margins_obs: np.ndarray, p2f: np.ndarray) -> dict:
      """Calibrate logit demand from observed shares, prices, and a margin vector.

      Strategy: average each product's single-product alpha estimate to pin down
      a single price coefficient, then invert the full pricing FOC for marginal
      costs.
      """
      omega = ownership_matrix(p2f)
      s0_total = np.sum(shares_obs)
      alpha_estimates = -1.0 / (margins_obs * (1.0 - shares_obs))
      alpha = float(np.mean(alpha_estimates))
  ```
  `run.py:211-222`

- **Data evidence:** The single-product logit FOC for product j is `0 = s_j + (p_j - c_j) * alpha * s_j * (1-s_j)`, which gives `alpha = -1 / ((p_j - c_j) * (1 - s_j)) = -1 / (m_j * p_j * (1 - s_j))`. The code computes `alpha_estimates = -1.0 / (margins_obs * (1.0 - shares_obs))`, dropping the `prices_obs` factor from the denominator. With `prices_obs = [1.0, 1.2, 0.9, 1.1, 1.3, 1.4]`, this is numerically wrong for all products except P1 (price = 1.0). The resulting `alpha = -3.1611` (code) vs correct average `-2.7556`. When `mc` is recovered by FOC inversion using the wrong `alpha`, implied margins deviate from observed margins by up to 3.82 percentage points (max 13.6% relative error, at product P3). Verified: `tables/merger-effects.csv` row Logit: `Avg Actual Price Inc. (%) = 11.15`; re-running with the correct alpha formula gives `~12.79%` — a 1.64 pp shift in the lead result number.

  The 4-product `calibrate_logit_simple` (used in Part B) does NOT have this bug because it derives alpha directly from `(p1 - c1)` rather than from `margin * (1-share)`, and `p1 = 1.0` anyway (`run.py:202-203`).

- **Category:** DILUTED — the code does perform logit calibration and FOC inversion; it omits one load-bearing factor from the alpha estimate formula, so the claim that it "matches margins by construction" is false for logit while true for linear and log-linear.

- **Severity:** HIGH — the bug propagates to the posted result `11.15%` in both `README.md` and `tables/merger-effects.csv`.

- **Result-changing:** yes — logit avg price increase shifts from 11.15% (reported) to ~12.79% (correct formula), a 14.7% relative change in the primary output of the tutorial's lead demand system.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not any(abs(p - 1.0) > 1e-6 for p in [1.0, 1.2, 0.9, 1.1, 1.3, 1.4]) or \
      abs(np.mean(-1.0 / (margins * (1.0 - shares))) - np.mean(-1.0 / (margins * prices * (1.0 - shares)))) < 1e-6
  # PASSES on current buggy code when prices != 1 (the two averages differ); FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert max(abs((prices_obs - mc_logit) / prices_obs - margins_obs)) < 0.01
  # PASSES on honest fix (margins matched); FAILS on current buggy code (max error 0.0382)
  ```

## Cross-cutting patterns

- The bug is confined to `calibrate_logit` (the multi-product version at `run.py:211`). The four-product `calibrate_logit_simple` at `run.py:193` is correct because it uses `(p - c)` directly, not the `margin * (1-share)` shortcut. The pattern is: the shortcut `margin * (1-share)` is only valid at `p = 1.0`; any tutorial that calibrates logit against non-unit prices using this shortcut will carry the same bug.
- Linear and log-linear calibrations both solve for own-price slopes and cross-price slopes from the margin data directly (not from a scalar alpha average), so they achieve exact margin-matching. The claim "by construction" holds for both; it fails only for logit.
- All HHI/concentration arithmetic is correct and self-consistent end to end.
- All CSV table values exactly match the numbers printed in `README.md` — the data artifact is consistent with the code that generated it. The only mismatch is between the code's stated intent (margin-matching) and what the code actually achieves.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 40%.** Surface to the user before writing fix code. The bug changes a posted result by 14.7%.

1. Turn the violated invariant into a pytest test: verify that `calibrate_logit` with non-unit prices produces `max(abs((p - mc)/p - margins)) > 0.01`. Confirm this PASSES on current code (proves the bug is real).

2. Turn the honest-fix pass condition into a second pytest test: verify that after the fix, `max(abs((p - mc)/p - margins)) < 0.01`. Confirm this FAILS on current code.

3. Fix `run.py:221`: change `alpha_estimates = -1.0 / (margins_obs * (1.0 - shares_obs))` to `alpha_estimates = -1.0 / (margins_obs * prices_obs * (1.0 - shares_obs))`. Verify both tests flip.

4. Regenerate `README.md` and `tables/merger-effects.csv` with `python run.py`. The logit row will change. Re-run this skill to confirm Finding 1 now reads HOLDS and the score drops to <=10%.
