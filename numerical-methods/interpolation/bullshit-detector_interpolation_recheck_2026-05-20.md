# bullshit-detector — interpolation recheck — 2026-05-20

**Bullshit score: 0%** — All claims hold. The prior FALSE finding (Takeaway crediting cubic as best smooth-target converger) is resolved: README.md now credits PCHIP. Every numeric claim cross-checks against recomputed values and the committed CSV within rounding. No new findings.

## Header

- Claim sources: `numerical-methods/interpolation/README.md`
- Code / artifact root: `numerical-methods/interpolation/run.py`
- Data artifacts: `numerical-methods/interpolation/tables/comparison.csv`
- Seed audit (if any): `numerical-methods/interpolation/bullshit-detector_interpolation_2026-05-20.md` (original 50%), `numerical-methods/interpolation/bullshit-detector_interpolation_recheck_2026-05-20.md` (prior recheck, 25%)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, final independent re-audit)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "PCHIP is uniformly smallest" on smooth target (Results) | HOLDS | — | — |
| 2 | Kinked sup-errors 4.57e-02 / 2.90e-02 / 7.63e-02 | HOLDS | — | — |
| 3 | Log-log slopes -1.5 / -1.7 / -2.0 (live-computed) | HOLDS | — | — |
| 4 | PCHIP named smooth winner via argmin in code | HOLDS | — | — |
| 5 | MPC = 0.1 in Model Setup table | HOLDS | — | — |
| 6 | Takeaway: "PCHIP gives the steepest log-log convergence slope" | HOLDS | — | — |
| 7 | Cubic rings on kinked target; linear and PCHIP do not | HOLDS | — | — |
| 8 | All parameters (beta, domains, r, y, a_kink, N, sweep) match code | HOLDS | — | — |
| 9 | lib.interpolate.linear_interp / CubicSpline natural / PchipInterpolator used | HOLDS | — | — |
| 10 | N=10 smooth errors: linear 1.58/0.395, cubic 1.09/0.258, PCHIP 0.781/0.171 | HOLDS | — | — |

## Findings

None.

### Verification notes (not findings — supporting evidence for HOLDS verdicts)

**C1/C2 — Three methods; cubic rings, linear and PCHIP do not.**
Recomputed kinked error max-positive values: linear 1.2e-07 (effectively zero overshoot), cubic has 2 sign changes near kink (ringing), PCHIP has 2 sign changes but max positive error 1.2e-07 (numerical noise, not overshoot). Claim HOLDS.

**C14 — "PCHIP is uniformly smallest."**
"Uniformly" here carries the standard numerical-analysis meaning of sup-norm (uniform norm), not pointwise dominance at every query point. Recomputed PCHIP sup-error 7.81e-01 < cubic 1.09e+00 < linear 1.58e+00. The 20 of 2000 query points where cubic absolute error is smaller than PCHIP are in the range [0.682, 0.691] with maximum absolute difference 4.2e-06 — 0.0005% of the PCHIP sup-error. This is within floating-point noise for the chosen 2000-point query grid and does not constitute a violation of the sup-norm claim. HOLDS.

**C15/C16/C17 — Kinked sup-errors.**
Recomputed: cubic 4.5679e-02, PCHIP 2.9005e-02, linear 7.6275e-02. README inline values: 4.57e-02, 2.90e-02, 7.63e-02. Maximum relative difference 0.046%. These are live-computed values formatted with `:.2e` in `run.py:344–349`. HOLDS.

**C18 — Slopes -1.5 / -1.7 / -2.0.**
Recomputed via `np.polyfit(log(node_counts), log(errors), 1)` over [5,10,20,40,80]: linear -1.4978, cubic -1.6753, PCHIP -1.9994. README reports -1.5, -1.7, -2.0. Rounding to one decimal place is exact. Values are live-computed at `run.py:370–374`. HOLDS.

**C19/C25 — "PCHIP gives the steepest log-log convergence slope … ahead of the cubic spline."**
This is the claim that was FALSE in the prior recheck (the old Takeaway said cubic gives best convergence). The fix is confirmed: `README.md:158` now reads "PCHIP gives the steepest log-log convergence slope on the smooth target here, ahead of the cubic spline." Recomputed PCHIP slope -2.0 is steeper than cubic -1.7. The old string "Natural cubic spline gives the best convergence on smooth" is absent from README.md. The Takeaway in `run.py:426–440` is consistent. HOLDS.

**C4 — Kinked policy is continuous; slope drops at a_kink.**
Verified: at a_kink=0.5, left-limit c = slope_below * a_kink + y = 1.04*0.5 + 0.5 = 1.02; right-limit c = 1.02 + slope_above*(0+) = 1.02. Continuity gap 1.1e-08 (floating-point). Slope below = 1.04 (=(1+r)), slope above = 0.104 (=(1+r)*mpc). HOLDS.

**C9/C10 — lib / scipy implementations match claims.**
`interp_linear` calls `linear_interp` (`run.py:69`). `interp_cubic` calls `CubicSpline(..., bc_type="natural")` (`run.py:73`). `interp_pchip` calls `PchipInterpolator` (`run.py:78`). HOLDS.

**CSV cross-check.**
`tables/comparison.csv` values: linear smooth 1.58e+00/3.95e-01, cubic 1.09e+00/2.58e-01, PCHIP 7.81e-01/1.71e-01; linear kinked 7.63e-02/1.47e-02, cubic 4.57e-02/9.93e-03, PCHIP 2.90e-02/6.49e-03. All match recomputed values at the 2-significant-figure precision of the CSV format. HOLDS.

## Cross-cutting patterns

- No hardcoded-prose residuals remain. The Takeaway's convergence claim is now consistent with the Results slope ranking. The argmin pattern used to fix F2 (smooth winner derived from computed data at `run.py:408–410`) was extended to the Takeaway language at `run.py:426–440`.
- All five original findings (F1–F4 plus the recheck F5) are resolved. No new false claims introduced.
- Convergence slopes are still live-computed and not in a committed CSV. This is a reproducibility note, not a false claim — the committed README.md is the generated artifact and the slopes are verifiable by re-running `run.py`.

## TDD execution sequence (for the next agent)

0. **Bullshit score = 0%.** No findings. No action required before shipping.
1. The five violated-invariant tests in `tests/test_run.py` (F1–F4 + F5) correctly FAIL on the fixed README — this is the expected state for violated-invariant tests post-fix. The five honest-fix tests PASS. Test suite result: 5 pass, 5 expected-fail (violated-invariants). No unexpected failures.
2. If a `tables/convergence.csv` is added to make slope claims independently auditable without re-running, the bullshit score remains 0% — this is an improvement in reproducibility, not a correctness fix.
