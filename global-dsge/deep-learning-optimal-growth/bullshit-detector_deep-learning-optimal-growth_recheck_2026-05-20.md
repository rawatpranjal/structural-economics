# bullshit-detector -- deep-learning-optimal-growth -- recheck -- 2026-05-20

**Bullshit score: 0%** -- all three prior findings are resolved; every claim now holds against code and committed CSV artifacts.

## Header
- Claim sources: `global-dsge/deep-learning-optimal-growth/README.md`
- Code / artifact root: `global-dsge/deep-learning-optimal-growth/run.py`
- Data artifacts: `global-dsge/deep-learning-optimal-growth/tables/training-summary.csv`
- Seed audit (if any): `bullshit-detector_deep-learning-optimal-growth_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Stability guard disclosed in Equations and pseudocode | HOLDS | -- | -- |
| 2 | Initial loss in CSV; ratio confirms "several orders of magnitude" | HOLDS | -- | -- |
| 3 | Mean saving share in CSV; matches README 0.3420 | HOLDS | -- | -- |
| 4 | Loss fn = mean(r^2) + 1e-3*stability guard | HOLDS | -- | -- |
| 5 | 1-16-16-1 tanh MLP architecture | HOLDS | -- | -- |
| 6 | k_ss=0.5524, c_ss=1.0629 | HOLDS | -- | -- |
| 7 | Training interval [0.138, 1.381] | HOLDS | -- | -- |
| 8 | Adam bias correction | HOLDS | -- | -- |
| 9 | Audit grid separate from training loss | HOLDS | -- | -- |
| 10 | CSV values match README table | HOLDS | -- | -- |

## Findings

### Prior finding 1 (DILUTED MED): stability guard undisclosed -- RESOLVED

- **Original claim:** README described the minimized object as pure `Xi_n = mean(r^2)` without disclosing the stability guard.
- **Current state:** `README.md:66-68` states verbatim: "The objective the code actually minimizes adds a light stability guard to $\Xi_n(\theta)$; that guard is zero during normal training and is described in the Solution Method section." Pseudocode at `README.md:158` states: `Set Xi_n(theta) = (1/n) sum_i r_i^2 + 1e-3 * (1/n) sum_i g_i^2`. Code at `run.py:137`: `return jnp.mean(residual**2) + 1e-3 * jnp.mean(lower_guard**2 + upper_guard**2)`. Disclosure is complete and accurate.
- **Category:** HOLDS

### Prior finding 2 (DATA DRIFT LOW): initial loss not in CSV -- RESOLVED

- **Original claim:** CSV had no `Initial loss` column; "several orders of magnitude" claim was ungrounded.
- **Current state:** `run.py:214`: `"initial_loss": float(log_losses[0])` captured at training step 1. `run.py:264-265`: `"Initial loss": float(train_log["initial_loss"])` in summary DataFrame. CSV column `Initial loss` present with value `0.07588020712137222`. Final loss `2.3155900308324817e-08`. Ratio ≈ 3.27 × 10^6 -- confirms "several orders of magnitude." Claim is now grounded in committed artifact.
- **Category:** HOLDS

### Prior finding 3 (DATA DRIFT LOW): mean saving share not in CSV -- RESOLVED

- **Original claim:** `mean_saving_share` was a runtime value not stored in any CSV artifact.
- **Current state:** `run.py:271`: `"Mean saving share": mean_saving_share` in summary DataFrame. CSV column `Mean saving share` present with value `0.34202349185943604`. README prose states `0.3420`; matches to 4 decimal places. Claim is now backed by committed artifact.
- **Category:** HOLDS

## Cross-cutting patterns

- All three prior findings shared the same root cause (undisclosed regularization; unarchived runtime scalars). Both have been corrected together: the guard is documented in two places, and the summary CSV now captures every scalar quoted in prose.
- No new findings. The tutorial's internal consistency (README prose, pseudocode, code, CSV) is complete.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required. All findings are HOLDS.
1. The three violated-invariant tests from the original audit now FAIL (as expected post-fix): the tests asserted the buggy state (guard undisclosed, columns absent) and the fixes have removed those states.
2. The three honest-fix tests now PASS: guard disclosed in README, `Initial loss` in CSV, `Mean saving share` in CSV.
3. No re-run or further code changes needed.
