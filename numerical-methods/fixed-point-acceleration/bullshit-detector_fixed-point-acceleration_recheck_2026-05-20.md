# bullshit-detector - fixed-point-acceleration-recheck - 2026-05-20

**Bullshit score: 0%** - All 30 claims ground cleanly against code and data. Both original findings (FALSE status label, DILUTED stress-sweep prose) are resolved. No remaining gaps.

## Header

- Claim sources: `numerical-methods/fixed-point-acceleration/README.md`
- Code / artifact root: `numerical-methods/fixed-point-acceleration/run.py`
- Data artifacts: `numerical-methods/fixed-point-acceleration/tables/method_comparison.csv`, `tables/stress_test.csv`, `tables/cournot_summary.csv`
- Seed audit: `numerical-methods/fixed-point-acceleration/bullshit-detector_fixed-point-acceleration_2026-05-20.md`
- Run by: bullshit-detector agent (Claude Sonnet 4.6), 2026-05-20 (recheck pass)
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Picard reaches tolerance in 146 iterations | HOLDS | - | - |
| 2 | Damped Picard exhausts 200-iter budget without crossing tolerance | HOLDS | - | - |
| 3 | Damped Picard residual 2.51e-09 at stop | HOLDS | - | - |
| 4 | Anderson at m=5 converges in 14 iterations | HOLDS | - | - |
| 5 | Anderson faster than Picard by factor 10.4 | HOLDS | - | - |
| 6 | Damped Picard Status = "stopped at max_iter = 200" | HOLDS | - | - |
| 7 | Picard Status = "converged" | HOLDS | - | - |
| 8 | Anderson Status = "converged" | HOLDS | - | - |
| 9 | Picard final residual 8.44e-13 | HOLDS | - | - |
| 10 | Picard distance to closed form 4.57e-12 | HOLDS | - | - |
| 11 | Anderson final residual 7.95e-14 | HOLDS | - | - |
| 12 | Anderson distance to closed form 5.08e-13 | HOLDS | - | - |
| 13 | Stress test sweeps from 0.1 down to 0.01 | HOLDS | - | - |
| 14 | Stress row s0=0.10: Picard 200 / 3.92e-11; Anderson 16 / 9.6e-14 | HOLDS | - | - |
| 15 | Stress row s0=0.05: Picard 200 / 1.38e-06; Anderson 20 / 4.11e-15 | HOLDS | - | - |
| 16 | Stress row s0=0.02: Picard 200 / 0.000328; Anderson 22 / 1.33e-15 | HOLDS | - | - |
| 17 | Stress row s0=0.01: Picard 200 / 0.00147; Anderson 21 / 1.47e-13 | HOLDS | - | - |
| 18 | Cournot Nash q* = 3.0000 for both firms | HOLDS | - | - |
| 19 | Vanilla Picard Cournot: 44 iters, residual 5.12e-13 | HOLDS | - | - |
| 20 | Damped Picard Cournot: 22 iters, residual 5.12e-13 | HOLDS | - | - |
| 21 | Closed-form benchmark delta*_j = log(s_j^obs) - log(s_0^obs) | HOLDS | - | - |
| 22 | Fixed-point map T_j(delta) = delta_j + log(s_j^obs) - log(s_j(delta)) | HOLDS | - | - |
| 23 | Anderson gamma_t = argmin_gamma ||f_t - F_t gamma||_2 | HOLDS | - | - |
| 24 | Anderson update delta^{t+1} = g_t - G_t gamma_t | HOLDS | - | - |
| 25 | Safeguard reverts to damped Picard when residual more than doubles | HOLDS | - | - |
| 26 | Cournot BR_i(q_{-i}) = (a-c-q_{-i})/2 | HOLDS | - | - |
| 27 | Cournot q* = (a-c)/3 | HOLDS | - | - |
| 28 | Vanilla Picard first Cournot step overshoots to (4.5, 4.5) | HOLDS | - | - |
| 29 | Anderson reduces to Picard when m=0 | HOLDS | - | - |
| 30 | Status column reflects actual termination condition | HOLDS | - | - |

## Findings

### Finding 1 (original): Damped Picard labeled "converged" despite hitting max_iter

- **Original category:** FALSE, HIGH
- **Recheck verdict:** RESOLVED. `run.py:600-610` now defines `termination_status()` which returns `"converged"` only when `residual < tol_`, else returns `f"stopped at max_iter = {max_it}"`. `dp_residuals[-1] = 2.51e-09 > 1e-12 = tol` triggers the else branch. `tables/method_comparison.csv:3` now reads `Damped Picard,damping alpha = 0.5,200,2.51e-09,2.97e-08,stopped at max_iter = 200`. `README.md:199` displays `stopped at max_iter = 200`. The hardcoded `["converged", "converged", "converged"]` list from the original buggy code is gone; the string is absent from `run.py`. **HOLDS.**

### Finding 2 (original): Stress-test prose claimed sweep "from a benign 0.5 down to 0.01"

- **Original category:** DILUTED, MED
- **Recheck verdict:** RESOLVED. `run.py:551` now reads `"The stress test sweeps the outside share from 0.1 down to 0.01. "`. The phrase `"from a benign 0.5"` is absent from `run.py`. `README.md:184` reads "The stress test sweeps the outside share from 0.1 down to 0.01." The four-row CSV (`tables/stress_test.csv`) starts at 0.10, confirming prose and data agree. **HOLDS.**

## Cross-cutting patterns

- All algorithm algebra (Anderson F/G matrices, gamma solve, Picard update, damped Picard update, Cournot BR map, closed-form benchmark) continues to match the equations precisely. No regressions introduced by the fix.
- Status column is now derived from the same tolerance used by the stopping rule, making the termination report self-consistent by construction.
- The stress prose now states the actual feasible sweep range. No silent skip is misrepresented anywhere.
- All 30 extractable claims ground to verbatim code or CSV rows. The fix addressed both original findings without introducing new gaps.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** Both original findings resolved. No further fixing required.
1. The test suite at `tests/test_run.py` encodes both findings:
   - `test_finding1_violated_invariant_status_hardcoded_converged` — marked `xfail(strict=True)`; passes (xfails) on fixed code.
   - `test_finding1_honest_fix_damped_picard_status_reflects_max_iter` — passes on fixed code.
   - `test_finding1_status_is_consistent_with_residual_for_every_row` — passes on fixed code.
   - `test_finding2_violated_invariant_prose_claims_0_5_start` — marked `xfail(strict=True)`; passes (xfails) on fixed code.
   - `test_finding2_honest_fix_prose_states_actual_start` — passes on fixed code.
   - `test_finding2_stress_table_starts_at_actual_first_share` — passes on fixed code.
2. `python -m pytest tests/ -q` reports `4 passed, 2 xfailed`. This is the expected green state.
3. No further action required. The tutorial is faithful.
