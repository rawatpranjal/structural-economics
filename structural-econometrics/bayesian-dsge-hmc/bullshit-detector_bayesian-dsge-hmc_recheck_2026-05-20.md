# bullshit-detector - bayesian-dsge-hmc - recheck - 2026-05-20

**Bullshit score: 5%** - Three prior findings (F2 MISLABELED, F3 DILUTED HIGH, F4 DILUTED LOW) verified fixed. F5 DATA DRIFT/LOW (runtime numbers) remains unverifiable without a live re-run; all other numeric and algorithmic claims HOLD against code and posterior_summary.csv.

## Header
- Claim sources: `structural-econometrics/bayesian-dsge-hmc/README.md`
- Code / artifact root: `structural-econometrics/bayesian-dsge-hmc/run.py`
- Data artifacts: `structural-econometrics/bayesian-dsge-hmc/tables/posterior_summary.csv`
- Seed audit: `structural-econometrics/bayesian-dsge-hmc/bullshit-detector_bayesian-dsge-hmc_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Klein A, B matrices match code | HOLDS | - | - |
| 2 | 2x2 worked example (psi=4/3, gradients 8/9, 4/3, 8/9) | HOLDS | - | - |
| 3 | Kalman log-likelihood formula matches code | HOLDS | - | - |
| 4 | Prior hyperparameters and means match code | HOLDS | - | - |
| 5 | Jacobian log-correction formula matches code | HOLDS | - | - |
| 6 | Table numbers match posterior_summary.csv | HOLDS | - | - |
| 7 | RW-MH draws = 8000 (single chain) | HOLDS | - | - |
| 8 | "Four chains run independently ... one after another in a sequential loop" (prior F2, fixed) | HOLDS | - | - |
| 9 | Takeaway qualifies ESS comparison as per-sample, disclaims per-second (prior F3, fixed) | HOLDS | - | - |
| 10 | "The figure y-axis uses a log scale" (prior F4, fixed) | HOLDS | - | - |
| 11 | Runtime numbers (728.9s NUTS, 3.2s RW-MH, 0.91 accept, 0.02 accept) (prior F5, DATA DRIFT) | DATA DRIFT | LOW | no |

## Findings

### Residual Finding: Runtime numbers

- **Claim source (verbatim):** "NUTS ran 4 chains of 1000 warm-up plus 2000 kept draws in 728.9 seconds wall time. Random-walk Metropolis ran a single chain of 8000 draws in 3.2 seconds. NUTS reached an average acceptance of 0.91; RW-MH at the chosen step size reached 0.02." - `README.md:113`
- **Code evidence:** `run.py:628-634` - values are interpolated at report-generation time from live variables `t_nuts`, `t_rwmh`, `nuts_accept.mean()`, `rwmh_accept`. Not hardcoded.
- **Data evidence:** No `tables/` file stores runtime or acceptance. Cannot ground against a committed artifact without re-running.
- **Category:** DATA DRIFT - README reflects a past run; no committed artifact to verify against.
- **Severity:** LOW
- **Result-changing:** no (runtimes are informational; chain configuration 4x1000+2000 NUTS, 1x8000 RW-MH matches code constants `NUM_WARMUP=1000`, `NUM_SAMPLES=2000`, `NUM_CHAINS=4`)

## Cross-cutting patterns

**Prior F2 resolved (MISLABELED: "parallel" chains).** `README.md:85` now reads: "Four chains run independently from near-zero unconstrained starting points, one after another in a sequential loop." `run.py:280-284` still uses a sequential Python `for`-loop over `num_chains`. Prose now matches code. Fix is option A (prose-only) from the original TDD sequence; no `jax.vmap` or `pmap` was added. The test `test_f2_honest_fix_prose_does_not_call_chains_parallel` asserts "parallel" is absent from the relevant README sentence and passes.

**Prior F3 resolved (DILUTED HIGH: "per compute unit" unqualified).** `README.md:142` (Takeaway) now reads: "gradient-based sampling delivers one to several orders of magnitude more effective draws per raw sample than the random-walk baseline at the same total draw count. That per-sample mixing gain does not carry over to a per-wall-clock-second comparison: NUTS pays a large warm-up and JIT-compilation cost, so on this run RW-MH produces more effective draws per second on several parameters. The gradient-based advantage is in samples drawn, not in wall time." The prior unqualified claim "per compute unit" is gone. The disclaimer that RW-MH wins per-second is explicit. The test `test_f3_honest_fix_takeaway_qualifies_the_comparison` asserts "per raw sample" appears in the takeaway and passes.

**Prior F4 resolved (DILUTED LOW: ambiguous log-scale sentence).** `README.md:136` now reads: "The figure y-axis uses a log scale." The prior bare sentence "ESS is on a log scale." which could be misread as describing the table is gone. The test `test_f4_honest_fix_sentence_attributes_log_axis_to_figure` asserts "figure" or "y-axis" appears in the sentence and passes.

- All numeric claims groundable against committed artifacts (matrix entries, prior parameters, posterior table, ESS ratios) are exact. `posterior_summary.csv` matches `README.md:126-134` to full floating-point precision.
- The DATA DRIFT residual (F5) is LOW severity and informational. It resolves automatically on every `python run.py` re-run; no action required.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5%. No action required beyond the residual DATA DRIFT.**
1. All 6 tests at `tests/test_bayesian-dsge-hmc.py` pass: F2 violated-invariant + honest-fix, F3 violated-invariant + honest-fix, F4 violated-invariant + honest-fix. That is the correct green state.
2. F5 (runtime DATA DRIFT) has no test and cannot have one without a live re-run. Accept as residual LOW severity.
3. No sim re-runs or data artifact changes needed. All three fixes were prose-only changes in `run.py` report strings; no CSV content was affected.
