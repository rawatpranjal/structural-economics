# bullshit-detector — optimal-growth — recheck — 2026-05-20

**Bullshit score: 0%** — All structural, algorithmic, and numerical claims HOLD. All four original findings are resolved: (1) pseudocode now shows `kp_max <- min(0.9999 * y_i, k_max)`; (2-3) full-errors.csv carries all 500 grid points, so the comparison.csv max errors are correctly below the full-grid max; (4) tables/convergence-log.csv is committed with vfi_iterations, vfi_sup_norm_error, max_value_error_above_bottom_decile, and max_policy_error_above_bottom_decile — all matching README prose.

## Header
- Claim sources: `dynamic-programming/optimal-growth/README.md`
- Code / artifact root: `dynamic-programming/optimal-growth/run.py`
- Data artifacts: `dynamic-programming/optimal-growth/tables/convergence-log.csv`, `dynamic-programming/optimal-growth/tables/full-errors.csv`, `dynamic-programming/optimal-growth/tables/comparison.csv`
- Seed audit: `bullshit-detector_optimal-growth_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Pseudocode: kp_max <- min(0.9999 * y_i, k_max) | HOLDS | none | no |
| 2 | Max value error = 1.91e-05 (outside bottom decile) | HOLDS | none | no |
| 3 | Max policy error = 2.87e-02 (outside bottom decile) | HOLDS | none | no |
| 4 | VFI converges in 143 iterations, residual 9.32e-07 | HOLDS | none | no |
| 5 | convergence-log.csv persists all four baseline scalars | HOLDS | none | no |
| 6 | full-errors.csv covers all 500 grid points | HOLDS | none | no |
| 7 | comparison.csv sampled max errors below full-grid max | HOLDS | none | no |

## Findings

### Finding 1 (original): Pseudocode omitted the 0.9999 feasibility factor — RESOLVED

- **Original finding:** DILUTED — pseudocode showed `kp_max <- min(y_i, k_max)`, omitting the `0.9999` strict-interior factor that prevents the policy from reaching the grid boundary.
- **Resolution:** `README.md:95` now reads:
  ```
  kp_max <- min(0.9999 * y_i, k_max)
  ```
  The `0.9999` factor is present. `run.py` computes `kp_max = jnp.minimum(0.9999 * net_output, capital_grid[-1])` at the corresponding line. README pseudocode matches code. HOLDS.
- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Findings 2-3 (original): comparison.csv max errors exceeded README claims — RESOLVED

- **Original finding:** DATA DRIFT — README claimed max value error 1.91e-05 and max policy error 2.87e-02 "outside the bottom decile," but `comparison.csv` (8 sampled rows) showed a max value error of 1.53e-05 and max kp error of 1.98e-02, both below the stated maxima. No artifact existed with the full 500-point grid to confirm the maxima.
- **Resolution:** `tables/full-errors.csv` now exists with 500 data rows (header + 500 = 501 lines confirmed by `wc -l`). The full-grid values:
  - Max value error (rows[50:], outside bottom decile): `1.9073e-05` → formatted `:.2e` → `1.91e-05` → README: "1.91e-05". Match.
  - Max policy error (rows[50:], outside bottom decile): `0.028688` → formatted `:.2e` → `2.87e-02` → README: "2.87e-02". Match.
  - The `comparison.csv` sampled max (1.53e-05 value, 1.98e-02 policy) is correctly below the full-grid max, confirming the 8-row sample was never intended to be the extremum source.
  Both README claims are now grounded in a committed 500-row artifact. HOLDS.
- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 4 (original): Convergence statistics not persisted to a committed artifact — RESOLVED

- **Original finding:** DATA DRIFT — VFI iteration count (143), sup-norm residual (9.32e-07), max value error, and max policy error appeared only in README prose from f-strings; no committed CSV held them for cross-checking.
- **Resolution:** `tables/convergence-log.csv` now exists. Contents:
  ```
  Quantity,Value
  vfi_iterations,143.0
  vfi_sup_norm_error,9.318680e-07
  max_value_error_above_bottom_decile,1.9073486e-05
  max_policy_error_above_bottom_decile,0.028688430
  ```
  Cross-check against README:
  - `vfi_iterations = 143.0` → README: "143 iterations". Match.
  - `vfi_sup_norm_error = 9.318680e-07` → formatted `:.2e` → `9.32e-07` → README: "9.32e-07". Match.
  - `max_value_error_above_bottom_decile = 1.9073486e-05` → `:.2e` → `1.91e-05` → README: "1.91e-05". Match.
  - `max_policy_error_above_bottom_decile = 0.028688430` → `:.2e` → `2.87e-02` → README: "2.87e-02". Match.
  All four baseline scalars are now grounded in a committed artifact. HOLDS.
- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

## Cross-cutting patterns

- None. All four original findings resolved cleanly. No new findings.
- `full-errors.csv` (500 rows) is the key addition that resolves findings 2-3. It is the ground-truth artifact for the max-error claims; `comparison.csv` (8 sampled rows) is a pedagogical subset, correctly showing values below the full-grid max.
- `convergence-log.csv` is written directly via `convergence_df.to_csv(...)` outside the `ModelReport` pipeline, so it is regenerated on every `python run.py` call.
- All numeric claims in the README are now grounded: fixed parameters in `run.py`, the 500-row `full-errors.csv`, and `convergence-log.csv`. No floating prose numbers remain.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No further action required.
1. The violated-invariant tests (`test_finding1_violated_invariant` for missing `0.9999`, `test_finding4_violated_invariant` for absent convergence-log.csv) both correctly FAIL on the current repo state, confirming both fixes were applied.
2. All honest-fix tests (`test_finding1_honest_fix`, `test_finding2_3_violated_invariant`, `test_finding2_honest_fix`, `test_finding3_honest_fix`, `test_finding4_honest_fix`) pass, confirming correctness.
3. No additional findings identified. No further TDD cycles needed.
