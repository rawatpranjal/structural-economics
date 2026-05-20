# bullshit-detector — job-search-mccall — recheck — 2026-05-20

**Bullshit score: 0%** — All structural, algorithmic, and numerical claims HOLD. The two original findings are resolved: (1) the expected-duration prose format changed from `:.0f` (which rounded 16.468 to "16") to `:.1f` (which rounds to "16.5"), now consistent with the reservation-wages table; (2) tables/baseline-stats.csv is committed with vfi_iterations, vfi_sup_norm_error, w_star_grid, w_star_cont, abs_grid_error, accept_frac_cont, and expected_duration_cont — every baseline numeric in the Solution Method and Results sections is now grounded in a committed artifact.

## Header
- Claim sources: `dynamic-programming/job-search-mccall/README.md`
- Code / artifact root: `dynamic-programming/job-search-mccall/run.py`
- Data artifacts: `dynamic-programming/job-search-mccall/tables/reservation-wages.csv`, `dynamic-programming/job-search-mccall/tables/baseline-stats.csv`
- Seed audit: `bullshit-detector_job-search-mccall_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | VFI converges in 178 iterations | HOLDS | none | no |
| 2 | Sup-norm error = 9.84e-09 | HOLDS | none | no |
| 3 | w*_grid = 4.7054 | HOLDS | none | no |
| 4 | w*_cont = 4.7055 | HOLDS | none | no |
| 5 | Absolute grid error = 9.1e-05 | HOLDS | none | no |
| 6 | Acceptance probability = 6.1% | HOLDS | none | no |
| 7 | Expected duration = 16.5 periods | HOLDS | none | no |
| 8 | 50 bins; conditional mean preserves E[W] | HOLDS | none | no |
| 9 | Mean offer E[W] = 1.6487 | HOLDS | none | no |
| 10 | Median offer = 1.0000 | HOLDS | none | no |
| 11 | Continuous benchmark solved by Brent's method | HOLDS | none | no |
| 12 | Reservation-wages CSV: 9 rows x 8 columns, all values consistent | HOLDS | none | no |
| 13 | baseline-stats.csv with vfi_iterations and vfi_sup_norm_error | HOLDS | none | no |

## Findings

### Finding 1 (original): Expected-duration prose used :.0f format — RESOLVED

- **Original finding:** DATA DRIFT — the prose used `:.0f` format on `expected_duration_cont = 16.468`, producing "16 periods" via Python banker's rounding, while the reservation-wages table showed "16.5" (formatted `:.1f`).
- **Resolution:** `run.py:336-337`:
  ```python
  f"Expected unemployment duration is about **{expected_duration_cont:.1f} "
  "periods**."
  ```
  Format string is now `:.1f`. `baseline-stats.csv:8`: `expected_duration_cont,16.46818023940088` → `:.1f` → `16.5`. README:97: "about **16.5 periods**". `reservation-wages.csv:6` (beta=0.95, b=1.0): `E[duration],16.5`. All three consistent.
- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 2 (original): Baseline sup-norm error not persisted to a committed artifact — RESOLVED

- **Original finding:** DATA DRIFT — `9.84e-09` appeared only in README prose from an f-string; no committed artifact held the baseline error or any other baseline scalar.
- **Resolution:** `run.py:478-503` writes `tables/baseline-stats.csv` directly (not via `report.add_table`, so it is always written regardless of report structure). The file contains 7 rows covering `vfi_iterations`, `vfi_sup_norm_error`, `w_star_grid`, `w_star_cont`, `abs_grid_error`, `accept_frac_cont`, and `expected_duration_cont`. Cross-check:
  - `vfi_iterations = 178.0` → README:93: "178 iterations". Match.
  - `vfi_sup_norm_error = 9.838e-09` → formatted `:.2e` → `9.84e-09` → README:93: "9.84e-09". Match.
  - `w_star_grid = 4.7054...` → `:.4f` → `4.7054` → README:93: "4.7054". Match.
  - `w_star_cont = 4.7055...` → `:.4f` → `4.7055` → README:93: "4.7055". Match.
  - `abs_grid_error = 9.132e-05` → `:.1e` → `9.1e-05` → README:93: "9.1e-05". Match.
  - `accept_frac_cont = 0.06072` → `100 * .1f` → `6.1%` → README:97: "6.1%". Match.
  - `expected_duration_cont = 16.468` → `:.1f` → `16.5` → README:97: "16.5 periods". Match.
- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

## Cross-cutting patterns

- None. Both original findings resolved cleanly. No new findings.
- The baseline-stats.csv is written directly via `baseline_stats.to_csv(...)` at `run.py:502-504`, outside the `ModelReport` pipeline. This means it is regenerated on every `python run.py` call regardless of whether figures or tables are skipped in the report. The pipeline is robust.
- All seven baseline scalars in the README are now grounded in `baseline-stats.csv`. The reservation-wages table (9 rows × 8 columns) is grounded in `reservation-wages.csv`. No floating prose numbers remain.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No further action required.
1. The violated-invariant tests (`test_finding1_violated_invariant` for the `:.0f` format, `test_finding2_violated_invariant` for baseline-stats absence) both correctly fail on the current repo state, confirming both fixes were applied.
2. All honest-fix tests (`test_finding1_honest_fix`, `test_finding2_honest_fix`) pass, confirming correctness.
3. No additional findings identified. No further TDD cycles needed.
