# bullshit-detector — consumption-savings — recheck — 2026-05-20

**Bullshit score: 0%** — All structural, algorithmic, and numerical claims HOLD. The three original findings (MISLABELED MPC label, DILUTED pseudocode feasibility, DATA DRIFT convergence statistics) are fully resolved: MPC is now computed as dc/dw = (dc/da)/R; pseudocode shows `{ 0 <= g_l <= R a_i + z_j }`; all eight scalars are persisted in `tables/key-scalars.csv` with values consistent with the README.

## Header
- Claim sources: `dynamic-programming/consumption-savings/README.md`
- Code / artifact root: `dynamic-programming/consumption-savings/run.py`
- Data artifacts: `dynamic-programming/consumption-savings/tables/key-scalars.csv`
- Seed audit: `bullshit-detector_consumption-savings_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | VFI: 260 iterations, residual 9.91e-07 | HOLDS | none | no |
| 2 | MPC near zero assets = 0.51 (median z) | HOLDS | none | no |
| 3 | MPC near top assets = 0.04 (median z) | HOLDS | none | no |
| 4 | Refined-grid max gap = 2.55e-02 | HOLDS | none | no |
| 5 | Panel: 3,000 agents, 400 periods | HOLDS | none | no |
| 6 | Median wealth = 0.20 | HOLDS | none | no |
| 7 | P90 wealth = 1.85 | HOLDS | none | no |
| 8 | ~20.5% agents near constraint | HOLDS | none | no |
| 9 | Pseudocode feasibility set: { 0 <= g_l <= R a_i + z_j } | HOLDS | none | no |
| 10 | Bellman TV is beta-contraction | HOLDS | none | no |
| 11 | MPC is dc*/dw = (dc*/da)/R | HOLDS | none | no |
| 12 | CSV values match README table | HOLDS | none | no |

## Findings

### Finding 1 (original): MPC label was dc*/da, not dc*/dw — RESOLVED

- **Original claim source (verbatim):** "For the median income state, average MPC is **0.52** near zero assets and **0.04** near the top." — original README, as documented in seed audit.
- **Original finding:** MISLABELED — code reported `np.gradient(policy_c[:, 2], a_grid)` (dc/da), not the standard MPC dc/dw = (dc/da)/R.
- **Resolution:** `run.py:222-226`:
  ```python
  # MPC is the consumption response to cash-on-hand w = R a + z, not to a.
  # Since dw/da = R, the marginal propensity to consume out of cash-on-hand
  # is dc/dw = (dc/da) / R.
  dc_da_mid = np.gradient(policy_c[:, median_z_idx], a_grid)
  mpc_mid = dc_da_mid / gross_return
  ```
  The division by `gross_return` (= 1.03) is now explicit with an explanatory comment. The reported low-asset MPC (0.5051) and high-asset MPC (0.0416) are dc/dw values. README prose reports "0.51" and "0.04" (correct rounding). CSV stores 0.5051 and 0.0416 (four decimal places). All consistent.
- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 2 (original): Pseudocode feasibility omitted lower bound — RESOLVED

- **Original claim source (verbatim):** `"feasible(g_l) := { g_l <= R a_i + z_j }"` — original run.py pseudocode string.
- **Original finding:** DILUTED — lower bound `0 <= g_l` absent from pseudocode despite comment "respects no-borrowing".
- **Resolution:** `run.py:354`:
  ```python
  "            feasible(g_l) := { 0 <= g_l <= R a_i + z_j }       # no-borrowing and budget\n"
  ```
  Both bounds (`0 <= g_l` and `g_l <= R a_i + z_j`) are now present. The comment is updated to "# no-borrowing and budget". README renders as `README.md:98`: `feasible(g_l) := { 0 <= g_l <= R a_i + z_j }`. HOLDS.
- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 3 (original): Runtime scalars not persisted to a committed artifact — RESOLVED

- **Original finding:** DATA DRIFT — all eight convergence and simulation scalars were f-string-embedded in README with no CSV for cross-checking.
- **Resolution:** `run.py:470-504`. `tables/key-scalars.csv` now contains all eight quantities:
  ```
  Quantity,Value
  Main-grid VFI iterations,260
  Main-grid sup-norm residual,9.91e-07
  Refined-grid max gap (median z),2.55e-02
  MPC near zero assets (median z),0.5051
  MPC near top assets (median z),0.0416
  Simulated median wealth,0.2043
  Simulated P90 wealth,1.8467
  Share near constraint,0.2053
  ```
  Cross-check against README (`README.md:134-143`) — all eight values match exactly:
  - Iterations: CSV=260, README="260". Match.
  - Residual: CSV=9.91e-07, README="9.91e-07". Match.
  - Gap: CSV=2.55e-02, README="2.55e-02". Match.
  - MPC low: CSV=0.5051 -> formatted 0.51, README="0.51". Match.
  - MPC high: CSV=0.0416 -> formatted 0.04, README="0.04". Match.
  - Median: CSV=0.2043 -> formatted 0.20, README="0.20". Match.
  - P90: CSV=1.8467 -> formatted 1.85, README="1.85". Match.
  - Share: CSV=0.2053 -> formatted 20.5%, README="20.5%". Match.
- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

## Cross-cutting patterns

- All three original findings resolved cleanly. No new findings introduced.
- The MPC fix and pseudocode fix required changes only to `run.py`; the CSV fix required adding the `scalars_df` table and `report.add_table` call. All three changes are internal and consistent.
- No floating prose numbers remain. Every numeric claim in the README is either a fixed parameter (from `run.py:131-153`), a derivation from parameters, or a value persisted to `tables/key-scalars.csv`.
- The `key-scalars.csv` stores values at higher precision (4 decimal places) than the README prose, which uses 2 significant figures. This is correct audit practice: the CSV is the ground truth, the README displays rounded values.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No further action required. All three original findings are resolved.
1. The violated-invariant tests (`test_finding2_violated_invariant_pseudocode_omits_lower_bound`, `test_finding3_violated_invariant_no_scalars_csv`) correctly fail on the current repo state, confirming both fixes were applied.
2. All honest-fix tests (`test_finding1_run_py_divides_gradient_by_gross_return`, `test_finding1_honest_fix_reported_mpc_is_dc_dw`, `test_finding2_honest_fix_pseudocode_shows_lower_bound`, `test_finding3_honest_fix_scalars_csv_exists`, `test_finding3_readme_iterations_match_csv`) pass, confirming correctness.
3. No additional findings identified. No further TDD cycles needed.
