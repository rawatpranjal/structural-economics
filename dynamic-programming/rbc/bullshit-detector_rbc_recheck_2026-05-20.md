# bullshit-detector — rbc — recheck — 2026-05-20

**Bullshit score: 0%** — Original DATA DRIFT finding fully remediated: `tables/fine-grid-audit.csv` now committed, all seven fine-grid audit numbers in README confirmed against that artifact. All 20 claims hold.

## Header

- Claim sources: `dynamic-programming/rbc/README.md`
- Code / artifact root: `dynamic-programming/rbc/run.py`
- Data artifacts: `dynamic-programming/rbc/tables/business-cycle-stats.csv`, `dynamic-programming/rbc/tables/fine-grid-audit.csv`
- Seed audit: `bullshit-detector_rbc_2026-05-20.md` (1 non-HOLDS finding: DATA DRIFT)
- Run by: bullshit-detector skill (claude-sonnet-4-6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | bench_V_rel = 2.1e-04 (README) matches CSV | HOLDS | — | — |
| 2 | bench_k_max_abs = 0.0461 (README) matches CSV | HOLDS | — | — |
| 3 | bench_l_max_abs = 0.0150 (README) matches CSV | HOLDS | — | — |
| 4 | coarse VFI 515 iterations (README) matches CSV | HOLDS | — | — |
| 5 | coarse error 9.95e-06 (README) matches CSV | HOLDS | — | — |
| 6 | fine VFI 525 iterations (README) matches CSV | HOLDS | — | — |
| 7 | fine error 9.96e-06 (README) matches CSV | HOLDS | — | — |
| 8 | k_ss = 10.4980 | HOLDS | — | — |
| 9 | l_ss = 0.3330 | HOLDS | — | — |
| 10 | c_ss = 0.8073 | HOLDS | — | — |
| 11 | i_ss = 0.2446 | HOLDS | — | — |
| 12 | EV = V @ P.T broadcasts correctly | HOLDS | — | — |
| 13 | flow_utility tensor shape (n_k, n_z, n_l, n_k) | HOLDS | — | — |
| 14 | i_t = k_{t+1} - (1-delta)*k_t in simulation | HOLDS | — | — |
| 15 | HP filter (I + lam*D'D)^{-1}*y with lam=1600 | HOLDS | — | — |
| 16 | moments table matches business-cycle-stats.csv | HOLDS | — | — |
| 17 | T_sim=5000, T_burn=500 | HOLDS | — | — |
| 18 | Coarse grid 50x50 | HOLDS | — | — |
| 19 | Fine grid 200x100 | HOLDS | — | — |
| 20 | Pseudocode matches code tensor operations | HOLDS | — | — |

## Findings

None.

## Cross-cutting patterns

- Original Finding 1 (DATA DRIFT, LOW): The `fine-grid-audit.csv` artifact is now committed at `run.py:543–570`. The same `bench_V_rel`, `bench_k_max_abs`, `bench_l_max_abs`, `info['iterations']`, `info['error']`, `info_fine['iterations']`, `info_fine['error']` variables that feed the README string at `run.py:366–375` are written verbatim to the CSV. All seven README fine-grid numbers are independently confirmed against the CSV in this audit:
  - bench_V_rel: CSV `0.00020686` → formatted `2.1e-04` = README value.
  - bench_k_max_abs: CSV `0.046149` → formatted `0.0461` = README value.
  - bench_l_max_abs: CSV `0.015007` → formatted `0.0150` = README value.
  - coarse_iterations: CSV `515.0` → README `515`. Match.
  - coarse_error: CSV `9.9546e-06` → formatted `9.95e-06` = README value.
  - fine_iterations: CSV `525.0` → README `525`. Match.
  - fine_error: CSV `9.9626e-06` → formatted `9.96e-06` = README value.
- No other patterns identified. All core algorithmic claims (EV broadcast, flow_utility shape, HP filter formula, investment formula, steady-state derivation) were already HOLDS in the original audit and remain so.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required.
1. The honest-fix test in `tests/test_rbc.py` (`test_finding1_honest_fix`) now passes: `tables/fine-grid-audit.csv` exists, contains all seven required metrics, both errors are below `1e-5`, and the coarse iteration count matches the README.
2. The violated-invariant test (`test_finding1_violated_invariant`) now correctly fails, confirming the fix was applied: `(FOLDER / "tables" / "fine-grid-audit.csv").exists()` is True.
3. No further action needed.
