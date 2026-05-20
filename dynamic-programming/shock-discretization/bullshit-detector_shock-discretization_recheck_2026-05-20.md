# bullshit-detector — shock-discretization — recheck — 2026-05-20

**Bullshit score: 0%** — Original DILUTED finding (pseudocode said "interior rows" while lib normalizes all rows) is fully remediated. All 15 claims hold against code, lib, and CSV.

## Header

- Claim sources: `dynamic-programming/shock-discretization/README.md`
- Code / artifact root: `dynamic-programming/shock-discretization/run.py`, `lib/discretize.py`
- Data artifacts: `dynamic-programming/shock-discretization/tables/moment-comparison.csv`
- Seed audit: `bullshit-detector_shock-discretization_2026-05-20.md` (1 non-HOLDS finding: DILUTED LOW)
- Run by: bullshit-detector skill (claude-sonnet-4-6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | sigma_z = 0.0641 | HOLDS | — | — |
| 2 | half-life approx 14 periods | HOLDS | — | — |
| 3 | Tauchen pseudocode: midpoint CDF differences | HOLDS | — | — |
| 4 | Rouwenhorst base uses p=(1+rho)/2 | HOLDS | — | — |
| 5 | Rouwenhorst grid formula matches lib | HOLDS | — | — |
| 6 | Rouwenhorst invariant distribution is binomial | HOLDS | — | — |
| 7 | Pseudocode says "normalize all rows of P_n" | HOLDS | — | — |
| 8 | lib normalizes all rows via row-sum division | HOLDS | — | — |
| 9 | Rouwenhorst zero error at all N | HOLDS | — | — |
| 10 | Tauchen N=7 persistence = 0.9622 | HOLDS | — | — |
| 11 | Tauchen N=3 near-absorbing (persistence near one) | HOLDS | — | — |
| 12 | Finite chains use common random numbers | HOLDS | — | — |
| 13 | Invariant pi satisfies pi = pi P | HOLDS | — | — |
| 14 | Tauchen extra tail mass creates positive variance error | HOLDS | — | — |
| 15 | README table numbers match moment-comparison.csv | HOLDS | — | — |

## Findings

None.

## Cross-cutting patterns

- Original Finding 1 (DILUTED, LOW): pseudocode previously said "row-normalize interior rows of P_n". The fix updated `run.py:255` to read: `"      normalize all rows of P_n              (interior rows summed two\n"`. The README now correctly states "normalize all rows of P_n" with an inline note "(interior rows summed two contributions and need scaling; endpoint rows already sum to 1)". This matches `lib/discretize.py:41` which applies `trans / trans.sum(axis=1, keepdims=True)` unconditionally to all rows.
- The numerical fact underpinning the fix holds: endpoint rows of the pre-normalization Rouwenhorst matrix sum to exactly 1.0 for all N >= 3, so normalizing all rows and normalizing only interior rows produce identical transition matrices to floating-point precision. The fix makes the pseudocode accurate without changing the library or the results.
- All numeric claims (sigma_z, half-life, all 10 table rows, Tauchen N=7 persistence 0.9622) remain confirmed against CSV and code.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required.
1. The violated-invariant test `test_finding1_violated_invariant_pseudocode_says_interior` now correctly fails (the string "row-normalize interior rows of P_n" no longer appears in `run.py`).
2. The honest-fix tests `test_finding1_honest_fix_pseudocode_normalizes_all_rows` and `test_finding1_honest_fix_readme_matches_lib` now pass: `run.py` contains "normalize all rows of P_n" and `README.md` does not contain "interior rows of P_n".
3. The numeric invariant test `test_finding1_violated_invariant_endpoint_rows_already_sum_to_one` still passes, confirming the mathematical basis for the equivalence.
4. No further action needed.
