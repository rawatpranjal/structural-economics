# bullshit-detector — solow-growth — recheck — 2026-05-20

**Bullshit score: 0%** — Original DILUTED finding (geometric residual 2.21e-04 conflated with actual terminal gap 2.73e-04) is fully remediated. Prose now explicitly names both quantities as distinct. All 13 claims hold.

## Header

- Claim sources: `dynamic-programming/solow-growth/README.md`
- Code / artifact root: `dynamic-programming/solow-growth/run.py`
- Data artifacts: `dynamic-programming/solow-growth/tables/steady-state-comparison.csv`
- Seed audit: `bullshit-detector_solow-growth_2026-05-20.md` (1 non-HOLDS finding: DILUTED MED)
- Run by: bullshit-detector skill (claude-sonnet-4-6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Delta = 0.0902 | HOLDS | — | — |
| 2 | k* = 4.3086 | HOLDS | — | — |
| 3 | lambda ≈ 0.941 | HOLDS | — | — |
| 4 | half-life ≈ 11.5 periods | HOLDS | — | — |
| 5 | terminal gap 2.73e-04 at period 159 | HOLDS | — | — |
| 6 | table values match steady-state-comparison.csv | HOLDS | — | — |
| 7 | comparative statics k* in {2.80, 4.31, 6.01} | HOLDS | — | — |
| 8 | linear approx predicts 2.21e-04; actual gap is 2.73e-04 | HOLDS | — | — |
| 9 | both quantities named distinctly in prose | HOLDS | — | — |
| 10 | no Bellman equation | HOLDS | — | — |
| 11 | c_t = (1-s)*y_t | HOLDS | — | — |
| 12 | phi(k) law of motion matches code | HOLDS | — | — |
| 13 | c* = (1-s)*y* | HOLDS | — | — |

## Findings

None.

## Cross-cutting patterns

- Original Finding 1 (DILUTED, MED): The buggy prose read "The geometric residual is about 2.21e-04", presenting the linearization's prediction as if it explained the actual gap (2.73e-04). The fix rewrites `run.py:412–418` to: "The linear approximation predicts a remaining gap of about 2.21e-04; the actual gap is 2.73e-04, larger because k_0=1.0 starts far from k* in the nonlinear region where the linearization underestimates the true distance." Both numbers are explicitly named as distinct quantities with a causal explanation. README.md:114 reflects this corrected prose.
- Both numbers are confirmed: linear approximation `|1.0-4.308592|*0.941^159 ≈ 2.21e-04` computed dynamically at `run.py:415`; actual gap `2.73e-04` at `run.py:416` and confirmed in CSV row 1.
- No other patterns identified. All arithmetic claims (Delta, k*, lambda, half-life, comparative statics, table entries) are derived deterministically from parameters and confirmed against the CSV artifact.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required.
1. The violated-invariant test `test_finding1_violated_invariant_prose_presents_residual_as_the_gap` now correctly fails: the string "The geometric residual is about" no longer appears in `run.py`.
2. The honest-fix tests `test_finding1_honest_fix_run_py_distinguishes_prediction_from_gap` and `test_finding1_honest_fix_readme_shows_both_numbers_distinctly` now pass: `run.py` contains "linear approximation" and "actual gap"; `README.md` contains both "2.21e-04" and "2.73e-04" and "linear approximation".
3. The numeric invariant test `test_finding1_violated_invariant_geometric_residual_underestimates_gap` still passes, confirming the mathematical basis of the two-number distinction.
4. No further action needed.
