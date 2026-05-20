# bullshit-detector -- smolyak-sparse-grids -- recheck -- 2026-05-20

**Bullshit score: 0%** -- All four prior findings verified fixed. All numeric, algorithmic, grid-count, and Euler-error claims HOLD against code and both CSV artifacts.

## Header
- Claim sources: `computational-methods/smolyak-sparse-grids/README.md`
- Code / artifact root: `computational-methods/smolyak-sparse-grids/run.py`
- Data artifacts: `computational-methods/smolyak-sparse-grids/tables/accuracy.csv`, `computational-methods/smolyak-sparse-grids/tables/grid-counts.csv`
- Seed audit: `computational-methods/smolyak-sparse-grids/bullshit-detector_smolyak-sparse-grids_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "worst-case relative Euler error" (prior F1, MISLABELED "absolute" fixed) | HOLDS | - | - |
| 2 | A^(1/(1-alpha)) approx (1.000, 0.848, 1.161, 0.706) (prior F2, corrected) | HOLDS | - | - |
| 3 | Z approx 3.714 (prior F3, corrected from 3.710) | HOLDS | - | - |
| 4 | omega approx (0.269, 0.228, 0.312, 0.190) (prior F4, corrected from 0.313) | HOLDS | - | - |
| 5 | Residual equation (Y-S)*S^(alpha-1) = 1/(beta*alpha*Z^(1-alpha)*E_n) | HOLDS | - | - |
| 6 | Scalar Euler solved by brentq at each node | HOLDS | - | - |
| 7 | Smolyak node counts H(d,mu) match grid-counts.csv | HOLDS | - | - |
| 8 | Basis count = node count (square Phi) | HOLDS | - | - |
| 9 | Closed-form policy S* = alpha*beta*Y | HOLDS | - | - |
| 10 | Tensor node formula (2^mu+1)^d used in comparison table | HOLDS | - | - |
| 11 | Admissible level indices: mu+1 <= sum(i) <= mu+d, i_k >= 1 | HOLDS | - | - |
| 12 | Gauss-Hermite quadrature normalized for eps ~ N(0,1) | HOLDS | - | - |
| 13 | degree_level function matches Judd et al. convention | HOLDS | - | - |
| 14 | Accuracy table values (mu=1,2,3) match accuracy.csv | HOLDS | - | - |

## Findings

None.

**Prior F1 resolved (MISLABELED: "absolute" Euler error).** `euler_errors_at` at `run.py:347-349` computes `np.abs(lhs / rhs - 1.0)`, a dimensionless relative metric. `README.md:287` now reads "worst-case relative Euler error." `run.py:477` (figure x-axis) reads "log10 relative Euler error." `run.py:905` (Results prose) reads "worst-case relative Euler error." All three sites corrected.

**Prior F2 resolved (DATA DRIFT: A^exp worked values).** `run.py:632` now reads `A^{1/(1-\alpha)} \approx (1.000, 0.848, 1.161, 0.706)`. Verification: `0.9^(1/0.64) = 0.84821`, rounds to 0.848. `0.8^(1/0.64) = 0.70563`, rounds to 0.706. Both correct.

**Prior F3 resolved (DATA DRIFT: Z approx 3.710 -> 3.714).** `run.py:634` now reads `Z \approx 3.714`. Verification: sum = 1.000 + 0.848 + 1.161 + 0.706 = 3.715 (at 3dp) but exact Z_const() = 3.71443, which rounds to 3.714. Correct.

**Prior F4 resolved (DATA DRIFT: omega[2] = 0.313 -> 0.312).** `run.py:636` now reads `\omega \approx (0.269, 0.228, 0.312, 0.190)`. Verification: exact omega[2] = 0.31245, rounds to 0.312. Consistent with Model Setup table at `README.md:180`. Both sites now agree.

## Cross-cutting patterns

- All three DATA DRIFT findings (F2, F3, F4) originated in the same Equations worked-example block (`run.py:632-636`). All corrected in one pass.
- The MISLABELED finding (F1) required changing three sites (README prose, run.py figure x-axis label, run.py Results prose). All three corrected.
- No DATA DRIFT between accuracy.csv and README accuracy table: mu=1 max Euler 1.20e-02 = 0.012, mu=2 1.04e-03 = 0.00104, mu=3 1.65e-04 = 0.000165. All match.
- Grid-counts.csv d=5 mu=2 Smolyak=61 nodes matches `smolyak_count(5,2)`. d=5 mu=3 Smolyak=241 nodes matches the 241 cited in README:287 and accuracy table.
- The Euler error function computes `|lhs/rhs - 1|` at `run.py:349`. The column header in accuracy.csv is `Max Euler error` (neutral, no "absolute"). Consistent with the corrected prose.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%. No action required.**
1. All 8 tests at `tests/test_smolyak-sparse-grids.py` pass: 4 violated-invariant tests (confirming original bugs were real) + 4 honest-fix tests (confirming fixes hold). Test suite is the correct green state.
2. No sim re-runs or data artifact changes needed. The corrected values are all in the Equations prose block of `run.py`; no CSV content was affected by any of the four fixes.
