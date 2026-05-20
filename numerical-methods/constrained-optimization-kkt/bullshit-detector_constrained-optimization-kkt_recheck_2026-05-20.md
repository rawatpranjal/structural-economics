# bullshit-detector — constrained-optimization-kkt — recheck — 2026-05-20

**Bullshit score: 0%** — All twenty claims hold against code and data; the three original findings (FALSE prose count, DATA DRIFT figure/table multiplier split, DILUTED pseudocode) are fully resolved; no new gaps found.

## Header
- Claim sources: `numerical-methods/constrained-optimization-kkt/README.md`
- Code / artifact root: `numerical-methods/constrained-optimization-kkt/run.py`
- Data artifacts: `tables/solution_comparison.csv`, `tables/kkt_check.csv`, `tables/shadow_prices.csv`
- Seed audit (if any): `bullshit-detector_constrained-optimization-kkt_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, independent recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Utility formula $u(x) = a^\top x - \tfrac{1}{2} x^\top B x$ | HOLDS | — | — |
| 2 | Calibration $a=(4,3,0.5)$, $B=I_3$, $I=3$ | HOLDS | — | — |
| 3 | Closed form $x^*=(2,1,0)$, $\lambda^*=2$, $\mu^*=(0,0,1.5)$ | HOLDS | — | — |
| 4 | $\mu_3^*=1.5$ is shadow price of project-3 non-negativity bound | HOLDS | — | — |
| 5 | Baseline gives $\lambda=1.5$, $x=(2.5,1.5,-1)$ | HOLDS | — | — |
| 6 | Duality gap along central path is exactly $n \cdot t$ | HOLDS | — | — |
| 7 | Projected gradient converges in 95 iterations | HOLDS | — | — |
| 8 | After 9 barrier values, iterate within 8.90e-09 of closed form | HOLDS | — | — |
| 9 | Figure complementarity uses exact $\mu_t = t/x_t$, equals $n\,t$ on central path | HOLDS | — | — |
| 10 | KKT table barrier row uses heuristic recovery; complementarity differs from $n\,t$ by factor $n$ | HOLDS | — | — |
| 11 | 9 barrier values to machine-precision feasibility (fixed: was "about a dozen") | HOLDS | — | — |
| 12 | SLSQP pseudocode names stationarity for binding-bound mu recovery (fixed: was "complementary slackness") | HOLDS | — | — |
| 13 | Baseline utility 9.25 exceeds feasible optimum 8.5 | HOLDS | — | — |
| 14 | Method 1 projected gradient: 95 iterations in table | HOLDS | — | — |
| 15 | Method 2 interior-point: 9 barrier values in table | HOLDS | — | — |
| 16 | Method 3 SLSQP: 2 iterations in table | HOLDS | — | — |
| 17 | All three methods reach allocation (2,1,0), utility 8.5 | HOLDS | — | — |
| 18 | KKT table barrier complementarity error = 1.00e-08 | HOLDS | — | — |
| 19 | Shadow prices: $\lambda^*=2$, $\mu_3^*=1.5$ | HOLDS | — | — |
| 20 | Table complementarity (1e-08) = $t$ = $n\cdot t/n$, i.e. off from $n\cdot t$ by factor $n$ | HOLDS | — | — |

## Findings

None. All claims hold.

## Cross-cutting patterns

- All three original findings were prose/pseudocode narrative mismatches, not numerical or algorithmic errors. The fix corrected only those narrative mismatches. The computations were correct throughout and remain correct.
- The "factor n" explanation (README.md:257) is now present, accurate, and internally consistent: the figure uses exact barrier multipliers ($\sum \mu_t |x_t| = n\cdot t = 3 \times 10^{-8}$), the table uses heuristic recovery ($= 1 \times 10^{-8} = t$). The ratio is exactly $n = 3$. Verified by fresh computation.
- The SLSQP pseudocode formula "mu_j = lambda - (a_j - (B x)_j)" at README.md:240 matches `run.py:79-80` (`mu[j] = max(0.0, lam - float(grad_neg[j]))` where `grad_neg = a - B @ x`) exactly, up to the `max(0, ...)` dual-feasibility clamp.
- The barrier count claim "9 barrier values" appears in README.md:251, README.md:257, and the solution_comparison.csv table — all three sources agree, and all agree with the `barriers` list at `run.py:137` (length 9, verified by fresh computation).

## TDD execution sequence (for the next agent)

0. **Bullshit score: 0%.** No halt required. No fixes needed. Ship.
1. All three honest-fix tests now pass (confirmed by `python -m pytest tests/ -q` in the tutorial directory).
2. All three violated-invariant tests now fail as expected (they are designed to fail on correctly fixed code).
3. No new TDD work needed. The tutorial is clean.
