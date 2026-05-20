# bullshit-detector — asset-pricing — recheck — 2026-05-20

**Bullshit score: 0%** — All structural, mathematical, and numerical claims HOLD. The original DATA DRIFT finding (convergence statistics not persisted to a committed artifact) is resolved: `tables/convergence.csv` now exists and all three values (405 iterations, 9.76e-10 residual, 0.011% max relative error) match the README prose exactly.

## Header
- Claim sources: `dynamic-programming/asset-pricing/README.md`
- Code / artifact root: `dynamic-programming/asset-pricing/run.py`
- Data artifacts: `dynamic-programming/asset-pricing/tables/convergence.csv`, `dynamic-programming/asset-pricing/tables/price-dividend-ratio.csv`
- Seed audit: `bullshit-detector_asset-pricing_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Endowment AR(1) with stationary sd = 0.2294 | HOLDS | none | no |
| 2 | CRRA u'(c) = c^{-gamma} | HOLDS | none | no |
| 3 | SDF M_{t+1} = beta*(y_{t+1}/y_t)^{-gamma} | HOLDS | none | no |
| 4 | Scaling f(y) = u'(y)*p(y) linearises the fixed point | HOLDS | none | no |
| 5 | Fixed-point recursion f(y)=beta*E[f(y')+u'(y')y' \| y] | HOLDS | none | no |
| 6 | Price recovery p(y) = f(y)/u'(y) = f*y^gamma | HOLDS | none | no |
| 7 | T operator is a beta-contraction | HOLDS | none | no |
| 8 | Log-utility p/y = beta/(1-beta) approx 19.0 (flat) | HOLDS | none | no |
| 9 | Gauss-Hermite quadrature correctly transforms to N(0,sigma^2) | HOLDS | none | no |
| 10 | Pseudocode d_{ij} = (y')^{1-gamma} matches dividend_term in code | HOLDS | none | no |
| 11 | gamma<1 p/y falls with y; gamma>1 rises; gamma=1 flat | HOLDS | none | no |
| 12 | CSV table matches README table (7 rows x 5 columns) | HOLDS | none | no |
| 13 | Convergence: 405 iters, residual 9.76e-10, max rel error 0.011% | HOLDS | none | no |

## Findings

### Finding 1 (original): convergence statistics persisted to CSV — RESOLVED

- **Original claim source (verbatim):** "The baseline $\gamma=2.0$ solution converges in **405 iterations** to sup-norm residual **9.76e-10**. On the central $\pm 3\,\mathrm{sd}(\log y)$ region, the maximum relative error is **0.011%**." — `README.md:109`

- **Original finding:** DATA DRIFT — numbers embedded in README via f-strings only; no committed artifact held them.

- **Resolution:** `tables/convergence.csv` now exists (`run.py:529-553`). It carries three rows:

  ```
  Quantity,Value
  Baseline iterations,405
  Baseline sup-norm residual,9.76e-10
  Central max relative error (%),0.011
  ```

  The README values (405, 9.76e-10, 0.011%) match the CSV values exactly. The README also contains a table that renders these values inline (`README.md:149-153`). The data pipeline from `solution.iterations` and `solution.error` → `convergence_df` → `tables/convergence.csv` → README is self-consistent.

- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

## Structural and mathematical claims — all HOLDS

Pass 2 and Pass 3 grounding for all remaining claims:

1. **Stationary sd = 0.2294.** `run.py:129`: `stat_std = sigma / np.sqrt(1.0 - rho**2)` = 0.10 / sqrt(1-0.81) = 0.10/sqrt(0.19) = 0.2294. HOLDS.

2. **CRRA marginal utility.** `run.py:35`: `return c ** (-gamma)`. Matches `u'(c)=c^{-gamma}`. HOLDS.

3. **SDF.** README states $M_{t+1}=\beta(y_{t+1}/y_t)^{-\gamma}$. This is the standard SDF for CRRA in an exchange economy with $c_t=y_t$. The code never evaluates the SDF directly — it substitutes it into the Euler equation to produce the fixed-point recursion, which is algebraically equivalent. HOLDS (by derivation).

4. **Scaled price fixed point.** `run.py:62`: `dividend_term = crra_marginal_utility(y_next, gamma) * y_next` = $u'(y')y'$. `run.py:76`: `f_new = beta * np.sum((continuation + dividend_term) * weights[None, :], axis=1)` = $\beta \mathbb{E}[f(y')+u'(y')y' | y]$. HOLDS.

5. **Price recovery.** `run.py:82`: `price = f / crra_marginal_utility(y_grid, gamma)` = $f / y^{-\gamma} = f \cdot y^\gamma = p(y)$. HOLDS.

6. **Pseudocode d_{ij}.** Pseudocode (`README.md:97`): `d_{ij} <- (y'_{ij})^{1 - gamma}`. Code (`run.py:62`): `crra_marginal_utility(y_next, gamma) * y_next` = $y'^{-\gamma} \cdot y' = y'^{1-\gamma}$. Identical. HOLDS.

7. **Log-utility benchmark.** `run.py:205`: `pd_log_utility = beta / (1.0 - beta)` = 0.95/0.05 = 19.0. README: "$\beta/(1-\beta)\approx 19.0$". HOLDS.

8. **CSV table vs README table.** `tables/price-dividend-ratio.csv` contains 7 rows × 5 columns. README table (`README.md:135-143`) matches CSV exactly, row by row. Verified by direct comparison of all 35 cells. HOLDS.

9. **Gamma comparative statics.** README: "When $\gamma<1$, the ratio falls with current dividends. When $\gamma>1$, it rises." Code solves for gamma in [0.5, 1.0, 2.0, 5.0] (`run.py:165-188`). CSV confirms: at y=0.504, p/y gamma=0.5 is 24.302 > 19.0 (log), and falls to 15.157 at y=1.983; at gamma=2, p/y=12.334 at y=0.504 rises to 31.724 at y=1.983. Both comparative statics hold. HOLDS.

## Cross-cutting patterns

- The fix followed the TDD sequence exactly: `tables/convergence.csv` was added to `run.py` (lines 529-553) and the values were embedded into the report table. The CSV is now a grounded artifact that future audits can cross-check without re-running.
- No new claims were introduced that are not grounded. The README description of the convergence table (`README.md:145`) is honest: "The iteration count, sup-norm residual, and central relative error are persisted here so the convergence claims in the Solution Method section can be cross-checked against a committed artifact."
- All numeric quantities in the README are now either (a) fixed parameters in `run.py:115-127`, (b) closed-form derivations from those parameters, or (c) live-computed values persisted to a committed CSV. No floating prose numbers remain.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No further action required. All findings from the original audit are resolved. The tutorial is ready to ship.
1. The violated-invariant test (`test_finding1_violated_invariant_no_convergence_csv`) correctly fails on the current repo state, confirming the fix was applied.
2. Both honest-fix tests (`test_finding1_honest_fix_convergence_csv_exists`, `test_finding1_readme_iterations_match_csv`) pass, confirming correctness of the new artifact.
3. No additional findings identified in this recheck. No further TDD cycles needed.
