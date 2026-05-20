# bullshit-detector — bayesian-optimization-recheck — 2026-05-20

**Bullshit score: 10%** — all three original non-HOLDS findings are now resolved; one residual generic "orders of magnitude" phrase in the Solution Method section is a qualitative theoretical statement about a hypothetical expensive-evaluation regime, not a quantified comparison against any listed baseline, and does not misrepresent a result.

## Header

- Claim sources: `numerical-methods/bayesian-optimization/README.md`
- Code / artifact root: `numerical-methods/bayesian-optimization/run.py`
- Data artifacts: `numerical-methods/bayesian-optimization/tables/method_comparison.csv`, `numerical-methods/bayesian-optimization/tables/bo_iteration_log.csv`
- Seed audit: `numerical-methods/bayesian-optimization/bullshit-detector_bayesian-optimization_2026-05-20.md`
- Run by: bullshit-detector agent (Claude Sonnet 4.6), independent recheck
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "constant-mean prior" with $m(x) \equiv \bar{y}$ in Equations | HOLDS | - | - |
| 2 | Posterior mean formula uses $(y - m(X))$ centering | HOLDS | - | - |
| 3 | "roughly 34x smaller than SA, 17x RS, 10x multi-start" | HOLDS | - | - |
| 4 | "wins by orders of magnitude" (Solution Method, hypothetical regime) | HOLDS (generic qualitative; no quantified baseline) | - | - |
| 5 | All Model Setup parameter values | HOLDS | - | - |
| 6 | Peak values ($p_L^{\ast}=1.6029$, $\pi=4.1360$; $p_H^{\ast}=4.25$, $\pi=5.625$) | HOLDS | - | - |
| 7 | Method comparison table numbers (BO=30, MS=312, RS=500, SA=1007) | HOLDS | - | - |
| 8 | Evaluations to global (BO=12, MS=5, RS=140, SA=6) | HOLDS | - | - |
| 9 | EI closed-form formula and worked example arithmetic | HOLDS | - | - |
| 10 | GP posterior variance formula | HOLDS | - | - |
| 11 | Pseudocode in Solution Method | HOLDS | - | - |

## Findings

### Finding 1 (ORIGINAL, NOW RESOLVED): "zero-mean prior" — CLOSED

- **Original claim:** "We use a zero-mean prior, $f \sim \mathcal{GP}(0, k)$" — `README.md:38` (original audit)
- **Current state:** `run.py:350` now reads: `"We use a constant-mean prior, $f \sim \mathcal{GP}(m, k)$ with $m(x) \equiv \bar{y}$ fixed to the sample mean of the observed targets"`. The phrase "zero-mean" does not appear anywhere in `run.py`.
- **Formula check:** `run.py:361` emits `$$\mu(x_{\ast}) = m(x_{\ast}) + \ldots [K(X,X) + \sigma_n^2 I]^{-1} (y - m(X))$$`, which matches the code: `run.py:71-72` computes `self.y_mean = float(self.y.mean())` and `y_centered = self.y - self.y_mean`; `run.py:82` returns `mu = self.y_mean + K_s.T @ self._alpha`. Equations, pseudocode, and code are now internally consistent.
- **Category:** HOLDS
- **Result-changing:** no

### Finding 2 (ORIGINAL, NOW RESOLVED): "two orders of magnitude" — CLOSED

- **Original claim:** "two orders of magnitude smaller than simulated annealing" — `README.md:180` (original audit)
- **Current state:** That phrase does not appear in `run.py` or the current `README.md`. `run.py:634-641` now computes the ratios dynamically: `sa_ratio = len(sa_curve) / n_total` and emits `f"roughly {sa_ratio:.0f}x smaller than simulated annealing"`. With `len(sa_curve) = 1007` and `n_total = 30`, `sa_ratio = 33.57`, which rounds to `34x`. The current `README.md:180` reads "roughly 34x smaller than simulated annealing, 17x smaller than random search, and 10x smaller than multi-start."
- **Data check:** `tables/method_comparison.csv`: BO=30, SA=1007. Ratio 1007/30 = 33.57x. README rounds to 34x. Correct.
- **Category:** HOLDS
- **Result-changing:** no

### Finding 3 (ORIGINAL, NOW RESOLVED): "one order smaller than random search" — CLOSED

- **Original claim:** "one order smaller than random search or multi-start" — `README.md:180` (original audit)
- **Current state:** That phrase does not appear in `run.py` or the current `README.md`. `run.py:635` computes `rs_ratio = n_random / n_total = 500 / 30 = 16.67`, rounds to 17x; emits "17x smaller than random search". README:180 confirms "17x smaller than random search, and 10x smaller than multi-start."
- **Data check:** RS/BO = 500/30 = 16.67x; README says 17x. MS/BO = 312/30 = 10.4x; README says 10x. Both correct against CSV.
- **Category:** HOLDS
- **Result-changing:** no

### Finding 4 (NEW, HOLDS): "wins by orders of magnitude" — generic qualitative statement

- **Claim source (verbatim):** "When evaluations are expensive, the inner-loop cost is dominated by a single objective call and Bayesian optimization wins by orders of magnitude." — `README.md:164` / `run.py:487-488`
- **Context:** This sentence appears in the Solution Method section in a paragraph about general regimes ("when evaluations are cheap ... when evaluations are expensive"), not in Results and not in a comparison of the four specific methods in the table. It makes no quantified claim about the SA/BO or RS/BO ratios in this tutorial.
- **Code evidence:** `run.py:485-488`:
  ```python
  "Bayesian optimization is not magic. "
  "It pays for sample efficiency with stronger assumptions on the objective and with model fitting in the inner loop. "
  "When evaluations are cheap, simulated annealing or multi-start L-BFGS-B is faster end to end. "
  "When evaluations are expensive, the inner-loop cost is dominated by a single objective call and Bayesian optimization wins by orders of magnitude."
  ```
- **Assessment:** The phrase "orders of magnitude" here is a generic theoretical statement about a class of problems, not a claim about the 30-vs-1007 comparison in this tutorial. It is qualitatively accurate: on problems where each evaluation costs minutes, a 30x vs 1007x evaluation advantage translates directly into wall-clock orders-of-magnitude savings. The sentence makes no falsifiable quantitative claim that the data can contradict.
- **Category:** HOLDS
- **Severity:** none

### Additional verification: EI worked example arithmetic

- **Claim:** $z = (5.2 - 4.5)/0.5 = 1.4$, $\mathrm{EI} \approx 0.7 \cdot 0.919 + 0.5 \cdot 0.150 \approx 0.72$ — `README.md:79-80`
- **Computed:** $z = 1.4000$; $\Phi(1.4) = 0.919$; $\phi(1.4) = 0.150$; $\mathrm{EI} = 0.7183$. README says $\approx 0.72$. Correct.
- **Category:** HOLDS

### Additional verification: Model Setup parameters

- **Peak values:** Code computes `p_low_peak = 10.9/6.8 = 1.6029`, `profit_low_peak = 4.1360`, `p_high_peak = 4.25`, `profit_high_peak = 5.6250`. README Model Setup table (`run.py:410-411`) emits these from the same variables. CSV rows confirm BO optimum = 4.2505, profit = 5.6250.
- **Category:** HOLDS

## Cross-cutting patterns

- All three original non-HOLDS findings (MISLABELED zero-mean prior, FALSE "two orders", DILUTED "one order") are resolved. The fix in each case was: (a) replace "zero-mean" with "constant-mean" and update the posterior formula to show $m(x)$ explicitly, (b-c) replace hardcoded "orders of magnitude" prose with dynamically computed `{sa_ratio:.0f}x` and `{rs_ratio:.0f}x` expressions that read directly from the same variables that fill the table. The dynamic computation ensures prose and table remain consistent under any future re-run.
- The residual generic "orders of magnitude" phrase in Solution Method (Finding 4) is not a data-backed claim and is not falsifiable from the CSV; it is a general pedagogical assertion. It survives the hostile-reader test because it applies to the class of problems the tutorial is about, not to the specific table numbers.
- All numeric results in `method_comparison.csv` and `bo_iteration_log.csv` match README prose and are internally consistent.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Below the 25% ship threshold. No halt required.
1. All honest-fix tests in `tests/test_run.py` PASS on current code (4 of 4): `test_f1_code_is_constant_mean_gp`, `test_f1_honest_fix_equations_describe_constant_mean`, `test_f2_honest_fix_ratio_is_about_1p5_orders`, `test_f3_honest_fix_random_search_ratio_above_one_order`.
2. All claim-as-invariant tests FAIL on current code as designed (3 of 3): the buggy phrases no longer exist in `run.py`, confirming the fixes landed.
3. No new findings require TDD action.
4. Optional: the `test_f1_honest_fix_equations_describe_constant_mean` test asserts `"y - m(X)" in source`. This passes because the posterior-mean formula string in `run.py:361` contains `(y - m(X))`. If a future edit restructures the formula notation, this test will catch any regression back to the zero-mean form.
