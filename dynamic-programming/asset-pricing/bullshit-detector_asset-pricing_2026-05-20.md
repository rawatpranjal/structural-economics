# bullshit-detector — asset-pricing — 2026-05-20

**Bullshit score: 10%** — All structural and mathematical claims HOLD; two numeric claims (iteration count 405, residual 9.76e-10, max relative error 0.011%) are live-computed via f-strings and cannot be grounded without a re-run; one minor DATA DRIFT finding.

## Header
- Claim sources: `dynamic-programming/asset-pricing/README.md` (Equations, Model Setup, Solution Method, Results sections)
- Code / artifact root: `dynamic-programming/asset-pricing/run.py`
- Data artifact: `dynamic-programming/asset-pricing/tables/price-dividend-ratio.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
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
| 13 | Convergence: 405 iterations, residual 9.76e-10, max rel error 0.011% | DATA DRIFT | LOW | no (consistent if re-run unchanged) |

## Findings

### Finding 1: Convergence statistics are live-computed and cannot be grounded from committed artifacts

- **Claim source (verbatim):** "The baseline $\gamma=2.0$ solution converges in **405 iterations** to sup-norm residual **9.76e-10**. On the central $\pm 3\,\mathrm{sd}(\log y)$ region, the maximum relative error is **0.011%**." — `README.md:109`

- **Code evidence (verbatim):**
  ```python
  f"The baseline $\\gamma={gamma}$ solution converges in "
  f"**{solution.iterations} iterations** to sup-norm residual "
  f"**{solution.error:.2e}**. On the central "
  f"$\\pm 3\\,\\mathrm{{sd}}(\\log y)$ region, the maximum relative error is "
  f"**{max_relative_error_pct:.3f}%**."
  ```
  `run.py:341-345`

- **Data evidence (if applicable):** None. The numbers 405, 9.76e-10, and 0.011% appear only in the committed `README.md`. There is no `tables/convergence.csv` or stdout log in the repo. The CSV (`tables/price-dividend-ratio.csv`) contains only price-dividend values, not solver diagnostics.

- **Category:** DATA DRIFT — the committed `README.md` numbers are the output of a past run; if `run.py` or any parameter changed between that run and now, the reported numbers could diverge from what the current code produces. The solver logic has not changed since at least commit `b578ef8` (verified via `git diff`), so the drift risk is low but non-zero. **needs re-run to verify**

- **Severity:** LOW — the numbers are self-consistent (9.76e-10 < tol=1e-9 holds; 0.011% < 1% is a plausible fine-grid error for 120 nodes), and the code that produced them is structurally unchanged. No result-changing consequence identified.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "405" in open("dynamic-programming/asset-pricing/README.md").read()  # PASSES now; would FAIL if re-run produces different count
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert solution.iterations == int(open("dynamic-programming/asset-pricing/README.md").read().split("converges in **")[1].split(" iterations")[0])  # PASSES after re-run that refreshes README
  ```

## Cross-cutting patterns

- The tutorial uses f-strings to embed all live-computed convergence statistics directly into the report prose (run.py:341-345). This is correct practice for auto-generated READMEs — the numbers were accurate at last generation time. The only risk is README staleness if code is edited without re-running. No pattern of static hardcoding found.
- All table numbers in README are bit-for-bit identical to `tables/price-dividend-ratio.csv` (verified row-by-row). The data pipeline from solver output to report to CSV is consistent.
- All mathematical claims (SDF, fixed-point structure, contraction property, log-utility benchmark, comparative statics direction) are algebraically correct and grounded in code. No gap found between the stated equations and what `solve_price_function` executes.
- The pseudocode in Solution Method matches the vectorised NumPy implementation exactly: `d_ij = (y')^{1-gamma}` corresponds to `dividend_term = crra_marginal_utility(y_next, gamma) * y_next` (line 62); `f_new = beta * sum_j w_j * (f_hat + d)` corresponds to line 76.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Below the 25% threshold — no halt required. Proceed with any planned work.
1. Finding 1 (DATA DRIFT, LOW): no test needed beyond a re-run check. To confirm the committed numbers are still accurate, run `python run.py` inside `dynamic-programming/asset-pricing/` and verify the printed iteration count and error match the README values. No code fix required.
2. Optional hardening: add a `tables/convergence.csv` emitted by `run.py` containing `iterations`, `error`, `max_relative_error_pct` columns. This makes the convergence statistics a grounded data artifact that any future audit can cross-check without a re-run.
3. No other non-HOLDS findings. No TDD red/green cycle needed.
