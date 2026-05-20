# bullshit-detector — numerical-optimization — recheck — 2026-05-20

**Bullshit score: 0%** — both original findings are resolved: the Newton line search now enforces Armijo sufficient-decrease (`c1 = 1e-4` at `run.py:133`), and the DA success flag is conditioned on `reached_mode` at `run.py:231`; all equations, parameters, and numeric results hold against code and CSV.

## Header
- Claim sources: `computational-methods/numerical-optimization/README.md`
- Code / artifact root: `computational-methods/numerical-optimization/run.py`
- Data artifacts: `computational-methods/numerical-optimization/tables/optimizer-summary.csv`
- Seed audit (if any): `computational-methods/numerical-optimization/bullshit-detector_numerical-optimization_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Newton backtracking uses Armijo sufficient-decrease | HOLDS | — | — |
| 2 | DA success flag conditioned on reaching a mode | HOLDS | — | — |
| 3 | mu1=(1.5,1.5), mu2=(-1.5,-1.5), Sigma, omega=0.5 | HOLDS | — | — |
| 4 | f(theta) = -log p(theta) | HOLDS | — | — |
| 5 | Newton update theta_{n+1} = theta_n - H^{-1} grad f | HOLDS | — | — |
| 6 | Hessian regularized in Newton | HOLDS | — | — |
| 7 | BFGS updates inverse-Hessian from gradient changes | HOLDS | — | — |
| 8 | Nelder-Mead uses only objective values | HOLDS | — | — |
| 9 | DA uses stochastic uphill moves and local polishing | HOLDS | — | — |
| 10 | Newton/BFGS upper-right; Nelder-Mead/DA lower-left | HOLDS | — | — |
| 11 | All four rows: objective=2.38467, distance=0.01082 | HOLDS | — | — |
| 12 | Iterations: Newton=38, BFGS=7, Nelder-Mead=74, DA=80 | HOLDS | — | — |
| 13 | Best objective = 2.38467 | HOLDS | — | — |

## Findings

### Finding 1 (original): Newton backtracking = pure-decrease only — NOW HOLDS

- **Claim source (verbatim):** `"Newton: update with a regularized Hessian and backtracking line search"` — `README.md:61`
- **Code evidence (verbatim):**
  ```python
  current = objective(x)
  # Armijo sufficient-decrease constant: accept a step only when it
  # reduces the objective by at least c1 * alpha * (grad . step).
  c1 = 1e-4
  directional_decrease = float(np.dot(grad, step))
  alpha = 1.0
  accepted = False
  while alpha > 1e-5:
      candidate = x - alpha * step
      if objective(candidate) <= current - c1 * alpha * directional_decrease:
          x = candidate
          accepted = True
          break
      alpha *= 0.5
  ```
  `run.py:130-143`
- **Category:** HOLDS — the Armijo condition `f(x - alpha*d) <= f(x) - c1 * alpha * dot(grad, step)` is now enforced at `run.py:139`. The constant `c1 = 1e-4` (`run.py:133`) is the standard Armijo value from Nocedal and Wright (2006). The claim "backtracking line search" now accurately describes the implementation.
- **Original finding resolved:** yes.

### Finding 2 (original): DA success flag = scipy raw flag only — NOW HOLDS

- **Claim source (verbatim):** `"Dual annealing | box [-5,5]^2 | (-1.49, -1.49) | lower-left | 2.38467 | 0.01082 | 80 | True"` — `README.md:95`
- **Code evidence (verbatim):**
  ```python
  distance_to_mode = min(
      float(np.linalg.norm(result.x - mode)) for mode in (MU1, MU2)
  )
  reached_mode = distance_to_mode < 0.1
  assert reached_mode, (
      f"dual_annealing finished at {np.round(result.x, 3)} with distance "
      f"{distance_to_mode:.4f} to the nearest mode; it did not reach a "
      f"basin, so Success=True would be a stale flag."
  )

  return OptimizationRun(
      ...
      success=bool(result.success) and reached_mode,
  )
  ```
  `run.py:215-231`
- **Data evidence:** `tables/optimizer-summary.csv:5`: `Dual annealing,"box [-5,5]^2","(-1.49, -1.49)",lower-left,2.38467,1.082e-02,80,True`. The distance 1.082e-02 = 0.01082 < 0.1, so `reached_mode = True` at this run, and `success=True and True = True`. The reported Success=True is now a certified mode-reach verdict, not just scipy's termination flag.
- **Category:** HOLDS — the `reached_mode` guard at `run.py:218-223` is an assertion that fails loudly if the polished solution fails to reach a basin. The reported `success` at `run.py:231` is `bool(result.success) and reached_mode`, which certifies both a clean scipy run and actual mode proximity. The original DATA DRIFT (scipy's flag without mode check) is resolved.
- **Original finding resolved:** yes.

### Numeric cross-check (new this pass)

`tables/optimizer-summary.csv` vs `README.md:92-95`:

| Row | CSV objective | README objective | CSV iterations | README iterations | CSV success | README success |
|-----|--------------|------------------|----------------|-------------------|-------------|----------------|
| Newton | 2.38467 | 2.38467 | 38 | 38 | True | True |
| BFGS | 2.38467 | 2.38467 | 7 | 7 | True | True |
| Nelder-Mead | 2.38467 | 2.38467 | 74 | 74 | True | True |
| Dual annealing | 2.38467 | 2.38467 | 80 | 80 | True | True |

All distance values: CSV shows `1.082e-02`; README shows `0.01082`. These are equal. No drift.

## Cross-cutting patterns

- Both original findings were documentation and implementation gaps that did not change published numbers. The fixes added an Armijo constant check to the line search and a mode-proximity guard to the DA success flag. No result in the table changed.
- The `reached_mode` assertion at `run.py:219-223` is now a runtime guardrail: if a future seed or `maxiter` change causes DA to terminate without reaching a basin, the script will fail loudly rather than silently reporting `Success=True`. This is a strengthening of the artifact's internal consistency contract.
- No new gaps found. All 13 claims extracted in Pass 1 are HOLDS.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required. Both honest-fix tests now pass:
   - `test_f1_honest_fix_line_search_uses_armijo_sufficient_decrease` PASSES (`c1` and `sufficient` present in `newton_with_backtracking` source).
   - `test_f2_honest_fix_success_flag_certifies_mode_reached` PASSES (`reached_mode` and `success=bool(result.success) and reached_mode` present in `run_dual_annealing` source).
1. No further action needed for this tutorial.
