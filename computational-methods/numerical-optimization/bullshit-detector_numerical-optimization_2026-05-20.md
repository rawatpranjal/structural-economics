# bullshit-detector — numerical-optimization — 2026-05-20

**Bullshit score: 15%** — One DILUTED finding (backtracking line search claims Armijo-standard behaviour but implements pure-decrease only); one DATA DRIFT at LOW severity (dual annealing nit == maxiter with Success=True is ambiguous without re-run). No FALSE or UNIMPLEMENTED findings. Score anchored at mid-range of the MISLABELED/DATA DRIFT band (10–30%).

## Header
- Claim sources: `computational-methods/numerical-optimization/README.md`
- Code / artifact root: `computational-methods/numerical-optimization/run.py`
- Data artifacts: `computational-methods/numerical-optimization/tables/optimizer-summary.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Newton uses backtracking line search | DILUTED | LOW | no |
| 2 | Dual annealing Success=True with nit==maxiter | DATA DRIFT | LOW | no — needs re-run to verify |
| 3 | p(theta) is a Gaussian mixture | HOLDS | — | — |
| 4 | f(theta) = -log p(theta) | HOLDS | — | — |
| 5 | Newton update: theta_{n+1} = theta_n - H^{-1} grad f | HOLDS | — | — |
| 6 | mu1=(1.5,1.5), mu2=(-1.5,-1.5), Sigma=[[1,0.5],[0.5,1]], omega=0.5 | HOLDS | — | — |
| 7 | Start = (3.00, -2.50) for local methods | HOLDS | — | — |
| 8 | Newton/BFGS reach upper-right; Nelder-Mead/DA reach lower-left | HOLDS | — | — |
| 9 | Best objective = 2.38467 | HOLDS | — | — |
| 10 | Iteration counts: Newton=38, BFGS=7, Nelder-Mead=74, DA=80 | HOLDS | — | — |
| 11 | Distance to mode = 0.01082 for all four | HOLDS | — | — |
| 12 | DA uses stochastic uphill moves and local polishing | HOLDS | — | — |
| 13 | Hessian is regularized in Newton | HOLDS | — | — |
| 14 | BFGS updates inverse-Hessian approximation from gradient changes | HOLDS | — | — |
| 15 | Nelder-Mead uses only objective values | HOLDS | — | — |

## Findings

### Finding 1: Newton backtracking implements pure-decrease, not Armijo sufficient-decrease

- **Claim source (verbatim):** "Newton: update with a regularized Hessian and backtracking line search" — `README.md:61` (pseudocode line 1, also `run.py:367`)

- **Code evidence (verbatim):**
  ```python
  current = objective(x)
  alpha = 1.0
  accepted = False
  while alpha > 1e-5:
      candidate = x - alpha * step
      if objective(candidate) < current:
          x = candidate
          accepted = True
          break
      alpha *= 0.5

  if not accepted:
      # Fall back to a small gradient step when the Newton direction is bad.
      x = x - 0.05 * grad
  ```
  `run.py:130-143`

- **Data evidence (if applicable):** None — the iteration count (38) and final objective (2.38467) are consistent with convergence regardless of line-search variant; no result-changing delta is observable from the CSV alone.

- **Category:** DILUTED

- **Severity:** LOW

- **Result-changing:** no — the Newton optimizer converges to the same basin and the same objective value. The weaker acceptance criterion does not visibly change the published numbers, but it means the theoretical convergence guarantees of Armijo backtracking (guaranteed decrease per step bounded away from zero) do not apply.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not any("c1" in line or "armijo" in line.lower() or "grad" in line for line in inspect.getsource(newton_with_backtracking).splitlines() if "alpha" in line and "candidate" in line)
  # PASSES on current code (no Armijo check present); FAILS on honest fix that adds c1*alpha*dot(grad, step)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any("c1" in line or "sufficient" in line.lower() for line in inspect.getsource(newton_with_backtracking).splitlines())
  # PASSES on an honest fix adding the Armijo constant; FAILS on current code
  ```

---

### Finding 2: Dual annealing nit == maxiter with Success=True is ambiguous

- **Claim source (verbatim):** Table row "Dual annealing | box [-5,5]^2 | (-1.49, -1.49) | lower-left | 2.38467 | 0.01082 | 80 | True" — `README.md:95`; `tables/optimizer-summary.csv:5`

- **Code evidence (verbatim):**
  ```python
  result = dual_annealing(
      objective,
      bounds=[(-5.0, 5.0), (-5.0, 5.0)],
      maxiter=80,
      seed=seed,
      callback=callback,
      no_local_search=False,
  )
  ```
  `run.py:194-201`

  ```python
  iterations=int(result.nit),
  success=bool(result.success),
  ```
  `run.py:210-211`

- **Data evidence (if applicable):** CSV line 5: `Dual annealing,"box [-5,5]^2","(-1.49, -1.49)",lower-left,2.38467,1.082e-02,80,True`. The reported `nit=80` equals `maxiter=80`. Whether scipy's `result.success=True` is accurate when `nit==maxiter` cannot be determined without re-running; scipy `dual_annealing` may set `success=True` even on a maxiter exit if the termination criterion was satisfied. **Needs re-run to verify.**

- **Category:** DATA DRIFT

- **Severity:** LOW

- **Result-changing:** no — the final solution and objective are internally consistent with the other three runs; the ambiguity is limited to whether "Success=True" accurately reflects convergence vs maxiter-termination.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert result.nit == 80 and result.nit == dual_annealing_maxiter  # True on current run; ambiguous success flag
  # PASSES now; a genuine convergence run with looser maxiter would produce nit < maxiter
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert result.nit < maxiter or (result.nit == maxiter and result.message.lower().__contains__("converge"))
  # PASSES if either nit < maxiter OR scipy explicitly says converged at maxiter; FAILS if success is a stale flag
  ```

## Cross-cutting patterns

- All numeric claims in `README.md` (objective values, iteration counts, distances, basin labels) match `tables/optimizer-summary.csv` verbatim. The CSV and README are internally consistent.
- The tutorial's main pedagogical claims (basin sensitivity, equal objective across modes, restart-grid diagnostics) are all implemented and match the code.
- The only structural gap is in the Newton line search: the pseudocode uses the term "backtracking line search" which conventionally implies the Armijo sufficient-decrease condition (Nocedal and Wright, 2006, Algorithm 3.1), but the implementation uses only a pure-decrease check (`objective(candidate) < current`). This pattern could recur in any future tutorial that reuses `newton_with_backtracking` and cites Armijo guarantees.
- The dual-annealing `seed=609` in `main()` differs from the function default `seed=123` (`run.py:186`). The README makes no claim about seeds, so this is not a finding, but if the function is extracted and called with its default, results will differ from the committed CSV.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 15% (< 50%).** No halt required. Findings are LOW severity and do not change published results. Surface to the author as a touch-up item before citing Armijo guarantees in prose.

1. For Finding 1 (DILUTED): turn the violated invariant into a pytest test. Confirm it PASSES on current `run.py` (no Armijo check present). This proves the gap is real.

2. Convert the honest-fix pass condition into a second pytest test that FAILS on current code. The pair is the red/green spec for adding a proper Armijo check: `f(x - alpha*d) <= f(x) - c1 * alpha * np.dot(grad, step)` with `c1 = 1e-4`.

3. For Finding 2 (DATA DRIFT): add `assert result.nit < maxiter or "converge" in result.message.lower()` as a runtime assertion inside `run_dual_annealing`. Re-run to confirm scipy's `result.success` semantics at `maxiter=80`.

4. After fixes, re-run `python run.py` and confirm committed CSV numbers are unchanged (basin labels, objective, distances all identical). The fix to the line search may change iteration count for Newton; update CSV if so.

5. Re-run this skill on the updated code. Expected new bullshit score: 0–10%.
