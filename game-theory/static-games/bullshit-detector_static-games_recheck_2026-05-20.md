# bullshit-detector — static-games — recheck — 2026-05-20

**Bullshit score: 0%** — original Finding 1 (BR equation omitted non-negativity clip) is resolved; README now displays `\max\lbrace 0,\ \tfrac{a-c-bq_j}{2b}\rbrace` at line 29-31. All other claims verified HOLDS against code and CSV.

## Header
- Claim sources: `game-theory/static-games/README.md`, `game-theory/static-games/run.py`
- Code / artifact root: `game-theory/static-games/run.py`
- Data artifacts: `game-theory/static-games/tables/convergence-summary.csv`
- Seed audit: `bullshit-detector_static-games_2026-05-20.md` (one DILUTED/LOW finding)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | BR equation includes non-negativity: `\max{0, (a-c-bq_j)/(2b)}` | HOLDS | — | — |
| 2 | q* = 2.667 | HOLDS | — | — |
| 3 | P* = 4.667 | HOLDS | — | — |
| 4 | pi* = 7.111 | HOLDS | — | — |
| 5 | Convergence table residuals match CSV | HOLDS | — | — |
| 6 | Damped update q_{t+1} = (1-lambda)q_t + lambda BR(q_t) | HOLDS | — | — |
| 7 | Simultaneous best responses | HOLDS | — | — |
| 8 | Closed-form row uses exact q* | HOLDS | — | — |
| 9 | Figure axes consistent with BR curve labeling | HOLDS | — | — |

## Findings

### Finding 1 (original DILUTED/LOW): RESOLVED

- **Original claim (verbatim):** "The interior first-order condition gives the best response $BR_i(q_j)=\frac{a-c-bq_j}{2b}.$" — original `README.md`
- **Current README evidence (verbatim):** "The interior first-order condition gives $(a-c-bq_j)/(2b)$, and clipping at the non-negativity constraint $q_i \geq 0$ gives the best response $$BR_i(q_j)=\max\lbrace 0,\ \tfrac{a-c-bq_j}{2b} \rbrace.$$" — `README.md:26-31`
- **Code evidence (verbatim):**
  ```python
  return np.maximum(0.0, (a - c - b * q_other) / (2.0 * b))
  ```
  `run.py:24`
- **Category:** HOLDS — README equation now matches the code's `np.maximum(0.0, ...)` call exactly.
- **Severity:** — (resolved)
- **Result-changing:** no

### HOLDS block (all claims verified)

**Finding 2: q* = 2.667**
- README Model Setup: `$q^{*}$ | 2.667` — `README.md:48`. Code: `q_star = (a - c) / (3.0 * b)` = 8/3 = 2.6667 — `run.py:87`. CSV: `2.6667`. HOLDS.

**Finding 3: P* = 4.667**
- README: `$P^{*}$ | 4.667` — `README.md:49`. Code: `p_star = price(2.0 * q_star, a, b)` = 10 - 1*(2*2.6667) = 4.6667 — `run.py:88`. HOLDS.

**Finding 4: pi* = 7.111**
- README: `$\pi^{*}$ | 7.111` — `README.md:50`. Code: `profit(q_star, q_star, a, b, c)` = (4.6667-2)*2.6667 = 7.1111 — `run.py:89`. HOLDS.

**Finding 5: Convergence table residuals**
- README table — `README.md:84-88`; `tables/convergence-summary.csv` rows 2-5 match exactly. Residuals 5.60e-06 and 4.44e-16 confirmed. HOLDS.

**Finding 6: Damped update formula**
- README pseudocode step 3: `q_{t+1} = (1-lambda) q_t + lambda BR(q_t)` — `README.md:64`. Code: `path[t + 1] = (1.0 - damping) * path[t] + damping * target` — `run.py:47`. Exact match. HOLDS.

**Finding 7: Simultaneous best responses**
- Code lines 41-46: `target = [BR(q2,...), BR(q1,...)]` uses `path[t]` for both before updating either. Correct simultaneous update. HOLDS.

**Finding 8: Closed-form table row**
- Code lines 234-240: row inserted with `q_star` and `fixed_point_residual(np.array([q_star,q_star]),...)`. CSV: `2.6667, 2.6667, 4.44e-16`. HOLDS.

**Finding 9: Figure axis labels**
- `ax1.set_xlabel("$q_2$")`, `ax1.set_ylabel("$q_1$")` — `run.py:185-186`. `BR_1(q_2)` plotted on y-axis (q1) vs x-axis (q2). Paths: `path[:,1], path[:,0]` — x=q2, y=q1. Consistent throughout. HOLDS.

## Cross-cutting patterns

- None. The single original finding is fully resolved. The tutorial is compact and all equations map to results.

## TDD execution sequence

0. **Bullshit score: 0%.** No issues. No action required.
1. Test `test_honest_fix_equation_shows_non_negativity` now PASSES (confirmed by pytest run 2026-05-20).
2. Test `test_violated_invariant_code_clips_but_equation_does_not` now FAILS as expected (confirms fix is in place).
3. Test `test_code_best_response_still_clips_at_zero` PASSES (code unchanged, correct).
