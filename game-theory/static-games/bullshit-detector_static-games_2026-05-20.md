# bullshit-detector — static-games — 2026-05-20

**Bullshit score: 10%** — one DILUTED LOW finding (BR equation omits non-negativity constraint that code enforces); never binds at reported parameters; all numeric claims verified against CSV and manual replication.

## Header
- Claim sources: `game-theory/static-games/README.md` (prose, Equations, Model Setup, Results table)
- Code / artifact root: `game-theory/static-games/run.py`
- Data artifacts: `game-theory/static-games/tables/convergence-summary.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | BR formula omits non-negativity constraint | DILUTED | LOW | no (never binds at reported params/starts) |
| 2 | q* = 2.6667 | HOLDS | — | — |
| 3 | P* = 4.667 | HOLDS | — | — |
| 4 | pi* = 7.111 | HOLDS | — | — |
| 5 | Residuals match table (5.60e-06, 4.44e-16) | HOLDS | — | — |
| 6 | Damped update q_{t+1} = (1-lambda)q_t + lambda BR(q_t) | HOLDS | — | — |
| 7 | Simultaneous best responses (each firm responds to other's current output) | HOLDS | — | — |
| 8 | Closed-form row in table uses exact q* | HOLDS | — | — |
| 9 | Figure axes consistent with BR curve labeling | HOLDS | — | — |

## Findings

### Finding 1: Equations section BR formula omits non-negativity constraint present in code

- **Claim source (verbatim):** "The interior first-order condition gives the best response $BR_i(q_j)=\frac{a-c-bq_j}{2b}.$" — `README.md:26-29`
- **Code evidence (verbatim):**
  ```python
  def cournot_best_response(
      q_other: float | np.ndarray,
      a: float,
      b: float,
      c: float,
  ) -> float | np.ndarray:
      """Firm i's best response in a linear Cournot duopoly."""
      return np.maximum(0.0, (a - c - b * q_other) / (2.0 * b))
  ```
  `run.py:17-24`
- **Data evidence:** All six reported final quantities (from CSV `tables/convergence-summary.csv`) equal 2.6667; at these values (a=10, b=1, c=2, q*=2.667) the non-negativity constraint never binds for any starting point in `starts = [(0.5, 7.0), (7.0, 0.5), (7.0, 7.0)]`. Replication confirmed: all residuals match the CSV to the digit shown.
- **Category:** DILUTED — the Equations claim says "interior FOC" but the code silently adds the boundary constraint `max(0, .)`. The prose would mislead a reader who tries to extend this to parameters where the corner solution binds.
- **Severity:** LOW — does not change any reported number at the published calibration.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "np.maximum" in inspect.getsource(cournot_best_response) and "0.0" in inspect.getsource(cournot_best_response)
  # PASSES on current code (proves the constraint is there); FAILS if code is rewritten to match the bare interior formula
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "BR_i(q_j)=\\frac{a-c-bq_j}{2b}" not in readme_equations_section or r"\max\{0," in readme_equations_section
  # PASSES on honest fix (equation includes non-negativity); FAILS on current README
  ```

### HOLDS block (all other claims verified)

**Finding 2: q* = 2.667**
- README Model Setup table: `$q^{\ast}$ | 2.667` — `README.md:47`
- Code: `q_star = (a - c) / (3.0 * b)` → `(10-2)/(3*1) = 2.6667` — `run.py:87`
- Independently replicated: `2.6666666667` truncated to 4 dp = `2.6667`. HOLDS.

**Finding 3: P* = 4.667**
- README: `$P^{\ast}$ | 4.667` — `README.md:48`
- Code: `p_star = price(2.0 * q_star, a, b)` = `10 - 1*(2*2.6667)` = `4.6667` — `run.py:88`
- README equation `P^{*}=a-2bq^{*}` matches code `a - b*(2*q_star)`. HOLDS.

**Finding 4: pi* = 7.111**
- README: `$\pi^{\ast}$ | 7.111` — `README.md:49`
- Code: `profit(q_star, q_star, a, b, c)` = `(4.6667-2)*2.6667` = `7.1111` — `run.py:89`
- Replicated: `7.1111111111`. HOLDS.

**Finding 5: Convergence table residuals**
- README table rows — `README.md:84-87`; CSV `tables/convergence-summary.csv` rows match exactly.
- Residual for `(0.5,7.0)` and `(7.0,0.5)`: independently replicated as `5.60e-06`. README shows `5.6e-06` (1 fewer sig fig in display only; same value). HOLDS.
- Residual for `(7.0,7.0)` and closed form: `4.44e-16` (machine epsilon). HOLDS.

**Finding 6: Damped update formula**
- README pseudocode step 3: `q_{t+1} = (1-lambda) q_t + lambda BR(q_t)` — `README.md:63`
- Code line 47: `path[t + 1] = (1.0 - damping) * path[t] + damping * target` — `run.py:47`
- Exact match. HOLDS.

**Finding 7: Simultaneous best responses**
- README claims firms respond simultaneously (static game). Code lines 40-46 compute `target=[BR(q2,...), BR(q1,...)]` using `path[t]` for both firms before updating either. Correct simultaneous update. HOLDS.

**Finding 8: Closed-form table row**
- Code lines 233-240 insert row with `q_star` and `fixed_point_residual(np.array([q_star,q_star]),...)`. CSV confirms `2.6667, 2.6667, 4.44e-16`. HOLDS.

**Finding 9: Figure axis labels**
- README: figure described as "Cournot best-response curves and damped iteration paths" — `README.md:74`
- Code: `ax1.xlabel="$q_2$"`, `ax1.ylabel="$q_1$"`. `ax1.plot(q_grid, br, label="$BR_1(q_2)$")` plots BR1 on y-axis as function of q2 on x-axis. `ax1.plot(br, q_grid, label="$BR_2(q_1)$")` plots BR2 on x-axis as function of q1 on y-axis. Iteration paths: `ax1.plot(path[:,1], path[:,0], ...)` — x=q2, y=q1. Consistent with axis labels throughout. HOLDS.

## Cross-cutting patterns

- Tutorial is compact and well-matched to code. Every equation in `Equations` drives a result in `Results`. No surplus derivations. No phantom claims.
- The single DILUTED finding (missing non-negativity in BR equation) is a pattern seen in pedagogical Cournot expositions that derive the interior solution only: the boundary condition is standard but often omitted from the formula display while being present in any correct numerical implementation. Adding `\max\{0, \frac{a-c-bq_j}{2b}\}` to the equation would close the gap.
- All numeric claims verified by independent replication from first principles (no re-run of `run.py` required).

## TDD execution sequence (for the next agent)

0. **Bullshit score: 10%.** Below 25% threshold. No halt required. Proceed.
1. Turn the violated invariant into a pytest: assert `np.maximum` appears in `cournot_best_response` source and the README Equations section lacks the `\max\{0,\cdot\}` form. Confirm test passes on current state.
2. Honest-fix test: assert README Equations BR formula includes non-negativity (e.g., `\max\{0,\frac{a-c-bq_j}{2b}\}`). Confirm this fails on current README.
3. Fix: edit `README.md` Equations and the corresponding string in `run.py:130` to show `BR_i(q_j)=\max\left\{0,\frac{a-c-bq_j}{2b}\right\}`. Re-run `python scripts/validate_catalog.py`.
4. All other findings are HOLDS; no data artifacts need refresh.
5. Re-run this skill after fix to confirm score drops to 0%.
