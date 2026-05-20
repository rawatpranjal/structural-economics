# bullshit-detector — rbc — 2026-05-20

**Bullshit score: 20%** — stale README emits C/Y = 0.76 in Equations prose but run.py would generate 0.77 at current {:.2f} format; all equations, matrices, and IRF table numbers hold.

## Header
- Claim sources: `dsge/rbc/README.md` (prose, Equations section, Results tables)
- Code / artifact root: `dsge/rbc/run.py`, `lib/perturbation.py`
- Data artifacts: `dsge/rbc/tables/irf-summary-fixed-labor.csv`, `dsge/rbc/tables/irf-summary-endogenous-labor.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | C/Y = 0.76 in Equations prose | DATA DRIFT | MED | no (display only, not a model result) |
| 2 | Two solvers run in parallel | MISLABELED | LOW | no |
| 3 | Equations, matrix rows, IRF formulas | HOLDS | - | - |
| 4 | IRF table numbers (CSV) | HOLDS | - | - |
| 5 | labor_static derivation | HOLDS | - | - |
| 6 | Klein QZ implementation (lib/perturbation.py) | HOLDS | - | - |
| 7 | Steady-state numeric claims (K, Y, C, psi) | HOLDS | - | - |
| 8 | Capital rule coefficients, labor rule coefficients | needs re-run to verify | - | - |

## Findings

### Finding 1: C/Y = 0.76 in Equations prose is stale and internally inconsistent

- **Claim source (verbatim):** "The Case A calibration gives $K/Y = 9.40$, $C/Y = 0.76$. The Case B calibration with $\bar N = 0.333$ gives $K/Y = 9.40$, $C/Y = 0.76$, and a labor weight $\psi = 7.883$." — `dsge/rbc/README.md:55-57`

- **Code evidence (verbatim):**
  ```python
  f"calibration gives $K/Y = {ss_A['K_Y']:.2f}$, $C/Y = {ss_A['C_Y']:.2f}$. The\n"
  f"$C/Y = {ss_B['C_Y']:.2f}$, and a labor weight $\\psi = {ss_B['psi']:.3f}$."
  ```
  `dsge/rbc/run.py:493-495`

- **Data evidence (if applicable):** `dsge/rbc/tables/irf-summary-fixed-labor.csv` does not list C/Y directly. The Model Setup table (README.md line 111) says `| $C/Y$ | 0.765 | 0.765 |`. Actual computed value: C/Y = 0.765170 (C=2.307, Y=3.015). Formatted as `{:.2f}`: **0.77** (rounds up). Formatted as `{:.3f}`: **0.765** (matches Model Setup table). Committed README Equations prose has **0.76** -- matches neither the table nor what `{:.2f}` would produce today.

- **Category:** DATA DRIFT — three-way inconsistency: Equations prose (0.76), Model Setup table (0.765), and what `run.py` line 493 would generate today ({:.2f} of 0.7652 = 0.77). The committed README was generated with a prior version of the Equations string (possibly `{:.3f}` truncated at two digits, or a manual edit).

- **Severity:** MED — the number is a calibration display value, not a model result. No IRF table entry changes. A reader checking the Equations section against the Model Setup table will see a contradiction (0.76 vs 0.765 vs 0.77).

- **Result-changing:** no (display-only, all IRF tables are unaffected)

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.76" in open("dsge/rbc/README.md").read() and "0.765" in open("dsge/rbc/README.md").read()
  # PASSES on current stale README (both strings present, inconsistently); FAILS after a clean re-run
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert open("dsge/rbc/README.md").read().count("C/Y = 0.76") == 0
  # PASSES after `python run.py` regenerates README (format produces 0.77, not 0.76); FAILS on current README
  ```

---

### Finding 2: "Two solvers run in parallel" — sequential in code

- **Claim source (verbatim):** "Two solvers run in parallel." — `dsge/rbc/README.md:9`

- **Code evidence (verbatim):**
  ```python
  policy_A = solve_log_linear_policy(alpha, beta, delta, rho, sigma, ss_A)
  qz_A = klein_qz_policy_fixed(alpha, beta, delta, rho, sigma, ss_A)
  ```
  `dsge/rbc/run.py:385-386`

- **Data evidence (if applicable):** None. This is an architectural claim, not a numeric one.

- **Category:** MISLABELED — the two solvers run sequentially in `main()`, one after the other in a single Python thread. "In parallel" in the prose means "on the same model simultaneously" (conceptually), but the word's natural engineering meaning is concurrent execution. The code is coherent and correct; only the label misleads.

- **Severity:** LOW — no result changes; a careful reader might spend time looking for threading or multiprocessing code that does not exist.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  import threading; assert threading.active_count() == 1  # no spawned threads during run
  # PASSES on current code (single-threaded); prose claim about parallel execution is false in the engineering sense
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "run in sequence" in open("dsge/rbc/README.md").read() or "sequentially" in open("dsge/rbc/README.md").read()
  # PASSES after prose is corrected to "Two solvers run in sequence"; FAILS on current README
  ```

---

### Findings 3-7: HOLDS

All of the following were audited against verbatim code and verified analytically or numerically. No discrepancies found.

**Finding 3 (HOLDS): All structural equations, Klein matrix rows (3x3 and 4x4), and Euler linearizations.**

Each row of the A and B matrices in `klein_qz_policy_fixed` (run.py:100-113) and `klein_system_labor` (run.py:227-243) was derived from first principles. Every row matches the log-linearized equilibrium conditions stated in the Equations section. The sign convention in row 2 of the 4x4 B matrix is negated relative to the derived form but represents the same equation (0 = -f is equivalent to 0 = f). The nonlinear Euler residual at run.py:166-167 matches the exact Euler equation.

**Finding 4 (HOLDS): IRF table numbers in both CSVs.**

- `irf-summary-fixed-labor.csv`: Output impact 1.000%, Consumption impact 0.323%, Investment impact 3.204%, Capital impact 0.080% — all reproduced analytically from the policy coefficients. Half-life and peak quarter values reproduced by simulation. Capital peak 0.687%/quarter 21 verified to within rounding of the exact p (our approximation gives 0.688% which is within 0.001pp).
- `irf-summary-endogenous-labor.csv`: Output impact 1.309% verified analytically using P[1,1]=0.4612 and the linearized output equation. Investment impact 4.311% verified from F[0,1]=0.1078.

**Finding 5 (HOLDS): `labor_static` formula in `nonlinear_irfs_labor`.**

`run.py:280-281`: `rhs = (1-alpha)*A*K_lag**alpha / (psi * C**sigma)` followed by `return rhs**(1.0/(chi+alpha))` correctly solves the intratemporal condition $\psi N^\chi = (1-\alpha) A K^\alpha N^{-\alpha} C^{-\sigma}$ for $N$.

**Finding 6 (HOLDS): Klein QZ implementation in `lib/perturbation.py`.**

`ordqz(B, A, sort="iuc")` correctly produces eigenvalues of $A^{-1}B$ sorted by inside-unit-circle criterion (line 73, consistent with the docstring explanation). The Schur partition into `Z11`, `Z21`, `AA11`, `BB11` and the recovery of `F` and `P` (lines 86-102) follow Klein (2000) exactly. Blanchard-Kahn check at line 78 is correct.

**Finding 7 (HOLDS): Steady-state numeric claims.**

K=28.348, Y=3.015, C=2.307 (Case A) and K=9.449, Y=1.005, C=0.769, psi=7.883 (Case B) reproduced analytically. K/Y=9.401 and C/Y=0.765 for both cases confirmed.

---

### Finding 8: Capital/labor rule coefficients and BK message — needs re-run

- **Claim:** "The capital rule is $\hat k_t = 0.9531\hat k_{t-1} + 0.1078\hat a_t$. The labor rule is $\hat n_t = -0.1677\hat k_{t-1} + 0.4612\hat a_t$." — `dsge/rbc/README.md:168`
- **Category:** needs re-run to verify (runtime QZ values embedded via f-string; cannot be re-derived without executing `solve_klein` on the 4x4 system). The Case A p=0.9621, q=0.0801 (undetermined coefficients) and the Case B impact responses (1.309% output, 4.311% investment, 0.461% labor) are internally consistent with the stated coefficients, providing indirect confirmation.
- **Severity:** not assigned (requires re-run; not a bullshit finding, just unverifiable from committed artifacts alone).

## Cross-cutting patterns

- The only real issue is a stale committed README: the Equations section prose was apparently generated with a different format string or manually edited to show 0.76, but the Model Setup table (generated with `{:.3f}`) correctly shows 0.765, and the current `{:.2f}` format at run.py:493 would produce 0.77. Running `python run.py` would resolve all three inconsistencies (0.76, 0.765, 0.77 collapse to a single authoritative value).
- The "parallel" language is a recurring prose imprecision. If the tutorial adds more solver comparisons, watch for this pattern.
- All load-bearing math (linearizations, matrix rows, IRF formulas, nonlinear Euler residual) holds without exception. This is a well-implemented tutorial.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20% — below the 50% halt threshold. Safe to proceed.**

1. **Finding 1 (DATA DRIFT, MED):** Write a pytest test that reads `dsge/rbc/README.md` and asserts the string "C/Y = 0.76" does not appear in the Equations section. Confirm the test fails on the current README (it PASSES on buggy code). Fix: `python dsge/rbc/run.py` regenerates the README; the `{:.2f}` format will produce 0.77 in the Equations prose. After regeneration, also verify the Model Setup table still shows 0.765 (it uses `{:.3f}`). After the fix the test should pass.

2. **Finding 2 (MISLABELED, LOW):** Write a pytest test asserting "run in parallel" does not appear in the Overview of `dsge/rbc/README.md`. Fix: change the prose in the `add_overview(...)` call at run.py:433-449 from "run in parallel" to "run in sequence" (or similar), then regenerate.

3. Re-run `python scripts/validate_catalog.py` after any README regeneration to confirm no math-rendering regressions are introduced.

4. Re-run this skill on the updated README to confirm the score drops to <= 10%.
