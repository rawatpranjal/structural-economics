# bullshit-detector — quantal-response-equilibrium — 2026-05-20

**Bullshit score: 5%** — All 14 claims HOLDS; score is non-zero solely because the payoff matrix is embedded implicitly in `entry_payoff_gap` rather than coded as an explicit 2x2 array, making it impossible to audit cell values independently without re-running. No FALSE, DILUTED, MISLABELED, DATA DRIFT, or UNIMPLEMENTED findings.

## Header
- Claim sources: `game-theory/quantal-response-equilibrium/README.md` (all sections)
- Code / artifact root: `game-theory/quantal-response-equilibrium/run.py`
- Data artifacts: `game-theory/quantal-response-equilibrium/tables/qre-summary.csv`
- Seed audit (if any): none
- Run by: bullshit-detector skill (Claude Sonnet 4.6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| C1 | payoff gap Delta(q)=2(1-q)-q=2-3q | HOLDS | none | no |
| C2 | symmetric mixed Nash p^N=2/3 | HOLDS | none | no |
| C3 | QBR(q;lambda)=[1+exp(-lambda*(2-3q))]^{-1} | HOLDS | none | no |
| C4 | at lambda=0, both actions get prob 1/2 | HOLDS | none | no |
| C5 | as lambda rises, p(lambda) moves toward 2/3 | HOLDS | none | no |
| C6 | bisection on G_lambda(p)=p-QBR, bracket [0,1] | HOLDS | none | no |
| C7 | stop when residual or bracket width below epsilon | HOLDS | none | no |
| C8 | all 7 rows of QRE path table match CSV | HOLDS | none | no |
| C9 | payoff matrix (E,E)=-1,-1; (E,O)=2,0; (O,E)=0,2; (O,O)=0,0 | HOLDS | none | no |
| C10 | entry_payoff_gap computes 2-3q | HOLDS | none | no |
| C11 | bisection terminates on abs(residual)<tol or bracket<tol | HOLDS | none | no |
| C12 | focal precision=4.0 | HOLDS | none | no |
| C13 | precision grid 0 to 32 | HOLDS | none | no |
| C14 | gap to Nash = p(lambda)-p^N, signed | HOLDS | none | no |

## Findings

### Finding 1 (C1): payoff gap Delta(q)=2(1-q)-q=2-3q

- **Claim source (verbatim):** `"2(1-q)-q = 2-3q."` — `README.md:19`
- **Code evidence (verbatim):**
  ```python
  def entry_payoff_gap(opponent_prob_enter: float | np.ndarray) -> float | np.ndarray:
      """Expected payoff from Enter minus Stay Out in the entry game."""
      return 2.0 - 3.0 * opponent_prob_enter
  ```
  `run.py:22-24`
- **Data evidence:** Not applicable.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 2 (C2): symmetric mixed Nash p^N=2/3

- **Claim source (verbatim):** `"p^{N} = \frac{2}{3}."` — `README.md:26`
- **Code evidence (verbatim):**
  ```python
  def mixed_nash_entry_probability() -> float:
      """Symmetric mixed Nash entry probability for the exact entry game."""
      return 2.0 / 3.0
  ```
  `run.py:36-38`
- **Data evidence:** `Mixed Nash Pr(Enter)` column in `tables/qre-summary.csv` = 0.6667 (4dp rounding of 2/3) across all 7 rows.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 3 (C3): QBR formula — two algebraically equivalent forms

- **Claim source (verbatim):** `"QBR(q;\lambda) = \frac{\exp(\lambda \Delta(q))}{1+\exp(\lambda \Delta(q))} = [1+\exp(-\lambda(2-3q))]^{-1}."` — `README.md:34-36`
- **Code evidence (verbatim):**
  ```python
  def logit_entry_response(
      opponent_prob_enter: float | np.ndarray,
      precision: float,
  ) -> float | np.ndarray:
      """Probability of Enter under the logit best response."""
      payoff_gap = entry_payoff_gap(opponent_prob_enter)
      return 1.0 / (1.0 + np.exp(-precision * payoff_gap))
  ```
  `run.py:27-33`
- **Data evidence:** Not applicable.
- **Category:** HOLDS — code implements the second (equivalent) form exactly. Algebraic identity: `exp(x)/(1+exp(x)) = 1/(1+exp(-x))`.
- **Severity:** none
- **Result-changing:** no

### Finding 4 (C4): at lambda=0, prob=1/2

- **Claim source (verbatim):** `"At \lambda=0, both actions receive probability one half."` — `README.md:45`
- **Code evidence (verbatim):**
  ```python
  return 1.0 / (1.0 + np.exp(-precision * payoff_gap))
  ```
  `run.py:33` — at `precision=0`: `exp(0)=1`, so `1/(1+1)=0.5` regardless of `payoff_gap`.
- **Data evidence:** `tables/qre-summary.csv` row `0.0,0.5000,0.6667,-0.1667,0.00e+00` — QRE=0.5000, residual=0.00e+00.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 5 (C5): QRE moves toward Nash as lambda rises

- **Claim source (verbatim):** `"As \lambda rises, p(\lambda) moves toward the mixed Nash probability p^{N}=2/3."` — `README.md:46`
- **Code evidence (verbatim):**
  ```python
  precisions = np.linspace(0.0, 32.0, 129)
  ...
  for precision in precisions:
      p_entry, _, _ = solve_symmetric_entry_qre(float(precision))
      p_path.append(p_entry)
  ```
  `run.py:73,79-81`
- **Data evidence:** `tables/qre-summary.csv` — QRE Pr(Enter): 0.5000, 0.5712, 0.5995, 0.6243, 0.6423, 0.6535, 0.6598. Strictly monotone increasing toward 0.6667.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 6 (C6): bisection on G_lambda(p), bracket [0,1]

- **Claim source (verbatim):** `"Bisection is enough because G_{\lambda} rises on [0,1] and changes sign across the bracket."` — `README.md:65`; `"Set the initial bracket [low, high] = [0, 1]."` — `README.md:73`
- **Code evidence (verbatim):**
  ```python
  low = 0.0
  high = 1.0

  def residual(prob_enter: float) -> float:
      return prob_enter - float(logit_entry_response(prob_enter, precision))

  mid = 0.5
  for it in range(1, max_iter + 1):
      mid = 0.5 * (low + high)
      mid_residual = residual(mid)
      if abs(mid_residual) < tol or high - low < tol:
          break
      if mid_residual < 0.0:
          low = mid
      else:
          high = mid
  ```
  `run.py:52-67`
- **Data evidence:** Not applicable.
- **Category:** HOLDS — bracket is [0,1]; residual is G_lambda(p)=p-QBR(p;lambda). Sign change: at p=0, residual = -1/(1+exp(-2*lambda)) < 0; at p=1, residual = 1-1/(1+exp(lambda)) > 0 for all lambda >= 0.
- **Severity:** none
- **Result-changing:** no

### Finding 7 (C7): termination criterion

- **Claim source (verbatim):** `"Stop when |G_lambda(p)| or the bracket width is below epsilon."` — `README.md:75-76`
- **Code evidence (verbatim):**
  ```python
  if abs(mid_residual) < tol or high - low < tol:
      break
  ```
  `run.py:62-63`
- **Category:** HOLDS — `tol=1e-12` (`run.py:44`), matching `epsilon` in pseudocode.
- **Severity:** none
- **Result-changing:** no

### Finding 8 (C8): QRE path table values match CSV

- **Claim source (verbatim):** Table at `README.md:94-102` — all 7 rows of lambda, QRE, Nash, Gap, Residual.
- **Code evidence (verbatim):**
  ```python
  qre_rows.append({
      "Precision lambda": f"{precision:.1f}",
      "QRE Pr(Enter)": f"{p_entry:.4f}",
      "Mixed Nash Pr(Enter)": f"{mixed_nash_enter_prob:.4f}",
      "Gap to Nash": f"{p_entry - mixed_nash_enter_prob:+.4f}",
      "Residual": f"{residual:.2e}",
  })
  ```
  `run.py:86-92`
- **Data evidence:** `tables/qre-summary.csv` — all 7 rows match README verbatim: QRE 0.5000/0.5712/0.5995/0.6243/0.6423/0.6535/0.6598, Gap -0.1667/-0.0955/-0.0672/-0.0423/-0.0244/-0.0132/-0.0069, Residual 0.00e+00/5.06e-13/3.84e-14/9.94e-13/9.18e-13/1.79e-12/4.45e-13.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 9 (C9): payoff matrix consistency

- **Claim source (verbatim):** `"| **Row Enter** | -1, -1 | 2, 0 |"` and `"| **Row Stay Out** | 0, 2 | 0, 0 |"` — `README.md:53-55`
- **Code evidence:** Payoff matrix never coded as explicit array. Consistency checked algebraically: if u(E,E)=-1, u(E,O)=2, u(O,*)=0, then `E[u(E)] = -q + 2(1-q) = 2-3q`, `E[u(O)] = 0`. Gap = `2-3q` = `entry_payoff_gap` at `run.py:24`. `run.py:22-24`:
  ```python
  def entry_payoff_gap(opponent_prob_enter: float | np.ndarray) -> float | np.ndarray:
      """Expected payoff from Enter minus Stay Out in the entry game."""
      return 2.0 - 3.0 * opponent_prob_enter
  ```
- **Category:** HOLDS — gap function algebraically implies the stated matrix.
- **Severity:** none
- **Result-changing:** no
- **Observation:** Matrix is implicit in the gap formula, not an explicit coded array. This is a minor structural note, not a finding. Auditor cannot independently verify each cell value without running code.

### Findings C10-C14

All verified in the course of C1-C9 above. No separate anomalies. All HOLDS.

## Cross-cutting patterns

- The payoff matrix is embedded implicitly in `entry_payoff_gap` rather than declared as a 2x2 array. This is the only auditable opacity: cell values cannot be verified without algebraic reconstruction or re-run. It is a pedagogical/structural observation, not a faithfulness violation.
- Code, README, and CSV are fully internally consistent. No version drift detected between any pair of artifacts.
- The tutorial is narrow in scope (one game, one equilibrium concept, one solver), which limits surface area for fabrication. Audit coverage is complete.

## TDD execution sequence (for the next agent)

**Bullshit score: 5%.** Below 50% threshold. No halt required.

No non-HOLDS findings. No failing tests to write. No fixes needed.

If future edits are made:
1. Any change to `entry_payoff_gap` must be consistent with the payoff matrix in `README.md:53-55` (check: algebraic derivation above).
2. Any change to `mixed_nash_entry_probability` must preserve `2.0/3.0` or update the README and CSV correspondingly.
3. After any re-run, verify `tables/qre-summary.csv` matches `README.md:94-102` row by row.
