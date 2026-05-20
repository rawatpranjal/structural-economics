# bullshit-detector -- smolyak-sparse-grids -- 2026-05-20

**Bullshit score: 30%** -- Three DATA DRIFT findings (wrong worked-example numerics in Equations) plus one MISLABELED metric ("absolute" Euler error is actually relative). No FALSE or UNIMPLEMENTED findings; algorithm, node counts, and collocation machinery all hold.

## Header

- Claim sources: `computational-methods/smolyak-sparse-grids/README.md`
- Code / artifact root: `computational-methods/smolyak-sparse-grids/run.py`
- Data artifacts: `computational-methods/smolyak-sparse-grids/tables/accuracy.csv`, `computational-methods/smolyak-sparse-grids/tables/grid-counts.csv`
- Seed audit: none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "absolute Euler error" in Results and accuracy table | MISLABELED | MED | no (numbers correct; label misleads readers on units) |
| 2 | A^(1/(1-alpha)) worked values are (1.000, 0.846, 1.161, 0.703) | DATA DRIFT | LOW | no (pedagogical example only; code uses correct values) |
| 3 | Z approx 3.710 | DATA DRIFT | LOW | no (pedagogical approximation only) |
| 4 | omega[2] approx 0.313 in Equations; Setup table says 0.312 | DATA DRIFT | LOW | no (Setup table is correct; Equations rounding is wrong) |
| 5 | Residual equation (Y-S)*S^(alpha-1) = 1/(beta*alpha*Z^(1-alpha)*E_n) | HOLDS | -- | -- |
| 6 | Scalar Euler solved by brentq at each node | HOLDS | -- | -- |
| 7 | Smolyak node counts H(d,mu) match table | HOLDS | -- | -- |
| 8 | Basis count equals node count (square Phi) | HOLDS | -- | -- |
| 9 | Closed-form policy S* = alpha*beta*Y | HOLDS | -- | -- |
| 10 | Tensor node formula (2^mu+1)^d used in comparison table | HOLDS | -- | -- |
| 11 | Admissible level indices: mu+1 <= sum(i) <= mu+d, i_k >= 1 | HOLDS | -- | -- |
| 12 | Gauss-Hermite quadrature normalized for eps ~ N(0,1) | HOLDS | -- | -- |
| 13 | degree_level function matches Judd et al. convention | HOLDS | -- | -- |

## Findings

### Finding 1: "absolute Euler error" mislabels a relative metric

- **Claim source (verbatim):** "The worst-case absolute Euler error on the 10,000-point Sobol test set is 1.65e-04" -- `README.md:287`
- **Code evidence (verbatim):**
  ```python
  rhs = BETA * ALPHA * Z ** (1.0 - ALPHA) * S ** (ALPHA - 1.0) * E_term
  lhs = 1.0 / c
  return np.abs(lhs / rhs - 1.0)
  ```
  `run.py:347-349`
- **Data evidence:** `tables/accuracy.csv` column header reads `Max Euler error`; value for mu=3 is `1.65e-04`. The CSV header does not say "absolute," but the README prose calls it "absolute."
- **Category:** MISLABELED
- **Severity:** MED
- **Result-changing:** no -- the numbers in the table are correct (code produces them and CSV matches README). The label misleads readers who interpret "absolute Euler error" as the raw difference `|lhs - rhs|` (in marginal utility units) rather than the dimensionless ratio `|lhs/rhs - 1|`.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "abs(lhs / rhs - 1" in inspect.getsource(euler_errors_at)  # PASSES on current code
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "relative" in open("README.md").read().lower() or "lhs/rhs" in open("README.md").read()  # PASSES after fix, currently FAILS
  ```

### Finding 2: Worked A^(1/(1-alpha)) values in Equations are wrong at 3rd decimal

- **Claim source (verbatim):** "A^{1/(1-alpha)} approx (1.000, 0.846, 1.161, 0.703)" -- `README.md:50`
- **Code evidence (verbatim):**
  ```python
  weighted = A_SECTORS ** (1.0 / (1.0 - ALPHA))
  ```
  `run.py:149`
  With `A_SECTORS = [1.0, 0.9, 1.1, 0.8]` and `ALPHA = 0.36`, the exact values are `[1.000000, 0.848211, 1.160583, 0.705632]`. Rounded to 3 decimal places: `(1.000, 0.848, 1.161, 0.706)`.
- **Data evidence:** `tables/grid-counts.csv` does not contain these intermediates. The discrepancy is entirely within the README Equations worked example.
- **Category:** DATA DRIFT
- **Severity:** LOW -- the code uses `A_SECTORS ** (1/(1-ALPHA))` correctly; only the prose-inlined approximation is wrong.
- **Result-changing:** no -- the worked example is pedagogical. The model solution uses the exact computed values throughout.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(0.9 ** (1/(1-0.36)) - 0.846) < 0.001  # PASSES on buggy README claim (0.848 vs 0.846 diff = 0.002 -- actually FAILS, showing the claim is wrong)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(0.9 ** (1/(1-0.36)) - 0.848) < 0.001  # PASSES on corrected value
  ```

### Finding 3: Z approx 3.710 is wrong; correct 3-decimal value is 3.714

- **Claim source (verbatim):** "Z approx 3.710" -- `README.md:52`
- **Code evidence (verbatim):**
  ```python
  def Z_const() -> float:
      return float(np.sum(A_SECTORS ** (1.0 / (1.0 - ALPHA))))
  ```
  `run.py:154-155`
  Exact result: `3.714425...`. Rounded to 3 decimal places: `3.714`, not `3.710`.
- **Data evidence:** not directly in any CSV; the worked example lives only in README.md.
- **Category:** DATA DRIFT
- **Severity:** LOW -- tilde hedge ("approx") is present, but 3.710 vs 3.714 exceeds normal rounding tolerance.
- **Result-changing:** no.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(sum(np.array([1.0, 0.9, 1.1, 0.8]) ** (1/(1-0.36))) - 3.710) < 0.001  # FAILS (diff = 0.004)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(sum(np.array([1.0, 0.9, 1.1, 0.8]) ** (1/(1-0.36))) - 3.714) < 0.001  # PASSES
  ```

### Finding 4: omega[2] is 0.312 (Setup table) vs 0.313 (Equations text) -- Equations is wrong

- **Claim source (verbatim):** "omega approx (0.269, 0.228, 0.313, 0.190)" -- `README.md:54` (Equations section)
- **Code evidence (verbatim):**
  ```python
  def sector_shares() -> np.ndarray:
      weighted = A_SECTORS ** (1.0 / (1.0 - ALPHA))
      return weighted / weighted.sum()
  ```
  `run.py:148-150`
  Exact omega[2] = 0.312453. Rounded to 3 decimal places: `0.312`. The Model Setup table at `README.md:180` correctly states `0.312`.
- **Data evidence:** `README.md:180` (Setup table): omega = `(0.269, 0.228, 0.312, 0.190)` -- this is the correct 3-dp rounding. The Equations section at line 54 uses `0.313` which is the wrong rounding.
- **Category:** DATA DRIFT (two locations in the same README disagree; code is correct; Setup table is correct; Equations is wrong).
- **Severity:** LOW -- pedagogical example only.
- **Result-changing:** no.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs((np.array([1.0,0.9,1.1,0.8])**(1/0.64) / (np.array([1.0,0.9,1.1,0.8])**(1/0.64)).sum())[2] - 0.313) < 5e-4  # PASSES on buggy claim (0.3124 vs 0.313 diff=0.0006>0.0005 -- actually FAILS)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs((np.array([1.0,0.9,1.1,0.8])**(1/0.64) / (np.array([1.0,0.9,1.1,0.8])**(1/0.64)).sum())[2] - 0.312) < 5e-4  # PASSES
  ```

## Cross-cutting patterns

- All three DATA DRIFT findings (F2, F3, F4) originate in the same Equations worked-example block (`README.md` lines 46-55). The block was likely computed by hand or with a different rounding convention than the code. A single pass regenerating those inline numbers from the code would fix all three.
- The MISLABELED finding (F1) is a terminology error that appears in both the inline Results prose (`README.md:287`) and the accuracy table description (`README.md:279`). The CSV header is neutral (`Max Euler error`), so no CSV change is needed -- only the README prose.
- No finding touches the algorithm, node counts, basis construction, or Euler solver. The computational core holds end to end.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 30%.** Below the 50% halt threshold. Proceed with fixes; no need to surface to user before touching code.
1. **F1 (MISLABELED):** Write a test asserting `euler_errors_at` returns `np.abs(lhs / rhs - 1)` (confirmed PASSES today). Then change the README prose at line 287 from "absolute Euler error" to "relative Euler error" and verify the test still confirms the code is unchanged.
2. **F2, F3, F4 (DATA DRIFT):** Regenerate the Equations worked-example block by running:
   ```python
   import numpy as np
   alpha, A = 0.36, np.array([1.0, 0.9, 1.1, 0.8])
   w = A**(1/(1-alpha)); Z = w.sum(); omega = w/Z
   print(np.round(w, 3), round(Z, 3), np.round(omega, 3))
   ```
   Replace lines 46-55 of `README.md` with the correct rounded values: `A^exp = (1.000, 0.848, 1.161, 0.706)`, `Z = 3.714`, `omega = (0.269, 0.228, 0.312, 0.190)`.
3. After fixes, re-run `python scripts/validate_catalog.py` to confirm no math-rendering regressions.
4. Re-run this skill on the updated README to confirm score drops to <=10%.
