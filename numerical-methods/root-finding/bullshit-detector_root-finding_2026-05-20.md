# bullshit-detector - root-finding - 2026-05-20

**Bullshit score: 25%** - One prose claim ("roughly an order of magnitude") is quantifiably false against the tutorial's own table (4-6x ratio, not 10x); all formulas, calibration values, and table numbers hold; one runtime-computed value cannot be grounded without re-run.

## Header
- Claim sources: `numerical-methods/root-finding/README.md` (prose, Equations, Results, numbers)
- Code / artifact root: `numerical-methods/root-finding/run.py`
- Data artifacts: `numerical-methods/root-finding/tables/comparison.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|-----------------|
| 1 | "Brent and Newton finish in roughly an order of magnitude fewer iterations than bisection" | FALSE | MED | no (table numbers correct; prose narrative wrong) |
| 2 | "hand-coded Brent root matches scipy.optimize.brentq to 0.00e+00" | DATA DRIFT | LOW | no - needs re-run to verify |
| 3 | Equations: Newton = Method 3, Brent = Method 4; Solution Method / table: Brent before Newton | DATA DRIFT | LOW | no |
| 4 | K_d formula, Z formula, Zprime formula, r_star, calibration params | HOLDS | - | - |
| 5 | Bisection iteration count = 29, Secant = 9, Brent = 7, Newton = 5 | HOLDS | - | - |
| 6 | All residual and error values in table match CSV | HOLDS | - | - |
| 7 | Divergence count: 1 of 9 Newton starts diverge | HOLDS | - | - |
| 8 | Secant convergence order (1+sqrt(5))/2 ≈ 1.618 | HOLDS | - | - |
| 9 | Brent uses IQI through last three ordinates; falls back to secant/bisection | HOLDS | - | - |

## Findings

### Finding 1: "roughly an order of magnitude" - FALSE

- **Claim source (verbatim):** "All four methods reach the closed-form root within tolerance. Brent and Newton finish in roughly an order of magnitude fewer iterations than bisection." - `README.md:164`
- **Code evidence (verbatim):**
  ```python
  table_data = {
      "Method": [m[0] for m in methods],
      "Inputs": [m[4] for m in methods],
      "Iterations": [int(histories[m[0]][-1, 0]) for m in methods],
      ...
  }
  ```
  `run.py:487-494`
- **Data evidence (verbatim):**
  ```
  Bisection,sign-change bracket,29,...
  Secant,two starting points,9,...
  Brent,sign-change bracket,7,...
  Newton-Raphson,x_0 and Z',5,...
  ```
  `tables/comparison.csv:1-5`
- **Category:** FALSE
- **Severity:** MED
- **Result-changing:** no (the table numbers are correct and immediately visible to the reader; the prose overstates the ratio)
- **Quantification:** 29/7 = 4.14x (Brent), 29/5 = 5.8x (Newton). One order of magnitude = 10x. Actual ratios are 41-58% of one order of magnitude.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert 29 / 7 >= 10  # PASSES on current code (claim requires >=10x); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert 4 <= 29 / 7 < 10  # prose says "4-6x fewer" or "roughly 4x"; PASSES on honest fix
  ```

### Finding 2: Brent-scipy match value - DATA DRIFT / needs re-run

- **Claim source (verbatim):** "The hand-coded Brent root matches $\mathrm{scipy.optimize.brentq}$ to **0.00e+00**." - `README.md:142`
- **Code evidence (verbatim):**
  ```python
  f"**{abs(bre_root - scipy_root):.2e}**."
  ```
  `run.py:369`
- **Data evidence:** Value `0.00e+00` is dynamically computed at runtime and not stored in `tables/comparison.csv`. Cannot be grounded against any committed artifact.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no - the value is plausible (both solvers use same Z and same tol), but the README number **needs re-run to verify**.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.00e+00" in open("numerical-methods/root-finding/README.md").read()  # PASSES on current README
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(bre_root - scipy_root) < 1e-14  # must hold after re-run; PASSES on honest run
  ```

### Finding 3: Method numbering inconsistency - DATA DRIFT

- **Claim source (verbatim):** "### Method 3: Newton-Raphson" and "### Method 4: Brent" - `README.md:53,61`
- **Code evidence (verbatim):**
  ```python
  methods = [
      ("Bisection", bis_root, bis_hist, "linear (1/2)", "sign-change bracket"),
      ("Secant", sec_root, sec_hist, "superlinear (~1.618)", "two starting points"),
      ("Brent", bre_root, bre_hist, "superlinear", "sign-change bracket"),
      ("Newton-Raphson", new_root, new_hist, "quadratic", "x_0 and Z'"),
  ]
  ```
  `run.py:172-177`
- **Data evidence:** `tables/comparison.csv` row order: Bisection, Secant, Brent, Newton-Raphson. The Equations section numbers Newton as 3 and Brent as 4; the Solution Method pseudocode section and the results table both present Brent before Newton-Raphson.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no - labels only; no numeric output is affected
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "### Method 3: Newton-Raphson" in open("numerical-methods/root-finding/README.md").read()
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "### Method 3: Brent" in open("numerical-methods/root-finding/README.md").read()
  ```

## Cross-cutting patterns

- All formula claims (K_d, Z, Z', bisection midpoint, secant step, Newton step, IQI branch) are grounded verbatim in `run.py` and hold. The mathematical content of this tutorial is clean.
- All numeric values in `tables/comparison.csv` match `README.md` exactly. The only unverifiable number is the dynamically computed Brent-vs-scipy residual, which is not persisted to any CSV.
- The single FALSE finding (Finding 1) is a prose exaggeration that the tutorial's own table immediately contradicts. A reader who looks at 29 vs 7/5 will not be misled, but the sentence as written is not defensible.
- The method-ordering inconsistency (Finding 3) is cosmetic: Equations labels Newton=3, Brent=4 while every other section reverses the pair. This is the kind of drift that accumulates when the Equations subsection numbering is written independently of the methods list.

## TDD execution sequence (for the next agent)

0. **Bullshit score = 25%.** Below the 50% halt threshold. Proceed with fix, but surface Finding 1 to the user before merging.

1. **Finding 1 (FALSE - prose claim).**
   Turn violated invariant into a pytest: assert `29/7 >= 10` passes on current README text. Confirm. Then fix the prose in `run.py`'s `add_results` call (run.py:496-500) to say "4-6x fewer" or "well under half an order of magnitude fewer" instead of "roughly an order of magnitude fewer." Re-run `python run.py` and confirm the honest-fix assertion passes.

2. **Finding 2 (DATA DRIFT - unverified runtime value).**
   Re-run `python run.py` and inspect the emitted value for `abs(bre_root - scipy_root)`. If it prints `0.00e+00`, the README claim is confirmed and the DATA DRIFT finding resolves to HOLDS. Persist the value to `tables/comparison.csv` or a separate `tables/scipy_match.csv` so future audits can ground it without re-running.

3. **Finding 3 (DATA DRIFT - label ordering).**
   In `run.py` `add_equations` string (lines 237-293), swap the Method 3 / Method 4 labels: rename `### Method 3: Newton-Raphson` to `### Method 3: Brent` and `### Method 4: Brent` to `### Method 4: Newton-Raphson`, then move the corresponding subsection blocks to match the order Bisection, Secant, Brent, Newton-Raphson. This aligns Equations with Solution Method and the table.

4. After fixes, re-run this skill on the updated code to confirm all findings now read HOLDS and the bullshit score is <=10%.
