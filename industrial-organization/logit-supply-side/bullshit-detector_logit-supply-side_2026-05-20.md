# bullshit-detector -- logit-supply-side -- 2026-05-20

**Bullshit score: 20%** -- Two findings: one MISLABELED (notation inconsistency in Omega definition, numerically harmless due to logit symmetry) and one DATA DRIFT (MAE 0.455 not in any committed artifact, needs re-run to verify). No FALSE, no UNIMPLEMENTED, no result-changing errors.

## Header

- Claim sources: `industrial-organization/logit-supply-side/README.md` (all sections)
- Code / artifact root: `industrial-organization/logit-supply-side/run.py`
- Data artifacts: `industrial-organization/logit-supply-side/tables/estimation-results.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Omega_jk = -O_jk * ds_k/dp_j (derivative subscript convention) | MISLABELED | MED | no (logit symmetry makes D = D^T) |
| 2 | MAE of recovered costs is 0.455 dollars (market 0) | DATA DRIFT | LOW | needs re-run to verify |
| 3 | OLS estimates alpha at 1.009, true 1.500 | HOLDS | -- | -- |
| 4 | IV/2SLS estimates alpha at 1.465 | HOLDS | -- | -- |
| 5 | Berry inversion: ln(s_jt) - ln(s_0t) = delta_jt | HOLDS | -- | -- |
| 6 | Logit derivative own: -alpha*s_j*(1-s_j), cross: alpha*s_k*s_j | HOLDS | -- | -- |
| 7 | Omega m = s solved for markups | HOLDS | -- | -- |
| 8 | mc = p - markup (cost recovery) | HOLDS | -- | -- |
| 9 | Firms 1 and 2 own 2 products each; 3 firms total | HOLDS | -- | -- |
| 10 | IV instruments: cost_shifter excluded from demand (independent of xi) | HOLDS | -- | -- |
| 11 | Cross-elasticity = alpha*p_k*s_k, identical across rows (IIA) | HOLDS | -- | -- |
| 12 | All four parameter estimates match tables/estimation-results.csv | HOLDS | -- | -- |

## Findings

### Finding 1: Omega derivative subscript -- README convention vs code indexing

- **Claim source (verbatim):** "$$\Omega_{jk}=-O_{jk}\frac{\partial s_k}{\partial p_j}.$$" -- `README.md:33` (also `run.py:427`)

- **Code evidence (verbatim):**
  ```python
  def compute_share_derivatives(alpha: float, shares: np.ndarray) -> np.ndarray:
      """ds_j/dp_k matrix.  Own: -alpha*s_j*(1-s_j).  Cross: alpha*s_j*s_k."""
      J = len(shares)
      D = np.zeros((J, J))
      for j in range(J):
          for k in range(J):
              if j == k:
                  D[j, k] = -alpha * shares[j] * (1.0 - shares[j])
              else:
                  D[j, k] = alpha * shares[j] * shares[k]
      return D


  def compute_markups(alpha: float, shares: np.ndarray,
                      ownership: np.ndarray) -> np.ndarray:
      """Recover markups from Bertrand-Nash FOC: p - mc = Omega^{-1} s.

      Omega_jk = -ds_k/dp_j * O_jk  (only internalise own-firm products).
      """
      D = compute_share_derivatives(alpha, shares)
      omega = -D * ownership
      return np.linalg.solve(omega, shares)
  ```
  `run.py:274-295`

- **Data evidence (if applicable):** None -- this is a structural/notation finding.

- **Category:** MISLABELED

- **Severity:** MED

- **Result-changing:** no. The docstring of `compute_share_derivatives` says "ds_j/dp_k matrix" (row=share-changing product j, col=price-changing product k). The README equation uses the opposite subscript convention: `ds_k/dp_j` (row=price-changing product j, col=share-changing product k). The matrix `omega = -D * ownership` therefore uses `D[j,k] = ds_j/dp_k` where the README equation calls for `Omega_jk = -O_jk * ds_k/dp_j`. These are transposes of each other in the derivative subscripts. For simple logit, `ds_j/dp_k = ds_k/dp_j` (both equal `alpha*s_j*s_k` for cross terms; own terms are trivially symmetric). The Bertrand-Nash FOC in matrix form requires `(O .* D^T) * m = -s`; since `D^T = D` for simple logit, `omega = -(O .* D)` and `omega * m = s` is correct. The result is numerically exact; only the prose description of the subscript convention is internally inconsistent.

  This would become a FALSE/HIGH bug the moment the tutorial is extended to random-coefficients logit (where `D` is not symmetric), and the reader following the README notation would implement the wrong matrix.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "ds_j/dp_k" in compute_share_derivatives.__doc__ and "ds_k/dp_j" not in compute_share_derivatives.__doc__
  # PASSES on current code (docstring says ds_j/dp_k); FAILS if notation is harmonized to README convention
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "ds_k/dp_j" in compute_share_derivatives.__doc__ and compute_share_derivatives.__doc__.startswith("ds_k/dp_j")
  # PASSES if docstring is updated to match README convention; FAILS on current code
  ```

---

### Finding 2: MAE 0.455 dollars -- not present in any committed artifact

- **Claim source (verbatim):** "In market 0, recovered marginal costs have mean absolute error 0.455 dollars." -- `README.md:74`

- **Code evidence (verbatim):**
  ```python
  report.add_results(
      f"OLS estimates alpha at {ols_alpha:.3f}, below the true value {TRUE_ALPHA:.3f}. "
      "Unobserved quality raises both demand and price. IV/2SLS estimates alpha at "
      f"{iv['alpha']:.3f}.\n\n"
      "In market 0, recovered marginal costs have mean absolute "
      f"error {np.abs(est_mc - true_mc).mean():.3f} dollars. These outputs show how "
      "demand bias moves recovered costs."
  )
  ```
  `run.py:465-472`

- **Data evidence (if applicable):** `tables/estimation-results.csv` contains only the four demand parameters (alpha, beta_sugar, beta_fiber, beta_const). The MAE figure (0.455) appears only in `README.md:74`. No committed CSV or log file records the market-0 cost recovery error.

- **Category:** DATA DRIFT

- **Severity:** LOW -- the number is generated inline during `run.py` and embedded in the README at generation time. It is correct at the moment of the last run. The gap is that it cannot be independently verified from committed artifacts without a re-run.

- **Result-changing:** needs re-run to verify. If `run.py` is re-run with the same seed (42), the number should reproduce. If the seed or DGP has changed since the last README regeneration, the committed 0.455 may be stale.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not any("0.455" in row for row in open("tables/estimation-results.csv").readlines())
  # PASSES on current state (0.455 is absent from CSV); shows the gap
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(float(open("tables/cost-recovery-market0.csv").read().strip().split(",")[1]) - 0.455) < 0.001
  # PASSES if a cost-recovery table is committed alongside the README; FAILS currently
  ```

## Cross-cutting patterns

- The single structural risk is the notation mismatch in the derivative subscript convention. The current code is correct for simple logit because the derivative matrix is symmetric. Any future extension to random-coefficients logit (where `ds_j/dp_k != ds_k/dp_j`) would silently produce wrong markups unless the notation inconsistency is resolved first.

- The MAE claim (0.455) is the only numeric result in the README not backed by a committed CSV. The four demand parameter estimates are all in `tables/estimation-results.csv` and verified against the README table. Adding a `tables/cost-recovery-market0.csv` would complete the artifact coverage.

- No false claims about the economics. Berry inversion formula, logit share formula, FOC statement, ownership structure, instrument exclusion, and IIA remark are all grounded verbatim in code with correct implementations.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20%.** Below the 50% halt threshold. Proceed.

1. **Finding 1 -- notation fix (low priority, cosmetic).**
   Turn violated invariant into a pytest: confirm `compute_share_derivatives.__doc__` currently says "ds_j/dp_k". Confirm the honest-fix condition fails (it does). Fix: either (a) update the docstring to `ds_k/dp_j` to match the README convention, or (b) update the README equation to `\Omega_{jk} = -O_{jk} \frac{\partial s_j}{\partial p_k}` to match the code. Either is consistent. Option (b) is safer because the code is correct and the FOC derivation in README uses the `ds_k/dp_j` convention consistently -- updating the README equation to match code would require also updating the FOC display. Recommend option (a): update docstring to match README.

2. **Finding 2 -- commit a cost-recovery CSV (low priority, audit hygiene).**
   Add a `tables/cost-recovery-market0.csv` with columns `product_name, price, markup, est_mc, true_mc, abs_error` and a footer row with the mean. Export it from `run.py` via `report.add_table(...)`. This makes the 0.455 figure independently verifiable without a re-run.

3. After fixes: re-run this skill. Expected new score: 0-10% (all HOLDS).
