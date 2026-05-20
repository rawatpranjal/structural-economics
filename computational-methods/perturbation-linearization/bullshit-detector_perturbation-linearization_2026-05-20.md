# bullshit-detector -- perturbation-linearization -- 2026-05-20

**Bullshit score: 20%** -- One MISLABELED finding (MED severity): algorithm pseudocode step 1 claims the code differentiates F at the steady state; the code hardcodes the Taylor coefficients directly. All numeric results and equations hold.

## Header

- Claim sources: `computational-methods/perturbation-linearization/README.md`
- Code / artifact root: `computational-methods/perturbation-linearization/run.py`
- Data artifacts: `computational-methods/perturbation-linearization/tables/approximation-errors.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Exact transition F(x) = rho*x + gamma*x^2 - eta*x^3 + kappa*x^4 | HOLDS | -- | no |
| 2 | Taylor order 1 = rho*x; order 2 adds gamma*x^2; order 3 adds -eta*x^3 | HOLDS | -- | no |
| 3 | Algorithm step 1: "Differentiate F at x_bar and keep terms through order n" | MISLABELED | MED | no |
| 4 | Shock size 0.18 in either direction; IRF periods 28 | HOLDS | -- | no |
| 5 | kappa term (0.35) left out by third-order perturbation | HOLDS | -- | no |
| 6 | All 24 README table cells match tables/approximation-errors.csv | HOLDS | -- | no |
| 7 | First-order map is symmetric; positive and negative responses cancel | HOLDS | -- | no |
| 8 | Asymmetry = positive path + negative path; nonzero sum measures nonlinearity | HOLDS | -- | no |
| 9 | Higher orders lower error near steady state | HOLDS | -- | no |
| 10 | local domain abs(x) <= 0.20; wide domain abs(x) <= 0.60 | HOLDS | -- | no |

## Findings

### Finding 1: Algorithm pseudocode claims differentiation; code hardcodes coefficients

- **Claim source (verbatim):** "1. Differentiate F at x_bar and keep terms through order n" -- `README.md:64`

- **Code evidence (verbatim):**
  ```python
  def perturbation_transition(x: np.ndarray | float, order: int) -> np.ndarray | float:
      """Taylor approximation to the transition rule around x=0."""
      z = np.asarray(x)
      out = RHO * z
      if order >= 2:
          out = out + GAMMA * z**2
      if order >= 3:
          out = out - ETA * z**3
      return out
  ```
  `run.py:34-43`

- **Data evidence (if applicable):** Not applicable. The numerical results in `tables/approximation-errors.csv` are correct for the hardcoded parameter values; the mismatch is in method description only.

- **Category:** MISLABELED -- the code does something coherent (manually reconstructs the correct Taylor polynomial using known parameters) but the pseudocode describes a different operation (symbolic or numerical differentiation of F). The mathematical outcome is identical because the Taylor coefficients of `F(x) = rho*x + gamma*x^2 - eta*x^3 + kappa*x^4` at x=0 are exactly `rho`, `gamma`, `-eta`, `kappa` -- which is what the code uses. No general differentiation step exists anywhere in the codebase.

- **Severity:** MED -- the description misleads a reader about what the code does, but no published number changes.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not any(kw in inspect.getsource(perturbation_transition) for kw in ["diff", "derivative", "grad", "np.gradient", "sympy"])
  # PASSES on current code (proves no differentiation); would FAIL only if a genuine diff step were added
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "Differentiate" not in open("README.md").read() or "hardcoded" in open("README.md").read()
  # PASSES after the pseudocode is corrected to say the Taylor coefficients are read off directly
  ```

## Cross-cutting patterns

- Only one non-HOLDS finding. It is a description gap, not a math gap. The correct Taylor coefficients (rho, 2*gamma/2=gamma, -6*eta/6=-eta) happen to equal the parameter values, which is why the hardcoding works and the results are exact. A reader who knows perturbation theory will not be deceived; a reader who does not may incorrectly conclude the code contains a differentiation step they could generalize.
- No parametric leak, no missing formula, no stale numbers. All 24 numeric cells in the results table match the on-disk CSV exactly (verified by direct cross-check).

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20% (below 50%).** No halt required. Surface the single finding to the author as a documentation fix.

1. The violated-invariant assertion already passes on current code (no diff step exists). No pytest test needed to prove the bug.

2. Fix: update `README.md` Solution Method pseudocode step 1 from "Differentiate F at x_bar and keep terms through order n" to "Read the Taylor coefficients of F directly from the known parameter values (rho, gamma, eta) and construct the order-n polynomial."

3. No code change required. No re-run required. The fix is documentation only.

4. After fix: re-run `scripts/validate_catalog.py` to confirm no math-rendering regressions from the prose edit.

5. Re-run this skill after the fix to confirm the finding now reads HOLDS and the bullshit score drops to 0-10%.
