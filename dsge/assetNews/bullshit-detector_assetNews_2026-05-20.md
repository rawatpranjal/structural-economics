# bullshit-detector — assetNews — 2026-05-20

**Bullshit score: 10%** — one MISLABELED finding of LOW severity (model.mod declares `stoch_simul(order=3)` while the tutorial and every computation use first-order perturbation); all numeric claims hold to floating-point precision; diagram-only cap does not apply.

## Header
- Claim sources: `dsge/assetNews/README.md` (all sections)
- Code / artifact root: `dsge/assetNews/run.py`, `lib/perturbation.py`
- Data artifacts: `dsge/assetNews/tables/impact-responses.csv`, `dsge/assetNews/model.mod`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | p=beta/(1-beta)=99.00 | HOLDS | none | no |
| 2 | A=1.917, B=-0.009 | HOLDS | none | no |
| 3 | Euler linearization formula | HOLDS | none | no |
| 4 | A, B closed-form expressions | HOLDS | none | no |
| 5 | Klein QZ matches hand-derived coefficients | HOLDS | none | no |
| 6 | Blanchard-Kahn satisfied | HOLDS | none | no |
| 7 | Surprise IRF: x[0]=sigma2=0.1 | HOLDS | none | no |
| 8 | News IRF: n[0]=1, x[0]=0, x[1]=sigma1 | HOLDS | none | no |
| 9 | Impact table: all four rows, three columns | HOLDS | none | no |
| 10 | README table matches tables/impact-responses.csv | HOLDS | none | no |
| 11 | Component decomposition (fig2 bar chart) | HOLDS | none | no |
| 12 | B<0; good news lowers price | HOLDS | none | no |
| 13 | Nonlinear terminal condition negligible error | HOLDS | none | no |
| 14 | model.mod uses stoch_simul(order=3), tutorial claims first-order | MISLABELED | LOW | no |

## Findings

### Finding 1: model.mod declares order=3 perturbation; tutorial is first-order throughout

- **Claim source (verbatim):** "First-order perturbation turns the Euler equation into coefficient matching." — `dsge/assetNews/README.md:76`
- **Code evidence (verbatim):**
  ```text
  stoch_simul(order=3);
  ```
  `dsge/assetNews/model.mod:41`
- **Data evidence (if applicable):** run.py docstring lines 6-7: "The accompanying `model.mod` spec records a representative-agent Lucas tree ... The spec is documentation only; `run.py` does not execute it." No table row is affected.
- **Category:** MISLABELED — the `.mod` file specifies third-order perturbation; the tutorial's stated method and every executed computation are first-order. A reader running `dynare model.mod` from Octave/MATLAB would get a third-order approximation, contradicting the tutorial's exposition.
- **Severity:** LOW — `run.py` does not execute `model.mod`; no computed number is wrong.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "order=3" in open("dsge/assetNews/model.mod").read()
  # PASSES on current code (proves mismatch); FAILS if .mod is corrected to order=1
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "order=1" in open("dsge/assetNews/model.mod").read() and "order=3" not in open("dsge/assetNews/model.mod").read()
  # PASSES after fix; FAILS on current code
  ```

## Cross-cutting patterns

- All numeric claims in the README are computed inline in `run.py` via f-string interpolation at report generation time (`run.py:281-296`). There is no hard-coded number that could drift from a re-run; the README and CSV are regenerated each time `python run.py` executes.
- The sole non-HOLDS finding is confined to `model.mod`, a documentation-only artifact that is never executed. It does not propagate into any figure, table, or reported number.
- The Klein QZ cross-check (`run.py:223`) computes `qz_diff` and prints it to stdout but does not assert a threshold anywhere in `run.py`. The README asserts it "matches" without a quantitative bound. Direct execution confirms `qz_diff = 2.22e-16`; the claim holds, but the absence of an in-code assertion means future edits could silently break the match without any test failing.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10% — below the 50% halt threshold.** No need to surface to user before fixing.
1. **Finding 1 (MISLABELED):** Write a pytest test under `tests/` that asserts `"order=1"` appears and `"order=3"` does not appear in `dsge/assetNews/model.mod`. Confirm test PASSES on current code (i.e., the violated-invariant test passes — `order=3` is found), meaning the bug is real.
2. Convert the honest-fix pass condition into a second pytest test. It FAILS on current code.
3. Fix: change line 41 of `dsge/assetNews/model.mod` from `stoch_simul(order=3);` to `stoch_simul(order=1);`.
4. After fix, the violated-invariant test FAILS and the pass-condition test PASSES.
5. No re-run of `python run.py` is required; the fix touches only the documentation file and no computed artifact changes.
6. Optional: add an assert in `run.py` or a test that `qz_diff < 1e-10` to guard the Klein QZ cross-check claim against future regressions.
