# bullshit-detector — allen-arkolakis — 2026-05-20

**Bullshit score: 15%** — two low-severity findings: DATA DRIFT on `path_gap=0.304` (runtime number, no CSV artifact, needs re-run to verify); DILUTED on "center share" label (code uses `labor.max()`, harmless by symmetry). No FALSE, no UNIMPLEMENTED. All core math, gamma formulas, residuals, CES price index, and every CSV number hold.

## Header
- Claim sources: `spatial-economics/allen-arkolakis/README.md` (Overview, Equations, Model Setup, Solution Method, Results)
- Code / artifact root: `spatial-economics/allen-arkolakis/run.py`
- Data artifacts: `spatial-economics/allen-arkolakis/tables/scenario-diagnostics.csv`, `tables/trade-cost-counterfactual.csv`, `tables/parameters.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | path_gap=0.304 (agglomeration labor divergence) | DATA DRIFT | LOW | no (qualitative illustration only; no CSV artifact) |
| 2 | "center share is 15.1% / 43.6%" but code uses `labor.max()` | DILUTED | LOW | no (symmetric model guarantees max == center) |
| 3 | CES price index formula | HOLDS | — | — |
| 4 | Trade shares / gravity formula | HOLDS | — | — |
| 5 | Trade balance residual | HOLDS | — | — |
| 6 | Mobility / utility equalization residual | HOLDS | — | — |
| 7 | Wage normalization (geometric mean = 1) | HOLDS | — | — |
| 8 | gamma1/gamma2 formulas and uniqueness threshold | HOLDS | — | — |
| 9 | All CSV numbers (HHI, welfare %, kappa, utility, largest share) | HOLDS | — | — |
| 10 | Continuation description ("two steps") | HOLDS | — | — |
| 11 | Softmax parameterization and 2N-1 system size | HOLDS | — | — |

## Findings

### Finding 1: path_gap=0.304 not archived in any CSV

- **Claim source (verbatim):** "Their final labor profiles differ by as much as 0.304." — `README.md:135`
- **Code evidence (verbatim):**
  ```python
  left_final = migration["agglomeration_left"]["final_labor"]
  right_final = migration["agglomeration_right"]["final_labor"]
  path_gap = float(np.max(np.abs(left_final - right_final)))
  ```
  `run.py:809-811`
- **Data evidence:** No entry for `path_gap` in `tables/scenario-diagnostics.csv`, `tables/trade-cost-counterfactual.csv`, or `tables/parameters.csv`. Number is computed at runtime and string-formatted directly into the README at `run.py:824`. Cannot be cross-checked without re-running.
- **Category:** DATA DRIFT
- **Severity:** LOW — code computes the number correctly; the gap is absence of a durable artifact. The tutorial's main results (equilibrium wages, HHI, welfare, center shares) are all archived and verified against CSV.
- **Result-changing:** no — this number illustrates non-uniqueness qualitatively; it does not appear in any equilibrium table or policy table.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not Path("tables/convergence-path-dependence.csv").exists()
  # PASSES on current code (no archive); FAILS on honest fix (file exists)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(float(pd.read_csv("tables/convergence-path-dependence.csv")["path_gap"][0]) - 0.304) < 0.001
  # PASSES on honest fix (CSV archived); FAILS on current code (file absent)
  ```

### Finding 2: Results text says "center share" but code uses `labor.max()`

- **Claim source (verbatim):** "The dispersion-dominant center share is 15.1%. The strong-agglomeration largest share is 43.6%." — `README.md:123-124`
- **Code evidence (verbatim):**
  ```python
  report.add_results(
      f"The dispersion-dominant center share is {disp_eq.labor.max():.1%}. "
      f"The strong-agglomeration largest share is {agg_eq.labor.max():.1%}. "
  ```
  `run.py:707-710`
- **Data evidence:** `scenario-diagnostics.csv:2` column `Largest share` = `0.151`; `scenario-diagnostics.csv:3` = `0.436`. The `counterfactual_table` function at `run.py:450` uses `center = len(geo.x) // 2` = 7 (actual center index). Both `labor.max()` and `labor[7]` equal the same value here because the productivity fundamental is a symmetric Gaussian centered on location 7. But the prose uses "center share" for the first sentence and "largest share" for the second — inconsistent labeling within a single sentence.
- **Category:** DILUTED — code operationalizes "center share" as `labor.max()`, a weaker test that does not verify the peak is at the center. Holds by symmetry of the model but not by the code's own logic.
- **Severity:** LOW — result-changing only if a non-symmetric scenario were introduced; symmetric fundamentals guarantee `argmax(labor) == 7` for both scenarios run here.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "center share" in inspect.getsource(run.main) and "labor.max()" in inspect.getsource(run.main)
  # PASSES on current code (mismatch exists); FAILS on honest fix (label unified)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "center share" not in open("README.md").read().split("largest share")[0].split("Results")[1]
  # PASSES on honest fix ("center share" replaced by "largest share" in that sentence); FAILS on current code
  ```

## Cross-cutting patterns

- All three CSV tables (`parameters.csv`, `scenario-diagnostics.csv`, `trade-cost-counterfactual.csv`) cross-check arithmetically against README prose and against each other. Every derivable number (gamma ratios, welfare percentages, kappa values, HHI changes within 3-decimal rounding) is internally consistent. No external DATA DRIFT between CSVs and README text.
- The only unarchived runtime number is `path_gap=0.304` (Finding 1). All equilibrium-level results are durable.
- The prose mislabels `labor.max()` as "center share" once, then correctly labels it "largest share" in the very next clause (same sentence). Suggests copy-edit drift, not computational error.
- No FALSE or UNIMPLEMENTED findings anywhere. Core equilibrium math (CES price index, gravity shares, balanced trade, utility equalization, wage normalization) all match their README Equations section line for line.
- Continuation uses 2 bridge scenarios (`bridge1`, `bridge2`) at `run.py:480-481`, consistent with README's "two steps" description.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 15%.** Below the 50% halt threshold. Proceed.

1. **Finding 1 (DATA DRIFT):** Archive `path_gap` to a CSV (e.g., `tables/convergence-path-dependence.csv`) inside `main()` after `run.py:811`.
   - Violated-invariant test: confirm `tables/convergence-path-dependence.csv` does not exist (passes currently).
   - Pass-condition test: after fix, `pd.read_csv("tables/convergence-path-dependence.csv")["path_gap"][0]` is approximately 0.304.

2. **Finding 2 (DILUTED):** Change `"The dispersion-dominant center share is"` to `"The dispersion-dominant largest labor share is"` at `run.py:707` for label consistency.
   - Violated-invariant test: `"center share" in` the relevant `add_results` string (passes currently).
   - Pass-condition test: `"largest labor share" in` that string (fails currently).

3. Re-run `python run.py` from `spatial-economics/allen-arkolakis/`. Confirm regenerated `README.md` no longer says "center share" and `tables/convergence-path-dependence.csv` exists with `path_gap` close to 0.304.

4. Re-run `python scripts/validate_catalog.py` from repo root. No math rendering regressions expected.

5. Re-run this skill on the updated tutorial. Expected new bullshit score: 0-10%.
