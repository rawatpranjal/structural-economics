# bullshit-detector — adversarial-estimation — 2026-05-20

**Bullshit score: 25%** — one prose claim ("all Monte Carlo numbers slightly below the asymptotic prediction") is false for the neural-net row, where the Monte Carlo sd (2.562) exceeds the asymptotic value (2.449) by 4.6%; all other claims hold against code and data.

## Header
- Claim sources: `structural-econometrics/adversarial-estimation/README.md` (prose, Equations, Results, Model Setup table)
- Code / artifact root: `structural-econometrics/adversarial-estimation/run.py`
- Data artifacts: `tables/standard-errors.csv`, `tables/smm-comparison.csv`, `tables/bootstrap-se.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | All MC numbers slightly below asymptotic prediction | FALSE | MED | no (tables show the real number; only the prose is wrong) |
| 2 | Oracle D* formula | HOLDS | - | - |
| 3 | Common-random-numbers trick (shocks reused at every theta) | HOLDS | - | - |
| 4 | MLP architecture: one hidden layer, H tanh units, L2 on W and c | HOLDS | - | - |
| 5 | Logistic discriminator inner problem convex, L-BFGS | HOLDS | - | - |
| 6 | Outer grid 25 points, parabolic refinement | HOLDS | - | - |
| 7 | SMM d=7/d=3 standard error ratio "roughly six" | HOLDS | - | - |
| 8 | Asymptotic sd values (MLE 1.732, adversarial 2.449) | HOLDS | - | - |
| 9 | Bootstrap: joint resample of real and shock vectors | HOLDS | - | - |
| 10 | Table numbers (all three CSVs match README tables) | HOLDS | - | - |

## Findings

### Finding 1: "All Monte Carlo numbers are slightly below the asymptotic prediction"

- **Claim source (verbatim):** "All Monte Carlo numbers are slightly below the asymptotic prediction at this sample size, which is the usual finite-sample behavior." — `README.md:202`
- **Code evidence (verbatim):**
  ```python
  rows = [
      ("MLE (reference)", cheap["mle"], THEORETICAL_SE_MLE),
      ("Oracle adversarial", cheap["oracle"], THEORETICAL_SE_ADV),
      ...
  ]
  if mlp is not None:
      rows.append(("Neural net disc.", mlp["mlp"], THEORETICAL_SE_ADV))
  ```
  `run.py:513-521`
- **Data evidence (verbatim):** From `tables/standard-errors.csv`:
  ```
  Neural net disc.,-0.004,2.562,2.449,2.541
  ```
  Neural net Monte Carlo sd × sqrt(n) = 2.562; Asymptotic sd × sqrt(n) = 2.449. The Monte Carlo value exceeds the asymptotic value by 0.113 (4.6%).
  By contrast: MLE 1.572 < 1.732 (below); Oracle 2.173 < 2.449 (below). Only neural net violates the direction.
- **Category:** FALSE
- **Severity:** MED
- **Result-changing:** no — the correct numbers appear in the table; the prose misdescribes the direction for one estimator only. A reader who trusts the prose over the table will misunderstand the finite-sample behavior of the neural-net discriminator, but the tables themselves are accurate.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert 2.562 < 2.449  # PASSES on current data (claim says below); actually false — this assertion FAILS, proving the bug
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "Neural net disc." not in prose_claiming_all_mc_below or neural_mc_sd <= neural_asymp_sd
  # Equivalent: fix the prose to say "most" or "MLE and oracle" instead of "all"
  ```

  Reformulated as directly testable:
  ```python
  assert float(neural_mc_sd) <= float(neural_asymp_sd)  # FAILS on current data (2.562 > 2.449); PASSES after prose fix acknowledges the exception
  ```

## Cross-cutting patterns

- Every other numeric claim (asymptotic standard deviations, table values, ratios, grid parameters, bootstrap B) is grounded in committed code constants and matches the CSV artifacts exactly. This tutorial has unusually tight claim-code-data consistency except for the single prose over-generalization in Finding 1.
- The over-generalization ("all") in the Results prose is the only dilution marker. The underlying table is correct; the verbal summary overstates the regularity. The fix is purely editorial: replace "All Monte Carlo numbers" with "The MLE and oracle Monte Carlo numbers" or add an explicit exception clause for the neural-net result.
- No parametric-access leaks, no mislabeled method classes, no unimplemented algorithms found. The oracle discriminator, logistic discriminator, MLP, SMM, and bootstrap are all faithfully implemented relative to their prose descriptions.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Below the 50% halt threshold. Proceed with fix.
1. Turn the violated invariant into a pytest test:
   ```python
   # tests/test_adversarial_estimation_claims.py
   import pandas as pd
   df = pd.read_csv("structural-econometrics/adversarial-estimation/tables/standard-errors.csv")
   neural_row = df[df["Estimator"] == "Neural net disc."].iloc[0]
   def test_neural_mc_below_asymptotic():
       # This test PASSES on current data, proving the prose claim is wrong
       assert neural_row["Monte Carlo sd $\\times \\sqrt{n}$"] < neural_row["Asymptotic sd $\\times \\sqrt{n}$"]
   ```
2. Fix the prose in `run.py` at `report.add_results(...)` (lines 872-878): change "All Monte Carlo numbers are slightly below the asymptotic prediction" to accurately describe that MLE and oracle are below, while the neural-net discriminator is slightly above due to the non-convex inner problem and limited iterations.
3. Regenerate `README.md` via `python run.py` and confirm the updated prose is correct.
4. After the fix, the test in step 1 should FAIL (the prose no longer makes the false claim) and a new test asserting the corrected prose condition should PASS.
5. Re-run this skill to confirm Finding 1 now reads HOLDS and the bullshit score drops to 0-10%.
