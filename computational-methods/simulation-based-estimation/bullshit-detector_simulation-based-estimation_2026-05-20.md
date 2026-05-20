# bullshit-detector — simulation-based-estimation — 2026-05-20

**Bullshit score: 20%** — Two findings: one MISLABELED (MED, confusing residual_table call pattern — output numerically correct), one DILUTED (MED, criterion "same scale" claim fails for MSM-vs-II comparison). No FALSE, no UNIMPLEMENTED. Results table numbers match CSV artifacts exactly. Score anchors at 20% for one DILUTED at MED that misleads a reader comparing criterion values across methods.

## Header
- Claim sources: `computational-methods/simulation-based-estimation/README.md`
- Code / artifact root: `computational-methods/simulation-based-estimation/run.py`
- Data artifacts: `tables/parameter-recovery.csv`, `tables/method-comparison.csv`, `tables/msm-residuals.csv`, `tables/indirect-inference-residuals.csv`, `tables/abc-summary.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Simulator, equations, structural model | HOLDS | — | no |
| 2 | MSM 5 moments, W_m scaling | HOLDS | — | no |
| 3 | II 6 auxiliary stats | HOLDS | — | no |
| 4 | ABC-SMC adaptive tolerance (alpha-quantile) | HOLDS | — | no |
| 5 | Round-0 samples ceil(N/alpha) from prior | HOLDS | — | no |
| 6 | Kernel covariance = 2 * weighted Cov | HOLDS | — | no |
| 7 | Uniform-prior weight simplification | HOLDS | — | no |
| 8 | All numeric Results table values match CSVs | HOLDS | — | no |
| 9 | residual_table called with (ii, ii) for II table | MISLABELED | MED | no (output correct) |
| 10 | "criterion at posterior mean on same scale as MSM and II" | DILUTED | MED | yes (reader misled comparing 0.00062 vs 0.00017) |

## Findings

### Finding 1: residual_table called with (ii, ii) for the II residuals table

- **Claim source (verbatim):** "Parameter estimates and residuals give a compact diagnostic. MSM and indirect inference return point estimates; ABC-SMC returns a posterior whose mean is reported alongside a 90% credible interval. Small scaled residuals show that each target vector is matched closely." — `README.md:216`

- **Code evidence (verbatim):**
  ```python
  report.add_table(
      "tables/indirect-inference-residuals.csv",
      "Indirect-inference auxiliary residuals",
      residual_table(
          ["LPM intercept", "LPM slope", "Mean log wage", "SD log wage", "Acceptance rate", "Mean accepted log wage"],
          target_aux,
          ii,
          ii,
      ).query("Estimator == 'MSM'").drop(columns=["Estimator"]).round(5),
  )
  ```
  `run.py:829-838`

  ```python
  def residual_table(names, target, msm, ii):
      rows = []
      for estimator, result in [("MSM", msm), ("Indirect inference", ii)]:
          ...
  ```
  `run.py:369-385`

- **Data evidence:** `tables/indirect-inference-residuals.csv` contains correct II residuals (e.g., LPM intercept observed=-1.75243, simulated=-1.74529, scaled=0.00408). These match `ii["simulated_stats"]` and `ii["residual"]` values as produced by `estimate_by_simulation(target_aux, ...)`. The function call with `(ii, ii)` happens to produce the correct numbers because the query `"Estimator == 'MSM'"` extracts the first loop iteration which maps to the first positional arg (here `ii`, not `msm`).

- **Category:** MISLABELED — the function signature names the second positional arg `msm` and the third `ii`. Passing `(ii, ii)` makes the internal loop label `ii`'s results as `"MSM"`. The `.query("Estimator == 'MSM'")` then correctly extracts `ii` data. The output CSV is numerically correct but the implementation mislabels ii as MSM inside the intermediate DataFrame.

- **Severity:** MED — output is numerically correct; mislabeling is internal to the construction pipeline. A future refactor that passes a real `msm` result as the second argument (the natural reading of the signature) would silently break the II residuals table.

- **Result-changing:** no — committed CSV artifacts match the intended ii residuals.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert residual_table(["A"], np.array([1.0]), ii_result, ii_result).iloc[0]["Estimator"] == "MSM"
  # PASSES on current code (ii is mislabeled MSM); FAILS on honest fix (first row would be labeled correctly)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert residual_table(["A"], np.array([1.0]), msm_result, ii_result).query("Estimator == 'Indirect inference'").iloc[0]["Simulated at estimate"] == ii_result["simulated_stats"][0]
  # PASSES on honest fix (ii results under correct label); FAILS on current call pattern (ii not passed as ii)
  ```

---

### Finding 2: Criterion values claimed "on the same scale" for all three methods

- **Claim source (verbatim):** "ABC-SMC reports the total number of simulator calls across all rounds and the same criterion evaluated at the posterior mean, which is on the same scale as the MSM and II numbers." — `README.md:247`

- **Code evidence (verbatim):**
  ```python
  # MSM criterion — scaled by target_moments-based scale, 5 economic moments
  scale = np.maximum(np.abs(target), 0.1)   # target = target_moments
  diff = (statistic_fn(sample) - target) / scale   # statistic_fn = economic_moments
  return float(diff @ diff)
  ```
  `run.py:87-88, 75-76` (inside `estimate_by_simulation` called with `target_moments, economic_moments`)

  ```python
  # II criterion — scaled by target_aux-based scale, 6 auxiliary stats
  scale = np.maximum(np.abs(target), 0.1)   # target = target_aux
  diff = (statistic_fn(sample) - target) / scale   # statistic_fn = auxiliary_statistics
  return float(diff @ diff)
  ```
  `run.py:87-88, 75-76` (inside `estimate_by_simulation` called with `target_aux, auxiliary_statistics`)

  ```python
  # ABC criterion at posterior mean — scaled by target_moments-based scale, 5 economic moments
  criterion_at_mean = float(simulated_distance(post_mean, target, scale, draws, choice_scale) ** 2)
  ```
  `run.py:295` where `scale = np.maximum(np.abs(target), 0.1)` at line 225, `target = target_moments`

- **Data evidence:** `method-comparison.csv` row 2: MSM criterion = 0.00062. Row 3: II criterion = 0.00017. Row 4: ABC criterion = 0.00065. The MSM and ABC criteria use the same 5 economic moments and the same scale vector → directly comparable. The II criterion uses 6 different auxiliary statistics scaled by `|target_aux|` magnitudes → different dimensionality and different scale denominators. The value 0.00017 is a sum of 6 squared scaled auxiliary residuals; 0.00062 is a sum of 5 squared scaled economic-moment residuals. They are not on the same scale.

- **Category:** DILUTED — the claim that all three criteria are "on the same scale" is true for ABC vs MSM (identical formula, identical moments, identical scale vector) but not for II vs MSM or II vs ABC. The prose asserts tripartite comparability that the math only partially supports.

- **Severity:** MED — a reader comparing 0.00017 (II) with 0.00062 (MSM) and concluding II fits "better" by a factor of ~3.6x is misled. The values reflect different normalized objectives with different denominators and different numbers of terms.

- **Result-changing:** yes — the comparison of criterion values across methods in the table (README.md:251-255) is the advertised diagnostic. Asserting "same scale" when the II criterion is not on the same scale as MSM/ABC misleads readers about relative fit quality.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert np.array_equal(np.maximum(np.abs(target_moments), 0.1), np.maximum(np.abs(target_aux), 0.1))
  # PASSES if the scales were the same (would vindicate the claim); FAILS on actual data (different targets)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "MSM and ABC criteria use the same 5-moment scale; II uses a different 6-statistic scale and is not directly comparable" in readme_text
  # Structural: the README should disclaim that II criterion is not on the same scale as MSM/ABC
  ```

---

## Cross-cutting patterns

- No parametric leaks, no data fabrication, no FALSE findings. The tutorial is honest about its model and correctly implements all three estimators.
- The `residual_table` confusion (Finding 1) and the scale-comparability overstatement (Finding 2) are both in the reporting layer, not the estimation layer. The numerical outputs committed to CSV artifacts are correct for all three methods.
- The `residual_table` signature hazard is a latent correctness bug: any caller who reads the signature and passes the natural `(msm_result, ii_result)` pair would get the wrong table for II. A future tutorial that reuses this function is at risk.
- The "same scale" language (Finding 2) appears once in Results prose and is implicit in the method-comparison table heading. Both should be corrected together.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20% — below the 50% halt threshold.** Proceed with fixes.

1. **Finding 1 — violated invariant test:**
   ```python
   # tests/test_simulation_based_estimation.py
   import importlib, sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
   import computational_methods.simulation_based_estimation.run as sbe
   # proves the mislabeling bug: calling with (ii, ii) labels ii as "MSM"
   dummy = {"simulated_stats": [1.0], "residual": [0.1]}
   df = sbe.residual_table(["X"], [1.0], dummy, dummy)
   assert df.iloc[0]["Estimator"] == "MSM"   # PASSES on current buggy code; FAILS on honest fix
   ```

2. **Finding 1 — honest-fix pass condition test:**
   ```python
   dummy_msm = {"simulated_stats": [2.0], "residual": [0.2]}
   dummy_ii  = {"simulated_stats": [3.0], "residual": [0.3]}
   df = sbe.residual_table(["X"], [1.0], dummy_msm, dummy_ii)
   assert df.query("Estimator == 'Indirect inference'").iloc[0]["Simulated at estimate"] == 3.0
   # PASSES on honest fix; FAILS if called with (ii, ii)
   ```

3. **Finding 2 — violated invariant test:**
   ```python
   import numpy as np
   target_moments_sample = np.array([0.40, 3.00, 0.45, 3.36, 0.32])
   target_aux_sample     = np.array([-1.75, 0.72, 3.00, 0.45, 0.40, 3.36])
   scale_m = np.maximum(np.abs(target_moments_sample), 0.1)
   scale_a = np.maximum(np.abs(target_aux_sample), 0.1)
   assert not np.array_equal(scale_m, scale_a[:5])  # PASSES (scales differ); FAILS if they were the same
   ```

4. Fix `run.py:835-836` to call `residual_table(..., msm, ii)` with the real `msm` result as second arg, and update the `.query` to use `"Estimator == 'Indirect inference'"`. Fix README.md:247 to disclaim that II criterion uses a different scale from MSM/ABC.

5. Rerun `python run.py` inside the tutorial folder; confirm CSVs regenerate with identical values. Re-run this skill; expected new score ≤ 10%.
