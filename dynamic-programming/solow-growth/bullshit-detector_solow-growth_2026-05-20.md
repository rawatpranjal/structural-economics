# bullshit-detector — solow-growth — 2026-05-20

**Bullshit score: 20%** — one DILUTED finding (MED severity): the prose sentence "the geometric residual is about 2.21e-04" is presented as explaining the terminal gap, but the actual gap (2.73e-04, in the table immediately above) exceeds it by 23% due to nonlinearity at the far-from-steady-state start. All other claims hold exactly.

## Header
- Claim sources: `dynamic-programming/solow-growth/README.md`
- Code / artifact root: `dynamic-programming/solow-growth/run.py`
- Data artifacts: `dynamic-programming/solow-growth/tables/steady-state-comparison.csv`
- Seed audit: none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Delta = 0.0902 | HOLDS | — | no |
| 2 | k* = 4.3086 (setup table) | HOLDS | — | no |
| 3 | lambda approx 0.941, half-life approx 11.5 periods | HOLDS | — | no |
| 4 | Terminal gap: k within 2.73e-04 of k* at period 159 | HOLDS | — | no |
| 5 | Table values match CSV | HOLDS | — | no |
| 6 | Comparative statics k* in {2.80, 4.31, 6.01} | HOLDS | — | no |
| 7 | "Geometric residual is about 2.21e-04" explains the terminal gap | DILUTED | MED | no (table above shows correct gap 2.73e-04; reader confusion, not numeric error) |
| 8 | No Bellman equation; model is scalar map phi | HOLDS | — | no |
| 9 | Output and consumption move together because c_t=(1-s)y_t | HOLDS | — | no |
| 10 | Law of motion phi(k) formula matches code | HOLDS | — | no |

## Findings

### Finding 1: "geometric residual is about 2.21e-04" conflated with actual terminal gap

- **Claim source (verbatim):** "The table compares the closed form with the terminal simulation. Any gap comes from finite horizon truncation. The geometric residual is about 2.21e-04." — `README.md:114`

- **Code evidence (verbatim):**
  ```python
  report.add_results(
      "The table compares the closed form with the terminal simulation. Any gap "
      "comes from finite horizon truncation. The geometric residual is about "
      f"{abs(k0 - k_star) * local_lambda ** (periods - 1):.2e}."
  )
  ```
  `run.py:412-415`

- **Data evidence:** CSV at `tables/steady-state-comparison.csv` row 1: `Capital per effective worker k,4.308592,4.308319,2.73e-04`. The actual absolute gap is `2.73e-04`. The geometric residual formula `|k0 - k_star| * lambda^(T-1) = |1.0 - 4.308592| * 0.941338^159 = 2.21e-04` is the linearization's prediction of the remaining gap, not the actual gap. These two quantities differ by 23% because k0=1.0 is far from k_star=4.309, placing the starting point deep in the nonlinear region where the linearization materially underestimates the true distance to steady state.

- **Category:** DILUTED — the geometric residual (2.21e-04) is a real and correctly computed quantity, but the prose presents it as the explanation for "any gap" when the actual gap displayed in the table immediately above (2.73e-04) is 23% larger. The residual does not explain the gap; it underestimates it.

- **Severity:** MED — the correct gap (2.73e-04) is already in the table and the CSV; this is a prose-vs-data consistency failure, not a missing result. A reader who compares the sentence to the table row above it will see two different numbers without explanation.

- **Result-changing:** no — the table is correct, the CSV is correct, and the computed numbers are internally consistent. Only the explanatory sentence is misleading. No published quantitative result changes.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(abs(1.0 - 4.308592) * 0.941338**159 - 2.73e-4) / 2.73e-4 < 0.05  # PASSES on buggy code: fails because ratio is 0.23
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "2.73e-04" in open("README.md").read() and "2.21e-04" not in open("README.md").read()  # or: prose distinguishes linear approx from actual gap
  ```

## Cross-cutting patterns

- Only one non-HOLDS finding, and it is a prose/explanation gap, not a code gap. The numeric chain from parameters to steady state to simulation to table is arithmetically consistent end to end (independently verified by re-running all formulas from primitives in this audit).
- The linearization (lambda, half-life) is used in two places: the Results prose (README:94, README:104) and the figure overlay (run.py:288-294). Both uses are mathematically correct. The only problem is the third use (run.py:415) where the linearization residual is presented as the terminal gap without noting the discrepancy.
- The tutorial correctly notes "There is no Bellman equation here" (README:65, run.py:220). Dynamic-programming folder placement is a catalog-organization choice; the claim itself is accurate and the code confirms it.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20% (below 50%).** Safe to proceed without halting.

1. Turn the violated invariant into a pytest test:
   ```python
   # tests/test_solow_growth.py
   def test_geometric_residual_matches_actual_gap():
       # Currently PASSES (proves the discrepancy is real)
       import numpy as np
       alpha, s, delta, n, g = 0.33, 0.24, 0.06, 0.01, 0.02
       gross_dil = (1+g)*(1+n)
       eff_dep = gross_dil - 1 + delta
       k_star = (s/eff_dep)**(1/(1-alpha))
       lam = ((1-delta) + s*alpha*k_star**(alpha-1)) / gross_dil
       k0, periods = 1.0, 160
       geom_res = abs(k0 - k_star) * lam**(periods-1)
       k = k0
       for _ in range(periods-1):
           k = ((1-delta)*k + s*k**alpha) / gross_dil
       actual_gap = abs(k - k_star)
       assert abs(geom_res - actual_gap) / actual_gap > 0.05  # >5% discrepancy exists
   ```

2. Honest-fix pass condition test:
   ```python
   def test_prose_uses_actual_gap_not_linear_residual():
       # Currently FAILS (prose says 2.21e-04); passes after fix
       readme = open("dynamic-programming/solow-growth/README.md").read()
       assert "2.73e-04" in readme  # actual gap appears in prose context
       # and the linearization residual is described as approximation, not "the gap"
   ```

3. Fix: in `run.py:412-415`, either (a) remove the geometric-residual sentence and let the table speak for itself, or (b) rewrite to distinguish the two quantities: "The linear approximation predicts a remaining gap of 2.21e-04; the actual gap is 2.73e-04, larger because k0=1.0 starts far from k_star."

4. After fix, regenerate README.md (`python run.py`). Re-run this skill to confirm score drops to 0-10%.
