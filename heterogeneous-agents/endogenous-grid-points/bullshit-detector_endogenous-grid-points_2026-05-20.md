# bullshit-detector — endogenous-grid-points — 2026-05-20

**Bullshit score: 25%** — The perfect-foresight MPC limit formula `R*(beta*R)^(-1/gamma) - 1` is not the Carroll (1997/2006) formula; it has G_c and R swapped in the denominator, producing 0.0413 instead of 0.0396. This is a MISLABELED / FALSE formula in a named citation context, reported in the summary table and marked as a reference line in the figure. All EGP algorithm steps, calibration parameters, and simulation statistics HOLD.

## Header
- Claim sources: `heterogeneous-agents/endogenous-grid-points/README.md` (prose, Equations section, Model Setup table, Results narrative, summary table)
- Code / artifact root: `heterogeneous-agents/endogenous-grid-points/run.py`
- Data artifacts: `heterogeneous-agents/endogenous-grid-points/tables/summary-statistics.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | kappa* = R*(beta*R)^{-1/gamma} - 1 is the PF-MPC limit | FALSE | MED | no (reference line only; main EGP result unaffected) |
| 2 | beta*R < 1 rules out unbounded saving per Carroll (1997) | DILUTED | LOW | no |
| 3 | Average local MPC label | MISLABELED | LOW | no |
| 4 | EGP algorithm (all four steps) | HOLDS | — | — |
| 5 | All calibration parameters | HOLDS | — | — |
| 6 | All table values (mean assets, Gini, large MPC, fraction constrained, gaps) | HOLDS | — | — |
| 7 | IID income implementation | HOLDS | — | — |
| 8 | Simulation setup (50k agents, 550 periods, seed 2020) | HOLDS | — | — |

## Findings

### Finding 1: Perfect-foresight MPC limit formula is wrong

- **Claim source (verbatim):** "The dotted line marks the perfect-foresight limit, $\kappa^{\ast}\approx0.041$. Here $\kappa^{\ast}=R(\beta R)^{-1/\gamma}-1$ is the perfect-foresight MPC limit for CRRA utility." — `README.md:140`

  Also in summary table: "Perfect-foresight MPC limit | 0.0413" — `README.md:158` and `tables/summary-statistics.csv:10`

- **Code evidence (verbatim):**
  ```python
  mpclim = gross_return * (beta_r ** (-1.0 / gamma)) - 1.0
  ```
  `run.py:335`

- **Data evidence:** `tables/summary-statistics.csv:10` reads `Perfect-foresight MPC limit,0.0413`, consistent with the code formula. README narrative and table agree internally. All three are internally consistent but all three use the wrong formula.

- **Category:** FALSE — the formula `R*(beta*R)^{-1/gamma} - 1` equals `R/G_c - 1` where `G_c = (beta*R)^{1/gamma}`. The Carroll (1997/2006) perfect-foresight MPC for infinite-horizon CRRA is `kappa* = 1 - G_c/R = 1 - (beta*R)^{1/gamma}/R`. The code has `G_c` and `R` swapped in the denominator. At this calibration, `G_c = 0.9892`, `R = 1.03`; the formula `R/G_c - 1 = 0.0413` vs correct `1 - G_c/R = 0.0396`. The discrepancy is 0.0016 absolute, 4.1% relative.

- **Severity:** MED — the formula is wrong and is named in the README prose with an explicit citation to Carroll (1997). It appears in the summary table and is plotted as a dotted reference line in `figures/mpc-distribution.png`. A reader who copies the formula will reproduce the wrong benchmark.

- **Result-changing:** no — the value 0.0413 is a reference benchmark only; it does not feed back into any EGP iteration, simulation, or Gini calculation. The main economic outputs (policy, wealth distribution, large MPC) are unaffected.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(1.03 * (0.9785 ** (-0.5)) - 1 - (1 - 0.9785 ** 0.5 / 1.03)) > 1e-4
  # PASSES on current code (formula differs from Carroll by ~0.0016); FAILS if formula is corrected
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs((1.03 * (0.9785 ** (-0.5)) - 1) - (1 - 0.9785 ** 0.5 / 1.03)) < 1e-10
  # FAILS on current code; PASSES only when mpclim uses 1 - (beta_r ** (1/gamma)) / gross_return
  ```

---

### Finding 2: Carroll (1997) patience condition is stated as beta*R < 1, not the growth-impatience condition

- **Claim source (verbatim):** "Patience-return product $\beta R$ | 0.9785 | $<1$ rules out the unbounded-saving target of Carroll (1997)" — `README.md:57`

- **Code evidence (verbatim):**
  ```python
  beta_r = beta * gross_return
  ```
  `run.py:207`
  The code computes `beta_r = 0.9785` and this is placed in the Model Setup table with the description above. No code references the growth-impatience condition.

- **Data evidence:** Not applicable (architectural/theoretical claim).

- **Category:** DILUTED — Carroll (1997) uses the growth-impatience condition `G_c < R`, i.e., `(beta*R)^{1/gamma} < R`, which for `gamma > 1` is weaker than `beta*R < 1`. The README conflates a sufficient condition (`beta*R < 1`) with the necessary-and-sufficient condition Carroll uses. At `gamma = 2`, the Carroll condition is `beta*R < R^2 = 1.0609`, which is satisfied even for some `beta*R > 1` values. The claim is logically valid as a sufficient condition (`beta*R < 1 => G_c < 1 < R for R > 1`) but misrepresents what Carroll (1997) actually invokes.

- **Severity:** LOW — the sufficient condition holds and the conclusion (finite target wealth) is correct. No published number is wrong.

- **Result-changing:** no.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "beta*R < R**gamma" not in readme_text  # Carroll's exact condition absent from README
  # PASSES on current README; FAILS after an honest fix that states the growth-impatience condition
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "(beta_r ** (1/gamma)) < gross_return" in description_string
  # PASSES on honest fix (states G_c < R); FAILS on current README text
  ```

---

### Finding 3: "Average local MPC" label does not explain computation

- **Claim source (verbatim):** "Average local MPC | 0.252" — `README.md:157` and `tables/summary-statistics.csv:5`

- **Code evidence (verbatim):**
  ```python
  transfer_small = 1.0e-6
  ...
  mpc_small[mask] = (
      con_interp[iy](assets_iy + transfer_small) - con_interp[iy](assets_iy)
  ) / transfer_small
  ```
  `run.py:316-326`

- **Data evidence:** `tables/summary-statistics.csv:5` row `Average local MPC,0.252` matches code computation of `mean(mpc_small)` with `transfer_small = 1e-6`.

- **Category:** MISLABELED — "Average local MPC" suggests an MPC with respect to an income transfer. The code computes a numerical wealth derivative `dc/da` evaluated at a wealth increment of `1e-6`. This is the slope of the consumption function, not the response to an income shock. Nowhere in the README is the computation explained or the transfer size disclosed.

- **Severity:** LOW — the value is economically interpretable (dc/da at the current wealth level), but a reader who expects "local MPC" to mean the MPC wrt a small income shock (or who wants to replicate the computation) has no information from the README to identify what "local" means.

- **Result-changing:** no — the value is internally consistent (code, CSV, README all agree at 0.252), and the large MPC (0.228) is the primary reported quantity.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "1e-6" not in open("README.md").read() and "transfer_small" not in open("README.md").read()
  # PASSES on current README (1e-6 not disclosed); FAILS after an honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "1e-6" in open("README.md").read() or "numerical derivative" in open("README.md").read()
  # PASSES on honest fix that discloses dc/da computation; FAILS on current README
  ```

---

## Cross-cutting patterns

- The MPC formula error (Finding 1) is the only formula-level discrepancy in the entire tutorial. All other Equations-section formulas (Bellman equation, Euler equation, EGP inversion formula, constrained branch rule, CRRA marginal utility and its inverse) are implemented exactly as written.
- All table values are runtime-generated from the same code that produces the README, so internal consistency between CSV and README is structurally guaranteed. Numeric findings would only arise from formula bugs (as in Finding 1) or simulation non-stationarity (not observed).
- Carroll (1997) appears in both the patience-condition claim (Finding 2) and the References section. The paper's exact conditions are softened in both cases (Finding 1: wrong formula attributed to PF limit; Finding 2: sufficient condition presented as Carroll's condition). A single pass over the Carroll (1997) equations would have caught both.
- The "Average local MPC" (Finding 3) is opaque to readers but not claimed to match any external source. It is a self-contained computation whose only defect is insufficient labeling.

## TDD execution sequence (for the next agent)

0. **Read the bullshit score first.** Score is 25% — ship after touch-up; no halt required.

1. **Finding 1 (FALSE, MED):** The fix is one line in `run.py:335`. Change:
   ```
   mpclim = gross_return * (beta_r ** (-1.0 / gamma)) - 1.0
   ```
   to the Carroll formula (violated invariant and pass condition above specify the exact change). After fixing, the table row "Perfect-foresight MPC limit" will read `0.0396` instead of `0.0413`, and the dotted line in `figures/mpc-distribution.png` will shift left slightly. Regenerate `README.md` and `figures/mpc-distribution.png`.

2. **Finding 2 (DILUTED, LOW):** Update the Model Setup table description string in `run.py` near line 413 to state the growth-impatience condition `G_c < R` rather than `beta*R < 1 rules out unbounded saving`.

3. **Finding 3 (MISLABELED, LOW):** Either rename "Average local MPC" to "Avg dc/da (local slope)" in the table, or add a parenthetical "(numerical derivative at transfer=1e-6)" in the table description in `run.py` near line 698.

4. Re-run `python run.py` to regenerate `README.md`, figures, and `tables/summary-statistics.csv`.

5. Re-run `scripts/validate_catalog.py` to confirm math rendering passes.

6. Re-run this skill on the updated tutorial; expected new score is 0-10%.
