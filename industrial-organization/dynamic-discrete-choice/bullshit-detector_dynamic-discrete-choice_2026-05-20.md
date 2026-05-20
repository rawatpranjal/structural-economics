# bullshit-detector — dynamic-discrete-choice — 2026-05-20

**Bullshit score: 25%** — One DILUTED/MED finding: the tutorial title and prose assert a
numerical MCE-IRL equivalence ("returns the same theta to within solver tolerance") but
no MCE-IRL estimator is implemented, no theta_IRL column appears in the results table,
and the numerical claim cannot be verified from committed artifacts without a re-run.
All other claims HOLD against code and CSV artifacts.

## Header

- Claim sources: `industrial-organization/dynamic-discrete-choice/README.md`
- Code / artifact root: `industrial-organization/dynamic-discrete-choice/run.py`
- Data artifacts: `tables/parameter-estimates.csv`, `tables/simulation-moments.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | u(x,1)=0 (replacement flow) | HOLDS | - | no |
| 2 | u(x,0)=theta_0+theta_1*x, theta_1<0 | HOLDS | - | no |
| 3 | Bellman uses logsumexp+gamma inclusive value | HOLDS | - | no |
| 4 | p_replace = logit over conditional values | HOLDS | - | no |
| 5 | Full-solution likelihood = panel binary logit | HOLDS | - | no |
| 6 | CCP first stage = logit in mileage and mileage-squared | HOLDS | - | no |
| 7 | HM ex ante value solves linear system W=u_bar+beta*F_hat*W | HOLDS | - | no |
| 8 | HM model-implied prob = Lambda(beta*F1*W - theta_0 - theta_1*x - beta*F0*W) | HOLDS | - | no |
| 9 | MPEC: theta and v joint; Bellman eq constraints | HOLDS | - | no |
| 10 | MCE-IRL "returns same theta as NFXP to within solver tolerance" | DILUTED | MED | no (pedagogical claim) |
| 11 | Mileage states = 61 | HOLDS | - | no |
| 12 | beta=0.9, theta_true=(2.00,-0.15), buses=1500, periods=35 | HOLDS | - | no |
| 13 | Repair rate = 0.253181 | HOLDS | - | no |
| 14 | Average mileage = 2.21011 | HOLDS | - | no |
| 15 | Share mileage>=10 = 0.00293333 (prose: 0.29%) | HOLDS | - | no |
| 16 | VFI iterations = 228 | HOLDS | - | no |
| 17 | Full ML: theta_0=2.01812, theta_1=-0.15346 | HOLDS | - | no |
| 18 | CCP: theta_0=2.01767, theta_1=-0.15334 | HOLDS | - | no |
| 19 | MPEC: theta_0=2.01808, theta_1=-0.15346 | HOLDS | - | no |
| 20 | MPEC iterations=4, Bellman residual=1.89644e-11 | HOLDS | - | no |
| 21 | Replacement resets to low-mileage transition | HOLDS | - | no |

## Findings

### Finding 1: MCE-IRL "returns the same theta to within solver tolerance"

- **Claim source (verbatim):** "the soft-Bellman equations and the MCE-IRL objective
  coincide algebraically with NFXP. The likelihood is identical and the estimator
  returns the same $\theta$ to within solver tolerance." — `README.md:148-149`

  Also, from the pseudocode block:
  "Output: reward parameters theta_IRL == theta_NFXP up to solver tolerance"
  — `README.md:153`

- **Code evidence (verbatim):**
  ```python
  # All functions in run.py (lines 27-284):
  def build_transition_matrices(...): ...
  def solve_ddc(...): ...
  def draw_next_state(...): ...
  def simulate_buses(...): ...
  def panel_log_likelihood(...): ...
  def estimate_full_solution(...): ...
  def estimate_first_stage_logit(...): ...
  def hotz_miller_ccp(...): ...
  def estimate_ccp(...): ...
  def bellman_residual(...): ...
  def pack_mpec(...): ...
  def unpack_mpec(...): ...
  def estimate_mpec(...): ...
  def main(): ...
  ```
  `run.py:27-284`

  No `estimate_mce_irl` function or any function whose body executes the
  MCE-IRL estimation exists anywhere in the file. The string "mce" and "irl"
  appear only in report prose strings (lines 324, 479-499, 641-644), not in
  any computational function.

- **Data evidence:** `tables/parameter-estimates.csv` (read in full, 3 rows) has columns
  `Parameter, True, Full-solution ML, Full ML error, CCP, CCP error, MPEC, MPEC error`.
  No `MCE-IRL` or `theta_IRL` column present. The claim "theta_IRL == theta_NFXP up to
  solver tolerance" has no corresponding data row in any committed artifact.
  **needs re-run to verify** the numerical equivalence claim.

- **Category:** DILUTED

- **Severity:** MED

  The prose self-hedges at `README.md:162`: "The point of including MCE-IRL here is
  not a new estimator." This hedge partially acknowledges the gap. However, the tutorial
  title (`README.md:1`) lists "the MCE-IRL Equivalence" alongside NFXP, CCP, and MPEC
  as four parallel topics, and the prose asserts a *quantitative* claim ("returns the
  same theta to within solver tolerance") that is never demonstrated. A reader expecting
  a verification column in the results table will find none.

  Doubt between DILUTED and MISLABELED: the code does not *mislabel* anything — there is
  no function called `estimate_mce_irl` that does something else. The gap is that the
  prose promises a numerical demonstration that the code simply does not run. DILUTED is
  the correct category.

- **Result-changing:** no — the three actual estimators (NFXP, CCP, MPEC) and all
  their numbers are correctly implemented and verified. The missing MCE-IRL run is a
  pedagogical gap, not an error in a published number.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not any("estimate_mce_irl" in fn or "mce_irl" in fn for fn in dir(__import__("importlib").import_module("run")))
  # PASSES on current code (no MCE-IRL fn); FAILS if an honest fix adds one
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "MCE-IRL" in pd.read_csv("tables/parameter-estimates.csv").columns
  # PASSES on honest fix (theta_IRL column present); FAILS on current code
  ```

## Cross-cutting patterns

- All 21 numeric claims in `README.md` are consistent with committed CSV artifacts to
  display precision. The code-to-README pipeline (auto-generated report) ensures tight
  coupling between computed values and displayed numbers.
- The only gap is the MCE-IRL claim: the tutorial uses a pedagogical equivalence
  assertion as if it were a numerical result. This pattern (prose claim not backed by
  a code path or data column) is isolated to the MCE-IRL section. No other section
  shows this pattern.
- The tutorial title advertises MCE-IRL as a 4th topic coequal with three real
  estimators. The self-hedge is buried in the middle of the Solution Method section
  (`README.md:162`). A future reader skimming the title and Results table will miss it.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** The substance of NFXP, CCP, and MPEC is correct and
   result-safe. Proceed with the fix, but surface the MCE-IRL gap to the user before
   adding a new estimator (it requires algebraic verification, not just a code stub).

1. Turn the violated invariant into a pytest test:
   ```python
   # tests/test_ddc.py
   import importlib, sys, pathlib
   sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
   def test_no_mce_irl_estimator():
       import industrial_organization.dynamic_discrete_choice.run as run
       assert not hasattr(run, "estimate_mce_irl")
   # This PASSES on current code (proves no implementation)
   ```

2. Convert the honest-fix pass condition into a second test:
   ```python
   import pandas as pd
   def test_mce_irl_column_in_results():
       df = pd.read_csv("industrial-organization/dynamic-discrete-choice/tables/parameter-estimates.csv")
       assert "MCE-IRL" in df.columns
   # This FAILS on current code (column absent)
   ```

3. Two honest fixes are available. Pick one and surface to user:
   - **Numerical demonstration fix:** add `estimate_mce_irl()` that calls `estimate_full_solution`
     (same code, different label), adds a `theta_IRL` column to the results table, and
     asserts `np.allclose(theta_irl, theta_nfxp, atol=1e-4)` before writing the report.
   - **Prose-only fix:** drop the quantitative phrasing ("returns the same theta to within
     solver tolerance") and replace with a purely algebraic statement; revise the title to
     remove "MCE-IRL Equivalence" or rename it "MCE-IRL Interpretation"; no new code needed.

4. After the chosen fix, re-run `python run.py`, verify `tables/parameter-estimates.csv`
   reflects the change, and re-run this skill to confirm Finding 1 reads HOLDS and the
   new bullshit score is 0-10%.
