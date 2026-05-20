# bullshit-detector — interpolation — 2026-05-20

**Bullshit score: 50%** — Two FALSE findings: README claims cubic spline is the lowest-error method on the smooth target in two places, but the committed `tables/comparison.csv` shows PCHIP wins on both sup-norm and L2 error on that target; the prose directly contradicts its own data artifact.

## Header
- Claim sources: `numerical-methods/interpolation/README.md`
- Code / artifact root: `numerical-methods/interpolation/run.py`, `lib/interpolate.py`
- Data artifacts: `numerical-methods/interpolation/tables/comparison.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Cubic is uniformly smallest" on smooth target | FALSE | HIGH | yes (PCHIP < cubic on both smooth metrics) |
| 2 | "Cubic spline is the lowest-error choice on the smooth target" | FALSE | HIGH | yes (PCHIP wins; wrong method recommended) |
| 3 | MPC value (0.1) omitted from Model Setup parameter table | DILUTED | LOW | no |
| 4 | "Cubic and PCHIP drop at roughly slope -4" | DILUTED | MED | needs re-run to verify |

## Findings

### Finding 1: "Cubic is uniformly smallest" on smooth target

- **Claim source (verbatim):** "On the smooth target all three errors concentrate near $W = 0$, where curvature is largest. Cubic is uniformly smallest." — `README.md:127`
- **Code evidence (verbatim):**
  ```python
  sup_lin = fits["kinked"]["Piecewise linear"]["sup_err"]
  sup_cub = fits["kinked"]["Cubic spline (natural)"]["sup_err"]
  sup_pch = fits["kinked"]["PCHIP (shape-preserving)"]["sup_err"]
  ```
  `run.py:335-337`

  The `errors_at_nodes` function computes:
  ```python
  "sup_err": float(np.max(np.abs(err))),
  "l2_err": float(np.sqrt(np.mean(err ** 2))),
  ```
  `run.py:103-104`

- **Data evidence (verbatim):**
  ```
  Piecewise linear,1.58e+00,3.95e-01,7.63e-02,1.47e-02
  Cubic spline (natural),1.09e+00,2.58e-01,4.57e-02,9.93e-03
  PCHIP (shape-preserving),7.81e-01,1.71e-01,2.90e-02,6.49e-03
  ```
  `tables/comparison.csv:2-4`

  On the smooth target: PCHIP sup-error = 7.81e-01, cubic = 1.09e+00. PCHIP L2 = 1.71e-01, cubic = 2.58e-01. PCHIP is strictly lower on both metrics. The claim "Cubic is uniformly smallest" is false on the smooth target by its own committed data artifact.

- **Category:** FALSE — the code computes errors honestly; the prose misreads the ranking of the results it computes.
- **Severity:** HIGH
- **Result-changing:** yes — the reader is told cubic is the best approximator on smooth functions when PCHIP beats it by 28% on sup-norm and 34% on L2 at N=10 nodes.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert smooth_sup["Cubic spline (natural)"] < smooth_sup["PCHIP (shape-preserving)"]
  # PASSES on current buggy prose claim; FAILS because cubic (1.09) > PCHIP (0.781)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert smooth_sup["PCHIP (shape-preserving)"] < smooth_sup["Cubic spline (natural)"]
  # PASSES on corrected prose acknowledging PCHIP wins on smooth sup-norm
  ```

---

### Finding 2: "Cubic spline is the lowest-error choice on the smooth target"

- **Claim source (verbatim):** "Cubic spline is the lowest-error choice on the smooth target; PCHIP is the lowest-error choice on the kinked one." — `README.md:145`

  Same claim in run.py:
  ```python
  "and L2 errors for each method on both targets. Cubic spline is "
  "the lowest-error choice on the smooth target; PCHIP is the "
  "lowest-error choice on the kinked one."
  ```
  `run.py:394-396`

- **Code evidence (verbatim):**
  The code computes smooth-target errors for all three methods in `errors_at_nodes` and stores them in `fits["smooth"]`. The hardcoded prose at `run.py:394-396` then makes a claim not derived from `fits["smooth"]` by any conditional check. There is no `argmin` or sorting logic in `run.py` that picks the winner — the claim is hardcoded string literal.

- **Data evidence (verbatim):**
  ```
  PCHIP (shape-preserving),7.81e-01,1.71e-01,...
  Cubic spline (natural),1.09e+00,2.58e-01,...
  ```
  `tables/comparison.csv:4,3`

  PCHIP has the lower smooth sup-error (0.781 vs 1.09) and lower smooth L2 (0.171 vs 0.258). The claim names cubic as winner on the smooth target; the data names PCHIP.

- **Category:** FALSE — the prose claim contradicts the committed CSV on its exact metric.
- **Severity:** HIGH
- **Result-changing:** yes — incorrect method recommendation in the Takeaway and the table caption; a reader following the tutorial would conclude cubic is the right default for smooth value functions when PCHIP outperforms it at this node count.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "Cubic spline is the lowest-error choice on the smooth target" in open("README.md").read()
  # PASSES on current buggy README; FAILS on honest fix that names PCHIP
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "PCHIP" in open("README.md").read().split("lowest-error choice on the smooth target")[0].split("\n")[-1]
  # PASSES when README names PCHIP as smooth-target winner; FAILS on current text
  ```

---

### Finding 3: MPC value omitted from Model Setup parameter table

- **Claim source (verbatim):** "Above the kink they save with marginal propensity to consume $\mathrm{MPC} < 1$." — `README.md:32`
- **Code evidence (verbatim):**
  ```python
  def make_consumption_policy(r=0.04, y=0.5, a_kink=0.5, mpc=0.1):
  ```
  `run.py:47`

  The Model Setup table in `README.md:63-72` lists `beta`, `smooth_domain`, `kinked_domain`, `a_kink`, `r`, `y`, `n_show`, and `node_counts`. The `mpc=0.1` parameter is absent.

- **Data evidence (if applicable):** None — the omission is documentation only; the numeric results all use `mpc=0.1` correctly.
- **Category:** DILUTED — the parameter drives the slope above the kink (`slope_above = (1+r)*mpc = 1.04*0.1 = 0.104`) and therefore the shape of the kinked target, but the error values in the CSV are consistent with `mpc=0.1` being used. The reader cannot reproduce the kinked target without knowing this value.
- **Severity:** LOW
- **Result-changing:** no — the CSV numbers are internally consistent with `mpc=0.1`; the omission affects reproducibility, not correctness of reported results.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "MPC" not in [row["Symbol"] for row in parse_model_setup_table("README.md")]
  # PASSES on current README (MPC row absent); FAILS on honest fix that adds the row
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any("0.1" in line and "MPC" in line for line in open("README.md"))
  # PASSES when README Model Setup lists MPC=0.1; FAILS currently
  ```

---

### Finding 4: "Cubic and PCHIP drop at roughly slope -4" (convergence)

- **Claim source (verbatim):** "Cubic and PCHIP drop at roughly slope $-4$." — `README.md:139`
- **Code evidence (verbatim):**
  ```python
  node_counts = np.array([5, 10, 20, 40, 80])
  convergence = {name: np.zeros(len(node_counts)) for name, *_ in METHODS}
  for i, n in enumerate(node_counts):
      for method_name, method_fn, *_ in METHODS:
          res = errors_at_nodes(V, method_fn, int(n), *smooth_domain)
          convergence[method_name][i] = res["sup_err"]
  ```
  `run.py:132-137`

  The code computes the convergence table and passes it to a log-log figure. No slope is computed or asserted in code — the slope estimate is purely a visual reading of the figure.

- **Data evidence (if applicable):** The committed CSV contains only N=10 errors, not the full convergence table. The convergence-vs-nodes figure is a binary blob (`figures/convergence-vs-nodes.png`). The slope values cannot be verified from committed text artifacts without re-running. **Needs re-run to verify.**

  Theoretical context: cubic spline on a smooth function is O(h^4) = O(N^{-4}) on sup-norm — slope -4 is plausible. PCHIP is typically O(h^3) or O(h^2) depending on smoothness and implementation; slope -4 for PCHIP on this target is not guaranteed by theory and must be verified empirically.

- **Category:** DILUTED — the slope claim for cubic is theoretically sound; the same claim for PCHIP is not guaranteed and is not derivable from committed artifacts.
- **Severity:** MED — if PCHIP does not achieve slope -4, the comparative convergence narrative changes.
- **Result-changing:** needs re-run to verify.
- **Violated invariant (one-line pytest assertion):**
  ```python
  # Run-dependent; cannot write without re-run. Placeholder:
  assert pchip_slope_estimate >= -4.5 and pchip_slope_estimate <= -3.5  # needs re-run
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(pchip_log_slope - (-4.0)) < 0.5  # PCHIP slope is near -4 on smooth target
  # needs re-run to determine if this PASSES or FAILS
  ```

---

## Cross-cutting patterns

- **Hardcoded prose, no assertion against computed results.** Findings 1 and 2 share a root cause: `run.py` computes the correct error table but the prose strings at `run.py:338-348` and `run.py:391-396` are hardcoded English claims, not derived by reading `fits["smooth"]`. A winner-selection check (e.g., `argmin` over `sup_err`) would have caught the mismatch at generation time.
- **PCHIP consistently underrated.** Both FALSE findings understate PCHIP performance on the smooth target: it beats cubic on sup-norm (28%) and L2 (34%) but is described as the runner-up. The Takeaway and the table caption both recommend cubic for smooth functions, contradicting the data.
- **No convergence table in committed artifacts.** Finding 4 cannot be verified without re-running because the convergence-vs-nodes data is only in the binary figure. Committing the convergence CSV alongside comparison.csv would make slope claims auditable from text artifacts.

## TDD execution sequence (for the next agent)

0. **Bullshit score = 50%.** Surface to user before any fix work. The wrong-winner claims (Findings 1 and 2) affect the Takeaway and table caption, which are the reader-facing recommendations.

1. **Finding 1 + 2 — write failing tests:**
   ```python
   # test_interpolation_claims.py
   import csv
   rows = list(csv.DictReader(open("tables/comparison.csv")))
   smooth_sup = {r["Method"]: float(r["Smooth sup-error"]) for r in rows}
   smooth_l2  = {r["Method"]: float(r["Smooth L2 error"])  for r in rows}
   # These PASS on current buggy code:
   assert smooth_sup["Cubic spline (natural)"] < smooth_sup["PCHIP (shape-preserving)"]  # FALSE claim
   # These FAIL on current code (honest-fix targets):
   assert smooth_sup["PCHIP (shape-preserving)"] < smooth_sup["Cubic spline (natural)"]
   assert smooth_l2["PCHIP (shape-preserving)"]  < smooth_l2["Cubic spline (natural)"]
   ```

2. **Fix run.py prose.** At `run.py:338-348`, replace "Cubic is uniformly smallest" with a winner derived from `fits["smooth"]`. At `run.py:391-396`, derive the winner string from `argmin(sup_err)` rather than hardcoding "Cubic spline".

3. **Fix README.md.** Regenerate by running `python run.py`; confirm `README.md:127` and `README.md:145` now name PCHIP as smooth-target winner.

4. **Finding 3 fix.** Add `mpc` row to the Model Setup table string in `run.py:214-225`.

5. **Finding 4 — re-run to verify.** After regeneration, read the log-log slopes numerically and assert them in the convergence block. If PCHIP slope is not near -4, update the prose to the observed slope.

6. Re-run this skill on the updated code. Score should drop to <= 10% if Findings 1 and 2 are resolved.
