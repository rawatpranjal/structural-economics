# bullshit-detector -- sequence-space-jacobian-hank -- 2026-05-20

**Bullshit score: 20%** -- worst finding is MISLABELED (MED): the anticipation-curve figure is described as "columns of J^{C,r}_{0,s}" when the code plots the policy perturbation dc(a) averaged over skill, a distinct object. No FALSE or UNIMPLEMENTED findings; core economics and algorithm are faithfully implemented.

## Header
- Claim sources: `heterogeneous-agents/sequence-space-jacobian-hank/README.md`
- Code / artifact root: `heterogeneous-agents/sequence-space-jacobian-hank/run.py`
- Data artifacts: `heterogeneous-agents/sequence-space-jacobian-hank/tables/diagnostics.csv`
- Seed audit (if any): None
- Run by: claude-sonnet-4-6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Anticipation-curve figure = "columns of J^{C,r}_{0,s}" | MISLABELED | MED | no (figure is correct; label is wrong) |
| 2 | Prose implies O(T|state|) total for forward sweep | DILUTED | LOW | no (only compute-time claim) |
| 3 | Lowest quintile cuts consumption "roughly four times as much as the highest" | DATA DRIFT | LOW | needs re-run to verify |
| 4 | "HANK shows larger inflation and real-rate responses than RA" | DATA DRIFT | LOW | needs re-run to verify |

## Findings

### Finding 1: Anticipation-curve figure mislabeled as J^{C,r}_{0,s} columns

- **Claim source (verbatim):** "These curves are the columns of $J^{C, r}_{0, s}$ before the forward distribution propagation." -- `README.md:201`
- **Code evidence (verbatim):**
  ```python
  for s in sample_lags:
      ax3.plot(a_grid, J_r["curl"]["c"][s].mean(axis=1) / C_star,
               label=f"lag s = {s}")
  ```
  `run.py:901-903`
  ```python
  ax3.set_ylabel("$\\partial c_0(a) / \\partial r_s$ (skill-averaged)")
  ```
  `run.py:907`
- **Data evidence (if applicable):** None. The figure is not represented in `diagnostics.csv`.
- **Category:** MISLABELED
- **Severity:** MED
- **Result-changing:** no -- the figure itself is correct and useful; only the prose label misidentifies the plotted object.
- **Analysis:** `J_r["curl"]["c"][s]` has shape `(n_a, n_e)`. `.mean(axis=1)` yields shape `(n_a,)` -- the policy perturbation `dc(a,e)` averaged over the skill dimension, plotted as a function of assets `a`. A "column" of the Jacobian `J^{C,r}[:,s]` is a `T`-vector of scalars (one entry per time period `t`). The entry `J^{C,r}[0,s]` is a single scalar = `sum_{a,e} curl["c"][s](a,e) * D_bar(a,e)`. The plotted object is `dc(a)` averaged over `e`, not integrated over the distribution -- a `(n_a,)` function, not a `(T,)` vector. The code's own y-axis label (`partial c_0(a) / partial r_s, skill-averaged`) correctly describes what is plotted; the README prose mis-applies the Jacobian notation.
- **Violated invariant (one-line pytest assertion):**
  ```python
  # J column is a T-vector of scalars; the plot array has shape (n_a,) != (T,)
  assert J_r["curl"]["c"][0].mean(axis=1).shape == (T_HORIZON,)  # PASSES on buggy README; FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # Honest fix: claim says "policy-space anticipation curve dc(a), skill-averaged" not "Jacobian column"
  assert J_r["curl"]["c"][0].mean(axis=1).shape == (N_A,)  # PASSES always; honest fix changes only the prose
  ```

### Finding 2: Complexity prose implies O(T|state|) total but forward sweep is O(T^2|state|)

- **Claim source (verbatim):** "anticipation curves are translation-invariant, so they are computed once and then convolved with the time-varying input path during the forward sweep." -- `README.md:170-171`
- **Code evidence (verbatim):**
  ```python
  for s in range(T_hz):
      delta_D = np.zeros_like(D_bar)
      for t in range(T_hz):
          ...
          if t < T_hz - 1:
              delta_D = (
                  forward_distribution_step(delta_D, idx_low, omega_lo, P_e)
                  + policy_change_to_distribution_shift(
                      da_t, D_bar, a_grid, idx_low, P_e
                  )
              )
  ```
  `run.py:521-544`
- **Data evidence (if applicable):** None; complexity is not in `diagnostics.csv`.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no -- IRF numbers are unaffected; only the efficiency characterization is misleading.
- **Analysis:** The anticipation curves (`curl` arrays) are indeed computed once via backward iteration (`anticipation_curves`, `run.py:424-483`). But the forward distribution sweep (the `for s in range(T_hz)` outer loop at `run.py:521`) restarts `delta_D = np.zeros_like(D_bar)` for each `s`, performing `T_hz` full inner loops of length `T_hz`. Total cost is `O(T^2 |state|)`, not `O(T |state|)`. The prose correctly attributes the Toeplitz trick to the full SSJ library and says "this algorithm is the simplest version" (`README.md:173-174`), but the preceding "computed once and then convolved" sentence creates the false impression that this implementation achieves the O(T) total. The pseudocode itself correctly says "O(T |state|) per pulse date" at Step 3 (`README.md:158`), which -- multiplied by T pulse dates -- is O(T^2|state|) total.
- **Violated invariant (one-line pytest assertion):**
  ```python
  # Code does T_hz outer iterations, each restarting delta_D, costing O(T_hz^2 |state|)
  assert sum(1 for line in open("run.py").read().split("\n") if "delta_D = np.zeros_like" in line) == 1  # inside the s-loop = O(T^2)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # Prose should say "O(T^2 |state|) total forward sweep, compared to O(T |state|) with Toeplitz"
  assert "O(T^2" in open("README.md").read()
  ```

### Finding 3: "Lowest quintile cuts consumption roughly four times as much as the highest" -- unverifiable

- **Claim source (verbatim):** "the lowest wealth quintile cuts consumption roughly four times as much as the highest on impact" -- `README.md:232`
- **Code evidence (verbatim):**
  ```python
  irf[q, t] += float(np.sum(dc_t * quintile_mass[q]))
  ```
  `run.py:696`
- **Data evidence (if applicable):** `diagnostics.csv` contains no per-quintile data. The ratio "four times" cannot be verified from any committed artifact.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** needs re-run to verify -- the ratio is a published comparative claim about the cross-sectional decomposition but no artifact captures it.
- **Analysis:** The quintile IRF is computed correctly (policy channel, lines 674-697), but its output is never written to a table or CSV. The "four times" ratio is present only in the `add_results` prose at `run.py:1275`. Whether the ratio is accurate depends on the committed figures, which cannot be re-derived from `diagnostics.csv` alone.
- **Violated invariant (one-line pytest assertion):**
  ```python
  import pandas as pd; assert "Q1_peak" in pd.read_csv("tables/diagnostics.csv").columns  # FAILS on current code (not stored)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # quintile ratio verifiable from CSV: Q1_peak / Q5_peak approx 4
  df = pd.read_csv("tables/diagnostics.csv"); assert abs(df.set_index("Quantity").loc["Q1/Q5 peak consumption ratio", "Value"] - 4.0) < 1.5
  ```

### Finding 4: "HANK shows larger inflation and real-rate responses than RA" -- unverifiable from CSV

- **Claim source (verbatim):** "HANK shows larger inflation and real-rate responses than RA because the cross-section forces real marginal cost to move more to clear the goods market." -- `README.md:193`
- **Code evidence (verbatim):**
  ```python
  axes[0, 1].plot(h, pct * 4.0 * delta_pi[:plot_T], color="steelblue", label="HANK")
  axes[0, 1].plot(h, pct * 4.0 * ra_irf["pi"][:plot_T], color="darkorange", linestyle="--", label="Representative-agent NK")
  ```
  `run.py:851-854`
- **Data evidence (if applicable):** `diagnostics.csv` rows `Peak HANK output response (% of Y*): -0.190` and `Peak RA NK output response (%): -0.249`. No peak inflation or peak real-rate rows for either economy.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** needs re-run to verify -- inflation and real-rate peak comparisons are visible only in the committed figure, not in any CSV row.
- **Violated invariant (one-line pytest assertion):**
  ```python
  import pandas as pd; assert "Peak HANK inflation response" in pd.read_csv("tables/diagnostics.csv")["Quantity"].values  # FAILS
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  df = pd.read_csv("tables/diagnostics.csv").set_index("Quantity"); assert float(df.loc["Peak HANK inflation response (annualized %)", "Value"]) > abs(float(df.loc["Peak RA NK inflation response (annualized %)", "Value"]))
  ```

## Cross-cutting patterns

- All four findings are presentation-layer issues (wrong notation, implicit complexity claim, missing CSV rows). The underlying economic model, EGM solver, fake-news Jacobian, NKPC linearization, monetary block, and equilibrium solve are all faithfully implemented. No finding challenges a number in the diagnostics table.
- Findings 3 and 4 share a root: the `add_table` call writes only solver diagnostics and peak output/consumption; peak inflation and peak real-rate comparisons, and all quintile ratios, are embedded only in figure images and prose strings. Adding three CSV rows would make all four directional claims verifiable without re-running.
- Finding 1 and Finding 2 share a root: the prose reaches for terminology from the full SSJ library (Jacobian columns, Toeplitz/O(T)) to describe a simpler implementation. Both would be fixed by tightening the prose to describe what the code actually does, not what the canonical package does.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 20%** -- below the 50% halt threshold. Proceed with fixes.
1. **Finding 1 (MISLABELED):** Prose fix only. Change `README.md:201` to describe the plotted object as "the date-0 policy perturbation dc(a,e) averaged over skill -- the raw anticipation curve before distribution weighting" rather than "columns of J^{C,r}_{0,s}". No code change required.
2. **Finding 2 (DILUTED):** Prose fix only. Change `README.md:170-171` to read "anticipation curves are computed once (O(T|state|) backward iteration); the forward distribution sweep then runs T separate passes of length T, for O(T^2|state|) total -- versus O(T|state|) in the full SSJ library's Toeplitz implementation."
3. **Findings 3 and 4 (DATA DRIFT):** Add rows to the diagnostics table in `run.py`'s `diag_df` for peak HANK inflation, peak RA inflation, peak HANK real rate, and Q1/Q5 consumption ratio at t=0. These rows are already computed (`delta_pi`, `ra_irf["pi"]`, `delta_r`, `quintile_irf`) but not written to CSV.
4. After prose fixes, re-run `python scripts/validate_catalog.py`. Re-run this skill to confirm all findings read HOLDS and bullshit score drops to 0-10%.
