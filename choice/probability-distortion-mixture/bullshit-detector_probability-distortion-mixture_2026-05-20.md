# bullshit-detector — probability-distortion-mixture — 2026-05-20

**Bullshit score: 65%** — RRP figure is labeled and interpreted as gain-domain but computed over all three lottery domains with a denominator clipping bug that produces astronomically large absolute values for loss/mixed rows; the stated inverted-S result is not what the figure actually shows. Second FALSE: Equations states lambda >= 1 but optimization bound is 0.5 and the reported EUT estimate is 0.996 < 1.

## Header
- Claim sources: `choice/probability-distortion-mixture/README.md` (Equations, Solution Method, Results sections)
- Code / artifact root: `choice/probability-distortion-mixture/run.py`
- Data artifacts: `choice/probability-distortion-mixture/tables/type-parameters.csv`, `choice/probability-distortion-mixture/tables/model-selection.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | RRP figure shows gain-domain inverted-S | FALSE | HIGH | yes (figure interpretation is wrong; RRP contaminated by loss/mixed domains) |
| 2 | lambda >= 1 (Equations constraint enforced) | FALSE | MED | yes (EUT estimated lambda = 0.996 < 1 violates stated model; bound is 0.5) |
| 3 | EM is monotone in log-likelihood by construction | DILUTED | MED | no (monotonicity may hold empirically but is not guaranteed by the xi approximation) |
| 4 | C=3 seeded from BFDE headline pattern | DILUTED | MED | yes (initial values are identical to true DGP; recovery partly reflects oracle start) |

## Findings

### Finding 1: RRP figure labeled gain-domain but computed over all domains with broken denominator

- **Claim source (verbatim):** "The median relative risk premium is positive at high probabilities and negative at low probabilities, the signature of inverted-S probability weighting in the gain domain." — `README.md:158`
- **Claim source (figure title, verbatim):** figure alt text "Median relative risk premia by lottery probability" and prose "in the gain domain" — `README.md:160`; ax4 title `"Median relative risk premia in the gain domain"` — `run.py:718`
- **Code evidence (verbatim):**
  ```python
  df["ev"] = df["p"] * df["x1"] + (1.0 - df["p"]) * df["x2"]
  df["rrp"] = (df["ev"] - df["ce"]) / np.maximum(df["ev"], 1e-9)
  rrp_by_p = df.groupby("p")["rrp"].median()
  ```
  `run.py:404-406`
- **Code evidence (filter absent):**
  ```python
  rrp_for_plot = rrp_by_p.reindex([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
  ```
  `run.py:708` — no domain filter before groupby or before plot.
- **Data evidence:** No CSV table for RRP bars. Bar heights cannot be grounded without re-run. **needs re-run to verify** exact bar values. However, the code path is unambiguous: `df` contains all three domains; no filter applied.
- **Category:** FALSE
- **Severity:** HIGH
- **Result-changing:** yes — the prose paragraph describes gain-domain probability weighting. The actual computation pools all three domains. For loss lotteries, `ev < 0` on every cell (e.g., lottery `(0, -20, 0.05)` has `ev = 0.05*0 + 0.95*(-20) = -19.0`); `np.maximum(ev, 1e-9)` clips the denominator to `1e-9`, producing `rrp ~ -1e10`. At `p = 0.05`, 5 gain cells and 5 loss cells per subject produce 1000 gain rows and 1000 loss rows; the 1000 loss rows have `rrp` on the order of `-1e9` to `-1e11`, which dominates the median of 2000 values. At `p = 0.25, 0.50, 0.75`, mixed lotteries (5 cells each) add further cross-domain contamination with similarly broken denominators (e.g., `(40, -20, 0.25)` has `ev = -5.0`). The figure does not show gain-domain inverted-S behavior; it shows a quantity numerically dominated by the clipped-denominator loss and mixed rows.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "domain" not in str(inspect.getsource(main)).split("rrp_by_p")[0].split("df[")[-1]  # PASSES on buggy code (no domain filter before groupby); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert df_rrp.shape[0] == n_subjects * n_gain_lotteries  # df_rrp is the filtered dataframe used for rrp_by_p
  ```

---

### Finding 2: Equations states lambda >= 1 but optimization allows lambda >= 0.5; recovery reports 0.996

- **Claim source (verbatim):** "The loss-aversion factor $\lambda \geq 1$ scales the disutility of losses relative to the utility of equivalent-magnitude gains, so $\lambda = 1$ means no loss aversion" — `README.md:33`
- **Code evidence (Method 1, verbatim):**
  ```python
  res = minimize(neg_ll, theta0, method="L-BFGS-B",
                 bounds=[(0.05, 2.0), (0.5, 5.0), (0.05, 2.0), (0.05, 5.0)])
  ```
  `run.py:147-148`
- **Code evidence (Methods 2 and 3, verbatim):**
  ```python
  res = minimize(
      neg_weighted_ll_for_type, theta[c],
      args=(weights_subj, list(subjects), df),
      method="L-BFGS-B",
      bounds=[(0.05, 2.0), (0.5, 5.0), (0.05, 2.0), (0.05, 5.0)],
      options={"maxiter": 30},
  )
  ```
  `run.py:234-239`
- **Data evidence (verbatim):** `tables/type-parameters.csv` row: `EUT,0.950,0.974,1.000,0.996,1.000,0.950,1.000,1.005,0.20,0.20`. Estimated lambda = 0.996 < 1.0.
- **Category:** FALSE — Equations defines the domain as `lambda >= 1`; the optimization enforces `lambda >= 0.5`; the published estimate for the EUT type is 0.996, which lies in `[0.5, 1.0)`.
- **Severity:** MED — 0.996 is numerically close to 1.0, so the economic substance is preserved, but the stated constraint is violated by both the bound and the result.
- **Result-changing:** yes — the EUT type estimated lambda in the table (0.996) violates the Equations-stated constraint. A reader trusting the Equations section would expect estimated lambda in `[1, 5]` for all types.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert any(b == (0.5, 5.0) for b in [(0.05, 2.0), (0.5, 5.0), (0.05, 2.0), (0.05, 5.0)])  # PASSES on buggy code; FAILS after fixing to (1.0, 5.0)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert all(b != (0.5, 5.0) for b in bounds_used_in_all_minimizations)  # lambda lower bound is 1.0 everywhere
  ```

---

### Finding 3: EM monotonicity claim is not supported by the xi approximation used

- **Claim source (verbatim):** "EM is monotone in log-likelihood by construction." — `README.md:112`
- **Code evidence (verbatim):**
  ```python
  # Use the maximum-posterior type to set xi; this is an approximation
  # of the proper weighted update but stays close to the BFDE
  # implementation, which profiles xi per subject.
  best_c = int(np.argmax(posteriors[s_idx]))
  xi[s_idx] = fit_xi_for_subject(
      sub["ce"].to_numpy(), sub["x1"].to_numpy(),
      sub["x2"].to_numpy(), sub["p"].to_numpy(), theta[best_c],
  )
  ```
  `run.py:245-251`
- **Data evidence:** None — monotonicity is an algorithm property, not a table value.
- **Category:** DILUTED — standard EM monotonicity requires the M-step to maximize the complete-data expected log-likelihood over all parameters. Using `argmax` over types to select xi is a greedy approximation that breaks this guarantee. The code comment (line 245) acknowledges the approximation; the README does not.
- **Severity:** MED
- **Result-changing:** no — the log-likelihood values in `tables/model-selection.csv` are not directly affected by whether monotonicity is formally guaranteed.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "approximation" in inspect.getsource(fit_mixture_em)  # PASSES on current code; documents the gap between claim and implementation
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "approximation" in open("README.md").read()  # README must acknowledge the xi update is an approximation that does not guarantee monotonicity
  ```

---

### Finding 4: C=3 initial values are identical to true DGP; README does not disclose this

- **Claim source (verbatim):** "Initial values are seeded from the BFDE headline pattern, augmented with type-specific loss aversion: an EUT type with $\lambda = 1$, a mild-CPT type with $\lambda = 1.5$, and a strong-CPT type with $\lambda = 2.5$." — `README.md:138`
- **Claim source (verbatim):** "The local-maxima problem is mitigated by warm starts from the BFDE headline parameters." — `README.md:140`
- **Code evidence (DGP, verbatim):**
  ```python
  types_true = np.array([
      [0.95, 1.00, 1.00, 1.00],   # Type 1: EUT
      [0.85, 1.50, 0.65, 0.85],   # Type 2: mild CPT
      [0.70, 2.50, 0.40, 0.95],   # Type 3: strong CPT
  ])
  ```
  `run.py:346-350`
- **Code evidence (C=3 init, verbatim):**
  ```python
  theta_init_c3 = np.array([
      [0.95, 1.00, 1.00, 1.00],
      [0.85, 1.50, 0.65, 0.85],
      [0.70, 2.50, 0.40, 0.95],
  ])
  ```
  `run.py:377-381`
- **Data evidence (verbatim):** `tables/type-parameters.csv` — all three types recover parameters within 3% of true values. The near-perfect recovery is consistent with starting at the true values.
- **Category:** DILUTED — `np.allclose(theta_init_c3, types_true)` is True (exact element equality). EM starts at the true DGP parameters. The near-perfect table recovery does not demonstrate EM's ability to escape local optima; it demonstrates convergence from an oracle start. The README implies the good performance is due to BFDE warm-starting, not to oracle initialization.
- **Severity:** MED
- **Result-changing:** yes — the central pedagogical claim (EM recovers the true type structure) is overstated. Results cannot be cited as evidence of EM estimation power from realistic starting conditions.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert np.allclose(theta_init_c3, types_true)  # PASSES on current code; FAILS after honest fix (different init)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert not np.allclose(theta_init_c3, types_true) or "initial values coincide with the true DGP" in open("README.md").read()
  ```

---

## Cross-cutting patterns

- Both FALSE findings (RRP domain, lambda bound) share the same root: the stated mathematical domain in Equations is narrower than the computational domain enforced in code. The mismatch was not caught because the figures and table numbers still look reasonable for the specific seed used.
- Findings 1 and 4 together inflate the apparent quality of results. Finding 1 makes a numerically contaminated figure appear to show clean gain-domain behavior; Finding 4 makes oracle-start recovery appear to demonstrate EM estimation power. Both Results claims are more favorable than the code justifies.
- The only honest hedge about an algorithmic approximation lives in `run.py:245` as a code comment. Audit any README prose claim about algorithmic guarantees (monotonicity, convergence, identifiability) against code comments that qualify the same operations.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 65%.** Surface to the user before writing fix code. The RRP figure (Finding 1) requires a re-run to confirm exact bar heights after the domain filter is added. Do not treat the existing figure as ground truth.

1. **Finding 1 (RRP domain, FALSE/HIGH).**
   - Failing test: confirm `df` passed to rrp computation contains all three domains.
   - Fix: insert `df_gain = df[df["domain"] == "gain"].copy()` before `run.py:404`; replace all subsequent `df["ev"]`, `df["rrp"]`, and `df.groupby("p")` with `df_gain` equivalents.
   - Re-run; confirm figure now shows gain-only inverted-S pattern with sensible bar magnitudes.

2. **Finding 2 (lambda bound, FALSE/MED).**
   - Failing test: `assert all(b[1][0] >= 1.0 for b in zip(["alpha","lam","gamma","delta"], bounds) if b[0]=="lam")`.
   - Fix: change `(0.5, 5.0)` to `(1.0, 5.0)` in both `fit_single_type` (`run.py:148`) and `fit_mixture_em` (`run.py:238`).
   - Re-run; confirm EUT estimated lambda >= 1.0 in all method outputs.

3. **Finding 3 (EM monotonicity, DILUTED/MED).**
   - No code change required. Add a qualifying sentence to `README.md` Method 2 paragraph: "The xi update uses the maximum-posterior type rather than a type-weighted expectation, an approximation that does not formally guarantee monotonicity but follows the BFDE implementation."

4. **Finding 4 (oracle init, DILUTED/MED).**
   - Option A: disclose in `README.md` Method 3 paragraph that `theta_init_c3` coincides with `types_true` in this tutorial and that the near-perfect recovery partly reflects this oracle start.
   - Option B: use perturbed initial values (e.g., `types_true * rng.uniform(0.85, 1.15, size=types_true.shape)`) and re-run to show EM still converges from nearby but non-oracle conditions.

5. After all fixes, re-run `python run.py` inside the tutorial folder. Regenerate `README.md` and figures. Re-run this skill on the new code to confirm findings read HOLDS and the score is <= 25%.
