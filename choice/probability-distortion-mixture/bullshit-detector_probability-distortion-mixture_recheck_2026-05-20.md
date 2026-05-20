# bullshit-detector - probability-distortion-mixture - recheck - 2026-05-20

**Bullshit score: 10%** - All four original findings are resolved. One residual LOW-severity DILUTED finding: Method 1 hat-lambda claim ("between 1 and 2.5") is asserted in prose but has no data artifact to ground it.

## Header

- Claim sources: `choice/probability-distortion-mixture/README.md`
- Code / artifact root: `choice/probability-distortion-mixture/run.py`
- Data artifacts: `choice/probability-distortion-mixture/tables/type-parameters.csv`, `choice/probability-distortion-mixture/tables/model-selection.csv`
- Seed audit: `choice/probability-distortion-mixture/bullshit-detector_probability-distortion-mixture_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | [ORIGINAL F1] RRP figure labeled gain-domain, computed on gain only | HOLDS | - | no |
| 2 | [ORIGINAL F2] lambda >= 1 enforced in optimizer | HOLDS | - | no |
| 3 | [ORIGINAL F3] README acknowledges xi update is an approximation, not monotone by construction | HOLDS | - | no |
| 4 | [ORIGINAL F4] Oracle start disclosed in README | HOLDS | - | no |
| 5 | Method 1 hat-lambda "between the EUT value of 1 and the strong-CPT value of 2.5" | DILUTED | LOW | no (intermediate result, not in any displayed table) |
| 6 | BIC formula and values match across code, prose, and table | HOLDS | - | no |
| 7 | lambda >= 1 satisfied in all estimated table values | HOLDS | - | no |
| 8 | NEC = 0.0020 at C = 3, confirms sharp classification | HOLDS | - | no |
| 9 | C = 3 selected by BIC | HOLDS | - | no |
| 10 | Gain-domain EV denominator safe (all positive) after filter | HOLDS | - | no |

## Findings

### [ORIGINAL F1] RESOLVED - RRP domain filter is in place

**Claim source (verbatim):** "The median relative risk premium is positive at high probabilities and negative at low probabilities, the signature of inverted-S probability weighting in the gain domain." - `README.md:160`

**Code evidence (verbatim):**
```python
df_gain = df[df["domain"] == "gain"].copy()
df_gain["ev"] = df_gain["p"] * df_gain["x1"] + (1.0 - df_gain["p"]) * df_gain["x2"]
df_gain["rrp"] = (df_gain["ev"] - df_gain["ce"]) / df_gain["ev"]
rrp_by_p = df_gain.groupby("p")["rrp"].median()
```
`run.py:408-411`

**Category:** HOLDS. Gain domain filter is present before the EV and RRP computation. All gain lotteries have strictly positive EV (minimum EV = 1.00 across all 35 gain cells), so the denominator is safe with no clipping required.

---

### [ORIGINAL F2] RESOLVED - lambda lower bound is 1.0 in both optimizers

**Claim source (verbatim):** "The loss-aversion factor $\lambda \geq 1$" - `README.md:33`

**Code evidence (verbatim):**
```python
# fit_single_type, run.py:148
bounds=[(0.05, 2.0), (1.0, 5.0), (0.05, 2.0), (0.05, 5.0)]

# fit_mixture_em M-step, run.py:238
bounds=[(0.05, 2.0), (1.0, 5.0), (0.05, 2.0), (0.05, 5.0)],
```

**Data evidence:** `tables/type-parameters.csv` - EUT estimated lambda = 1.000, Mild CPT = 1.487, Strong CPT = 2.533. All satisfy lambda >= 1.

**Category:** HOLDS. Both optimizer calls enforce lambda in [1.0, 5.0]. All three estimated values in the table are >= 1.0. EUT lambda = 1.000 hits the lower bound exactly, consistent with the true DGP value of 1.000.

---

### [ORIGINAL F3] RESOLVED - README acknowledges xi approximation and non-guaranteed monotonicity

**Claim source (verbatim):** "That is an approximation, also used in the BFDE implementation, which does not formally guarantee monotone improvement; in practice the log-likelihood still rises at every iteration on this design." - `README.md:112`

**Code evidence (verbatim):**
```python
# Use the maximum-posterior type to set xi; this is an approximation
# of the proper weighted update but stays close to the BFDE
# implementation, which profiles xi per subject.
best_c = int(np.argmax(posteriors[s_idx]))
xi[s_idx] = fit_xi_for_subject(...)
```
`run.py:245-251`

**Category:** HOLDS. README now explicitly qualifies the xi update as an approximation that does not formally guarantee monotone improvement. The code comment and the prose are consistent.

---

### [ORIGINAL F4] RESOLVED - Oracle start disclosed

**Claim source (verbatim):** "In this tutorial the C = 3 initial values coincide with the true data-generating parameters, so the recovery reported below is an oracle start: it shows EM converges and stays at the truth, not that EM finds the truth from a cold start." - `README.md:140`

**Code evidence (verbatim):**
```python
types_true = np.array([
    [0.95, 1.00, 1.00, 1.00],
    [0.85, 1.50, 0.65, 0.85],
    [0.70, 2.50, 0.40, 0.95],
])
# run.py:346-350

theta_init_c3 = np.array([
    [0.95, 1.00, 1.00, 1.00],
    [0.85, 1.50, 0.65, 0.85],
    [0.70, 2.50, 0.40, 0.95],
])
# run.py:377-381
```

`np.allclose(theta_init_c3, types_true)` is True (element-wise identical). README now discloses this explicitly and correctly scopes what the near-perfect recovery demonstrates.

**Category:** HOLDS.

---

### Finding 5: Method 1 hat-lambda "between 1 and 2.5" - no data artifact

- **Claim source (verbatim):** "When the data are heterogeneous, Method 1 averages incompatible types and produces a $\hat\lambda$ between the EUT value of 1 and the strong-CPT value of 2.5, describing no actual subject well." - `README.md:94`
- **Code evidence:** `theta_single` is computed at `run.py:362` via `fit_single_type`. The variable is used only to compute `ll_single` and `bic_single`, which feed the model-selection table (`run.py:364`, `run.py:771`). The actual parameter vector `theta_single` (including its lambda component) is never written to any table or figure.
- **Data evidence:** Neither `tables/type-parameters.csv` nor `tables/model-selection.csv` contains the single-type estimated lambda. No artifact exists to verify the specific claim that hat-lambda lies in (1, 2.5).
- **Category:** DILUTED - the code computes theta_single and the claim is mechanically plausible given the DGP (a weighted average of EUT lambda=1 and strong-CPT lambda=2.5 would lie between them), but the specific claim cannot be grounded against any on-disk artifact. The reader is being asked to trust an intermediate result that was computed but never reported.
- **Severity:** LOW - this is not a published table number. The finding affects a single sentence of qualitative prose in the Solution Method section. The model-selection results (BIC, NEC, log-likelihoods) are not affected.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "theta_single" not in open("tables/type-parameters.csv").read() and "theta_single" not in open("tables/model-selection.csv").read()  # PASSES on current code (theta_single is not in any table)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any("theta_single" in line or "Single-type lambda" in line for line in open("tables/type-parameters.csv"))  # PASSES if theta_single is added to the table
  ```

---

## Cross-cutting patterns

- All four original FALSE/DILUTED findings are now HOLDS. The fixes are clean: (1) domain filter inserted before RRP computation, (2) lambda lower bound raised from 0.5 to 1.0 in both optimizer calls, (3) README now qualifies EM monotonicity as an approximation-dependent claim, (4) README now explicitly names the oracle start and scopes what the recovery demonstrates.
- The one residual finding (Finding 5) is a new observation not in the original audit. It is LOW severity because the claim is qualitative and the intermediate result (`theta_single`) is not a displayed number. The fix is either to add `theta_single` to a table or to rephrase the claim as a theoretical expectation rather than a computed result.
- The type-parameters table, model-selection table, and BIC computation all cross-check cleanly against the formulas in `run.py`. No data drift between code and artifact.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** No halt required. The one residual finding is LOW severity and does not affect any displayed result.

1. **Finding 5 (optional LOW fix):** Either (a) add `theta_single[1]` (the lambda component) as a row in the model-selection table or a footnote to the Method 1 prose, grounding the "between 1 and 2.5" claim in an artifact; or (b) rephrase the prose to "would be expected to lie between" rather than asserting a computed fact. If option (a), re-run and confirm the table entry matches the prose.

2. No other remediation required. All honest-fix acceptance criteria from the original audit are satisfied.
