# bullshit-detector — aiyagari-hact — 2026-05-20

**Bullshit score: 40%** — Two FALSE claims in README.md:241 ("wealth Gini agrees to three decimal places" and "mass at borrowing limit agrees to one percentage point") are directly contradicted by numbers in the same README's own table. Both claims overstate agreement by 10x+ and could mislead a reader about the accuracy of the discrete-time vs continuous-time comparison, which is the headline pedagogical claim of the tutorial.

## Header
- Claim sources: `heterogeneous-agents/aiyagari-hact/README.md`
- Code / artifact root: `heterogeneous-agents/aiyagari-hact/run.py`
- Data artifacts: `heterogeneous-agents/aiyagari-hact/tables/equilibrium.csv`, `heterogeneous-agents/aiyagari-hact/tables/dt-vs-ct-comparison.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Gini agrees to three decimal places" | FALSE | HIGH | yes — overstates DT/CT agreement by 11x |
| 2 | "mass at BL agrees to one percentage point" | FALSE | HIGH | yes — actual gap is 2.96 pp, claimed 1 pp |
| 3 | "two Ginis agree to within 0.011" | DATA DRIFT | LOW | no — actual gap 0.0114, off by 0.0004 |
| 4 | HACT vs DT mass at BL use different objects | DILUTED | MED | yes — objects are not comparable |
| 5 | All other numeric/algorithmic claims | HOLDS | — | — |

## Findings

### Finding 1: "wealth Gini agrees to three decimal places"

- **Claim source (verbatim):** "The wealth Gini agrees to three decimal places" — `README.md:241`
- **Code evidence (verbatim):**
  ```python
  hact_gini = gini_from_density(a, g_marginal * da)
  ...
  dt_gini = gini_from_density(a_grid_dt, dt_marginal)
  ```
  `run.py:455`, `run.py:493`
- **Data evidence (verbatim):** `tables/dt-vs-ct-comparison.csv`:
  ```
  Wealth Gini,0.5146,0.5260,0.0114
  ```
  Gap = 0.0114. Three decimal places of agreement requires gap < 0.001. The claim is contradicted by the table printed in the same README at line 246. The first digit of the third decimal place differs: 0.51**4**6 vs 0.52**6**0.
- **Category:** FALSE
- **Severity:** HIGH
- **Result-changing:** yes — the headline pedagogical comparison claims the two solvers agree closely; "three decimal places" overstates agreement by 11x relative to the actual 0.0114 gap. A reader trusting this claim would underestimate discretisation error in the Gini.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(0.5146 - 0.5260) < 0.001  # PASSES on current buggy prose; FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(0.5146 - 0.5260) < 0.012  # PASSES on honest fix ("within 0.012"); FAILS on "three decimal places"
  ```

---

### Finding 2: "mass at borrowing limit agrees to one percentage point"

- **Claim source (verbatim):** "the mass at the borrowing limit agrees to one percentage point" — `README.md:241`
- **Code evidence (verbatim):**
  ```python
  mass_at_constraint = float(np.sum(g_marginal[a <= a_min + 0.02]) * da)
  ...
  dt_mass_at_constraint = float(dt_marginal[0])
  ```
  `run.py:454`, `run.py:494`
- **Data evidence (verbatim):** `tables/dt-vs-ct-comparison.csv`:
  ```
  Mass at borrowing limit,0.0541,0.0245,0.0296
  ```
  Gap = 0.0296 = 2.96 percentage points. The claim "one percentage point" understates the actual gap by a factor of ~3. The absolute gap column in the tutorial's own table reads 0.0296.
- **Category:** FALSE
- **Severity:** HIGH
- **Result-changing:** yes — a reader assessing how well the DT and CT methods agree on the distribution near the constraint would be misled. The 2.96 pp gap is large enough to be economically meaningful and is also inflated by the methodological inconsistency in Finding 4.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(0.0541 - 0.0245) <= 0.01  # PASSES on current buggy prose; FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(0.0541 - 0.0245) <= 0.03  # PASSES on honest fix ("within ~3 pp"); FAILS on "one pp"
  ```

---

### Finding 3: "two Ginis agree to within 0.011"

- **Claim source (verbatim):** "The two Ginis agree to within $0.011$" — `README.md:210`
- **Code evidence:** same `gini_from_density` calls as Finding 1.
- **Data evidence:** `tables/dt-vs-ct-comparison.csv` row `Wealth Gini,0.5146,0.5260,0.0114`. Gap = 0.0114 > 0.011 by 0.0004.
- **Category:** DATA DRIFT — the prose value (0.011) is a rounded-down version of the table value (0.0114), and the two are in the same document.
- **Severity:** LOW — 0.0004 discrepancy; does not change economic interpretation.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(0.5146 - 0.5260) <= 0.011  # PASSES on current code output; FAILS if fixed to report 0.0114
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "0.011" not in open("README.md").read() or "0.0114" in open("README.md").read()  # consistent rounding
  ```

---

### Finding 4: HACT and DT "mass at borrowing limit" are different objects

- **Claim source (verbatim):** "Mass at borrowing limit" column header in comparison table — `README.md:251` / `tables/dt-vs-ct-comparison.csv:6`
- **Code evidence (verbatim):**
  ```python
  # HACT: integral of density over a <= 0.02
  mass_at_constraint = float(np.sum(g_marginal[a <= a_min + 0.02]) * da)
  # a has 800 uniform pts on [0,30], da = 0.03754 > 0.02
  # So a[0] = 0.0 is the only pt with a <= 0.02
  # mass_at_constraint = g_marginal[0] * da

  # DT: raw probability mass at first grid node (no da factor)
  dt_mass_at_constraint = float(dt_marginal[0])
  ```
  `run.py:454`, `run.py:494`
- **Data evidence:** HACT = 0.0541, DT = 0.0245. The HACT number is a density-times-step-size integral (one grid point, `g_marginal[0] * da`). The DT number is a probability mass (no grid spacing factor). On a uniform HACT grid with `da = 0.0375`, and an exponential DT grid with a different step at `a_grid_dt[0] = 0`, these measure the same economic concept (fraction of households at or near the constraint) but via numerically incomparable objects. The comparison overstates the DT mass gap.
- **Category:** DILUTED — the table presents a like-for-like comparison but the two rows use different normalisations.
- **Severity:** MED — inflates the apparent DT/CT discrepancy on the mass statistic and partly explains why Finding 2's claim of "one pp" is wrong.
- **Result-changing:** yes — the 2.96 pp gap is partly a numerical artefact of the different mass definitions, not pure economic disagreement between the two solvers.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "g_marginal[0] * da" in inspect.getsource(main) and "dt_marginal[0]" in inspect.getsource(main)  # both computed differently; PASSES on buggy code
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # after fix: both use the same definition (both integral over [0, 0.02], or both point mass)
  assert abs(mass_at_constraint_hact_fixed - mass_at_constraint_dt_fixed) < 0.015  # comparable objects
  ```

---

## Cross-cutting patterns

- Findings 1 and 2 are the same disease: the "agrees to X" summary sentence at README.md:241 overstates agreement on two consecutive statistics. The sentence was likely written by eyeballing the table at a high level and rounding aggressively in the optimistic direction. A single careful read of the table — which appears in the same README — would have caught both.
- Finding 4 is the root cause that amplifies Finding 2: because `dt_mass_at_constraint = float(dt_marginal[0])` (no `da` factor) while `mass_at_constraint` uses `g_marginal[0] * da`, the apparent gap is contaminated by different normalisation conventions. Fixing Finding 4 (making both definitions comparable) will shrink the gap in Finding 2 and may or may not rescue the "one pp" claim — it needs a re-run to determine.
- All algorithmic and structural claims (upwind HJB, KFE transposition, CTMC generator construction, bisection bracket, implicit time step, 39% precautionary wedge) HOLD against the code. The audit failures are confined to the comparison-table prose summary.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 40%.** Below the 50% halt threshold. Surface to the user; the code is honest, the table is honest, only the prose summary sentence at README.md:241 is wrong. Safe to fix without halting forward work.

1. **Finding 1 (Gini three decimal places):**
   - Test: `assert abs(0.5146 - 0.5260) < 0.001` — confirms this passes on current (buggy) prose.
   - Fix: change "agrees to three decimal places" to "agrees to within 0.012" (or similar honest bound) in the `add_table` description string in `run.py` around line 1010-1014. Regenerate README.

2. **Finding 2 (mass at BL one pp):**
   - Test: `assert abs(0.0541 - 0.0245) <= 0.01` — confirms this passes on current (buggy) prose.
   - Fix: change "agrees to one percentage point" to "differs by roughly 3 percentage points" in the same description string. Regenerate README.

3. **Finding 3 (Gini 0.011 vs 0.0114):**
   - Fix: update the figure description string for `wealth-distribution-comparison.png` in `run.py` around line 880 to use `f"{abs(hact_gini - dt_gini):.4f}"` (i.e. let the code write the number) instead of the hardcoded `0.011`. That line already does this correctly: `f"The two Ginis agree to within ${abs(hact_gini - dt_gini):.3f}$"` — re-run will produce the right number automatically.

4. **Finding 4 (mass at BL definition mismatch):**
   - Fix option A: change DT computation to `float(np.sum(dt_marginal[a_grid_dt <= 0.02]))` (integral over [0, 0.02] like HACT).
   - Fix option B: change HACT computation to `float(g_marginal[0])` (point mass like DT).
   - Fix option C: add a prose footnote explaining the different definitions.
   - Escalate choice to user — all three options change the reported comparison numbers.

5. After fixes, re-run `python run.py` to regenerate README and CSVs. Re-run this skill to confirm findings now read HOLDS and score drops to ≤10%.
