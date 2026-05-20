# bullshit-detector — ridge-lasso-sparsity — 2026-05-20

**Bullshit score: 20%** — one DILUTED finding (MED severity): "False inclusions by lasso = 0" is a mathematical certainty given the all-nonzero DGP, not a measurement of lasso precision; the table row actively misleads without disclosure. All algorithmic and numeric claims hold.

## Header
- Claim sources: `time-series/ridge-lasso-sparsity/README.md` (all sections)
- Code / artifact root: `time-series/ridge-lasso-sparsity/run.py`
- Data artifacts: `time-series/ridge-lasso-sparsity/tables/forecast_metrics.csv`, `time-series/ridge-lasso-sparsity/tables/selection_summary.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Ridge formula: argmin (1/n)sum(...)^2 + lambda*sum(b_j^2) | HOLDS | — | no |
| 2 | Lasso formula: argmin (1/n)sum(...)^2 + lambda*sum(|b_j|) | HOLDS | — | no |
| 3 | "intercept is never penalized" | HOLDS | — | no |
| 4 | "Lasso uses cyclic coordinate descent" | HOLDS | — | no |
| 5 | "Ridge has closed-form solution" | HOLDS | — | no |
| 6 | time-block splits (train=125, valid=55, test=79) | HOLDS | — | no |
| 7 | Ridge lambda=0.0381, Lasso lambda=0.0079 | HOLDS | — | no |
| 8 | Ridge RMSE=0.2699, Lag RMSE=0.5989 | HOLDS | — | no |
| 9 | Ridge shock correlation=0.773 | HOLDS | — | no |
| 10 | Lasso selects 56 indicators | HOLDS | — | no |
| 11 | True nonzero policy indicators=120 | HOLDS | — | no |
| 12 | Dense-signal share missed=0.510 (51.0%) | HOLDS | — | no |
| 13 | Ridge coef corr with truth=0.644 | HOLDS | — | no |
| 14 | Lasso coef corr with truth=0.740 | HOLDS | — | no |
| 15 | "False inclusions by lasso = 0" presented as measurement | DILUTED | MED | no (table row is correct but misleading) |
| 16 | All tabular numbers in README match tables/*.csv | HOLDS | — | no |

## Findings

### Finding 1: "False inclusions by lasso = 0" is a DGP tautology, not a lasso performance measurement

- **Claim source (verbatim):** "False inclusions by lasso | 0" in the selection table, followed by "The selection table separates statistical selection from economic sparsity." — `README.md:115` and `README.md:109-119`

- **Code evidence (verbatim):**
  ```python
  true_nonzero = np.abs(true_text_beta) > 1e-10
  ...
  {
      "Statistic": "False inclusions by lasso",
      "Value": int(np.sum(lasso_selected & ~true_nonzero)),
  },
  ```
  `run.py:353,369-372`

  And the DGP that makes `true_nonzero` always all-True:
  ```python
  beta = rng.normal(0.0, 0.008, n_features)
  ...
  weak = concept_signs[concept] * rng.uniform(0.006, 0.018, indicators_per_concept)
  beta[sl] += weak
  ```
  `run.py:170,180-181`

- **Data evidence:** `tables/selection_summary.csv:5`: `False inclusions by lasso,0`

- **Category:** DILUTED — the computation `lasso_selected & ~true_nonzero` is always zero because `~true_nonzero` is always all-False: by DGP construction every one of the 120 indicators receives a minimum per-concept addition of at least 0.006 (rng.uniform lower bound), making every `|beta_j|` >> 1e-10. The reader is shown a "0" that looks like evidence lasso avoids spurious selection, but it is impossible for lasso to produce a false inclusion under this DGP regardless of how poorly lasso performs. The metric conveys no information about lasso's selectivity.

- **Severity:** MED — the number in the table is arithmetically correct and the result-table numbers are unaffected, but the prose framing ("separates statistical selection from economic sparsity") invites the reader to treat the zero as a meaningful diagnostic when it is not.

- **Result-changing:** no — the RMSE, shock-correlation, and dense-signal-share numbers are unaffected. Only the interpretation of the zero false-inclusion count is wrong.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert np.all(np.abs(true_text_beta) > 1e-10)  # PASSES on current code: all 120 betas nonzero by construction, making false-inclusion count structurally zero
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "always zero by DGP construction" in readme_text or np.any(~true_nonzero)  # PASSES on honest fix: either README discloses the tautology OR the DGP is redesigned to include some truly zero betas
  ```

## Cross-cutting patterns

- All algorithmic implementations (ridge closed-form, lasso coordinate descent, intercept centering) match the README equations exactly. There is no gap between the formula stated and the formula executed.
- All numeric values in `README.md` are dynamically interpolated from the same computed objects that populate `tables/*.csv` (e.g. `run.py:574-584`, `run.py:640-649`, `run.py:704-712`). Numeric drift between prose and tables is structurally impossible given the code architecture.
- The one finding (false inclusions) is a DGP design issue, not a code faithfulness issue. The code computes exactly what it claims; the problem is that the DGP renders the statistic uninformative and the README does not disclose this.
- The "Dense-signal share missed" metric is measured as an L1 mass fraction of the bottom-90th-percentile betas, not a count fraction. The prose ("51.0% of the weak dense signal") correctly uses mass-fraction language; no mislabeling.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 20%.** Below the 50% halt threshold. Forward work may continue; address Finding 1 before publishing or teaching from this tutorial.

1. Turn the violated invariant into a pytest test under `tests/`:
   ```python
   # test: false_inclusions_is_tautological
   # PASSES on current code (proves the structural issue):
   assert np.all(np.abs(true_text_beta) > 1e-10)
   ```
   Confirm it passes on current code.

2. The honest-fix pass condition test:
   ```python
   # FAILS on current code (no README disclosure):
   assert "always zero by DGP construction" in open("README.md").read()
   ```
   Confirm it fails on current code.

3. Hand off to `writing-plans` for one of two fix choices:
   - **Option A (preferred):** Redesign the DGP to include a fraction of exactly-zero betas (e.g., set `beta[j] = 0` for a random 20% of indicators). This makes the false-inclusion count genuinely informative and teaches the reader what lasso actually achieves.
   - **Option B (minimal):** Add a footnote to the selection table: "False inclusions are structurally zero in this DGP because all 120 indicators have nonzero true coefficients; the zero reflects the DGP, not lasso's precision."

4. After fix: re-run `python run.py`, re-run this skill. The finding should read HOLDS and the score should drop to 0-10%.
