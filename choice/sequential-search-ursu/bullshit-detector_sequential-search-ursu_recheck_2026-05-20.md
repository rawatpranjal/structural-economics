# bullshit-detector — sequential-search-ursu — recheck — 2026-05-20

**Bullshit score: 0%** — original Finding 1 (DILUTED, HIGH: prose claimed inside purchase share falls as search costs rise while every CSV row showed 1.0) resolved by correcting the Results prose to match the committed table; all other claims verified HOLDS.

## Header
- Claim sources: `choice/sequential-search-ursu/README.md`
- Code / artifact root: `choice/sequential-search-ursu/run.py`
- Data artifacts: `choice/sequential-search-ursu/tables/parameter-recovery.csv`, `choice/sequential-search-ursu/tables/moment-fit.csv`, `choice/sequential-search-ursu/tables/search-cost-counterfactual.csv`
- Seed audit: `bullshit-detector_sequential-search-ursu_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Inside share falls as search costs rise | HOLDS (fixed) | — | no |
| 2 | Weitzman reservation equation implementation | HOLDS | — | no |
| 3 | W diagonal via moment scaling | HOLDS | — | no |
| 4 | Search stops when highest remaining z_j <= b_i | HOLDS | — | no |
| 5 | Outside option enters through zero in b_i | HOLDS | — | no |
| 6 | Parameter recovery numbers (README vs CSV) | HOLDS | — | no |
| 7 | Moment fit numbers (README vs CSV) | HOLDS | — | no |
| 8 | Counterfactual grid numbers (README vs CSV) | HOLDS | — | no |
| 9 | True parameter values in Model Setup match code | HOLDS | — | no |
| 10 | Fixed parameters (alpha, gamma, sigma) held fixed | HOLDS | — | no |
| 11 | SMM uses fixed simulation draws | HOLDS | — | no |
| 12 | Average searches falls monotonically across counterfactual grid | HOLDS | — | no |
| 13 | Inside purchase share stays at 1 with explanation | HOLDS | — | no |

## Findings

### Finding 1 (original): DILUTED — prose claimed inside share falls; table showed 1.0 — RESOLVED

- **Original claim:** "Increasing search costs lowers the amount of inspection and pushes some consumers to stop earlier. The inside purchase share falls because consumers are less likely to discover a product match that beats the outside option." — `README.md:297-299` (original)

- **Current code evidence:**
  ```python
  # run.py:632-643
  report.add_results(
      "Increasing search costs lowers the amount of inspection and pushes some "
      "consumers to stop earlier: average searches falls from about three "
      "products at half the baseline cost to under two at twice the baseline "
      "cost. The inside purchase share stays at one across this grid. The "
      "outside option is coded as the zero floor in the best-value rule, but at "
      "this calibration the inside products have high enough mean utilities that "
      "every consumer still uncovers a positive match before stopping, even at "
      "twice the baseline search cost. The counterfactual lesson here is about "
      "search depth, not the inside-outside split: with a less generous product "
      "calibration, or a wider cost grid, the same mechanism would eventually "
      "send some consumers to the outside option."
  )
  ```

- **Current README evidence:** `README.md:297` — "The inside purchase share stays at one across this grid. The outside option is coded as the zero floor in the best-value rule, but at this calibration the inside products have high enough mean utilities that every consumer still uncovers a positive match before stopping, even at twice the baseline search cost."

- **Data evidence:** `tables/search-cost-counterfactual.csv:2-6` — inside purchase share = 1.0 in every row. Prose and table now agree.

- **Resolution:** The prose no longer asserts a falling inside demand curve. It correctly states the inside share stays at one and explains the mechanism: high mean utilities ensure every consumer finds a positive match before stopping, even at 2x cost. The counterfactual's actual lesson (search depth falls) is the conclusion the table supports. Finding fully resolved.

- **Category:** HOLDS (post-fix)

### Finding 2 — Weitzman reservation equation (HOLDS)

- **Claim source (verbatim):** "$c_j = E[\max(u_{ij}-z_j,0)]$" — `README.md:62-65`
- **Code evidence:**
  ```python
  # run.py:28-30
  def expected_gain_over_threshold(k: float) -> float:
      """E[max(Z-k,0)] for Z distributed standard normal."""
      return norm.pdf(k) - k * (1.0 - norm.cdf(k))
  ```
  `run.py:37`: `target = cost / match_sd` — standardized form.
  `run.py:43`: `values[j] = mean_utility[j] + match_sd * k_star` — de-standardized reservation value.
- **Category:** HOLDS

### Finding 3 — W diagonal via moment scaling (HOLDS)

- **Claim source (verbatim):** "The weighting matrix W is diagonal in the implementation." — `README.md:111-113`
- **Code evidence:** `run.py:173`: `scale = np.maximum(np.abs(target), MOMENT_SCALE_FLOOR)` then `run.py:160`: `diff = (moments(sample) - target) / scale` and `return float(diff @ diff)` — equivalent to `(m-t)^T diag(1/a^2) (m-t)`. Exactly diagonal.
- **Category:** HOLDS

### Finding 4 — Search stopping rule (HOLDS)

- **Claim source (verbatim):** "If the highest remaining reservation value is below $b_i$, every other uninspected product has even lower option value, so she stops." — `README.md:88-90`
- **Code evidence:** `run.py:77`: `order = np.argsort(-reservation)` — descending order. `run.py:87`: `active = reservation[product] > best_value`. Products are visited high-to-low; when `active` is false for the current product (lowest remaining unvisited reservation), all subsequent products have even lower reservation values. Stopping rule exact match.
- **Category:** HOLDS

### Finding 5 — Outside option through zero in b_i (HOLDS)

- **Claim source (verbatim):** "The outside option enters through the zero in $b_i$." — `README.md:83`
- **Code evidence:** `run.py:82-83`: `best_value = np.zeros(n_consumers)`, `best_product = np.full(n_consumers, n_products, dtype=int)`. Consumer takes outside option iff all inspected match values are negative (below 0). Correct.
- **Category:** HOLDS

### Finding 6 — Parameter recovery numbers (HOLDS)

- **Claim source (verbatim):** `README.md:305-309` — Quality taste True=1.18, Est=1.193, Error=0.013; Base search cost True=0.08, Est=0.0791, Error=-0.0009; Complexity slope fixed 0.48.
- **Data evidence:** `tables/parameter-recovery.csv:2-4` — exact match.
- **Category:** HOLDS

### Finding 7 — Moment fit numbers (HOLDS)

- **Claim source (verbatim):** `README.md:313-327` — 13 moment rows.
- **Data evidence:** `tables/moment-fit.csv:1-14` — all 13 rows match exactly.
- **Category:** HOLDS

### Finding 8 — Counterfactual grid numbers (HOLDS)

- **Claim source (verbatim):** `README.md:331-337` — 5 rows with multipliers 0.5, 0.75, 1.0, 1.5, 2.0.
- **Data evidence:** `tables/search-cost-counterfactual.csv:1-6` — exact match. Inside share = 1.0 in all rows; average searches falls from 3.0218 to 1.818.
- **Category:** HOLDS

### Finding 9 — True parameter values in Model Setup match code (HOLDS)

- **Claim source (verbatim):** `README.md:128-130` — True quality taste 1.18, True base search cost 0.080, Complexity slope 0.48.
- **Code evidence:** `run.py:254`: `theta_true = np.array([1.18, np.log(0.08)])`. `exp(log(0.08)) = 0.08`. `run.py:18`: `COST_COMPLEXITY_SLOPE = 0.48`.
- **Category:** HOLDS

### Finding 10 — Fixed parameters held fixed (HOLDS)

- **Claim source (verbatim):** `README.md:115-117` — alpha=0.32, sigma=0.85, gamma=0.48 fixed.
- **Code evidence:** `run.py:57`: `price_taste = 0.32` hardcoded in `primitives()`; `run.py:253`: `match_sd = 0.85`; `run.py:18`: `COST_COMPLEXITY_SLOPE = 0.48`. None appear in estimated `theta`.
- **Category:** HOLDS

### Finding 11 — Fixed simulation draws (HOLDS)

- **Claim source (verbatim):** "fixed simulation shocks" — `README.md:242`
- **Code evidence:** `run.py:258`: `simulation_draws = np.random.default_rng(712).normal(size=(9000, 5))` seeded before optimization, passed as constant `draws` argument to all `criterion` calls.
- **Category:** HOLDS

### Finding 12 — Average searches falls monotonically (HOLDS)

- **Claim source (verbatim):** "average searches falls from about three products at half the baseline cost to under two at twice the baseline cost" — `README.md:297`
- **Data evidence:** `tables/search-cost-counterfactual.csv:2-6` — average searches: 3.0218, 2.6519, 2.3953, 2.0396, 1.818 — strictly decreasing. "About three at half cost" (3.02) and "under two at twice cost" (1.818) both exact.
- **Category:** HOLDS

### Finding 13 — Inside share stays at 1 with mechanism explanation (HOLDS)

- **Claim source (verbatim):** "The inside purchase share stays at one across this grid." — `README.md:297`
- **Data evidence:** `tables/search-cost-counterfactual.csv:2-6` — inside purchase share = 1.0 in all five rows. Prose and data agree exactly.
- **Category:** HOLDS

## Cross-cutting patterns

- The one non-HOLDS finding from the original audit was a prose-vs-data mismatch. The fix corrected the prose to match the data. No code or data artifact needed to change.
- The corrected prose at `README.md:297` not only removes the false claim but explains the mechanism: high mean utilities and the 2x cost grid are insufficient to push consumers to the outside option at this calibration. The explanation names the counterfactual's actual lesson (search depth) and acknowledges what would change the result (less generous calibration, wider cost grid). This is honest reporting.
- All mathematical derivations, algorithmic pseudocode, parameter values, and tabulated numbers are faithful to the code and CSV artifacts. No gaps remain.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings read HOLDS. No further fixes required.
1. Test suite status: `test_f1_violated_invariant_prose_claims_inside_share_falls` FAILS (correct — "The inside purchase share falls" no longer appears in `run.py`); `test_f1_honest_fix_prose_matches_flat_inside_share` PASSES (correct — prose matches table, "near one" / "stays at one" language present in README).
2. `test_committed_table_has_flat_inside_share` and `test_average_searches_does_fall` both PASS — data artifacts unchanged and consistent with corrected prose.
3. No re-run needed. Prose fix only; tables and code logic untouched.
</content>
