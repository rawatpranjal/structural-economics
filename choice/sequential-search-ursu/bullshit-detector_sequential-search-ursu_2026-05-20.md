# bullshit-detector — sequential-search-ursu — 2026-05-20

**Bullshit score: 35%** — One DILUTED finding at HIGH severity: prose claims the inside purchase share falls as search costs rise, but the committed CSV shows inside share = 1.0 for every row of the counterfactual grid; average searches falls correctly but the inside/outside split mechanism is calibrated away from the described effect.

## Header
- Claim sources: `choice/sequential-search-ursu/README.md` (Overview, Equations, Model Setup, Solution Method, Results)
- Code / artifact root: `choice/sequential-search-ursu/run.py`
- Data artifacts: `tables/parameter-recovery.csv`, `tables/moment-fit.csv`, `tables/search-cost-counterfactual.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Inside purchase share falls as search costs rise | DILUTED | HIGH | yes — described counterfactual effect absent from every row of published table |
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

## Findings

### Finding 1: Inside purchase share claimed to fall; committed table shows 1.0 throughout

- **Claim source (verbatim):** "Increasing search costs lowers the amount of inspection and pushes some consumers to stop earlier. The inside purchase share falls because consumers are less likely to discover a product match that beats the outside option." — `README.md:297-299`

- **Code evidence (verbatim):**
  ```python
  best_value = np.zeros(n_consumers)
  best_product = np.full(n_consumers, n_products, dtype=int)
  ...
  improved = active & (match_value[:, product] > best_value)
  best_value[improved] = match_value[improved, product]
  best_product[improved] = product
  ```
  `run.py:82-96`

  A consumer buys the outside option only if `best_product` stays at `n_products`, which requires all inspected match values to be negative. With simulation seed 712 and 9,000 consumers (`run.py:258`), zero consumers have `max_j(match_value_j) < 0`: mean utilities range from 0.777 (Basic) to 1.716 (Premium) at the estimated theta; with `sigma=0.85` the probability that ALL five products produce a negative match is essentially zero at this calibration. Even at 2x search cost, Premium's reservation value is approximately 1.88 (remains well above the initial `best_value=0`), so Premium is always the first product searched, and its match value is positive for essentially all 9,000 draws.

- **Data evidence (verbatim):**
  ```
  Search cost multiplier,Average searches,Inside purchase share,Outside share
  0.5,3.0218,1.0,0.0
  0.75,2.6519,1.0,0.0
  1.0,2.3953,1.0,0.0
  1.5,2.0396,1.0,0.0
  2.0,1.818,1.0,0.0
  ```
  `tables/search-cost-counterfactual.csv:1-6`

  Also: `Purchase share: Outside,0.0,0.0,0.0` in `tables/moment-fit.csv:12`. The outside option has zero share in both the observed synthetic data (seed 711, 4,000 consumers) and the fitted simulation, confirming this is a structural feature of the calibration, not a sampling artifact.

- **Category:** DILUTED — the code does implement the correct outside-option mechanism (`best_product` initialized to `n_products`, `best_value` initialized to 0). The general economic mechanism described in the prose is real Weitzman economics. But the prose presents a specific counterfactual prediction ("inside purchase share falls") that is load-bearing — it is the stated purpose of the policy experiment — and that prediction is not realized in any row of the committed output. The piece that is missing is not bad code; it is a calibration that puts the outside option within operational reach of the counterfactual grid.

- **Severity:** HIGH — the counterfactual experiment is framed as the tutorial's policy lesson. The prose tells the reader what to expect from the figure; the figure shows the opposite.

- **Result-changing:** yes — a reader of the Results section believes the published table demonstrates a falling inside-demand curve as search costs rise. The table shows a flat line at 1.0. Average searches does fall correctly (3.02 to 1.82), but the inside/outside split mechanism that the prose features is absent.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert all(row["Inside purchase share"] == 1.0 for _, row in cf.iterrows())
  # PASSES on current committed output; FAILS on any honest fix that produces outside > 0
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert cf.loc[cf["Search cost multiplier"] == 2.0, "Inside purchase share"].iloc[0] < 1.0
  # PASSES if high-cost multiplier produces outside buyers; FAILS on current output
  ```

---

### Findings 2-11: HOLDS

**Finding 2 — Weitzman reservation equation.**
Claim `README.md:60-65`: `c_j = E[max(u_ij - z_j, 0)]`. Code `run.py:28-30`: `expected_gain_over_threshold(k) = norm.pdf(k) - k*(1-norm.cdf(k))` — correct `E[max(Z-k, 0)]` for Z~N(0,1). `run.py:37`: `target = cost / match_sd` implements the standardized form. `run.py:43`: `z_j = mu_j + match_sd * k_star`. HOLDS.

**Finding 3 — Diagonal W via moment scaling.**
Claim `README.md:111-113`: "weighting matrix W is diagonal." Code `run.py:173, 160-161`: `scale = max(|target|, floor)`; `diff = (moments - target)/scale`; `diff @ diff` computes `sum((m_l-t_l)^2/a_l^2)` equivalent to `(m-t)^T diag(1/a^2) (m-t)`. HOLDS.

**Finding 4 — Search stopping rule.**
Claim `README.md:87-89`: stops when highest remaining z_j is below b_i. Code `run.py:77,86-89`: `order = argsort(-reservation)` descending; `active = reservation[product] > best_value` per consumer. Because order is descending and `best_value` is non-decreasing, once a consumer's `active` flag goes False it stays False for all subsequent (lower reservation) products. HOLDS.

**Finding 5 — Outside option through zero in b_i.**
Claim `README.md:83-84`: "outside option enters through the zero in b_i." Code `run.py:82-83`: `best_value = zeros(n_consumers)`, `best_product = full(n_consumers, n_products)`. Consumer takes outside option if no match value exceeds 0. Mechanism is coded correctly; see Finding 1 for why it is unreachable in this calibration. HOLDS.

**Finding 6 — Parameter recovery numbers match CSV.**
`README.md:305-309`: Quality taste True=1.18, Est=1.193, Error=0.013; Base cost True=0.08, Est=0.0791, Error=-0.0009; Complexity slope True=0.48, Est=0.48, Error=0. All values match `tables/parameter-recovery.csv:1-4` exactly. Error formula `run.py:203` gives 1.193-1.18=0.013 and 0.0791-0.08=-0.0009. HOLDS.

**Finding 7 — Moment fit numbers match CSV.**
All 13 rows of `README.md:313-327` match `tables/moment-fit.csv:1-14` exactly. HOLDS.

**Finding 8 — Counterfactual grid numbers match CSV.**
All 5 rows of `README.md:331-337` match `tables/search-cost-counterfactual.csv:1-6` exactly. HOLDS.

**Finding 9 — True parameter values in Model Setup match code.**
`README.md:127-129`: True quality taste 1.18, True base search cost 0.080, Complexity slope 0.48. Code `run.py:254`: `theta_true = array([1.18, log(0.08)])`; `run.py:7`: `COST_COMPLEXITY_SLOPE = 0.48`. `exp(log(0.08)) = 0.08`. HOLDS.

**Finding 10 — Fixed parameters held fixed.**
`README.md:115-117`: alpha=0.32, sigma=0.85, gamma=0.48 fixed. Code: `run.py:57` `price_taste = 0.32` hardcoded in `primitives()`; `run.py:253` `match_sd = 0.85`; `run.py:7` `COST_COMPLEXITY_SLOPE = 0.48`. None appear in estimated `theta = (beta, log_c0)` at `run.py:254`. HOLDS.

**Finding 11 — Fixed simulation draws.**
`README.md:242`: "fixed simulation shocks." Code `run.py:258`: `simulation_draws = default_rng(712).normal(size=(9000, 5))` seeded before optimization, passed as constant `draws` argument to all `criterion` calls. HOLDS.

## Cross-cutting patterns

- The single non-HOLDS finding is isolated to the counterfactual prose narrative. All mathematical derivations, algorithmic pseudocode, parameter values, and tabulated numbers are faithful to the code and CSV artifacts.
- The failing mechanism (outside option) is coded correctly but is calibrated away from the described effect. The calibration produces products with high positive mean utilities; even at 2x search cost, every consumer inspects at least 1.8 products on average and finds a positive match. The prose describes economics that is correct in principle but does not materialize in this simulation's output.
- The `continue` vs `break` in `run.py:88` is not a faithfulness violation but is a latent readability trap: the loop's correctness depends on the sort order of `order`. A future refactor that changes the loop order would silently break the stopping logic.

## TDD execution sequence (for the next agent)

0. Bullshit score = 35%. Below the 50% halt threshold. Notify user before proposing fixes; do not halt forward work.

1. Write failing test for Finding 1 (proves prose-table mismatch):
   ```python
   # tests/test_sequential_search_counterfactual.py — PASSES on current code
   def test_inside_share_never_falls(cf_table):
       assert all(row["Inside purchase share"] == 1.0 for _, row in cf_table.iterrows())
   ```

2. Write pass-condition test (fails on current code):
   ```python
   def test_inside_share_falls_at_2x_cost(cf_table):
       assert cf_table.loc[cf_table["Search cost multiplier"] == 2.0, "Inside purchase share"].iloc[0] < 1.0
   ```

3. Present two candidate fixes to user:
   - **Prose fix (no re-run):** Replace `README.md:297-299` with text that matches the actual result: average searches falls monotonically; the inside purchase share remains near one throughout this calibration because the high-quality calibration leaves no consumer unable to find a positive match. Economic message shifts from "inside share falls" to "search depth falls."
   - **Calibration fix (re-run required):** Extend counterfactual multipliers to 5x or 10x, or lower product quality values, until some consumers find `z_j < 0` for all products and take the outside option. Verify `tables/search-cost-counterfactual.csv` shows outside > 0 at the high end before updating prose.

4. After prose fix: verify `README.md:297-299` no longer claims inside share falls. After calibration fix: re-run `python run.py`, confirm CSV updates, re-run this skill.

5. Re-run this skill on the updated tutorial to confirm Finding 1 reads HOLDS and score drops to <= 10%.
