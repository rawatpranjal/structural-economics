# bullshit-detector -- nested-logit -- 2026-05-20

**Bullshit score: 65%** -- Three of four claimed instruments are product-level constants with zero cross-market variation; the system is underidentified (1 valid IV for 2 endogenous variables); the README instrument table is FALSE for three of four rows; sigma estimate is inconsistent and the 30% bias is a direct consequence.

## Header
- Claim sources: `choice/nested-logit/README.md` (prose, Equations, Results, Model Setup, instrument table)
- Code / artifact root: `choice/nested-logit/run.py`
- Data artifacts: `choice/nested-logit/tables/parameter-estimates.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, 2026-05-20)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Rival sugar, all products" instruments price with market variation | FALSE | HIGH | yes (zero variation; cost_shifter is only valid IV) |
| 2 | "Number of products in nest" instruments ln(s_{j|g}) | FALSE | HIGH | yes (always 2; collinear with intercept) |
| 3 | "Same-nest rival sugar" instruments ln(s_{j|g}) | FALSE | HIGH | yes (product-constant; no market variation) |
| 4 | "Nested logit recovers the parameters" (table heading) | DILUTED | MED | no (prose says "signs and ranking"; framing incomplete on 52% beta_const bias) |
| 5 | Share formulas D_g, s_{j|g}, s_g match code | HOLDS | -- | -- |
| 6 | Elasticity formulas (own, same-nest, cross-nest) match code | HOLDS | -- | -- |
| 7 | Diversion ratio formula matches code | HOLDS | -- | -- |
| 8 | Berry inversion LHS matches code | HOLDS | -- | -- |
| 9 | T=50, J=4, true params (alpha=1.5, sigma=0.7, etc.) match code | HOLDS | -- | -- |
| 10 | Table numbers match CSV | HOLDS | -- | -- |
| 11 | 2SLS coefficient extraction (alpha=-coeff[2], sigma=coeff[3]) | HOLDS | -- | -- |

## Findings

### Finding 1: "Rival sugar, all products" instrument has zero cross-market variation

- **Claim source (verbatim):** "Rival sugar, all products | Price | Summarizes rival characteristics in the market" -- `README.md:75`
- **Code evidence (verbatim):**
  ```python
  products = {
      ...
      "sugar": [10.0, 8.0, 1.0, 2.0],
      ...
  }
  # in generate_instruments():
  others = mkt[mkt["product_id"] != row["product_id"]]
  rival_sugar.append(others["sugar"].sum())
  ```
  `run.py:44` (sugar is a fixed constant per product) and `run.py:137-138` (instrument construction).

  Sugar is never randomized across markets. `rival_sugar_sum` for product j equals `sum_{k!=j} sugar_k` in every market -- a product-level constant. For Choco-Bombs: 8+1+2=11 in all 50 markets. This constant adds no identifying variation beyond the sugar regressor already in the model.

- **Data evidence:** `tables/parameter-estimates.csv:5`: `sigma,0.700,---,0.913`. The 30.4% upward bias is consistent with underidentification of ln(s_{j|g}).

- **Category:** FALSE -- claimed to summarize market-level rival characteristics; in fact it is a product fixed effect with zero cross-market variation.

- **Severity:** HIGH

- **Result-changing:** yes -- with `rival_sugar_sum` degenerate, cost_shifter is the only valid IV for two endogenous regressors (price and ln(s_{j|g})). The order condition fails. sigma is inconsistently estimated.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert df.groupby("product_id")["rival_sugar_sum"].std().max() == 0.0
  # PASSES on current code (zero within-product variation across markets); FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert df.groupby("product_id")["rival_sugar_sum"].std().min() > 0.0
  # PASSES on honest fix (instrument varies by market within product); FAILS on current code
  ```

---

### Finding 2: "Number of products in nest" instrument is identically 2 for all observations

- **Claim source (verbatim):** "Number of products in nest | ln(s_{j|g,t}) | Changes the local competitive set" -- `README.md:76`
- **Code evidence (verbatim):**
  ```python
  same_nest = mkt[mkt["nest_id"] == row["nest_id"]]
  num_in_nest.append(len(same_nest))
  ```
  `run.py:140-141`

  The DGP always assigns the same 4 products to 2 nests of 2. `num_in_nest` equals 2 for every row in every market. It is collinear with the intercept column already present in W.

- **Data evidence:** Same sigma bias as Finding 1 (`tables/parameter-estimates.csv:5`).

- **Category:** FALSE -- described as changing the local competitive set; it is a scalar constant across all 200 observations (50 markets x 4 products).

- **Severity:** HIGH

- **Result-changing:** yes -- this is one of two intended instruments for ln(s_{j|g}). With zero variation it provides no identifying information for sigma.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert df["num_in_nest"].nunique() == 1
  # PASSES on current code (always 2); FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert df["num_in_nest"].nunique() > 1
  # PASSES on honest fix (nest size varies across markets or products); FAILS on current code
  ```

---

### Finding 3: "Same-nest rival sugar" instrument has zero cross-market variation

- **Claim source (verbatim):** "Same-nest rival sugar | ln(s_{j|g,t}) | Moves the attractiveness of close substitutes" -- `README.md:77`
- **Code evidence (verbatim):**
  ```python
  same_nest_others = same_nest[same_nest["product_id"] != row["product_id"]]
  same_nest_rival_sugar.append(same_nest_others["sugar"].sum())
  ```
  `run.py:143-144`

  Sugar is product-fixed. `same_nest_rival_sugar` for product j equals the sugar of its sole same-nest competitor, constant across all markets. For Choco-Bombs: 8.0 in all 50 markets. For Store-Frosted: 10.0 in all 50 markets.

- **Data evidence:** Same sigma bias (`tables/parameter-estimates.csv:5`). Together with Finding 2, ln(s_{j|g}) has no valid instrument at all.

- **Category:** FALSE -- described as moving attractiveness of close substitutes across markets; it has zero cross-market variation.

- **Severity:** HIGH

- **Result-changing:** yes -- with both intended instruments for ln(s_{j|g}) being product-level constants, the system fails the order condition (2 endogenous regressors, 1 valid IV). sigma is not consistently identified.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert df.groupby("product_id")["same_nest_rival_sugar"].std().max() == 0.0
  # PASSES on current code; FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert df.groupby("product_id")["same_nest_rival_sugar"].std().min() > 0.0
  # PASSES on honest fix; FAILS on current code
  ```

---

### Finding 4: "Recovers the parameters" table framing is DILUTED

- **Claim source (verbatim):** "The table checks whether estimation recovers the parameters used to generate the synthetic shares." -- `README.md:93`
- **Code evidence (verbatim):**
  ```python
  table_data = {
      "Parameter": [r"alpha", r"beta_sugar", r"beta_const", r"sigma"],
      "True": [f"{TRUE_ALPHA:.3f}", f"{TRUE_BETA_SUGAR:.3f}",
               f"{TRUE_BETA_CONST:.3f}", f"{TRUE_SIGMA:.3f}"],
      "Logit": [...],
      "Nested Logit": [f"{nested_res['alpha']:.3f}", f"{nested_res['beta_sugar']:.3f}",
                       f"{nested_res['beta_const']:.3f}", f"{nested_res['sigma']:.3f}"],
  }
  ```
  `run.py:629-637`

- **Data evidence (verbatim from `tables/parameter-estimates.csv`):**
  ```
  beta_const,1.000,-0.034,1.518
  sigma,0.700,---,0.913
  ```
  `tables/parameter-estimates.csv:4-5`

  `beta_const`: true=1.000, estimated=1.518 (+52%). `sigma`: true=0.700, estimated=0.913 (+30.4%). The surrounding prose at `README.md:93` is careful ("recovers the signs and the same-nest ranking"), but the table heading says "recovers the parameters" without flagging that two of four parameters show large bias. No sentence in the README warns the reader of finite-sample bias or weak instruments.

- **Category:** DILUTED -- the narrow prose claim ("signs and ranking") is technically defensible; the table heading framing is not.

- **Severity:** MED -- the numbers are displayed; a careful reader can see the bias. But the framing primes interpretation as a success story.

- **Result-changing:** no -- the table values are correct; only the framing is incomplete.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(float(open("tables/parameter-estimates.csv").readlines()[4].split(",")[3].strip()) - 0.700) < 0.05
  # PASSES if sigma were well-recovered; FAILS on current code (gap = 0.213)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "bias" in open("README.md").read().lower() or "finite-sample" in open("README.md").read().lower()
  # PASSES when README explicitly warns reader; FAILS on current README
  ```

---

## Cross-cutting patterns

- **Three degenerate instruments from one root cause.** Sugar is a product-level constant in the DGP (`run.py:44`). Any instrument derived from rival sugar values (rival_sugar_sum, same_nest_rival_sugar) inherits zero cross-market variance. Any instrument derived from nest composition (num_in_nest) is constant because nest membership is fixed. All three degenerate instruments trace back to the same DGP decision: sugar and nest membership do not vary across markets. Adding cross-market variation to any one characteristic would fix multiple instruments simultaneously.

- **Underidentification masked by pinv.** `np.linalg.pinv` (`run.py:165`) handles the near-singular W.T @ W without error or warning. The 2SLS runs, produces estimates, and the code never warns the user that the instrument set is degenerate. A first-stage F-statistic check (standard weak-instrument diagnostic) would expose this immediately and is a natural pedagogical addition for a tutorial on IV demand estimation.

- **sigma bias is mechanically explainable.** The 30.4% upward bias in sigma (0.913 vs 0.700) is consistent with the identification failure: with cost_shifter as the only valid IV, the first stage for ln(s_{j|g}) captures only an indirect channel (cost_shock -> prices -> delta -> within-nest shares), not a direct channel targeting the nesting structure. The estimate is biased but the sign is preserved, which explains why the README's narrow claim ("signs and ranking") holds while "recovers the parameters" does not.

- **The instrument failure also applies to the logit estimation.** `run.py:193` shows the logit uses `Z = df[["cost_shifter", "rival_sugar_sum"]].values`. Since `rival_sugar_sum` is a product constant, the logit is also using a degenerate second instrument, though it has only one endogenous variable (price), so identification from cost_shifter alone holds. The logit estimates are consistent; the nested logit estimates are not.

## TDD execution sequence (for the next agent)

0. **Read the bullshit score first.** Score is 65% -- HALT code work, surface to user before proposing fixes.

1. **Finding 1:** Write `tests/test_instruments.py::test_rival_sugar_has_market_variation`. Assert `df.groupby("product_id")["rival_sugar_sum"].std().max() == 0.0`. Run -- confirm PASSES on current code.

2. **Finding 2:** Write `test_num_in_nest_varies`. Assert `df["num_in_nest"].nunique() == 1`. Run -- confirm PASSES.

3. **Finding 3:** Write `test_same_nest_rival_has_market_variation`. Assert `df.groupby("product_id")["same_nest_rival_sugar"].std().max() == 0.0`. Run -- confirm PASSES.

4. Write the three honest-fix pass conditions as second tests -- all should FAIL on current code. This is the red/green spec.

5. Hand off to `writing-plans` to design the fix. Two candidate approaches: (a) add market-varying characteristics to the DGP -- e.g., market-level sugar content shocks or random product entry/exit changing nest size; (b) replace the three degenerate instruments with BLP-style alternatives -- e.g., sum of other markets' prices for the same product (if panel is balanced), or number of firms competing in a broader geographic market. Option (a) is simpler for a pedagogical tutorial and preserves the existing code structure.

6. After fix: re-run `python run.py`, confirm sigma estimate moves closer to 0.700, re-run this skill, confirm score drops to <= 25%.
