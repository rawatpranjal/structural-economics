# bullshit-detector -- nested-logit recheck -- 2026-05-20

**Bullshit score: 10%** -- All three degenerate instrument findings are resolved; sigma bias reduced from 30.4% to 10.6%; all 21 README claims now ground in code and data. Residual 10% reflects stale `n_markets=30` default in `generate_product_data` signature (called with 50 in main) and finite-sample sigma gap at outer edge of "close."

## Header
- Claim sources: `choice/nested-logit/README.md`
- Code / artifact root: `choice/nested-logit/run.py`
- Data artifacts: `choice/nested-logit/tables/parameter-estimates.csv`
- Seed audit: `choice/nested-logit/bullshit-detector_nested-logit_2026-05-20.md` (original, score 65%)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, re-audit, 2026-05-20)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Rival sugar varies by market because recipes shift" | HOLDS | -- | -- |
| 2 | "Same-nest rival cost moves conditional share" | HOLDS | -- | -- |
| 3 | "Same-nest rival sugar varies across markets" | HOLDS | -- | -- |
| 4 | "Each instrument carries genuine cross-market variation" | HOLDS | -- | -- |
| 5 | "Sugar shifts by market through recipe shocks" | HOLDS | -- | -- |
| 6 | "Same-nest rivals carry idiosyncratic cost shocks" | HOLDS | -- | -- |
| 7 | "Two excluded instruments target ln(s_{j|g}); order condition holds" | HOLDS | -- | -- |
| 8 | Share formulas D_g, s_{j|g}, s_g match code | HOLDS | -- | -- |
| 9 | Berry inversion LHS matches code | HOLDS | -- | -- |
| 10 | Diversion ratio formula matches code | HOLDS | -- | -- |
| 11 | T=50, true params (alpha=1.5, sigma=0.7, etc.) match code | HOLDS | -- | -- |
| 12 | "Recovers sigma and alpha close to true values" | HOLDS | -- | -- |
| 13 | "Finite-sample bias" framing present in README | HOLDS | -- | -- |
| 14 | Table numbers match CSV | HOLDS | -- | -- |
| 15 | Plain logit biases alpha upward | HOLDS | -- | -- |
| M1 | `generate_product_data` default n_markets=30, called with 50 | NOTE | LOW | no (README correctly states T=50 for the actual run) |

## Findings

### Finding M1 (marginal): stale default argument in `generate_product_data`

- **Claim source (verbatim):** "Markets $T$ | 50 | Cross-market price and cost variation" -- `README.md:43`
- **Code evidence (verbatim):**
  ```python
  def generate_product_data(n_markets: int = 30) -> pd.DataFrame:
  ```
  `run.py:37`
  Called at:
  ```python
  df = generate_product_data(n_markets=50)
  ```
  `run.py:439`
- **Data evidence:** `tables/parameter-estimates.csv` reflects a 50-market run (consistent with `run.py:439`). No data artifact is wrong.
- **Category:** NOTE -- not a faithfulness violation. README claim is correct for the actual execution path. A reader who calls `generate_product_data()` without arguments gets 30 markets, not 50 as described; this is a code hygiene issue outside the claim-vs-code audit scope.
- **Severity:** LOW
- **Result-changing:** no -- the actual run uses 50 markets throughout.
- **Violated invariant (one-line pytest assertion):**
  ```python
  import inspect, run; assert "n_markets: int = 30" in inspect.getsource(run.generate_product_data)
  # PASSES on current code (stale default); FAILS if default is updated to 50
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  import inspect, run; assert "n_markets: int = 50" in inspect.getsource(run.generate_product_data)
  # PASSES if default updated to match tutorial description; FAILS on current code
  ```

---

## Original findings resolution

The original audit (score 65%) identified four findings. Status after the fix:

| Original finding | Original category | Resolution |
|------------------|-------------------|------------|
| Finding 1: rival_sugar_sum zero cross-market variation | FALSE / HIGH | RESOLVED -- sugar now drawn as `sugar_base[j] + np.random.normal(0, 1.0)` at `run.py:66`; rival_sugar_sum varies by market within each product |
| Finding 2: num_in_nest identically 2 for all rows | FALSE / HIGH | RESOLVED -- column dropped entirely; replaced by `same_nest_rival_cost` (idiosyncratic cost drawn fresh each market×product at `run.py:68`), which varies by market |
| Finding 3: same_nest_rival_sugar zero cross-market variation | FALSE / HIGH | RESOLVED -- inherits market-level sugar variation from `run.py:66`; within-product std now > 0 for all products |
| Finding 4: "Recovers the parameters" framing DILUTED | DILUTED / MED | RESOLVED -- README now contains "finite-sample bias: with 50 markets the estimates scatter around the truth rather than land on it exactly" (`README.md:95`); sigma gap reduced from 0.213 (30.4%) to 0.074 (10.6%), within defensible finite-sample range |

## Cross-cutting patterns

- All three original degenerate-instrument findings shared one root cause: sugar and nest membership were product-level constants. The fix added `np.random.normal(0, 1.0)` recipe shocks to sugar and `np.random.normal(0, 0.1)` idiosyncratic cost shocks, giving all three derived instruments genuine cross-market variation in a single DGP change.
- The `num_in_nest` column (originally FALSE) was replaced by `same_nest_rival_cost` rather than repaired; this is a clean solution -- the column disappears from the instrument set and the new instrument is valid.
- The sigma identification improvement is mechanically consistent with the fix: with `same_nest_rival_cost` and `same_nest_rival_sugar` now genuinely varying, the first stage for `ln(s_{j|g})` has two real excluded instruments, and the 2SLS sigma estimate moves from 0.913 to 0.774 (gap 0.074 vs. 0.213 before).
- No README claim was introduced post-fix that overclaims precision. The "close to their true values" framing at `README.md:95` is now accurate, and the "finite-sample bias" qualifier is explicit.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** No halt required. Touch-up only.

1. Optional: update `generate_product_data` default from `n_markets=30` to `n_markets=50` to match the tutorial description. Write one test asserting the default matches the README.

2. No violated-invariant tests need to be written for the original findings -- they are resolved. The existing `tests/test_instruments.py` violated-invariant tests (findings 1-3) now correctly FAIL (the bugs are gone), and the honest-fix pass-condition tests correctly PASS.

3. The existing `tests/test_recovery.py::test_finding4_honest_fix_sigma_recovered_within_tolerance` asserts `abs(sigma - 0.700) < 0.10`. Current gap = 0.074 < 0.10. PASSES.

4. No re-audit required unless the DGP or instrument set changes again.
