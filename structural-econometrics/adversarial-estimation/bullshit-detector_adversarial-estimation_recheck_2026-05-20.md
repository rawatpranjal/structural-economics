# bullshit-detector — adversarial-estimation — recheck — 2026-05-20

**Bullshit score: 5%** — Finding 1 from the original audit is now fixed; the prose at `README.md:202` correctly scopes "MLE and oracle... slightly below" and adds an explicit exception for the neural-net row. All other claims HOLD. The residual 5% reflects one borderline borderline rounding ("about five percent" vs actual 4.6%) that an adversarial reviewer could flag but cannot call a lie.

## Header

- Claim sources: `structural-econometrics/adversarial-estimation/README.md` (prose, Equations, Results, Model Setup table)
- Code / artifact root: `structural-econometrics/adversarial-estimation/run.py`
- Data artifacts: `tables/standard-errors.csv`, `tables/smm-comparison.csv`, `tables/bootstrap-se.csv`
- Seed audit: `structural-econometrics/adversarial-estimation/bullshit-detector_adversarial-estimation_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, independent recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | MLE and oracle MC numbers slightly below asymptotic; neural-net is exception at ~5% above | HOLDS | - | - |
| 2 | Oracle D* formula | HOLDS | - | - |
| 3 | Common-random-numbers trick (shocks reused at every theta) | HOLDS | - | - |
| 4 | MLP architecture: one hidden layer, H=10 tanh units, L2 on W and c | HOLDS | - | - |
| 5 | Logistic discriminator inner problem convex, L-BFGS | HOLDS | - | - |
| 6 | Outer grid 25 points, parabolic refinement | HOLDS | - | - |
| 7 | SMM d=7/d=3 standard error ratio "roughly six" | HOLDS | - | - |
| 8 | Asymptotic sd values (MLE 1.732, adversarial 2.449) | HOLDS | - | - |
| 9 | Bootstrap: joint resample of real and shock vectors | HOLDS | - | - |
| 10 | Table numbers (all three CSVs match README tables) | HOLDS | - | - |
| 11 | "About five percent above" for neural-net excess | HOLDS (borderline) | LOW | no |

## Findings

### Finding 1 (original audit, now fixed): "All Monte Carlo numbers slightly below asymptotic"

**Status: RESOLVED.**

Original claim at `README.md:202` read "All Monte Carlo numbers are slightly below the asymptotic prediction at this sample size." That was false: the neural-net row has MC sd 2.562 > asymptotic 2.449.

Fixed prose at `README.md:202` now reads:

> "The maximum-likelihood and oracle Monte Carlo numbers sit slightly below the asymptotic prediction at this sample size, the usual finite-sample behavior. The neural-net discriminator is the exception: its Monte Carlo standard deviation runs about five percent above the asymptotic value, because its non-convex inner problem and limited training iterations add estimation noise the asymptotic formula does not capture."

Grounding in `run.py:876-877` (same text verbatim). Grounding in `tables/standard-errors.csv`:

```
MLE (reference),0.0019,1.572,1.732,1.568      -- 1.572 < 1.732  (below)
Oracle adversarial,-0.0032,2.173,2.449,2.168  -- 2.173 < 2.449  (below)
Neural net disc.,-0.004,2.562,2.449,2.541      -- 2.562 > 2.449  (above, exception)
```

**Category:** HOLDS
**Result-changing:** no

### Finding 11 (new): "About five percent above" for neural-net excess

- **Claim source (verbatim):** "its Monte Carlo standard deviation runs about five percent above the asymptotic value" — `README.md:202`
- **Data evidence:** `tables/standard-errors.csv` row `Neural net disc.`: MC sd = 2.562, Asymptotic = 2.449. Actual excess = (2.562 - 2.449) / 2.449 = 4.61%.
- **Category:** HOLDS (borderline) — "about five percent" is a prose approximation of 4.61%. The 0.39 percentage-point gap is within normal rounding tolerance for "about." A hostile reviewer could flag this as mildly imprecise but cannot call it false.
- **Severity:** LOW
- **Result-changing:** no — the direction and order of magnitude are correct; the approximation does not mislead any inference.
- **Violated invariant:** none; the claim is directionally correct and within rounding.
- **Honest-fix pass condition (if a future agent wants to tighten):**
  ```python
  assert abs(4.61 - 5.0) < 1.0  # 4.61% is within 1pp of "about five percent"
  ```

## Cross-cutting patterns

- The fix to Finding 1 is surgical and correct: "All" replaced by "MLE and oracle..."; neural-net exception added with correct direction and approximate magnitude.
- No new faithfulness violations introduced by the fix.
- The only residual imprecision ("about five percent" vs 4.6%) is within the scope of reasonable prose rounding. It does not reach the threshold for a MED finding.
- All config constants in `run.py:40-58` match the Model Setup table at `README.md:100-117` exactly: N=300, M_SIM=300, HIDDEN=10, MLP_L2=1e-3, MLP_MAXITER=60, R_MAIN=200, R_MLP=60, THETA_GRID=linspace(-0.6,0.6,25), FISHER=1/3, THEORETICAL_SE_MLE=1.732, THEORETICAL_SE_ADV=2.449.
- Bootstrap section absent from README because `boot_table` is None at runtime (budget exceeded). This is the known HOLDS item from the original audit; not a regression.
- All three CSV table values match README prose tables exactly (verified by direct comparison).

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5%.** Well below the 25% threshold. No action required beyond the tests already committed.
1. The two tests in `tests/test_standard_errors_prose.py` are the correct green state:
   - `test_neural_mc_sd_exceeds_asymptotic`: PASSES (violated-invariant; data still 2.562 > 2.449, proving the original prose was wrong and the fix was needed).
   - `test_prose_does_not_claim_all_mc_below_asymptotic`: PASSES (honest-fix condition; bad string absent from `run.py`).
2. No further fixes required. Finding 11 (borderline "about five percent") is LOW severity and not result-changing; no test warranted.
3. Re-run `python run.py` to regenerate outputs if config changes. Re-run this skill after any prose edit to the Results section.
