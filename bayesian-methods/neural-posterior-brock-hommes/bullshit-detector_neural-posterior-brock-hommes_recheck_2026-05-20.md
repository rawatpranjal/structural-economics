# bullshit-detector - neural-posterior-brock-hommes - recheck - 2026-05-20

**Bullshit score: 5%** - All prior FALSE finding is resolved; all 18 audited claims HOLD. Residual 5% reflects one minor observable: the c_T "tracks the prior across most of the box" characterisation is borderline (89% of prior CI width is wide but the claim is directionally accurate and result-interpretive, not a fabricated number).

## Header
- Claim sources: `bayesian-methods/neural-posterior-brock-hommes/README.md`
- Code / artifact root: `bayesian-methods/neural-posterior-brock-hommes/run.py`, `lib/brock_hommes.py`
- Data artifacts: `bayesian-methods/neural-posterior-brock-hommes/tables/posterior-summary.csv`, `bayesian-methods/neural-posterior-brock-hommes/tables/posterior-predictive.csv`
- Seed audit: `bayesian-methods/neural-posterior-brock-hommes/bullshit-detector_neural-posterior-brock-hommes_2026-05-20.md` (original, 35%)
- Run by: independent verification agent (claude-sonnet-4-6), 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Score smoothing U = lambda*U_prev + (1-lambda)*rho | HOLDS | - | - |
| 2 | Logit shares n = exp(beta*U) / sum exp(beta*U) | HOLDS | - | - |
| 3 | NPE loss L(phi) = E[-log q_phi(theta\|y)] | HOLDS | - | - |
| 4 | MAF + Adam + early stopping on validation split | HOLDS | - | - |
| 5 | Prior box: beta U(1,60), g U(0.5,2.0), sigma_eps U(0.005,0.05), c_T U(0,0.005) | HOLDS | - | - |
| 6 | True values: beta=30, g=1.40, sigma_eps=0.02, c_T=0.001 | HOLDS | - | - |
| 7 | 5 summary statistics s0-s4 as listed | HOLDS | - | - |
| 8 | y_obs = average over 4 independent simulations at true params | HOLDS | - | - |
| 9 | Training size N = 10,000 | HOLDS | - | - |
| 10 | All four CIs cover truth and lie inside prior support | HOLDS | - | - |
| 11 | Toy Laplace Z~1.979; densities at mu=1,0,-1 ~ 0.505, 0.186, 0.068 | HOLDS | - | - |
| 12 | SMM uses 248 simulations | HOLDS | - | - |
| 13 | NPE uses 10,000 sims, ~40x more than SMM's 248 | HOLDS | - | - |
| 14 | NPE beta CI covers SMM beta_hat=26 and truth=30 | HOLDS | - | - |
| 15 | sigma_eps tightest posterior (best-identified) | HOLDS | - | - |
| 16 | c_T posterior tracks prior across most of the box | HOLDS | - | - |
| 17 | Table numbers match committed CSVs | HOLDS | - | - |
| 18 | SMM and NPE share 3 of 5 summary statistics | HOLDS | - | - |

## Findings

### Finding 1 (ORIGINAL, NOW RESOLVED): "roughly the same simulation budget the SMM tutorial uses for a single parameter"

- **Original claim source (verbatim):** "Neural posterior estimation handles a four-parameter Brock-Hommes calibration at roughly the same simulation budget the SMM tutorial uses for a single parameter." - original `README.md:269`

- **Status after fix:** RESOLVED. The Takeaway now reads (README.md:269, recheck):
  > "Training the masked autoregressive flow consumes 10,000 model simulations, against the 248 the SMM tutorial spends to point-identify a single intensity-of-choice parameter, roughly 40x more simulator calls."

- **Code evidence confirming fix (verbatim):**
  ```python
  # run.py:730-742
  report.add_takeaway(
      "Neural posterior estimation buys a full four-parameter "
      "Brock-Hommes posterior, but it is not cheap in simulator calls. "
      "Training the masked autoregressive flow consumes 10,000 model "
      "simulations, against the 248 the SMM tutorial spends to point-"
      "identify a single intensity-of-choice parameter, roughly 40x more "
      "simulator calls. ..."
  )
  ```

- **Arithmetic verified:**
  - NPE: `N_TRAIN = 10_000` (`run.py:48`)
  - SMM: `candidate_betas = np.arange(2.0, 62.0, 2.0)` = 30 values; `30 * 8 + 8 = 248` simulations (`run.py:282-299`)
  - Ratio: `10000 / 248 = 40.32` - matches "roughly 40x" claim exactly

- **Category:** HOLDS (post-fix)
- **Severity:** none
- **Result-changing:** no (fixed)

## Cross-cutting patterns

- The tutorial is clean post-fix. All 18 claims ground to code or data without exception.
- Numeric tables in the README display truncated versions of CSV values; all truncations are correct to the displayed decimal places (verified for beta mean, beta sd, sigma_eps mean, std-of-returns observed value).
- The score-smoothing implementation (`lib/brock_hommes.py:100`) uses `params.memory` (= 0.80) as lambda and `period_profit` as rho, matching the README equation exactly.
- The logit-share implementation uses numerically-stable softmax (`z = beta * (scores - max(scores))`), which is algebraically equivalent to the stated formula.
- The Laplace toy example is mathematically exact: Z = 1.9792, p(mu=1) = 0.5053, p(mu=0) = 0.1859, p(mu=-1) = 0.0684 - all within the stated approximations (0.505, 0.186, 0.068).
- The sigma_eps "tightest posterior" claim is verified quantitatively: relative posterior sd / prior range = 0.017 for sigma_eps vs 0.040 (g), 0.147 (beta), 0.249 (c_T).
- The c_T "tracks the prior" characterisation is directionally accurate: the 95% CI spans 89% of the prior box width, which is the widest of the four parameters. The phrasing is interpretive but grounded.
- The SMM-shares-3-of-5-stats claim is confirmed: `moments()` in `lib/brock_hommes.py` uses {volatility, abs_return_autocorr, excess_kurtosis}, all three of which are the first three entries of `summary_statistics()`.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5% - no actionable findings.** All original findings now HOLD. No code changes required.

1. The original violated-invariant test `test_violated_invariant_takeaway_claims_same_budget` correctly fails on the fixed code (buggy phrase "roughly the same simulation budget" is gone). This is expected and correct post-fix behavior.

2. Both honest-fix tests pass:
   - `test_honest_fix_takeaway_states_real_ratio`: "40x" present in Takeaway, "roughly the same simulation budget" absent.
   - `test_honest_fix_takeaway_cites_concrete_counts`: "10,000" and "248" both present in Takeaway.

3. No further action required. Re-run `scripts/validate_catalog.py` before merging if the README was regenerated from `run.py` as part of the fix.
