# bullshit-detector - neural-posterior-brock-hommes - 2026-05-20

**Bullshit score: 35%** - One FALSE claim in the Takeaway misstates NPE's simulation cost relative to SMM by a factor of ~42x, misleading readers on cost-effectiveness; all other claims HOLD.

## Header
- Claim sources: `bayesian-methods/neural-posterior-brock-hommes/README.md`
- Code / artifact root: `bayesian-methods/neural-posterior-brock-hommes/run.py`, `lib/brock_hommes.py`
- Data artifacts: `bayesian-methods/neural-posterior-brock-hommes/tables/posterior-summary.csv`, `bayesian-methods/neural-posterior-brock-hommes/tables/posterior-predictive.csv`
- Seed audit (if any): None
- Run by: bullshit-detector subagent (claude-sonnet-4-6), 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "roughly the same simulation budget the SMM tutorial uses for a single parameter" | FALSE | HIGH | yes - overstates NPE cost-efficiency by ~42x |
| 2 | Score-smoothing equation U_{h,t} = lambda*U_{h,t-1} + (1-lambda)*rho_{h,t} | HOLDS | - | - |
| 3 | Logit share n_{h,t} = exp(beta*U) / sum exp(beta*U) | HOLDS | - | - |
| 4 | NPE loss L(phi) = E[-log q_phi(theta|y)] | HOLDS | - | - |
| 5 | Masked autoregressive flow, Adam + early stopping | HOLDS | - | - |
| 6 | 4-parameter prior box with stated bounds and true values | HOLDS | - | - |
| 7 | 5 summary statistics as listed | HOLDS | - | - |
| 8 | y_obs averaged over 4 independent simulations | HOLDS | - | - |
| 9 | All four CIs cover truth, lie inside prior support | HOLDS | - | - |
| 10 | Table numbers match committed CSV files | HOLDS | - | - |
| 11 | Toy Laplace Z~1.979, densities at mu=1,0,-1 | HOLDS | - | - |
| 12 | SMM and NPE share 3 of 5 summary statistics | HOLDS | - | - |
| 13 | SMM CI covers both beta_hat=26 and truth=30 | HOLDS | - | - |
| 14 | sigma_eps is best-identified (tightest posterior) | HOLDS | - | - |
| 15 | c_T posterior tracks the prior across most of the box | HOLDS | - | - |

## Findings

### Finding 1: "roughly the same simulation budget the SMM tutorial uses for a single parameter"

- **Claim source (verbatim):** "Neural posterior estimation handles a four-parameter Brock-Hommes calibration at roughly the same simulation budget the SMM tutorial uses for a single parameter." - `bayesian-methods/neural-posterior-brock-hommes/README.md:269`

- **Code evidence (verbatim):**

  NPE training budget (`run.py:48`):
  ```python
  N_TRAIN = 10_000
  ```

  SMM grid in `smm_grid_for_beta` (`run.py:282-299`):
  ```python
  candidate_betas = np.arange(2.0, 62.0, 2.0)
  data_rng = np.random.default_rng(2028)
  sim_rng = np.random.default_rng(2029)
  pseudo_data_shocks = data_rng.normal(0.0, base.shock_sigma, size=(8, base.periods))
  smm_shocks = sim_rng.normal(0.0, base.shock_sigma, size=(8, base.periods))
  target = average_moments(true_beta, base, pseudo_data_shocks)
  ...
  for beta in candidate_betas:
      fitted = average_moments(float(beta), base, smm_shocks)
  ```

  SMM sibling (`agent-based-models/brock-hommes-asset-pricing/run.py:20-25`):
  ```python
  candidate_betas = np.arange(2.0, 62.0, 2.0)
  pseudo_data_shocks = data_rng.normal(0.0, params.shock_sigma, size=(8, params.periods))
  smm_shocks = sim_rng.normal(0.0, params.shock_sigma, size=(8, params.periods))
  target = average_moments(true_beta, params, pseudo_data_shocks)
  for beta in candidate_betas:
      fitted = average_moments(float(beta), params, smm_shocks)
  ```

- **Data evidence:** Not applicable (budget is a code-parameter claim, not a results table claim).

- **Category:** FALSE

- **Severity:** HIGH

- **Result-changing:** yes - the claim is the primary cost-benefit argument for NPE in the Takeaway. The actual counts are: NPE runs 10,000 complete BH simulations of T=700 periods (7,000,000 period-draws total). The SMM tutorial in `agent-based-models/brock-hommes-asset-pricing/` runs 30 candidate betas * 8 shock banks + 8 pseudo-data sims = 248 complete BH simulations (173,600 period-draws total). The ratio is 10,000/248 = 40.3x. On a per-parameter basis: SMM uses 248 sims for 1 parameter; NPE uses 10,000 for 4, i.e. 2,500/param - still 10x more per parameter. No charitable reading makes these budgets "roughly the same."

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(10_000 / 248 - 1.0) < 0.5  # PASSES on buggy claim (if "same" means within 50%); FAILS since ratio=40.3
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "NPE uses roughly 40x more simulator calls" in open("bayesian-methods/neural-posterior-brock-hommes/README.md").read() or "10,000" in open("bayesian-methods/neural-posterior-brock-hommes/README.md").read().split("Takeaway")[1]
  ```

## Cross-cutting patterns

- This tutorial is unusually clean. All 14 non-budget claims cross-check to source in either `run.py` or `lib/brock_hommes.py` without dilution or mislabeling.
- The single false claim is in the Takeaway only (a qualitative prose section), not in the Equations or Results tables. The numeric results tables (`tables/posterior-summary.csv`, `tables/posterior-predictive.csv`) match the README display exactly.
- The toy Laplace worked example is mathematically correct: Z=1.9792, posterior densities at mu=1,0,-1 match the stated ~0.505, ~0.186, ~0.068 (verified by direct computation).
- The sigma_eps pseudocode notation `N(0, sigma_eps^2 I)` is conventional (variance parameterization), consistent with `numpy.normal(0, sigma_eps)` (std parameterization) - no finding.
- The budget false claim is the exact pattern the Takeaway's final sentence is structured to make: a comparison of NPE's power against SMM's cheapness that inverts the actual cost relationship.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 35% - one HIGH FALSE finding.** Surface to user before touching code. The finding is in the Takeaway prose only; the Results tables and Equations are clean.

1. Write a test confirming the budget ratio:
   ```python
   # tests/test_npe_budget_claim.py
   def test_budget_ratio():
       N_TRAIN = 10_000
       smm_sims = 30 * 8 + 8  # 30 candidate betas * 8 banks + 8 pseudo-data
       ratio = N_TRAIN / smm_sims
       assert ratio > 30  # proves the "same budget" claim is false
   ```

2. Honest-fix pass condition: the Takeaway must state the actual budget (N=10,000 vs ~248 SMM simulations, ~40x difference) or remove the budget comparison entirely.

3. Fix: revise `run.py:730-739` `add_takeaway(...)` string to either (a) remove the budget comparison, (b) state that NPE uses ~40x more simulator calls than the SMM grid search while returning a full joint posterior, or (c) compare against the total SMM cost if the full beta search were treated as the budget.

4. After fix, re-run `python run.py` to regenerate `README.md` and re-run `scripts/validate_catalog.py`.

5. Re-run this skill on the updated README. Score should drop to 0-10% (all HOLDS, no false claims).
