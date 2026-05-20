# bullshit-detector -- dynamic-games-estimation -- recheck -- 2026-05-20

**Bullshit score: 5%** -- both original findings resolved; entropy term now explicit in the Equations equation; prose RMSE now printed at 5 decimal places, matching CSV to within 0.3%.

## Header
- Claim sources: `industrial-organization/dynamic-games-estimation/README.md`
- Code / artifact root: `industrial-organization/dynamic-games-estimation/run.py`
- Data artifacts: `tables/parameter-recovery.csv`, `tables/estimator-diagnostics.csv`
- Seed audit: `bullshit-detector_dynamic-games-estimation_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | W_theta equation discloses entropy + Euler-gamma correction | HOLDS | none | no |
| 2 | Model RMSE prose value matches CSV within 0.5% | HOLDS | none | no |
| 3 | flow_payoff = theta_q*q_i - theta_c*a_i + theta_g*max{q_j-q_i,0}*a_i | HOLDS | none | no |
| 4 | Choice-specific values use rival's first-stage CCP and W_theta | HOLDS | none | no |
| 5 | Pseudo likelihood is binary logit scored on invest/no-invest | HOLDS | none | no |
| 6 | First-stage CCP uses Laplace smoothing | HOLDS | none | no |
| 7 | True theta_q=0.70, theta_g=0.40, theta_c=1.00 | HOLDS | none | no |
| 8 | Parameter recovery errors: 0.00387, 0.00502, -0.00115 | HOLDS | none | no |
| 9 | 591 MPE iterations, 72 second-stage iterations | HOLDS | none | no |
| 10 | 60,000 firm-period observations | HOLDS | none | no |
| 11 | Forward simulation RMSE 0.045, 1,000 paths, 70-period horizon | HOLDS | none | no |
| 12 | Symmetric competitors, soft MPE for truth solve | HOLDS | none | no |

## Findings

### Finding 1 (RESOLVED): W_theta equation entropy disclosure

**Original finding:** Equation omitted the `H(p) + gamma` entropy correction that `policy_transition_and_flow` adds at `run.py:151,161`.

**Recheck evidence (verbatim):**
```
README.md:30-34
$$\bar\pi_\theta(\omega;\hat p) = (1-\hat p)\,\pi_i(\omega,0;\theta) + \hat p\,\pi_i(\omega,1;\theta) + H(\hat p) + \gamma,$$

where $H(\hat p) = -\hat p\log\hat p-(1-\hat p)\log(1-\hat p)$ is the Bernoulli
entropy of the investment rate and $\gamma$ is the Euler-Mascheroni constant.
The $H(\hat p)+\gamma$ term is the expected value of the Type-I extreme value
shock under the first-stage policy.
```

Code at `run.py:151`: `entropy = -(ccp * np.log(ccp) + (1.0 - ccp) * np.log(1.0 - ccp)) + EULER_GAMMA`

Equation now explicitly shows `H(\hat p) + \gamma` with full definition. Code and equation agree.

- **Category:** HOLDS
- **Verdict:** RESOLVED

---

### Finding 2 (RESOLVED): Model RMSE prose precision

**Original finding:** Prose printed `0.001` at 3dp (26% understatement of `0.0013543`).

**Recheck evidence (verbatim):**
- `run.py:445`: `f"The model-implied RMSE against truth is **{policy_rmse:.5f}**."` -- format changed from `.3f` to `.5f`
- `README.md:93`: `"The model-implied RMSE against truth is **0.00135**."` -- `f"{0.0013543:.5f}"` = `"0.00135"` (5 decimal places, rounds correctly)
- `tables/estimator-diagnostics.csv:8`: `"Model policy RMSE,0.0013543355175390252"`
- Discrepancy: `|0.00135 - 0.001354| = 0.0000043` -- 0.3% relative error, well within rounding tolerance

- **Category:** HOLDS
- **Verdict:** RESOLVED

---

## Cross-cutting patterns

None. Both original findings fixed. No new findings identified.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5%.** No action required. Both original findings read HOLDS.
1. `test_fixed_equation_discloses_entropy_term` PASSES -- confirms Finding 1 resolved.
2. `test_fixed_model_rmse_matches_csv` PASSES -- confirms Finding 2 resolved.
3. `test_violated_code_adds_entropy_not_in_equation` FAILS -- correct (violated-invariant test designed to fail after fix; failure confirms fix landed).
4. `test_violated_model_rmse_understated_in_prose` FAILS -- correct (violated-invariant test designed to fail after fix; failure confirms prose precision improved).
5. No further work needed on this tutorial.
