# bullshit-detector -- dynamic-games-estimation -- 2026-05-20

**Bullshit score: 25%** -- W_theta equation in Equations section omits the entropy correction that the code actually adds; a replicator following the equation gets wrong continuation values.

## Header
- Claim sources: `industrial-organization/dynamic-games-estimation/README.md` (Equations, Results, Model Setup sections)
- Code / artifact root: `industrial-organization/dynamic-games-estimation/run.py`
- Data artifacts: `tables/parameter-recovery.csv`, `tables/estimator-diagnostics.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | flow_payoff = theta_q*q_i - theta_c*a_i + theta_g*max{q_j-q_i,0}*a_i | HOLDS | none | no |
| 2 | W_theta = pi_bar(p_hat) + beta P_hat W_theta (logit shock integrated out) | DILUTED | MED | no (code correct; equation incomplete for replication) |
| 3 | Choice-specific values use rival's first-stage CCP and W_theta | HOLDS | none | no |
| 4 | Pseudo likelihood is binary logit scored on invest/no-invest | HOLDS | none | no |
| 5 | First-stage CCP uses Laplace smoothing | HOLDS | none | no |
| 6 | True theta_q=0.70, theta_g=0.40, theta_c=1.00 | HOLDS | none | no |
| 7 | Model policy RMSE = **0.001** (Results prose) | DATA DRIFT | LOW | no |
| 8 | Parameter recovery errors: 0.00387, 0.00502, -0.00115 | HOLDS | none | no |
| 9 | 591 MPE iterations, 72 second-stage iterations | HOLDS | none | no |
| 10 | 60,000 firm-period observations | HOLDS | none | no |
| 11 | Forward simulation RMSE 0.045, 1,000 paths, 70-period horizon | HOLDS | none | no |
| 12 | Symmetric competitors, soft MPE for truth solve | HOLDS | none | no |

## Findings

### Finding 1: W_theta equation omits entropy correction

- **Claim source (verbatim):** "Holding CCPs fixed gives a policy transition $\hat P$ and expected flow payoff $\bar\pi_\theta(\omega;\hat p)$. The logit action shock is integrated out. The value under the first-stage policy is $W_\theta = \bar\pi_\theta(\hat p) + \beta \hat P W_\theta.$" -- `README.md:26-29`

- **Code evidence (verbatim):**
  ```python
  entropy = -(ccp * np.log(ccp) + (1.0 - ccp) * np.log(1.0 - ccp)) + EULER_GAMMA

  for s_idx, state in enumerate(states):
      own, rival = state
      own_policy = ccp[s_idx]
      rival_policy = ccp[index[(rival, own)]]
      expected_flow[s_idx] = (
          (1.0 - own_policy) * flow_payoff(theta, state, 0)
          + own_policy * flow_payoff(theta, state, 1)
          + entropy[s_idx]
      )
  ```
  `run.py:151-160`

- **Data evidence (if applicable):** None -- W values are intermediate computations, not directly reported in CSV.

- **Category:** DILUTED

- **Severity:** MED

- **Result-changing:** no -- code computes W correctly; published estimates and RMSEs come from correct code. A replicator following the equation exactly (omitting `entropy`) would get wrong W values and thus wrong choice-specific values and wrong estimates.

- **Detail:** The equation as written defines `pi_bar` as the expected flow payoff, with a prose note that "logit action shock is integrated out." The code adds `entropy[s_idx] = H_bernoulli(p) + EULER_GAMMA` to the expected flow before solving the Bellman system. This entropy correction is the explicit form of "integrating out" the logit shock. The equation does not display this term or define `pi_bar` to include it. A reader who implements `pi_bar = E_p[flow(a)]` (no entropy) will solve the wrong Bellman system. The code is correct; the equation is incomplete.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "entropy" in inspect.getsource(policy_transition_and_flow) and "EULER_GAMMA" in inspect.getsource(policy_transition_and_flow)
  # PASSES on current code (proves entropy is added); would FAIL if code matched the bare equation
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert r"\text{H}(p)" in open("README.md").read() or "EULER_GAMMA" in open("README.md").read() or "entropy" in open("README.md").read()
  # PASSES on honest fix (equation shows entropy term); FAILS on current README
  ```

### Finding 2: Results prose shows model RMSE as 0.001 while diagnostics table shows 0.00135

- **Claim source (verbatim):** "The model-implied RMSE against truth is **0.001**." -- `README.md:85`

- **Code evidence (verbatim):**
  ```python
  policy_rmse = float(np.sqrt(np.mean((np.asarray(estimate["p_model"]) - true_policy) ** 2)))
  ...
  f"The model-implied RMSE against truth is **{policy_rmse:.3f}**. The first-stage "
  f"empirical RMSE is **{first_stage_rmse:.3f}**."
  ```
  `run.py:297-298, 437-439`

- **Data evidence (verbatim):** `"Model policy RMSE,0.0013543355175390252"` -- `tables/estimator-diagnostics.csv:8`

- **Category:** DATA DRIFT

- **Severity:** LOW

- **Result-changing:** no -- both values come from the same variable in the same run; the discrepancy is 3-decimal formatting truncation (0.001354 -> 0.001 at 3dp, a 26% understatement of the RMSE). First-stage RMSE does not have this problem (0.017087 -> 0.017, a 0.5% difference). The structural conclusions are unaffected.

- **Detail:** `f"{0.001354:.3f}"` produces `"0.001"` because the fourth decimal digit is 3 (rounds down). The diagnostics table uses full-precision floats. The prose bolded value "0.001" understates the actual value by 26% relative to the CSV. A reader comparing the prose RMSE (0.001) against the table (0.00135) sees apparent inconsistency. The first-stage RMSE does not have this asymmetry because 0.01708 rounds to 0.017 with <1% distortion.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(float(open("README.md").read().split("model-implied RMSE against truth is **")[1].split("**")[0]) - 0.0013543) > 0.0002
  # PASSES on current README (prose says 0.001, which differs from 0.00135 by >0.0002)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(float(open("README.md").read().split("model-implied RMSE against truth is **")[1].split("**")[0]) - 0.0013543) < 0.0001
  # PASSES on honest fix using f"{policy_rmse:.4f}" or f"{policy_rmse:.5f}"
  ```

## Cross-cutting patterns

- The entropy correction is mentioned in prose ("logit action shock is integrated out") but never made explicit in the equation or in a definition of `pi_bar`. This pattern -- prose acknowledgment without equation disclosure -- is a recurring risk in tutorials that integrate out private shocks in the Bellman but display the equation in its deterministic skeleton form.
- Both findings are documentation-vs-code gaps, not code errors. The code is correct; the README understates what the code does in two places. This is consistent with a workflow where equations are written before or separately from code, and the entropy/formatting details are not back-propagated to the prose.
- No parametric leaks, no wrong transition matrices, no mismatched parameter ordering found.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Below the 50% halt threshold. Surface to user as advisory; forward work may continue with fixes.

1. **Finding 1 (DILUTED MED):** Turn the violated-invariant assertion into a pytest test confirming `entropy` appears in `policy_transition_and_flow` source. Confirm it passes on current code. Then add a second test confirming the README Equations section contains an explicit entropy or Euler-gamma term after any fix. The second test should currently fail.

2. **Finding 2 (DATA DRIFT LOW):** Add a pytest that parses the bold RMSE value from README.md prose and asserts it is within 5% of the CSV value. Currently fails because 0.001 differs from 0.00135 by 26%.

3. Fix path for Finding 1: update `run.py`'s `add_equations` call to show the entropy term explicitly. Either expand the definition of `pi_bar` to include `E[eps_a] = H_\text{Bernoulli}(p) + \gamma`, or add a displayed equation for `expected_flow` before the Bellman equation.

4. Fix path for Finding 2: change `f"{policy_rmse:.3f}"` to `f"{policy_rmse:.4f}"` (or higher precision) in the Results prose for the model RMSE. The first-stage RMSE format is fine at 3dp.

5. After fixes, regenerate README.md via `python run.py`. Re-run this skill on the new README to confirm both findings now read HOLDS and score drops to 0-10%.
