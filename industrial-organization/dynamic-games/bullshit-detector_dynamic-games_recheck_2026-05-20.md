# bullshit-detector -- dynamic-games -- recheck -- 2026-05-20

**Bullshit score: 5%** -- all three original findings resolved; fallback is disclosed in pseudocode and verified at convergence by a runtime assertion; all five top-rung states added to the results table; alpha=0.35 stated in the Model Setup table.

## Header
- Claim sources: `industrial-organization/dynamic-games/README.md`
- Code / artifact root: `industrial-organization/dynamic-games/run.py`
- Data artifacts: `tables/policy-by-state.csv`
- Seed audit: `bullshit-detector_dynamic-games_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Output is pure-strategy MPE; fallback disclosed and verified at convergence | HOLDS | none | no |
| 2 | Firm 1 waits at all 5 top-rung states; all present in results table | HOLDS | none | no |
| 3 | alpha=0.35 stated in Model Setup table | HOLDS | none | no |
| 4 | Flow profit formula, transition probs, G_i, beta, kappa | HOLDS | none | no |
| 5 | All 8 table values match CSV | HOLDS | none | no |
| 6 | Convergence criterion max Bellman residual < 1e-8 | HOLDS | none | no |
| 7 | Deviation gains zero at reported actions | HOLDS | none | no |

## Findings

### Finding 1 (RESOLVED): undisclosed no-NE fallback

**Original finding:** `select_equilibrium` carried a silent `if not equilibria` fallback that was never disclosed in the README, making the "pure-strategy MPE" claim unverifiable.

**Recheck evidence (verbatim):**
- `README.md:65`: "If no pure NE exists (fallback), take the joint-payoff-maximising profile."
- `README.md:72`: "At the converged values every state game has a pure NE and the fallback no longer fires, which the solver checks before returning."
- `run.py:99-110`:
  ```python
  converged_fallback: list[bool] = []
  for q1 in range(q_max + 1):
      for q2 in range(q_max + 1):
          pay1, pay2 = payoff_matrices(V, q1, q2, beta, invest_cost)
          select_equilibrium(pay1, pay2, fallback_log=converged_fallback)
  assert not converged_fallback, (
      "no pure-strategy Nash equilibrium at the converged values"
  )
  ```

The fallback is now explicitly named in the pseudocode. A post-convergence re-solve over all 25 states uses `fallback_log` to confirm the fallback never fires at the fixed point. The `assert not converged_fallback` halts the program if any state lacks a pure NE at convergence. "Pure-strategy MPE" claim is now verifiably enforced at runtime.

- **Category:** HOLDS
- **Verdict:** RESOLVED

---

### Finding 2 (RESOLVED): "waits at top rung" claim broader than evidence

**Original finding:** README claimed "Firm 1 waits at the top rung" but only (4,4) appeared in the CSV, leaving four top-rung states unverified.

**Recheck evidence (verbatim):**
- `run.py:307`: `for state in [(0, 0), (1, 2), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]:`
- CSV rows:
  ```
  "(4,0)",Wait,Invest,99.76,42.63,57.13,0.00e+00
  "(4,1)",Wait,Invest,93.34,52.83,40.51,0.00e+00
  "(4,2)",Wait,Invest,86.99,62.92,24.07,0.00e+00
  "(4,3)",Wait,Invest,81.61,71.95,9.66,0.00e+00
  "(4,4)",Wait,Wait,78.52,78.52,0.00,0.00e+00
  ```

All five (4,q2) states are in the results table. All show Firm 1 "Wait". All have `Max deviation gain = 0.00e+00`. The "waits at top rung" claim now has complete evidence coverage.

- **Category:** HOLDS
- **Verdict:** RESOLVED

---

### Finding 3 (RESOLVED): damping alpha=0.35 undisclosed

**Original finding:** Pseudocode named `alpha` without stating its value; Model Setup table omitted it.

**Recheck evidence (verbatim):**
- `README.md:49`: `| Iteration damping weight | $\alpha=0.35$ | Step size in the value update |`
- `run.py:239`: `"  Update V_i^{n+1} = alpha T_i V^n + (1-alpha) V_i^n.  (alpha=0.35: damping weight)\n"`
- `run.py:94`: `V = 0.35 * V_new + 0.65 * V`

Model Setup table now has an explicit row. Pseudocode now states `alpha=0.35` inline. Code matches.

- **Category:** HOLDS
- **Verdict:** RESOLVED

---

## Cross-cutting patterns

None. All three original findings resolved. No new findings identified.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5%.** No action required.
1. `test_fixed_mpe_claim_unconditional` PASSES -- fallback absent from source or fallback disclosed in README (both now true: fallback disclosed, and runtime assertion prevents any silent fire).
2. `test_fixed_all_top_rung_states_in_table` PASSES -- all 5 top-rung states in README table.
3. `test_fixed_firm1_waits_at_every_top_rung_state` PASSES -- all top-rung states have policy=Wait for Firm 1.
4. `test_fixed_damping_alpha_value_disclosed` PASSES -- README states `alpha=0.35`.
5. Violated-invariant tests (test_violated_silent_fallback_present_and_undisclosed, test_violated_top_rung_coverage_incomplete, test_violated_damping_alpha_value_missing) FAIL -- correct; they are designed to fail after the fix.
6. No further work needed on this tutorial.
