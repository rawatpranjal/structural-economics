# bullshit-detector -- rbc-capital-tax -- 2026-05-20

**Bullshit score: 15%** -- one DILUTED/LOW finding: Euler-refinement pseudocode implies Jacobi update of g_K; code executes Gauss-Seidel (policy_k updated inside iz-loop). All numeric claims, steady-state formulas, and table values hold exactly.

## Header
- Claim sources: `global-dsge/rbc-capital-tax/README.md` (prose, Equations, Model Setup, Results, Solution Method pseudocode)
- Code / artifact root: `global-dsge/rbc-capital-tax/run.py`
- Data artifacts: `global-dsge/rbc-capital-tax/tables/steady-state.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Pseudocode implies Jacobi update of g_K; code does Gauss-Seidel | DILUTED | LOW | no -- GS converges same fixed point, faster |
| 2 | K loss 42.7%, Y loss 18.2%, C loss 9.7% at tau=30% | HOLDS | -- | -- |
| 3 | Steady-state formula Kss(tau) | HOLDS | -- | -- |
| 4 | Capital grid 40 pts, TFP grid 5 Tauchen states | HOLDS | -- | -- |
| 5 | Same shock seed for every tax regime | HOLDS | -- | -- |
| 6 | Euler refinement uses after-tax return (1-tau_k)MPK | HOLDS | -- | -- |
| 7 | Resource constraint c+K'=zK^alpha+(1-delta)K | HOLDS | -- | -- |
| 8 | T_ss = tau_k * alpha * Y_ss | HOLDS | -- | -- |
| 9 | Table values match analytical formula and CSV | HOLDS | -- | -- |
| 10 | Simulated mean K above deterministic K_ss | HOLDS | -- | -- |

## Findings

### Finding 1: Euler pseudocode implies Jacobi update of g_K; code uses Gauss-Seidel

- **Claim source (verbatim):**
  > "repeat Euler refinement: for each state (z_i,K_m): K_plus = g_K(z_i,K_m) ... g_K_new(z_i,K_m) = z_i K_m^alpha + (1-delta)K_m - g_c_new(z_i,K_m) until the consumption policy update is below epsilon"
  >
  > -- `README.md:80-87`

  The pseudocode presents a full sweep over all (z_i, K_m), then an implicit update of g_K. This implies a Jacobi scheme: all states computed using old g_K, then g_K refreshed atomically.

- **Code evidence (verbatim):**
  ```python
  for iz in range(n_z):
      resources = resources_all[iz]
      for ik in range(n_k):
          kp = policy_k[iz, ik]
          Ec = 0.0
          for jz in range(n_z):
              z_next = z_grid[jz]
              c_next = np.interp(kp, K_grid, policy_c[jz, :])
              mpk_next = (1.0 - tau_k) * alpha * z_next * kp ** (alpha - 1.0) + 1.0 - delta
              Ec += trans_z[iz, jz] * c_next ** (-sigma) * mpk_next
          c_euler = (beta * Ec) ** (-1.0 / sigma)
          c_euler = np.clip(c_euler, 1e-10, resources[ik] - K_min)
          policy_c_new[iz, ik] = c_euler
      policy_k[iz, :] = np.clip(resources - policy_c_new[iz, :], K_min, K_max)
  ```
  `run.py:127-141`

  `policy_k[iz, :]` is updated inside the `for iz` loop at line 141. When `iz=1` executes, `policy_k[0, :]` already reflects the current euler_iter. When `iz=2` executes, `policy_k[0, :]` and `policy_k[1, :]` are already updated. This is Gauss-Seidel on the iz dimension for policy_k, not the Jacobi scheme the pseudocode implies.

  `policy_c` by contrast is updated atomically: `policy_c = policy_c_new.copy()` at line 144 runs after the full iz-loop. So policy_c is Jacobi; policy_k is Gauss-Seidel.

- **Data evidence:** Not applicable. The numeric outputs in the table are analytically verifiable from the steady-state formula (confirmed correct); the Gauss-Seidel vs Jacobi distinction does not change the fixed point, only the convergence path.

- **Category:** DILUTED

- **Severity:** LOW -- same fixed point; convergence may differ; results table unaffected.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  # policy_k[1,:] uses kp from policy_k[1,:] which is still old -- but policy_k[0,:] is already new
  assert "policy_k[iz, :] = np.clip" in inspect.getsource(solve_rbc_tax)  # line inside iz-loop, not after
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # Honest fix moves policy_k update outside the iz-loop (full Jacobi)
  # Test: after fix, policy_k_new array is fully computed before any row of policy_k is overwritten
  assert all_rows_computed_before_update  # requires refactor to policy_k_new buffer, then swap
  ```

## Cross-cutting patterns

- All steady-state formula claims are analytically exact and self-consistent. The code computes Kss, Yss, Css, Tss with formulas that match the Equations section one-to-one.
- The Tauchen discretization uses sigma_e (innovation SD) in the CDF denominator: correct for the conditional distribution N(rho*z_t, sigma_e^2). Row sums verified to equal 1.0 for all 5 states.
- The z-grid is identical across all tau regimes (depends only on rho, sigma_e, m_z, n_z which are fixed), so seed=42 generates the same z_idx path and thus the same z_sim for every tax rate. The claim "same shock seed for every tax regime" is upheld at the z-path level.
- simulate() hardcodes alpha=0.36, delta=0.025 at lines 167-168 rather than inheriting from the solution dict. This is a latent parameter-inconsistency bug if either primitive were changed, but for the current parameterization it is not a claim violation.
- The pseudocode/code discrepancy (Finding 1) is the only gap. It is on the implementation detail level (update order), not on the economic mechanism or result.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 15%.** Below the 50% halt threshold. Safe to proceed.

1. For Finding 1 (DILUTED, LOW): turn the violated invariant into a test:
   ```python
   import inspect, sys
   sys.path.insert(0, ".")
   from run import solve_rbc_tax
   src = inspect.getsource(solve_rbc_tax)
   # The Gauss-Seidel update appears inside the iz-loop, confirmed by pattern:
   assert src.count("policy_k[iz, :] = np.clip") == 1  # inside iz-loop
   ```
   This test passes on current code (proves GS update exists).

2. Honest-fix pass condition: refactor Euler loop to accumulate into `policy_k_new` and swap after full iz-loop -- mirrors the existing `policy_c_new` / `policy_c` pattern.

3. After fix, re-run `python run.py` and verify table values unchanged (same fixed point, different convergence path).

4. Update pseudocode in README to explicitly label the update order (or add a comment in code noting the Gauss-Seidel character).

5. Re-run this skill to confirm score drops to 0-10%.
