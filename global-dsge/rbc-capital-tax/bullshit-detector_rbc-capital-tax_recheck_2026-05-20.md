# bullshit-detector -- rbc-capital-tax -- recheck -- 2026-05-20

**Bullshit score: 0%** -- prior finding (Gauss-Seidel vs Jacobi) is resolved; every claim now holds against code and committed CSV artifacts.

## Header
- Claim sources: `global-dsge/rbc-capital-tax/README.md`
- Code / artifact root: `global-dsge/rbc-capital-tax/run.py`
- Data artifacts: `global-dsge/rbc-capital-tax/tables/steady-state.csv`
- Seed audit (if any): `bullshit-detector_rbc-capital-tax_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Euler refinement is Jacobi (buffer + atomic swap) | HOLDS | -- | -- |
| 2 | After-tax return (1-tau_k)*alpha*z*K'^(alpha-1) in Euler | HOLDS | -- | -- |
| 3 | K loss 42.7% at tau=30% | HOLDS | -- | -- |
| 4 | Y loss 18.2%, C loss 9.7% at tau=30% | HOLDS | -- | -- |
| 5 | Steady-state formulas K_ss, Y_ss, C_ss, T_ss | HOLDS | -- | -- |
| 6 | Grid 40 K pts, 5 Tauchen z states | HOLDS | -- | -- |
| 7 | Simulation 5000 periods, seed=42, burn=500 | HOLDS | -- | -- |
| 8 | CSV table values all five tax rates | HOLDS | -- | -- |
| 9 | Mean K(sim) above K_ss for all regimes | HOLDS | -- | -- |

## Findings

### Prior finding 1 (Gauss-Seidel vs Jacobi pseudocode, DILUTED LOW): RESOLVED

- **Original state:** `policy_k[iz, :]` was updated inside the `for iz` loop (Gauss-Seidel), so later rows of the sweep already saw the refreshed earlier rows. The pseudocode implied Jacobi (full sweep then atomic update).
- **Current state:** `run.py:130-131` allocates `policy_c_new = np.zeros_like(policy_c)` and `policy_k_new = np.zeros_like(policy_k)` before the `for iz` loop. Writes go to `policy_c_new[iz, ik]` (line 144) and `policy_k_new[iz, :]` (line 145) inside the loop. Atomic swaps at `run.py:148-149`: `policy_c = policy_c_new.copy()`, `policy_k = policy_k_new`, after the full loop. Comment at `run.py:127-129` explicitly labels this Jacobi. Pseudocode and code now agree.
- **Category:** HOLDS

## Cross-cutting patterns

- All prior findings are resolved in a single structural fix: the `_new` buffer pattern now matches the pseudocode's `g_K_new` / `g_c_new` notation and atomic-swap semantics.
- CSV values verified for all five tax rates. Y loss (18.18%) and C loss (9.71%) derivable from CSV rows match the README's 18.2% and 9.7% to one decimal.
- simulate() hardcodes `alpha=0.36, delta=0.025` at lines 172-173 (latent inconsistency if primitives change, noted in original audit). No claim is violated under current parameterization; this is an engineering note, not a finding.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required.
1. The violated-invariant test (`"            policy_k[iz, :] = np.clip" in block`) now FAILS, confirming the in-loop Gauss-Seidel write is gone.
2. The honest-fix test (`"policy_k_new[iz, :]" in block and "        policy_k = policy_k_new" in block`) now PASSES.
3. No further code changes needed.
