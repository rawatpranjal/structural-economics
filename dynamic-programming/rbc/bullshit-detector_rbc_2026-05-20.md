# bullshit-detector -- rbc -- 2026-05-20

**Bullshit score: 10%** -- One DATA DRIFT finding (fine-grid audit numbers in README cannot be verified without a re-run; all other claims held against code and CSV).

## Header
- Claim sources: `dynamic-programming/rbc/README.md`
- Code / artifact root: `dynamic-programming/rbc/run.py`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Fine-grid audit numbers (iterations 515/525, errors 9.95e-06/9.96e-06, value error 2.1e-04, capital gap 0.0461, hours gap 0.0150) match committed README | DATA DRIFT | LOW | no (audit-only figures, not used in simulation) |
| 2 | Steady-state values k_ss=10.4980, l_ss=0.3330, c_ss=0.8073, i_ss=0.2446 | HOLDS | -- | -- |
| 3 | EV computation EV=V@P.T broadcasts correctly as EV[j,s] = sum_t P[s,t]*V[j,t] | HOLDS | -- | -- |
| 4 | flow_utility tensor shape (n_k,n_z,n_l,n_k) correct | HOLDS | -- | -- |
| 5 | Investment formula i_t = k_{t+1} - (1-delta)*k_t in simulation | HOLDS | -- | -- |
| 6 | HP filter formula (I + lam*D'D)^{-1}*y with lam=1600 | HOLDS | -- | -- |
| 7 | Business-cycle moments table matches tables/business-cycle-stats.csv | HOLDS | -- | -- |
| 8 | Simulation: T_sim=5000, T_burn=500; grid: 50x50 coarse, 200x100 fine | HOLDS | -- | -- |
| 9 | Bellman update pseudocode matches code tensor operations | HOLDS | -- | -- |
| 10 | Steady-state formulas in Equations section match code derivation | HOLDS | -- | -- |

## Findings

### Finding 1: Fine-grid audit numbers cannot be verified without re-run

- **Claim source (verbatim):** "The max relative value error is **2.1e-04**. The max capital-policy gap is **0.0461**. The max hours gap is **0.0150**. The coarse VFI converged in **515 iterations** with sup-norm error **9.95e-06**. The fine-grid VFI converged in **525 iterations** with error **9.96e-06**." -- `README.md:110-112`
- **Code evidence (verbatim):**
  ```python
  bench_V_rel = np.max(np.abs((V - V_bench) / np.abs(V_bench)))
  bench_k_max_abs = np.max(np.abs(k_policy - k_policy_bench))
  bench_l_max_abs = np.max(np.abs(l_policy - l_policy_bench))
  ```
  `run.py:173-175`
  ```python
  info = {"iterations": iteration, "converged": error < tol, "error": float(error)}
  ```
  `run.py:91`
  The five numbers are dynamically generated and interpolated into the README string at `run.py:366-375`. They are not hardcoded in the source.
- **Data evidence:** No stdout log, no tables CSV for audit metrics. The business-cycle CSV (`tables/business-cycle-stats.csv`) contains only simulation moments, not fine-grid diagnostics.
- **Category:** DATA DRIFT -- the committed README contains specific numbers that cannot be cross-checked against any committed artifact other than the README itself. The code will regenerate them on the next run; whether the committed values reflect the current code state is unverifiable without execution.
- **Severity:** LOW -- these figures appear only in the Solution Method section as a convergence audit. They do not feed the simulation, the policy rules, or the business-cycle moments table. The result-changing risk is nil.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert Path("dynamic-programming/rbc/logs/vfi_diagnostics.txt").exists(), "No committed artifact for fine-grid audit numbers"
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert float(open("dynamic-programming/rbc/logs/vfi_diagnostics.txt").read().split("bench_V_rel=")[1].split()[0]) == pytest.approx(2.1e-4, rel=0.05)
  ```

## Cross-cutting patterns

- The README numbers that CAN be grounded (steady-state values, all table rows, grid dimensions, parameter values) are exact matches to code or CSV. The only gap is the five fine-grid diagnostics for which no committed artifact exists outside the README itself. Consider committing a `logs/vfi_diagnostics.txt` or adding a `tables/fine-grid-audit.csv` row so future audits can verify without re-running.
- The code generates the README dynamically from live computed values (`run.py:366-375`). This means any committed README is self-consistent with the code AT THE TIME OF THE LAST RUN, but can drift if parameters change without a re-run. No current drift detected; flag as a structural fragility.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Below the 50% halt threshold. No halt required. The one DATA DRIFT finding is low-severity and not result-changing.
1. To close Finding 1: add a committed artifact -- e.g. `tables/fine-grid-audit.csv` with columns `metric,value` and rows for `bench_V_rel`, `bench_k_max_abs`, `bench_l_max_abs`, `coarse_iterations`, `coarse_error`, `fine_iterations`, `fine_error`. Populate it from `run.py` at generation time using `report.add_table(...)` or a direct `pd.DataFrame.to_csv(...)` call.
2. Write a pytest that loads `tables/fine-grid-audit.csv` and asserts each value is within 10% of the value currently printed in `README.md`. This test fails if the code changes parameters without a re-run.
3. Re-run this skill after step 1 to confirm the finding reads HOLDS and the score drops to 0%.
