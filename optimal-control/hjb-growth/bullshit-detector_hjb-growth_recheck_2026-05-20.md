# bullshit-detector -- hjb-growth -- recheck -- 2026-05-20

**Bullshit score: 10%** -- all HJB equations, upwind rule, boundary conditions, FOC, analytical steady-state values, and grid parameters hold exactly; the original Finding 1 (k_max=14.80 display) is confirmed non-finding (correct rounding of 14.7997); the original Finding 2 (iterations=16, residual=5.34e-07) persists as LOW DATA DRIFT because these runtime values have no independent CSV cross-check, but they are self-consistent across README and CSV.

## Header
- Claim sources: `optimal-control/hjb-growth/README.md` (full file, 275 lines)
- Code / artifact root: `optimal-control/hjb-growth/run.py` (734 lines)
- Data artifacts: `optimal-control/hjb-growth/tables/steady-state.csv`
- Seed audit: `optimal-control/hjb-growth/bullshit-detector_hjb-growth_2026-05-20.md`
- Run by: bullshit-detector skill (Claude Sonnet 4.6), 2026-05-20 (recheck)
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | HJB formulation, FOC c* = (V')^(-1/sigma) | HOLDS | - | - |
| 2 | Drift s(k) = f(k) - delta*k - c*(k) | HOLDS | - | - |
| 3 | Upwind rule (forward if s>0, backward if s<0, dV0 if s=0) | HOLDS | - | - |
| 4 | Boundary: left uses forward, right uses backward | HOLDS | - | - |
| 5 | Implicit linear system [(1/Delta+rho)I - G^n] V^{n+1} = u(c^n) + V^n/Delta | HOLDS | - | - |
| 6 | Delta = 1000 | HOLDS | - | - |
| 7 | Initial guess V^0 = u(f(k))/rho | HOLDS | - | - |
| 8 | Tolerance = 1e-6 | HOLDS | - | - |
| 9 | Parameters: rho=0.05, sigma=2.0, alpha=0.36, delta=0.05, A=1.0 | HOLDS | - | - |
| 10 | Grid: 500 points, k in [0.1, 14.80] (correctly rounded 14.7997) | HOLDS | - | - |
| 11 | Analytical k_ss=7.3998, c_ss=1.6855, y_ss=2.0555, i_ss=0.3700, i/y=0.1800, MPK=0.1000 | HOLDS | - | - |
| 12 | Steady-state condition f'(k_ss) = rho + delta | HOLDS | - | - |
| 13 | k_ss = (alpha*A/(rho+delta))^(1/(1-alpha)) | HOLDS | - | - |
| 14 | 6000-point grid dk ~ 2.5e-3 | HOLDS | - | - |
| 15 | HJB converged in 16 iterations, residual 5.34e-07 | DATA DRIFT | LOW | no |

## Findings

### Finding 1 (RESOLVED from original audit): k_max=14.80 display

The original audit Finding 1 flagged `k_max=14.80` as a display discrepancy because `k_max = 2.0 * k_ss = 14.7997`. This was already conceded as a "correct two-decimal rounding" in the original audit. Independent recomputation: `k_ss = (0.36*1.0/0.10)^(1/0.64) = 7.39969...`, `k_max = 14.7994...`, `{:.2f}` = 14.80. The README displays the correctly rounded value. **Not a finding. RESOLVED.**

The test `test_readme_k_max_display_is_correct_rounding` verifies `abs(k_max - 14.80) < 1e-3` and `f"{k_max:.2f}]$"` appears in README. Both conditions hold.

### Finding 2 (UNCHANGED from original audit): Runtime convergence diagnostics

- **Claim source (verbatim):** "The HJB converged in **16 iterations** with final sup-norm change $5.34e-07$." -- `README.md:228`
- **Code evidence (verbatim):**
  ```python
  f"The HJB converged in **{info_ct['iterations']} iterations** with "
  f"final sup-norm change ${info_ct['error']:.2e}$. "
  ```
  `run.py:563-564`
- **Data evidence:** `tables/steady-state.csv:8-9` -- `HJB iterations,--,16` and `HJB residual,--,5.34e-07`. README and CSV are self-consistent. Neither can be independently cross-checked without executing the solver.
- **Category:** DATA DRIFT
- **Severity:** LOW -- the values are plausible (convergence in O(10) iterations is expected for this class of HJB solver); residual 5.34e-07 < tol 1e-6 is consistent. The only gap is the absence of a pre-run verification artifact.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert open("optimal-control/hjb-growth/tables/steady-state.csv").read().count("16") == 1  # PASSES (one occurrence in CSV); FAILS if re-run changes iteration count
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert float(open("optimal-control/hjb-growth/tables/steady-state.csv").readlines()[8].split(",")[-1].strip()) < 1e-6  # PASSES (residual < tol=1e-6)
  ```

## Cross-cutting patterns

- Original Finding 1 (k_max=14.80) was a non-finding in the original audit's own analysis and remains a non-finding here. It never warranted DATA DRIFT classification.
- Original Finding 2 (runtime convergence) remains at DATA DRIFT / LOW. No code change resolved it because it requires a CI re-run to be truly verified. The test `test_convergence_diagnostics_consistent_readme_and_csv` passes by checking cross-consistency between README and CSV, but this only confirms the two artifacts were generated together -- it cannot independently verify the iteration count.
- All analytical steady-state values (k_ss, c_ss, y_ss, i_ss, i/y, MPK) hold to 4 decimal places against fresh computation.
- All algorithmic claims (upwind rule, FOC inversion, boundary conditions, implicit system, Delta, tolerance, initial guess) are grounded verbatim in `run.py:87-206`.
- No FALSE, DILUTED, MISLABELED, or UNIMPLEMENTED findings.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Below the 50% halt threshold. All load-bearing math is clean.
1. **Finding 2 (DATA DRIFT, LOW):** To fully resolve, add a CI step that executes `python run.py` and asserts `info_ct['iterations'] == 16` and `info_ct['error'] < 1e-6`. Alternatively, archive `info_ct` to a secondary JSON artifact that can be cross-checked without re-running the full solver.
2. No prose fixes or algorithm fixes needed. The README faithfully describes the code.
3. Re-run `python scripts/validate_catalog.py` before any commit.
