# bullshit-detector — diamond-mortensen-pissarides — recheck — 2026-05-20

**Bullshit score: 0%** — All structural, mathematical, and numerical claims HOLD. The two original findings are resolved: (1) the contraction-modulus prose now distinguishes the linear-term bound 0.9621 from the effective modulus ~0.293, with the effective modulus computed live in run.py; (2) tables/solver-diagnostics.csv is committed with policy_gap_pct, grid_gap_pct, and iterations columns matching README prose exactly.

## Header
- Claim sources: `dynamic-programming/diamond-mortensen-pissarides/README.md`
- Code / artifact root: `dynamic-programming/diamond-mortensen-pissarides/run.py`
- Data artifacts: `dynamic-programming/diamond-mortensen-pissarides/tables/solver-diagnostics.csv`, `tables/amplification-by-surplus.csv`, `tables/business-cycle-stats.csv`
- Seed audit: `bullshit-detector_diamond-mortensen-pissarides_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Linear-term modulus = beta*(1-sigma) = 0.9621 | HOLDS | none | no |
| 2 | Effective modulus ~0.293; explains few-dozen iterations | HOLDS | none | no |
| 3 | N_z=41 fixed point converges in 26 iterations | HOLDS | none | no |
| 4 | Policy gap (nonlinear vs log-linear) = 3.23% | HOLDS | none | no |
| 5 | Grid gap (coarse vs fine theta) = 3.97e-04% | HOLDS | none | no |
| 6 | Log-linear elasticity C = 1.554 at baseline | HOLDS | none | no |
| 7 | A=1.1098, B=0.5262 at baseline | HOLDS | none | no |
| 8 | Amplification table: b=0.40->C=1.55, b=0.95->C=18.65 | HOLDS | none | no |
| 9 | Business-cycle stats: tightness std/std_z = 1.72 | HOLDS | none | no |
| 10 | u_ss=0.0649, w_ss=0.9837 | HOLDS | none | no |
| 11 | Simulation length: 4500 months post-burn-in | HOLDS | none | no |
| 12 | solver-diagnostics CSV carries all three diagnostic columns | HOLDS | none | no |

## Findings

### Finding 1 (original): Contraction modulus stated as linear-term only — RESOLVED

- **Original finding (verbatim):** "The operator is a contraction with modulus $\beta(1-\sigma)=0.9621$." Original README stated only the linear-term modulus; the effective modulus of the nonlinear operator (~0.293) was absent, making the 26-iteration convergence inexplicable from the stated bound (which would require ~654 iterations).

- **Resolution:** `run.py:268-284`:
  ```python
  linear_term_modulus = beta * (1.0 - separation_rate)
  expected_job_value_ss = vacancy_cost / (beta * matching_efficiency)
  theta_feedback_term = (
      bargaining_power
      * vacancy_cost
      * (1.0 / (1.0 - matching_elasticity))
      * (beta * matching_efficiency / vacancy_cost)
      ** (1.0 / (1.0 - matching_elasticity))
      * expected_job_value_ss ** (matching_elasticity / (1.0 - matching_elasticity))
  )
  effective_modulus = abs(linear_term_modulus - theta_feedback_term)
  ```
  The effective modulus is now computed analytically in code and injected into the Solution Method prose at `run.py:474-480`:
  ```python
  "The operator is a contraction. Its linear term $\\beta(1-\\sigma)"
  f"E[J']$ alone has modulus $\\beta(1-\\sigma)={linear_term_modulus:.4f}$, "
  "but the substituted free-entry term $\\theta(E[J'])$ inside the "
  "Bellman adds a negative correction. The total derivative of the "
  "update with respect to $E[J']$ at the steady state gives an effective "
  f"modulus of about ${effective_modulus:.3f}$, well below the linear-term "
  "bound. The tight effective modulus is why the fixed point converges in "
  "a few dozen iterations rather than the several hundred the linear-term "
  "modulus would imply."
  ```
  README:97 renders as: "Its linear term ... alone has modulus $\beta(1-\sigma)=0.9621$, but ... gives an effective modulus of about $0.293$, well below the linear-term bound." Both moduli stated and distinguished. The phrase "linear-term" appears. HOLDS.

- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 2 (original): Solver diagnostics not persisted to a committed artifact — RESOLVED

- **Original finding:** DATA DRIFT — policy gap, grid gap, and iteration counts were only in README prose; no CSV committed.

- **Resolution:** `run.py:634-671`. `tables/solver-diagnostics.csv` now exists with three columns:
  ```
  Quantity,policy_gap_pct,grid_gap_pct,iterations
  Coarse-grid policy gap vs. log-linear,3.2287,,
  Coarse-grid interpolation gap vs. fine grid,,0.000397,
  Coarse-grid fixed-point iterations,,,26
  Fine-grid fixed-point iterations,,,31
  ```
  Cross-check against README:
  - `3.2287` → formatted `3.23%` → README:132: "3.23%". Match (round(3.2287,2)=3.23).
  - `0.000397` → formatted `3.97e-04%` → README:130: "3.97e-04%". Match.
  - `26` → README:132: "26 iterations". Match.
  - `31` (fine-grid) → README table `iterations` column row 4 = "31". Match.
  All four diagnostic quantities are now grounded in a committed artifact.

- **Category after fix:** HOLDS
- **Severity:** none
- **Result-changing:** no

## Cross-cutting patterns

- None. Both original findings resolved cleanly. No new findings.
- The effective-modulus computation (`run.py:268-284`) is the key addition. It is algebraically derived at the steady state (theta_ss=1, EJ_ss=k/(beta*chi)) and injected live into the prose. The value 0.293 is not hardcoded; it recomputes on each `python run.py` call from current parameters.
- All three data tables (business-cycle-stats, amplification-by-surplus, solver-diagnostics) now carry grounded numeric values. The README table for business-cycle moments and amplification match their CSVs exactly, row by row.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No further action required.
1. The violated-invariant tests (`test_finding1_violated_invariant_modulus_is_linear_term_only`, `test_finding2_violated_invariant_no_diagnostics_csv`) both correctly fail on the current repo state, confirming both fixes were applied.
2. All honest-fix tests (`test_finding1_stated_modulus_inconsistent_with_iteration_count`, `test_finding1_honest_fix_readme_states_effective_modulus`, `test_finding2_honest_fix_diagnostics_csv_exists`, `test_finding2_readme_policy_gap_matches_csv`) pass.
3. No additional findings identified. No further TDD cycles needed.
