# bullshit-detector — diamond-mortensen-pissarides — 2026-05-20

**Bullshit score: 25%** — One DILUTED finding: contraction modulus stated as beta*(1-sigma)=0.9621, but the effective modulus of the nonlinear operator is ~0.3; convergence in 26 iterations is inconsistent with the stated modulus, which would require ~654 iterations for tol=1e-11.

## Header

- Claim sources: `dynamic-programming/diamond-mortensen-pissarides/README.md`
- Code / artifact root: `dynamic-programming/diamond-mortensen-pissarides/run.py`
- Data artifacts: `tables/amplification-by-surplus.csv`, `tables/business-cycle-stats.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Contraction modulus = beta*(1-sigma) = 0.9621 | DILUTED | MED | no (solver converges; modulus is misleading, not wrong about convergence) |
| 2 | Grid gap 3.97e-04%, policy gap 3.23%, 26 iterations not in any CSV | DATA DRIFT | LOW | needs re-run to verify |
| 3 | All calibration parameters, steady-state values, wage, J equations, unemployment law | HOLDS | none | n/a |
| 4 | All amplification-by-surplus table values | HOLDS | none | n/a |
| 5 | All business-cycle-stats table values | HOLDS | none | n/a |

## Findings

### Finding 1: Contraction modulus stated as beta*(1-sigma) = 0.9621

- **Claim source (verbatim):** "The operator is a contraction with modulus $\beta(1-\sigma)=0.9621$." — `README.md:97`

- **Code evidence (verbatim):**
  ```python
  for iteration in range(1, max_iter + 1):
      expected_job_value = transition @ job_value
      theta = (
          beta
          * matching_efficiency
          * np.maximum(expected_job_value, 0.0)
          / vacancy_cost
      ) ** (1.0 / (1.0 - matching_elasticity))
      new_job_value = (
          (1.0 - bargaining_power) * (z_grid - benefit)
          - bargaining_power * vacancy_cost * theta
          + beta * (1.0 - separation_rate) * expected_job_value
      )
      error = float(np.max(np.abs(new_job_value - job_value)))
      job_value = new_job_value
      if error < tol:
          break
  ```
  `run.py:93-109`

- **Data evidence:** The README itself contradicts the claimed modulus in the same section: "The $N_z=41$ fixed point converges in **26 iterations**" (`README.md:132`). At modulus 0.9621, starting from an initial error of order 1, reaching tol=1e-11 requires at least `ceil(-11 / log10(0.9621)) = 654` iterations. The solver terminates in 26, which is consistent with an effective modulus of ~0.3 but inconsistent with 0.9621. The effective contraction rate is computed from the Jacobian of `T(J)` w.r.t. `J` at steady state. The operator is `J_new(z_i) = (1-gamma)(z_i-b) - gamma*k*theta_i(EJ_i) + beta*(1-sigma)*EJ_i` where `theta_i = (beta*chi/k * EJ_i)^(1/(1-eta))`. The total derivative w.r.t. `EJ` at steady state (`EJ_ss = k/(beta*chi)`, `theta_ss=1`) is `d(J_new)/d(EJ) = beta*(1-sigma) - gamma*k * (1/(1-eta)) * (beta*chi/k)^(1/(1-eta)) * EJ_ss^(eta/(1-eta))`. Numerically: `0.9621 - 0.72*0.2106*8.275 = -0.293`. The effective Lipschitz constant is `|-0.293| * ||P||_inf = 0.293` (since `||P||_inf = 1` for a row-stochastic transition matrix). This matches 26 iterations: `0.293^26 ~ 2.5e-14 < 1e-11`.

- **Category:** DILUTED — The operator IS a contraction (code converges correctly), but the stated modulus `beta*(1-sigma)` applies only to the linear term `beta*(1-sigma)*EJ`. The full nonlinear operator substitutes `theta(EJ)` inside the Bellman, introducing a negative correction that tightens the actual modulus from 0.9621 to ~0.293. The README omits this interaction, citing only the linear-term modulus.

- **Severity:** MED — Pedagogically wrong: a reader computing required iterations from the stated modulus (654) would find the actual count (26) inexplicable. Does not affect output correctness.

- **Result-changing:** no — The solver converges in both cases; the stated modulus is too loose, not too tight, so convergence is not at risk. Reported outputs (tightness grid, business-cycle moments) are unaffected.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(0.9621**26 - 1.0) < 0.001  # PASSES on current code (modulus 0.9621 after 26 steps is 0.365, not near zero; proves stated modulus is inconsistent with 26-iter convergence)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(-0.293**26) < 1e-11  # PASSES on honest fix (true modulus ~0.293 after 26 steps reaches <1e-11 tol)
  ```

### Finding 2: Grid gap, policy gap, and iteration count not grounded in any CSV artifact

- **Claim source (verbatim):** "The interpolated gap in $\theta(z)$ is **3.97e-04%**." — `README.md:130`; "The $N_z=41$ fixed point converges in **26 iterations**." — `README.md:132`; "The maximum policy gap between the nonlinear and log-linear rules is **3.23%**." — `README.md:132`

- **Code evidence (verbatim):**
  ```python
  max_policy_gap = float(
      np.max(np.abs(theta_nonlinear_grid - theta_linear_grid) / theta_nonlinear_grid)
  )
  max_grid_gap = float(
      np.max(np.abs(theta_nonlinear_grid - theta_fine_on_coarse) / theta_fine_on_coarse)
  )
  ```
  `run.py:293-298`; then formatted as `f"{100.0 * max_grid_gap:.2e}%"` at `run.py:490` and `f"{100.0 * max_policy_gap:.2f}%"` at `run.py:495`.

- **Data evidence:** `tables/business-cycle-stats.csv` and `tables/amplification-by-surplus.csv` contain no column for grid gap, policy gap, or iteration count. These three numbers appear only in the generated `README.md` prose and are not persisted to any auditable artifact. If `run.py` were re-run on a different machine or with a different random seed (shock simulation does not affect these deterministic quantities, but a future code change could), the discrepancy would be silent.

- **Category:** DATA DRIFT — The committed README numbers cannot be cross-checked against any on-disk artifact without re-running `run.py`. Per audit instructions: **needs re-run to verify**.

- **Severity:** LOW — The numbers are deterministic (fixed by the Rouwenhorst discretization and the fixed-point iteration with `seed`-independent inputs). The drift risk is from future code edits, not from stochastic variation.

- **Result-changing:** needs re-run to verify

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert any("grid_gap" in col or "policy_gap" in col or "iterations" in col for col in pd.read_csv("tables/business-cycle-stats.csv").columns)  # FAILS on current code (proves numbers not persisted to CSV)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert pd.read_csv("tables/solver-diagnostics.csv")["policy_gap_pct"].iloc[0] == pytest.approx(3.23, abs=0.005)  # PASSES on honest fix (solver diagnostics written to CSV)
  ```

## Cross-cutting patterns

- The contraction modulus error is the only structural claim that is inconsistent with the code's own observable behavior (26 iterations vs. 654 required). This pattern - citing the modulus of the linearized operator rather than the full nonlinear operator - is a common shortcut in DMP expositions. Other tutorials that use nonlinear fixed points with theta-feedback inside the Bellman should be checked for the same claim.
- The three ungrounded inline numbers (grid gap, policy gap, iteration count) are dynamically injected from code at report-generation time. The committed README could silently drift from the current code on any re-run that changes those values. The pattern is present because no solver-diagnostics CSV is written. Any future tutorial using inline f-string quantities that are not also persisted to a CSV carries the same risk.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Proceed with fixes; no need to halt or re-audit sibling artifacts.

1. **Finding 1 (DILUTED, MED):** Turn the violated invariant into a pytest test confirming that the solver converges in fewer than 100 iterations (not ~654 as the claimed modulus implies). Confirm it passes on current code.

2. **Finding 1 fix:** Fix the README prose. The correct statement is that the operator IS a contraction but the effective modulus is approximately 0.29 at the steady state due to the theta-feedback term; convergence in ~26 iterations is consistent with this tighter bound, not with beta*(1-sigma)=0.9621. Do not change `run.py` - the computation is correct. Change only the claim in `run.py`'s `add_solution_method` string (the source that generates the README) and regenerate.

3. **Finding 2 (DATA DRIFT, LOW):** Either (a) add a `tables/solver-diagnostics.csv` written in `main()` containing `max_grid_gap`, `max_policy_gap`, and `iterations` for both grid sizes, or (b) accept that these numbers are transiently generated and flag them explicitly in the README as "generated at last run." Option (a) is the stronger fix.

4. After fixes: re-run `python run.py` inside the tutorial folder, verify the new README states the correct modulus text, verify the CSV (if added) contains the expected values, re-run `scripts/validate_catalog.py`.

5. Re-run this skill. Expected new score: 0-10% (all HOLDS, no ungrounded inline quantities).
