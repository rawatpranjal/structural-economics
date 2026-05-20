# bullshit-detector — rbc-irreversible-investment — 2026-05-20

**Bullshit score: 20%** — all steady-state and structural claims hold; the only material finding is a pseudocode description that mislabels which model receives the off-grid boundary refinement, and several simulation-output numbers that cannot be verified without a re-run (DATA DRIFT risk).

## Header
- Claim sources: `global-dsge/rbc-irreversible-investment/README.md`
- Code / artifact root: `global-dsge/rbc-irreversible-investment/run.py`
- Data artifacts: `global-dsge/rbc-irreversible-investment/tables/stationary-moments.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Pseudocode: boundary refinement added only to A_irr | MISLABELED | MED | no (std model solution actually more accurate, not less) |
| 2 | VFI convergence in 49 iters (both models) | DATA DRIFT | LOW | no (iteration count is cosmetic) |
| 3 | Binding covers 9.5% of grid states | DATA DRIFT | LOW | no (descriptive statistic, not a table value) |
| 4 | Stress-path binding 13.3% | DATA DRIFT | LOW | no (stress path result, not in stationary table) |
| 5 | Stationary binding 0.42%; table values (mean K, std(Y), etc.) | DATA DRIFT | LOW | no (README/CSV consistent; freshness unverifiable without re-run) |
| 6 | Bellman eq, output, Tauchen, steady-state values, grid bounds | HOLDS | — | — |

## Findings

### Finding 1: Pseudocode says boundary refinement is only added to A_irr; code applies it to both models

- **Claim source (verbatim):** "set A_irr(K_m) to choices in A_std with K' >= (1-delta)K_m\nadd the exact boundary K'=(1-delta)K_m to A_irr when it is between grid nodes" — `README.md:74-75`

- **Code evidence (verbatim):**
  ```python
  ev_boundary = np.zeros(n_k)
  for jz in range(n_z):
      ev_boundary += trans_z[iz, jz] * np.interp(lower_bound, k_grid, v[jz])
  boundary_c = resources[iz] - lower_bound
  boundary_val = utility(boundary_c, sigma) + beta * ev_boundary
  boundary_val = np.where(boundary_allowed, boundary_val, neg_large)
  use_boundary = boundary_val >= best_val - 1e-10
  best_val = np.where(use_boundary, boundary_val, best_val)
  best_k = np.where(use_boundary, lower_bound, best_k)
  if constrained:
      is_binding = use_boundary
  ```
  `run.py:121-131`

- **Data evidence (if applicable):** None — the boundary refinement affects V_std and V_irr values in figures, not the stationary-moments CSV.

- **Category:** MISLABELED

- **Severity:** MED

- **Result-changing:** no — the boundary refinement for the standard model is economically correct (it can only win when the std model's true optimum is at or above I=0, since lower K' from the grid beats boundary_val when negative investment is optimal). The pseudocode description is inaccurate: it says the refinement belongs only to A_irr, but the code computes it for both. The std model solution is more precise (not corrupted). Value-loss figure (fig4: V_std - V_irr) may be very slightly understated because V_std gains from the refinement in states where std model voluntarily chooses I=0, but the effect is bounded by the fraction of states where the unconstrained optimum happens to land exactly at (1-delta)K — negligible at 72 grid points.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "if constrained" not in inspect.getsource(solve_rbc).split("ev_boundary")[1].split("is_binding")[0]
  # PASSES on current code (no guard around boundary computation), FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # After fix: boundary block wrapped with `if constrained:` guard before lines 121-131
  assert "if constrained:\n            ev_boundary" in inspect.getsource(solve_rbc)
  # PASSES on honest fix, FAILS on current code
  ```

### Finding 2: VFI convergence iteration count (49 for both models)

- **Claim source (verbatim):** "The irreversible model converged in **49** VFI iterations. The standard comparison converged in **49**." — `README.md:83`

- **Code evidence (verbatim):**
  ```python
  f"The irreversible model converged in **{sol_irr['iterations']}** VFI iterations. "
  f"The standard comparison converged in **{sol_std['iterations']}**."
  ```
  `run.py:418-419`

- **Data evidence (if applicable):** No artifact records the iteration count. The README number is embedded from a previous run. needs re-run to verify.

- **Category:** DATA DRIFT

- **Severity:** LOW

- **Result-changing:** no — iteration count is a convergence diagnostic, not a results table value.

- **Violated invariant (one-line pytest assertion):**
  ```python
  # No executable invariant possible without re-run; recorded as DATA DRIFT.
  assert sol_irr["iterations"] == 49  # FAILS if re-run converges at different count
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert sol_irr["iterations"] == solve_rbc(constrained=True, **params, verbose=False)["iterations"]
  # PASSES after re-run regenerates README with fresh iteration count
  ```

### Finding 3: Binding covers 9.5% of grid states

- **Claim source (verbatim):** "the boundary covers 9.5% of grid states" — `README.md:112`

- **Code evidence (verbatim):**
  ```python
  binding_share_states = float(sol_irr["binding"].mean())
  ```
  `run.py:303`

- **Data evidence (if applicable):** Not in `tables/stationary-moments.csv`. Embedded in README from previous run. needs re-run to verify.

- **Category:** DATA DRIFT

- **Severity:** LOW

- **Result-changing:** no — descriptive statistic about the grid, not a published comparison number.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(sol_irr["binding"].mean() - 0.095) < 0.001  # FAILS if re-run gives different share
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "binding_share_states" in inspect.getsource(main) and sol_irr["binding"].mean() > 0
  # PASSES after re-run regenerates README; always true structurally
  ```

### Finding 4: Stress-path binding 13.3%

- **Claim source (verbatim):** "It binds for 13.3% of the stress path" — `README.md:112`

- **Code evidence (verbatim):**
  ```python
  binding_share_stress = float(stress_irr["binding"].mean())
  ```
  `run.py:304`

- **Data evidence (if applicable):** Not in CSV. Embedded from previous run. needs re-run to verify.

- **Category:** DATA DRIFT

- **Severity:** LOW

- **Result-changing:** no — stress-path result, not in stationary-moments table.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(stress_irr["binding"].mean() - 0.133) < 0.01  # FAILS if re-run gives different share
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert 0.0 < stress_irr["binding"].mean() < 1.0  # PASSES after any valid re-run
  ```

### Finding 5: Stationary-moments table values (mean K, std(Y), std(C)/std(Y), mean I/Y, I=0 frequency)

- **Claim source (verbatim):** README.md table rows at `README.md:107-110` and CSV `tables/stationary-moments.csv:2-3`.

- **Code evidence (verbatim):**
  ```python
  stationary_idx = draw_markov_path(sol_irr["trans_z"], 6000, seed=42)
  stat_irr = simulate(sol_irr, stationary_idx, k0=k_ss)
  stat_std = simulate(sol_std, stationary_idx, k0=k_ss)
  burn = 1000
  ```
  `run.py:299-302`

- **Data evidence (if applicable):** CSV records: Irreversible: mean K=39.499, std(Y)=17.110%, std(C)/std(Y)=0.729, mean I/Y=0.251, I=0 frequency=0.42%. Standard: same first four columns, I=0 frequency=0.00%. README matches CSV exactly. needs re-run to verify freshness; if code has been edited since CSV was generated, both could be stale.

- **Category:** DATA DRIFT

- **Severity:** LOW

- **Result-changing:** no — README and CSV are mutually consistent; no internal contradiction detected.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert pd.read_csv("tables/stationary-moments.csv").iloc[0]["std(Y) %"] == pytest.approx(17.110, abs=0.001)
  # FAILS if re-run produces different value
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert float(pd.read_csv("tables/stationary-moments.csv").iloc[0]["mean K"]) == pytest.approx(39.499, abs=0.01)
  # PASSES after fresh re-run regenerates CSV
  ```

## Cross-cutting patterns

- All confirmed-HOLDS findings (Bellman equation, Tauchen discretization, steady-state values, grid bounds, Howard improvement, off-grid boundary via interpolation, investment floor enforcement, simulation logic) are verified analytically or against explicit code lines without needing a re-run.
- The four DATA DRIFT findings all share the same root cause: simulation-output numbers embedded in README from a previous run are not re-verifiable against any committed artifact other than the CSV (which is itself from the same run). This is a structural property of the tutorial contract (README generated by `run.py`), not a sign of fabrication.
- The one MISLABELED finding (pseudocode vs code structure for boundary refinement) is a documentation-only gap. The code behavior is correct; the prose description is inaccurate. The same pattern of "boundary computed unconditionally, guard only on binding indicator" appears once in the VFI loop — not a systemic issue across the codebase.
- No FALSE or UNIMPLEMENTED findings. Every core claim about the model (Bellman, feasibility set, irreversibility constraint, Howard acceleration, stress path construction) is implemented faithfully.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20%.** Below the 50% halt threshold. Code work can continue; address Finding 1 (pseudocode correction) in the next editorial pass.

1. Finding 1 (MISLABELED): Turn violated invariant into a pytest test that confirms `ev_boundary` computation is not guarded by `if constrained`. Confirm test passes on current code.
2. Convert honest-fix pass condition into a second test that fails on current code (expects the `if constrained:` guard before `ev_boundary`).
3. Fix is one of: (a) wrap lines 121-131 in `run.py` with `if constrained:` — but verify this doesn't degrade std model accuracy; or (b) correct the pseudocode in `run.py`'s `add_solution_method` call to say the boundary is added to both models' candidate sets. Option (b) is the lower-risk fix since the code behavior is economically correct.
4. Findings 2-5 (DATA DRIFT): Re-run `python run.py` inside the tutorial folder to regenerate README and CSV. Confirm the frozen numbers (49 iters, 9.5%, 13.3%, 0.42%, table rows) match the fresh output. No code change needed unless numbers diverge.
5. After fixes, re-run this skill. Expected new score: 5-10%.
