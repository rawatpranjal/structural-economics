# bullshit-detector - consumption-savings - 2026-05-20

**Bullshit score: 20%** - Two low-severity definitional gaps (MPC label is dc/da not dc/dw; pseudocode omits the no-borrowing lower bound). No FALSE, no UNIMPLEMENTED, no result-changing errors. Core VFI, interpolation, masking, simulation all HOLD. All runtime numbers need re-run to verify but code is deterministic.

## Header

- Claim sources: `dynamic-programming/consumption-savings/README.md` (prose, Equations, Results)
- Code / artifact root: `dynamic-programming/consumption-savings/run.py`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "average MPC is 0.52 near zero assets" | MISLABELED | LOW | no (~3% diff from R=1.03 factor) |
| 2 | Pseudocode feasibility shows only upper bound | DILUTED | LOW | no (grid lower bound is implicit) |
| 3 | Runtime numbers: 260 iters, 9.91e-07, 0.52, 0.04, 0.20, 1.85, 20.5%, 2.55e-02 | DATA DRIFT | LOW | needs re-run to verify |
| 4 | VFI algorithm, interpolation, infeasibility masking, Rouwenhorst, initialization | HOLDS | - | - |
| 5 | Simulation: 3000 agents, 400 periods, ergodic init | HOLDS | - | - |

## Findings

### Finding 1: MPC label is dc*/da, not dc*/d(cash-on-hand)

- **Claim source (verbatim):** "For the median income state, average MPC is **0.52** near zero assets and **0.04** near the top." - `README.md:114`

- **Code evidence (verbatim):**
  ```python
  mpc_mid = np.gradient(policy_c[:, median_z_idx], a_grid)
  low_asset_mpc = float(np.mean(mpc_mid[5:20]))
  high_asset_mpc = float(np.mean(mpc_mid[-40:]))
  ```
  `run.py:221-223`

- **Data evidence (if applicable):** No CSV. Numbers embedded in README from runtime. Cannot verify without re-run.

- **Category:** MISLABELED

- **Severity:** LOW

- **Result-changing:** no. The standard MPC out of cash-on-hand is dc*/dw where w = Ra + z. Since policy_c = Ra + z - policy_a, dc*/da = R - d(policy_a)/da. The true MPC wrt cash-on-hand is dc*/dw = dc*/da * (da/dw) = (dc*/da) / R = (dc*/da) / 1.03. At the reported values, true MPC near zero = 0.52/1.03 ≈ 0.50; near top = 0.04/1.03 ≈ 0.039. The discrepancy is 3%, driven entirely by R - 1 = 0.03. Not result-changing economically but the label is wrong.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not np.allclose(np.gradient(sol["policy_c"][:, 2], a_grid)[5:20], np.gradient(sol["policy_c"][:, 2], a_grid)[5:20] / 1.03, rtol=1e-6)
  # PASSES on current buggy code (dc/da != dc/da / R); FAILS on honest fix that reports dc/dw
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert np.allclose(reported_mpc_near_zero, np.mean(np.gradient(sol["policy_c"][:, 2], a_grid)[5:20]) / 1.03, rtol=0.01)
  # PASSES on honest fix (MPC = dc/dw = (dc/da)/R); FAILS on current code (uses dc/da)
  ```

### Finding 2: Pseudocode feasibility omits no-borrowing lower bound

- **Claim source (verbatim):** "feasible(g_l) := { g_l <= R a_i + z_j }            # respects no-borrowing" - `README.md:98` (pseudocode block)

- **Code evidence (verbatim):**
  ```python
  consumption = cash_on_hand[:, [iz]] - a_choice_grid[None, :]
  feasible = consumption > 1e-10
  ```
  `run.py:66-67`

- **Data evidence (if applicable):** None applicable.

- **Category:** DILUTED

- **Severity:** LOW

- **Result-changing:** no. The lower bound a' >= 0 is enforced implicitly because `a_choice_grid = exponential_grid(a_min=0, a_max=20, ...)` starts at 0 (`run.py:168-170`). The feasibility set in the pseudocode only shows the cash-on-hand upper bound but does not state the a' >= 0 lower bound that the algorithm also imposes via the grid construction. The comment "# respects no-borrowing" is misleading: the lower bound IS respected, but through grid construction, not through the feasibility mask shown.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "a_choice_grid[None, :] >= 0" not in inspect.getsource(solve_income_fluctuation)
  # PASSES on current code (no explicit lower bound check); FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "a_choice_grid[0]" in inspect.getsource(main) and a_choice_grid[0] == 0.0
  # PASSES on honest fix (either explicit check or pseudocode adds: g_l >= 0 AND g_l <= Ra+z); FAILS if grid start is non-zero
  ```

### Finding 3: Runtime numbers cannot be verified without re-run

- **Claim source (verbatim):** "The main grid converges in **260 iterations** to sup-norm residual **9.91e-07**." - `README.md:106`. "Median wealth is **0.20**, and the 90th percentile is **1.85**. About **20.5%** of agents sit near $\underline{a}$." - `README.md:126`. "maximum gap from the main grid is **2.55e-02**" - `README.md:116`.

- **Code evidence (verbatim):**
  ```python
  f"The main grid converges in **{solution['iterations']} iterations** to "
  f"sup-norm residual **{solution['error']:.2e}**."
  ```
  `run.py:356-357`

  ```python
  f"Median wealth is **{median_assets:.2f}**, and the 90th percentile is "
  f"**{p90_assets:.2f}**. About **{constraint_share:.1%}** of agents sit near "
  ```
  `run.py:453-455`

- **Data evidence (if applicable):** No tables CSV. Numbers only in README.md. Code is deterministic (fixed seeds: `seed=42` at `run.py:226`, `seed=1234` at `run.py:234`, rng seed 123 at `run.py:230`) so re-run should reproduce. **Needs re-run to verify.**

- **Category:** DATA DRIFT

- **Severity:** LOW

- **Result-changing:** needs re-run to verify. Code is fully deterministic; if the committed README was generated from the committed run.py with identical library versions, numbers should match. The risk is library version drift in `lib/discretize.rouwenhorst` or `lib/grids.exponential_grid` changing outputs silently.

- **Violated invariant (one-line pytest assertion):**
  ```python
  # Cannot write without re-running. Placeholder only.
  assert False, "needs re-run to verify: run python run.py and compare README numbers"
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert solution["iterations"] == 260 and abs(solution["error"] - 9.91e-7) < 5e-9
  # PASSES if code reproduces committed numbers; FAILS on version drift
  ```

## Cross-cutting patterns

- All numeric results (convergence stats, MPC values, simulation quantiles) are embedded at report-generation time from live computation. No separate CSV tables exist for cross-checking. Any code or library change silently propagates to README without a diff signal. Adding a `tables/` output with key scalars would make future audits deterministic without re-runs.
- The pseudocode in Solution Method omits one constraint (lower bound a' >= 0) while correctly showing the other (upper bound a' <= Ra+z). The same pattern could affect other pseudocode blocks if this tutorial is used as a template.
- The MPC mislabeling (dc/da vs dc/dw) is a definitional gap that compounds with the 3% R-factor. For this tutorial r=0.03 makes the error negligible, but any companion tutorial with higher r would show a larger discrepancy.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20%.** Below 50%. No halt required. Findings are low-severity; proceed after addressing.

1. **Finding 1 - MPC label.** Write test: run `solve_income_fluctuation` on small grid, compute `np.gradient(policy_c[:, mid], a_grid)` and compare to `np.gradient(policy_c[:, mid], a_grid) / gross_return`. Confirm they differ by factor R. Then update `run.py:399` to report `(dc/da)/R` or relabel the quantity as `dc/da` in prose.

2. **Finding 2 - Pseudocode lower bound.** No code fix needed. Fix the pseudocode string at `run.py:349` to read `feasible(g_l) := { 0 <= g_l <= R a_i + z_j }` and regenerate README.

3. **Finding 3 - Runtime numbers.** Add a `tables/` output in `run.py` with a CSV of key scalars (iterations, error, median_assets, p90_assets, constraint_share, low_asset_mpc, high_asset_mpc, max_refined_gap_mid). Commit the CSV alongside README. Future audits can cross-check without re-run.

4. After fixes, re-run `python run.py` inside the folder and confirm README matches tables CSV. Re-run this skill; score should drop to 0-10%.
