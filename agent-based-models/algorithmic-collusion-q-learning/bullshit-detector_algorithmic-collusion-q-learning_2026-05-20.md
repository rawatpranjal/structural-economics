# bullshit-detector — algorithmic-collusion-q-learning — 2026-05-20

**Bullshit score: 15%** — one DILUTED prose misdescription of the grid construction ("between" vs "from ... to ... inclusive"); all equations, hyperparameters, and numeric results hold exactly against code and CSV artifacts.

## Header

- Claim sources: `agent-based-models/algorithmic-collusion-q-learning/README.md` (prose, Equations, Model Setup, Results, tables)
- Code / artifact root: `agent-based-models/algorithmic-collusion-q-learning/run.py`
- Data artifacts: `agent-based-models/algorithmic-collusion-q-learning/tables/benchmark-summary.csv`, `tables/run-summary.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "13 evenly spaced prices between those two prices" | DILUTED | LOW | no |
| 2 | Logit demand formula matches code | HOLDS | — | — |
| 3 | Bertrand FOC matches code | HOLDS | — | — |
| 4 | Monopoly FOC matches code | HOLDS | — | — |
| 5 | Q-learning update formula matches code | HOLDS | — | — |
| 6 | Collusion index formula matches code | HOLDS | — | — |
| 7 | Action set P = {p_B-Delta} ... {p_M+Delta} matches code | HOLDS | — | — |
| 8 | All hyperparameters (alpha, beta, delta, k, steps) match Params | HOLDS | — | — |
| 9 | Numeric results in prose match CSV (learned price 1.708, CI 0.52, min post-shock 1.492, recovery 2) | HOLDS | — | — |
| 10 | Impulse response pseudocode matches impulse_response() | HOLDS | — | — |
| 11 | Optimistic Q-initialization matches initialize_q() | HOLDS | — | — |

## Findings

### Finding 1: "13 evenly spaced prices between those two prices"

- **Claim source (verbatim):** "Then form 13 evenly spaced prices between those two prices and add one padding point below and above." — `README.md:67` (generated from `run.py:481`)

- **Code evidence (verbatim):**
  ```python
  core_grid = np.linspace(p_bertrand.min(), p_monopoly.max(), params.k - 2)
  step = core_grid[1] - core_grid[0]
  grid = np.linspace(core_grid[0] - step, core_grid[-1] + step, params.k)
  ```
  `run.py:130-132`

- **Data evidence (if applicable):** Not applicable — no numeric CSV row affected. The grid has 15 points total (verified numerically); `grid[1]` == p_Bertrand and `grid[-2]` == p_Monopoly (verified: True).

- **Category:** DILUTED

- **Severity:** LOW

- **Result-changing:** no — the grid construction is correct; only the prose word "between" mischaracterizes `np.linspace(p_B, p_M, 13)`, which includes both endpoints, giving 11 prices strictly between the benchmarks and 2 at the benchmark values, not 13 strictly between.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert sum(1 for g in grid if p_b < g < p_m) == 13  # PASSES on buggy prose reading; FAILS on honest count (actual is 11)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert sum(1 for g in grid if p_b < g < p_m) == 11  # PASSES when prose is corrected to "11 prices strictly between" or "13 prices spanning from Bertrand to monopoly inclusive"
  ```

## Cross-cutting patterns

- No systematic pattern of prose-vs-code drift. The single finding is an isolated word choice ("between" instead of "from ... to ... inclusive") in the Model Setup section. It does not recur in Equations (where the action set notation is correct) or in Results (where all numbers match CSV exactly).
- All economic formulas (logit demand, Bertrand FOC, monopoly FOC, Q-update, collusion index) are transcribed into code without loss or alteration.
- All committed CSV data (`benchmark-summary.csv`, `run-summary.csv`) are consistent with the README prose numbers to the precision reported (6 significant figures in tables, 3 in prose).

## TDD execution sequence (for the next agent)

0. **Bullshit score: 15%.** Below the 50% halt threshold. No immediate action required; surface the single DILUTED finding to the author as a low-priority prose fix.

1. **Finding 1 violated invariant test:**
   ```python
   # tests/test_grid_description.py
   def test_core_grid_includes_endpoints():
       from run import solve_benchmarks, Params
       params = Params()
       bench = solve_benchmarks(params)
       # Verify that grid[1] is p_B and grid[-2] is p_M (endpoints IN grid, not outside)
       import numpy as np
       assert np.isclose(bench.grid[1], bench.bertrand_price, atol=1e-6)
       assert np.isclose(bench.grid[-2], bench.monopoly_price, atol=1e-6)
       # Count strictly interior points
       n_strictly_between = sum(1 for g in bench.grid if bench.bertrand_price < g < bench.monopoly_price)
       assert n_strictly_between == 11  # not 13 as prose claims
   ```
   This test PASSES on current code (proves the description gap is real).

2. **Honest-fix pass condition test:** Change the prose in `run.py:481` from `"13 evenly spaced prices between those two prices"` to `"13 prices spanning from the Bertrand to the monopoly benchmark (endpoints included)"` (or equivalent). After the fix, re-run `python run.py` and confirm README.md updates. No code logic changes needed.

3. After the prose fix, re-run this skill to confirm the finding now reads HOLDS and the bullshit score drops to 0-10%.
