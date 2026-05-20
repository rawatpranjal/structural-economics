# bullshit-detector — algorithmic-collusion-q-learning — recheck — 2026-05-20

**Bullshit score: 0%** — all 21 claims hold; the prose fix from the original audit ("between those two prices" → "spanning from the Bertrand to the monopoly benchmark with both endpoints included") is correctly applied; all numeric results match committed CSVs.

## Header
- Claim sources: `agent-based-models/algorithmic-collusion-q-learning/README.md`
- Code / artifact root: `agent-based-models/algorithmic-collusion-q-learning/run.py`
- Data artifacts: `agent-based-models/algorithmic-collusion-q-learning/tables/benchmark-summary.csv`, `agent-based-models/algorithmic-collusion-q-learning/tables/run-summary.csv`
- Seed audit: `agent-based-models/algorithmic-collusion-q-learning/bullshit-detector_algorithmic-collusion-q-learning_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Grid: 13 evenly spaced prices spanning Bertrand to monopoly, both endpoints included | HOLDS | - | - |
| 2 | Grid size = 15 (13 core + 2 padding) | HOLDS | - | - |
| 3 | Bertrand price = 1.473 | HOLDS | - | - |
| 4 | Monopoly price = 1.925 | HOLDS | - | - |
| 5 | Logit demand formula matches code | HOLDS | - | - |
| 6 | Bertrand FOC: 1 - (p-c)(1-s)/mu = 0 | HOLDS | - | - |
| 7 | Monopoly FOC adds cross-product term (p_j-c)s_j/mu | HOLDS | - | - |
| 8 | Q-update: (1-alpha)*Q + alpha*[pi + delta*max Q'] | HOLDS | - | - |
| 9 | CI = (p_learned - p_B) / (p_M - p_B) | HOLDS | - | - |
| 10 | epsilon_t = exp(-beta*t), beta = 4e-06 | HOLDS | - | - |
| 11 | Learned average price = 1.708 | HOLDS | - | - |
| 12 | Collusion index = 0.52 | HOLDS | - | - |
| 13 | Min post-shock average price = 1.492 | HOLDS | - | - |
| 14 | Recovery horizon = 2 periods | HOLDS | - | - |
| 15 | Optimistic Q-init: discounted average one-period profits | HOLDS | - | - |
| 16 | State = previous-period price-index pair | HOLDS | - | - |
| 17 | Bertrand price (CSV) = 1.47293 | HOLDS | - | - |
| 18 | Monopoly price (CSV) = 1.92498 | HOLDS | - | - |
| 19 | Competitive profit (CSV) = 0.222927 | HOLDS | - | - |
| 20 | Monopoly profit (CSV) = 0.33749 | HOLDS | - | - |
| 21 | Seed 202: learned price 1.70837, CI 0.520833, min post-shock 1.49176, recovery 2 | HOLDS | - | - |

## Findings

### Finding 1 (original): Grid prose "between those two prices" — RESOLVED

- **Original claim:** "Then form 13 evenly spaced prices between those two prices" — original README.md:67. The word "between" mischaracterized `np.linspace(p_B, p_M, 13)`, which includes both endpoints.
- **Current README (verbatim):** "Then form 13 evenly spaced prices spanning from the Bertrand to the monopoly benchmark with both endpoints included, and add one padding point below and above." — `README.md:67`
- **Code (verbatim):**
  ```python
  core_grid = np.linspace(p_bertrand.min(), p_monopoly.max(), params.k - 2)
  step = core_grid[1] - core_grid[0]
  grid = np.linspace(core_grid[0] - step, core_grid[-1] + step, params.k)
  ```
  `run.py:130-132`
- **Verification:** `params.k - 2 = 13`. `np.linspace(p_B, p_M, 13)` includes both endpoints. Prose now says "with both endpoints included". Code and prose are consistent. Strictly-interior count = 11 (not 13). HOLDS.
- **Category:** HOLDS (was DILUTED LOW in original audit)

## Cross-cutting patterns

None. No systematic gap between prose and code. The single prose fix resolved the only non-HOLDS finding in the original audit.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 0%.** Ship. No action required.
1. Tests `test_violated_invariant_only_11_prices_strictly_between` and `test_violated_invariant_benchmarks_are_on_the_grid` now PASS — confirming the grid geometry is correct.
2. Test `test_honest_fix_readme_prose_says_endpoints_included` now PASSES — confirming the prose fix is applied.
3. No further code or prose changes warranted.
