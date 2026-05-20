# bullshit-detector — cobweb-arifovic-ga-learning — recheck — 2026-05-20

**Bullshit score: 0%** — all 19 claims hold; the election notation ambiguity (original Finding 1) is fully resolved: sigma(i) now appears in the notation table, step (6) uses pi_{sigma(i),t}, and a clarifying paragraph at README.md:90 explains that the threshold is the tournament-selected parent's profit, not firm i's own profit.

## Header
- Claim sources: `agent-based-models/cobweb-arifovic-ga-learning/README.md`
- Code / artifact root: `agent-based-models/cobweb-arifovic-ga-learning/run.py`
- Data artifacts: `agent-based-models/cobweb-arifovic-ga-learning/tables/parameter-grid.csv`, `agent-based-models/cobweb-arifovic-ga-learning/tables/iv-estimates.csv`
- Seed audit: `agent-based-models/cobweb-arifovic-ga-learning/bullshit-detector_cobweb-arifovic-ga-learning_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Election step (6): keep child iff pi(decode(b_i'), p_t) >= pi_{sigma(i),t} | HOLDS | - | - |
| 2 | sigma(i) defined in notation table as tournament-selected parent index | HOLDS | - | - |
| 3 | Election threshold pi_{sigma(i),t} clarified as tournament-selected parent's profit | HOLDS | - | - |
| 4 | 3-way tournament selection | HOLDS | - | - |
| 5 | p_c = 0.6, p_m = 0.02 | HOLDS | - | - |
| 6 | L=8, 256 levels in [0,2] | HOLDS | - | - |
| 7 | N = n = 30, T = 500 | HOLDS | - | - |
| 8 | Stable: beta=0.50, p*=1.67 | HOLDS | - | - |
| 9 | Unstable: beta=1.50, p*=3.00 | HOLDS | - | - |
| 10 | GA mean price (last 100): stable=1.66655, unstable=3.01176 | HOLDS | - | - |
| 11 | OLS intercept=20.9452, slope=6.5707 | HOLDS | - | - |
| 12 | 2SLS intercept=67.059, slope=34.2436 | HOLDS | - | - |
| 13 | Fitness = realized profit at cleared price (no demand-parameter leak) | HOLDS | - | - |
| 14 | Naive cobweb: beta = n/(b*y), REE p* = (ay+nx)/(by+n) | HOLDS | - | - |
| 15 | 2SLS uses lagged price p_{t-1} as instrument | HOLDS | - | - |

## Findings

### Finding 1 (original): Election notation pi_{i,t} ambiguous vs code — RESOLVED

- **Original finding:** README step (6) wrote `pi_{i,t}` as the election threshold. The symbol table defined `pi_{i,t}` as "firm i's realized profit." The code uses `parent_profits = profits[parent_indices]` — the tournament winner's profit at slot i, not the original firm i's own profit. These are different quantities when tournament selection permutes parent indices.

- **Resolution — three changes applied:**

  1. **Symbol table (README.md:23):** `sigma(i)` is now a defined symbol: `| sigma(i) | Index of the tournament-selected parent placed at slot i of the next generation. |`

  2. **Step (6) in the abstract GA loop (README.md:82):** Now reads `keep child b_i' iff pi(decode(b_i'), p_t) >= pi_{sigma(i),t}; else keep parent.`

  3. **Clarifying paragraph (README.md:90):** "The election threshold pi_{sigma(i),t} in step (6) is the realized profit of the *tournament-selected parent* placed at slot i, not of the original firm i. Tournament selection at step (4) writes a parent index sigma(i) into each slot i of the new population, and the child at slot i is built from that parent. The election filter then compares the child against its own parent, pi_{sigma(i),t}, which is the standard Arifovic (1994) election operator. Because sigma(i) is a tournament winner, pi_{sigma(i),t} weakly exceeds the profit of a randomly drawn firm, so the filter is conservative: it accepts offspring only when they beat a parent that already survived selection."

- **Code (verbatim):**
  ```python
  parent_indices = selection(profits, rng)
  parents = population[parent_indices]
  parent_profits = profits[parent_indices]
  ...
  if child_profit[0] >= parent_profits[i]:
      new_pop[i] = ca
  if child_profit[1] >= parent_profits[i + 1]:
      new_pop[i + 1] = cb
  ```
  `run.py:152-176`

- **Verification:** `parent_profits[i] = profits[parent_indices[i]]` = profit of the tournament winner at slot i = `pi_{sigma(i),t}`. Prose, notation table, and code now agree exactly.

- **Category:** HOLDS (was DILUTED LOW in original audit)

## Cross-cutting patterns

None. The single notation gap from the original audit is resolved at three levels: symbol definition, equation notation, and clarifying prose paragraph. No systematic gap between prose and code.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 0%.** Ship. No further action required.
1. Test `test_election_compares_against_tournament_winner_profit` PASSES — confirms `parent_profits = profits[parent_indices]` and `child_profit[0] >= parent_profits[i]` are both present.
2. Test `test_finding1_violated_invariant_election_notation_unclarified` now FAILS — confirms "tournament-selected parent" is present in README.
3. Test `test_finding1_honest_fix_election_notation_clarified` now PASSES — confirms "selected parent" is present in README.
4. No further code or prose changes warranted.
