# bullshit-detector — cobweb-arifovic-ga-learning — 2026-05-20

**Bullshit score: 15%** — One DILUTED/LOW finding: election operator step (6) notation uses subscript i to mean "tournament-selected parent" but prose says "firm i's profit," creating a directionally-strict but ambiguously-labeled comparison. All numeric claims, table values, REE formulas, and GA mechanics otherwise HOLD with verbatim code evidence.

## Header
- Claim sources: `agent-based-models/cobweb-arifovic-ga-learning/README.md`
- Code / artifact root: `agent-based-models/cobweb-arifovic-ga-learning/run.py`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Election step (6): "keep child iff π(decode(b_i'), p_t) ≥ π_{i,t}" where π_{i,t} = firm i's realized profit | DILUTED | LOW | no |
| 2 | β = 0.50 (stable), β = 1.50 (unstable) | HOLDS | — | — |
| 3 | REE p* = 1.67 (stable), p* = 3.00 (unstable) | HOLDS | — | — |
| 4 | GA mean price last 100: stable = 1.66655, unstable = 3.01176 | HOLDS | — | — |
| 5 | OLS intercept = 20.9452, slope = 6.5707; 2SLS intercept = 67.059, slope = 34.2436 | HOLDS | — | — |
| 6 | L = 8 giving 256 encoded quantity levels in [0, 2] | HOLDS | — | — |
| 7 | Population size N = n = 30 | HOLDS | — | — |
| 8 | crossover p_c = 0.6, mutation p_m = 0.02, T = 500 | HOLDS | — | — |
| 9 | Fitness = realized profit at cleared price p_t | HOLDS | — | — |
| 10 | 3-way tournament selection | HOLDS | — | — |
| 11 | 2SLS uses lagged price p_{t-1} as instrument | HOLDS | — | — |
| 12 | OLS biased by simultaneity (Cov(p_t, e_t) > 0) | HOLDS | — | — |

## Findings

### Finding 1: Election step (6) — π_{i,t} notation ambiguous vs code

- **Claim source (verbatim):** "keep child $b_i'$ iff $\pi(\mathrm{decode}(b_i'), p_t) \geq \pi_{i,t}$; else keep parent." — `README.md:81`
- **Code evidence (verbatim):**
  ```python
  parent_indices = selection(profits, rng)
  parents = population[parent_indices]
  parent_profits = profits[parent_indices]
  new_pop = parents.copy()
  ...
  if ga.use_election:
      children = np.stack([ca, cb])
      child_q = decode(children, ga)
      child_profit = firm_profit(child_q, realized_price, params)
      if child_profit[0] >= parent_profits[i]:
          new_pop[i] = ca
      if child_profit[1] >= parent_profits[i + 1]:
          new_pop[i + 1] = cb
  ```
  `run.py:152-176`
- **Data evidence:** Not applicable (architectural claim).
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no — the tournament-selected parent has profit >= any non-selected firm at the same realized price. Using tournament-winner profits makes the election filter stricter than if it compared against each original firm i's own profit. The bias is directionally conservative (harder to accept offspring), not permissive. GA convergence result is unaffected.
- **Detail:** The notation π_{i,t} in step (6) reads as "firm i's realized profit at period t." In code, `parent_profits[i]` is the profit of the firm that WON the tournament for position i — that is, `profits[parent_indices[i]]` — not firm i's own original profit. The subscript i in the prose denotes a slot in the new population, not a fixed firm identity. This is consistent with Arifovic (1994)'s own description where election compares a candidate child against its tournament-selected parent. The code is faithful to the Arifovic paper; the README notation is ambiguous relative to the actual mechanics.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert profits[parent_indices[0]] == parent_profits[0]  # PASSES on current code (tournament winner profits used); standard reading of π_{i,t} would imply profits[0] == parent_profits[0]
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "profits[parent_indices" in inspect.getsource(reproduce) or "parent_profits = profits[parent_indices" in inspect.getsource(reproduce)  # clarify in prose that π_{i,t} = tournament-winner profit
  ```

## Cross-cutting patterns

- All numeric claims in Results (REE prices, GA mean prices, distances, OLS/2SLS coefficients) match committed CSV artifacts exactly (truncated in README text, full precision in CSVs). No drift between prose and data artifacts.
- The GA mechanics (decode, selection, crossover, mutation, election) all match the pseudocode algorithm in Solution Method verbatim and are correctly implemented.
- The IV estimation block uses the standard two-stage OLS formula: stage-2 OLS of Q on X_hat = (X_hat'X_hat)^{-1} X_hat'Q with HC0 SEs using the stage-2 residuals Q - X@beta. This is the correct 2SLS two-stage implementation, not a mislabeling.
- The only gap is the election notation ambiguity (Finding 1). No parametric-knowledge leaks of the form caught in the canonical worked example: election in `reproduce()` uses only `realized_price` (the cleared market price) and the decoded child quantities — no true parameters (a, b, x, y) are read directly in the election comparison.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 15%.** Below the 50% halt threshold. No stop needed; the single finding is LOW severity and non-result-changing.
1. The one non-HOLDS finding is notation only. No failing pytest test needs to be written for a bug fix — the code is correct per Arifovic (1994). The fix is prose: clarify in README step (6) that π_{i,t} denotes the realized profit of the tournament-selected parent at position i, not of the original firm i.
2. Optional prose fix: change "iff $\pi(\mathrm{decode}(b_i'), p_t) \geq \pi_{i,t}$" to "iff $\pi(\mathrm{decode}(b_i'), p_t) \geq \pi(\mathrm{decode}(b_{\sigma(i),t}), p_t)$" where $\sigma(i)$ is the tournament winner at slot i, or add a prose clarification sentence after the display equation.
3. No re-run required. All data artifacts are consistent with committed CSVs.
4. Re-run this skill after the prose fix to confirm the score drops to 0-10%.
