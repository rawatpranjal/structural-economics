# bullshit-detector -- cfr-asymmetric-auction -- recheck -- 2026-05-20

**Bullshit score: 0%** -- both Stage-3 findings resolved; prose describes empirical decay rate correctly and alpha is now in the committed CSV.

## Header
- Claim sources: `game-theory/cfr-asymmetric-auction/README.md`
- Code / artifact root: `game-theory/cfr-asymmetric-auction/run.py`
- Data artifacts: `game-theory/cfr-asymmetric-auction/tables/methods-summary.csv`, `game-theory/cfr-asymmetric-auction/tables/asymmetric-exploitability.csv`
- Seed audit: `bullshit-detector_cfr-asymmetric-auction_2026-05-20.md`
- Run by: claude-sonnet-4-6 (bullshit-detector skill, Stage-3 recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Original finding (short) | Category | Resolution | Recheck result |
|---|--------------------------|----------|------------|----------------|
| 1 | Exploitability tracks O(1/sqrt(T)) -- empirical slope is -0.83, not -0.5 | DILUTED | Prose fixed in Results | HOLDS |
| 2 | alpha = 3/2 claim had no committed artifact | DATA DRIFT | alpha_opt added to methods-summary.csv | HOLDS |

## Findings

### Finding 1: Exploitability decay rate -- RESOLVED

- **Original claim (verbatim):** "Exploitability of the average strategy falls steadily across iterations and tracks the textbook rate of order one over the square root of iterations on a log-log plot." (original Results prose)
- **Fixed prose (verbatim):** "On a log-log plot the committed run decays at an empirical rate of approximately O(T^{-0.8}), faster than the O(1/sqrt(T)) Hannan bound rather than tracking it." -- `run.py:502-503` (recheck)
- **Note on remaining "square root" at line 412:** That mention is in Solution Method and describes the theoretical Hannan bound per information set -- a correct mathematical claim about CFR theory, not about this run's empirical rate. No conflict.
- **Data evidence:** Log-log slope over all rows of `asymmetric-exploitability.csv` = -0.834 (re-measured). New prose "approximately O(T^{-0.8})" is consistent with -0.834.
- **Category:** HOLDS
- **Violated-invariant test:** `test_finding1_violated_invariant` FAILS (old phrase absent from run.py).
- **Honest-fix test:** `test_finding1_honest_fix` PASSES.

### Finding 2: alpha coefficient artifact -- RESOLVED

- **Data evidence:** `methods-summary.csv` row 4: `MMRS shooting coefficient $\alpha$, 1.5000`. Value 1.5000 satisfies `abs(1.5000 - 1.5) < 1e-3`.
- **Category:** HOLDS
- **Violated-invariant test:** `test_finding2_violated_invariant` FAILS (alpha now in CSV).
- **Honest-fix test:** `test_finding2_honest_fix` PASSES.

## Cross-cutting patterns

None. Both findings were presentation-layer issues. The CFR algorithm, MMRS BVP shooting, exploitability computation, and bid-function residuals remain faithfully implemented.

## TDD execution sequence (for the next agent)

Stage-3 fixes are complete. No further action required. Bullshit score is 0%.
