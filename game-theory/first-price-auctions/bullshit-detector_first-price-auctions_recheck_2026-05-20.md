# bullshit-detector -- first-price-auctions -- recheck -- 2026-05-20

**Bullshit score: 0%** -- single Stage-3 finding resolved; prose no longer claims the grid best response "sits on" the analytic bid.

## Header
- Claim sources: `game-theory/first-price-auctions/README.md`
- Code / artifact root: `game-theory/first-price-auctions/run.py`
- Data artifacts: `game-theory/first-price-auctions/tables/auction-summary.csv`, `game-theory/first-price-auctions/tables/best-response-residuals.csv`
- Seed audit: `bullshit-detector_first-price-auctions_2026-05-20.md`
- Run by: claude-sonnet-4-6 (bullshit-detector skill, Stage-3 recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Original finding (short) | Category | Resolution | Recheck result |
|---|--------------------------|----------|------------|----------------|
| 1 | "Grid best response sits on the analytic bid" -- focal gap is 1.333e-04 | DILUTED | Prose changed to "within one grid spacing" | HOLDS |

## Findings

### Finding 1: Focal-case prose accuracy -- RESOLVED

- **Original claim (verbatim):** "The grid best response sits on the analytic bid." -- `run.py:189` (original)
- **Fixed prose (verbatim):** "within one grid spacing of the analytic bid." -- `run.py:189` (recheck)
- **Code evidence:** `gap = abs(grid_best_response(0.8, 3)[0] - equilibrium_bid(0.8, 3)) = 1.333e-04` (re-measured). Old phrase "sits on" absent. New phrase "within one grid spacing" present. The description is now consistent with the non-zero gap the residuals table already disclosed.
- **Category:** HOLDS
- **Violated-invariant test:** `test_finding1_violated_invariant` FAILS (old phrase gone).
- **Honest-fix test:** `test_finding1_honest_fix` PASSES.

## Cross-cutting patterns

None. All formulas (bid rule, shading, win probability, expected payoff), grid parameters, and residuals table remain faithful and unchanged.

## TDD execution sequence (for the next agent)

Stage-3 fixes are complete. No further action required. Bullshit score is 0%.
