# bullshit-detector -- deep-optimal-auctions -- recheck -- 2026-05-20

**Bullshit score: 0%** -- single Stage-3 finding resolved; table column label now matches the computation.

## Header
- Claim sources: `game-theory/deep-optimal-auctions/README.md`
- Code / artifact root: `game-theory/deep-optimal-auctions/run.py`
- Data artifacts: `game-theory/deep-optimal-auctions/tables/revenue-regret-audit.csv`
- Seed audit: `bullshit-detector_deep-optimal-auctions_2026-05-20.md`
- Run by: claude-sonnet-4-6 (bullshit-detector skill, Stage-3 recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Original finding (short) | Category | Resolution | Recheck result |
|---|--------------------------|----------|------------|----------------|
| 1 | Table column "Mean regret" stored max-of-means, not mean | MISLABELED | Column renamed to "Largest mean bidder regret" | HOLDS |

## Findings

### Finding 1: Audit-table column label -- RESOLVED

- **Original claim (verbatim):** Column header "Mean regret" in `run.py:296` stored `np.max(mean_regrets)`.
- **Fixed code (verbatim):** `run.py:296`: `"Largest mean bidder regret": f"{float(np.max(mean_regrets)):.4f}"`. Old key `"Mean regret"` absent. Figure y-axis `run.py:587` already said "Largest mean bidder regret" -- now consistent with the table.
- **Data evidence:** `tables/revenue-regret-audit.csv` column header: `Largest mean bidder regret`. Value 0.0012 unchanged (computation unchanged; only label fixed).
- **Category:** HOLDS
- **Violated-invariant test:** `test_finding1_violated_invariant` FAILS (old label gone).
- **Honest-fix test:** `test_finding1_honest_fix` PASSES.

## Cross-cutting patterns

None. Single MISLABELED finding. All numeric results, analytic benchmarks, augmented Lagrangian loss, multiplier update, and softmax feasibility are faithfully implemented and unchanged.

## TDD execution sequence (for the next agent)

Stage-3 fixes are complete. No further action required. Bullshit score is 0%.
