# bullshit-detector -- normal-form-games -- recheck -- 2026-05-20

**Bullshit score: 0%** -- single Stage-3 finding resolved; the combined deviation quantity max{d_1, d_2} is now defined in the Equations section.

## Header
- Claim sources: `game-theory/normal-form-games/README.md`
- Code / artifact root: `game-theory/normal-form-games/run.py`
- Data artifacts: `game-theory/normal-form-games/tables/equilibrium-summary.csv`
- Seed audit: `bullshit-detector_normal-form-games_2026-05-20.md`
- Run by: claude-sonnet-4-6 (bullshit-detector skill, Stage-3 recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Original finding (short) | Category | Resolution | Recheck result |
|---|--------------------------|----------|------------|----------------|
| 1 | Equations never defined combined quantity max{d1,d2} used in heat map and pseudocode | DILUTED | Definition added to Equations section | HOLDS |

## Findings

### Finding 1: Combined deviation quantity definition -- RESOLVED

- **Original claim (verbatim):** Equations section defined d1 and d2 individually but never introduced max{d1(i,j), d2(i,j)} as a named object, yet the heat map and pseudocode both used it.
- **Fixed prose (verbatim):** "The combined deviation gain at $(i,j)$ is the larger of the two, $d(i,j)=\max\lbrace d_1(i,j), d_2(i,j) \rbrace.$" -- `README.md:29-32` (recheck). Pseudocode still uses "max{d1(i,j), d2(i,j)}" at `run.py:234` -- now a back-reference to the defined object.
- **Code evidence:** `README.md:29` contains "combined deviation gain"; `README.md:32` contains `\max\lbrace d_1(i,j), d_2(i,j) \rbrace`.
- **Category:** HOLDS
- **Violated-invariant test:** `test_violated_invariant_combined_quantity_undefined` FAILS (definition now present).
- **Honest-fix test:** `test_honest_fix_combined_quantity_defined` PASSES.
- **Pseudocode test:** `test_pseudocode_still_uses_combined_quantity` PASSES.

## Cross-cutting patterns

None. All numeric results, equilibrium formulas, mixed-NE solver, pure-Nash enumeration, and CSV values remain faithful and unchanged across all four games.

## TDD execution sequence (for the next agent)

Stage-3 fixes are complete. No further action required. Bullshit score is 0%.
