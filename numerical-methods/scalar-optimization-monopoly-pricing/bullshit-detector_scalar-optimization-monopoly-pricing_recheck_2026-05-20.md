# bullshit-detector — scalar-optimization-monopoly-pricing — recheck — 2026-05-20

**Bullshit score: 0%** — F1 (DATA DRIFT/MED "1/N in one dimension" vs "1/sqrt(N)" elsewhere) resolved; Equations section now uses $1/\sqrt{N}$ consistently; all other claims held and still hold.

## Header
- Claim sources: `numerical-methods/scalar-optimization-monopoly-pricing/README.md`
- Code / artifact root: `numerical-methods/scalar-optimization-monopoly-pricing/run.py`
- Data artifacts: `tables/method_comparison.csv`, `tables/elasticity_sensitivity.csv`, `tables/newton_sensitivity.csv`
- Seed audit: `bullshit-detector_scalar-optimization-monopoly-pricing_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Random-search rate 1/sqrt(N) — consistent throughout | HOLDS | - | - |
| 2 | pi'(p) formula | HOLDS | - | - |
| 3 | pi''(p) formula | HOLDS | - | - |
| 4 | p* = epsilon/(epsilon-1)*c closed form | HOLDS | - | - |
| 5 | p_inflect = (epsilon+1)/(epsilon-1)*c | HOLDS | - | - |
| 6 | p* < p_inflect (concave region claim) | HOLDS | - | - |
| 7 | phi = (sqrt(5)-1)/2 ≈ 0.618 | HOLDS | - | - |
| 8 | Bracket shrinks by factor phi per step | HOLDS | - | - |
| 9 | Safeguard pseudocode matches code | HOLDS | - | - |
| 10 | Bad start x_1 = 5.4, diverged after 1 iteration | HOLDS | - | - |
| 11 | Safeguarded Newton: 9 iterations | HOLDS | - | - |
| 12 | 3 of 9 starts diverge | HOLDS | - | - |
| 13 | Newton good start: 6 iterations | HOLDS | - | - |
| 14 | Method comparison table numbers match CSV | HOLDS | - | - |
| 15 | Elasticity sensitivity table numbers match CSV | HOLDS | - | - |

## Findings

No non-HOLDS findings.

### F1 resolution: Equations random-search rate claim

- **Original finding:** DATA DRIFT/MED — Equations section said "scales as $1/N$ in one dimension, the same order as the grid" while four other locations in the same README used $1/\sqrt{N}$.
- **Fix applied:** `README.md:75` (generated from `run.py`'s `add_equations` call) now reads:
  "The expected error scales as $1/\sqrt{N}$ in one dimension, slower than the deterministic grid but with a rate that does not degrade as price dimensions are added."
- **README count of `1/\sqrt{N}`:** 5 occurrences (lines 75, 140, 152, 213, 265).
- **"1/N in one dimension, the same order as the grid":** absent from README.
- **Honest-fix test status:** PASSES (`"scales as $1/N$ in one dimension, the same order as the grid" not in README` and `README.count(r"1/\sqrt{N}") >= 5`)
- **Violated-invariant test status:** FAILS (the old string absent) — fix confirmed.

## Cross-cutting patterns

Single prose fix. Code (`run.py:659-661` reference line `ref_sqrt = (p_high - p_low) / np.sqrt(n_arr)`) always computed the $1/\sqrt{N}$ reference; the fix aligns Equations prose with the code and the rest of the document. No numeric output changed.

## TDD execution sequence

All tests pass. No further action required for this tutorial.
