# bullshit-detector — dcegm-retirement-saving — recheck — 2026-05-20

**Bullshit score: 0%** — F1 (DILUTED/LOW "centered at 2.8") resolved; prose now reads "median 2.8 (arithmetic mean 3.06)"; all other claims held in the original audit and still hold.

## Header
- Claim sources: `structural-econometrics/dcegm-retirement-saving/README.md`
- Code / artifact root: `structural-econometrics/dcegm-retirement-saving/run.py`
- Data artifacts: `tables/bruteforce-comparison.csv`, `tables/lifecycle-moments.csv`
- Seed audit: `bullshit-detector_dcegm-retirement-saving_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "median 2.8 (arithmetic mean 3.06)" | HOLDS | - | - |
| 2 | EGM Euler inversion formula | HOLDS | - | - |
| 3 | Endogenous asset formula | HOLDS | - | - |
| 4 | Branch value at endogenous point | HOLDS | - | - |
| 5 | CRRA utility and terminal bequest | HOLDS | - | - |
| 6 | Income formula y_t(work) | HOLDS | - | - |
| 7 | Work disutility psi_t(work) sign | HOLDS | - | - |
| 8 | Upper envelope: choose retire when V_retire >= V_work | HOLDS | - | - |
| 9 | Next-period status indexing (m'(work)=0, m'(retire)=1) | HOLDS | - | - |
| 10 | Absorbing retirement in simulation | HOLDS | - | - |
| 11 | Pseudocode constraint region V formula | HOLDS | - | - |
| 12 | All table numbers match CSV artifacts | HOLDS | - | - |
| 13 | Budget constraint c + a+ = R*a + y | HOLDS | - | - |
| 14 | Retire_gap sign convention | HOLDS | - | - |

## Findings

No non-HOLDS findings.

### F1 resolution: "centered at 2.8" → "median 2.8 (arithmetic mean 3.06)"

- **Original finding:** DILUTED/LOW — "centered at" conventionally reads as arithmetic mean; lognormal median is 2.8 but arithmetic mean is 3.06.
- **Fix applied:** `run.py:683-685` now generates:
  `"Initial assets lognormal with median {p.initial_asset_mean:.1f} (arithmetic mean {p.initial_asset_mean * np.exp(p.initial_asset_sigma ** 2 / 2):.2f})"`
- **README:202 current text:** "Initial assets lognormal with median 2.8 (arithmetic mean 3.06)"
- **Honest-fix test status:** PASSES (`"centered at 2.8" not in README` and `"median" in README.lower()`)
- **Violated-invariant test status:** FAILS (no longer finds "centered at 2.8") — confirms fix applied.
- **Category now:** HOLDS

## Cross-cutting patterns

Single fix applied cleanly. No new issues introduced. All 14 original HOLDS claims re-verified against current `run.py` and `README.md`; none have drifted.

## TDD execution sequence

All tests pass. No further action required for this tutorial.
