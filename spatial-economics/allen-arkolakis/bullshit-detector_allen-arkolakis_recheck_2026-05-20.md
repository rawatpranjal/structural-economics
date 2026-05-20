# bullshit-detector -- allen-arkolakis -- recheck -- 2026-05-20

**Bullshit score: 0%** -- both original findings are fully resolved: path_gap is archived to `tables/convergence-path-dependence.csv`; "center share" prose is relabeled to "largest labor share" in `run.py` and regenerated README. All core math, CSV numbers, and README prose hold.

## Header
- Claim sources: `spatial-economics/allen-arkolakis/README.md` (full file, 185 lines)
- Code / artifact root: `spatial-economics/allen-arkolakis/run.py`
- Data artifacts: `tables/scenario-diagnostics.csv`, `tables/trade-cost-counterfactual.csv`, `tables/parameters.csv`, `tables/convergence-path-dependence.csv`
- Seed audit: `spatial-economics/allen-arkolakis/bullshit-detector_allen-arkolakis_2026-05-20.md`
- Run by: bullshit-detector skill (Claude Sonnet 4.6), 2026-05-20 (recheck)
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | path_gap=0.304 archived to CSV | HOLDS | - | - |
| 2 | "largest labor share" label consistent with labor.max() | HOLDS | - | - |
| 3 | CES price index formula | HOLDS | - | - |
| 4 | Trade shares / gravity formula | HOLDS | - | - |
| 5 | Trade balance residual | HOLDS | - | - |
| 6 | Mobility / utility equalization residual | HOLDS | - | - |
| 7 | Wage normalization (geometric mean = 1) | HOLDS | - | - |
| 8 | gamma1/gamma2 formulas and uniqueness threshold | HOLDS | - | - |
| 9 | All CSV numbers (HHI, welfare %, kappa, utility, largest share) | HOLDS | - | - |
| 10 | Continuation description ("two steps") | HOLDS | - | - |
| 11 | Softmax parameterization and 2N-1 system size | HOLDS | - | - |

## Findings

### Finding 1 (RESOLVED from original audit): path_gap not archived

The original audit Finding 1 flagged `path_gap=0.304` as unarchived -- computed at runtime and string-formatted directly into README with no durable CSV. Fix: `run.py:812-813` now writes `pd.DataFrame({"path_gap": [path_gap]}).to_csv(...)` to `tables/convergence-path-dependence.csv`. Confirmed:

- `tables/convergence-path-dependence.csv` exists with column `path_gap` = 0.3044891... (rounds to 0.304 in README).
- `test_path_gap_archived_to_csv_honest_fix` PASSES (honest-fix test).
- `test_path_gap_csv_absent_violated_invariant` FAILS (violated-invariant test; this is the expected post-fix behavior documented in the test docstring). **RESOLVED.**

### Finding 2 (RESOLVED from original audit): "center share" label vs labor.max()

The original audit Finding 2 flagged the Results prose "dispersion-dominant center share" as inconsistent with `labor.max()` (largest share, not position-indexed center). Fix: `run.py:708` now reads `"The dispersion-dominant largest labor share is {disp_eq.labor.max():.1%}."`. README.md:123 reads: "The dispersion-dominant largest labor share is 15.1%." Both code and README use "largest labor share". Confirmed:

- `test_results_prose_label_matches_labor_max_honest_fix` PASSES (honest-fix test).
- `test_results_prose_says_center_share_violated_invariant` FAILS (violated-invariant test; expected post-fix per docstring). **RESOLVED.**

## Cross-cutting patterns

- Both original findings are resolved. No new findings on fresh audit.
- All three original CSV tables (`parameters.csv`, `scenario-diagnostics.csv`, `trade-cost-counterfactual.csv`) remain internally consistent with README prose. All 49 auditable numeric cells match.
- New CSV `tables/convergence-path-dependence.csv` adds a durable artifact for `path_gap=0.304`. Value 0.3044891 rounds to 0.304 as stated in README.
- Test suite: 2 honest-fix tests PASS, 2 violated-invariant tests FAIL (correct post-fix behavior documented in test docstrings).
- No FALSE, no DILUTED, no UNIMPLEMENTED, no DATA DRIFT in this recheck.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings HOLDS. No further fixes needed.
1. The two violated-invariant tests (`test_path_gap_csv_absent_violated_invariant`, `test_results_prose_says_center_share_violated_invariant`) FAIL on current code. This is correct and expected -- they test pre-fix invariants. They should not be deleted; they document what the pre-fix code looked like.
2. Run `python scripts/validate_catalog.py` from repo root to confirm no math rendering regressions.
