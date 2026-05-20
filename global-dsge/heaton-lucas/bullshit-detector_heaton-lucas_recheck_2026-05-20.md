# bullshit-detector -- heaton-lucas -- recheck -- 2026-05-20

**Bullshit score: 0%** -- all six prior findings resolved; every claim now holds against code and committed CSV artifacts.

## Header
- Claim sources: `global-dsge/heaton-lucas/README.md`
- Code / artifact root: `global-dsge/heaton-lucas/run.py`
- Data artifacts: `global-dsge/heaton-lucas/tables/euler-errors.csv`, `global-dsge/heaton-lucas/tables/scalars.csv`
- Seed audit (if any): `bullshit-detector_heaton-lucas_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | KKT complementarity via min-formulation | HOLDS | -- | -- |
| 2 | Euler errors over full 24 paths | HOLDS | -- | -- |
| 3 | Equity premium 0.47% to 1.04% (scalars.csv) | HOLDS | -- | -- |
| 4 | Mean omega=0.504, p10=0.110, p90=0.897 (scalars.csv) | HOLDS | -- | -- |
| 5 | No-short 0.0%, borrowing 48.5% (scalars.csv) | HOLDS | -- | -- |
| 6 | Policy change 1.78e-02, residual 5.31e-07 (scalars.csv) | HOLDS | -- | -- |
| 7 | Euler errors table matches euler-errors.csv | HOLDS | -- | -- |
| 8 | Grid 201 pts, 19 unknowns, 8 shocks | HOLDS | -- | -- |
| 9 | 80 iterations, stopped at cap | HOLDS | -- | -- |

## Findings

### Prior finding 1 (KKT sign enforcement, DILUTED MED): RESOLVED

- **Original state:** Code used `ms1 * s1p` (product form) which does not enforce `mu >= 0`.
- **Current state:** `run.py:139-142`:
  ```python
  jnp.minimum(ms1, s1p),      # 80
  jnp.minimum(ms2, s2p),      # 81
  jnp.minimum(mb1, nb1p),     # 82
  jnp.minimum(mb2, nb2p),     # 83
  ```
  Min-formulation: `min(mu, constraint) = 0` is satisfied iff either `mu = 0` or `constraint = 0`. When `mu < 0` and `constraint > 0`, the residual is `mu < 0`, not zero -- the solver is driven away from sign-violated solutions. The full KKT condition (`mu >= 0, constraint >= 0, mu * constraint = 0`) is enforced implicitly.
- **Category:** HOLDS

### Prior finding 2 (4-path Euler loop, DILUTED LOW): RESOLVED

- **Original state:** `for p in range(min(n_paths, 4))` capped Euler errors at 4 paths despite advertising 24.
- **Current state:** `run.py:298`: `for p in range(n_paths)` with no cap. `n_paths = 24` at `run.py:283`. Euler errors are now computed over all 24 paths.
- **Category:** HOLDS

### Prior findings 3-6 (unarchived runtime scalars, DATA DRIFT LOW): RESOLVED

- **Original state:** Equity premium range, wealth share percentiles, constraint-binding shares, and convergence metrics were written into README from runtime values with no committed CSV backing.
- **Current state:** `run.py:340-358` writes `tables/scalars.csv` with nine metrics before building the report. All nine README-quoted scalars are grounded in the committed CSV:

| Metric | scalars.csv | README | Match |
|--------|-------------|--------|-------|
| eq_premium_min_pct | 0.4669 | 0.47% | yes |
| eq_premium_max_pct | 1.0447 | 1.04% | yes |
| omega_mean | 0.5041 | 0.504 | yes |
| omega_p10 | 0.1103 | 0.110 | yes |
| omega_p90 | 0.8970 | 0.897 | yes |
| no_short_share_pct | 0.0 | 0.0% | yes |
| borrow_share_pct | 48.48 | 48.5% | yes |
| final_policy_change | 0.01782 | 1.78e-02 | yes |
| max_pointwise_residual | 5.31e-07 | 5.31e-07 | yes |

- **Category:** HOLDS

## Cross-cutting patterns

- All prior findings resolved together: the complementarity fix (min-formulation) and the scalar archival (scalars.csv) address every audit gap in one pass.
- Euler error table values in README match euler-errors.csv row for row (Mean equity 4.63e-03, bond 2.51e-03; Median equity 3.14e-03, bond 8.46e-05; Max equity 1.24e-01, bond 1.43e-01).
- No new structural gaps found.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required.
1. The violated-invariant tests (product form present, 4-path cap present, scalars.csv absent) all now FAIL -- confirming the fixes removed the original bugs.
2. The honest-fix tests (min-formulation present, no path cap, scalars.csv with required keys) now PASS.
3. No further code changes needed.
