# bullshit-detector — particle-filter — recheck — 2026-05-20

**Bullshit score: 0%** — the original DILUTED finding is resolved: `run.py:284` now calls `measurement_noise_sweep(n_periods, n_particles=500, n_runs=50)`, matching the Model Setup table; the sweep CSV was regenerated and all 8 rows match the README table exactly; all algorithmic claims, parameters, and baseline results hold.

## Header
- Claim sources: `computational-methods/particle-filter/README.md`
- Code / artifact root: `computational-methods/particle-filter/run.py`
- Data artifacts: `computational-methods/particle-filter/tables/filter-summary.csv`, `computational-methods/particle-filter/tables/measurement-noise-sweep.csv`
- Seed audit (if any): `computational-methods/particle-filter/bullshit-detector_particle-filter_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Sweep uses 500 particles / 50 runs (matching Model Setup) | HOLDS | — | — |
| 2 | PSI = [1.0, 0.9] | HOLDS | — | — |
| 3 | PHI = diag(0.4, 0.5) | HOLDS | — | — |
| 4 | Measurement std = 0.10 | HOLDS | — | — |
| 5 | Process std = (0.30, 0.25) | HOLDS | — | — |
| 6 | Baseline particles = 500 | HOLDS | — | — |
| 7 | Repeated runs = 50 | HOLDS | — | — |
| 8 | Bootstrap weights = observation likelihood only | HOLDS | — | — |
| 9 | Optimal proposal conditions on y_t before drawing state | HOLDS | — | — |
| 10 | ESS = 1/sum_i(w_i)^2 on normalized weights | HOLDS | — | — |
| 11 | Bootstrap PF RMSE = 0.0273 | HOLDS | — | — |
| 12 | Optimal PF RMSE = 0.0100 | HOLDS | — | — |
| 13 | Bootstrap Mean ESS = 121.829 | HOLDS | — | — |
| 14 | Optimal Mean ESS = 492.636 | HOLDS | — | — |
| 15 | Bootstrap Loglike sd = 0.6914 | HOLDS | — | — |
| 16 | Optimal Loglike sd = 0.0397 | HOLDS | — | — |
| 17 | All 8 sweep table rows match CSV | HOLDS | — | — |

## Findings

### Finding 1 (original): Sweep uses undisclosed 350-particle / 20-run config — NOW HOLDS

- **Claim source (verbatim):** `"The baseline repeats each filter 50 times with 500 particles."` — `README.md:149`; `"| Baseline particles | 500 |"` — `README.md:107`; `"| Repeated runs | 50 |"` — `README.md:108`
- **Code evidence (verbatim):**
  ```python
  measurement_table = measurement_noise_sweep(n_periods, n_particles=500, n_runs=50)
  ```
  `run.py:284`
- **Category:** HOLDS — the call at `run.py:284` now uses `n_particles=500, n_runs=50`, matching both the Model Setup table and the Results prose. The original finding (`n_particles=350, n_runs=20`) no longer holds; the string `"n_particles=350"` does not appear anywhere in `run.py`.
- **Original finding resolved:** yes.

### Data cross-check: sweep CSV regenerated with 500/50

The committed `tables/measurement-noise-sweep.csv` is consistent with a 500-particle run. Optimal ESS values across all four measurement_std settings range from 477 to 496, approaching the particle ceiling of 500. These values are incompatible with a 350-particle run (which would produce a ceiling of 350). The CSV was regenerated after the fix.

Full cross-check against `README.md:162-171`:

| measurement_std | Method | CSV PF RMSE | README | CSV Mean ESS | README |
|----------------|--------|-------------|--------|--------------|--------|
| 0.2500 | bootstrap | 0.0209 | 0.0209 ✓ | 245.6403 | 245.64 ✓ |
| 0.2500 | optimal | 0.0112 | 0.0112 ✓ | 477.1401 | 477.14 ✓ |
| 0.1500 | bootstrap | 0.0254 | 0.0254 ✓ | 165.9904 | 165.99 ✓ |
| 0.1500 | optimal | 0.0106 | 0.0106 ✓ | 485.5555 | 485.555 ✓ |
| 0.1000 | bootstrap | 0.0312 | 0.0312 ✓ | 115.9987 | 115.999 ✓ |
| 0.1000 | optimal | 0.0099 | 0.0099 ✓ | 491.2554 | 491.255 ✓ |
| 0.0500 | bootstrap | 0.0425 | 0.0425 ✓ | 60.2258 | 60.2258 ✓ |
| 0.0500 | optimal | 0.0097 | 0.0097 ✓ | 496.2063 | 496.206 ✓ |

Loglike sd and Kalman RMSE columns also match exactly. No drift on any cell.

### Baseline filter-summary.csv cross-check

- `filter-summary.csv:2`: `bootstrap,500,0.0273,121.8292,0.6914` — README shows `0.0273 / 121.829 / 0.6914` ✓
- `filter-summary.csv:3`: `optimal,500,0.0100,492.6356,0.0397` — README shows `0.01 / 492.636 / 0.0397` ✓

Display rounding only (121.8292 → 121.829, 492.6356 → 492.636). Not a discrepancy.

## Cross-cutting patterns

- The fix unified the sweep call to use the same `n_particles` and `n_runs` variables that the baseline block uses (`run.py:260-261`). The Model Setup table is also generated from those same variables (`run.py:409-410`). All three locations now share a single source of truth.
- The sweep CSV was regenerated. The ESS ceiling evidence confirms regeneration with 500 particles.
- No new gaps found. All 17 claims hold.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required.
1. `test_sweep_uses_documented_particle_count` — the violated-invariant test (`"n_particles=350" in src`) now FAILS on current code (the fix removed `n_particles=350`). This is the expected post-fix state for a violated-invariant test.
2. `test_sweep_config_matches_readme` — the honest-fix test (`"measurement_noise_sweep(n_periods, n_particles=500, n_runs=50)" in src`) PASSES on current code. This is the expected green state.
3. No further action needed for this tutorial.
