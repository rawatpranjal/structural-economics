# bullshit-detector — mixed-logit-simulation — recheck — 2026-05-20

**Bullshit score: 0%** — All claims hold. The original Finding 1 (SD bounds row omitted the log-sigma qualifier) is resolved: the row now reads "Effective sigma range; the bound is enforced on log-sigma for both random coefficients." All numeric claims match committed CSVs exactly. All algorithmic claims grounded line-by-line in code.

## Header
- Claim sources: `choice/mixed-logit-simulation/README.md`
- Code / artifact root: `choice/mixed-logit-simulation/run.py`
- Data artifacts: `tables/parameter-recovery.csv`, `tables/share-fit.csv`, `tables/price-substitution-matrix.csv`
- Seed audit: `bullshit-detector_mixed-logit-simulation_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | n_consumers=1500, J=4, R=120 | HOLDS | — | — |
| 2 | True theta=(-1.00, 1.10, 0.36, 0.55) | HOLDS | — | — |
| 3 | L-BFGS-B, maxiter=220, start=(-0.75, 0.85, log 0.25, log 0.35) | HOLDS | — | — |
| 4 | Price-taste bound [-3.00, -0.05]; quality-taste bound [0.05, 2.50] | HOLDS | — | — |
| 5 | SD bounds [0.03, 1.30] — effective sigma range; bound enforced on log-sigma | HOLDS | — | — |
| 6 | Probability floor 1e-14 | HOLDS | — | — |
| 7 | Profile grid 21x21 | HOLDS | — | — |
| 8 | Draws made once, held fixed (common random numbers) | HOLDS | — | — |
| 9 | sigma=exp(log_sigma) inside estimator | HOLDS | — | — |
| 10 | P_hat=(1/R) sum P_ij(theta, nu_r) | HOLDS | — | — |
| 11 | SML objective = argmax sum log P_hat_{i,yi} | HOLDS | — | — |
| 12 | Q_R=-ell_R/N (division by N) | HOLDS | — | — |
| 13 | D_jk formula (numerator/denominator) | HOLDS | — | — |
| 14 | D_kk=-1 | HOLDS | — | — |
| 15 | Price step Delta_p=0.10 | HOLDS | — | — |
| 16 | Parameter recovery table numbers vs CSV | HOLDS | — | — |
| 17 | Share fit table numbers vs CSV | HOLDS | — | — |
| 18 | Price substitution table numbers vs CSV | HOLDS | — | — |

## Findings

### Original Finding 1 (from seed audit): SD bounds row omitted log-space qualifier — RESOLVED

- **Original claim:** Model Setup SD bounds row said "Applied to both random-coefficient standard deviations" without naming the log transform.
- **Current state:** `run.py:488` now generates: `f"| SD bounds | [{np.exp(MIXED_BOUNDS[2][0]):.2f}, {np.exp(MIXED_BOUNDS[2][1]):.2f}] | Effective sigma range; the bound is enforced on log-sigma for both random coefficients |"`. README.md row reads: `| SD bounds | [0.03, 1.30] | Effective sigma range; the bound is enforced on log-sigma for both random coefficients |`. Both the effective range and the log-sigma mechanism are now stated.
- **Resolution:** RESOLVED. DILUTED → HOLDS.

## Cross-cutting patterns

None. All claims verified end-to-end. All numeric claims in README tables are byte-exact copies of committed CSV rows. No DATA DRIFT detected.

## TDD execution sequence (for the next agent)

None required. Score is 0%. The honest-fix test (`assert re.search(r"log[- ]?sigma|log scale|log standard deviation", row)`) now passes on the SD bounds row. The violated-invariant test now correctly fails, confirming the fix is in place.
