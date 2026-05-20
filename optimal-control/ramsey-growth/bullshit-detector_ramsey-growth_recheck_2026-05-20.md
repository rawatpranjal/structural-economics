# bullshit-detector -- ramsey-growth -- recheck -- 2026-05-20

**Bullshit score: 0%** -- all structural equations, parameter values, steady-state formulas, ODE system, algorithm description, and every table cell hold exactly against code and CSV; both original findings (DATA DRIFT on c0 truncation, MISLABELED column label) are fully resolved.

## Header
- Claim sources: `optimal-control/ramsey-growth/README.md` (full file, 139 lines)
- Code / artifact root: `optimal-control/ramsey-growth/run.py` (420 lines)
- Data artifacts: `optimal-control/ramsey-growth/tables/shooting-results.csv`
- Seed audit: `optimal-control/ramsey-growth/bullshit-detector_ramsey-growth_2026-05-20.md`
- Run by: bullshit-detector skill (Claude Sonnet 4.6), 2026-05-20 (recheck)
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | ODE: k-dot = f(k) - delta*k - c, c-dot = [(f'(k)-delta-rho)/sigma]*c | HOLDS | - | - |
| 2 | Steady state: f'(k*) = rho+delta, k*=(alpha*A/(rho+delta))^(1/(1-alpha)) | HOLDS | - | - |
| 3 | k* = 8.2898, c* = 1.5952 (analytical) | HOLDS | - | - |
| 4 | Terminal gap G = k(T; c0) - k* (positive = c0 too low) | HOLDS | - | - |
| 5 | Brent's method (brentq, xtol=rtol=1e-11, maxiter=200) | HOLDS | - | - |
| 6 | lambda_stable = min(eigvals(J)) at steady state | HOLDS | - | - |
| 7 | Parameters: alpha=0.33, delta=0.05, rho=0.03, sigma=2.0, A=1.0, T=150 | HOLDS | - | - |
| 8 | Column label "Relative terminal capital gap" (|k(T)-k*|/k*) | HOLDS | - | - |
| 9 | All 5 c0 values match between README and CSV | HOLDS | - | - |
| 10 | All 5 terminal gap values match between README and CSV | HOLDS | - | - |

## Findings

### Finding 1 (RESOLVED from original audit): c0 truncation DATA DRIFT

The original audit Finding 1 claimed README c0 values were truncated to 5 decimal places while the CSV used 6. Independent check: the current format string at `run.py:366` is `{c0:.6g}` (6 significant figures). For c0=1.168448..., `{:.6g}` produces "1.16845" -- 6 significant figures but 5 decimal digits. The CSV also records "1.16845". README and CSV are identical. The original audit confused decimal places with significant figures.

Current README.md:121: `1.16845`. CSV row 3: `1.16845`. IDENTICAL. **RESOLVED.**

### Finding 2 (RESOLVED from original audit): Column label MISLABELED

The original audit Finding 2 flagged the column "Terminal capital gap" as mislabeled because the code computes `|k(T)-k*|/k*` (a dimensionless ratio), not an absolute gap. Current `run.py:377` names the column `"Relative terminal capital gap"`. The README prose at README.md:114 says "the relative terminal capital gap, the distance $|k(T)-k^{*}|$ expressed as a fraction of $k^{*}$". The CSV header is "Relative terminal capital gap". Code, README, and CSV are all consistent and accurate. **RESOLVED.**

## Cross-cutting patterns

- Both original findings are resolved. No new findings arose on fresh audit.
- The terminal gap values differ in minor text representation only (README: "3.7e-07", CSV: "3.70e-07") but are numerically equal. This is a cosmetic formatting artifact, not a DATA DRIFT finding.
- All load-bearing math: ODE system (`run.py:53-54`), steady state (`run.py:33-34`), terminal gap (`run.py:80-81`), Jacobian (`run.py:130-135`), Brent root search (`run.py:102-111`) match their corresponding README descriptions exactly.
- The `{:.6g}` format at `run.py:366` produces 6 significant figures, which for numbers near 1.0 gives 5 or 6 decimal digits depending on the leading digit count. This is correct behavior and consistent between README and CSV.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** Nothing to fix. All findings HOLDS.
1. The four tests in `tests/test_ramsey_growth.py` cover the resolved findings. All 4 pass on current code.
2. Re-run `python scripts/validate_catalog.py` before any commit to confirm no regressions.
