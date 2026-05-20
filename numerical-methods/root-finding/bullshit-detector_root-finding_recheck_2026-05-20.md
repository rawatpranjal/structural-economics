# bullshit-detector - root-finding - recheck - 2026-05-20

**Bullshit score: 5%** - All numeric claims, formulas, iteration counts, residuals, method-numbering order, and ratio arithmetic now hold against code and data. One pre-existing style violation (inline LaTeX in the Overview section, prohibited by CLAUDE.md) does not affect faithfulness of any numeric or algorithmic claim.

## Header
- Claim sources: `numerical-methods/root-finding/README.md`
- Code / artifact root: `numerical-methods/root-finding/run.py`
- Data artifacts: `numerical-methods/root-finding/tables/comparison.csv`, `numerical-methods/root-finding/tables/scipy_match.csv`
- Seed audit (if any): `numerical-methods/root-finding/bullshit-detector_root-finding_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, independent recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|-----------------|
| 1 | r* = 1/beta - 1 = 0.041667 | HOLDS | - | - |
| 2 | K* = 5.4468 | HOLDS | - | - |
| 3 | alpha=0.36, beta=0.96, delta=0.08 | HOLDS | - | - |
| 4 | K_d(r) = (alpha/(r+delta))^(1/(1-alpha)) | HOLDS | - | - |
| 5 | Z(r) = K_d(r) - K*, Z(r*)=0 | HOLDS | - | - |
| 6 | Z'(r) = -1/(1-alpha) * K_d(r)/(r+delta) < 0 | HOLDS | - | - |
| 7 | Bisection formula m_n=(a_n+b_n)/2; bracket halves each step | HOLDS | - | - |
| 8 | Bisection convergence linear rate 1/2 | HOLDS | - | - |
| 9 | Secant formula x_{n+1} = x_n - Z(x_n)*(x_n-x_{n-1})/(Z(x_n)-Z(x_{n-1})) | HOLDS | - | - |
| 10 | Secant order (1+sqrt(5))/2 approx 1.618 | HOLDS | - | - |
| 11 | Brent: IQI through last three ordinates; secant fallback; bisection fallback | HOLDS | - | - |
| 12 | Brent bracket invariant maintained every iteration | HOLDS | - | - |
| 13 | Newton formula x_{n+1} = x_n - Z(x_n)/Z'(x_n) | HOLDS | - | - |
| 14 | Newton quadratic convergence when Z'(r*)!=0 | HOLDS | - | - |
| 15 | Bisection=29, Secant=9, Brent=7, Newton=5 iterations | HOLDS | - | - |
| 16 | Brent matches scipy.brentq to 0.00e+00 | HOLDS | - | - |
| 17 | scipy_match.csv persists 0.00e+00 | HOLDS | - | - |
| 18 | 1 of 9 Newton starts diverge | HOLDS | - | - |
| 19 | Residuals: Bisection=7.23e-09, Secant=2.04e-14, Brent=9.06e-14, Newton=8.88e-16 | HOLDS | - | - |
| 20 | Error in r: Bisection=1.03e-10, Secant=2.91e-16, Brent=1.28e-15, Newton=6.94e-18 | HOLDS | - | - |
| 21 | "4-6x fewer iterations than bisection: 29/7=4.1x, 29/5=5.8x, well under one order of magnitude" | HOLDS | - | - |
| 22 | Method 3: Brent, Method 4: Newton-Raphson (Equations ordering matches table) | HOLDS | - | - |
| 23 | Overview contains inline LaTeX math ($r^{\ast}$, $r$, $\mathrm{scipy...}$) | DATA DRIFT | LOW | no |

## Findings

### Finding 1: Overview contains inline LaTeX - pre-existing CLAUDE.md contract violation

- **Claim source (verbatim):** "A representative-firm economy with Cobb-Douglas production has a closed-form clearing rate $r^{\ast} = 1/\beta - 1$. The market-clearing condition is one scalar equation in $r$." — `README.md:5`
- **Code evidence (verbatim):**
  ```python
  report.add_overview(
      "A representative-firm economy with Cobb-Douglas production has a "
      "closed-form clearing rate $r^{\\ast} = 1/\\beta - 1$. "
      "The market-clearing condition is one scalar equation in $r$.\n\n"
  ```
  `run.py:224-228`
- **Data evidence:** Not applicable. No numeric claim is affected.
- **Category:** DATA DRIFT - between CLAUDE.md style contract ("The Overview is prose only and contains no math, no equations, no inline LaTeX, and no symbolic notation") and the generated README.
- **Severity:** LOW - no faithfulness violation; no numeric, algorithmic, or comparison-table claim is wrong. Pre-existing before the fix (confirmed via `git show d09c5bd`). The fix agent did not introduce this.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "$r^{\\ast}" not in open("numerical-methods/root-finding/README.md").read().split("## Equations")[0]  # FAILS on current code
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "$r^{\\ast}" not in open("numerical-methods/root-finding/README.md").read().split("## Equations")[0]  # PASSES after removing inline math from add_overview
  ```

## Cross-cutting patterns

- All three findings from the original audit are resolved. Finding 1 (FALSE prose "order of magnitude") is now corrected to dynamically computed "4-6x / 4.1x / 5.8x / well under one order of magnitude" (run.py:511-518). Finding 2 (DATA DRIFT, unverifiable Brent-scipy residual) is resolved by `tables/scipy_match.csv` persisting the value `0.00e+00`, grounded by re-computation. Finding 3 (DATA DRIFT, method numbering) is resolved: Equations now reads Method 3: Brent, Method 4: Newton-Raphson (run.py:277,285; README.md:52,60), matching the table order.
- All numeric values in the README are generated dynamically from actual solver output (not hardcoded), which means future re-runs will keep them consistent automatically.
- The one remaining finding (inline math in Overview) is pre-existing, not introduced by the fix, and is a CLAUDE.md style rule issue rather than a faithfulness violation.
- Mathematical content (K_d, Z, Z', bisection midpoint, secant chord, IQI, Newton tangent) verified end-to-end against code. All hold.

## TDD execution sequence (for the next agent)

0. **Bullshit score = 5%.** Below 25% threshold. No halt required.

1. **Finding 1 (DATA DRIFT - pre-existing Overview inline math).**
   Optional cleanup: in `run.py`'s `add_overview` call (lines 224-235), replace `$r^{\ast} = 1/\beta - 1$` with prose "equal to one over beta minus one", replace `$r$` with "r", and replace `$\mathrm{scipy.optimize.brentq}$` with "`scipy.optimize.brentq`" (backtick-quoted as code). Re-run `python run.py` and confirm the Overview section of the generated README contains no dollar signs. This is a style fix only and does not require a new audit pass.
