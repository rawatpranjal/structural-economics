# bullshit-detector — fred-macro-data — recheck — 2026-05-20

**Bullshit score: 0%** — original Finding 1 (sigma notation overload) resolved: `run.py:250` and `README.md:28` now use `\sigma^{y}` for the DGP scaling vector and `\sigma^{c}_j` for the HP-cycle standard deviation, with explicit prose at `README.md:31` explaining the distinction.

## Header
- Claim sources: `time-series/fred-macro-data/README.md`, `time-series/fred-macro-data/run.py`
- Code / artifact root: `time-series/fred-macro-data/run.py`
- Data artifacts: `time-series/fred-macro-data/tables/business-cycle-stats.csv`
- Seed audit: `bullshit-detector_fred-macro-data_2026-05-20.md` (one DILUTED/LOW finding)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | sigma^y (DGP scaling) and sigma^c (cycle std) are notionally distinct in equations | HOLDS | — | — |
| 2 | DGP formula s_t = rho*s_{t-1} + sqrt(1-rho^2)*eps_t, eps~N(0,C) | HOLDS | — | — |
| 3 | HP filter solves (I + lambda K'K) tau = y | HOLDS | — | — |
| 4 | K is correct (T-2) x T second-difference matrix | HOLDS | — | — |
| 5 | T=200, T_B=5000, lambda=1600 match code | HOLDS | — | — |
| 6 | Series means, stds, persistence, corr matrix match code | HOLDS | — | — |
| 7 | All table numbers match tables/business-cycle-stats.csv | HOLDS | — | — |
| 8 | Okun regression direction: c_u on c_g | HOLDS | — | — |
| 9 | Okun slope -0.142 consistent with table moments | HOLDS | — | — |
| 10 | Unemployment most persistent cycle (autocorr 0.649) | HOLDS | — | — |
| 11 | Sign claims (countercyclical unemp, procyclical CPI/Fed) | HOLDS | — | — |
| 12 | Innovation correlation matrix correctly labelled (not cycle corr) | HOLDS | — | — |
| 13 | Benchmark uses same process for sampling variation | HOLDS | — | — |

## Findings

### Finding 1 (original DILUTED/LOW): RESOLVED

- **Original claim (verbatim, buggy):** The Equations section used `$\sigma$` for both the DGP scaling vector and the HP-cycle standard deviation without distinguishing notation.
- **Current run.py evidence (verbatim):**
  ```
  y_t=\mu+\sigma^{y}\odot s_t.
  ```
  `run.py:250`
  ```
  The vector $s_t$ is a standardized latent state and $\sigma^{y}$ is the 4-vector of series standard deviations (3.0, 1.5, 1.5, 3.0). It is a separate quantity from the HP-cycle standard deviation $\sigma^{c}_j = \mathrm{sd}(c_{j,t})$ defined below; the superscripts $y$ and $c$ keep the DGP scaling and the cycle moment distinct.
  ```
  `run.py:253`
  ```
  \sigma^{c}_j=\mathrm{sd}(c_{j,t}),\qquad
  ```
  `run.py:270`
- **README evidence (verbatim):** `\sigma^{y}` at `README.md:28`; `\sigma^{c}_j` at `README.md:48`; prose distinguishing the two at `README.md:31`. HOLDS.
- **Category:** HOLDS — both superscripted forms present; prose explicitly names the distinction.

### HOLDS block (all other claims verified, unchanged from original audit)

**Finding 2: DGP formula** — `README.md:24-29`, `run.py:85-93`. Element-wise match. HOLDS.
**Finding 3: HP filter linear system** — `README.md:41-43`, `run.py:55`. Exact match. HOLDS.
**Finding 4: K matrix** — `run.py:51-54` uses `spdiags([e, -2e, e], [0,1,2], T-2, T)`. Correct. HOLDS.
**Finding 5: Scalar parameters** — T=200, T_B=5000, lambda=1600 at `run.py:177-179`. HOLDS.
**Finding 6: Series primitives** — all four series means, stds, persistence, corr match `run.py:68-80`. HOLDS.
**Finding 7: Table numbers** — `tables/business-cycle-stats.csv` matches `README.md:138-143`. HOLDS.
**Finding 8: Okun direction** — x=GDP, y=Unemployment in `run.py:138-141`. HOLDS.
**Finding 9: Okun slope** — -0.423 * 0.975 / 2.896 = -0.142. HOLDS.
**Finding 10: Unemployment most persistent** — autocorr 0.649 > 0.599 > 0.483 > 0.282. HOLDS.
**Finding 11: Sign claims** — corr(Unemp,GDP)=-0.423 (counter), corr(CPI,GDP)=+0.174 (pro), corr(Fed,GDP)=+0.187 (pro). HOLDS.
**Finding 12: Innovation correlation labelling** — variable `corr` passed to `rng.multivariate_normal`. Correctly characterised. HOLDS.
**Finding 13: Benchmark sampling** — `generate_synthetic_macro_data(T=T_benchmark, seed=2026, dated=False)` same function, different seed. HOLDS.

## Cross-cutting patterns

- None. The single original DILUTED finding is fully resolved. The notation fix is propagated consistently across both `run.py` (source of truth) and the generated `README.md`.

## TDD execution sequence

0. **Bullshit score: 0%.** No issues. No action required.
1. Test `test_honest_fix_sigma_distinguished` now PASSES (confirmed by pytest run 2026-05-20).
2. Test `test_honest_fix_both_variants_present` now PASSES.
3. `test_violated_invariant_sigma_not_distinguished` now FAILS as expected (confirms fix applied).
