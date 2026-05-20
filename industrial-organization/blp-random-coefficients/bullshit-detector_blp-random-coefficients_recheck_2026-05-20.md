# bullshit-detector — blp-random-coefficients — recheck — 2026-05-20

**Bullshit score: 0%** — all four original DATA DRIFT findings (runtime diagnostic scalars unbacked by any CSV) resolved by adding `tables/convergence-diagnostics.csv`; grid-evaluation count disclosure added to prose; all formula, parameter, and diagnostic claims verified HOLDS against code and both CSV artifacts.

## Header
- Claim sources: `industrial-organization/blp-random-coefficients/README.md`
- Code / artifact root: `industrial-organization/blp-random-coefficients/run.py`
- Data artifacts: `industrial-organization/blp-random-coefficients/tables/parameter-estimates.csv`, `industrial-organization/blp-random-coefficients/tables/convergence-diagnostics.csv`
- Seed audit: `bullshit-detector_blp-random-coefficients_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Contraction converged in 627 iterations | HOLDS (fixed) | — | no |
| 2 | max delta recovery error = 2.45e-11 | HOLDS (fixed) | — | no |
| 3 | GMM evaluated objective 46 times; grid separate at 25 | HOLDS (fixed) | — | no |
| 4 | Largest own-elasticity error = 0.315 | HOLDS (fixed) | — | no |
| 5 | Share formula: s_jt = (1/ns) Σ logit | HOLDS | — | no |
| 6 | Contraction update: delta += log(s_obs) - log(s_pred) | HOLDS | — | no |
| 7 | mu_ijt = sigma_x*nu_i1*x_jt + sigma_p*nu_i2*p_jt | HOLDS | — | no |
| 8 | Initial delta from simple-logit inversion | HOLDS | — | no |
| 9 | Instruments include cost shifter and sums of rival characteristics | HOLDS | — | no |
| 10 | Price endogenous Cov(p,xi) != 0 | HOLDS | — | no |
| 11 | Parameter table values match CSV | HOLDS | — | no |
| 12 | Outside good utility normalized to zero | HOLDS | — | no |
| 13 | Logit IIA: each column identical off-diagonal entries | HOLDS | — | no |
| 14 | GMM objective Q = n g(sigma)' W g(sigma) | HOLDS | — | no |

## Findings

### Finding 1 (original): DATA DRIFT — contraction iteration count unbacked — RESOLVED

- **Original gap:** `len(conv_history)` (627) appeared in README prose at `run.py:609-610` with no CSV backing.

- **Current code evidence:**
  ```python
  # run.py:750-766
  diag_df = pd.DataFrame({
      "Diagnostic": [
          "contraction_iters",
          "max_delta_error",
          "grid_evals",
          "gmm_nfev",
          "max_own_elast_error",
      ],
      "Value": [
          f"{len(conv_history)}",
          f"{max_delta_error:.2e}",
          f"{grid_evals}",
          f"{result.nfev}",
          f"{max_own_elast_error:.3f}",
      ],
  })
  report.add_table("tables/convergence-diagnostics.csv", ...)
  ```

- **Data evidence:** `tables/convergence-diagnostics.csv:2` — `contraction_iters,627`. `README.md:91` — "the contraction converged in **627 iterations**". All three sources agree.

- **Resolution:** `convergence-diagnostics.csv` row `contraction_iters,627` backs the prose claim. Finding fully resolved.

- **Category:** HOLDS (post-fix)

### Finding 2 (original): DATA DRIFT — max delta recovery error unbacked — RESOLVED

- **Original gap:** `max_delta_error` (2.45e-11) appeared in README prose at `run.py:611` with no CSV backing.

- **Current code evidence:** `run.py:491` — `max_delta_error = np.max(np.abs(delta_recovered - delta_true))`. `run.py:761` — `f"{max_delta_error:.2e}"` written to `convergence-diagnostics.csv`.

- **Data evidence:** `convergence-diagnostics.csv:3` — `max_delta_error,2.45e-11`. `README.md:91` — `2.45e-11`. All agree.

- **Resolution:** Finding fully resolved.

- **Category:** HOLDS (post-fix)

### Finding 3 (original): DATA DRIFT — GMM nfev unbacked, grid undisclosed — RESOLVED

- **Original gap (two-part):** (a) `result.nfev` (46) appeared in README prose with no CSV backing. (b) README failed to disclose that the 25-point starting grid was a separate set of evaluations.

- **Current code evidence (disclosure fix):**
  ```python
  # run.py:612-617
  f"The GMM search first ran a coarse starting grid, where the grid "
  f"evaluated the objective {grid_evals} times. The Nelder-Mead refinement "
  f"from the best grid point then evaluated the objective {result.nfev} "
  f"more times. The convergence diagnostics in the Results table record "
  f"these counts so they can be checked against a fresh run."
  ```

- **Current code evidence (CSV backing):** `run.py:762-763` — `f"{grid_evals}"` and `f"{result.nfev}"` both written to `convergence-diagnostics.csv`.

- **Data evidence:** `convergence-diagnostics.csv:4-5` — `grid_evals,25` and `gmm_nfev,46`. `README.md:93` — "the grid evaluated the objective 25 times. The Nelder-Mead refinement from the best grid point then evaluated the objective 46 more times." All agree.

- **Resolution:** Both parts resolved. Grid evaluations (25) and Nelder-Mead evaluations (46) are now separately named in prose and backed by CSV. Finding fully resolved.

- **Category:** HOLDS (post-fix)

### Finding 4 (original): DATA DRIFT — own-elasticity error unbacked — RESOLVED

- **Original gap:** `max_own_elast_error` (0.315) appeared in README prose at `run.py:665` with no CSV backing.

- **Current code evidence:** `run.py:492` — `max_own_elast_error = np.max(np.abs(own_elast_blp - own_elast_true))`. `run.py:764` — `f"{max_own_elast_error:.3f}"` written to `convergence-diagnostics.csv`.

- **Data evidence:** `convergence-diagnostics.csv:6` — `max_own_elast_error,0.315`. `README.md:103` — "The largest own-elasticity error is 0.315 in this market." All agree.

- **Resolution:** Finding fully resolved.

- **Category:** HOLDS (post-fix)

## Grounded HOLDS findings (key structural claims verified fresh)

**H1 — Share formula.** `README.md:36` — `s_jt = (1/ns) Σ_i exp(δ+μ)/(1+Σ_k exp(δ+μ))`. Code `run.py:121-133` — `mu` broadcast correctly; `denom = 1.0 + exp_V.sum(axis=2, keepdims=True)`; `prob = exp_V / denom`; `shares = prob.mean(axis=0)`. Exact match. HOLDS.

**H2 — Contraction update.** `README.md:41` — `δ^(r+1) = δ^(r) + log(s_obs) - log(s_pred)`. Code `run.py:230-231` — `update = np.log(s_obs) - np.log(s_pred)` / `delta = delta + update`. Exact match. HOLDS.

**H3 — mu decomposition.** `README.md:32` — `μ_ijt = σ_x ν_i1 x_jt + σ_p ν_i2 p_jt`. Code `run.py:121-122` — `mu = (sigma_x * nu[:,0][:, None, None] * x[None,...] + sigma_p * nu[:,1][:, None, None] * p[None,...])`. Exact match. HOLDS.

**H4 — Initial delta.** `README.md:77` — "Initialize delta with the simple-logit inversion log(s_jt) - log(s_0t)". Code `run.py:219-221` — `s0 = 1.0 - s_obs.sum(axis=1, keepdims=True)` / `delta = np.log(np.maximum(s_obs, 1e-15)) - np.log(s0)`. Exact match. HOLDS.

**H5 — Instruments.** `README.md:50` — "cost shifter and sums of rival characteristics". Code `run.py:255-268` — `z_flat` (cost shifter), `sum_x_others.flatten()`, `sum_z_others.flatten()` (BLP-style rival sums). HOLDS.

**H6 — Price endogeneity.** `README.md:51` — `Cov(p,ξ) ≠ 0`. Code `run.py:68` — `p = 1.0 + 0.5 * x + 0.8 * z + 0.5 * xi + ...`. Coefficient on `xi` is 0.5. HOLDS.

**H7 — Parameter table.** `README.md:119-125` — β₀=2.000→1.969; β_x=1.500→1.576; α=-0.800→-0.835; σ_x=0.800→0.951; σ_p=0.300→0.196. `parameter-estimates.csv:2-6` — exact match. HOLDS.

**H8 — Outside good normalized.** `README.md:28` — "outside good has utility normalized to zero". Code `run.py:129` — `denom = 1.0 + exp_V.sum(...)`. The `1.0` is `exp(0)` for the outside good. HOLDS.

**H9 — Logit IIA.** `README.md:111` — "each column has identical off-diagonal entries". Code `run.py:354-358` — off-diagonal entry j≠k: `-alpha * prices[k] * shares[k]`. Independent of j. Column-constant. HOLDS.

**H10 — GMM objective.** `README.md:82` — `Q = n g(σ)' W g(σ)`. Code `run.py:319-321` — `moments = Z_iv.T @ xi / n` (this is g); `W = np.linalg.inv(Z_iv.T @ Z_iv / n)`; `obj = n * moments.T @ W @ moments`. Exact match. HOLDS.

## Cross-cutting patterns

- All four original DATA DRIFT findings shared one root cause: five runtime scalars committed to README prose with no backing CSV. A single new `tables/convergence-diagnostics.csv` written at `run.py:750-777` resolves all four simultaneously.
- The prose at `README.md:93` now correctly separates grid evaluations (25) from Nelder-Mead evaluations (46) and names each count explicitly, addressing the mild imprecision flagged in the original audit.
- `compute_share_jacobian` (`run.py:138-182`) remains dead code (defined but never called). This is not a faithfulness violation — no README claim depends on it — but it is a latent trap. Not counted as a finding.
- No new faithfulness gaps identified. All mathematical, algorithmic, and numeric claims verified HOLDS.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings read HOLDS. No further fixes required.
1. Test suite status: all four violated-invariant tests FAIL (correct — `convergence-diagnostics.csv` now exists); all four honest-fix tests PASS (correct — CSV contains all five diagnostic rows matching README prose).
2. No re-run needed beyond the already-committed CSV and prose updates.
</content>
