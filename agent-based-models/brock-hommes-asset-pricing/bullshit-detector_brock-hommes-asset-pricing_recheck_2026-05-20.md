# bullshit-detector — brock-hommes-asset-pricing — recheck — 2026-05-20

**Bullshit score: 0%** — all 27 claims hold; both original omissions (c_F = 0 undisclosed, initial conditions x_0/x_1 undisclosed) are now present as explicit rows in the Model Setup table; all equations and SMM table values match committed CSV.

## Header
- Claim sources: `agent-based-models/brock-hommes-asset-pricing/README.md`
- Code / artifact root: `agent-based-models/brock-hommes-asset-pricing/run.py`, `lib/brock_hommes.py`
- Data artifacts: `agent-based-models/brock-hommes-asset-pricing/tables/smm-fit.csv`
- Seed audit: `agent-based-models/brock-hommes-asset-pricing/bullshit-detector_brock-hommes-asset-pricing_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | p* = d/(R-1) | HOLDS | - | - |
| 2 | f_F,t = 0 (fundamentalist forecast zero) | HOLDS | - | - |
| 3 | f_T,t = xbar * tanh(tilde_f / xbar) | HOLDS | - | - |
| 4 | Market clearing: x_t = (n_F f_F + n_T f_T)/R + eps | HOLDS | - | - |
| 5 | pi_h,t = e_t*(f_h,t - R*x_{t-1})/(a*sigma^2) - c_h | HOLDS | - | - |
| 6 | Score smoothing: U = lambda*U + (1-lambda)*pi | HOLDS | - | - |
| 7 | Logit shares: n_h = exp(beta*U_h) / sum_j exp(beta*U_j) | HOLDS | - | - |
| 8-15 | All calibration table values match Params defaults | HOLDS | - | - |
| 16 | c_F = 0.000 (Fundamentalist cost row now present in table) | HOLDS | - | - |
| 17 | c_T = 0.001 (Trend cost) | HOLDS | - | - |
| 18 | x_0 = 0.10 (Initial deviation lag row now present) | HOLDS | - | - |
| 19 | x_1 = 0.12 (Initial deviation row now present) | HOLDS | - | - |
| 20 | T_sim = 700 | HOLDS | - | - |
| 21 | T_0 = 100 | HOLDS | - | - |
| 22-23 | Plotted betas 2/20/50; SMM true=30, grid 2-60 | HOLDS | - | - |
| 24-27 | SMM table values match smm-fit.csv exactly | HOLDS | - | - |

## Findings

### Finding 1 (original): c_F = 0 not disclosed in Model Setup — RESOLVED

- **Original omission:** Model Setup table listed `c_T = 0.001` but omitted `c_F = 0`. Code `lib/brock_hommes.py:88` uses `costs = np.array([0.0, params.trend_cost])`, making the asymmetry (fundamentalist has zero cost, trend-follower pays 0.001) invisible to a reader reproducing from the table alone.
- **Current README (verbatim):** `| Fundamentalist cost | $c_F$ | 0.000 | Zero cost for the fundamental rule |` — `README.md:88`
- **Current run.py source (verbatim):**
  ```python
  f"| Fundamentalist cost | $c_F$ | 0.000 | Zero cost for the fundamental rule |\n"
  ```
  `run.py:221`
- **Code (verbatim):** `costs = np.array([0.0, params.trend_cost])` — `lib/brock_hommes.py:88`
- **Category:** HOLDS (was DILUTED LOW in original audit)

### Finding 2 (original): Initial conditions x_0, x_1 not in Model Setup — RESOLVED

- **Original omission:** Model Setup algorithm step 1 said "initialize x_0, x_1" without giving values. `lib/brock_hommes.py:33-34` sets `x_lag=0.10`, `x0=0.12`.
- **Current README (verbatim):**
  - `| Initial deviation lag | $x_0$ | 0.10 | Starting price deviation |` — `README.md:90`
  - `| Initial deviation | $x_1$ | 0.12 | Starting price deviation |` — `README.md:91`
- **Current run.py source (verbatim):**
  ```python
  f"| Initial deviation lag | $x_0$ | {params.x_lag:.2f} | Starting price deviation |\n"
  f"| Initial deviation | $x_1$ | {params.x0:.2f} | Starting price deviation |\n"
  ```
  `run.py:223-224`
- **Code (verbatim):** `x_lag: float = 0.10` / `x0: float = 0.12` — `lib/brock_hommes.py:33-34`; used at `x[0] = params.x_lag` / `x[1] = params.x0` — `lib/brock_hommes.py:82-83`
- **Category:** HOLDS (was DILUTED LOW in original audit)

## Cross-cutting patterns

None. Both original findings were omissions from the Model Setup table. Both are now present as explicit rows with values derived from live `params.*` f-strings. No systematic gap remains.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 0%.** Ship. No further action required.
1. Tests `test_finding1_violated_invariant_initial_conditions_absent` and `test_finding2_violated_invariant_fundamentalist_cost_absent` now FAIL — confirming the disclosures are present.
2. Tests `test_finding1_honest_fix_initial_conditions_disclosed` and `test_finding2_honest_fix_fundamentalist_cost_disclosed` now PASS — confirming the fix is correctly applied.
3. No further code or prose changes warranted.
