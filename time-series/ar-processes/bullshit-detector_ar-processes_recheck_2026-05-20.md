# bullshit-detector — ar-processes — recheck — 2026-05-20

**Bullshit score: 0%** — both original findings resolved: pseudocode now uses `eta_t` for the g_t update (line 87) and Results prose now says "6.6 periods" (line 94). All other HOLDS verified.

## Header
- Claim sources: `time-series/ar-processes/README.md`, `time-series/ar-processes/run.py`
- Code / artifact root: `time-series/ar-processes/run.py`
- Data artifacts: `time-series/ar-processes/tables/ar-properties.csv`
- Seed audit: `bullshit-detector_ar-processes_2026-05-20.md` (two findings: DILUTED/MED + DATA DRIFT/LOW)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Pseudocode step 3 uses `eta_t` for g_t update (distinct from `eps_t`) | HOLDS | — | — |
| 2 | Results prose says "6.6 periods" not "seven" | HOLDS | — | — |
| 3 | AR(1) closed-form variance, half-life, ACF | HOLDS | — | — |
| 4 | Multiplier-accelerator recursion | HOLDS | — | — |
| 5 | Steady-state values y_bar=5.00, c_bar=4.00 | HOLDS | — | — |
| 6 | Income roots 0.346, 0.694; largest modulus 0.694 | HOLDS | — | — |
| 7 | Spectral density formula | HOLDS | — | — |
| 8 | Table values for variance, half-life at rho=0.5, 0.9, 0.99 | HOLDS | — | — |
| 9 | AR1 and MA simulations use independent shocks | HOLDS | — | — |

## Findings

### Finding 1 (original DILUTED/MED): RESOLVED

- **Original claim (verbatim, buggy):** "3. Update x_t = rho x_{t-1} + eps_t and g_t = rho_g g_{t-1} + eps_t." — original `README.md:87`
- **Current README evidence (verbatim):** "3. Update x_t = rho x_{t-1} + eps_t and g_t = rho_g g_{t-1} + eta_t." — `README.md:87`
- **Code evidence (verbatim):** `simulate_ar1` uses `seed=42`; `simulate_multiplier_accelerator` uses `seed=43` — `run.py:21,43`. Two independent RNG streams. HOLDS.
- **Category:** HOLDS — pseudocode now correctly uses `eta_t` for the g_t update, matching the Equations section's "drawn independently of $\varepsilon_t$" at `README.md:31`.

### Finding 2 (original DATA DRIFT/LOW): RESOLVED

- **Original claim (verbatim, buggy):** "Raising $\rho$ from 0.5 to 0.9 lengthens the half-life from one to seven periods." — original `README.md:94`
- **Current README evidence (verbatim):** "Raising $\rho$ from 0.5 to 0.9 lengthens the half-life from 1.0 to 6.6 periods." — `README.md:94`
- **Corroboration:** Table row "Half-life (periods) | 1.0 | 6.6 | 69.0" — `README.md:122`. Takeaway "a shock has half of its initial effect after 6.6 periods" — `README.md:128`. All consistent. HOLDS.

### HOLDS block (all other claims verified)

**Finding 3: AR(1) closed-form variance**
- `README.md:76`: `Var(x_t) = sigma^2/(1-rho^2) = 0.000526` at rho=0.9, sigma=0.01. Calculation: 0.0001/0.19 = 0.000526. HOLDS.

**Finding 4: Multiplier-accelerator recursion**
- `README.md:44`: `y_t = beta(1+alpha)y_{t-1} - alpha*beta*y_{t-2} + g_t`. `run.py:56-59` computes c, g, investment, y per period. Algebraic expansion matches. HOLDS.

**Finding 5: Steady states**
- `README.md:67-68`: y_bar=5.00, c_bar=4.00. 1/(1-0.8)=5, 0.8*5=4. HOLDS.

**Finding 6: Income roots 0.346, 0.694**
- `README.md:78`: roots 0.346, 0.694. `np.roots([1, -0.8*1.3, 0.3*0.8])` = [0.694, 0.346]. HOLDS.

**Finding 7: Spectral density**
- `README.md:76` (implied). Code implements `sigma**2 / (2.0 * np.pi * |1 - rho*exp(-i*omega)|**2)`. Standard AR(1). HOLDS.

**Finding 8: Table values**
- `README.md:120-124`: rho=0.5 variance=0.000133, rho=0.9 variance=0.000526, rho=0.99 variance=0.005025. Verified analytically. HOLDS.

**Finding 9: Independent shocks**
- `simulate_ar1` seed=42; `simulate_multiplier_accelerator` seed=43. Different RNG streams. HOLDS.

## Cross-cutting patterns

- None. Both original findings are fully resolved. The pseudocode shock notation and the Results prose half-life value are now internally consistent with the Equations section and the analytical table.

## TDD execution sequence

0. **Bullshit score: 0%.** No issues. No action required.
1. Test `test_honest_fix_pseudocode_uses_eta_for_g` now PASSES (confirmed by pytest run 2026-05-20).
2. Test `test_honest_fix_prose_consistent_half_life` now PASSES (confirmed).
3. Both violated-invariant tests FAIL as expected (confirms fixes applied).
