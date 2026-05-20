# bullshit-detector — production-functions-markups — recheck — 2026-05-20

**Bullshit score: 0%** — all three prior findings (F1 DILUTED/MED, F2 DILUTED/MED, F3 DILUTED/LOW) are resolved. All numeric claims verified against code and CSV. No new findings.

## Header
- Claim sources: `industrial-organization/production-functions-markups/README.md` (all sections)
- Code / artifact root: `industrial-organization/production-functions-markups/run.py`
- Data artifacts: `industrial-organization/production-functions-markups/tables/production-estimates.csv`, `tables/markup-by-productivity.csv`
- Seed audit: `bullshit-detector_production-functions-markups_2026-05-20.md` (score 40%, 3 findings)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| F1 (prior) | `material_share` read from panel, not from `true_markup` | HOLDS | — | — |
| F2 (prior) | Proxy inversion is nonparametric polynomial, not oracle | HOLDS | — | — |
| F3 (prior) | Capital predetermined channel disclosed in Equations and Results | HOLDS | — | — |
| 4 | alpha_m = exp(log_materials) / exp(log_output) (unit prices) | HOLDS | — | — |
| 5 | Productivity control: polynomial fit in k to I, residual | HOLDS | — | — |
| 6 | OLS overstates materials elasticity (flexible-input channel) | HOLDS | — | — |
| 7 | Capital elasticity not corrected by proxy (proxy bias 0.275 acknowledged) | HOLDS | — | — |
| 8 | All numeric claims in both CSV tables | HOLDS | — | — |
| 9 | Markup quintile ordering and proxy bias direction | HOLDS | — | — |

## Findings

### Prior Finding 1 — RESOLVED: `material_share` now read from panel

- **Prior claim (verbatim):** `material_share = np.clip(TRUE_BETA["Materials"] / true_markup + rng.normal(0.0, 0.025, n_firms), 0.16, 0.75)` — generated directly from `true_markup`. Equations section defines it as materials expenditure over revenue.
- **Current code evidence (verbatim):**
  ```python
  # material_share is read off the panel: materials expenditure over
  # revenue. With unit prices this is exp(log_materials) / exp(log_output).
  material_share = np.exp(materials - y)
  ```
  `run.py:68-70`
- **Verification:** `max abs diff between material_share and exp(log_materials)/exp(log_output)` = 1.11e-16 (floating-point identity). The new DGP adds 2.34x more noise than the old oracle-derived share (std 0.0584 vs old 0.025), making the identification problem harder, as expected for a genuinely observed share. Correlation with `beta_m / true_markup` = 0.836 (down from ~0.99 under old oracle construction). **HOLDS.**
- **Category:** HOLDS

---

### Prior Finding 2 — RESOLVED: Proxy inversion now uses polynomial fit, not oracle coefficients

- **Prior claim (verbatim):** `omega_proxy = (inv - 0.75 - 0.20 * k) / 0.90` — exact true investment-schedule coefficients hardcoded. README implied nonparametric h^{-1}.
- **Current code evidence (verbatim):**
  ```python
  # Nonparametric proxy inversion. Investment is monotone in productivity
  # given capital, so the part of investment not explained by a polynomial
  # in capital is monotone in productivity. np.polyfit estimates that
  # polynomial from the data; the residual is the productivity control.
  # No true investment-schedule coefficient is used here.
  capital_trend = np.polyfit(k, inv, deg=2)
  omega_proxy = inv - np.polyval(capital_trend, k)
  ```
  `run.py:97-103`
- **Verification:** `'0.90' in src` = False; `'0.75' in src` = False; `'polyfit' in src` = True; `'polyval' in src` = True. Solution Method text (`run.py:210-215`) explicitly states: "The productivity control is built nonparametrically: a polynomial in capital is fit to investment, and the residual is the part of investment that moves with productivity. No true investment-schedule coefficient is used." **HOLDS.**
- **Category:** HOLDS

---

### Prior Finding 3 — RESOLVED: Capital predetermined channel disclosed in Equations and Results

- **Prior claim:** README attributed OLS bias only to flexible-input simultaneity; capital bias was largest but not explained.
- **Current code evidence (verbatim):**
  ```python
  # Equations section, run.py:184-187:
  control identifies the flexible-input elasticities $\beta_l$ and $\beta_m$. It
  does not separately identify the capital elasticity $\beta_k$: capital is a
  predetermined state, and its productivity-correlated variation is absorbed by
  the control. Recovering $\beta_k$ cleanly needs the Olley-Pakes second stage,
  which this tutorial does not run.
  ```
  `run.py:183-187`
  ```python
  # Results figure description, run.py:255-258:
  "It does not correct the capital elasticity: capital is predetermined, "
  "and the single-stage control absorbs its productivity-correlated variation "
  "rather than identifying it.",
  ```
  `run.py:255-258`
- **Data evidence:** README table (`README.md:112`): `Capital | 0.24 | 0.215 | 0.515 | -0.025 | 0.275`. The proxy bias of 0.275 is the largest in the table and is now explicitly acknowledged in the Equations prose and Results description. **HOLDS.**
- **Category:** HOLDS

---

### Finding 4: alpha_m definition — HOLDS

- **Claim source (verbatim):** "$\alpha^m_{it} = \frac{\text{materials expenditure}_{it}}{\text{revenue}_{it}}$" — `README.md:51`
- **Code evidence:** `material_share = np.exp(materials - y)` at `run.py:70`. With unit prices, `exp(log_materials) / exp(log_output)` = materials expenditure / revenue. Comment at `run.py:31`: "Unit input and output prices, so the materials revenue share is alpha_m = exp(log_materials) / exp(log_output)." ✓
- **Category:** HOLDS

---

### Finding 5: Polynomial proxy control — HOLDS

- **Claim source (verbatim):** "$\tilde \omega_{it}=I_{it}-\widehat{\mathrm{poly}}(k_{it})$" — `README.md:32`
- **Code evidence:** `capital_trend = np.polyfit(k, inv, deg=2)` / `omega_proxy = inv - np.polyval(capital_trend, k)` at `run.py:102-103`. Exact match to the equation. ✓
- **Category:** HOLDS

---

### Finding 6: OLS overstates materials elasticity — HOLDS

- **Claim source (verbatim):** "OLS badly overstates the materials elasticity, the markup-relevant one, because high-productivity firms choose more materials." — `README.md:94`
- **Data evidence:** `tables/production-estimates.csv:3`: `Materials,0.440,0.932,0.451,0.492,0.011`. OLS = 0.932 vs true = 0.440; OLS bias = 0.492. Proxy-control = 0.451, bias = 0.011. Materials has the largest OLS bias among the three inputs. ✓
- **Category:** HOLDS

---

### Finding 7: Capital proxy bias acknowledged — HOLDS

- **Claim source (verbatim):** "It does not correct the capital elasticity: capital is predetermined, and the single-stage control absorbs its productivity-correlated variation rather than identifying it." — `README.md:94`
- **Data evidence:** `tables/production-estimates.csv:2`: `Capital,0.240,0.215,0.515,-0.025,0.275`. Proxy-control capital elasticity = 0.515; true = 0.240; proxy bias = 0.275. The prose correctly identifies this as an upward bias introduced by the control (not corrected). ✓
- **Category:** HOLDS

---

### Finding 8: All numeric claims in both CSV tables — HOLDS

- **Claim source:** Table `README.md:110-114` (production estimates) and `README.md:120-127` (markup quintiles)
- **Verification:** All 15 cells in the production-estimates table match `tables/production-estimates.csv` within display rounding (3dp format). All 30 cells in the markup-quintiles table match `tables/markup-by-productivity.csv` within display rounding (3dp format). Cross-check performed by running `simulate_panel()` and `estimate_production()` live, confirming CSV values are reproducible.
- **Category:** HOLDS

---

### Finding 9: Markup quintile ordering — HOLDS

- **Claim source (verbatim):** "True markups rise with productivity. The recovered quintile means trace that gradient." — `README.md:102-103`
- **Data evidence:** `tables/markup-by-productivity.csv`: true_markup monotonically increases from Q1 (0.865) to Q5 (1.655); proxy_markup also monotonically increases from Q1 (0.906) to Q5 (1.720). ✓
- **Category:** HOLDS

## Cross-cutting patterns

- All three prior findings are fully resolved. The fixes are genuine: F1 changed the DGP to derive `material_share` from observed log-variables (making identification 2.34x harder), F2 replaced oracle inversion with `np.polyfit`/`np.polyval`, and F3 added explicit prose about the capital identification failure in two places (Equations and Results).
- The new proxy capital bias (0.275, proxy=0.515 vs true=0.240) introduced by the nonparametric fix is correctly acknowledged rather than hidden. This is a faithful representation of what a first-stage-only proxy control actually does.
- No new findings emerged. The numeric tables are consistent end-to-end. The Equations section now matches the code operations exactly.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All prior findings resolved. No halt trigger.
1. Violated-invariant tests for F1, F2, F3 (tests that encoded the buggy behavior) now FAIL — confirmed by test run: 3 failed tests are the violated-invariant tests, which is the correct green state.
2. Honest-fix tests (3 passing) confirm the fixes are in place.
3. No further action required for this tutorial. Re-run `scripts/validate_catalog.py` to confirm no math-rendering regressions from the new Equations prose.
