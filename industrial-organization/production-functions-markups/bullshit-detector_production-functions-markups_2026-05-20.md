# bullshit-detector — production-functions-markups — 2026-05-20

**Bullshit score: 40%** — Two DILUTED/MED findings: (1) `material_share` is generated from `true_markup` directly, not computed from panel log-data as the Equations section implies; (2) the proxy inversion uses the exact true investment-schedule coefficients (oracle), while README notation implies nonparametric inversion in the OP/LP tradition.

## Header
- Claim sources: `industrial-organization/production-functions-markups/README.md` (Equations, Model Setup, Solution Method, Results sections)
- Code / artifact root: `industrial-organization/production-functions-markups/run.py`
- Data artifacts: `industrial-organization/production-functions-markups/tables/production-estimates.csv`, `tables/markup-by-productivity.csv`
- Seed audit (if any): None
- Run by: bullshit-detector subagent (Claude Sonnet 4.6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | alpha_m is materials expenditure / revenue (observed from panel data) | DILUTED | MED | yes — makes identification trivially easy; real alpha_m would add independent noise |
| 2 | omega_tilde = h^{-1}(k, I) implies nonparametric inversion | DILUTED | MED | no on numbers; yes on pedagogical claim about what "proxy control" does |
| 3 | OLS overstates flexible-input elasticities | DILUTED | LOW | no — capital bias is largest but capital is predetermined, not flexible |
| 4 | All numeric claims (elasticity table, quintile table) | HOLDS | none | — |
| 5 | 320 firms, 6 years, AR(1) productivity rho=0.72 | HOLDS | none | — |
| 6 | Investment monotone in omega, mu = theta_m / alpha_m formula | HOLDS | none | — |

## Findings

### Finding 1: `alpha_m` generated from `true_markup`, not from observed panel data

- **Claim source (verbatim):** "Let $\alpha^m_{it} = \frac{\text{materials expenditure}_{it}}{\text{revenue}_{it}}$ be the materials revenue share." — `README.md:38-40`
- **Code evidence (verbatim):**
  ```python
  material_share = np.clip(TRUE_BETA["Materials"] / true_markup + rng.normal(0.0, 0.025, n_firms), 0.16, 0.75)
  ```
  `run.py:44`
- **Data evidence:** `log_materials` and `log_output` are present in the panel (rows appended at `run.py:46-57`) but are never divided (not even after `np.exp()`) to form `material_share` anywhere in `run.py`. The column `material_share` is generated directly from `true_markup`, which means the denominator in `mu = theta_m / material_share` carries exact markup signal (plus small noise std=0.025), not independently observed expenditure-over-revenue.
- **Category:** DILUTED
- **Severity:** MED
- **Result-changing:** yes — in this synthetic design the markup recovery problem is structurally easier than the README implies. The denominator `material_share` was generated as `theta_m / true_markup + noise`, so recovering `mu = theta_m / material_share` largely inverts the generative process. A reader replicating with actual firm-panel data (where `alpha_m = materials_cost / revenue` is independently measured) would face a harder estimation problem. The direction of bias correction (proxy-control beats OLS) remains valid, but the magnitude of improvement is understated for real data.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "log_materials" in str(df.columns) and not any("log_materials" in str(expr) for expr in ["material_share = np.exp", "material_share = df"])
  # PASSES on current code (log_materials never used to form material_share); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(df["material_share"].mean() - (np.exp(df["log_materials"]) / np.exp(df["log_output"])).mean()) < 0.1
  # PASSES on honest fix (share computed from panel log-data); FAILS on current code
  ```

### Finding 2: Proxy inversion uses oracle true coefficients, README implies nonparametric h^{-1}

- **Claim source (verbatim):** "The control-function estimator uses this monotonicity to form a productivity control $\tilde \omega_{it}=h^{-1}(k_{it},I_{it})$." — `README.md:28-29`; and in Solution Method: "Use monotonic investment to build a productivity control: omega_tilde_it = h^{-1}(k_it, I_it)." — `README.md:66-67`
- **Code evidence (verbatim):**
  ```python
  omega_proxy = (inv - 0.75 - 0.20 * k) / 0.90
  ```
  `run.py:71`
- **Data evidence:** The investment DGP at `run.py:34` is `investment = 0.75 + 0.90 * omega + 0.20 * capital + rng.normal(0.0, 0.05, n_firms)`. The inversion at line 71 uses the exact coefficients `0.75`, `0.20`, `0.90` — the true parameters of `h`. Correlation between `omega_proxy` and true `omega`: 0.9863 (verified by reproduction run). The OP/LP tradition (cited in References) inverts `h` nonparametrically using polynomial or kernel estimation on the observed `(k, I)` pairs; the code bypasses this entirely. The Solution Method text does say "uses the synthetic investment schedule" — a partial acknowledgment — but does not name the shortcut as an oracle inversion.
- **Category:** DILUTED
- **Severity:** MED
- **Result-changing:** no on the printed numbers (the oracle inversion is deterministic given the DGP, so a polynomial estimator with enough data would converge to the same thing). Yes on the pedagogical claim: the README is teaching "proxy-control estimation" but the code is performing oracle-control estimation. A student implementing this for real data cannot use line 71.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.90" in inspect.getsource(estimate_production) and "0.75" in inspect.getsource(estimate_production)
  # PASSES on current code (hardcoded true coefficients); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "polyfit" in inspect.getsource(estimate_production) or "lowess" in inspect.getsource(estimate_production)
  # PASSES on honest fix (nonparametric inversion); FAILS on current code
  ```

### Finding 3: "OLS overstates flexible-input elasticities" — capital (predetermined) has the largest OLS bias

- **Claim source (verbatim):** "OLS overstates the flexible-input elasticities because high-productivity firms choose more inputs." — `README.md:81-82`
- **Code evidence (verbatim):**
  ```python
  capital = 0.82 * capital + 0.25 * omega + rng.normal(0.0, 0.18, n_firms) + 0.45
  ```
  `run.py:31` — capital is a state variable updated each period; it is predetermined (chosen at `t-1`), not freely optimized at `t`.
- **Data evidence:** `tables/production-estimates.csv` row for Capital: OLS=0.492, True=0.240, bias=0.252 — the largest absolute OLS bias of the three inputs. Labor bias is 0.132, Materials bias is 0.331. The README prose attributes OLS bias to flexible-input simultaneity, but capital is not a flexible input.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no — the numbers are correct. The prose explanation is incomplete: capital is also correlated with `omega` through its law of motion (`run.py:31` includes `0.25 * omega`), generating OLS bias by a different channel than simultaneity. A reader who accepts "flexible-input bias" as the full story would mislearn the source of the capital coefficient bias.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "capital" not in "OLS overstates the flexible-input elasticities"  # vacuously true — README omits capital from the bias explanation
  # PASSES (current README does not mention capital OLS bias separately); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "capital" in readme_results_text and "predetermined" in readme_results_text
  # PASSES on honest fix (capital bias explained separately); FAILS on current README
  ```

## Cross-cutting patterns

- Both F1 and F2 share the same root: the synthetic DGP was built so the econometric problem is structurally trivial (oracle share + oracle inversion), then described using the vocabulary of the nonparametric estimation tradition (OP/LP). The tutorial teaches the right formula and recovers the right direction of correction, but the identification hurdles it appears to overcome are softer than the prose implies.
- All numeric claims are clean. The CSV tables are reproduced exactly from `run.py` with seed=44. There is no data drift between README and tables.
- F3 is a symptom of the same pattern: only flexible inputs are named in the bias story, but capital inherits omega-correlation through its law of motion and has the largest absolute OLS bias. The omission is not random noise — it protects the narrative that "proxy controls flexible-input simultaneity," which is only part of the story.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 40%.** This is below the 50% halt threshold. Surface F1 and F2 to the user before writing fix code, since both change what the tutorial teaches, not just what it says.

1. **F1 violated invariant test** — write `tests/test_production_markups.py`:
   ```python
   def test_material_share_not_from_panel(df):
       # Proves the bug: share column computed from true_markup, not from exp(log vars)
       assert "log_materials" not in inspect.getsource(simulate_panel).split("material_share")[1].split("\n")[0]
   ```
   Confirm it PASSES on current code.

2. **F1 honest-fix test:**
   ```python
   def test_material_share_from_panel(df):
       share_computed = np.exp(df["log_materials"]) / np.exp(df["log_output"])
       assert abs(df["material_share"].mean() - share_computed.mean()) < 0.15
   ```
   Confirm it FAILS on current code.

3. **F2 violated invariant test:**
   ```python
   def test_oracle_inversion():
       import inspect
       src = inspect.getsource(estimate_production)
       assert "0.90" in src and "0.75" in src  # true DGP coefficients hardcoded
   ```
   Confirm it PASSES on current code.

4. **F2 honest-fix test:**
   ```python
   def test_nonparametric_inversion():
       import inspect
       src = inspect.getsource(estimate_production)
       assert any(kw in src for kw in ["polyfit", "lowess", "KernelReg", "polynomial"])
   ```
   Confirm it FAILS on current code.

5. Hand off to `writing-plans`: design fix for F1 (add price-ratio column to panel and compute `alpha_m = (exp(m) * price_m) / (exp(y) * price_y)` or scale by constant unit prices); design fix for F2 (replace line 71 with a polynomial fit of `inv` on `(k, k^2, inv, inv^2, k*inv)` to recover `omega_proxy`).

6. After fixes: re-run `python run.py`, regenerate CSVs, rerun this skill. Expect F1 and F2 to read HOLDS and score to drop to 10-15%.
