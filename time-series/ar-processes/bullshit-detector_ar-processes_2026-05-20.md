# bullshit-detector — ar-processes — 2026-05-20

**Bullshit score: 20%** — Two DILUTED/DATA DRIFT findings; code and math are correct; no false results; pseudocode uses wrong shock symbol and prose rounds 6.6 to "seven."

## Header
- Claim sources: `time-series/ar-processes/README.md`
- Code / artifact root: `time-series/ar-processes/run.py`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Pseudocode step 3 uses `eps_t` for g_t update (same symbol as AR1 shock) | DILUTED | MED | no |
| 2 | Results prose says half-life goes from "one to seven periods"; table says 6.6 | DATA DRIFT | LOW | no |
| 3 | AR(1) closed-form variance, half-life, ACF | HOLDS | - | - |
| 4 | Multiplier-accelerator recursion y_t = beta(1+alpha)y_{t-1} - alpha*beta*y_{t-2} + g_t | HOLDS | - | - |
| 5 | Steady-state values y_bar=5.00, c_bar=4.00 | HOLDS | - | - |
| 6 | Income roots 0.346, 0.694; largest modulus 0.694 | HOLDS | - | - |
| 7 | Spectral density formula sigma^2 / (2*pi*|1 - rho*exp(-i*omega)|^2) | HOLDS | - | - |
| 8 | Table values for variance, half-life at rho=0.5, 0.9, 0.99 | HOLDS | - | - |
| 9 | AR1 and MA simulations use independent shocks | HOLDS | - | - |

## Findings

### Finding 1: Pseudocode step 3 uses `eps_t` for both AR(1) state and government spending update

- **Claim source (verbatim):** "3. Update x_t = rho x_{t-1} + eps_t and g_t = rho_g g_{t-1} + eps_t." — `README.md:87`
- **Claim source context:** The Equations section explicitly states "The spending innovation $\eta_t \sim N(0,\sigma^2)$ is drawn independently of $\varepsilon_t$." — `README.md:31`
- **Code evidence (verbatim):**
  ```python
  def simulate_ar1(rho, sigma, periods, seed=42, burn_in=200):
      rng = np.random.default_rng(seed)
      ...
      shocks = rng.normal(0.0, sigma, total_periods)

  def simulate_multiplier_accelerator(alpha, beta, rho_g, sigma, periods, seed=43, burn_in=200):
      rng = np.random.default_rng(seed)
      ...
      shocks = rng.normal(0.0, sigma, total_periods)
  ```
  `run.py:17-32` and `run.py:36-48`
- **Data evidence:** Not applicable (pseudocode notation error, not a numeric claim).
- **Category:** DILUTED — the code correctly uses independent shocks (different RNG seeds); the pseudocode conflates them under the same symbol `eps_t`, erasing the independence claim made two lines above it in Equations.
- **Severity:** MED — a reader following the pseudocode literally would implement a correlated-shock model, producing wrong cross-correlations between x_t and g_t. The code itself is correct.
- **Result-changing:** no — the code (not the pseudocode) drives the actual simulation.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "eps_t" in open("time-series/ar-processes/README.md").read().split("3. Update")[1].split("\n")[0] and "g_t" in open("time-series/ar-processes/README.md").read().split("3. Update")[1].split("\n")[0]
  # PASSES on current buggy pseudocode (eps_t used for g_t update); FAILS after fix (eta_t replaces eps_t)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "eta_t" in open("time-series/ar-processes/README.md").read().split("3. Update")[1].split("\n")[0]
  # PASSES after fix (eta_t used in pseudocode for g_t); FAILS on current pseudocode
  ```

---

### Finding 2: Results prose says half-life "from one to seven periods"; table and solution method say 6.6

- **Claim source (verbatim):** "Raising $\rho$ from 0.5 to 0.9 lengthens the half-life from one to seven periods." — `README.md:94`
- **Contradicting claim source (verbatim):** "The AR(1) half-life is $\log(0.5)/\log(\rho)=6.6$ periods." — `README.md:78`; table row "Half-life (periods) | 1.0 | 6.6 | 69.0" — `README.md:122`; Takeaway "a shock has half of its initial effect after 6.6 periods." — `README.md:128`
- **Code evidence (verbatim):**
  ```python
  ar1_half_life = np.log(0.5) / np.log(rho_ar1)
  ```
  `run.py:169` — with `rho_ar1 = 0.9` this evaluates to 6.5788, formatted as 6.6 everywhere except the Results paragraph.
- **Data evidence:** `tables/ar-properties.csv` row: `Half-life (periods),1.0,6.6,69.0` — confirms 6.6, not 7.
- **Category:** DATA DRIFT — three sources (Solution Method prose, table, Takeaway) all say 6.6; Results paragraph says "seven." The code emits 6.6 consistently.
- **Severity:** LOW — 6.5788 rounds to 7 in integer terms; the error is purely presentational and does not change any figure or result.
- **Result-changing:** no — figure labels and table values are correct; only one prose sentence uses the rounded integer.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "seven" in open("time-series/ar-processes/README.md").read()
  # PASSES on current file; FAILS after fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "seven" not in open("time-series/ar-processes/README.md").read() and "6.6" in open("time-series/ar-processes/README.md").read()
  # PASSES after fix (seven removed, 6.6 used consistently); FAILS on current file
  ```

---

## HOLDS (documented for coverage)

- **AR(1) closed-form variance:** `sigma^2/(1-rho^2)` — `README.md:76`. Code: `ar1_variance = sigma_ar1**2 / (1.0 - rho_ar1**2)` `run.py:167`. Table values 0.000133/0.000526/0.005025 verified analytically. HOLDS.
- **Multiplier-accelerator recursion:** `README.md:44` states `y_t = beta(1+alpha)y_{t-1} - alpha*beta*y_{t-2} + g_t`. Algebraic expansion of `run.py:90-95` confirms identity. HOLDS.
- **Steady states y_bar=5.00, c_bar=4.00:** `README.md:67-68`. `y_bar = 1.0/(1-0.8) = 5.0`, `c_bar = 0.8*5.0 = 4.0`. Code: `run.py:171-172`. HOLDS.
- **Income roots 0.346, 0.694; largest modulus 0.694:** `README.md:78`. `np.roots([1.0, -0.8*1.3, 0.3*0.8])` = [0.69436, 0.34564]. HOLDS.
- **Spectral density formula:** `README.md:76` (implied). `run.py:117-118` implements `sigma**2 / (2.0 * np.pi * |1 - rho*exp(-i*omega)|**2)`. Standard one-sided AR(1) spectral density. HOLDS.
- **Table values at rho=0.5, 0.9, 0.99:** `tables/ar-properties.csv`. All three columns verified analytically. HOLDS.
- **Independent shocks in code:** `simulate_ar1` seed=42, `simulate_multiplier_accelerator` seed=43. Shocks are independent. HOLDS (the code is right; only the pseudocode symbol is wrong — see Finding 1).

## Cross-cutting patterns

- The single diseased pattern is **pseudocode faithfulness**: the pseudocode in Solution Method uses the wrong shock symbol (`eps_t` instead of `eta_t` for the government spending update), contradicting the Equations prose two paragraphs above it. No other algorithmic faithfulness gap was found.
- The "seven vs 6.6" drift is a one-off rounding inconsistency; no other numeric inconsistency between prose and table was found.
- The mathematical core (recursions, closed-form objects, root calculation, spectral density) is fully faithful to the code.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20% (< 50%).** Safe to proceed with minor fixes without escalation.

1. **Finding 1 fix:** In `README.md` line 87, replace `g_t = rho_g g_{t-1} + eps_t` with `g_t = rho_g g_{t-1} + eta_t` (or equivalently, clarify that the pseudocode uses a single generic shock symbol but the implementation draws independently). The fix is documentation-only; no code change needed.

2. **Finding 2 fix:** In `README.md` line 94 (and correspondingly in `run.py` line 293-295 figure description), replace "seven periods" with "6.6 periods" to match the table, Solution Method prose, and Takeaway.

3. After fixes, re-run `python scripts/validate_catalog.py` to confirm no math rendering regressions.

4. Re-run this skill to confirm both findings now read HOLDS and score drops to 0-10%.
