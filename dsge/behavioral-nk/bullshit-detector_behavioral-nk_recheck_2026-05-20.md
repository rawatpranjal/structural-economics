# bullshit-detector -- behavioral-nk -- recheck -- 2026-05-20

**Bullshit score: 0%** -- All findings HOLDS. The prior DILUTED/HIGH finding (monotone-shrink claim) and the prior DILUTED/MED finding ("at every horizon" quantifier) are both resolved. The current prose at README:106 explicitly names the H=0 tie, the H=5 exception, the local minimum at H=6, and the sign reversal from H=7 onward. All 14 table numbers re-verified from code.

## Header
- Claim sources: `dsge/behavioral-nk/README.md` (129 lines, read in full)
- Code / artifact root: `dsge/behavioral-nk/run.py` (481 lines, read in full)
- Data artifacts: `dsge/behavioral-nk/tables/policy-responses.csv` (3 lines, read in full)
- Seed audit: `bullshit-detector_behavioral-nk_recheck_2026-05-20.md` (prior recheck, score 15%)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, second recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | IS-curve 2x2 LHS matrix matches code | HOLDS | - | no |
| 2 | PC row and backward recursion 2x2 LHS match code | HOLDS | - | no |
| 3 | u_t set to zero in both experiments | HOLDS | - | no |
| 4 | M=M_f=1 rational; M=M_f=0.85 behavioral | HOLDS | - | no |
| 5 | psi_i = phi_pi*psi_pi + phi_x*psi_x + 1 | HOLDS | - | no |
| 6 | Backward recursion x_{H+1}=pi_{H+1}=0, steps backward | HOLDS | - | no |
| 7 | sigma=1, beta=0.99, kappa=0.1, phi_pi=1.5, phi_x=0.125, rho_v=0.5 | HOLDS | - | no |
| 8 | shock_size=0.01 labeled one-percentage-point | HOLDS | - | no |
| 9 | "behavioral response smaller than rational at most horizons" | HOLDS | - | no |
| 10 | H=0 tie explicitly named | HOLDS | - | no |
| 11 | H=5 exception explicitly named | HOLDS | - | no |
| 12 | Local minimum near H=6 | HOLDS | - | no |
| 13 | Hump shape, rises before declining | HOLDS | - | no |
| 14 | Sign reversal in behavioral from H=7 onward | HOLDS | - | no |
| 15 | M=0.85 shrinks but does not remove FG puzzle | HOLDS | - | no |
| 16 | All 14 table numbers (Rational NK, Behavioral NK) | HOLDS | - | no |

## Findings

### Finding 1 (prior DILUTED/HIGH, RESOLVED): Monotone-shrink claim

- **Prior claim (buggy):** "The farther away the wedge is, the more the behavioral response shrinks." No longer present.
- **Current README:106 (verbatim):** "The shape is not a simple monotone decay, though. Both models trace a hump: the absolute date-0 response falls to a local minimum near a six-quarter horizon, then rises again before declining at long horizons."
- **Data verification:** Independent recomputation confirms behavioral |output[0]| at H=6 = 0.000055 (local minimum); H=7 = 0.000142, H=8 = 0.000241, H=9 = 0.000279, H=10 = 0.000280 (rising); H=11 = 0.000260 (declining). Prose is accurate. **RESOLVED.**
- **Category:** HOLDS

---

### Finding 2 (prior DILUTED/MED, RESOLVED): "at every horizon" quantifier

- **Prior claim (buggy):** "the behavioral response is smaller than the rational one at every horizon" -- failed at H=0 (tie) and H=5 (|behavioral| > |rational|).
- **Current README:106 (verbatim):** "the behavioral response is smaller than the rational one at most horizons. It is not smaller everywhere. At a zero-quarter horizon the wedge is contemporaneous, there is nothing to discount, and the two models tie exactly. Near the sign-change band the rational and behavioral curves cross sign at different dates, so a short window around a five-quarter horizon has the behavioral absolute response slightly above the rational one at very small magnitudes."
- **Code evidence:** `run.py:398`: `"behavioral response is smaller than the rational one at most horizons. "`. Phrase "at every horizon" is absent from `run.py` and `README.md` (confirmed by grep). **RESOLVED.**
- **Data verification:**
  - H=0: rational = -0.007843, behavioral = -0.007843 (tie, both identical).
  - H=5: |behavioral| = 0.000397 > |rational| = 0.000164 (behavioral larger; sign-reversal phase offset).
  - H=1 through H=4, H=6 through H=20: |behavioral| < |rational|.
  - The prose names both exceptions explicitly and explains the mechanism. **HOLDS.**
- **Category:** HOLDS

---

### Finding 3: 2x2 LHS coefficient system -- HOLDS

- **Claim source (verbatim):**
  ```
  [1-M*rho_v+sigma*phi_x,  sigma*(phi_pi-rho_v)]
  [-kappa,                 1-beta*M_f*rho_v     ]
  ```
  `README.md:73-79`
- **Code evidence:** `run.py:63-68`:
  ```python
  lhs = np.array(
      [
          [1.0 - setting.m * rho + c.sigma * c.phi_x, c.sigma * (c.phi_pi - rho)],
          [-c.kappa, 1.0 - c.beta * setting.m_f * rho],
      ]
  )
  ```
  Exact match. With rational (M=M_f=1): LHS = [[0.625, 1.0], [-0.1, 0.505]]; with behavioral (M=M_f=0.85): [[0.700, 1.0], [-0.1, 0.579]]. ✓
- **Category:** HOLDS

---

### Finding 4: psi_i formula -- HOLDS

- **Claim source (verbatim):** "compute $\psi_i=\phi_\pi\psi_\pi+\phi_x\psi_x+1$" -- `README.md:82`
- **Code evidence:** `run.py:71`: `psi_i = c.phi_pi * psi_pi + c.phi_x * psi_x + 1.0`. Exact match. Verified: rational psi_i = 0.487, behavioral psi_i = 0.560. ✓
- **Category:** HOLDS

---

### Finding 5: Backward recursion -- HOLDS

- **Claim source:** "Start from $x_{H+1}=\pi_{H+1}=0$. Then step backward" -- `README.md:84`
- **Code evidence:** `run.py:128-144`:
  ```python
  output = np.zeros(horizon + 1)
  inflation = np.zeros(horizon + 1)
  ...
  next_output = 0.0
  next_inflation = 0.0
  for t in range(horizon, -1, -1):
      output[t], inflation[t] = solve_one_period(...)
      next_output = output[t]
      next_inflation = inflation[t]
  ```
  Initializes at zero; steps backward from t=horizon to t=0. ✓ The solve_one_period LHS = [[1+sigma*phi_x, sigma*phi_pi], [-kappa, 1]] = [[1.125, 1.5], [-0.1, 1.0]], derived from substituting the Taylor rule into the IS and Phillips curves.
- **Category:** HOLDS

---

### Finding 6: u_t=0 -- HOLDS

- **Claim source (verbatim):** "Here $u_t$ is a cost-push shock, set to zero in both experiments run here." -- `README.md:29`
- **Code evidence:** `run.py:112-116`:
  ```python
  rhs = np.array(
      [
          setting.m * next_output + c.sigma * next_inflation - c.sigma * policy_wedge,
          c.beta * setting.m_f * next_inflation,
      ]
  )
  ```
  No u_t term. RHS[1] = beta*M_f*next_inflation + 0; correct for u_t=0. ✓
- **Category:** HOLDS

---

### Finding 7: Calibration parameters -- HOLDS

- **Claim source:** `README.md:51-64` model setup table.
- **Code evidence:** `run.py:29-37` Calibration dataclass: sigma=1.0, beta=0.99, kappa=0.10, phi_pi=1.50, phi_x=0.125, rho_v=0.50, shock_size=0.01, irf_horizon=32, news_horizon=20. All match exactly. `run.py:183-184`: rational M=M_f=1.0, behavioral M=M_f=0.85. ✓
- **Category:** HOLDS

---

### Finding 8: Local minimum near H=6 -- HOLDS

- **Claim source (verbatim):** "the absolute date-0 response falls to a local minimum near a six-quarter horizon" -- `README.md:106`
- **Data verification:** Behavioral |output[0]|: H=5=0.000397, H=6=0.000055 (minimum), H=7=0.000142 (rising). H=6 is the local minimum. ✓
- **Category:** HOLDS

---

### Finding 9: Sign reversal from H=7 onward -- HOLDS

- **Claim source (verbatim):** "the sign reversal still appears in the behavioral model from a seven-quarter horizon onward" -- `README.md:106`
- **Data verification:** Behavioral output: H=6 = -0.000055 (negative), H=7 = +0.000142 (positive). First positive value at H=7. All subsequent values H=7..20 are positive. "From a seven-quarter horizon onward" is exact. ✓
- **Category:** HOLDS

---

### Finding 10: All 14 table numbers -- HOLDS

- **Claim source:** `README.md:114-117` table and `tables/policy-responses.csv:1-3`.
- **Verification:** Independent recomputation from `solve_policy_coefficients` and `forward_guidance_path`:
  - Rational NK: Output -1.215, Inflation -0.241, Rate 0.487, Cum Output -2.43, Cum Inflation -0.481, FG H=8 output 0.171, FG H=8 inflation -0.166. All match CSV and README.
  - Behavioral NK: Output -1.146, Inflation -0.198, Rate 0.560, Cum Output -2.292, Cum Inflation -0.396, FG H=8 output 0.024, FG H=8 inflation -0.058. All match CSV and README. ✓
- **Category:** HOLDS

---

## Cross-cutting patterns

- The prior DILUTED/HIGH finding (monotone-shrink) and DILUTED/MED finding ("at every horizon") are both resolved. The current prose names both exceptions (H=0 tie, H=5 phase-offset) and correctly describes the hump shape and sign reversal.
- All quantitative claims -- 14 table numbers, calibration values, threshold horizons -- are dynamically generated from code with no hardcoded constants in the prose strings beyond calibration initialization. Drift between code and README is structurally impossible for these numbers.
- No FALSE, no DILUTED, no DATA DRIFT, no UNIMPLEMENTED in this pass.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings HOLDS. No action required.
1. The tests in `tests/test_behavioral_nk.py` confirm the resolved state:
   - `test_readme_does_not_claim_monotone_shrink`: passes ("the more the behavioral response shrinks" absent).
   - `test_readme_drops_every_horizon_quantifier`: passes ("smaller than the rational one at every horizon" absent).
   - `test_behavioral_not_smaller_at_every_horizon`: passes (data confirms H=0 tie and H=5 exception).
   - `test_behavioral_fg_hump_band_exists`: passes (b_out rises for H in range(6, 10)).
2. No further fixes needed.
