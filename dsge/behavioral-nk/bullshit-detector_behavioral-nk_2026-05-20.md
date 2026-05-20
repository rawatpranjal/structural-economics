# bullshit-detector -- behavioral-nk -- 2026-05-20

**Bullshit score: 25%** -- One DILUTED/HIGH finding: the figure plots |output[0]| which masks a sign reversal present in both rational and behavioral output paths at H>=6 and H>=7 respectively; the prose claim "the farther away the wedge is, the more the behavioral response shrinks" is false at H=7-10 for the behavioral model in absolute terms (|output[0]| rises from H=6 to H=10 before shrinking). All algebra, all table numbers, and all code structure otherwise HOLD.

## Header

- Claim sources: `dsge/behavioral-nk/README.md` (full file, 129 lines)
- Code / artifact root: `dsge/behavioral-nk/run.py` (465 lines)
- Data artifacts: `dsge/behavioral-nk/tables/policy-responses.csv` (3 lines)
- Seed audit: none
- Run by: bullshit-detector skill (Claude Sonnet 4.6), 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | IS-curve 2x2 coefficient system (AR(1) case) matches derivation | HOLDS | - | no |
| 2 | Backward recursion 2x2 system (FG case) matches derivation | HOLDS | - | no |
| 3 | All nine table numbers match independent computation | HOLDS | - | no |
| 4 | M=Mf=0.85 for behavioral attention | HOLDS | - | no |
| 5 | psi_i = phi_pi*psi_pi + phi_x*psi_x + 1 | HOLDS | - | no |
| 6 | shock_size=0.01 described as "one-percentage-point" | HOLDS | - | no |
| 7 | "farther away the wedge, more behavioral response shrinks" | DILUTED | HIGH | yes (claim fails H=7-10 in abs terms; figure plots abs()) |
| 8 | Figure shows behavioral attenuation of FG puzzle | DILUTED | MED | no (figure correct; sign reversal hidden by abs()) |

## Findings

### Finding 1: IS-curve 2x2 coefficient system (AR(1) case)

- **Claim source (verbatim):** "Plug the guess into the IS curve and Phillips curve. The Taylor rule then gives this 2 by 2 system: [[1-M*rho_v + sigma*phi_x, sigma*(phi_pi - rho_v)], [-kappa, 1 - beta*M_f*rho_v]] [psi_x, psi_pi]^T = [-sigma, 0]^T." -- `README.md:74-80`
- **Code evidence (verbatim):**
  ```python
  lhs = np.array(
      [
          [1.0 - setting.m * rho + c.sigma * c.phi_x, c.sigma * (c.phi_pi - rho)],
          [-c.kappa, 1.0 - c.beta * setting.m_f * rho],
      ]
  )
  rhs = np.array([-c.sigma, 0.0])
  ```
  `run.py:63-70`
- **Data evidence:** Independent computation with these parameters yields `psi_x=-1.2150, psi_pi=-0.2406, psi_i=0.4872` (rational) matching `tables/policy-responses.csv:2` exactly.
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

### Finding 2: Backward recursion 2x2 system (forward guidance case)

- **Claim source (verbatim):** "Start from x_{H+1}=pi_{H+1}=0. Then step backward: Given x_{t+1} and pi_{t+1}, solve the two date-t equations backward." -- `README.md:84-96`
- **Code evidence (verbatim):**
  ```python
  lhs = np.array(
      [
          [1.0 + c.sigma * c.phi_x, c.sigma * c.phi_pi],
          [-c.kappa, 1.0],
      ]
  )
  rhs = np.array(
      [
          setting.m * next_output + c.sigma * next_inflation - c.sigma * policy_wedge,
          c.beta * setting.m_f * next_inflation,
      ]
  )
  output, inflation = np.linalg.solve(lhs, rhs)
  ```
  `run.py:106-118`
- **Data evidence:** Substituting Taylor rule `i_t = phi_pi*pi_t + phi_x*x_t + v_t` into IS `x_t = M*x_{t+1} - sigma*(i_t - pi_{t+1})` yields `(1+sigma*phi_x)*x_t + sigma*phi_pi*pi_t = M*x_{t+1} + sigma*pi_{t+1} - sigma*v_t`. PC `pi_t = beta*M_f*pi_{t+1} + kappa*x_t` yields `-kappa*x_t + pi_t = beta*M_f*pi_{t+1}`. Both match code exactly.
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

### Finding 3: All nine table numbers match independent computation

- **Claim source (verbatim):** Table "Policy-Wedge Responses" rows for Rational NK and Behavioral NK with values `[-1.215, -0.241, 0.487, -2.43, -0.481, 0.171, -0.166]` and `[-1.146, -0.198, 0.56, -2.292, -0.396, 0.024, -0.058]`. -- `README.md:114-117`
- **Code evidence:** `format_pp` at `run.py:176-177` computes `round(100.0 * value, 3)`. Independent recomputation with stated calibration parameters matches all 14 numeric cells exactly.
- **Data evidence:** `tables/policy-responses.csv:2-3` -- values identical to README table and to independent computation.
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

### Finding 4: "The farther away the wedge is, the more the behavioral response shrinks"

- **Claim source (verbatim):** "Cognitive discounting breaks part of that backward chain. The farther away the wedge is, the more the behavioral response shrinks. This is the main tutorial result." -- `README.md:106-107`
- **Code evidence (verbatim):**
  ```python
  axes2[0].plot(
      horizons,
      100.0 * np.abs(subset["Output"].to_numpy()),
      ...
  )
  ```
  `run.py:365-371`
- **Data evidence:** Independent computation of `|output[0]|` across H=0..20 for behavioral (M=M_f=0.85):
  - H=5: 0.0397, H=6: 0.0055, H=7: 0.0142 (UP), H=8: 0.0241 (UP), H=9: 0.0279 (UP), H=10: 0.0280 (UP), H=11: 0.0260 (DOWN).
  The absolute output response is NOT monotonically shrinking with H. It reaches a local minimum near H=6 (0.0055), then RISES back to 0.0280 at H=10 before eventually declining. The prose claim "the more the behavioral response shrinks" is false for H=7 through H=10. The rational model has the same non-monotone shape (shrinks H=0..5, rises H=6..11, shrinks H=12+). The figure plots `np.abs()` of the output column, which presents this non-monotone hump without flagging it. The signed values in the table show that output[0] turns positive for both models at moderate horizons (rational at H>=6, behavioral at H>=7), a signature of the forward guidance puzzle -- present in both models, not resolved by M=0.85.
- **Category:** DILUTED -- the claim is directionally true only in absolute terms for short horizons (H=0..6 behavioral) and for long horizons (H>=11 behavioral). For H=7..10, |behavioral output[0]| is rising, not shrinking. The claim "the main tutorial result" is stated without qualification and is false over a 4-quarter band in the 20-quarter news horizon shown.
- **Severity:** HIGH -- this is the claimed main result of the tutorial. The figure uses `abs()` which masks the non-monotone shape. A reader following the abs-value figure cannot detect this from the figure alone.
- **Result-changing:** yes -- the claim "farther away = more shrinkage" fails for H=7..10 in absolute terms; the behavioral model's FG puzzle (sign reversal at H=7+) is not suppressed by M=0.85 and is unreported.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert all(abs(behavioral_output_0[h+1]) < abs(behavioral_output_0[h]) for h in range(6, 10))
  # PASSES on current behavior (claim as stated); FAILS because abs rises from H=6 to H=10
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert not all(abs(behavioral_output_0[h+1]) < abs(behavioral_output_0[h]) for h in range(20))
  # PASSES on honest implementation (non-monotone acknowledged); confirms sign-reversal band exists
  ```

### Finding 5: Figure description implies behavioral monotonically attenuates rational at all horizons

- **Claim source (verbatim):** "The forward-guidance figure reports absolute date-0 responses. A future rate wedge matters today only through expectations... Cognitive discounting breaks part of that backward chain. The farther away the wedge is, the more the behavioral response shrinks." -- `README.md:106-107`; figure alt text: "Date-0 response magnitudes to policy wedges announced for future quarters" -- `README.md:108`
- **Code evidence (verbatim):**
  ```python
  100.0 * np.abs(subset["Output"].to_numpy())
  ```
  `run.py:366`; rational |output[0]| at H=5 is 0.0164 (minimum) then rises to 0.2113 at H=11. Figure correctly plots this non-monotone shape but the prose describes it as simple attenuation.
- **Category:** DILUTED -- the figure itself is correct (it plots the computed values). The prose describing the figure mischaracterizes what the figure shows: neither model shows simple monotone attenuation of |output| across all 20 horizons.
- **Severity:** MED -- the figure is not wrong; only the accompanying prose description is wrong. A careful reader examining the figure would see the non-monotone shape for the rational model (rises then falls), but the behavioral model's hump (H=7..10) is subtle in scale (0.028 at peak vs 0.784 at H=0) and might not be visually obvious.
- **Result-changing:** no -- the figure is correct; only the prose narration is inaccurate.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert all(rational_abs_output[h] < rational_abs_output[h-1] for h in range(1, 21))
  # PASSES on claim (monotone shrink); FAILS because rational |output| rises H=6..11
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert rational_abs_output[11] > rational_abs_output[5]  # non-monotone confirmed
  # PASSES on honest description (the forward guidance puzzle is present); FAILS if prose were correct
  ```

## Cross-cutting patterns

- Both DILUTED findings (7, 8) share the same root: the `np.abs()` call at `run.py:366` hides a sign reversal that is itself the forward guidance puzzle the tutorial is implicitly about. The tutorial references Gabaix (2020), whose paper introduces M and M_f precisely to resolve the FG puzzle. With M=0.85, the behavioral model does NOT fully resolve the puzzle (sign reversal still occurs at H>=7). The tutorial never states this, instead presenting the behavioral model as if it simply attenuates the rational response at all horizons.
- The sign reversal (output[0] positive for rate hike at long horizons) appears in the signed table (FG output H=8 = 0.171 for rational, 0.024 for behavioral) without any annotation or explanation. A reader unfamiliar with the FG puzzle literature would not recognize these positive numbers as anomalous.
- The figure `forward-guidance-attenuation.png` is named "attenuation" which is accurate only in the sense that |behavioral| < |rational| at every horizon -- not in the sense that either model shows monotone decay.
- No other numeric claims, algebra, or code structure errors found. The codebase is clean; all bugs are in prose characterization of results, not in computation.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Prose fix only; no code fix needed. Surface to user before any prose rewrite.
1. Violated-invariant test for Finding 7:
   ```python
   # tests/test_behavioral_nk_fg_monotone.py
   # Confirm |behavioral output[0]| is NOT monotone across H=0..20
   assert not all(abs(b_out[h+1]) < abs(b_out[h]) for h in range(20))
   # Should PASS currently (non-monotone exists at H=7..10)
   ```
2. Honest-fix pass condition: add a note in README Results section that the behavioral model still exhibits a sign reversal at H>=7 (a residual FG puzzle), and that the prose "shrinks monotonically" should be qualified to "shrinks at long horizons (H>10)".
3. Prose fix: replace "The farther away the wedge is, the more the behavioral response shrinks" with a qualified version that notes the non-monotone shape and the residual FG puzzle in both models.
4. After fix, re-run `python run.py` to regenerate README.md. Re-run this skill to confirm findings 7 and 8 now read HOLDS and score drops to 0-10%.
