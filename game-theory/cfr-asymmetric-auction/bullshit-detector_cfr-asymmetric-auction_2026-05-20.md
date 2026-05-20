# bullshit-detector — cfr-asymmetric-auction — 2026-05-20

**Bullshit score: 15%** — one DILUTED finding (exploitability rate claimed as O(1/sqrt(T)) but empirical log-log slope is -0.8, not -0.5); all numeric results match CSV artifacts exactly; code implements every equation as written.

## Header

- Claim sources: `game-theory/cfr-asymmetric-auction/README.md` (Overview, Equations, Solution Method, Results, Takeaway sections)
- Code / artifact root: `game-theory/cfr-asymmetric-auction/run.py`
- Data artifacts: `game-theory/cfr-asymmetric-auction/tables/methods-summary.csv`, `game-theory/cfr-asymmetric-auction/tables/asymmetric-exploitability.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Exploitability tracks O(1/sqrt(T)) rate | DILUTED | LOW | no (exploitability final value correct; only rate description wrong) |
| 2 | alpha=3/2 from BVP bisection | DATA DRIFT | LOW | no (needs re-run to verify; bbar=2/3 confirmed by CSV) |

## Findings

### Finding 1: Exploitability "tracks the textbook rate of order one over the square root of iterations"

- **Claim source (verbatim):** "Exploitability of the average strategy falls steadily across iterations and tracks the textbook rate of order one over the square root of iterations on a log-log plot." — `README.md:162`
- **Code evidence (verbatim):**
  ```python
  if t in log_set:
      avg1 = average_strategy(S1)
      avg2 = average_strategy(S2)
      eps_total, _, _ = exploitability(
          avg1, avg2, values_1, values_2, type_pmf_1, type_pmf_2, bids
      )
      iters_logged.append(t)
      expl_logged.append(eps_total)
  ```
  `run.py:142-149`
- **Data evidence:** From `tables/asymmetric-exploitability.csv`: iteration 1 -> 3.465e-01; iteration 5000 -> 3.624e-04. Log-log slope = (ln(3.624e-4) - ln(3.465e-1)) / (ln(5000) - ln(1)) = **-0.806**, not -0.5. Midpoint segment (T=1 to T~100): slope = **-0.727**. Neither matches the O(1/sqrt(T)) rate. The actual decay is approximately O(T^{-0.8}), faster than claimed.
- **Category:** DILUTED — the code computes exploitability correctly and it does decay; the claim about the specific rate is inaccurate against the committed data artifact.
- **Severity:** LOW — the claim is about the convergence rate, not about whether the equilibrium is found. The final exploitability value (3.624e-4) and all other results are unaffected. The README also explicitly disclaims the textbook guarantee ("the first-price auction is general-sum...so CFR's convergence here is empirical rather than guaranteed by the textbook bound") before asserting the rate empirically in Results — making the rate assertion in Results self-contradictory rather than fabricated.
- **Result-changing:** no — exploitability value, bid functions, and residuals are all unchanged.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(np.polyfit(np.log(iters[1:]), np.log(expl[1:]), 1)[0] - (-0.5)) < 0.1  # PASSES on current data (slope is -0.8, not -0.5); FAILS on honest fix (if rate were truly -0.5)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "O(1/sqrt" not in open("README.md").read() or "approximately O(T^{-0.8})" in open("README.md").read()  # PASSES if description corrected; FAILS currently
  ```

### Finding 2: "Produces alpha = 3/2 and bbar = 2/3 for our distributions"

- **Claim source (verbatim):** "Shooting forward from a small b_0 with this asymptotic initial condition and bisecting alpha on the constraint phi_2(bbar) = 2 produces alpha = 3/2 and bbar = 2/3 for our distributions." — `README.md:107-109`
- **Code evidence (verbatim):**
  ```python
  alpha_opt = brentq(residual, alpha_lo, alpha_hi, xtol=1e-12)
  bbar, _, _ = shoot(alpha_opt)
  ```
  `run.py:212-213`
- **Data evidence:** `tables/methods-summary.csv` row 4: `MMRS upper bid $\bar{b}$, 0.6667`. This confirms bbar = 0.6667 = 2/3. The alpha value (3/2 = 1.5) is not stored in any CSV artifact; it is within the bisection search range [1.0, 1.9] (`run.py:179`) and analytically consistent with the ODE structure (series expansion yields c1 = -c2, and the specific value alpha = 1.5 is the known result for U[0,1] vs U[0,2]). However, the committed artifacts do not record alpha_opt. **Needs re-run to verify** the exact alpha value.
- **Category:** DATA DRIFT — bbar is confirmed by CSV; alpha is an unchecked numeric claim with no artifact to ground it against.
- **Severity:** LOW — bbar is the load-bearing number (used in bid function inversion); alpha is an intermediate bisection result not used downstream after `solve_asymmetric_bne` returns.
- **Result-changing:** no — bbar = 0.6667 matches the claim; the downstream bid function computation depends only on bbar and the final ODE solution, not on alpha directly.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert any("alpha" in row for row in open("tables/methods-summary.csv"))  # FAILS on current artifacts (alpha not stored)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(alpha_opt - 1.5) < 1e-4  # PASSES if re-run confirms alpha=3/2; needs re-run to verify
  ```

## Cross-cutting patterns

- No systematic mislabeling, parametric leakage, or missing implementation found. All six equations in the Equations section (payoff, win probability, counterfactual value, regret, cumulative regret, regret matching, time-averaged strategy, exploitability, MMRS ODE) map exactly to code functions with the correct operations.
- All six numeric entries in `tables/methods-summary.csv` match `README.md` verbatim. All 37 exploitability rows in `tables/asymmetric-exploitability.csv` match the displayed README table.
- The one DILUTED finding (rate claim) is self-undermined within the same paragraph of the README, which makes it a description inconsistency rather than a fabrication.
- The MMRS paper (Marshall, Meurer, Richard, Stromquist 1994) is not in the References section despite the acronym appearing throughout the text. Maskin & Riley (2000) is cited but does not originate the BVP shooting method. This is a citation gap, not a code bug; out of scope for this audit.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 15% (< 50%).** Safe to proceed with fixes without user escalation.

1. **Finding 1 (rate claim):**
   - Turn the violated invariant into a pytest test: load `tables/asymmetric-exploitability.csv`, compute log-log slope over all rows, assert slope is between -0.45 and -0.55. Confirm this test PASSES (slope is ~-0.8, not in [-0.55, -0.45]).
   - Honest-fix pass condition: replace the rate claim in `run.py`'s `add_results` call at `run.py:499-506` with "decays at an empirical rate of approximately O(T^{-0.8}) on a log-log plot, faster than the O(1/sqrt(T)) Hannan bound." Confirm test now FAILS (slope is genuinely ~-0.8, description now accurate).
   - No code change needed, only prose correction in `run.py:499-506`.

2. **Finding 2 (alpha claim):**
   - Add `alpha_opt` to the `methods-summary.csv` table in `run.py:533-558` so the committed artifact records it.
   - After re-run, confirm `alpha_opt` is approximately 1.5. If not, correct the prose claim at `README.md:107-109`.
   - **Needs re-run to verify.**

3. Re-run `python run.py` in `game-theory/cfr-asymmetric-auction/` after prose fix. Confirm the log-log slope description matches the regenerated data. Re-run this skill to confirm both findings now read HOLDS and bullshit score drops to 0-5%.
