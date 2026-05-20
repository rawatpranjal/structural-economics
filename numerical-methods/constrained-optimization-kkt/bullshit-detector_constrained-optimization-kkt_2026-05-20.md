# bullshit-detector — constrained-optimization-kkt — 2026-05-20

**Bullshit score: 25%** — Three findings, all LOW severity; worst is a FALSE prose count ("about a dozen") contradicted by the code (9) and the table (9); no result-changing failure.

## Header
- Claim sources: `numerical-methods/constrained-optimization-kkt/README.md`
- Code / artifact root: `numerical-methods/constrained-optimization-kkt/run.py`
- Data artifacts: `tables/solution_comparison.csv`, `tables/kkt_check.csv`, `tables/shadow_prices.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "about a dozen barrier values" needed for machine-precision feasibility | FALSE | LOW | no (table correctly says 9; prose is the only artifact affected) |
| 2 | "complementarity bounded above by n*t" — reported table value is not n*t | DATA DRIFT | LOW | no (figure uses exact formula; table uses heuristic recovery; both defensible) |
| 3 | SLSQP pseudocode "recover mu by complementary slackness on inactive bounds" | DILUTED | LOW | no (code is correct; pseudocode omits the stationarity-based recovery for binding bounds) |
| 4 | All numeric table values (allocations, utilities, KKT residuals, shadow prices) | HOLDS | — | — |
| 5 | Closed-form optimum x*=(2,1,0), lambda*=2, mu*=(0,0,1.5), u*=8.5 | HOLDS | — | — |
| 6 | Baseline gives lambda=1.5, x=(2.5,1.5,-1), utility=9.25 | HOLDS | — | — |
| 7 | Barrier distance at t=1e-8 is 8.90e-09 from closed form | HOLDS | — | — |
| 8 | Projected gradient converges in 95 iterations | HOLDS | — | — |
| 9 | Duality gap along central path is exactly n*t (theoretical claim) | HOLDS | — | — |

## Findings

### Finding 1: "about a dozen barrier values" contradicted by code and table

- **Claim source (verbatim):** "Reaching machine-precision feasibility takes about a dozen barrier values." — `README.md:257`
- **Code evidence (verbatim):**
  ```python
  barriers = [10.0, 1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
  ```
  `run.py:137`
- **Data evidence:** `tables/solution_comparison.csv` row for Method 2: `"9 barrier values"` in Iterations column. CSV row 3 verbatim: `"Method 2: Interior-point log barrier,2.0000,1.0000,0.0000,3.0000,8.5000,9 barrier values,yes"`
- **Category:** FALSE
- **Severity:** LOW
- **Result-changing:** no. The table correctly states 9. The figure and all numeric outputs are unaffected. Only the prose narrative is wrong.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert len([10.0, 1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]) == 12  # PASSES on buggy prose; 9 != 12 so this FAILS -- invert: assert equals 9
  ```
  ```python
  assert len([10.0, 1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]) != 9  # FAILS on current code (9 == 9), proving claim is false
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "9 barrier values" in open("README.md").read() and "dozen" not in open("README.md").read()
  ```

### Finding 2: complementarity reported in KKT table is not n*t — figure and table use different multiplier formulas

- **Claim source (verbatim):** "Complementarity is bounded above by n*t along the central path." — `README.md:256`; and implicitly the KKT table presents barrier complementarity as the diagnostic value consistent with this bound.
- **Code evidence (verbatim):**
  ```python
  # Figure trace: exact barrier multipliers
  mu_t = t / x_t
  barrier_kkt_trace.append(kkt_residuals(x_t, lam_t, mu_t))
  ```
  `run.py:147-148`
  ```python
  # Table: heuristic recovery ignores barrier multiplier
  barrier_lam, barrier_mu = recover_multipliers(barrier_x_final)
  barrier_stat, barrier_primal, barrier_dual, barrier_compl = kkt_residuals(barrier_x_final)
  ```
  `run.py:152-153`
- **Data evidence:** `tables/kkt_check.csv` row for Method 2: `"Method 2: Interior-point log barrier,...,1.00e-08"` (complementarity). Exact barrier n*t at t=1e-8 = 3e-8. Ratio is 3.0. Figure uses n*t-exact complementarity; table uses heuristic recovery giving ~1e-8. The two differ by factor 3.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no. Both values are consistent with the bound (1e-8 < 3e-8). No published table number is wrong on its own terms. The discrepancy is an internal inconsistency between figure and table, not an error in either artifact individually.
- **Violated invariant (one-line pytest assertion):**
  ```python
  import numpy as np; from scipy.optimize import brentq; a=np.array([4.,3.,.5]); t=1e-8; lam=brentq(lambda l:(0.5*((a-l)+np.sqrt((a-l)**2+4*t))).sum()-3.,-100,100); x=0.5*((a-lam)+np.sqrt((a-lam)**2+4*t)); assert abs(float(np.sum((t/x)*np.abs(x))) - 1e-8) < 1e-15  # FAILS: n*t=3e-8, not 1e-8
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(float(np.sum((t/x)*np.abs(x))) - 3*t) < 1e-12  # exact n*t = 3*1e-8
  ```

### Finding 3: SLSQP pseudocode omits stationarity-based recovery of binding-bound multipliers

- **Claim source (verbatim):** "recover mu by complementary slackness on inactive bounds" — `README.md:240` (Solution Method pseudocode for SLSQP)
- **Code evidence (verbatim):**
  ```python
  def recover_multipliers(x):
      active = x > eps_active
      grad_neg = a - B @ x
      if active.any():
          lam = float(np.mean(grad_neg[active]))
      else:
          lam = float(np.max(a))
      mu = np.zeros_like(x)
      for j in range(len(x)):
          if not active[j]:
              mu[j] = max(0.0, lam - float(grad_neg[j]))
      return lam, mu
  ```
  `run.py:70-81`
- **Data evidence:** `tables/shadow_prices.csv` row for Project 3: `"Project 3 bound $x_3 \geq 0$,1.50,binding,..."`. This mu_3=1.50 is computed via stationarity: `mu[2] = max(0, lam - (a[2] - x[2])) = max(0, 2.0 - 0.5) = 1.5`. The pseudocode phrase "complementary slackness on inactive bounds" describes only the trivially-zero case (mu_j=0 for non-binding j), not this computation.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no. The code correctly computes mu_3=1.5. Only the pseudocode description is incomplete. The Equations section (README.md:133) correctly states: "averaging a_j - (Bx)_j over the active set and solving for the bound multipliers on the inactive set."
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "complementary slackness" not in open("README.md").read().split("recover mu by")[1].split("\n")[0]  # FAILS on current text
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "stationarity" in open("README.md").read().split("recover mu")[1].split("\n")[0]  # pseudocode must name stationarity
  ```

## Cross-cutting patterns

- All three findings involve the narrative prose (Results, Takeaway, pseudocode) being inconsistent with the code or data; the actual computations and tables are correct throughout. No numerical finding is wrong.
- The "dozen" mismatch and the figure/table complementarity discrepancy both stem from the same habit: the prose was written to convey a general principle (n*t bound, ~O(10) steps) and the specific realized values were never back-checked against the generated artifacts.
- The SLSQP pseudocode is the only place where the prose is less accurate than the Equations section; the Equations section (README.md:133) correctly describes the recovery. The discrepancy is purely within the Solution Method subsection.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 25%.** All findings LOW severity; no halt required. Proceed to fix.
1. **Finding 1 fix:** In `run.py` line where Results text is assembled for the barrier KKT paragraph, change "about a dozen barrier values" to "9 barrier values" (matching the table). Re-run `python run.py` to regenerate README.md. Confirm `"a dozen"` no longer appears in README.md.
2. **Finding 2 fix (optional):** Add a prose note distinguishing figure complementarity (uses exact n*t-based multipliers mu_t=t/x_t) from table complementarity (uses heuristic recover_multipliers). Alternatively, make the table also use the exact barrier multipliers for the barrier row. Neither is required for correctness.
3. **Finding 3 fix:** In the SLSQP pseudocode string at `run.py:470`, change "recover mu by complementary slackness on inactive bounds" to "recover mu for binding bounds from stationarity: mu_j = lam - (a_j - (Bx)_j)". Re-run to regenerate README.md.
4. After fixes, re-run `python scripts/validate_catalog.py` to confirm no math rendering regressions.
5. Re-run this skill on the updated tutorial. Expected new score: 0-10%.
