# bullshit-detector — hamiltonian-monte-carlo — recheck — 2026-05-20

**Bullshit score: 0%** — All findings HOLDS. Prior F1 (DILUTED MED: "0.7-0.9 typical") and F2 (DILUTED MED: "ESS sweet spot 0.6-0.8") are fully resolved with qualified prose. Prior F3 (DATA DRIFT LOW: ACF lag claim unverifiable) is resolved: `tables/acf-summary.csv` is now committed and the committed value (hmc_x=10) matches `README.md:233` exactly. No new findings.

## Header

- Claim sources: `computational-methods/hamiltonian-monte-carlo/README.md` (281 lines, read in full)
- Code / artifact root: `computational-methods/hamiltonian-monte-carlo/run.py` (793 lines, read in full)
- Data artifacts: `computational-methods/hamiltonian-monte-carlo/tables/method-comparison.csv` (3 lines), `tables/stepsize-sweep.csv` (8 lines), `tables/acf-summary.csv` (5 lines) — all read in full
- Seed audit: `bullshit-detector_hamiltonian-monte-carlo_recheck_2026-05-20.md` (prior recheck, score 10%)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, second recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | F1 fix: "near 0.99; 0.7-0.9 is high-d asymptotic" | HOLDS | — | — |
| 2 | F2 fix: "0.96 to 0.99; 0.6-0.8 is high-d limit" | HOLDS | — | — |
| 3 | F3 fix: ACF CSV committed; hmc_x=10 matches README | HOLDS | — | — |
| 4 | Leapfrog half-kick/drift/half-kick matches code | HOLDS | — | — |
| 5 | Worked example grad_U=(0,2), r_half=(0.3,0.1), theta_new=(0.03,0.01) | HOLDS | — | — |
| 6 | Acceptance exp(H_t - H_star) matches code | HOLDS | — | — |
| 7 | HMC acceptance=0.991 matches CSV | HOLDS | — | — |
| 8 | RW-MH acceptance=0.671 matches CSV | HOLDS | — | — |
| 9 | ESS_x HMC=726, RW-MH=203 matches CSV | HOLDS | — | — |
| 10 | HMC grad evals=103,974 matches CSV | HOLDS | — | — |
| 11 | All 7 sweep rows match CSV | HOLDS | — | — |
| 12 | Best ESS_x at eps=0.25 (acc=0.986) in claimed 0.96-0.99 range | HOLDS | — | — |
| 13 | Banana params sigma_x=2.0, alpha=0.50, sigma_y=1.0 | HOLDS | — | — |
| 14 | Var(theta_2)=9.00 analytically | HOLDS | — | — |
| 15 | HMC draws 4,000 burn-in 500; RW-MH 40,000 burn-in 2,000 | HOLDS | — | — |
| 16 | Starting point (3.0, 4.0) | HOLDS | — | — |
| 17 | step_size=0.18, n_leapfrog=25 | HOLDS | — | — |
| 18 | ACF lag hmc_x=10 in README backed by CSV | HOLDS | — | — |
| 19 | RW-MH stays correlated beyond 200 lags | HOLDS | — | — |

## Findings

### Finding 1 (prior F1, RESOLVED): Acceptance "near 0.99; 0.7-0.9 is high-d asymptotic"

- **Prior claim (buggy):** "acceptance rates of 0.7 to 0.9 are typical" presented as applying to this run.
- **Current code evidence (verbatim):**
  ```python
  "for this two-dimensional banana posterior at the tuned hyperparameters it sits near 0.99, "
  "while the often-cited 0.7 to 0.9 range is the high-dimensional asymptotic benchmark.\n\n"
  ```
  `run.py:526-527`
- **Current README (verbatim):** "for this two-dimensional banana posterior at the tuned hyperparameters it sits near 0.99, while the often-cited 0.7 to 0.9 range is the high-dimensional asymptotic benchmark." — `README.md:180`
- **Data evidence:** `tables/method-comparison.csv:3`: `Hamiltonian Monte Carlo,3500,0.991,...` — "near 0.99" is accurate for 0.991.
- The 0.7-0.9 range is now explicitly labeled as the high-dimensional asymptotic, not the result of this run. **RESOLVED.**
- **Category:** HOLDS

---

### Finding 2 (prior F2, RESOLVED): ESS sweet spot "0.96 to 0.99; 0.6-0.8 is high-d limit"

- **Prior claim (buggy):** claimed the ESS peak occurred at acceptance around 0.6-0.8.
- **Current code evidence (verbatim):**
  ```python
  "Effective sample size is largest in this sweep at a step size that keeps acceptance around 0.96 to 0.99, "
  "higher than the 0.6 to 0.8 asymptotic-optimal acceptance results in the HMC literature, "
  "because those asymptotics describe the high-dimensional limit and this banana posterior is only two-dimensional."
  ```
  `run.py:748-750`
- **Data evidence:** `tables/stepsize-sweep.csv`: best ESS_x=331 at eps=0.25 with acc=0.986 (in 0.96-0.99 range); best ESS_y=817 at eps=0.30 with acc=0.963 (in 0.96-0.99 range). All sweep rows have acceptance >= 0.944, none in 0.6-0.8 band. **RESOLVED.**
- **Category:** HOLDS

---

### Finding 3 (prior F3, RESOLVED): ACF lag claim backed by committed artifact

- **Prior claim (finding):** "within 10 lags" — `README.md:233` — was dynamically generated with no committed artifact to cross-check against.
- **Current state:** `tables/acf-summary.csv` is now committed (`run.py:706-708` writes it). `acf-summary.csv:2`: `hmc_x,10`. `README.md:233` reads "within 10 lags". The committed value matches the README exactly.
- **Code evidence (verbatim):**
  ```python
  acf_csv = Path(__file__).resolve().parent / "tables" / "acf-summary.csv"
  acf_csv.parent.mkdir(parents=True, exist_ok=True)
  acf_summary.to_csv(acf_csv, index=False)
  ```
  `run.py:706-708`
- The README lag count is generated from `lag_hmc_x = _first_lag_below(acf_hmc_x)` at `run.py:694`, and the same `lag_hmc_x` value is written to `acf-summary.csv` at `run.py:698-703`. Both artifacts come from the same computation in the same run — they cannot disagree unless the CSV is from a different run than the README. The committed CSV (hmc_x=10) and README (10 lags) agree. **RESOLVED.**
- **Category:** HOLDS

---

### Finding 4: Leapfrog pseudocode matches code — HOLDS

- **Claim source (verbatim):** pseudocode half-kick/drift/half-kick at `README.md:187-200`.
- **Code evidence:** `run.py:128-142`:
  ```python
  p = p - 0.5 * step_size * grad_neg_log_target(q)
  grad_calls += 1
  for i in range(n_leapfrog):
      q = q + step_size * p
      if not np.all(np.isfinite(q)) or np.any(np.abs(q) > 1e6):
          diverged = True
          break
      if i < n_leapfrog - 1:
          p = p - step_size * grad_neg_log_target(q)
          grad_calls += 1
          if not np.all(np.isfinite(p)):
              diverged = True
              break
  if not diverged:
      p = p - 0.5 * step_size * grad_neg_log_target(q)
  ```
  Matches pseudocode exactly: initial half-kick, L position steps with full momentum kicks between (not after last), final half-kick. ✓
- **Category:** HOLDS

---

### Finding 5: Worked example arithmetic — HOLDS

- **Claim source (verbatim):** "the residual is ... = 0 - 0.5*(0 - 4) = 2, so grad_U = (0, 2). The half-kick gives r_{t+1/2} = (0.3, 0.2) - 0.05*(0,2) = (0.3, 0.1). The drift gives theta_{t+1} = (0,0) + 0.1*(0.3,0.1) = (0.03, 0.01)." — `README.md:102-104`
- **Verification:** At (0,0): `resid = 0 - 0.5*(0 - 4) = 2`; `dU_dx = 0/4 - 2*0.5*0*2/1 = 0`; `dU_dy = 2/1 = 2`. So `grad_U = (0, 2)`. Half-kick: `(0.3, 0.2) - 0.05*(0, 2) = (0.3, 0.1)`. Drift: `(0,0) + 0.1*(0.3, 0.1) = (0.03, 0.01)`. All exact. ✓
- **Category:** HOLDS

---

### Finding 6: Acceptance formula — HOLDS

- **Claim source (verbatim):** "$$\alpha(\theta_t, r;\, \theta^{\star}, r^{\star}) = \min\Big\lbrace 1,\, \exp\big(H(\theta_t, r) - H(\theta^{\star}, r^{\star})\big) \Big\rbrace$$" — `README.md:122-124`
- **Code evidence:** `run.py:153-158`:
  ```python
  H_current = -log_target(q_current) + 0.5 * float(np.sum(p_current ** 2))
  H_proposed = -log_target(q) + 0.5 * float(np.sum(p ** 2))
  if not np.isfinite(H_proposed):
      draws[t] = q_current
      continue
  if np.log(rng.uniform()) < (H_current - H_proposed):
  ```
  `np.log(uniform()) < H_current - H_proposed` is equivalent to `uniform() < exp(H_current - H_proposed)`, which is `min(1, exp(H_current - H_proposed))`. ✓
- **Category:** HOLDS

---

### Finding 7: All method-comparison table values — HOLDS

- **Claim source:** `README.md:243-246` table.
- **Data evidence:**
  - `method-comparison.csv:2`: `Random-walk MH,38000,0.671,0.097,0.340,203,201,40000.0,` — matches README row exactly (0.34 = 0.340 truncated).
  - `method-comparison.csv:3`: `Hamiltonian Monte Carlo,3500,0.991,0.040,0.082,726,841,,103974.0` — matches README row exactly. ✓
- **Category:** HOLDS

---

### Finding 8: All 7 step-size sweep rows — HOLDS

- **Claim source:** `README.md:254-262` table.
- **Data evidence:** All 7 rows of `stepsize-sweep.csv` match the README table exactly, verified row by row. ✓
- **Category:** HOLDS

---

### Finding 9: Banana parameters — HOLDS

- **Claim source:** `README.md:163`: "(2.0, 0.50, 1.0)".
- **Code evidence:** `run.py:40-42`: `SIGMA_X = 2.0; SIGMA_Y = 1.0; ALPHA = 0.5`. ✓
- **Category:** HOLDS

---

### Finding 10: Var(theta_2) = 9.00 — HOLDS

- **Claim source:** `README.md:165`: "9.00".
- **Code evidence:** `run.py:45`: `VAR_Y_MARGINAL = (ALPHA**2) * 2.0 * (SIGMA_X**4) + SIGMA_Y**2`. With ALPHA=0.5, SIGMA_X=2.0, SIGMA_Y=1.0: `0.25 * 2.0 * 16.0 + 1.0 = 8.0 + 1.0 = 9.0`. Independent computation confirms. ✓
- **Category:** HOLDS

---

### Finding 11: Configuration claims — HOLDS

- **Claim source:** `README.md:167-172`: n_draws_hmc=4000, burn_hmc=500, n_draws_mh=40000, burn_mh=2000, start=(3.0,4.0), step_size=0.18, n_leapfrog=25.
- **Code evidence:** `run.py:226-234`:
  ```python
  n_draws_hmc = 4000
  n_draws_mh = 40_000
  burn_hmc = 500
  burn_mh = 2000
  start = np.array([3.0, 4.0])
  step_size = 0.18
  n_leapfrog = 25
  ```
  All match. ✓
- **Category:** HOLDS

---

## Cross-cutting patterns

- The three prior findings (F1 DILUTED MED, F2 DILUTED MED, F3 DATA DRIFT LOW) are all resolved.
- The ACF artifact fix is robust: `run.py` writes `tables/acf-summary.csv` in the same execution that generates the README, using the same computed `lag_hmc_x` value. The two artifacts cannot disagree unless the CSV is replaced by a file from a different run.
- No new findings in this pass. The internal consistency between the three CSV artifacts and the README is clean across all numeric claims.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings HOLDS. No action required.
1. The three honest-fix tests in `tests/test_hamiltonian-monte-carlo.py` all pass:
   - `test_f1_honest_fix_prose_does_not_claim_0_7_to_0_9_typical`: passes.
   - `test_f2_honest_fix_prose_does_not_claim_sweep_matches_0_6_to_0_8`: passes.
   - `test_f3_honest_fix_acf_csv_exists_and_backs_readme_lag_claim`: passes (CSV exists, hmc_x=10 matches README).
2. No further fixes needed.
