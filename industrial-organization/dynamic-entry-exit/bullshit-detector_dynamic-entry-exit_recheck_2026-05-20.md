# bullshit-detector — dynamic-entry-exit — recheck — 2026-05-20

**Bullshit score: 5%** — all three original findings now HOLDS; one residual low-severity note on the entry-count approximation inside VFI (not result-changing); no FALSE or DILUTED findings remain.

## Header
- Claim sources: `industrial-organization/dynamic-entry-exit/README.md`
- Code / artifact root: `industrial-organization/dynamic-entry-exit/run.py`
- Data artifacts: `industrial-organization/dynamic-entry-exit/tables/equilibrium-statistics.csv`, `tables/value-by-N.csv`
- Seed audit (if any): `bullshit-detector_dynamic-entry-exit_2026-05-20.md` (original 60% audit)
- Run by: bullshit-detector (claude-sonnet-4-6), independent recheck
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Delta(N) drives both V(N) and p_exit via same Binomial-integrated E[V(N')\|stay] | HOLDS | - | - |
| 2 | VFI converged in 762 iterations with sup-norm error 9.94e-09 (undampened) | HOLDS | - | - |
| 3 | Pseudocode integrates over N-1 rivals, Binomial(s; N-1, 1-p_exit) | HOLDS | - | - |
| 4 | All numeric table values match live recomputation | HOLDS | - | - |
| 5 | Entry count inside VFI uses round(N * p_stay_others) where p_stay_others comes from previous sweep | DATA DRIFT (self) | LOW | no — converges to same fixed point |

## Findings

### Finding 1 (original Finding 1 — now HOLDS): `_exit_prob` uses Binomial-integrated E[V(N')|stay]

- **Claim source (verbatim):**
  > "If it stays, its deterministic surplus is $\Delta(N)=\pi(N)-f+\beta \mathbb{E}\left[V(N_{t+1})\mid N_t=N,\text{ stay}\right]$. ... The incumbent exit probability is $p_{\mathrm{exit}}(N)=\frac{1}{1+\exp\{\Delta(N)/\sigma_\varepsilon\}}$."
  >
  > — `README.md:25`, `README.md:39-41`

- **Code evidence (verbatim):**
  ```python
  def _exit_prob(N, profits, f, beta, V, sigma_eps, K, N_max, p_exit_rivals):
      ...
      EV = _continuation_value(N, p_stay_others, n_enter, V, N_max)
      delta = pi_N - f + beta * EV
      p_exit = 1.0 / (1.0 + np.exp(delta / sigma_eps))
      return p_exit, n_enter, EV
  ```
  `run.py:147-177`

  ```python
  def _continuation_value(N, p_stay_others, n_enter, V, N_max):
      EV = 0.0
      for s in range(N):  # s = survivors among the other N-1 rivals
          prob_s = binom.pmf(s, N - 1, p_stay_others) if N > 1 else (1.0 if s == 0 else 0.0)
          ...
          EV += prob_s * V[N_next - 1]
      return EV
  ```
  `run.py:125-144`

- **Verification:** `_continuation_value` uses `binom.pmf` over Binomial(N-1, p_stay_others). `_exit_prob` calls `_continuation_value` and computes `delta = pi_N - f + beta * EV` — the exact same `delta` that drives the log-sum value update in `solve_model` (`u_stay = pi_N - f + beta * EV` at `run.py:94`). Identity check: `exp(V/sigma) - 1 == 1/p_exit - 1` holds for all N where `exit_prob > 1e-6` with max relative error 6.96e-09. Self-loop proxy `V[N - 1]` is absent from `_exit_prob`. Self-consistency check at convergence: all 30 states pass to 1e-6.

- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** —

---

### Finding 2 (original Finding 2 — now HOLDS): convergence error is undampened sup-norm

- **Claim source (verbatim):**
  > "The value iteration converged in **762 iterations** with sup-norm error **9.94e-09**."
  >
  > — `README.md:109`

  > "until max_N |V_{n+1}(N)-V_n(N)| < epsilon"
  >
  > — `README.md:101`

- **Code evidence (verbatim):**
  ```python
  V_update = dampen * V_new + (1.0 - dampen) * V
  error = np.max(np.abs(V_new - V))
  ...
  V = V_update
  if error < tol:
      ...break
  ```
  `run.py:107-119`

- **Data evidence:** Live recomputation returns `iterations=762`, `error=9.9408e-09`. Both match README exactly. The error 9.94e-09 is below `tol=1e-8`, confirming it is the undampened residual (if dampened at 0.3, the undampened residual would be 3.31e-08, which exceeds `tol` — the old bug). The fix correctly reports the undampened `V_new - V`.

- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** —

---

### Finding 3 (original Finding 3 — now HOLDS): pseudocode explicitly states N-1 rivals

- **Claim source (verbatim):**
  > "for s = 0,...,N-1 surviving rivals (the focal firm always stays):"
  > "weight = Binomial(s; N-1 rivals, 1 - p_exit(N))"
  >
  > — `README.md:96-97`

- **Code evidence (verbatim):**
  ```python
  "        for s = 0,...,N-1 surviving rivals (the focal firm always stays):\n"
  "            weight = Binomial(s; N-1 rivals, 1 - p_exit(N))\n"
  ```
  `run.py:456-457` (inside the pseudocode string in `main()`)

  The VFI loop itself uses `binom.pmf(s, N - 1, p_stay_others)` at `run.py:138`, consistent. The transition matrix correctly uses `binom.pmf(s, N, p_stay)` at `run.py:213` (unconditional, all N firms), and the pseudocode at `README.md:102-105` explicitly flags the distinction.

- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** —

---

### Finding 4: all numeric table values match live recomputation

- **Claim source:** `README.md:133-162`, `tables/equilibrium-statistics.csv`, `tables/value-by-N.csv`

- **Data evidence (live recomputation vs CSV/README):**

  All 10 rows of `value-by-N.csv` match live recomputation to 3 decimal places:
  - `V(1)=21.133`, `V(10)=3.889`, `V(30)=1.428` — all exact.
  - `exit_prob(3)=0.0001`, `exit_prob(10)=0.0205` — all exact.
  - `entry_count(7)=1.00`, `entry_count(10)=0.00` — all exact.

  `equilibrium-statistics.csv`:
  - `E[N]=7.98` (live: 7.9789) — correct to stated precision.
  - `Std[N]=0.15` (live: 0.1452) — correct to stated precision.
  - `Mode=8` — correct.
  - `exit probability=0.0026` (live: 0.0026) — correct.
  - `VFI iterations=762` — correct.
  - `max_dist_gap=4.78e-04` (live: 4.7843e-04) — correct.
  - `zero-profit N=10.3` (analytic: 10.31) — correctly rounded.

- **Category:** HOLDS
- **Severity:** —
- **Result-changing:** —

---

### Finding 5 (new, low severity): entry-count inside VFI uses lagged exit_prob for expected_surv

- **Claim source (verbatim):**
  > "Potential entrants use the expected survivor count $\bar S(N_t)=\mathrm{round}\{N_t[1-p_{\mathrm{exit}}(N_t)]\}$"
  >
  > — `README.md:45`

- **Code evidence (verbatim):**
  ```python
  p_stay_others = 1.0 - p_exit_rivals  # p_exit_rivals = previous sweep's exit_prob[i]
  expected_surv = max(0, int(np.round(N * p_stay_others)))
  n_enter = _free_entry_count(expected_surv, ...)
  ```
  `run.py:169-173`

  During VFI sweeps, `expected_surv` uses the **previous sweep's** `exit_prob[i]` (passed as `p_exit_rivals`), not the current sweep's `p_exit`. The README presents `S_bar` as a function of the current `p_exit(N_t)`. This creates a one-lag discrepancy between the stated formula and the within-sweep computation. At convergence, `p_exit_rivals` equals the converged `exit_prob[i]`, so the lag vanishes and the converged fixed point is consistent. The self-consistency check (all 30 states, re-running `_exit_prob` with converged `exit_prob`) confirms zero discrepancy at convergence.

- **Data evidence:** Not result-changing. Live recomputation confirms all table values match what the converged code produces. The lag is a within-sweep approximation that disappears at the fixed point.

- **Category:** DATA DRIFT (between within-VFI formula and README description)
- **Severity:** LOW
- **Result-changing:** no — fixed point is self-consistent; all reported numbers verified against live recomputation.

- **Violated invariant (one-line pytest assertion):**
  ```python
  import inspect; src = inspect.getsource(run._exit_prob); assert "p_exit_rivals" in src and "1.0 - p_exit_rivals" in src
  ```
  (Passes on current code; a hypothetical fix using the current-sweep p_exit in a nested loop would remove `p_exit_rivals`.)

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # If fully consistent within each sweep (nested fixed point), p_exit_rivals would be removed:
  assert "p_exit_rivals" not in inspect.getsource(run._exit_prob)
  ```
  (Not required — the current approach is a standard outer-loop joint convergence that is theoretically sound. This finding is informational only.)

---

## Cross-cutting patterns

- All three original DILUTED findings from the 60% audit are now HOLDS. The fix is structurally correct and numerically verified end-to-end.
- The one new finding (Finding 5) is a documentation note about within-VFI lagging, not a faithfulness failure. The converged output is fully consistent with the stated equations.
- The convergence diagnostic change (dampened → undampened) causes the reported iteration count to increase from 667 (original buggy run) to 762. The README now reports 762. This is consistent: the undampened residual takes more iterations to drop below `tol=1e-8` than the dampened step.
- No aspirational pseudocode remains. The pseudocode at `README.md:85-107` now faithfully matches the implemented algorithm, including the explicit distinction between N-1 rivals (VFI step) and N firms (transition matrix).

## TDD execution sequence (for the next agent)

0. **Score is 5%.** All original findings resolved. No action required before shipping.
1. The three violated-invariant tests in `tests/test_run.py` now correctly FAIL (the bugs are gone).
2. The three honest-fix tests now correctly PASS.
3. Finding 5 (lagged `p_exit_rivals` in entry count) is informational. No test is required; the fixed point is self-consistent.
4. Re-run `python run.py` to regenerate outputs and confirm the README, figures, and tables regenerate cleanly from the fixed code.
5. Run `python scripts/validate_catalog.py` to confirm math rendering passes.
