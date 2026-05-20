# bullshit-detector - dynamic-entry-exit - 2026-05-20

**Bullshit score: 60%** - `_exit_prob` uses a self-loop EV proxy (V[N-1]) instead of the
integrated E[V(N')|stay] stated in the Equations section; the central policy object
p_exit is computed from a different Delta than the one V(N) converged on,
making all downstream equilibrium statistics (stationary distribution, exit rate,
E[N]) products of an undisclosed approximation.

## Header
- Claim sources: `industrial-organization/dynamic-entry-exit/README.md`
- Code / artifact root: `industrial-organization/dynamic-entry-exit/run.py`
- Seed audit (if any): none
- Run by: bullshit-detector (claude-sonnet-4-6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Delta(N) drives both V(N) and p_exit via same E[V(N')\|stay] | DILUTED | HIGH | yes (needs re-run to verify) |
| 2 | Sup-norm error 9.94e-09 at convergence | DILUTED | MED | no |
| 3 | Pseudocode integrates over "rival survivors S" correctly | DILUTED | MED | no |

## Findings

### Finding 1: `_exit_prob` uses self-loop EV, not integrated E[V(N')|stay]

- **Claim source (verbatim):**
  > "An incumbent's exit value is normalized to zero. If it stays, its deterministic
  > surplus is $\Delta(N)=\pi(N)-f+\beta \mathbb{E}\left[V(N_{t+1})\mid N_t=N,\text{ stay}\right].$
  > ...The incumbent exit probability is
  > $p_{\mathrm{exit}}(N)=\frac{1}{1+\exp\{\Delta(N)/\sigma_\varepsilon\}}.$"
  >
  > - `README.md:22-41`

  The README presents a single Delta(N) formula whose E[V(N')|N_t=N, stay] is an
  expectation over the Binomial survivor distribution of rival firms. The same Delta
  drives both V(N) (log-sum formula) and p_exit(N). The equations section does not
  disclose any approximation.

- **Code evidence (verbatim):**
  ```python
  def _exit_prob(N, profits, f, beta, V, sigma_eps):
      """Compute equilibrium exit probability at state N.

      With logistic idiosyncratic shocks, P(exit) = 1/(1 + exp(u_stay/sigma)).
      Here u_stay uses a rough E[V(N')] based on current V at the expected next state.
      """
      pi_N = profits[N - 1]

      # Rough continuation: assume N stays roughly the same (self-consistent approx)
      # This is used only for computing the exit probability
      EV_approx = V[N - 1]
      u_stay = pi_N - f + beta * EV_approx
      return 1.0 / (1.0 + np.exp(u_stay / sigma_eps))
  ```
  `run.py:144-156`

  Contrast with the VFI value update, which correctly integrates over the other N-1
  rivals' survivors:
  ```python
  EV = 0.0
  for s in range(N):  # s = survivors among other N-1 firms
      prob_s = binom.pmf(s, N - 1, p_stay_others) if N > 1 else (1.0 if s == 0 else 0.0)
      if prob_s < 1e-15:
          continue
      N_surv = s + 1
      N_next = min(N_surv + n_enter_current, N_max)
      EV += prob_s * V[N_next - 1]
  u_stay = pi_N - f + beta * EV
  ```
  `run.py:95-108`

  `_exit_prob` is called both during VFI (line 83: `p_exit_i = _exit_prob(...)`) to
  compute the rival stay probability fed into the survivor distribution, and during
  final policy extraction (lines 134-135). In both cases it uses `V[N-1]` (the
  current state's value as a self-loop proxy) rather than the Binomial-integrated EV.

  The two computations yield different u_stay values at the same N. The VFI loop
  converges V using the integrated EV while p_exit is computed throughout from the
  self-loop EV. These are not consistent with the same Delta(N).

- **Data evidence:** CSV `tables/equilibrium-statistics.csv` reports E[N]=7.98,
  Std=0.15, exit probability=0.0027. All three are downstream of p_exit(N).
  "needs re-run to verify" magnitude of shift from a consistent Delta implementation.

- **Category:** DILUTED
- **Severity:** HIGH
- **Result-changing:** yes (needs re-run to verify) - p_exit(N) is the central policy
  object; stationary distribution, E[N], Std[N], exit rate, and all figures in Results
  are downstream of the approximated p_exit. The approximation (`EV_approx = V[N-1]`)
  is never disclosed in the README.

- **Violated invariant (one-line pytest assertion):**
  ```python
  import inspect; src = inspect.getsource(run._exit_prob); assert "V[N - 1]" in src and "binom" not in src
  ```
  (Passes on current code; fails on honest fix integrating over survivors.)

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  import inspect; assert "binom.pmf" in inspect.getsource(run._exit_prob) or "binom" in inspect.getsource(run._exit_prob)
  ```
  (Passes on honest fix; fails on current code.)

---

### Finding 2: Dampened-step error reported as sup-norm convergence error

- **Claim source (verbatim):**
  > "The value iteration converged in **667 iterations** with sup-norm error **9.94e-09**."
  >
  > - `README.md:105`

  Pseudocode: "until max_N |V_{n+1}(N)-V_n(N)| < epsilon" - `README.md:100`

  Both passages present the stopping criterion as the sup-norm of the undampened update.

- **Code evidence (verbatim):**
  ```python
  # Dampened update
  V_update = dampen * V_new + (1.0 - dampen) * V
  error = np.max(np.abs(V_update - V))

  if iteration % 100 == 0:
      print(f"  VFI iteration {iteration:4d}, error = {error:.2e}")

  V = V_update

  if error < tol:
      print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
      break
  ```
  `run.py:117-127`

  `dampen = 0.3` at `run.py:63`. `V_update - V = 0.3*V_new + 0.7*V - V = 0.3*(V_new - V)`.
  Therefore `error = 0.3 * max|V_new - V|`. The true undampened sup-norm at convergence
  is `9.94e-09 / 0.3 = 3.31e-08`, which exceeds the stated tolerance `tol = 1e-8`.
  The code converges when the dampened step is below tol, but the undampened update
  residual is above tol.

- **Data evidence:** `tables/equilibrium-statistics.csv` row "VFI iterations: 667"
  and `README.md:141` match. The iteration count is internally consistent; the error
  number is the issue.

- **Category:** DILUTED
- **Severity:** MED
- **Result-changing:** no - VFI does converge to a fixed point; the stationary
  distribution calculation uses the converged Markov chain, not the VFI residual
  directly. The qualitative conclusions hold. The error number in the report is
  misleading but not result-falsifying.

- **Violated invariant (one-line pytest assertion):**
  ```python
  dampen = 0.3; assert info["error"] < tol and info["error"] / dampen > tol  # error is dampened; true residual exceeds tol
  ```
  (Passes on current code; fails on honest fix that measures undampened residual.)

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(info["error"] - max(abs(V_new - V_prev))) < 1e-15  # undampened sup-norm equals reported error
  ```
  (Passes on honest fix; fails on current code.)

---

### Finding 3: Pseudocode integration variable "S" is ambiguous - rivals vs total survivors

- **Claim source (verbatim):**
  > "for each possible number of rival survivors S:
  >     add V_n(min{S + 1 + e(N), N_max}) to the incumbent's continuation value"
  >
  > - `README.md:96-97`

  The phrase "rival survivors S" with the "+1" suggests S counts surviving rivals,
  so total survivors = S + 1 (this firm + S rivals). The pseudocode draws S from
  all rivals.

- **Code evidence (verbatim):**
  ```python
  EV = 0.0
  for s in range(N):  # s = survivors among other N-1 firms
      prob_s = binom.pmf(s, N - 1, p_stay_others) if N > 1 else (1.0 if s == 0 else 0.0)
  ```
  `run.py:95-97`

  The loop variable `s` ranges 0..N-1 and draws from `Binomial(N-1, p_stay)`. This
  is rivals only (N-1 other firms), consistent with the "+1" in pseudocode. So far
  so good. BUT: the pseudocode says `S + 1 + e(N)` for total next-period firms, where
  S is labelled "rival survivors." The code computes `N_surv = s + 1` (line 102)
  then `N_next = min(N_surv + n_enter_current, N_max)` (line 103), which is
  `min(s + 1 + n_enter, N_max)`. That matches the pseudocode formula.

  However, the loop comment "s = survivors among other N-1 firms" and range `range(N)`
  (0 to N-1) is correct. But `range(N)` runs s from 0 to N-1, covering all possible
  rival survivor counts for N-1 rivals. This is right. The pseudocode says
  "for each possible number of rival survivors S" but does not state S's range or
  distribution clearly. A reader could interpret S as total survivors (0 to N),
  yielding `Binomial(N, p_stay)` - which is what the transition matrix uses (line 191:
  `binom.pmf(s, N, p_stay)`) but NOT what the VFI loop uses.

  The VFI loop (N-1 rivals) and transition matrix computation (N firms, lines 190-191)
  use DIFFERENT binomial distributions for survivor counts: one conditions on the
  focal firm staying, one does not. The pseudocode presents one formula for both
  contexts without flagging this distinction.

- **Data evidence:** Not directly testable from CSV without re-run.

- **Category:** DILUTED
- **Severity:** MED
- **Result-changing:** no - the VFI integration (N-1 rivals) is correct for the
  incumbent's expected value conditional on staying. The transition matrix (N firms)
  is correct for the unconditional transition. The pseudocode ambiguity does not
  indicate the code is wrong, only that the documentation misrepresents the
  integration object.

- **Violated invariant (one-line pytest assertion):**
  ```python
  import inspect; src = inspect.getsource(run.solve_model); assert "binom.pmf(s, N - 1" in src  # VFI uses N-1, pseudocode unclear
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "N-1 rival" in README_pseudocode or "binom(N-1" in README_pseudocode  # pseudocode explicitly states N-1 rivals
  ```

---

## Cross-cutting patterns

- The inconsistency in Finding 1 is self-aware in the code: the `_exit_prob` docstring
  says "Here u_stay uses a rough E[V(N')] based on current V at the expected next
  state" and the comment says "Rough continuation: assume N stays roughly the same
  (self-consistent approx)". The README contains no disclosure of this approximation.
  The gap is between the README's presentation and the code's own admitted roughness.

- The same function `_exit_prob` is called both during VFI (to set p_stay_others
  used in the EV sum) and for final policy extraction. The approximation therefore
  propagates into the VFI loop itself, not just post-hoc. The converged V is not
  the fixed point of the stated Bellman; it is the fixed point of a Bellman where
  p_exit is computed from the self-loop EV.

- Findings 2 and 3 share the same root: the pseudocode and description in Solution
  Method were written to match the intended algorithm, not the implemented one.
  A pattern of aspirational pseudocode vs implemented approximation.

## TDD execution sequence (for the next agent)

0. **Score is 60%. Surface to user before fixing.** The stationary distribution,
   E[N], exit rate, and all figures in Results are products of the undisclosed
   approximation in `_exit_prob`. The user should decide whether to (a) fix
   `_exit_prob` to integrate over the survivor distribution (consistent with the
   stated Bellman), or (b) disclose the approximation in the README and relabel
   the pseudocode. Options have different implications for reported numbers.

1. **Finding 1 test pair:**
   - Violated invariant (write this first, confirm it passes on current code):
     ```python
     import inspect, run
     src = inspect.getsource(run._exit_prob)
     assert "V[N - 1]" in src and "binom" not in src
     ```
   - Honest-fix pass condition (confirm it fails on current code):
     ```python
     import inspect, run
     assert "binom" in inspect.getsource(run._exit_prob)
     ```

2. **Finding 2 test pair:**
   - Violated invariant:
     ```python
     _, _, _, _, info = run.solve_model(30, 10, 1, 2, 0.5, 5.0, 0.95, tol=1e-8)
     assert info["error"] < 1e-8 and info["error"] / 0.3 > 1e-8
     ```
   - Honest-fix pass condition:
     ```python
     _, _, _, _, info = run.solve_model(30, 10, 1, 2, 0.5, 5.0, 0.95, tol=1e-8)
     assert info["error"] < 1e-8 and info["error"] / 0.3 < 1e-8  # true residual also below tol
     ```

3. Hand off to `writing-plans` to decide fix strategy for Finding 1 (integrate vs
   disclose). If integrate: update `_exit_prob` to accept `profits, entry_count, N_max`
   and sum over Binomial(N-1, p_stay) rivals. Re-run to get updated stationary
   distribution and equilibrium statistics. Refresh README and CSVs.

4. After fixes, re-run this skill to confirm all findings read HOLDS and score
   drops to <= 25%.
