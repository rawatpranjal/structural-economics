# bullshit-detector — cake-eating-recheck — 2026-05-20

**Bullshit score: 10%** — all three non-HOLDS findings from the original audit (Finding 1 FALSE, Finding 2 DATA DRIFT MED, Finding 3 DATA DRIFT LOW) now HOLDS; the sole residual finding is wall-time values in the README with no disclaimer that they are hardware-dependent (DATA DRIFT LOW, unchanged from original Finding 4).

## Header

- Claim sources: `dynamic-programming/cake-eating/README.md`
- Code / artifact root: `dynamic-programming/cake-eating/run.py`
- Data artifacts: `dynamic-programming/cake-eating/tables/comparison.csv`, `dynamic-programming/cake-eating/tables/method-comparison.csv`
- Seed audit: `dynamic-programming/cake-eating/bullshit-detector_cake-eating_2026-05-20.md`
- Run by: bullshit-detector skill, Claude Sonnet 4.6 (independent re-check)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "k=0 recovers VFI exactly" (fixed from k=1 claim) | HOLDS | none | — |
| 2 | Pseudocode V_eval <- T V_n (fixed from V_eval <- V_n) | HOLDS | none | — |
| 3 | "k+1 times" in Equations and Takeaway (fixed from "k times") | HOLDS | none | — |
| 4 | Wall times (VFI 0.54s, MPI 0.11s, PI 0.15s) no hardware disclaimer | DATA DRIFT | LOW | no |
| 5 | Iteration counts 68, 13, 11 | HOLDS | none | — |
| 6 | Sup-norm vs closed form 2.52e-02 | HOLDS | none | — |
| 7 | Three methods agree to machine precision | HOLDS | none | — |
| 8 | Closed-form policy c*(W) = (1-β)W | HOLDS | none | — |
| 9 | (I - β P_π) V_π = u(π) linear system | HOLDS | none | — |
| 10 | Boundary fallback to analytical_v when W' < w_min | HOLDS | none | — |
| 11 | T_π applied k+1 times total per outer step | HOLDS | none | — |
| 12 | Formal iterate V_{n+1} = T_{π_{n+1}}^k (T V_n) | HOLDS | none | — |

## Findings

### Finding 1 (resolved): "k=0 recovers VFI exactly" — original FALSE now HOLDS

- **Original claim (verbatim):** "Choosing $k=1$ recovers value function iteration exactly." — original README.md (pre-fix)
- **Fixed claim (verbatim):** "Choosing $k=0$ does no evaluation sweep, so the outer step reduces to $V_{n+1} = T V_n$ and recovers value function iteration exactly." — `README.md:88`
- **Code evidence (verbatim):**
  ```python
  for iteration in range(1, 201):
      v_imp, policy_mpi = bellman_step(v_mpi)   # v_imp = T V_n
      v_eval = v_imp                              # eval starts from T V_n
      for _ in range(k_inner):                   # with k_inner=0: zero sweeps
          v_eval = policy_eval_sweep(v_eval, policy_mpi)
  ```
  `run.py:135-139`
  VFI step: `v_new, policy_vfi = bellman_step(v_vfi)` at `run.py:116`. With k=0, `v_eval = v_imp = T V_n = v_new`. The two paths produce identical values.
- **Data evidence:** Not applicable (architectural claim, not a numeric).
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 2 (resolved): Pseudocode V_eval initialization — original DATA DRIFT MED now HOLDS

- **Original claim (verbatim):** `V_eval <- V_n` (old buggy pseudocode, README.md pre-fix line 171)
- **Fixed pseudocode (verbatim):**
  ```text
  V_eval <- T V_n          # one improvement step, same update as VFI
  repeat k times :
      V_eval(W_i) <- u(pi(W_i)) + beta * interp(V_eval, W_i - pi(W_i))
  ```
  `run.py:408-410` (the pseudocode string embedded in Solution Method)
- **Code evidence (verbatim):**
  ```python
  v_imp, policy_mpi = bellman_step(v_mpi)   # run.py:136: v_imp = T V_n
  v_eval = v_imp                             # run.py:137: eval starts from T V_n
  for _ in range(k_inner):
      v_eval = policy_eval_sweep(v_eval, policy_mpi)
  ```
  `run.py:136-139`
- **Category:** HOLDS — pseudocode now correctly reads `V_eval <- T V_n`, matching `v_eval = v_imp` in code.
- **Severity:** none
- **Result-changing:** no

### Finding 3 (resolved): "k+1 times" consistency — original DATA DRIFT LOW now HOLDS

- **Old state:** Equations section used `k`, Takeaway used "k times", only Results/Convergence used "k+1".
- **Fixed state (verbatim):**
  - Equations section, `README.md:82`: "One improvement step is followed by $k$ such evaluation sweeps, so the policy contraction $T_{\pi}$ is applied a total of $k+1$ times per outer step"
  - Takeaway, `README.md:253`: "Modified policy iteration applies the policy contraction $T_{\pi}$ a total of $k+1$ times per outer step and shrinks the error roughly by $\beta^{k+1}$."
  - Results/Convergence, `README.md:220`: "each outer step composes the policy contraction $T_{\pi}$ a total of $k+1$ times" (unchanged, was already correct)
- **Code evidence:** `run.py:135-139` applies `bellman_step` (1 application) + `k_inner` sweeps = k+1 total. All three sections now agree with the code.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 4 (unchanged): Wall times with no hardware disclaimer — DATA DRIFT LOW

- **Claim source (verbatim):** "VFI took **0.54s**. MPI took **0.11s**. Exact PI took **0.15s**." — `README.md:220`
- **CSV evidence:** `tables/method-comparison.csv:2-4`: Wall time (s) = 0.54, 0.11, 0.15. README and CSV agree exactly (same run).
- **Code evidence (verbatim):**
  ```python
  info_vfi["time"] = time.perf_counter() - t0   # run.py:126
  info_mpi["time"] = time.perf_counter() - t0   # run.py:148
  info_pi["time"] = time.perf_counter() - t0    # run.py:168
  ```
  `run.py:126, 148, 168`
- **Category:** DATA DRIFT — wall times are machine- and load-dependent; the committed README reports one specific run with no disclaimer. A reader on different hardware will get different numbers. The README and CSV are internally consistent but the claim is not reproducible across environments.
- **Severity:** LOW
- **Result-changing:** no — the qualitative ordering (MPI fastest, VFI slowest on this grid) will hold on most hardware; the specific seconds will differ.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not re.search(r"(machine|hardware|platform).dependent", Path("README.md").read_text(), re.I)
  # PASSES on current README (no disclaimer); FAILS after honest fix adds one
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert re.search(r"(machine|hardware|platform).dependent|vary by (machine|hardware)", Path("README.md").read_text(), re.I)
  # PASSES after a disclaimer sentence is added; FAILS on current README
  ```

## Cross-cutting patterns

- All three MED/HIGH findings from the original audit root to a single implementation choice: the MPI evaluation phase starts from `v_imp = T V_n` rather than from `v_mpi = V_n`. The fix correctly resolved this by (a) changing the k=1/VFI equivalence claim to k=0, (b) updating the pseudocode initialization to `V_eval <- T V_n`, and (c) aligning all three sections of the README to "k+1 times." The root-cause fix was documentation-only; the code was correct throughout.
- The sole remaining finding (wall times) is a pre-existing LOW severity DATA DRIFT that was already recorded as Finding 4 in the original audit. It was not introduced by the fix and does not affect the tutorial's pedagogical claims.
- No numeric findings changed between the original and re-check audit. The iteration counts, sup-norm errors, and closed-form comparisons in both CSV files continue to match the code faithfully.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Below the 25% ship-after-touch-up threshold. The sole non-HOLDS finding (wall times, DATA DRIFT LOW) requires at most one sentence added to the Results section noting that wall times are hardware-dependent. No code change is needed.

1. **Optional Finding 4 fix:** Add one sentence after the wall-time numbers in `run.py`'s `add_results` call for the convergence figure, e.g. "Wall times vary by hardware; the ordering (MPI fastest, VFI slowest) is robust." Regenerate the README and re-run `python scripts/validate_catalog.py`.

2. **Re-run this skill** after the fix to confirm the score drops to 0%.
