# bullshit-detector — cake-eating — 2026-05-20

**Bullshit score: 35%** — one FALSE claim (k=1 recovers VFI) is contradicted by the code's own MPI loop, and two DATA DRIFT findings (pseudocode initialization; k vs k+1 sweep-count inconsistency across sections) mean the tutorial describes an algorithm that differs from the one it implements.

## Header
- Claim sources: `dynamic-programming/cake-eating/README.md`
- Code / artifact root: `dynamic-programming/cake-eating/run.py`
- Data artifacts: `dynamic-programming/cake-eating/tables/comparison.csv`, `dynamic-programming/cake-eating/tables/method-comparison.csv`
- Seed audit (if any): None
- Run by: bullshit-detector skill, Claude Sonnet 4.6
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Choosing k=1 recovers value function iteration exactly" | FALSE | MED | no (pedagogical claim, not a results number; but a reader who sets k=1 expecting VFI gets a different algorithm) |
| 2 | Pseudocode `V_eval <- V_n` vs code `v_eval = v_imp` | DATA DRIFT | MED | no (convergence numbers match CSV, but the described algorithm differs from the implemented one) |
| 3 | Takeaway "k times" vs Equations "k" vs Convergence prose "k+1" | DATA DRIFT | LOW | no (internal inconsistency between sections; code is consistent with k+1) |
| 4 | Wall times (VFI 0.45s, MPI 0.09s, Exact PI 0.10s) | DATA DRIFT | LOW | needs re-run to verify |
| 5 | VFI 68 iters, MPI 13 iters, PI 11 iters | HOLDS | none | — |
| 6 | Sup-norm vs closed form 2.52e-02 | HOLDS | none | — |
| 7 | Three methods agree to machine precision | HOLDS | none | — |
| 8 | Closed-form policy c*(W) = (1-beta)W, value function formula | HOLDS | none | — |
| 9 | Policy transition matrix (I - beta P_pi) linear system | HOLDS | none | — |
| 10 | Boundary fallback to analytical_v when W' < w_min | HOLDS | none | — |

## Findings

### Finding 1: "Choosing k=1 recovers value function iteration exactly"

- **Claim source (verbatim):** "Choosing $k=1$ recovers value function iteration exactly." — `README.md:87`
- **Also repeated:** "Failure mode: setting $k=1$ makes MPI identical to VFI and removes the speed-up." — `README.md:179`
- **Code evidence (verbatim):**
  ```python
  for iteration in range(1, 201):
      v_imp, policy_mpi = bellman_step(v_mpi)   # line 136: applies T once; v_imp = T V_n
      v_eval = v_imp                              # line 137: eval starts from T V_n, not V_n
      for _ in range(k_inner):                   # line 138: k_inner = 5 by default
          v_eval = policy_eval_sweep(v_eval, policy_mpi)  # line 139: k more T_pi applications
      err = float(np.max(np.abs(v_eval - v_mpi)))         # line 140
  ```
  `run.py:135-140`
- **VFI loop for comparison:**
  ```python
  for iteration in range(1, 501):
      v_new, policy_vfi = bellman_step(v_vfi)   # line 116: applies T once; returns T V_n
      err = float(np.max(np.abs(v_new - v_vfi)))  # line 117
  ```
  `run.py:115-118`
- **Data evidence:** Not applicable (this is an architectural claim, not a results number).
- **Category:** FALSE
- **Severity:** MED
- **Result-changing:** no — the convergence numbers in the CSV were produced with k=5, not k=1, so the published table is unaffected. However, the claim is false for the code as written.
- **Analysis:** Standard MPI (Puterman 1994) defines V_{n+1} = T_{pi}^k V_n where pi = greedy(V_n). With k=1 this reduces to T_{pi} V_n = T V_n = VFI. In this code, however, the MPI loop starts the evaluation phase from `v_imp = T V_n` (the result of `bellman_step`), not from `v_mpi = V_n`. So with k=1 the code computes T_{pi}(T V_n), which is two operator applications, not one. Setting k=0 (zero eval sweeps, v_eval remains v_imp) would recover VFI; k=1 does not. The claim would hold for the standard formulation but not for this implementation.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert v_eval_k1 == pytest.approx(v_new_vfi, rel=1e-10)  # PASSES on buggy code only if T_pi(T V_n)==T V_n, which fails in general; FAILS on honest fix where k=0 recovers VFI
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert v_eval_k0 == pytest.approx(v_new_vfi, rel=1e-10)  # k=0 (no eval sweeps) leaves v_eval=v_imp=T V_n = VFI update; PASSES on honest fix (clarify claim to k=0), FAILS on current code where the claim asserts k=1
  ```

### Finding 2: Pseudocode `V_eval <- V_n` contradicts code `v_eval = v_imp`

- **Claim source (verbatim):** (MPI pseudocode)
  ```text
  V_eval <- V_n
  repeat k times :
      V_eval(W_i) <- u(pi(W_i)) + beta * interp(V_eval, W_i - pi(W_i))
  err   <- max_i | V_eval(W_i) - V_n(W_i) |
  ```
  `README.md:171-174`
- **Code evidence (verbatim):**
  ```python
  v_imp, policy_mpi = bellman_step(v_mpi)  # v_imp = T V_n
  v_eval = v_imp                            # eval starts from T V_n, not v_mpi = V_n
  for _ in range(k_inner):
      v_eval = policy_eval_sweep(v_eval, policy_mpi)
  err = float(np.max(np.abs(v_eval - v_mpi)))
  ```
  `run.py:136-140`
- **Data evidence:** The published iteration counts (68, 13, 11) match the CSV. The discrepancy is between the described algorithm and the implemented one, not between the code and the results table.
- **Category:** DATA DRIFT
- **Severity:** MED
- **Result-changing:** no — the iteration counts and errors in the tables come from the code, not the pseudocode. But the pseudocode describes a different algorithm than the code runs.
- **Analysis:** The pseudocode initializes `V_eval <- V_n` (the previous outer iterate). The code initializes `v_eval = v_imp` where `v_imp = T V_n` (the Bellman update). The pseudocode therefore describes the standard MPI starting from V_n with k evaluation sweeps. The code runs k evaluation sweeps starting from T V_n, which is k+1 total T-like applications per outer step. These are different algorithms; the pseudocode is wrong relative to the code.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert id(v_eval_init) == id(v_mpi)  # pseudocode says eval starts from V_n; PASSES on pseudocode-faithful code, FAILS on current code where v_eval = v_imp
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert np.allclose(v_eval_after_0_sweeps, v_mpi)  # if eval initializes from V_n, 0 sweeps leaves v_eval = V_n; PASSES on honest fix, FAILS on current code (0 sweeps leaves v_eval = T V_n)
  ```

### Finding 3: Sweep count inconsistency — "k times" vs "k+1 times" across sections

- **Claim source 1 (verbatim):** "Modified policy iteration applies the policy contraction $T_{\pi}$ a total of $k$ times per outer step and shrinks the error roughly by $\beta^{k+1}$." — `README.md:252` (Takeaway)
- **Claim source 2 (verbatim):** "MPI with $k = 5$ inner sweeps drops faster because each outer step composes the policy contraction $T_{\pi}$ a total of $k+1$ times." — `README.md:219` (Results/Convergence)
- **Claim source 3 (verbatim):** "$V_{n+1} = T_{\pi_{n+1}}^{\,k} V_n$" — `README.md:84` (Equations)
- **Code evidence:** Code does `bellman_step` (one T application) + `k_inner` eval sweeps = k+1 total T-like applications. The convergence prose (README:219) is consistent with the code. The Equations section (README:84) and Takeaway (README:252) state k, not k+1.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no — all three sections describe the same convergence behavior qualitatively; the discrepancy is in the sweep count stated in two of three places.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "k" in readme_takeaway_mpi_sentence and "k+1" not in readme_takeaway_mpi_sentence  # PASSES on current doc (Takeaway says "k times"), FAILS after honest fix aligns to "k+1"
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert readme_equations_mpi.count("k+1") >= 1 and readme_takeaway_mpi.count("k+1") >= 1  # PASSES after all three sections are updated to k+1; FAILS on current doc
  ```

### Finding 4: Wall times (VFI 0.45s, MPI 0.09s, Exact PI 0.10s)

- **Claim source (verbatim):** "VFI took **0.45s**. MPI took **0.09s**. Exact PI took **0.10s**." — `README.md:219`
- **Code evidence:** Wall times are machine-dependent and embedded at runtime via `time.perf_counter()` (`run.py:126`, `run.py:147`, `run.py:167`). The committed README reflects one specific run on one machine.
- **Data evidence:** `tables/method-comparison.csv:2-4` shows Wall time (s) = 0.45, 0.09, 0.10 matching the README exactly.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** needs re-run to verify — wall times will differ on any other machine or Python environment.
- **Violated invariant:** n/a (timing claims are inherently machine-dependent; no test-level assertion is meaningful).
- **Honest-fix pass condition:** n/a — the fix is a documentation note that times are hardware-dependent, not a code change.

## Cross-cutting patterns

- The MPI loop is implemented in a non-standard form (eval starting from `T V_n` rather than `V_n`). This single off-by-one in the initialization propagates three findings: the false k=1/VFI equivalence claim (Finding 1), the pseudocode contradiction (Finding 2), and the k vs k+1 sweep count mismatch across sections (Finding 3). All three root to `run.py:137`.
- The convergence-prose section (README:219) accidentally states the truth ("k+1 times") while the Equations section and Takeaway state k. This is a sign the prose was written by someone who traced the code loop correctly at one point but then edited the formal sections without updating them.
- No false claims affect the published numeric results. The iteration counts, sup-norm errors, and closed-form comparisons in both CSV files match the code faithfully.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 35%.** This is below the 50% halt threshold. Surface Finding 1 (FALSE) to the user before proposing any fix, because the fix has two valid resolutions: (a) change the claim to "k=0 recovers VFI" and add a note that this implementation uses the alternative formulation; or (b) change the code to start eval from `v_mpi` instead of `v_imp`, which aligns the implementation with the standard Puterman definition and makes k=1 recover VFI. The user must decide which resolution is intended.

1. **Finding 1 (FALSE):** Write a test that runs MPI with k=1 and compares the resulting value function to VFI output. Assert they differ by more than tolerance. This PASSES on current code (they do differ), proving the bug is real.

2. **Finding 2 (DATA DRIFT):** Write a test that traces `v_eval` immediately after line 137 and checks whether it equals `v_mpi` (pseudocode expectation) or `v_imp` (code reality). Assert `np.allclose(v_eval_init, v_mpi)` FAILS on current code.

3. **Finding 3 (DATA DRIFT):** Grep-based test: assert that the string "T_{\pi_{n+1}}^{\,k+1}" or equivalent appears in both the Equations section and the Takeaway section of the generated README. FAILS on current README.

4. After user decides on resolution for Finding 1, hand off to `writing-plans` to draft the fix. Then `executing-plans` to apply. Re-run `python run.py` and confirm iteration counts are unchanged (since k=5 was used for results, not k=1). Re-run this skill to confirm all findings now read HOLDS.
