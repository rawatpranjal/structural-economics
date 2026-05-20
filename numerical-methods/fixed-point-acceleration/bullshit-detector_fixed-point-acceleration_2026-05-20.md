# bullshit-detector - fixed-point-acceleration - 2026-05-20

**Bullshit score: 35%** - One FALSE claim (Damped Picard labeled "converged" despite hitting max_iter with residual 5 orders of magnitude above tolerance) and one DILUTED range claim in the stress test. No wrong algorithm implementations; all algebra holds. Score is mid-range of the "one DILUTED at MED severity" band, rounded up for the FALSE status label which is directly reader-facing.

## Header

- Claim sources: `numerical-methods/fixed-point-acceleration/README.md`
- Code / artifact root: `numerical-methods/fixed-point-acceleration/run.py`
- Data artifacts: `numerical-methods/fixed-point-acceleration/tables/method_comparison.csv`, `tables/stress_test.csv`, `tables/cournot_summary.csv`
- Seed audit: none
- Run by: bullshit-detector agent (Claude Sonnet 4.6), 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Damped Picard status = "converged" in table | FALSE | HIGH | yes - reader told method converged; it did not reach tolerance |
| 2 | Stress test sweeps "from benign 0.5 down to 0.01" | DILUTED | MED | no - numbers shown are correct; claimed range is overstated |
| 3 | Anderson algebra (F, G, gamma, update rule) matches equations | HOLDS | - | - |
| 4 | Safeguard compares against current residual as pseudocode states | HOLDS | - | - |
| 5 | Picard/Damped Picard stopping rule matches pseudocode | HOLDS | - | - |
| 6 | Iteration counts in Results match tables/method_comparison.csv | HOLDS | - | - |
| 7 | Stress table numbers match tables/stress_test.csv | HOLDS | - | - |
| 8 | Cournot first-step overshoot to (4.5, 4.5) | HOLDS | - | - |
| 9 | Cournot Nash q* = 3.0 | HOLDS | - | - |
| 10 | Closed-form benchmark delta_j* = log(s_j) - log(s0) | HOLDS | - | - |

## Findings

### Finding 1: Damped Picard labeled "converged" despite not reaching tolerance

- **Claim source (verbatim):** `| Damped Picard    | damping alpha = 0.5              |          200 |         2.51e-09 |                  2.97e-08 | converged |` - `README.md:199`
- **Code evidence (verbatim):**
  ```python
  method_table = pd.DataFrame({
      "Method": ["Picard", "Damped Picard", "Anderson (m = 5)"],
      ...
      "Status": ["converged", "converged", "converged"],
  })
  ```
  `run.py:600-619`
- **Data evidence:** `tables/method_comparison.csv:3` - `Damped Picard,damping alpha = 0.5,200,2.51e-09,2.97e-08,converged`

  The `damped_picard` function iterates up to `max_iter=200` (`run.py:43`). The stopping condition is `if residuals[-1] < tol_: break` (`run.py:89`). `tol = 1e-12` (`run.py:42`). Final residual `2.51e-09` is `2.51e03` times above tolerance. The loop exhausted `max_iter` without triggering the break. `dp_iter = len(dp_residuals) = 200 = max_iter`. The status "converged" is hardcoded unconditionally for all three methods; no code path checks whether `break` was reached.

- **Category:** FALSE - the code exits via loop exhaustion (not the `break` on `residuals[-1] < tol_`); "converged" requires the break path to have been taken.
- **Severity:** HIGH
- **Result-changing:** yes - the table presented to the reader as the main comparison asserts the method converged; it did not. The final residual `2.51e-09` and distance-to-closed-form `2.97e-08` are both legitimate numbers, but the "converged" label misrepresents the termination condition and implies the method achieved the stated tolerance.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert method_table.loc[method_table["Method"] == "Damped Picard", "Status"].values[0] == "converged"
  # PASSES on current buggy code (hardcoded), FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert method_table.loc[method_table["Method"] == "Damped Picard", "Status"].values[0] in ("max_iter", "did not converge")
  # PASSES on honest fix (status reflects termination condition), FAILS on current buggy code
  ```

---

### Finding 2: Stress test claimed to sweep "from benign 0.5 down to 0.01" but 0.5 and 0.2 are silently skipped

- **Claim source (verbatim):** "The stress test sweeps the outside share from a benign 0.5 down to 0.01." - `README.md:184`
- **Code evidence (verbatim):**
  ```python
  stress_outsides = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
  stress_rows = []
  for s0_target in stress_outsides:
      ...
      e3 = (1.0 - s0_target) / s0_target - e.sum()
      if e3 <= 0:
          continue
      ...
      stress_rows.append({...})
  ```
  `run.py:145-224`
- **Data evidence:** `tables/stress_test.csv` contains exactly 4 data rows: `0.10`, `0.05`, `0.02`, `0.01`. No row for `0.50` or `0.20`.

  For `s0_target=0.5`: `e = exp([log(0.40/0.5), log(0.25/0.5), log(0.20/0.5)]) = [0.8, 0.5, 0.4]`; `e3 = (0.5/0.5) - 1.7 = -0.7 <= 0` - skipped. For `s0_target=0.2`: `e = [2.0, 1.25, 1.0]`; `e3 = (0.8/0.2) - 4.25 = -0.25 <= 0` - skipped. Both are silently skipped without any warning, log, or prose acknowledgment.

- **Category:** DILUTED - the code attempts to include 0.5 and 0.2 but the share parameterization makes them infeasible; the prose claims a sweep starting at 0.5 when the actual sweep starts at 0.1. The numbers shown are correct; the stated range is not.
- **Severity:** MED - the stress test conclusion (Anderson stays flatter as s0 shrinks) holds on the 4 points that are computed; the overstated range does not change the qualitative lesson.
- **Result-changing:** no - the stress test table numbers are all correct; the prose misrepresents the starting point of the sweep.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert any(r["s_outside"] == 0.5 for r in stress_rows)
  # PASSES if 0.5 were included (would fail on current code since it is skipped)
  # Current code: this FAILS (0.5 is not in stress_rows) - rewrite as:
  assert "The stress test sweeps the outside share from a benign 0.5" in readme_text and not any(abs(r["s_outside"] - 0.5) < 1e-9 for r in stress_rows)
  # PASSES on current buggy code (prose says 0.5, data omits it), FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert readme_text_stress_sentence.startswith("The stress test sweeps the outside share from") and "0.1" in readme_text_stress_sentence[:80]
  # PASSES on honest fix (prose states the actual starting point 0.1), FAILS on current buggy code
  ```

## Cross-cutting patterns

- The "converged" false label is the only hardcoded status string. All other numeric claims are computed dynamically from the run. The bug is localized to `run.py:618` where the `Status` list is constructed without checking whether `break` was reached.
- The stress parameterization bug (`e3 <= 0` for large `s0_target`) is a consequence of the share construction logic at `run.py:148-154` using fixed inside shares (0.40, 0.25, 0.20) that already sum to more than `1 - s0_target` for benign outside shares. The silent `continue` at `run.py:159` is not surfaced in any warning or printed output. The discrepancy between the 6-element `stress_outsides` list and the 4-row table is detectable by inspection.
- All algorithm algebra (Anderson F/G matrices, gamma solve, Picard update, damped Picard update, Cournot BR map) matches the equations precisely. No parametric leaks, no wrong norms, no missing terms.
- The safeguard comparison (`new_residual > 2.0 * prev_residual` at `run.py:127`) correctly compares against the current step's pre-candidate residual, consistent with the pseudocode at `README.md:162`.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 35%.** Below the 50% halt threshold. Surface both findings to the user before fixing. Finding 1 (FALSE status label) is the higher-priority fix.

1. **Finding 1 - violated invariant test:**
   ```python
   # Confirm bug: status is always "converged" regardless of termination
   assert dp_residuals[-1] > 1e-12  # Damped Picard did not meet tol
   assert dp_iter == 200             # hit max_iter
   assert method_table.loc[method_table["Method"] == "Damped Picard", "Status"].values[0] == "converged"
   # All three should pass on current code
   ```

2. **Finding 1 - honest-fix pass condition test:**
   ```python
   # After fix: status reflects actual termination
   assert method_table.loc[method_table["Method"] == "Damped Picard", "Status"].values[0] != "converged"
   # Should pass on honest fix (status updated to reflect max_iter hit)
   ```

3. **Finding 2 - violated invariant test:**
   ```python
   assert len(stress_rows) == 4  # only 4 rows, not 6
   assert not any(abs(r["s_outside"] - 0.5) < 1e-9 for r in stress_rows)
   # Both should pass on current code (proves the gap)
   ```

4. **Finding 2 - honest-fix pass condition test:**
   ```python
   # Prose describes the actual sweep range
   assert "0.1" in stress_description_sentence  # starts at 0.1, not 0.5
   # Should pass on honest fix (prose corrected to match actual data)
   ```

5. Fix `run.py:618` to compute status dynamically based on whether the final residual meets `tol`, e.g.:
   - For Picard: `"converged" if pi_residuals[-1] < tol else f"stopped at {max_iter} iterations"`
   - For Damped Picard: same pattern
   - Fix stress prose: change "from a benign 0.5 down to 0.01" to "from 0.1 down to 0.01" (the actual sweep range that survives the feasibility check)

6. Regenerate `README.md` and `tables/method_comparison.csv`. Re-run this skill to confirm both findings now read HOLDS and score is <= 10%.
