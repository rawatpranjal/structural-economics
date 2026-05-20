# bullshit-detector -- rbc -- recheck -- 2026-05-20

**Bullshit score: 10%** -- all structural equations, steady-state values, and IRF table numbers hold; the original two findings (DATA DRIFT on C/Y, MISLABELED "run in parallel") are both resolved; the residual 10% is for runtime-embedded values (undetermined-coefficients residual 2.9e-15, Klein QZ diff 1.5e-15, capital/labor rule coefficients) that are self-consistent across artifacts but cannot be re-derived without executing the QZ solver.

## Header
- Claim sources: `dsge/rbc/README.md` (full file, 237 lines)
- Code / artifact root: `dsge/rbc/run.py` (821 lines)
- Data artifacts: `dsge/rbc/tables/irf-summary-fixed-labor.csv`, `dsge/rbc/tables/irf-summary-endogenous-labor.csv`
- Seed audit: `dsge/rbc/bullshit-detector_rbc_2026-05-20.md`
- Run by: bullshit-detector skill (Claude Sonnet 4.6), 2026-05-20 (recheck)
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Two solvers run "one after the other" / "in sequence" | HOLDS | - | - |
| 2 | C/Y = 0.76 (Equations prose, two-decimal format) | HOLDS | - | - |
| 3 | C/Y = 0.765 (Model Setup table, three-decimal format) | HOLDS | - | - |
| 4 | All Case A steady-state values (K=28.348, Y=3.015, C=2.307, K/Y=9.401) | HOLDS | - | - |
| 5 | All Case B steady-state values (K=9.449, Y=1.005, C=0.769, psi=7.883) | HOLDS | - | - |
| 6 | All structural equations (Euler, resource constraint, labor supply, TFP AR(1)) | HOLDS | - | - |
| 7 | All Case A IRF table numbers (5 rows x 5 columns = 25 cells) | HOLDS | - | - |
| 8 | All Case B IRF table numbers (6 rows x 4 columns = 24 cells) | HOLDS | - | - |
| 9 | Klein matrix (A, B) rows match log-linearized equilibrium conditions | HOLDS | - | - |
| 10 | Undetermined-coefficients residual 2.9e-15; QZ diff 1.5e-15 | DATA DRIFT | LOW | no (runtime-only, no independent CSV artifact) |
| 11 | Capital rule k_t = 0.9531 k_{t-1} + 0.1078 a_t; labor rule n_t = -0.1677 k_{t-1} + 0.4612 a_t | DATA DRIFT | LOW | no (runtime QZ output, consistent with impact IRF numbers) |

## Findings

### Finding 1 (RESOLVED from original audit): C/Y inconsistency

Original audit Finding 1 claimed a three-way inconsistency: Equations prose (0.76), Model Setup table (0.765), and what `{:.2f}` would generate (predicted "0.77"). The audit's prediction of 0.77 was wrong.

Independent recomputation: α=0.33, β=0.99, δ=0.025. C/Y = (Y - δK)/Y = 0.764964. `{:.2f}` of 0.764964 = **0.76** (rounds down). `{:.3f}` of 0.764964 = **0.765**. Both representations in the current README are correct roundings of the same number.

- `run.py:494`: `f"$C/Y = {ss_A['C_Y']:.2f}$"` → 0.76. README.md:55-56: `C/Y = 0.76`. Match. HOLDS.
- `run.py:547`: `f"$C/Y$ | {ss_A['C_Y']:.3f}"` → 0.765. README.md:112: `C/Y | 0.765`. Match. HOLDS.

No inconsistency. **RESOLVED.**

### Finding 2 (RESOLVED from original audit): "run in parallel" mislabel

Original audit Finding 2 flagged README Overview saying "Two solvers run in parallel." Current README.md:9 reads: "Two solvers run on the same model, one after the other." Current README.md:116 reads: "Two methods run in sequence." Code at `run.py:385-386`: `policy_A = solve_log_linear_policy(...)` then `qz_A = klein_qz_policy_fixed(...)` — sequential calls in `main()`. Both prose descriptions now match the sequential code. **RESOLVED.**

### Finding 3 (LOW, DATA DRIFT): Runtime residuals 2.9e-15 and 1.5e-15

- **Claim source (verbatim):** "The undetermined-coefficients residual is 2.9e-15. Klein QZ agrees with the hand-derived (p, q) to 1.5e-15." -- `README.md:142`
- **Code evidence (verbatim):**
  ```python
  f"The undetermined-coefficients residual is {policy_A['max_residual']:.1e}. "
  f"Klein QZ agrees with the hand-derived (p, q) to {qz_diff_A:.1e}. Both "
  ```
  `run.py:581-582`
- **Data evidence:** No CSV artifact stores these values. They are runtime-generated and committed through string formatting. They are self-consistent (one sentence, single README occurrence) but cannot be cross-checked against any on-disk artifact.
- **Category:** DATA DRIFT
- **Severity:** LOW -- the values are plausible (machine-precision residuals for a 3x3 linear system solved by scipy.root and ordqz); the committed numbers are self-referential but harmless.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "tables/residuals.csv" not in open("dsge/rbc/README.md").read()  # PASSES (no CSV archive); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert float(pd.read_csv("dsge/rbc/tables/residuals.csv")["uc_residual"][0]) < 1e-14  # PASSES after archiving; FAILS currently (no file)
  ```

### Finding 4 (LOW, DATA DRIFT): Capital and labor rule coefficients not independently verifiable

- **Claim source (verbatim):** "The capital rule is $\hat k_t = 0.9531\hat k_{t-1} + 0.1078\hat a_t$. The labor rule is $\hat n_t = -0.1677\hat k_{t-1} + 0.4612\hat a_t$." -- `README.md:168`
- **Code evidence (verbatim):**
  ```python
  f"$\\hat k_t = {F[0, 0]:.4f}\\hat k_{{t-1}} + {F[0, 1]:.4f}\\hat a_t$. "
  f"The labor rule is $\\hat n_t = {P[1, 0]:.4f}\\hat k_{{t-1}} + "
  f"{P[1, 1]:.4f}\\hat a_t$."
  ```
  `run.py:613-615`
- **Data evidence:** These are QZ runtime outputs embedded directly in the README via f-string. No CSV stores F or P. The impact IRF numbers in `irf-summary-endogenous-labor.csv` provide indirect confirmation: Labor impact = 0.461% = `P[1,1] * shock * 100` = 0.4612 * 0.01 * 100 = 0.4612%, consistent to the reported 0.461%. Output impact 1.309% is consistent with `a_hat + alpha * k_lag + (1-alpha) * n_hat` at t=0 given `P[0,1]=P[1,1]` under the shock. The coefficients are internally consistent with the IRF tables.
- **Category:** DATA DRIFT
- **Severity:** LOW -- indirect consistency check passes; no direct artifact stores F and P.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not Path("dsge/rbc/tables/policy-coefficients.csv").exists()  # PASSES (no archive); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(float(pd.read_csv("dsge/rbc/tables/policy-coefficients.csv").set_index("name").loc["F_00", "value"]) - 0.9531) < 1e-4
  ```

## Cross-cutting patterns

- Both original findings (DATA DRIFT C/Y, MISLABELED "parallel") are fully resolved. The current README is faithful to the code.
- The two remaining LOW DATA DRIFT findings are both about unarchived runtime values (solver residuals, QZ policy coefficients). Neither affects any tabulated IRF number. Archiving them to CSV would bring the score to 0%.
- All 49 numeric cells in the two IRF summary tables match between README prose, README tables, and CSVs exactly. No drift found.
- All structural equations in the Equations section match the corresponding code rows in `klein_qz_policy_fixed` (run.py:100-114) and `klein_system_labor` (run.py:227-244), verified against first-principles log-linearization.
- The `C/Y = 0.76` vs `C/Y = 0.765` discrepancy the original audit flagged is not a discrepancy -- it is two valid roundings of 0.764964 at different decimal places, generated by different format strings for different table contexts. The original audit's claim that `{:.2f}` of 0.7652 = 0.77 was arithmetically wrong (actual value is 0.764964, not 0.7652).

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Below the 50% halt threshold. All load-bearing results are clean.
1. **Finding 3 (DATA DRIFT, LOW):** Optional -- archive `policy_A['max_residual']` and `qz_diff_A` to a small CSV (e.g., `tables/solver-diagnostics.csv`) before they are string-formatted into the README. This creates a cross-checkable artifact.
2. **Finding 4 (DATA DRIFT, LOW):** Optional -- archive `F` and `P` matrices to `tables/policy-coefficients.csv`. Provides a durable record and enables the indirect consistency test above to be explicit.
3. Neither finding requires a prose fix or a re-run of the model. The README is faithful to the code as-is. Archiving the runtime values is a hygiene improvement, not a correctness fix.
4. Re-run `python scripts/validate_catalog.py` before any commit.
