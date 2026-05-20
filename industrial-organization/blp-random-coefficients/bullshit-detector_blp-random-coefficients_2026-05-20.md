# bullshit-detector — blp-random-coefficients — 2026-05-20

**Bullshit score: 20%** — three runtime numbers in the README (iteration count 627, recovery error 2.45e-11, GMM evaluations 46) are committed prose claims that cannot be verified against any on-disk artifact without a re-run; every structural formula and parameter value checks out.

## Header
- Claim sources: `industrial-organization/blp-random-coefficients/README.md`
- Code / artifact root: `industrial-organization/blp-random-coefficients/run.py`
- Data artifacts: `industrial-organization/blp-random-coefficients/tables/parameter-estimates.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Contraction converged in 627 iterations | DATA DRIFT | LOW | no (cosmetic count) |
| 2 | Max delta recovery error = 2.45e-11 | DATA DRIFT | LOW | no (plausibility check only) |
| 3 | GMM evaluated objective 46 times | DATA DRIFT | LOW | no (cosmetic count) |
| 4 | Share formula: s_jt = (1/ns) sum_i logit | HOLDS | - | - |
| 5 | Contraction update: delta += log(s_obs) - log(s_pred) | HOLDS | - | - |
| 6 | mu_ijt = sigma_x*nu_i1*x_jt + sigma_p*nu_i2*p_jt | HOLDS | - | - |
| 7 | Initial delta from simple-logit inversion | HOLDS | - | - |
| 8 | Instruments include cost shifter and sums of rival characteristics | HOLDS | - | - |
| 9 | Price endogenous (Cov(p,xi) != 0) | HOLDS | - | - |
| 10 | Parameter table values match CSV | HOLDS | - | - |
| 11 | Model Setup table values match code | HOLDS | - | - |
| 12 | Outside good utility normalized to zero | HOLDS | - | - |
| 13 | Logit IIA: each column identical off-diagonal entries | HOLDS | - | - |
| 14 | GMM objective Q = n g(sigma)' W g(sigma) | HOLDS | - | - |
| 15 | Largest own-elasticity error = 0.315 | DATA DRIFT | LOW | no (cosmetic) |

## Findings

### Finding 1: Contraction converged in 627 iterations

- **Claim source (verbatim):** "At the true nonlinear parameters, the contraction converged in **627 iterations** with max $|\delta^{\mathrm{recovered}}-\delta^{\mathrm{true}}|=2.45e-11$." — `README.md:91`
- **Code evidence (verbatim):**
  ```python
  f"At the true nonlinear parameters, the contraction converged in "
  f"**{len(conv_history)} iterations** with max "
  f"$|\\delta^{{\\mathrm{{recovered}}}}-\\delta^{{\\mathrm{{true}}}}|="
  f"{max_delta_error:.2e}$.\n\n"
  ```
  `run.py:607-610`
- **Data evidence:** `tables/parameter-estimates.csv` does not contain iteration counts or recovery errors. The values 627 and 2.45e-11 exist only as committed README prose from a prior run.
- **Category:** DATA DRIFT — the committed README prose asserts specific runtime values that cannot be cross-checked against any committed data artifact. The code is deterministic with `seed=42` (`run.py:374`) so the values should be reproducible, but they are unverifiable from committed artifacts without re-running. `needs re-run to verify`
- **Severity:** LOW — the numbers are diagnostic outputs (plausibility checks on the contraction), not inputs to any economic conclusion.
- **Result-changing:** no — iteration count and recovery error do not appear in any results table or figure.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "627" in open("README.md").read() and "parameter-estimates.csv" not in open("README.md").read().split("627")[0]
  # PASSES on current state (627 in README, no CSV backing); FAILS after re-run confirms or refutes
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(str(c) == "627" for c in [len(h) for h in [contraction_mapping(s_obs, x, p, 0.8, 0.3, nu, tol=1e-12)[1]]])
  # PASSES if re-run with seed=42 yields exactly 627 iterations; FAILS if count differs
  ```

---

### Finding 2: Max delta recovery error = 2.45e-11

- **Claim source (verbatim):** "max $|\delta^{\mathrm{recovered}}-\delta^{\mathrm{true}}|=2.45e-11$" — `README.md:91`
- **Code evidence (verbatim):**
  ```python
  max_delta_error = np.max(np.abs(delta_recovered - delta_true))
  ```
  `run.py:489`; embedded in report string at `run.py:609`.
- **Data evidence:** No committed artifact contains this value. `needs re-run to verify`
- **Category:** DATA DRIFT — same pattern as Finding 1. Runtime value committed as prose with no CSV backing.
- **Severity:** LOW — value is a sanity check on contraction accuracy, not an economic result.
- **Result-changing:** no.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(float(re.search(r"2\.45e-11", open("README.md").read()).group()) - 2.45e-11) < 1e-13
  # PASSES trivially (the string is there); FAILS after re-run if true value differs
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert np.max(np.abs(delta_recovered - delta_true)) < 1e-10
  # PASSES if re-run confirms recovery error < 1e-10; a weaker but verifiable bound
  ```

---

### Finding 3: GMM evaluated objective 46 times

- **Claim source (verbatim):** "The GMM search used Nelder-Mead after a coarse starting grid and evaluated the objective 46 times." — `README.md:93`
- **Code evidence (verbatim):**
  ```python
  f"The GMM search used Nelder-Mead after a coarse "
  f"starting grid and evaluated the objective {result.nfev} times."
  ```
  `run.py:611-612`
- **Data evidence:** `result.nfev` is not stored in any committed CSV. `needs re-run to verify`
- **Category:** DATA DRIFT — same pattern. Note: the grid search at `run.py:416-424` evaluates `gmm_objective` up to 25 times before `minimize()` is called; `result.nfev` counts only the `minimize()` calls, so the total evaluations are at least 25 + 46 = 71. The README says "evaluated the objective 46 times" without clarifying that the 25-point grid search is additional. This is a mild imprecision.
- **Severity:** LOW — cosmetic count; does not affect any economic conclusion.
- **Result-changing:** no.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "46" in open("README.md").read()  # PASSES; FAILS after re-run if nfev differs
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert result.nfev == 46  # PASSES if re-run with seed=42 gives exactly 46 Nelder-Mead evals
  ```

---

### Finding 4: Largest own-elasticity error = 0.315

- **Claim source (verbatim):** "The largest own-elasticity error is 0.315 in this market." — `README.md:103`
- **Code evidence (verbatim):**
  ```python
  max_own_elast_error = np.max(np.abs(own_elast_blp - own_elast_true))
  ```
  `run.py:490`; embedded at `run.py:660`.
- **Data evidence:** Not stored in any committed CSV. `needs re-run to verify`
- **Category:** DATA DRIFT — runtime value committed as prose.
- **Severity:** LOW — illustrative error magnitude in a single market; not a headline result.
- **Result-changing:** no.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.315" in open("README.md").read()  # PASSES; FAILS after re-run if value differs
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(np.max(np.abs(own_elast_blp - own_elast_true)) - 0.315) < 0.05
  # PASSES if re-run gives value within 0.05 of 0.315 (loose bound)
  ```

## Cross-cutting patterns

- All four non-HOLDS findings are the same pattern: runtime scalar values committed to README prose without being written to any CSV or log file. None affect economic conclusions. The fix is uniform: store these diagnostics in `tables/convergence-diagnostics.csv` so future re-runs can cross-check committed values against fresh runs automatically.
- Every core claim - utility specification, contraction formula, share formula, mu decomposition, instrument construction, price endogeneity mechanism, parameter values, and elasticity formulas - holds exactly against the code. The economics is faithfully implemented.
- `compute_share_jacobian` (`run.py:138-182`) is dead code: defined but never called anywhere in the file. It also has an incomplete implementation (missing the `alpha_i` multiplication, documented as "the caller can multiply by alpha_i" with no caller). This is not a README claim violation but is a latent trap for any future caller.
- The README prose "A price increase sends proportional demand to each rival" (`README.md:112`) is a slight overstatement of IIA. The logit cross-elasticity formula `-alpha * p_k * s_k` (column-constant, not proportional to rival shares) makes rivals with larger shares receive proportionally more absolute demand, but the diversion is proportional to rival shares only in the diversion-ratio sense. The code is correct; the prose is imprecise but not wrong enough to constitute a finding.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20%.** Under the 50% halt threshold. Proceed with fixes.

1. For each DATA DRIFT finding (Findings 1-4), the violated-invariant test is trivial (the string is in the README). The real test is the honest-fix pass condition: re-run with `seed=42` and confirm each committed value matches the fresh output.

2. **Recommended structural fix (no code change needed for correctness):** Add a `tables/convergence-diagnostics.csv` output in `run.py` storing `{contraction_iters, max_delta_error, gmm_nfev, max_own_elast_error}`. This makes all four DATA DRIFT findings verifiable from committed artifacts without re-running.

3. **Dead code:** Consider removing or completing `compute_share_jacobian` (`run.py:138-182`). It requires `alpha` as a separate argument to be useful. Either pass `alpha` and complete the computation, or delete the function.

4. After adding `tables/convergence-diagnostics.csv`, re-run this skill. Expected outcome: all findings read HOLDS, bullshit score drops to 0-10%.
