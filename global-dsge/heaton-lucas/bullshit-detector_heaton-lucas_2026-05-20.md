# bullshit-detector — heaton-lucas — 2026-05-20

**Bullshit score: 25%** — All structural equations hold; the worst finding is DILUTED MED: the README states KKT non-negativity (mu_i >= 0) as a requirement but the code enforces only the complementarity product = 0, not the sign constraint. Several Results numbers cannot be cross-checked without a re-run.

## Header
- Claim sources: `global-dsge/heaton-lucas/README.md` (prose, Equations, Results, Model Setup)
- Code / artifact root: `global-dsge/heaton-lucas/run.py`, `lib/stpfi.py`
- Data artifacts: `global-dsge/heaton-lucas/tables/euler-errors.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | mu_i >= 0 stated as KKT requirement | DILUTED | MED | no (cannot change published numbers; solver still finds sensible equilibria in practice) |
| 2 | Euler errors computed over "simulated" paths (implied 24) | DILUTED | LOW | no (table values are internally consistent; 4-path subset is undisclosed) |
| 3 | Equity premium range 0.43%-1.42% | DATA DRIFT | LOW | needs re-run to verify |
| 4 | Mean wealth share 0.487, p10=-0.050, p90=1.050 | DATA DRIFT | LOW | needs re-run to verify |
| 5 | No-short multiplier positive at 0.3%, borrowing at 2.7% | DATA DRIFT | LOW | needs re-run to verify |
| 6 | Final policy change 2.06e-02, residual 1.27e-03 | DATA DRIFT | LOW | needs re-run to verify |
| 7 | All equations (Euler, budget, clearing, consistency) | HOLDS | none | — |
| 8 | Parameter table (beta, gamma, Kb, n_w, n_eq, shock_num) | HOLDS | none | — |
| 9 | Simulation 24 paths x 10,000 periods (ergodic) | HOLDS | none | — |
| 10 | Table CSV vs README Euler residual values | HOLDS | none | — |

## Findings

### Finding 1: KKT non-negativity mu_i >= 0 stated but not enforced

- **Claim source (verbatim):** "mu_i^s>=0, mu_i^s s_i'=0, mu_i^b>=0, mu_i^b(b_i'-K^b)=0." — `README.md:44-45`
- **Code evidence (verbatim):**
  ```python
  ms1 * s1p,                  # 80
  ms2 * s2p,                  # 81
  mb1 * nb1p,                 # 82
  mb2 * nb2p,                 # 83
  ```
  `run.py:139-142`
- **Data evidence (if applicable):** None directly. The sign of multipliers in the solution is not stored in any CSV artifact.
- **Category:** DILUTED
- **Severity:** MED
- **Result-changing:** no — in practice `scipy.optimize.root` with the hybr/lm fallback tends to find non-negative multipliers for interior points because the Euler residuals pull in the right direction; the omission does not change the published table values but it does mean the code cannot detect a sign violation if one occurs.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert np.all(x_sol[:, :, 5] >= 0) and np.all(x_sol[:, :, 7] >= 0)
  # PASSES if solver happens to return non-negative multipliers; may FAIL on current code at some grid points — proving no enforcement
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "maximum" in inspect.getsource(make_residual_fn) or "jnp.where" in inspect.getsource(make_residual_fn)
  # PASSES on an honest fix that uses Fischer-Burmeister or max-formulation; FAILS on current code
  ```

### Finding 2: Euler-error table uses 4 paths not the stated 24

- **Claim source (verbatim):** "The table reports simulated Euler-equation residuals." and "Simulation | 24 paths x 10,000 periods" — `README.md:99` and `README.md:64`
- **Code evidence (verbatim):**
  ```python
  for p in range(min(n_paths, 4)):
  ```
  `run.py:298`
- **Data evidence (if applicable):** `tables/euler-errors.csv` rows are internally consistent; no artifact records the path count used to produce them.
- **Category:** DILUTED
- **Severity:** LOW — the table values are not wrong, but the text implies the same 24-path simulation used for the ergodic distribution also generates the Euler errors, which is false.
- **Result-changing:** no — the table numbers are what they are from 4 paths; the claim is about scope, not accuracy.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert sum(1 for _ in range(min(24, 4))) == 24
  # FAILS on current code (min returns 4); proves the mismatch
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert sum(1 for _ in range(min(n_paths, n_paths))) == n_paths
  # PASSES when path limit removed or documented; FAILS on current code with hardcoded 4
  ```

### Finding 3: Equity premium range 0.43%-1.42%

- **Claim source (verbatim):** "The computed equity premium ranges from 0.43% to 1.42% on the interior grid." — `README.md:89`
- **Code evidence (verbatim):**
  ```python
  eq_min = float(np.nanmin(eq_prem_pct))
  eq_max = float(np.nanmax(eq_prem_pct))
  ```
  `run.py:330-331`
- **Data evidence (if applicable):** No CSV stores these values. Only the README text asserts them.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** needs re-run to verify
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert Path("global-dsge/heaton-lucas/tables/equity-premium-range.csv").exists()
  # FAILS on current repo; proves no artifact backs this number
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(eq_min_from_csv - 0.43) < 0.01 and abs(eq_max_from_csv - 1.42) < 0.01
  # PASSES after re-run if numbers match; FAILS until CSV artifact is created and verified
  ```

### Finding 4: Mean wealth share 0.487, p10=-0.050, p90=1.050

- **Claim source (verbatim):** "The mean wealth share is 0.487, with 10th and 90th percentiles at -0.050 and 1.050." — `README.md:91`
- **Code evidence (verbatim):**
  ```python
  omega_mean = float(np.mean(omega_all))
  omega_p10, omega_p90 = np.percentile(omega_all, [10, 90])
  ```
  `run.py:334-335`
  The simulation clips omega to `[-0.05, 1.05]` (run.py:289-291). The p10=-0.050 and p90=1.050 equalling the grid endpoints means wealth shares hit the boundary in at least 10% of simulated periods in each tail, which is economically significant and unremarked upon.
- **Data evidence (if applicable):** No CSV stores these values. Only README asserts them.
- **Category:** DATA DRIFT
- **Severity:** LOW — values cannot be verified without re-run; the fact that p10 and p90 equal clip boundaries is a substantive economic finding left unaddressed in the prose.
- **Result-changing:** needs re-run to verify
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert Path("global-dsge/heaton-lucas/tables/ergodic-distribution.csv").exists()
  # FAILS on current repo
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(omega_mean_from_csv - 0.487) < 0.001
  # PASSES after re-run and CSV creation if number is stable
  ```

### Finding 5: No-short at 0.3%, borrowing at 2.7% of interior collocation points

- **Claim source (verbatim):** "Agent 1's no-short-sale multiplier is positive at 0.3% of interior collocation points. The borrowing multiplier is positive at 2.7%." — `README.md:95`
- **Code evidence (verbatim):**
  ```python
  no_short_share = float(np.mean(ms1_sol[:, interior_mask] > 1e-5) * 100.0)
  borrow_share = float(np.mean(mb1_sol[:, interior_mask] > 1e-5) * 100.0)
  ```
  `run.py:332-333`
- **Data evidence (if applicable):** No CSV stores these values.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** needs re-run to verify
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert Path("global-dsge/heaton-lucas/tables/constraint-binding.csv").exists()
  # FAILS on current repo
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(no_short_share_from_csv - 0.3) < 0.1 and abs(borrow_share_from_csv - 2.7) < 0.1
  # PASSES after re-run if numbers are stable
  ```

### Finding 6: Final policy change 2.06e-02 and residual 1.27e-03

- **Claim source (verbatim):** "The final policy change was 2.06e-02, and the maximum pointwise residual was 1.27e-03." — `README.md:85`
- **Code evidence (verbatim):**
  ```python
  f"iterations. The final policy change was {info['error']:.2e}, and "
  f"the maximum pointwise residual was {info['residual']:.2e}."
  ```
  `run.py:434-437`
- **Data evidence (if applicable):** No CSV stores these values. `info['error']` and `info['residual']` are runtime outputs from `lib/stpfi.py:89-93`.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** needs re-run to verify — these numbers are consistent with the cap-stop logic (tol=5e-4 not met, max_iter=80), but the exact values depend on the run.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert Path("global-dsge/heaton-lucas/tables/convergence-info.csv").exists()
  # FAILS on current repo
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(final_change_from_csv - 0.0206) < 0.001
  # PASSES after re-run and CSV creation if solver is deterministic
  ```

## Cross-cutting patterns

- **Unarchived runtime scalars.** Findings 3, 4, 5, and 6 share the same root: scalars computed at runtime (equity premium range, omega percentiles, constraint-binding fractions, convergence metrics) are written into the README text but not saved to any CSV artifact. Any discrepancy between the committed README and a fresh re-run is invisible until re-run. Adding a `tables/scalars.csv` row per run would make all of these verifiable without re-run.

- **Complementarity without sign enforcement.** Finding 1 is a structural gap: the gmod formulation the code claims to translate uses Fischer-Burmeister or min-formulation to enforce mu >= 0 alongside mu*constraint = 0. The direct-product-equals-zero approach in the code is a smooth but incomplete implementation of the stated KKT conditions. The same gap would appear in any other STPFI tutorial in this catalog that copies this complementarity pattern (check `barro-rare-disasters/`, `bianchi-sudden-stops/`).

- **Simulation scope mismatch.** Finding 2 is a documentation gap: the Model Setup table advertises "24 paths x 10,000 periods" as the simulation spec, but Euler errors silently use only 4 paths. This same hardcoded-min pattern could recur in other STPFI tutorials.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Below 50% threshold; proceed, but address the two DILUTED findings before citing these numbers in any publication-facing context.

1. **Finding 1 (KKT sign, DILUTED MED).**
   Turn violated invariant into a pytest that checks whether any `ms1_sol` or `mb1_sol` entry in the committed solution is negative (requires loading the solution array, which is not currently serialized). Add a `tables/solution-arrays.npz` save step, then write the test. Confirm the test fails (negative multipliers exist at some grid points) or passes (they don't). If they exist, implement Fischer-Burmeister: `min(mu, -constraint) = 0` in place of `mu * constraint = 0`.

2. **Finding 2 (4-path Euler, DILUTED LOW).**
   Replace `min(n_paths, 4)` with `n_paths` in `run.py:298`, or add an explicit `n_paths_euler = 4` variable and document it in the Model Setup table. Either way, the test is:
   ```python
   assert "min(n_paths, 4)" not in open("run.py").read()
   ```

3. **Findings 3-6 (DATA DRIFT LOW, needs re-run).**
   Add a `tables/scalars.csv` save in `run.py` capturing `eq_min`, `eq_max`, `omega_mean`, `omega_p10`, `omega_p90`, `no_short_share`, `borrow_share`, `info['error']`, `info['residual']`. Re-run. Cross-check CSV against committed README values. Then the pytest assertions in Findings 3-6 become executable.

4. After fixes, re-run `python run.py` inside `global-dsge/heaton-lucas/`. Re-run this skill on the new code. Target score: <= 10%.
