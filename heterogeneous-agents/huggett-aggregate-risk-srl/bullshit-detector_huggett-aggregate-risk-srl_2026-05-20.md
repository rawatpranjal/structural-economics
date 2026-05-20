# bullshit-detector — huggett-aggregate-risk-srl — 2026-05-20

**Bullshit score: 55%** — two result-changing FALSE findings: (1) "smoother aggregate consumption" asserted in Takeaway when C/Y volatility ratio = 1.000 (not smoother); (2) "Grid and training settings: Matched" hardcoded in benchmark table when QUICK_PROFILE deviates from FULL_PROFILE on every grid dimension.

## Header
- Claim sources: `heterogeneous-agents/huggett-aggregate-risk-srl/README.md` (Overview, Equations, Model Setup, Results, Takeaway prose and tables)
- Code / artifact root: `heterogeneous-agents/huggett-aggregate-risk-srl/run.py`
- Data artifacts: `tables/diagnostics.csv`, `tables/hyperparameters.csv`, `tables/paper_benchmark.csv`
- Seed audit (if any): None
- Run by: bullshit-detector skill (Claude Sonnet 4.6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Takeaway: "smoother aggregate consumption" | FALSE | HIGH | yes — Takeaway contradicts the run's own C/Y ratio = 1.000 |
| 2 | Benchmark table: "Grid and training settings: Matched" | FALSE | HIGH | yes — QUICK_PROFILE differs from FULL_PROFILE on every grid/training dimension |
| 3 | Diagnostics table: "Convergence threshold 0.0003" | DATA DRIFT | MED | yes — training ran at 1e-4 (QUICK), table reports 3e-4 (FULL) |
| 4 | Benchmark "Market-clearing residual: Matched" vs 4.4e-6 | MISLABELED | MED | no — residual is algebraically zero by construction, not a meaningful equilibrium metric |
| 5 | Objective = Monte Carlo estimate of expected lifetime utility | DILUTED | LOW | no — code adds L2 regularizer `-1e-5 * mean(theta^2)` not mentioned in README |

## Findings

### Finding 1: Takeaway claims "smoother aggregate consumption" when C/Y ratio = 1.000

- **Claim source (verbatim):** "The tutorial reproduces the paper benchmark's calibration and grid settings, then checks the same economic objects: concave consumption, endogenous prices, smoother aggregate consumption, and a near-zero market-clearing residual." — `README.md:178`
- **Code evidence (verbatim):**
  ```python
  consumption_income_volatility = float(np.std(consumption) / max(np.std(income), 1.0e-12))
  ```
  `run.py:708`

  ```python
  volatility_ratio < 1.0,
  ```
  `run.py:756` — this is one of the `qualitative_passes` tests; if ratio >= 1.0, it fails.
- **Data evidence:** `tables/diagnostics.csv:14` — `Aggregate consumption volatility divided by income volatility,1`. `tables/paper_benchmark.csv:6` — C/Y volatility ratio 1.000; Assessment: Mixed.
- **Category:** FALSE
- **Severity:** HIGH
- **Result-changing:** yes — the Takeaway asserts a qualitative property ("smoother") that the run's own diagnostics disprove (ratio = 1.000, not < 1). A reader trusting the Takeaway believes the model reproduces the canonical smoothing result when it does not.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "smoother aggregate consumption" in open("README.md").read()  # PASSES on buggy code
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "smoother aggregate consumption" not in open("README.md").read() or float(open("tables/diagnostics.csv").read().split("income volatility,")[1].strip()) < 1.0  # PASSES on honest fix
  ```

---

### Finding 2: Benchmark table "Grid and training settings: Matched" is hardcoded and false for QUICK_PROFILE

- **Claim source (verbatim):** "Grid and training settings | ... | Uses the same grid, horizon, learning-rate schedule, and batch size | Matched" — `README.md:171`
- **Code evidence (verbatim):**
  ```python
  {
      "Benchmark item": "Grid and training settings",
      "Published SRL benchmark": (
          "200 bond points, b_max=50, 3 income states, 30 aggregate states, "
          "20 rate points on [0.01, 0.06], T=170, 1000 epochs, "
          "50 warm-up epochs, lr_ini=1e-3, lr_decay=0.5, batch=512"
      ),
      "Tutorial run": "Uses the same grid, horizon, learning-rate schedule, and batch size",
      "Assessment": "Matched",
  },
  ```
  `run.py:779-788` — the "Tutorial run" string and "Matched" assessment are hardcoded literals; they do not branch on which profile is active.
- **Data evidence:** `tables/hyperparameters.csv` rows confirm QUICK_PROFILE deviates on every dimension: asset_points 56 vs 200; asset_upper 12 vs 50; aggregate_states 7 vs 30; rate_points 9 vs 20; horizon 44 vs 170; epochs 90 vs 1000; warmup_epochs 12 vs 50; batch_size 14 vs 512; initial_learning_rate 0.045 vs 0.001; learning_rate_decay 0.55 vs 0.5; convergence_tol 1e-4 vs 3e-4.
- **Category:** FALSE
- **Severity:** HIGH
- **Result-changing:** yes — the benchmark comparison table tells the reader the tutorial run used the published grid and training settings when it used a heavily reduced QUICK profile. A reader comparing results across profile runs would be misled about what was actually solved.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "Uses the same grid, horizon, learning-rate schedule, and batch size" in open("tables/paper_benchmark.csv").read()  # PASSES on buggy code
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "Uses the same grid, horizon, learning-rate schedule, and batch size" not in open("tables/paper_benchmark.csv").read()  # PASSES on honest fix (text must be profile-conditional)
  ```

---

### Finding 3: Diagnostics table reports FULL_PROFILE convergence threshold (3e-4) when training used QUICK_PROFILE threshold (1e-4)

- **Claim source (verbatim):** "Convergence threshold | 0.0003" — `README.md:153` (diagnostics table) and `tables/diagnostics.csv:5`
- **Code evidence (verbatim):**
  ```python
  ("Convergence threshold", f"{FULL_PROFILE.convergence_tol:.6g}"),
  ```
  `run.py:713` — `diagnostics_table` hardcodes `FULL_PROFILE.convergence_tol`.

  ```python
  if epoch > profile.warmup_epochs and movement < profile.convergence_tol:
  ```
  `run.py:479` — training loop uses `profile.convergence_tol` (= 1e-4 for QUICK_PROFILE).
- **Data evidence:** `tables/hyperparameters.csv` row "Convergence threshold" shows Tutorial setting = 0.0001 (QUICK), but `tables/diagnostics.csv` row shows 0.0003 (FULL). The two tables disagree about which threshold was active.
- **Category:** DATA DRIFT
- **Severity:** MED
- **Result-changing:** yes — a reader using the diagnostics table to understand why the run did not converge would compute the wrong stopping criterion. Final movement 0.0273 is far above both thresholds, so convergence status is unaffected, but the reported threshold is wrong.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "FULL_PROFILE.convergence_tol" in open("run.py").read()  # PASSES on buggy code
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "FULL_PROFILE.convergence_tol" not in open("run.py").read()  # PASSES on honest fix
  ```

---

### Finding 4: Market-clearing residual 9.78e-18 labeled "Matched" against published 4.4e-6 — residual is algebraically zero by construction

- **Claim source (verbatim):** "Market-clearing residual | Average bond-market clearing gap about 4.4e-6 | Mean absolute residual 9.78e-18; maximum 5.55e-17; bracketing share 1.000 | Matched" — `README.md:173`
- **Code evidence (verbatim):**
  ```python
  weight_hi = float(np.clip(-demand[idx] / denominator, 0.0, 1.0))
  weights[idx] = 1.0 - weight_hi
  weights[idx + 1] = weight_hi
  ...
  residual = float(np.sum(weights * demand))
  ```
  `run.py:580-587` — `residual = (1 - weight_hi)*demand[idx] + weight_hi*demand[idx+1]`. Substituting `weight_hi = -demand[idx]/denominator` gives `residual = demand[idx]*demand[idx+1]/denominator - demand[idx]*demand[idx+1]/denominator = 0` exactly. The near-zero residual is a tautology of the linear interpolation formula, not an equilibrium property.
- **Data evidence:** `tables/diagnostics.csv:8` — `Mean interpolated market-clearing residual,9.77753e-18`. This is floating-point zero, not a meaningful convergence metric.
- **Category:** MISLABELED
- **Severity:** MED
- **Result-changing:** no — the residual is accurately reported as ~1e-17; the mislabeling is calling it "Matched" against 4.4e-6, which implies the two numbers measure the same thing when they do not.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "Matched" in open("tables/paper_benchmark.csv").read().split("Market-clearing residual")[1].split("\n")[0]  # PASSES on buggy code
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "Matched" not in open("tables/paper_benchmark.csv").read().split("Market-clearing residual")[1].split("\n")[0]  # PASSES on honest fix (label must note tautology)
  ```

---

### Finding 5: Objective equation omits L2 regularization term present in code

- **Claim source (verbatim):** "The Structural Reinforcement Learning objective is a Monte Carlo estimate of expected lifetime utility: $$J(\theta) = \mathbb{E}[\sum_{t=0}^{T-1}\beta^t u(c_t)]$$" — `README.md:29-32`
- **Code evidence (verbatim):**
  ```python
  objective_value = jnp.mean(returns) - 1.0e-5 * jnp.mean(theta**2)
  ```
  `run.py:336` — the actual objective includes a L2 penalty on parameters not present in the stated equation.
- **Data evidence:** None required; gap is code vs equation.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no — the L2 coefficient is 1e-5, small relative to utility magnitudes (~0.02), so the policy is not materially distorted. The omission is a documentation gap, not a result-changing error.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "1.0e-5 * jnp.mean(theta**2)" in open("run.py").read()  # PASSES on buggy code
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert r"1.0\times 10^{-5}" in open("run.py").read() or "regulariz" in open("README.md").read()  # PASSES on honest fix
  ```

---

## Cross-cutting patterns

- **Hardcoded prose for profile-conditional behavior.** Findings 1, 2, and 3 all stem from strings or values being hardcoded to the FULL_PROFILE context inside functions that receive the active profile as an argument. The `paper_benchmark_table` function receives `train_info` and `hard_sim` (computed from the active profile) but emits a "Tutorial run" string that is factually true only for FULL_PROFILE. The `diagnostics_table` function receives `train_info` but ignores the active profile's `convergence_tol`. Any future profile addition will silently produce the same wrong table rows.

- **Assessment column is partly hardcoded, partly computed.** "Calibration" and "Grid and training settings" rows have hardcoded `"Assessment": "Matched"`. "Convergence status", "Market-clearing residual", and "Qualitative figure match" rows compute the assessment at runtime. The hybrid approach means the computed rows correctly surface the QUICK run's failures (convergence: Not met; qualitative: Mixed) while the hardcoded rows claim success regardless of which profile ran.

- **Tautological residual vs meaningful residual.** The interpolated market-clearing residual is algebraically zero by construction (Finding 4), but the training-phase "soft market residual" (0.101) is a meaningful quantity. The README presents both without distinguishing that one is tautological. Readers comparing the 9.78e-18 hard residual against the published 4.4e-6 gap will draw incorrect inferences about solution quality.

## TDD execution sequence (for the next agent)

0. **Bullshit score = 55%. Halt code work; surface findings 1 and 2 to the user before writing fixes.**

1. For each non-HOLDS finding, turn the **violated invariant** into a pytest test under `tests/`. Confirm each test PASSES on current code (proves the bug is real).

2. Convert the **honest-fix pass condition** into a second pytest test that FAILS on current code.

3. **Priority fix order:**
   - Finding 2 (FALSE/HIGH): make `paper_benchmark_table` branch on whether the active profile matches FULL_PROFILE; emit a "Reduced (quick) grid" label and "Not matched" assessment when it does not.
   - Finding 1 (FALSE/HIGH): make the Takeaway prose conditional on `volatility_ratio < 1.0`; do not assert "smoother" when the ratio is >= 1.
   - Finding 3 (DATA DRIFT/MED): replace `FULL_PROFILE.convergence_tol` at `run.py:713` with the active `profile.convergence_tol` passed to `diagnostics_table`.
   - Finding 4 (MISLABELED/MED): relabel the market-clearing residual assessment row; note that the interpolated residual is zero by construction of the linear interpolation formula and is not comparable to the published 4.4e-6 gap metric.
   - Finding 5 (DILUTED/LOW): add a sentence to the Equations section noting the L2 regularization term.

4. After fixes, re-run `python run.py` (QUICK profile). Re-run this skill on the new output to confirm all findings now read HOLDS and the score is <= 25%.
