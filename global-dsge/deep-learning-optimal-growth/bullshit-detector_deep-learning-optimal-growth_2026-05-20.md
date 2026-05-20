# bullshit-detector -- deep-learning-optimal-growth -- 2026-05-20

**Bullshit score: 20%** -- one DILUTED finding: the loss actually minimized is `mean(r^2) + 1e-3*stability_penalty`, not the pure `Xi_n(theta) = (1/n) sum r_i^2` defined in Equations and Pseudocode; all other claims hold against code and CSV.

## Header
- Claim sources: `global-dsge/deep-learning-optimal-growth/README.md`
- Code / artifact root: `global-dsge/deep-learning-optimal-growth/run.py`
- Data artifacts: `global-dsge/deep-learning-optimal-growth/tables/training-summary.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Loss minimized is pure `Xi_n = mean(r^2)` | DILUTED | MED | no (holdout metrics unaffected) |
| 2 | Training curve drops "several orders of magnitude" | DATA DRIFT | LOW | needs re-run to verify (initial loss not in CSV) |
| 3 | Mean saving share = 0.3420 (neural) | DATA DRIFT | LOW | needs re-run to verify (not stored in CSV) |
| 4 | All numeric table values | HOLDS | -- | -- |
| 5 | Euler residual formula | HOLDS | -- | -- |
| 6 | MLP architecture 1-16-16-1 tanh | HOLDS | -- | -- |
| 7 | Saving-share parameterization | HOLDS | -- | -- |
| 8 | Exact policy formula | HOLDS | -- | -- |
| 9 | Steady-state values k_ss=0.5524, c_ss=1.0629 | HOLDS | -- | -- |
| 10 | Training interval [0.138, 1.381] | HOLDS | -- | -- |
| 11 | Adam bias correction | HOLDS | -- | -- |
| 12 | Audit not part of training loss | HOLDS | -- | -- |

## Findings

### Finding 1: Loss function is Xi_n + stability penalty, not pure Xi_n

- **Claim source (verbatim):** "Set Xi_n(theta) = (1/n) sum_i r_i^2" -- `README.md:152` (pseudocode block); also "The population risk is $\Xi(\theta) = E[r(k;\theta)^2]$... The program replaces that expectation with simulated capital draws" -- `README.md:53-63`.

- **Code evidence (verbatim):**
  ```python
  residual = euler_log_residual(params, k_batch, k_ss)
  kp = neural_policy(params, k_batch, k_ss)
  lower_guard = jax.nn.relu(0.5 * k_min - kp) / k_ss
  upper_guard = jax.nn.relu(kp - 1.15 * k_max) / k_ss
  return jnp.mean(residual**2) + 1e-3 * jnp.mean(lower_guard**2 + upper_guard**2)
  ```
  `run.py:133-137`

- **Data evidence:** The `Final loss` column in `tables/training-summary.csv` records `2.3155900308324817e-08`. This is the penalty-inclusive loss value (logged at `run.py:271`), not the pure Euler-residual mean. During normal training when `kp` is within bounds the penalty terms are zero via `relu`, so the logged value approximates `mean(r^2)`, but the claim that the minimized object IS `Xi_n` as defined is not exactly true.

- **Category:** DILUTED -- the code minimizes a superset of the stated objective; the Euler residual component is present but the stability guard is a load-bearing extra term not mentioned anywhere in Equations, Pseudocode, or Solution Method prose.

- **Severity:** MED -- the guard weight is `1e-3` (small), the guard fires only outside `[0.5*k_min, 1.15*k_max]`, and the holdout metrics (policy error, Euler residual) are computed independently from the loss on a separate grid. Published table values are not distorted. But the tutorial teaches the algorithm as pure Euler-residual minimization, and a reader replicating from pseudocode alone would not reproduce the actual training objective.

- **Result-changing:** no -- the policy-error column (`4.44e-05`) and Euler-residual column (`2.83e-04`) in the table are evaluated on `k_grid` (`run.py:248, 256-258`) using `euler_log_residual` and `neural_policy` directly, not from the loss. The stability guard does not enter those computations.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "lower_guard" in inspect.getsource(run.loss_fn)
  # PASSES on current code (guard present); FAILS on honest fix that removes/documents it
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "stability" in open("README.md").read() or "lower_guard" in open("README.md").read()
  # PASSES if the guard is disclosed in README; FAILS on current code where README omits it
  ```

---

### Finding 2: "Both series drop by several orders of magnitude" (training curves)

- **Claim source (verbatim):** "Both series drop by several orders of magnitude." -- `README.md:167`

- **Code evidence:** `train_log` records `loss` and `mean_policy_error` at steps 1, 200, 400, ..., 6000 (`run.py:202-208`). Neither the initial-step loss nor the initial policy error is stored in `tables/training-summary.csv` -- only the final values appear.

- **Data evidence:** `tables/training-summary.csv` contains only one row: final values. Initial loss is not recorded. The ratio `initial_loss / final_loss` cannot be computed from committed artifacts. **Needs re-run to verify.**

- **Category:** DATA DRIFT -- the claim is plausible given the final loss magnitude (`2.3e-08`) but is not grounded in any committed artifact.

- **Severity:** LOW -- qualitative direction claim about the training process; does not affect the quantitative results table.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "initial_loss" in pd.read_csv("tables/training-summary.csv").columns
  # PASSES if initial loss is recorded; FAILS on current CSV (only final logged)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  df = pd.read_csv("tables/training-summary.csv"); assert df["initial_loss"].iloc[0] / df["Final loss"].iloc[0] > 100
  # PASSES if ratio confirms "several orders of magnitude"; FAILS without the column
  ```

---

### Finding 3: "Its mean is 0.3420" (neural saving share)

- **Claim source (verbatim):** "The learned saving share is nearly flat. Its mean is 0.3420. The exact saving share is $\alpha\beta=0.3420$." -- `README.md:189`

- **Code evidence:** `mean_saving_share = float(share_grid.mean())` where `share_grid = saving_share(params, k_grid, k_ss)` (`run.py:249, 259`). This value depends on the random seed (`PRNGKey(2026)`) and stochastic minibatch draws. It is computed at runtime and emitted into the README text.

- **Data evidence:** `tables/training-summary.csv` does not contain a `Mean saving share` column. The value `0.3420` in the README is not recoverable from committed CSV artifacts without re-running. **Needs re-run to verify.**

- **Category:** DATA DRIFT -- the README prose contains a number not backed by any committed data artifact. It matches `alpha*beta=0.342` analytically, but the actual neural mean share is stochastic and commit-dependent.

- **Severity:** LOW -- the claim is economically interpretive (the neural share converges near the exact saving share); it does not affect the quantitative accuracy metrics.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "Mean saving share" in pd.read_csv("tables/training-summary.csv").columns
  # PASSES if the value is committed; FAILS on current CSV
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  df = pd.read_csv("tables/training-summary.csv"); assert abs(df["Mean saving share"].iloc[0] - 0.342) < 0.001
  # PASSES once the column is added and matches the claim
  ```

## Cross-cutting patterns

- The one structural gap (Finding 1) is a documentation omission, not a code error: the stability guard is an undisclosed regularization term. This pattern -- adding a small auxiliary penalty to aid training convergence without documenting it -- is common in deep-learning macro implementations and easy to miss in audit.
- Findings 2 and 3 share the same root: the CSV artifact captures only final-step scalars. Any prose claim about training dynamics or intermediate values cannot be grounded from committed artifacts alone. A simple fix would be to add `initial_loss` and `mean_saving_share` columns to the summary CSV.
- No parametric-access leaks (no analogue of the Arifovic GA bug). The neural policy receives only `k` and `k_ss`; closed-form values are used only in the audit, not in training.
- No mislabeled methods: the tutorial calls the method Euler-residual minimization and that is exactly what `euler_log_residual` implements.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20%.** Below the 50% halt threshold. Proceed, but address findings before the next tutorial release.

1. **Finding 1 (DILUTED, MED):**
   - Write a test confirming `"lower_guard" in inspect.getsource(run.loss_fn)` PASSES (proves guard is present).
   - Fix options: (a) disclose the stability guard in a single sentence in Solution Method prose and in the pseudocode line (change `Set Xi_n(theta) = (1/n) sum_i r_i^2` to include the penalty term), or (b) remove the guard if training is stable without it. Do not remove without verifying convergence.

2. **Finding 2 (DATA DRIFT, LOW):**
   - Add `initial_loss` and `initial_policy_error` to the `summary` DataFrame in `run.py` (`main()`, around line 262). Confirm both appear in the CSV after re-run.
   - Test: `assert "initial_loss" in pd.read_csv("tables/training-summary.csv").columns`.

3. **Finding 3 (DATA DRIFT, LOW):**
   - Add `mean_saving_share` to the `summary` DataFrame in `run.py`. Confirm it appears in the CSV.
   - Test: `assert "Mean saving share" in pd.read_csv("tables/training-summary.csv").columns`.

4. After fixes, re-run `python run.py` inside the folder. Confirm `README.md` and `tables/training-summary.csv` regenerate cleanly. Re-run this skill to confirm score drops to <= 10%.
