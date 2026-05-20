# bullshit-detector - q-learning-growth - 2026-05-20

**Bullshit score: 25%** - Two non-HOLDS findings: DILUTED (policy MAE computed on interior-only states, unlabeled) and MISLABELED (VFI "samples" column conflates deterministic sweep-evaluations with stochastic draws). No FALSE or UNIMPLEMENTED findings. Score sits at the upper edge of the 10-30% band because the DILUTED finding touches the headline numbers in the comparison table.

## Header

- Claim sources: `dynamic-programming/q-learning-growth/README.md` (Overview, Equations, Model Setup, Solution Method, Results)
- Code / artifact root: `dynamic-programming/q-learning-growth/run.py`
- Data artifacts: `dynamic-programming/q-learning-growth/tables/algorithm-comparison.csv`
- Seed audit (if any): None
- Run by: bullshit-detector skill (claude-sonnet-4-6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | policy MAE values are over the full state space | DILUTED | MED | yes (reported 0.0038/0.0154/0.0299 are interior-only; full-grid values higher) |
| 2 | comparison table "samples" column is commensurable across solvers | MISLABELED | MED | no (column is present and arithmetically correct; the label misrepresents VFI's nature) |
| 3 | Bellman update Q(s,a) <- Q(s,a) + alpha_t[r + beta max Q(s',a') - Q(s,a)] | HOLDS | - | - |
| 4 | step size is Robbins-Monro 1/n_{s,a}^0.6 | HOLDS | - | - |
| 5 | closed-form policy k'(k,z) = alpha*beta*z*A*k^alpha | HOLDS | - | - |
| 6 | grid and parameter values in Model Setup table | HOLDS | - | - |
| 7 | DQN uses two-layer MLP, replay buffer, Huber loss, target network | HOLDS | - | - |
| 8 | Q-learning uses no transition matrix analytically | HOLDS | - | - |
| 9 | Q-learning 6,000,000 total samples across 4 seeds | HOLDS | - | - |
| 10 | VFI converges in 361 sweeps | HOLDS | - | - |

## Findings

### Finding 1: policy MAE reported without boundary exclusion disclosure

- **Claim source (verbatim):** "Q-learning hits a policy MAE of 0.0154 after 6,000,000 sampled transitions across 4 seeds. DQN reaches 0.0299 after 250,000 steps." - `README.md:105`
- **Claim source (table header):** "policy MAE" column with values 0.0038, 0.0154, 0.0299 - `README.md:99-103` and `tables/algorithm-comparison.csv:2-4`
- **Code evidence (verbatim):**
  ```python
  interior_mask = np.ones_like(closed_form_kp, dtype=bool)
  interior_mask[:3] = False
  interior_mask[-3:] = False

  def policy_mae(policy: np.ndarray) -> float:
      return float(np.mean(np.abs(policy - closed_form_kp)[interior_mask]))

  vfi_mae = policy_mae(policy_kp_vfi)
  ql_mae = policy_mae(policy_kp_ql)
  dqn_mae = policy_mae(dqn_kp) if dqn_kp is not None else float("nan")
  ```
  `run.py:465-474`
- **Data evidence:** `tables/algorithm-comparison.csv` rows confirm policy MAE values 0.0038 / 0.0154 / 0.0299. These numbers are live-computed via `policy_mae()`, which applies `interior_mask`. The mask removes the 3 lowest and 3 highest k-grid rows (6 of 41 rows = 14.6% of state pairs, or 42 of 287 (k,z) pairs excluded).
- **Category:** DILUTED
- **Severity:** MED
- **Result-changing:** yes - the reported MAE numbers are interior-only. Full-grid MAE values are higher (boundary k states are where the closed-form policy k'(k,z)=alpha*beta*z*A*k^alpha can push next-period capital outside the discrete action grid, which is precisely where the Q-table and VFI are under the most pressure). The three published numbers (0.0038, 0.0154, 0.0299) are understated relative to a full-grid evaluation. The within-table comparison between solvers remains apples-to-apples because all three use the identical mask; only the absolute values are inflated.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not np.all(interior_mask)  # PASSES on current code (boundary rows excluded); FAILS if mask is all-True (honest full-grid)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "interior" in report_text or "boundary" in report_text  # table caption or results prose must disclose boundary exclusion
  ```

---

### Finding 2: VFI "samples" column conflates deterministic evaluations with stochastic draws

- **Claim source (verbatim):** table column "samples" with value 2,175,747 for value iteration - `README.md:99-103` and `tables/algorithm-comparison.csv:2`
- **Code evidence (verbatim):**
  ```python
  rows = [
      {
          "algorithm": "value iteration",
          "transition matrix": "yes",
          "policy MAE": round(vfi_mae, 4),
          "value sup-norm vs VFI": 0.0,
          "samples": int(vfi_info["iterations"]) * int(N_K * N_Z * N_A),
          "runtime sec": round(vfi_info["runtime"], 3),
      },
  ```
  `run.py:477-485`
- **Data evidence:** `tables/algorithm-comparison.csv:2` shows `samples = 2175747`. Arithmetic: 361 iterations * 41 * 7 * 21 = 2,175,747. The number is correct. The label "samples" is not.
- **Category:** MISLABELED
- **Severity:** MED
- **Result-changing:** no - the number itself is arithmetically correct and the runtime and MAE columns are unaffected. A reader comparing "2.175M samples (VFI)" vs "6M samples (Q-learning)" may incorrectly infer that VFI is cheaper per sample or uses stochastic draws, but the economic conclusion of the tutorial (Q-learning matches VFI without the transition matrix) is not altered.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "samples" in comparison_df.columns and comparison_df.loc[comparison_df["algorithm"] == "value iteration", "samples"].iloc[0] == 2175747
  # PASSES on current code; FAILS only if the column is renamed or formula changed
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "state-action evaluations" in comparison_df.columns or "sweeps" in comparison_df.columns  # VFI row uses a column name that does not imply stochastic sampling
  ```

## Cross-cutting patterns

- The interior-mask pattern is applied identically to all three solvers (VFI, Q-learning, DQN), so the intra-table ranking is not distorted. The only casualty is the absolute MAE values, which are presented without a boundary-exclusion footnote anywhere in the README or table caption.
- The "samples" mislabeling is a naming/framing issue, not a numerical one. VFI's entry in that column is deterministically derived from `iterations * N_K * N_Z * N_A`. The Q-learning and DQN entries are genuine stochastic sample counts. A reader cannot detect the conceptual difference from the table alone.
- No parametric leak, no missing algorithm component, no false convergence claim. The core Q-learning loop, Bellman update, step size schedule, exploration strategy, and closed-form benchmark are all faithfully implemented.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Below the 50% halt threshold. Forward work is clear; fix the two findings before the next paper-facing summary.

1. **Finding 1 - violated invariant test:**
   ```python
   # tests/test_q_learning_growth.py
   import numpy as np
   import sys; sys.path.insert(0, "dynamic-programming/q-learning-growth")
   import run as qrun
   # Prove boundary rows ARE excluded from MAE
   k_grid, z_grid, _ = qrun.build_grids()
   cf = qrun.closed_form_policy(k_grid[:, None], z_grid[None, :])
   mask = np.ones_like(cf, dtype=bool); mask[:3] = False; mask[-3:] = False
   assert not np.all(mask)  # PASSES now; FAILS on honest full-grid fix
   ```

2. **Finding 1 - honest-fix pass condition test:**
   ```python
   # After fix: README or table caption contains boundary-exclusion disclosure
   readme = open("dynamic-programming/q-learning-growth/README.md").read()
   assert "interior" in readme.lower() or "boundary" in readme.lower()
   # FAILS now (neither word appears); PASSES after fix
   ```

3. **Finding 2 - violated invariant test:**
   ```python
   import pandas as pd
   df = pd.read_csv("dynamic-programming/q-learning-growth/tables/algorithm-comparison.csv")
   assert "samples" in df.columns  # PASSES now; FAILS if column renamed
   ```

4. **Finding 2 - honest-fix pass condition test:**
   ```python
   df = pd.read_csv("dynamic-programming/q-learning-growth/tables/algorithm-comparison.csv")
   assert "samples" not in df.columns or "evaluations" in df.columns  # FAILS now; PASSES after rename
   ```

5. Fix sequence: (a) add a parenthetical to the `policy MAE` column header or table caption noting "interior grid points only (boundary 3 rows excluded)"; (b) rename the `samples` column to `state-action evaluations` for the VFI row, or split the column into two (one for stochastic samples, one for deterministic evaluations). Re-run `python run.py` and confirm CSV and README regenerate with the updated labels.

6. Re-run this skill after fixes. Expected new score: 0-10%.
