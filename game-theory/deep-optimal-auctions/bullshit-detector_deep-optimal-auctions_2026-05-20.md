# bullshit-detector — deep-optimal-auctions — 2026-05-20

**Bullshit score: 15%** — one MISLABELED finding (table column "Mean regret" stores `np.max(mean_regrets)`, not the mean); all other claims HOLD against code and committed artifacts. Score at midpoint of the 10–30% band for MISLABELED-only findings.

## Header

- Claim sources: `game-theory/deep-optimal-auctions/README.md` (Overview, Equations, Model Setup, Solution Method, Results, Takeaway)
- Code / artifact root: `game-theory/deep-optimal-auctions/run.py`
- Data artifacts: `game-theory/deep-optimal-auctions/tables/revenue-regret-audit.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Table column "Mean regret" = max over bidders, not mean | MISLABELED | LOW | no (2 symmetric bidders: max ~= mean numerically) |
| 2 | Myerson exact revenue 0.4167 | HOLDS | - | - |
| 3 | Second-price exact revenue 0.3333 | HOLDS | - | - |
| 4 | Myerson optimal reserve r*=0.5 | HOLDS | - | - |
| 5 | Architecture 2-32-32-5 tanh MLP | HOLDS | - | - |
| 6 | Augmented Lagrangian loss formula | HOLDS | - | - |
| 7 | Multiplier update max{0, lambda + rho*rgt} | HOLDS | - | - |
| 8 | Softmax enforces feasibility x1+x2 <= 1 | HOLDS | - | - |
| 9 | IR built into parameterization | HOLDS | - | - |
| 10 | Neural revenue 0.4498 matches committed CSV | HOLDS | - | - |
| 11 | Mean regret 0.0012 matches committed CSV | HOLDS | - | - |
| 12 | Max regret 0.0164 matches committed CSV | HOLDS | - | - |
| 13 | Regret formula max over grid B | HOLDS | - | - |
| 14 | Misreport rivals report true values | HOLDS | - | - |
| 15 | Training steps 5,000 | HOLDS | - | - |
| 16 | Misreport grid 41 points (training), 81 (audit) | HOLDS | - | - |

## Findings

### Finding 1: Table column "Mean regret" stores max-of-means, not mean-of-means

- **Claim source (verbatim):** `"| Neural auction | 0.4498 | 0.0012 | 0.0164 | 0 |"` where the third column header is `"Mean regret"` — `README.md:196`
- **Code evidence (verbatim):**
  ```python
  rows = [
      {
          "Mechanism": "Neural auction",
          "Revenue": f"{float(audit['neural_revenue']):.4f}",
          "Mean regret": f"{float(np.max(mean_regrets)):.4f}",
          "Max regret": f"{float(np.max(max_regrets)):.4f}",
          "Max IR violation": f"{float(audit['ir_violation']):.2e}",
      },
  ```
  `run.py:292-299`

  ```python
  mean_regrets = np.asarray(audit["mean_regrets"], dtype=float)
  ```
  `run.py:290`

  where `audit["mean_regrets"]` is populated at `run.py:265` from `revenue_and_regret` which returns `jnp.array(regrets)` — a per-bidder array where `regrets[i] = jnp.mean(jnp.maximum(gain_i, 0.0))` (`run.py:123`).

- **Data evidence:** `tables/revenue-regret-audit.csv:2` — `Neural auction,0.4498,0.0012,0.0164,0.00e+00`. The value 0.0012 in the "Mean regret" column is `np.max([mean_regret_bidder1, mean_regret_bidder2])`.
- **Category:** MISLABELED — the column computes the maximum over bidders of each bidder's mean regret, but is labelled "Mean regret". The figure y-axis at `run.py:588` correctly says "Largest mean bidder regret"; the table column header does not match.
- **Severity:** LOW
- **Result-changing:** no — with two bidders drawing IID from U[0,1], the per-bidder mean regrets are symmetric and nearly equal, so `max(mean_regrets) ~= mean(mean_regrets)` numerically. The committed value 0.0012 is not wrong; the label misnames what it computes.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "np.max(mean_regrets)" in open("run.py").read() and 'Mean regret' in open("run.py").read()
  # PASSES on current code (max used under a Mean label); FAILS on honest fix (label renamed)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert '"Largest mean bidder regret"' in open("run.py").read() or '"Max mean regret"' in open("run.py").read()
  # PASSES on honest fix (column renamed to match computation); FAILS on current code
  ```

## Cross-cutting patterns

- No cross-cutting pattern. The single MISLABELED finding is isolated to the `audit_table` function's column header string at `run.py:296`. The figure that displays the same quantity (`run.py:588`, y-axis "Largest mean bidder regret") uses the accurate label, so the inconsistency is between the figure and the table — not between the prose and the code.
- All numeric claims in README.md that cite committed CSV values are exact string matches to `tables/revenue-regret-audit.csv`. No DATA DRIFT between prose and artifact.
- The analytical benchmarks (Myerson exact 0.4167, second-price exact 0.3333) are computed from closed-form formulas at `run.py:50-57` and cross-checked correctly against the model-setup table. No drift.
- The augmented Lagrangian, multiplier update, and rho schedule are implemented verbatim to the equations in the README. No dilution.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 15%.** Below the 50% halt threshold. Safe to proceed with a targeted fix.
1. Turn the violated invariant into a pytest test under `tests/`:
   ```python
   def test_mean_regret_column_label_is_accurate():
       src = open("game-theory/deep-optimal-auctions/run.py").read()
       # This test PASSES on current buggy code (proves the mislabeling):
       assert '"Mean regret": f"{float(np.max(mean_regrets)):.4f}"' in src
   ```
2. Turn the honest-fix condition into a second test that FAILS on current code:
   ```python
   def test_mean_regret_column_label_after_fix():
       src = open("game-theory/deep-optimal-auctions/run.py").read()
       # FAILS on current code; PASSES after fix:
       assert '"Largest mean bidder regret": f"{float(np.max(mean_regrets)):.4f}"' in src
   ```
3. Fix: rename the dict key `"Mean regret"` to `"Largest mean bidder regret"` at `run.py:296` and regenerate `README.md` and `tables/revenue-regret-audit.csv`.
4. After fix: test 1 should FAIL (bug gone), test 2 should PASS (honest label present). Re-run `python run.py` and confirm the committed CSV column header updates.
5. Re-run this skill. Expected score: 0-10%.
