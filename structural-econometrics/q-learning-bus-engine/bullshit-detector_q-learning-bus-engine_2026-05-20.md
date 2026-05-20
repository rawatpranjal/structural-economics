# bullshit-detector — q-learning-bus-engine — 2026-05-20

**Bullshit score: 25%** — One DILUTED finding (masked MAE reported without disclosure) skews the headline number in the results table; a MISLABELED/LOW finding on MLP depth; one unverifiable slope claim tagged needs-re-run-to-verify. No FALSE or UNIMPLEMENTED findings. Worst reader quotes "0.0105 MAE" as a full-grid number when it excludes unvisited high-mileage states where the table predicts 0.5 against a truth near 1.0.

## Header
- Claim sources: `structural-econometrics/q-learning-bus-engine/README.md` (Equations, Model Setup, Solution Method, Results, Takeaway sections)
- Code / artifact root: `structural-econometrics/q-learning-bus-engine/run.py`
- Data artifacts: `structural-econometrics/q-learning-bus-engine/tables/method-comparison.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 / bullshit-detector skill
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Hazard MAE 0.0105 over full mileage range | DILUTED | MED | yes (masked subset vs full grid) |
| 2 | Small two-layer MLP Q_theta | MISLABELED | LOW | no |
| 3 | Log-log slope roughly square-root | DATA DRIFT | LOW | needs re-run to verify |
| 4 | Soft Q update = Robbins-Monro schedule | HOLDS | — | — |
| 5 | Flow payoffs u(replace)=0, u(keep)=theta_0+theta_1*x | HOLDS | — | — |
| 6 | NFXP Bellman operator with logsumexp + gamma | HOLDS | — | — |
| 7 | 61 grid points on [0,15] step 0.25 | HOLDS | — | — |
| 8 | Observed transitions 51,000 (1500 buses * 34 periods) | HOLDS | — | — |
| 9 | Q-learning samples 6,120,000 | HOLDS | — | — |
| 10 | DQN samples 4,080,000 | HOLDS | — | — |
| 11 | NFXP 228 iterations | HOLDS | — | — |
| 12 | DQN uses Huber loss / SmoothL1 | HOLDS | — | — |
| 13 | DQN target network updated every K steps | HOLDS | — | — |
| 14 | Replacement transition resets to low-mileage | HOLDS | — | — |
| 15 | Action convention: replace=0, keep=1 internally consistent | HOLDS | — | — |

## Findings

### Finding 1: Hazard MAE 0.0105 is a masked partial-grid number, reported as if full-grid

- **Claim source (verbatim):** "NFXP converges in 228 Bellman iterations. Soft Q-learning hits a hazard MAE of 0.0105 after 30 passes through 51,000 observed transitions. Soft DQN reaches 0.0046 on the same panel." — `README.md:114`

  Also in table: `tables/method-comparison.csv:3` — `soft Q-learning (4 seeds avg.),no,0.0105,...`

- **Code evidence (verbatim):**
  ```python
  visit_min = visits_per_action.min(axis=1)
  visit_total = visits_per_action.sum(axis=1)
  ...
  visited_mask = visit_min >= 5  # require both actions to have visits
  hazard_mae_ql = float(np.mean(
      np.abs(p_replace_ql[visited_mask] - nfxp["p_replace"][visited_mask])
  ))
  ```
  `run.py:463-488`

  Also: `visits_per_action` is captured from seed 0 only:
  ```python
  if visits_per_action is None:
      visits_per_action = ql_info["visits"].copy()
  ```
  `run.py:461-462`

- **Data evidence:** `tables/method-comparison.csv:3` reports `hazard MAE` of `0.0105` with no footnote that this excludes unvisited mileage states. The README Results section says "Past that range the table has no data to update" (referring to the figure) but does not disclose that the 0.0105 number itself excludes those states.

- **Category:** DILUTED — the code correctly computes MAE over visited states and the figure masks unvisited regions, but the prose presents 0.0105 as a standalone performance number without disclosing the coverage restriction. A reader comparing against another method's full-grid MAE is comparing unlike quantities. The unvisited high-mileage states have `q=0` so `p_replace = 0.5` (softmax of two equal values) while NFXP truth approaches 1.0; those states would each contribute ~0.5 error. The full-grid MAE is substantially higher than 0.0105.

- **Severity:** MED — the number in the table is misleading under the implicit comparison interpretation, but the figure and surrounding prose gesture at the coverage limit. A careful reader can reconstruct the masking.

- **Result-changing:** yes — a reader benchmarking soft Q-learning against any other method using a full-grid MAE would see an unfavorable gap. The 0.0105 figure cannot be directly compared to any full-grid evaluation number.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert np.sum(visit_min_seed0 >= 5) < len(x_grid)  # mask excludes some states; 0.0105 is not full-grid
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "visited states" in readme_results_text or "(masked" in readme_results_text  # disclosure present
  ```

---

### Finding 2: "Small two-layer MLP" has two hidden layers, which is three-layer by the Goodfellow convention

- **Claim source (verbatim):** "it fits a small two-layer MLP $Q_\theta(x, \cdot)$ that maps mileage to the two action values" — `README.md:72`

- **Code evidence (verbatim):**
  ```python
  class QNet(nn.Module):
      def __init__(self) -> None:
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(1, DQN_HIDDEN),
              nn.Tanh(),
              nn.Linear(DQN_HIDDEN, DQN_HIDDEN),
              nn.Tanh(),
              nn.Linear(DQN_HIDDEN, 2),
          )
  ```
  `run.py:262-271`

- **Data evidence:** Not applicable (architecture claim, no numeric artifact).

- **Category:** MISLABELED — the network has two hidden layers (`Linear(1,64)` and `Linear(64,64)`) plus an output layer, totaling three weight-bearing layers. Under the Goodfellow/Bishop convention, "two-layer MLP" means one hidden layer plus output. Under the common RL-paper convention, "two-layer" counts hidden layers only, which would match. The ambiguity is real; the claim is not FALSE. However, "small two-layer MLP" as a size descriptor undersells the actual depth.

- **Severity:** LOW — terminology ambiguity only; the architecture is fully specified in code and does not affect any result.

- **Result-changing:** no

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert sum(isinstance(m, torch.nn.Linear) for m in QNet().net) == 2  # fails: actual count is 3
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert sum(isinstance(m, torch.nn.Linear) for m in QNet().net) == 3  # three weight layers present
  ```

---

### Finding 3: "Log-log slope is roughly square-root" — slope unverifiable without re-run

- **Claim source (verbatim):** "Hazard recovery improves as more buses enter the panel. The log-log slope is roughly square-root. That rate matches the standard sample-complexity scaling of off-policy evaluation." — `README.md:96`

- **Code evidence (verbatim):**
  ```python
  bus_counts = [50, 150, 450, 1500]
  hazard_mae_per_count: list[float] = []
  for nb in bus_counts:
      sub = simulate_buses(...)
      _, info = soft_q_learning(sub_trans, flow, BETA, QL_EPOCHS, seed=999)
      hazard_mae_per_count.append(float(info["log_hazard_mae"][-1]))
  ```
  `run.py:473-479`

- **Data evidence:** No `tables/sample-efficiency.csv` artifact committed. The MAE values for each bus count are computed at runtime and written only to a figure (`figures/sample-efficiency.png`). The slope claim is a visual assertion about the figure that cannot be verified from committed artifacts.

- **Category:** DATA DRIFT — the code correctly generates the figure; the slope claim is about the figure's visual content which is not archived in a data artifact. The claim may be accurate but is unverifiable from committed files.

- **Severity:** LOW — slope characterization is qualitative and pedagogically framed ("roughly"). Does not affect any result number in the table.

- **Result-changing:** needs re-run to verify

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert Path("tables/sample-efficiency.csv").exists()  # fails: file not committed
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  slope = np.polyfit(np.log(bus_counts), np.log(hazard_mae_per_count), 1)[0]; assert abs(slope - (-0.5)) < 0.2
  ```

---

## Cross-cutting patterns

- The single DILUTED finding (masked MAE) follows a pattern common in RL-meets-econometrics tutorials: the figure masks unvisited regions correctly, but the scalar summary number in the results table and prose inherits the same restriction without saying so. Any reader skimming the table would miss the asterisk.
- No parametric knowledge leaks were found (contrast with the Arifovic GA canonical finding). The Q-learning agent receives only `(x_t, a_t, x_{t+1})` triples and the `flow` array (which encodes `THETA_TRUE`). The flow array is the analog of a known utility specification, not a structural parameter being estimated — its use is legitimate and consistent with the stated premise ("Soft Q-learning replaces the matrix with the simulated bus panel. The agent sees only the data the econometrician uses for estimation.").
- The action-index flip (`a_idx = 1 - a_t`) in `build_panel_transitions` is internally consistent with `flow_payoffs` throughout, but the inline comment "1 = replace, 0 = keep" on the line immediately before the flip (`run.py:218-220`) describes the *input* convention. A reader who reads only that comment and not the flip line could infer the wrong convention. This is a code-clarity issue, not a claim-vs-code gap, and is out of scope for this skill.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Below the 50% halt threshold. Proceed with targeted fixes.

1. **Finding 1 — disclose masked MAE.**
   Turn the violated invariant into a test:
   ```python
   def test_masked_mae_not_disclosed():
       readme = Path("structural-econometrics/q-learning-bus-engine/README.md").read_text()
       # The 0.0105 figure appears; check that a disclosure qualifier also appears nearby
       idx = readme.index("0.0105")
       context = readme[max(0, idx-200):idx+200]
       assert "visited" in context.lower() or "masked" in context.lower() or "coverage" in context.lower()
   # This test FAILS on current README (no disclosure near the number).
   ```
   Then add "(over visited mileage states)" or equivalent qualifier in `run.py:724-726` where `closing` is assembled.

2. **Finding 2 — clarify MLP depth.**
   Test:
   ```python
   import torch.nn as nn
   from run import QNet  # if importable
   net = QNet()
   assert sum(isinstance(m, nn.Linear) for m in net.net) == 3  # already passes
   ```
   Fix: change "two-layer MLP" to "two-hidden-layer MLP" in `run.py:636` and regenerate README.

3. **Finding 3 — archive sample-efficiency numbers.**
   Test:
   ```python
   assert Path("structural-econometrics/q-learning-bus-engine/tables/sample-efficiency.csv").exists()
   ```
   Fix: write `bus_counts` and `hazard_mae_per_count` to `tables/sample-efficiency.csv` before the sample-efficiency figure is produced, then re-run to verify the slope claim.

4. After fixes: re-run `python run.py` inside the tutorial folder, re-run `scripts/validate_catalog.py`, then re-run this skill. Target score: 0-10%.
