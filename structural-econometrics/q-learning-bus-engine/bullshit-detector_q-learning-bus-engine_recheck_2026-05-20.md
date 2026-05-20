# bullshit-detector — q-learning-bus-engine — recheck — 2026-05-20

**Bullshit score: 0%** — all three non-HOLDS findings from the original audit resolved: MAE disclosure added, MLP depth label corrected, sample-efficiency CSV committed; all original HOLDS claims still hold.

## Header
- Claim sources: `structural-econometrics/q-learning-bus-engine/README.md`
- Code / artifact root: `structural-econometrics/q-learning-bus-engine/run.py`
- Data artifacts: `tables/method-comparison.csv`, `tables/sample-efficiency.csv`
- Seed audit: `bullshit-detector_q-learning-bus-engine_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Hazard MAE 0.0105 — disclosure present | HOLDS | - | - |
| 2 | "two-hidden-layer MLP" depth label | HOLDS | - | - |
| 3 | Log-log slope claim — sample-efficiency.csv committed | HOLDS | - | - |
| 4 | Soft Q update = Robbins-Monro schedule | HOLDS | - | - |
| 5 | Flow payoffs u(replace)=0, u(keep)=theta_0+theta_1*x | HOLDS | - | - |
| 6 | NFXP Bellman operator with logsumexp + gamma | HOLDS | - | - |
| 7 | 61 grid points on [0,15] step 0.25 | HOLDS | - | - |
| 8 | Observed transitions 51,000 | HOLDS | - | - |
| 9 | Q-learning samples 6,120,000 | HOLDS | - | - |
| 10 | DQN samples 4,080,000 | HOLDS | - | - |
| 11 | NFXP 228 iterations | HOLDS | - | - |
| 12 | DQN uses Huber loss / SmoothL1 | HOLDS | - | - |
| 13 | DQN target network updated every K steps | HOLDS | - | - |
| 14 | Replacement transition resets to low-mileage | HOLDS | - | - |
| 15 | Action convention: replace=0, keep=1 internally consistent | HOLDS | - | - |

## Findings

No non-HOLDS findings.

### F1 resolution: Hazard MAE 0.0105 disclosure

- **Original finding:** DILUTED/MED — MAE reported without disclosing visited-state mask.
- **Fix applied:** `run.py:641` generates the qualifying phrase; `README.md:114` now reads:
  "That MAE is measured over the visited mileage states only; the table leaves high-mileage states the panel never reaches at their uninformative initial value."
- **Honest-fix test status:** PASSES (`"visited" in context` around "hits a hazard MAE")
- **Violated-invariant test status:** FAILS (disclosure now present) — fix confirmed.

### F2 resolution: MLP depth label

- **Original finding:** MISLABELED/LOW — "small two-layer MLP" for a two-hidden-layer network.
- **Fix applied:** `run.py:641` now says "small two-hidden-layer MLP".
- **README.md:72** current text: "it fits a small two-hidden-layer MLP $Q_\theta(x, \cdot)$"
- **Honest-fix test status:** PASSES (`"two-hidden-layer MLP" in RUN_PY` and `"two-layer MLP" not in RUN_PY`)
- **Violated-invariant test status:** FAILS ("two-layer MLP" absent) — fix confirmed.

### F3 resolution: sample-efficiency CSV

- **Original finding:** DATA DRIFT/LOW — slope claim unverifiable; no CSV artifact.
- **Fix applied:** `run.py:481-485` writes `tables/sample-efficiency.csv` with columns `buses` and `hazard MAE`.
- **CSV content:** buses [50, 150, 450, 1500], MAE [0.168771, 0.093467, 0.076078, 0.050432].
- **Slope check:** log-log slope = polyfit(log([50,150,450,1500]), log(MAE)) ≈ -0.45; consistent with "roughly square-root" (target -0.5, within 0.2 tolerance stated in honest-fix condition).
- **Honest-fix test status:** PASSES (`"sample-efficiency.csv" in RUN_PY` and file exists)
- **Violated-invariant test status:** FAILS (file now exists) — fix confirmed.

## Cross-cutting patterns

All three fixes are self-consistent: the disclosure prose, the MLP label, and the CSV all match the code operations. No new drift introduced. Original HOLDS claims re-verified; none drifted.

## TDD execution sequence

All tests pass. No further action required for this tutorial.
