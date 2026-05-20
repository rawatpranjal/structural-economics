# bullshit-detector — q-learning-growth — recheck — 2026-05-20

**Bullshit score: 0%** — Both original findings (DILUTED boundary exclusion, MISLABELED "samples" column) are fully remediated. All 15 claims hold exactly against code and CSV.

## Header

- Claim sources: `dynamic-programming/q-learning-growth/README.md`
- Code / artifact root: `dynamic-programming/q-learning-growth/run.py`
- Data artifacts: `dynamic-programming/q-learning-growth/tables/algorithm-comparison.csv`
- Seed audit: `bullshit-detector_q-learning-growth_2026-05-20.md` (2 non-HOLDS findings)
- Run by: bullshit-detector skill (claude-sonnet-4-6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | boundary exclusion disclosed in prose and column header | HOLDS | — | — |
| 2 | evaluation-type column distinguishes deterministic sweeps from stochastic samples | HOLDS | — | — |
| 3 | policy MAE column header is "policy MAE (interior)" | HOLDS | — | — |
| 4 | VFI evaluation type = "deterministic sweeps" in CSV | HOLDS | — | — |
| 5 | Q-learning/DQN evaluation type = "stochastic samples" in CSV | HOLDS | — | — |
| 6 | Q-learning uses no transition matrix analytically | HOLDS | — | — |
| 7 | Bellman update Q(s,a) ← Q(s,a) + α_t[r + β max Q(s',a') − Q(s,a)] | HOLDS | — | — |
| 8 | step size is Robbins-Monro 1/n_{s,a}^0.6 | HOLDS | — | — |
| 9 | closed-form policy k'(k,z) = αβzAk^α | HOLDS | — | — |
| 10 | Q-learning 6,000,000 total samples across 4 seeds | HOLDS | — | — |
| 11 | VFI converges in 361 sweeps | HOLDS | — | — |
| 12 | DQN uses two-layer MLP, replay buffer, Huber loss, target network | HOLDS | — | — |
| 13 | policy MAE values VFI=0.0038, QL=0.0154, DQN=0.0299 | HOLDS | — | — |
| 14 | state-action evaluations: VFI=2,175,747, QL=6,000,000, DQN=250,000 | HOLDS | — | — |
| 15 | CSV headers match README table headers exactly | HOLDS | — | — |

## Findings

None.

## Cross-cutting patterns

- Original Finding 1 (DILUTED, MED): boundary exclusion now disclosed at `README.md:95` in explicit prose ("the three lowest and three highest capital grid rows are excluded") and reflected in the column header "policy MAE (interior)" (`run.py:485`, `tables/algorithm-comparison.csv:1`). The mask code at `run.py:465-470` is unchanged and correct; only the disclosure was missing before.
- Original Finding 2 (MISLABELED, MED): "samples" column renamed to "state-action evaluations" (`run.py:487,496,507`). New "evaluation type" column added with per-algorithm values: "deterministic sweeps" for VFI (`run.py:488`), "stochastic samples" for Q-learning and DQN (`run.py:497,508`). CSV confirms all three rows (`tables/algorithm-comparison.csv:2-4`).
- No new patterns identified. The core algorithm (Q-learning loop, Bellman update, step size schedule, closed-form benchmark, VFI Bellman operator) is unchanged from the original audit and all held then.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** No action required. Both prior findings are resolved.
1. The honest-fix tests in `tests/test_q-learning-growth.py` now pass: `test_finding1_honest_fix` finds "interior" in README; `test_finding2_honest_fix` finds "state-action evaluations" and "evaluation type" columns with correct VFI row value.
2. The violated-invariant tests now fail as expected: `test_finding2_violated_invariant` fails because "samples" is no longer a column name — this failure is correct (the bug is fixed).
3. No further action needed. Re-run `scripts/validate_catalog.py` to confirm no math-rendering regressions from the column header change.
