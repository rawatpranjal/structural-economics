# bullshit-detector — bayesian-learning — 2026-05-20

**Bullshit score: 25%** — Model Setup table advertises one horizon (T=50) but the stopping-boundary figure (the central results figure) runs on a silent second horizon (T_stop=30); no disclosure anywhere in README. All equation claims hold.

## Header
- Claim sources: `choice/bayesian-learning/README.md`
- Code / artifact root: `choice/bayesian-learning/run.py`
- Seed audit (if any): prior stale version of this file (overwritten)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, fresh pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Signal horizon \| 50" covers stopping problem | DILUTED | MED | yes — stopping figure uses T_stop=30; boundary shape is horizon-dependent |
| 2 | Bayes update formula | HOLDS | — | no |
| 3 | Log-odds / sufficient-statistic equation | HOLDS | — | no |
| 4 | Action value A(p) = max[p*pi_H+(1-p)*pi_L, 0] | HOLDS | — | no |
| 5 | Continuation value C_t(p) with predictive P(R\|p)=p*pH+(1-p)*pL | HOLDS | — | no |
| 6 | Terminal value V_T(p) = max[A(p), 0] | HOLDS | — | no |
| 7 | Backward-induction recursion V_t = max[A(p), C_t(p)] | HOLDS | — | no |
| 8 | Simulated paths: 200 per state | HOLDS | — | no |

## Findings

### Finding 1: "Signal horizon | 50" — Model Setup table omits stopping horizon T_stop=30

- **Claim source (verbatim):** `"| Signal horizon | 50 | Draws used for belief paths |"` — `README.md:61`
- **Code evidence (verbatim):**
  ```python
  T = 50                # Number of signals
  ...
  T_stop = 30  # Shorter horizon for stopping problem
  upper_bounds, lower_bounds = compute_optimal_stopping_boundary(
      T_stop, payoff_invest_H, payoff_invest_L, payoff_wait, p_red_H, p_red_L
  )
  ```
  `run.py:191`, `run.py:219-221`
- **Data evidence:** README Model Setup table has exactly one "Signal horizon" row; value is 50. The stopping-boundary figure caption and alt text (`README.md:99`) name it a "finite-horizon stopping problem" without stating which horizon. No row in the table names T_stop=30. The word "30" never appears in README.md.
- **Category:** DILUTED — the table covers the belief-simulation horizon (T=50) but silently omits the stopping horizon (T_stop=30) used for the primary results figure. A reader attempting to reproduce the stopping-boundary figure with T=50 gets a different figure: the continuation region is wider at early periods and converges differently. The committed figure is internally consistent, but only with T_stop=30, which is undisclosed.
- **Severity:** MED — the stopping-boundary figure is the tutorial's primary computational output (backward induction is the advertised method). The horizon directly shapes the boundary; a one-third reduction from 50 to 30 is not a rounding difference.
- **Result-changing:** yes — reproducing `stopping-boundary.png` from the stated parameters (T=50) gives a different figure than the committed one. The specific boundary shape in the committed figure cannot be recovered from the Model Setup table alone. needs re-run to verify exact magnitude of the visual difference.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "30" not in open("choice/bayesian-learning/README.md").read()
  # PASSES on current README (30 is absent); FAILS after honest fix adds T_stop=30 row
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "30" in open("choice/bayesian-learning/README.md").read()
  # PASSES after fix adds stopping horizon row; FAILS on current README
  ```

## Cross-cutting patterns

- Single horizon-disclosure gap: tutorial silently uses two different T values (T=50 for simulation, T_stop=30 for stopping) and documents only one. Same pattern would appear in any tutorial running multiple sub-models with different horizons but presenting one combined parameter table.
- No parametric leak, no label mismatch, no false algorithmic claim. Bayesian update code (`run.py:27-37`), log-odds formula (`run.py:54-59`), action-value formula (`run.py:117-118`), continuation-value computation (`run.py:150-159`), terminal-value initialization (`run.py:133`), and backward-induction loop (`run.py:142-179`) all match stated equations verbatim.
- No tables CSV present (`tables/` directory absent); all numeric claims in Model Setup are injected from code via f-strings, so DATA DRIFT between code and README is structurally impossible for those values -- except for T_stop, which is a hardcoded local variable not surfaced in any f-string.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Below the 50% halt threshold. Fix is a one-row table addition; surface to user before pushing.
1. **Violated-invariant test:** `assert "30" not in open("choice/bayesian-learning/README.md").read()` -- confirm PASSES on current file (proves the omission).
2. **Honest-fix pass condition test:** `assert "30" in open("choice/bayesian-learning/README.md").read()` -- confirm FAILS on current file.
3. Fix: add a second row to the Model Setup table in `run.py` `add_model_setup(...)` f-string:
   `f"| Stopping horizon | {T_stop} | Periods used for backward-induction boundary |\n"`
   Note: `T_stop` is a local variable in `main()`; the f-string must be inside `main()` after `T_stop` is defined. Regenerate `README.md` via `python run.py`.
4. Re-run both tests; violated-invariant now FAILS, pass-condition now PASSES.
5. Re-run this skill on updated README; expect score 0-10%.
