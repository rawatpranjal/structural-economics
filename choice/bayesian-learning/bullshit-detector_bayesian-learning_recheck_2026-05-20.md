# bullshit-detector — bayesian-learning — recheck — 2026-05-20

**Bullshit score: 0%** — All claims hold. The original Finding 1 (stopping horizon T_stop=30 undisclosed) is resolved: Model Setup table now contains a "Stopping horizon | 30" row generated from the live `T_stop` variable. No new findings.

## Header
- Claim sources: `choice/bayesian-learning/README.md`
- Code / artifact root: `choice/bayesian-learning/run.py`
- Seed audit: `bullshit-detector_bayesian-learning_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Signal horizon = 50 | HOLDS | — | — |
| 2 | Stopping horizon = 30 | HOLDS | — | — |
| 3 | Simulated paths = 200 per state | HOLDS | — | — |
| 4 | p_H=0.7, p_L=0.3, prior=0.5 | HOLDS | — | — |
| 5 | Payoffs H=1.0, L=-0.5, reject=0.0 | HOLDS | — | — |
| 6 | Bayes update formula | HOLDS | — | — |
| 7 | Log-odds / sufficient statistic | HOLDS | — | — |
| 8 | Action value A(p) = max[p*pi_H+(1-p)*pi_L, 0] | HOLDS | — | — |
| 9 | Continuation value C_t(p) with Pr(R|p)=p*p_H+(1-p)*p_L | HOLDS | — | — |
| 10 | Recursion V_t(p) = max[A(p), C_t(p)] | HOLDS | — | — |
| 11 | 200 paths solid mean, dashed = exact binomial mean | HOLDS | — | — |

## Findings

### Original Finding 1 (from seed audit): Stopping horizon undisclosed — RESOLVED

- **Original claim (verbatim):** "Signal horizon | 50" was the only horizon row in the Model Setup table; T_stop=30 was never disclosed.
- **Current state:** `README.md:62` now reads `| Stopping horizon | 30 | Periods used for the backward-induction boundary |`. The value is injected via f-string at `run.py:299`: `f"| Stopping horizon | {T_stop} | Periods used for the backward-induction boundary |\n"`. T_stop is assigned at `run.py:219`: `T_stop = 30  # Shorter horizon for stopping problem`.
- **Resolution:** RESOLVED. The table now discloses both horizons. A reader can reproduce the stopping-boundary figure using T_stop=30 as stated.
- **Category change:** DILUTED → HOLDS.

## Cross-cutting patterns

None. All claims verified end-to-end. No parametric leak, no label mismatch, no stale hardcoded numbers.

## TDD execution sequence (for the next agent)

None required. Score is 0%. The honest-fix test (`assert "Stopping horizon" in text and "| 30 |" in text`) now passes. The violated-invariant test (`assert "30" not in README.read_text()`) now correctly fails, confirming the fix is in place.
