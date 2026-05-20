# bullshit-detector - keane-wolpin-career-choice - recheck - 2026-05-20

**Bullshit score: 5%** - All four original non-HOLDS findings are now HOLDS. One marginal non-monotonicity in absolute RMSE (dip at age 18: 0.0277 to 0.0275) means the series is not strictly monotone, but the first value (0.0175) is lower than the last (0.0862), the prose "grows with age" is directionally correct, and no result changes. Worst-reviewer nitpick only; no finding.

## Header

- Claim sources: `structural-econometrics/keane-wolpin-career-choice/README.md`
- Code / artifact root: `structural-econometrics/keane-wolpin-career-choice/run.py`
- Data artifacts: `tables/emax-diagnostics.csv`, `tables/emax-fit.csv`, `tables/lifecycle-moments.csv`
- Seed audit: `structural-econometrics/keane-wolpin-career-choice/bullshit-detector_keane-wolpin-career-choice_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | All wage/utility formulas (5 payoff functions) | HOLDS | - | - |
| 2 | Emax formula + Euler constant | HOLDS | - | - |
| 3 | N_s = 2,310 pre-terminal states | HOLDS | - | - |
| 4 | max normalized RMSE = 12.3%, min policy agreement = 90.0% | HOLDS | - | - |
| 5 | All calibration parameters (beta, sigma_e, gamma_E, horizon, M, lambda, N_workers) | HOLDS | - | - |
| 6 | Terminal value formula | HOLDS | - | - |
| 7 | "Absolute Emax RMSE grows with age because later ages have more states and exceed the sample cap" | HOLDS | - | - |
| 8 | "normalized error is largest at young ages" | HOLDS | - | - |
| 9 | "Early ages have fewer states than the cap, so every state is sampled and the regression fit is near-exact. Later ages exceed the cap" | HOLDS | - | - |
| 10 | phi(s,t) = 12 terms; in-code normalization disclosed | HOLDS | - | - |
| 11 | Empty-path feature_matrix returns (0, 12) | HOLDS | - | - |
| 12 | All lifecycle moments match data artifact | HOLDS | - | - |

## Findings

### Finding 1 (original) — "Later ages have fewer continuation states" [RESOLVED]

**Previous category:** FALSE / HIGH

**Resolution:** `run.py:844-849` now reads:

```python
description=(
    "The Emax regression uses at most the sample cap at each age. Early ages "
    "have fewer states than the cap, so every state is sampled and the "
    "regression fit is near-exact. Later ages exceed the cap, so only a "
    "subset is sampled and the in-sample fit error grows."
),
```

Data confirms direction: age 16 has 1 state (sampled=1); age 29 has 525 states (sampled=260). Cap binds at ages 26-29 (states: 282, 354, 435, 525). README.md:319 matches. **HOLDS.**

---

### Finding 2 (original) — "Approximation errors are largest at young ages" [RESOLVED]

**Previous category:** DILUTED / MED

**Resolution:** `run.py:780-791` now reads:

```python
description=(
    "The exact solve provides a benchmark. Absolute Emax RMSE grows with age "
    "because later ages have more states and exceed the sample cap. Relative "
    "to the exact Emax standard deviation at each age, the normalized error is "
    "largest at young ages, where the Emax surface has little spread. ..."
),
```

Data confirms both sub-claims:
- Absolute RMSE: 0.0175 (age 16) to 0.0862 (age 29). One dip at age 18 (0.0277 to 0.0275) means the series is not strictly monotone, but the overall trend is rising and the directional claim is defensible.
- Normalized RMSE: max 0.123 at age 17 (first non-NaN age). README.md:294 matches. **HOLDS.**

**Marginal note on non-monotonicity:** RMSE at age 17 = 0.0277, age 18 = 0.0275. Delta = -0.0002. This is not a false claim — the prose says "grows with age" not "is strictly monotone." No finding.

---

### Finding 3 (original) — phi(s,t) documented as raw state variables, code uses normalized features [RESOLVED]

**Previous category:** DILUTED / MED

**Resolution:** `run.py:561-566` now reads:

```python
written above in raw state units for readability. In code each input is
first normalized to a comparable scale before the polynomial terms are
formed: schooling as $(E-E_0)/(\bar E-E_0)$, each experience stock divided
by the horizon, and age divided by the maximum age. The fitted coefficient
vector $\widehat b_t$ therefore lives in normalized units, so evaluating the
surface requires applying the same normalization to any new state.
```

README.md:134-139 matches verbatim. Normalization fully disclosed. Code at `run.py:187-192` confirmed: `schooling = (arr[:, 0] - p.initial_schooling) / (p.max_schooling - p.initial_schooling)`, etc. **HOLDS.**

---

### Finding 4 (original) — np.empty fallback shape (0, 11) inconsistent with 12-column output [RESOLVED]

**Previous category:** DATA DRIFT / LOW

**Resolution:** `run.py:186` now reads:

```python
return np.empty((0, 12))
```

Verified: `feature_matrix([], 0, CareerPrimitives()).shape == (0, 12)`. Both empty-path and populated-path agree on 12 columns. **HOLDS.**

---

## Cross-cutting patterns

- All four original findings were prose/documentation gaps (two prose direction errors, one undisclosed normalization, one off-by-one in a dead code path). None changed a published number.
- All four are now resolved. No new structural gaps found in this recheck.
- The RMSE series has a single dip at age 18 (not strictly monotone) but the prose does not claim strict monotonicity. The min is at age 16, the max is at age 29. No finding.
- Lifecycle moments in `tables/lifecycle-moments.csv` match README.md:345-354 exactly (re-verified from artifact).
- N_s = 2,310 confirmed by code execution. The MODEL SETUP table at README.md:168 shows 2,310; `build_reachable_states` returns sum 2310 for pre-terminal ages 16-29.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5%.** Well below the 25% ship-with-touch-up threshold. No action required before shipping.

1. All four violated-invariant tests from the original audit now FAIL on the fixed code (the bugs they described are gone). All four honest-fix pass conditions now PASS. The test suite confirms this: 8/8 tests pass.

2. The one marginal observation (RMSE dip at age 18) does not meet the threshold for a new finding. If a future author wants to tighten the prose, they could write "grows on average" or "rises from youngest to oldest age" — but neither is required for correctness.

3. No further action needed. Re-running `python run.py` would regenerate all outputs consistently with the current code.
