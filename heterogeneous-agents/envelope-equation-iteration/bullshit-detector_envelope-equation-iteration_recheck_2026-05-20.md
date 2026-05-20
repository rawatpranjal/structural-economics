# bullshit-detector — envelope-equation-iteration — recheck — 2026-05-20

**Bullshit score: 0%** — all five non-HOLDS findings from the original audit resolved: title corrected to IID, metric mismatch disclosed, MPC and mean-assets precision unified to 4 decimal places, timing-claim contradiction removed; all original HOLDS claims still hold.

## Header
- Claim sources: `heterogeneous-agents/envelope-equation-iteration/README.md`
- Code / artifact root: `heterogeneous-agents/envelope-equation-iteration/run.py`
- Data artifacts: `heterogeneous-agents/envelope-equation-iteration/tables/solution-statistics.csv`
- Seed audit: `bullshit-detector_envelope-equation-iteration_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Title: "IID Income" — correct | HOLDS | - | - |
| 2 | Convergence figure: metric difference disclosed | HOLDS | - | - |
| 3 | MPC precision — single format 0.2197 | HOLDS | - | - |
| 4 | Mean assets precision — single format 0.4124 | HOLDS | - | - |
| 5 | Takeaway: no bare timing claim | HOLDS | - | - |

## Findings

No non-HOLDS findings.

### F1 resolution: Title corrected from "Persistent Income" to "IID Income"

- **Original finding:** MISLABELED/HIGH — title said "Persistent Income"; code implements IID.
- **Fix applied:** `run.py:432`: `"Buffer-Stock Saving with IID Income by Envelope-Equation Iteration"`
- **README.md:1** current text: `# Buffer-Stock Saving with IID Income by Envelope-Equation Iteration`
- **Violated-invariant test status:** FAILS (`"Persistent" not in README.splitlines()[0]`) — fix confirmed.
- **Honest-fix test status:** PASSES.

### F2 resolution: Convergence metric mismatch disclosed

- **Original finding:** DILUTED/MED — convergence figure compared value-level (VFI) and consumption-level (EEI/EGP) errors without disclosure.
- **Fix applied:** `run.py:690-698` convergence figure description now reads:
  "Both update policies through the Euler equation and track a consumption-level sup-norm error. Grid VFI updates the value level and tracks a value-level error, which is intrinsically larger-scaled, so VFI needs more iterations to cross the same absolute tolerance. The iteration counts are read against different error metrics, so this is a fixed-point comparison and not a clean iteration race or a timing claim."
- **README.md:150** contains "value-level error".
- **Violated-invariant test status:** FAILS (`"value-level error" in README`) — fix confirmed.
- **Honest-fix test status:** PASSES.

### F3 resolution: MPC precision unified

- **Original finding:** DATA DRIFT/LOW — prose used `:.3f` ("0.220") while table used `:.4f` ("0.2197").
- **Fix applied:** `run.py:665`: `f"The borrowing-limit mass raises the average MPC to {mean_mpc:.4f}."` and `run.py:730`: `f"{mean_mpc:.4f}"`.
- **README.md:146** current text: "The borrowing-limit mass raises the average MPC to 0.2197."
- **Violated-invariant test status:** FAILS (both "0.220" and "0.2197" not simultaneously present) — fix confirmed.
- **Honest-fix test status:** PASSES.

### F4 resolution: Mean assets precision unified

- **Original finding:** DATA DRIFT/LOW — prose used `:.2f` ("0.41") while table used `:.4f` ("0.4124").
- **Fix applied:** `run.py:662`: `f"Mean assets are {mean_assets:.4f}. "` and `run.py:728`: `f"{mean_assets:.4f}"`.
- **README.md:146** current text: "Mean assets are 0.4124."
- **Violated-invariant test status:** FAILS ("Mean assets are 0.41." absent) — fix confirmed.
- **Honest-fix test status:** PASSES.

### F5 resolution: Timing claim contradiction removed

- **Original finding:** DILUTED/LOW — Takeaway said "EGP is faster here" while Solution Method said "not a timing claim".
- **Fix applied:** `run.py:748-761` Takeaway now says "EGP replaces the inner bisection with one analytic marginal-utility inverse per state, so each iteration does less work than the EEI Euler step." The phrase "EGP is faster here" is absent. The phrase "not a timing claim" is retained only in the convergence figure description (correctly qualified by the metric-disclosure context from F2 fix).
- **Violated-invariant test status:** FAILS (not both "not a timing claim" AND "EGP is faster here" in README) — fix confirmed.
- **Honest-fix test status:** PASSES.

## Cross-cutting patterns

All five fixes address prose precision and labeling; no economic computation changed. The code's IID income process, envelope condition, and bisection Euler step are unchanged and correct throughout. The F2 and F5 fixes are coupled: the convergence figure now honestly characterizes both the metric difference and the fixed-point nature of the comparison, while the Takeaway makes only a factual claim about per-iteration cost (analytic inverse vs bisection) rather than an unverified wall-clock assertion.

## TDD execution sequence

All 10 tests pass (5 violated-invariants fail on fixed code = expected; 5 honest-fix tests pass). No further action required for this tutorial.
