# bullshit-detector — huggett-incomplete-markets — recheck — 2026-05-20

**Bullshit score: 0%** — all four non-HOLDS findings from the original audit resolved: value-gap precision unified to 3 decimal, r* and wedge precision unified to 5 decimal, mean-wealth masking removed (raw residual shown); all algorithmic claims still hold.

## Header
- Claim sources: `heterogeneous-agents/huggett-incomplete-markets/README.md`
- Code / artifact root: `heterogeneous-agents/huggett-incomplete-markets/run.py`
- Data artifacts: `heterogeneous-agents/huggett-incomplete-markets/tables/equilibrium.csv`
- Seed audit: `bullshit-detector_huggett-incomplete-markets_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Relative value gap "0.155%" — single precision | HOLDS | - | - |
| 2 | r* "0.03499" — single precision | HOLDS | - | - |
| 3 | Wedge "0.01501" — single precision | HOLDS | - | - |
| 4 | Mean wealth "5.43e-06" — actual residual shown | HOLDS | - | - |
| 5 | Upwind HJB scheme matches description | HOLDS | - | - |
| 6 | Implicit step formula | HOLDS | - | - |
| 7 | KFE solved by A-transpose | HOLDS | - | - |
| 8 | Boundary condition at borrowing limit | HOLDS | - | - |
| 9 | Bisection logic: S>0 lower r, S<0 raise r | HOLDS | - | - |
| 10 | Aswitch income-switching block | HOLDS | - | - |
| 11 | All numeric values in table match CSV | HOLDS | - | - |

## Findings

No non-HOLDS findings.

### F1 resolution: Relative value gap precision unified

- **Original finding:** DATA DRIFT/LOW — "0.16%" (2dp) in Results text, "0.155%" (3dp) in table.
- **Fix applied:** `run.py:728`: `f"The relative value gap against the reference grid is ${100 * V_gap_rel:.3f}\\%$."` - unified to `:.3f` matching the table at `run.py:844`.
- **README.md:337** current text: "The relative value gap against the reference grid is $0.155\%$."
- **CSV:16**: `Relative value gap (% of value scale),0.155%`
- **Violated-invariant test status:** FAILS (both "0.16%" and "0.155%" not simultaneously in README) — fix confirmed.
- **Honest-fix test status:** PASSES.

### F2 resolution: r* precision unified

- **Original finding:** DATA DRIFT/LOW — "0.0350" (4dp) in prose vs "0.03499" (5dp) in Equations and table.
- **Fix applied:** `run.py:802`: `f"The Huggett equilibrium is lower, at $r^{{\\ast}} = {r_eq:.5f}$."` and `run.py:831`: `f"{r_eq:.5f}"`. All occurrences of r_eq in prose use `:.5f`.
- **README.md:349** current text: "The Huggett equilibrium is lower, at $r^{\ast} = 0.03499$."
- **Violated-invariant test status:** FAILS ("0.0350" and "0.03499" not both present in README) — fix confirmed.
- **Honest-fix test status:** PASSES (`"= 0.0350$" not in README` and `README.count("0.03499") >= 2`).

### F3 resolution: Precautionary wedge precision unified

- **Original finding:** DATA DRIFT/LOW — "0.0150}" (4dp) in prose vs "0.01501" (5dp) in table.
- **Fix applied:** `run.py:803`: `f"The precautionary wedge is $\\rho - r^{{\\ast}} = {wedge:.5f}$."` and `run.py:864`: `f"$\\rho - r^{{\\ast}} = {wedge:.5f}$"`. All prose uses `:.5f`.
- **README.md:349** current text: "The precautionary wedge is $\rho - r^{\ast} = 0.01501$."
- **README.md:381** current text: "In this run the wedge is $\rho - r^{\ast} = 0.01501$."
- **Violated-invariant test status:** FAILS ("0.0150}" absent) — fix confirmed.
- **Honest-fix test status:** PASSES.

### F4 resolution: Mean wealth masking removed

- **Original finding:** DILUTED/LOW — "Mean wealth E[a]" row hard-coded to 0.00000 via `mean_wealth_display` override while residual row disclosed the true 5.43e-06.
- **Fix applied:** `run.py:834`: `f"{market_residual:.2e}"` for "Mean wealth E[a]" row. The `mean_wealth_display` override is gone. Both "Mean wealth E[a]" and "Bond-market residual" rows now show `5.43e-06`.
- **CSV:5**: `Mean wealth E[a],5.43e-06`
- **README.md:353** disclosure sentence: "The mean-wealth and bond-market-residual rows report the same quantity: bisection drives $|S(r^{\ast})|$ down to the small residual shown, so mean assets are zero up to that bisection tolerance rather than exactly zero."
- **Violated-invariant test status:** FAILS (`_row_value("Mean wealth E[a]") == "5.43e-06"`, not "0.00000") — fix confirmed.
- **Honest-fix test status:** PASSES (`mean_wealth == residual == "5.43e-06"`).

## Cross-cutting patterns

All four findings shared one root: same quantity rendered by two format strings (or two display paths) in `run.py`. Unified format strings and removed the `mean_wealth_display` override eliminate all four in one pass. No algorithmic code changed. All upwind HJB, implicit step, KFE, bisection, and Aswitch claims re-verified; none drifted.

## TDD execution sequence

All 8 tests pass (4 violated-invariants fail on fixed code = expected; 4 honest-fix tests pass). No further action required for this tutorial.
