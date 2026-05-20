# bullshit-detector — schelling-segregation — 2026-05-20

**Bullshit score: 5%** — all claims hold; one cosmetic DATA DRIFT between the tau=1/3 display label and its true floating-point value (0.333... rendered as "0.333") is the only non-trivial observation, and it is produced intentionally by the table-formatting code.

## Header
- Claim sources: `agent-based-models/schelling-segregation/README.md` (prose, Equations, Model Setup, Solution Method, Results sections)
- Code / artifact root: `agent-based-models/schelling-segregation/run.py`
- Data artifacts: `agent-based-models/schelling-segregation/tables/threshold-sweep.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, 2026-05-20)
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Isolated agent gets s_i=1 (acceptable) | HOLDS | none | no |
| 2 | 10% vacancy, equal group sizes (50/50) | HOLDS | none | no |
| 3 | Moore neighborhood, up to 8 cells | HOLDS | none | no |
| 4 | Random visit order; re-check satisfaction before each move | HOLDS | none | no |
| 5 | Uniform random draw from C_i(t) (satisfying vacancies) | HOLDS | none | no |
| 6 | Grid update (X_t, E_t) before visiting next agent | HOLDS | none | no |
| 7 | Moves keep M fixed (swap occupied for vacant) | HOLDS | none | no |
| 8 | Three stop conditions (D_t empty / n_moved=0 / t+1=T) | HOLDS | none | no |
| 9 | tau range 0.20-0.50; 5 replications per threshold | HOLDS | none | no |
| 10 | Four thresholds in path plot | HOLDS | none | no |
| 11 | CSV numbers match README table | HOLDS | none | no |
| 12 | tau=1/3 displayed as "0.333" in table | DATA DRIFT | LOW | no |
| 13 | S(t) "near one half" for random initial city | HOLDS | none | no |
| 14 | tau=1/3 final S < tau=1/2 final S | HOLDS | none | no |

## Findings

### Finding 1: tau=1/3 displayed as "0.333" in threshold-sweep table

- **Claim source (verbatim):** "Near the one-third region, the same rule raises same-group exposure sharply." — `README.md:111`; table row shows tau label "0.333" — `README.md:138`
- **Code evidence (verbatim):**
  ```python
  tau_grid = np.array([0.20, 0.225, 0.25, 0.275, 0.30, 1.0 / 3.0, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50])
  ```
  `run.py:390`
  ```python
  table["Threshold tau"] = table["Threshold tau"].map(lambda x: f"{x:.3f}")
  ```
  `run.py:438`
- **Data evidence:** `tables/threshold-sweep.csv:7` shows row `0.333,0.752,0.014,7.8,531,5,5`. The actual threshold used in computation is `1.0/3.0 = 0.33333...`; the CSV label is a 3-decimal-place rendering artifact.
- **Category:** DATA DRIFT — the internal float `0.33333...` and the displayed label `0.333` describe the same run, but a reader computing `S` at exactly `tau=0.333` (not `1/3`) would get a marginally different answer. The code uses the exact rational approximation; the table suggests a rounded decimal.
- **Severity:** LOW — the discrepancy is 0.00033..., sub-threshold for any agent decision on a finite grid. The plot correctly marks `1/3` with `ax.axvline(1.0/3.0, ...)`.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert float(pd.read_csv("tables/threshold-sweep.csv")["Threshold tau"].iloc[5]) == pytest.approx(1.0/3.0, abs=1e-10)
  # PASSES on current code (0.333 != 1/3 to 10dp), FAILS on honest fix (exact 1/3 label)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(float(pd.read_csv("tables/threshold-sweep.csv")["Threshold tau"].iloc[5]) - 1.0/3.0) < 1e-10
  # PASSES on honest fix (store full precision), FAILS on current code (3dp rounding)
  ```

## Cross-cutting patterns

- All algorithmic claims in the Equations and Solution Method sections are faithfully implemented. The isolation convention (s_i=1), the Moore neighborhood, the random-order agent visitation, the grid update within the iteration, and all three stop conditions match code exactly.
- The "draw e from C_i(t)" pseudocode phrasing (uniform random draw from satisfying vacancies) is implemented as shuffle-all-vacancies-then-take-first, which is mathematically equivalent to a uniform draw from the satisfying subset. No fidelity gap.
- The only non-trivial observation is the 3-decimal rounding of `1/3` in the table label. This is a display artifact, not a computation error. The phase-transition plot correctly uses the exact `1.0/3.0` float for the reference line.
- CSV numbers match README table verbatim across all 13 rows and 7 columns. No stale-data risk.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 5%.** Below the 25% ship-after-touchup threshold. No halt required.
1. Finding 1 is LOW severity and non-result-changing. The only actionable fix is to store the tau label at full floating-point precision in the CSV (e.g., `f"{x:.10f}"` or the raw float) and update the README table accordingly. This is cosmetic.
2. No other findings require test coverage or code changes.
3. If the tau label fix is applied, re-run `python run.py` inside the tutorial folder and confirm the CSV row 6 now reads `0.3333333333` (or similar) and the README table reflects it.
4. Re-run `scripts/validate_catalog.py` to confirm no math-rendering regressions.
