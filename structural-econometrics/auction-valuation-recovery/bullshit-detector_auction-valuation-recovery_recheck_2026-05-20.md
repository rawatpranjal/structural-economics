# bullshit-detector - auction-valuation-recovery - recheck - 2026-05-20

**Bullshit score: 0%** - Both prior findings verified fixed. All numeric, algorithmic, and formula claims HOLD against code and both CSV artifacts.

## Header
- Claim sources: `structural-econometrics/auction-valuation-recovery/README.md`
- Code / artifact root: `structural-econometrics/auction-valuation-recovery/run.py`
- Data artifacts: `tables/recovery-diagnostics.csv`, `tables/value-recovery-summary.csv`
- Seed audit: `structural-econometrics/auction-valuation-recovery/bullshit-detector_auction-valuation-recovery_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Value distribution Beta(2,5) on [0,1] | HOLDS | - | - |
| 2 | n_auctions=3,000 | HOLDS | - | - |
| 3 | n_bidders=4 | HOLDS | - | - |
| 4 | Observed bids=12,000 | HOLDS | - | - |
| 5 | Kept bids=10,800 | HOLDS | - | - |
| 6 | Trimmed share=0.1 | HOLDS | - | - |
| 7 | RMSE=0.001, MAE=0.001, Correlation=1 | HOLDS | - | - |
| 8 | Boundary trim 5% in each tail | HOLDS | - | - |
| 9 | GPV inversion hat v = b + G(b)/[(n-1)g(b)] | HOLDS | - | - |
| 10 | hat G 1-indexed: smallest maps to 1/N (prior F1, fixed) | HOLDS | - | - |
| 11 | Trimmed share denominator is bid count not auction count (prior F2, fixed) | HOLDS | - | - |
| 12 | Values hidden during recovery | HOLDS | - | - |
| 13 | KDE for g-hat, empirical ranks for G-hat | HOLDS | - | - |
| 14 | Equilibrium bid via BNE integration formula | HOLDS | - | - |
| 15 | True/recovered value summary matches CSVs | HOLDS | - | - |

## Findings

None.

**Prior Finding 1 resolved (DILUTED: pseudocode rank convention unspecified).** `run.py:280-281` now reads: `"3. Estimate \\hat G(b_i) = rank(b_i) / N, with rank(b_i) 1-indexed so the smallest bid maps to 1/N and the largest to 1."` `README.md:103` matches. The `empirical_cdf_at_sample` function at `run.py:64-67` uses `np.arange(1, len(x) + 1)` confirming 1-indexed convention. The pseudocode now precisely describes what the code does.

**Prior Finding 2 resolved (DILUTED: variable named `auctions` was bid-level).** `run.py:142` assigns the bid-level DataFrame to `all_bids`. `run.py:161` computes `round(1.0 - len(data) / len(all_bids), 3)`. The denominator `len(all_bids) = 12000` is the bid count. The ambiguous name `auctions` is gone; `all_bids` unambiguously signals bid-level semantics.

## Cross-cutting patterns

- Both DILUTED findings were documentation/naming gaps. Neither affected any committed numeric artifact. Both resolved in one pass.
- No parametric information leaks: `recover_pseudo_values` at `run.py:70-89` receives only `bids`, `n_bidders`, `trim_quantile`. The `beta` distribution object and `value` column are never passed to or accessed by the inversion function.
- All five numeric claims in the Results tables (RMSE, MAE, Correlation, Kept bids, Trimmed share) are internally consistent between README and both CSV files.
- The equilibrium bid schedule at `run.py:22-37` correctly implements the BNE formula via `cumulative_trapezoid(F(v)^{n-1}, v)`.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%. No action required.**
1. All 4 tests at `tests/test_auction-valuation-recovery.py` pass: 2 violated-invariant + 2 honest-fix tests. All pass means fixes applied and correct.
2. No sim re-runs or data artifact changes needed. Both fixes were documentation/naming only.
