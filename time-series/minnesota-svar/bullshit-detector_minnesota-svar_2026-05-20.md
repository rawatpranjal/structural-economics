# bullshit-detector — minnesota-svar — 2026-05-20

**Bullshit score: 15%** — all CSV-grounded numeric claims HOLD; two Takeaway numbers (stability radius 0.88→0.76, shrinkage ratio 0.70) are not stored in any committed data artifact and cannot be verified without re-running the script.

## Header

- Claim sources: `time-series/minnesota-svar/README.md` (prose, Equations, Results, Takeaway sections)
- Code / artifact root: `time-series/minnesota-svar/run.py`
- Data artifacts: `time-series/minnesota-svar/tables/forecast-rmse.csv`, `irf-summary.csv`, `coefficient-posteriors.csv`, `prior-hyperparameters.csv`, `shock-identification.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | RMSE values (OLS 0.434, BVAR 0.391) match CSV | HOLDS | — | no |
| 2 | IRF trough output -0.037, inflation -0.018 match CSV | HOLDS | — | no |
| 3 | Coefficient posterior means and intervals match CSV | HOLDS | — | no |
| 4 | Hyperparameter values match dataclass defaults | HOLDS | — | no |
| 5 | Prior variance formula code matches README equations | HOLDS | — | no |
| 6 | Posterior update formula code matches README equations | HOLDS | — | no |
| 7 | Shock scaling formula code matches README equations | HOLDS | — | no |
| 8 | Zero-impact on output/inflation (Cholesky lower-tri) | HOLDS | — | no |
| 9 | Own-lag prior mean indexing correct | HOLDS | — | no |
| 10 | Takeaway: stability radius 0.88→0.76 not in any CSV | DATA DRIFT | LOW | no (Takeaway prose only) |
| 11 | Takeaway: shrinkage ratio 0.70 not in any CSV | DATA DRIFT | LOW | no (Takeaway prose only) |

## Findings

### Finding 1: Stability radius numbers not stored in committed artifacts

- **Claim source (verbatim):** "The stability radius falls from 0.88 under OLS to 0.76 under the Minnesota BVAR." — `README.md:269`
- **Code evidence (verbatim):**
  ```python
  ols_radius = stability_radius(ols["beta"], lag_order, len(VARIABLES))
  bvar_radius = stability_radius(bvar["beta"], lag_order, len(VARIABLES))
  ```
  `run.py:349-350`
  ```python
  f"falls from {ols_radius:.2f} under OLS to {bvar_radius:.2f} under the "
  ```
  `run.py:928`
- **Data evidence:** No CSV table stores `ols_radius` or `bvar_radius`. The values 0.88 and 0.76 appear only in the committed `README.md` prose. Seed is fixed at `seed=2027` (`run.py:53`), so re-running should reproduce these values deterministically, but they cannot be verified against any on-disk artifact without re-running the script.
- **Category:** DATA DRIFT
- **Severity:** LOW — values are Takeaway prose, not cited in Results tables or figures. The seed is deterministic, so the numbers are likely stable across runs.
- **Result-changing:** no — these numbers are interpretive commentary in the Takeaway, not entries in any results table.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert any("stability" in col.lower() for col in pd.read_csv("tables/forecast-rmse.csv").columns)
  # PASSES on current code (table has no stability column), FAILS if a column is added
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(f.name == "stability-metrics.csv" for f in Path("tables").iterdir())
  # PASSES if a stability-metrics.csv is committed, FAILS on current code
  ```

### Finding 2: Shrinkage ratio not stored in committed artifacts

- **Claim source (verbatim):** "the shrinkage lowers the coefficient norm to 0.70 of the OLS norm" — `README.md:269`
- **Code evidence (verbatim):**
  ```python
  shrinkage_ratio = float(
      np.linalg.norm(bvar["beta"][1:, :]) / np.linalg.norm(ols["beta"][1:, :])
  )
  ```
  `run.py:351-353`
  ```python
  f"shrinkage lowers the coefficient norm to {shrinkage_ratio:.2f} of the "
  ```
  `run.py:926`
- **Data evidence:** No CSV stores `shrinkage_ratio`. The value 0.70 appears only in committed `README.md:269`. Seed is deterministic (`seed=2027`); needs re-run to verify.
- **Category:** DATA DRIFT
- **Severity:** LOW — Takeaway prose only, no results table entry.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert any("shrinkage" in col.lower() for col in pd.read_csv("tables/forecast-rmse.csv").columns)
  # PASSES on current code (no such column), FAILS if column added
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(f.name == "stability-metrics.csv" for f in Path("tables").iterdir())
  # PASSES if committed artifact stores shrinkage_ratio, FAILS on current code
  ```

## Cross-cutting patterns

- Both DATA DRIFT findings share the same root: interpretive Takeaway numbers computed in `run.py` but not persisted to any `tables/*.csv` file. A single `tables/stability-metrics.csv` writing `ols_radius`, `bvar_radius`, and `shrinkage_ratio` would close both gaps.
- All five `tables/*.csv` files that ARE committed match their README counterparts exactly to the precision shown (3 decimal places). The CSV-generation code uses `format_float(..., digits=3)` consistently, so the displayed and stored values are identical by construction.
- The simulation seed (`seed=2027`, `run.py:53`) is fixed and passed to `np.random.default_rng`. All numeric outputs are therefore deterministic across re-runs of the same code version. The DATA DRIFT findings are low-severity precisely because of this.
- The IRF scaling formula in the README (`q = tau * P*e3 / (e3'*P*e3)`) and the code are algebraically equivalent for a lower-triangular Cholesky factor: both reduce to `[0, 0, tau]'`. No discrepancy.
- The own-lag prior mean indexing (`own_first_lag = 1 + equation`, `run.py:150`) correctly aligns with the VAR design column ordering because `VARIABLES` indices match the interleaved lag column order.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 15%.** Below the 50% halt threshold. Safe to proceed with minor fixes.

1. For Finding 1 (stability radius): turn the violated invariant into a pytest test under `tests/`. Confirm it PASSES on current code.
2. For Finding 2 (shrinkage ratio): same pattern.
3. Honest-fix path: add a `tables/stability-metrics.csv` write in `run.py:main()` immediately after `run.py:353`, storing `ols_radius`, `bvar_radius`, and `shrinkage_ratio`. Both pass-condition tests should then PASS.
4. Re-run `python scripts/validate_catalog.py` to confirm no catalog regressions.
5. Re-run this skill on the updated tutorial. Expected new score: 0-10%.
