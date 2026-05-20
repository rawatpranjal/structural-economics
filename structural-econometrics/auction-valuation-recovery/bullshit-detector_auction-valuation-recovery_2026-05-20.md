# bullshit-detector — auction-valuation-recovery — 2026-05-20

**Bullshit score: 30%** — one DILUTED/MED finding (empirical-CDF normalizer off-by-one vs. pseudocode claim) plus one DILUTED/MED finding (trimmed-share formula inconsistency between README table and code); no FALSE; no result-changing errors on published magnitudes.

## Header
- Claim sources: `structural-econometrics/auction-valuation-recovery/README.md`
- Code / artifact root: `structural-econometrics/auction-valuation-recovery/run.py`
- Data artifacts: `tables/recovery-diagnostics.csv`, `tables/value-recovery-summary.csv`
- Seed audit (if any): None
- Run by: bullshit-detector skill (claude-sonnet-4-6)
- Date: 2026-05-20
- Diagram-only cap applied: no

---

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | `hat G(b_i) = rank(b_i) / N` (pseudocode step 3) | DILUTED | MED | no (tiny bias, not visible at reported precision) |
| 2 | "Trimmed share = 0.1" matches "5% in each tail" | DILUTED | MED | no (consistent at reported precision, but formula is wrong-sign) |
| 3 | Bid shading: bids shifted left of values | HOLDS | — | — |
| 4 | `n_auctions=3000`, `n_bidders=4` | HOLDS | — | — |
| 5 | `Observed bids = 12000` | HOLDS | — | — |
| 6 | `Kept bids = 10800` | HOLDS | — | — |
| 7 | `RMSE = 0.001`, `MAE = 0.001`, `Correlation = 1` | HOLDS | — | — |
| 8 | True values: Mean=0.28, Std=0.132, P10=0.112, Median=0.265, P90=0.473 | HOLDS | — | — |
| 9 | Recovered: Mean=0.28, Std=0.131, P10=0.112, Median=0.264, P90=0.473 | HOLDS | — | — |
| 10 | GPV inversion formula `hat v = b + G(b)/[(n-1)g(b)]` | HOLDS | — | — |
| 11 | Beta(2,5) value distribution | HOLDS | — | — |
| 12 | Values hidden during recovery; estimator sees only bids+n | HOLDS | — | — |
| 13 | KDE for `hat g`, empirical ranks for `hat G` | HOLDS | — | — |
| 14 | Boundary trim 5% each tail | HOLDS | — | — |
| 15 | Equilibrium bid via ODE/integration formula | HOLDS | — | — |

---

## Findings

### Finding 1: `hat G(b_i) = rank(b_i) / N` — pseudocode uses a biased normalizer

- **Claim source (verbatim):**
  > "3. Estimate `\hat G(b_i) = rank(b_i) / N`."
  > — `README.md:103`

- **Code evidence (verbatim):**
  ```python
  order = np.argsort(x, kind="mergesort")
  ranks = np.empty_like(order, dtype=float)
  ranks[order] = np.arange(1, len(x) + 1)
  return ranks / len(x)
  ```
  `run.py:64-67`

- **Data evidence:** The pseudocode in `README.md:103` writes `rank(b_i) / N`. The code computes `ranks / len(x)` where ranks run from 1 to N (1-indexed). So for the maximum bid the code returns `N/N = 1.0`, not `(N)/N = 1.0` — that specific value matches. But the pseudocode claim says "rank" (which by convention in the GPV literature is usually `(rank - 0.5)/N` or `rank/(N+1)` to avoid boundary mass at 1). The code uses `rank/N` with 1-indexed ranks, so the maximum bid maps to CDF=1.0 exactly. This pushes the empirical CDF to 1 at the sample maximum and makes the pseudo-value formula blow up (`G/(g*(n-1))` with G=1 at boundary). The trim step saves it in practice but the pseudocode claim `rank(b_i)/N` does not distinguish whether rank is 0-indexed or 1-indexed. The code uses 1-indexed, which is the standard choice but differs from the most natural reading of `rank(b_i)/N` as a 0-based rank divided by N (which would give 0 for the minimum and (N-1)/N for the maximum). This is a documentation precision gap: the pseudocode is ambiguous, the code resolves it one way (1-indexed), and the trim prevents any numerical consequence.

- **Category:** DILUTED — pseudocode step 3 does not state the rank convention; reader cannot tell if CDF evaluates to 0 at minimum or 1/N at minimum without reading code.

- **Severity:** MED

- **Result-changing:** no — trim at 5%/95% excises the boundary cases where the convention matters; reported RMSE=0.001 and Correlation=1 are unaffected.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert empirical_cdf_at_sample(np.array([1.0, 2.0, 3.0]))[0] == 1/3  # 1-indexed: min maps to 1/N not 0
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "rank(b_i)" in open("README.md").read() and "(1-indexed)" in open("README.md").read()
  ```

---

### Finding 2: Trimmed share formula — code computes wrong-direction fraction, agrees with README only by accident

- **Claim source (verbatim):**
  > "| Boundary trim | 5% in each tail | Avoids unstable density estimates |"
  > — `README.md:91`
  >
  > "| Trimmed share | 0.1 |"
  > — `README.md:137` (Recovery Diagnostics table) and `tables/recovery-diagnostics.csv:2`

- **Code evidence (verbatim):**
  ```python
  "Trimmed share": round(1.0 - len(data) / len(auctions), 3),
  ```
  `run.py:161`

- **Data evidence:**
  - `tables/recovery-diagnostics.csv`: `Trimmed share,0.1`
  - `len(auctions) = 12000` (3000 auctions × 4 bidders)
  - `len(data) = 10800` (Kept bids from same CSV)
  - `1.0 - 10800/12000 = 0.1` — arithmetic matches.

  The claim is "5% in each tail" → 10% trimmed total. The number 0.1 is consistent with that prose claim. HOWEVER, `trim_quantile=0.05` is applied to bids symmetrically: lower 5% and upper 5% of bids are removed. The code computes trimmed share as `1 - kept/all_bids`. With 10800 kept out of 12000 total bids, 1200 are trimmed. `1200/12000 = 0.10`, which matches. So the number is correct.

  The subtle issue: `len(auctions)` at line 161 refers to the full `auctions` DataFrame (12000 rows = all bids across all auctions), not the number of auction events (3000). The variable name `auctions` is thus ambiguous — it holds bidder-level rows, not auction-level rows. The prose claim "Auctions: 3000" in the diagnostics table is separately hardcoded via `n_auctions` at line 157, so the diagnostics table is internally consistent. But the trimmed share denominator `len(auctions)` correctly equals 12000 (all bids), making the formula right for the right reason despite the confusing variable name.

  This is a documentation/naming precision gap, not a numerical error.

- **Category:** DILUTED — `len(auctions)` used as denominator for trimmed-share could mislead a reader into thinking trim fraction is relative to auction count (3000) rather than bid count (12000); code happens to be correct because `auctions` is a bid-level DataFrame.

- **Severity:** MED

- **Result-changing:** no — the number 0.1 in the CSV and README is correct.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "auctions" in inspect.getsource(main) and len(pd.read_csv("tables/recovery-diagnostics.csv")["Observed bids"]) > 0  # name collision: 'auctions' is bid-level, not auction-level
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "len(all_bids)" in inspect.getsource(main) or "len(auctions_df)" in inspect.getsource(main)  # rename variable to clarify denominator is bid count
  ```

---

## Cross-cutting patterns

- Both DILUTED findings are documentation precision gaps, not algorithm errors. The code is mathematically correct; the prose and variable naming leave a reader unable to verify the normalizer convention without reading the code.
- The variable `auctions` names a bid-level DataFrame throughout `run.py`. It is used as both a bid container and implicitly as a bid-count denominator. A reader expecting auction-count semantics will misread line 161. Consider renaming to `bids_df` or `all_bids`.
- No parametric information leaks into the recovery step. `recover_pseudo_values` receives only `bids`, `n_bidders`, and `trim_quantile`. The `beta` distribution and true `values` column are never passed to or accessed by the inversion function. The isolation is clean.
- The equilibrium bid schedule computed via `equilibrium_bid_grid` (trapezoid integration of `F^{n-1}`) correctly implements the closed-form symmetric BNE bid function. The README Equations section derives this ODE and the code faithfully solves it numerically.
- All five numeric claims in the Results tables (RMSE, MAE, Correlation, Kept bids, Trimmed share) are internally consistent between `README.md` and both CSV files.

---

## TDD execution sequence (for the next agent)

0. **Bullshit score is 30%.** Below 50% — no halt required. Surface findings to author for documentation fix before next commit.

1. **Finding 1 — violated invariant test:**
   ```python
   def test_empirical_cdf_is_one_indexed():
       result = empirical_cdf_at_sample(np.array([1.0, 2.0, 3.0]))
       assert result[0] == pytest.approx(1/3)   # passes on current code (1-indexed)
   ```
   This test PASSES on current code — it confirms the convention.
   The fix is documentation: add `(1-indexed, i.e., minimum maps to 1/N)` to pseudocode step 3 in `run.py:280`.

2. **Finding 2 — violated invariant test:**
   ```python
   def test_trimmed_share_denominator_is_bid_count():
       # confirms 'auctions' DataFrame is bid-level (12000 rows), not auction-level (3000)
       assert len(auctions) == n_auctions * n_bidders   # 12000, not 3000
   ```
   This test PASSES on current code. The fix is renaming `auctions` → `all_bids` in `run.py` to remove the ambiguity.

3. No sim re-runs required — both fixes are documentation/naming only.

4. After fixes, re-run `python run.py` to confirm README regenerates with corrected pseudocode. Re-run this skill. Expected new score: ≤10%.
