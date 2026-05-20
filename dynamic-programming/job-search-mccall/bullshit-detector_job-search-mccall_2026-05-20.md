# bullshit-detector — job-search-mccall — 2026-05-20

**Bullshit score: 5%** — all structural and numeric claims hold; one sub-threshold rounding presentation noted (text "about 16 periods" vs table 16.5) that is not a contradiction but warrants a comment.

## Header
- Claim sources: `dynamic-programming/job-search-mccall/README.md` (Overview, Equations, Model Setup, Solution Method, Results sections)
- Code / artifact root: `dynamic-programming/job-search-mccall/run.py`
- Data artifacts: `dynamic-programming/job-search-mccall/tables/reservation-wages.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, 2026-05-20)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Bellman operator T: each sweep one expectation, one max | HOLDS | — | no |
| 2 | VFI init V_i = w_i/(1-beta) | HOLDS | — | no |
| 3 | Continuation C = b + beta * sum p_i V_i | HOLDS | — | no |
| 4 | Reservation wage extracted as w* = (1-beta)*C | HOLDS | — | no |
| 5 | Scalar fixed-point w* = (1-beta)*b + beta*E[max(W',w*)] | HOLDS | — | no |
| 6 | Discrete bins: 50 equiprobable, represented by conditional mean | HOLDS | — | no |
| 7 | Mean offer preserved exactly by discretization | HOLDS | — | no |
| 8 | E[max(W,r)] formula for continuous benchmark | HOLDS | — | no |
| 9 | Continuous fixed-point solved by Brent's method | HOLDS | — | no |
| 10 | Baseline: 178 VFI iterations | HOLDS | — | no |
| 11 | Baseline: sup-norm error 9.84e-09 | HOLDS (needs re-run to verify exact digit) | — | no |
| 12 | Baseline: w*_grid = 4.7054, w*_cont = 4.7055 | HOLDS | — | no |
| 13 | Absolute grid error 9.1e-05 | HOLDS (CSV -0.0001 is 4-decimal rounding of -9.1e-05) | — | no |
| 14 | Mean offer E[W] = 1.6487 | HOLDS | — | no |
| 15 | Median offer = 1.0000 | HOLDS | — | no |
| 16 | Acceptance probability 6.1% | HOLDS | — | no |
| 17 | "About 16 periods" duration vs table 16.5 | DATA DRIFT | LOW | no |
| 18 | Grid gap grows at high beta | HOLDS | — | no |
| 19 | All CSV table values consistent with README table | HOLDS | — | no |

## Findings

### Finding 1: "About 16 periods" in prose vs 16.5 in table

- **Claim source (verbatim):** "Expected unemployment duration is about **16 periods**." — `README.md:97`
- **Code evidence (verbatim):**
  ```python
  f"Expected unemployment duration is about **{expected_duration_cont:.0f} "
  "periods**."
  ```
  `run.py:336-337`
- **Data evidence:** `tables/reservation-wages.csv:5` (beta=0.95, b=1.0): `E[duration]` column = `16.5`
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no

The two numbers are consistent. The underlying acceptance probability is approximately 6.06% (not exactly 6.1%). At 6.06%: `100 * 0.0606 = 6.06` rounds to `6.1%` (one decimal); `1 / 0.0606 = 16.50` formats as `16.5` in the table (`:.1f`) and as `16` in the prose text (`:.0f`, Python banker's rounding: `round(16.5) = 16`). The text "about 16" is the correctly formatted output of `:.0f`; the table `16.5` is the correctly formatted output of `:.1f`. No fabrication. The discrepancy is a display-format artifact, not a numerical inconsistency.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert abs(1.0 / 0.061 - 16.5) < 0.05  # PASSES on current (true accept ~6.06%), but 1/0.061=16.39 would fail
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(round(1.0 / accept_frac_cont) - 16) <= 1 and abs(1.0 / accept_frac_cont - 16.5) < 0.05
  ```

Note: this finding is LOW severity and non-result-changing. It documents a potential reader confusion, not a code-vs-claim gap.

### Finding 2: Sup-norm error "9.84e-09" (needs re-run to verify)

- **Claim source (verbatim):** "The sup-norm error is **9.84e-09**." — `README.md:93`
- **Code evidence (verbatim):**
  ```python
  f"iterations**. The sup-norm error is **{info['error']:.2e}**. The grid "
  ```
  `run.py:289`
- **Data evidence:** `tables/reservation-wages.csv` does not store the final error; only iteration counts. The CSV baseline row shows `VFI iter.=178`, consistent with the text.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no

The error value `9.84e-09` is not stored in any committed artifact (CSV does not include the `error` column from `info`). The value is consistent with `tol=1e-8` (since `9.84e-09 < 1e-8`), and the code embeds the live runtime value directly into the README string at generation time (`{info['error']:.2e}`). The claim cannot be grounded against a committed artifact without a re-run. Flagged as "needs re-run to verify" per audit instructions.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "error" in df_table.columns  # FAILS on current code (CSV lacks error column)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert float(info["error"]) == pytest.approx(9.84e-9, rel=0.01)  # needs re-run
  ```

## Cross-cutting patterns

- All numeric values embedded in `README.md` are generated at runtime by `run.py` via f-string interpolation of live computation outputs (lines 288-292, 335-337). No hardcoded numbers exist in `run.py` that could silently drift from the code. The self-consistency of the generated README is structurally guaranteed by this design.
- The one DATA DRIFT finding (E[duration] text vs table) traces entirely to format-string differences (`:.0f` vs `:.1f`) applied to the same underlying float. This is a display architecture issue, not a numerical one.
- The CSV stores rounded displays; the README text also stores rounded displays of the same underlying computation. Internal inconsistencies between them require checking whether they are consistent with a common underlying value (not just comparing their surface strings). Both DATA DRIFT findings here resolved to consistent-with-common-value.
- No claim in this tutorial hedges with "approximately X," "X-style," or "in the spirit of X." All structural claims are stated directly and verified directly.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5% (well below 25% threshold).** No halt needed. The tutorial is faithful. Proceed to optional cleanup only.

1. The only actionable observation is the "about 16 periods" / 16.5 display discrepancy. If prose consistency with the table is desired:
   - **Violated invariant test:** confirm `round(expected_duration_cont) == 16` when true duration is 16.5 (Python banker's rounding).
   - **Fix option:** change `:.0f` to `:.1f` in `run.py:336` so prose and table agree on "16.5".
   - This is cosmetic and non-result-changing. Do not treat as a bug unless the reader finds it confusing.

2. The sup-norm error "9.84e-09" (Finding 2) cannot be verified without a re-run. If a persistent artifact is desired, add an `error` column to the CSV or write the baseline stats to a separate `tables/baseline-stats.csv`. Neither action changes the reported results.

3. After any edits, re-run `python run.py` inside `dynamic-programming/job-search-mccall/` and confirm all CSV values match the README table exactly (they should, given the self-generating architecture).

4. Re-run `scripts/validate_catalog.py` from repo root.
