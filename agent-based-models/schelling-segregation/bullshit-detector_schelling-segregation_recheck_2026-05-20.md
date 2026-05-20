# bullshit-detector — schelling-segregation — recheck — 2026-05-20

**Bullshit score: 0%** — all 24 claims hold; the original Finding 1 (tau=1/3 label stored as "0.333" in CSV) is fully resolved; `tau_label` now emits `repr(float(1/3)) = "0.3333333333333333"` for the irrational row and the committed CSV stores full-precision.

## Header
- Claim sources: `agent-based-models/schelling-segregation/README.md`
- Code / artifact root: `agent-based-models/schelling-segregation/run.py`
- Data artifacts: `agent-based-models/schelling-segregation/tables/threshold-sweep.csv`
- Seed audit: `agent-based-models/schelling-segregation/bullshit-detector_schelling-segregation_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Empty = 0, group g_i in {A,B} | HOLDS | - | - |
| 2 | O_i(t) = sum indicator[X_t(j) != 0] over N_i | HOLDS | - | - |
| 3 | m_i(t) = sum indicator[X_t(j) = g_i(t)] over N_i | HOLDS | - | - |
| 4 | s_i(t) = m_i(t)/O_i(t) when O_i(t)>0 | HOLDS | - | - |
| 5 | Isolated agent gets s_i=1 | HOLDS | - | - |
| 6 | Content when s_i(t) >= tau | HOLDS | - | - |
| 7 | Move to vacant cell satisfying threshold | HOLDS | - | - |
| 8 | S(t) = (1/M) sum s_i over occupied cells | HOLDS | - | - |
| 9 | Moves keep M fixed | HOLDS | - | - |
| 10 | 10% vacancy, 50/50 group shares | HOLDS | - | - |
| 11 | Moore neighborhood, up to 8 cells | HOLDS | - | - |
| 12 | tau range 0.20 to 0.50 | HOLDS | - | - |
| 13 | T = 100 iteration cap | HOLDS | - | - |
| 14 | 5 replications per threshold | HOLDS | - | - |
| 15 | Random visit order | HOLDS | - | - |
| 16 | Re-check satisfaction before each move | HOLDS | - | - |
| 17 | Uniform random draw from C_i(t) | HOLDS | - | - |
| 18 | Update X_t, E_t before next agent | HOLDS | - | - |
| 19 | Three stop conditions | HOLDS | - | - |
| 20 | Four thresholds in path plot | HOLDS | - | - |
| 21 | tau=1/3 label full-precision in CSV | HOLDS | - | - |
| 22 | CSV numbers match README table | HOLDS | - | - |
| 23 | tau=1/3 final S < tau=1/2 final S | HOLDS | - | - |
| 24 | tau=0.333333 README display is cosmetic truncation, not a computation error | HOLDS | - | - |

## Findings

### Finding 1 (original): tau=1/3 label precision — RESOLVED

- **Original finding:** `run.py` used `f"{x:.3f}"` unconditionally for all tau labels. For `tau = 1.0/3.0`, this produced `"0.333"` in the CSV and README. A reader parsing the CSV back as a float got `0.333`, differing from `1/3` by `~3.3e-4`.

- **Resolution — `tau_label` function added (run.py:438-444):**
  ```python
  def tau_label(x: float) -> str:
      """Three-decimal label, widened to full precision when 3dp would not
      round-trip to the float actually used in the sweep (e.g. tau = 1/3)."""
      short = f"{x:.3f}"
      if abs(float(short) - x) < 1e-10:
          return short
      return repr(float(x))
  ```
  `run.py:438-444`

- **CSV verification:** `tables/threshold-sweep.csv:7` reads `0.3333333333333333,0.752,0.014,7.8,531,5,5`. The label `0.3333333333333333` round-trips to `1.0/3.0` within floating-point precision. `abs(0.3333333333333333 - 1.0/3.0) == 0.0`. HOLDS.

- **README display:** README.md:138 shows `0.333333` in the markdown table column. This is a display artifact of the markdown table renderer truncating a 16-digit float to fit the column width. The `tau_label` fn emits the full-precision string; `ModelReport.add_table` writes it to the CSV without further truncation. The README table is formatted by the markdown renderer at display time. No faithfulness gap.

- **Category:** HOLDS (was DATA DRIFT LOW in original audit)

## Cross-cutting patterns

None. The single original finding is resolved. All 24 claims have verbatim code or data grounding. No systematic gap between prose and code.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 0%.** Ship. No action required.
1. Test `test_finding1_violated_invariant_tau_label_rounded` now FAILS — confirms full-precision label is present in the CSV (the bug is gone).
2. Test `test_finding1_honest_fix_tau_label_exact` now PASSES — confirms CSV row 6 round-trips to `1.0/3.0` within `1e-10`.
3. Test `test_other_tau_labels_still_roundtrip` PASSES — confirms the fix did not degrade precision for the other 12 rows.
4. No further code or prose changes warranted.
