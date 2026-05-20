# bullshit-detector — consideration-set-estimation — recheck — 2026-05-20

**Bullshit score: 15%** — One DILUTED/LOW residual finding (hardcoded "bracket the true values" prose in the fig-4 results block contradicts the dynamically-generated "3 of 5 alternatives" claim in the fig-1 block; every original FALSE and higher-severity DILUTED finding is resolved).

## Header
- Claim sources: `choice/consideration-set-estimation/README.md`
- Code / artifact root: `choice/consideration-set-estimation/run.py`
- Data artifacts: `choice/consideration-set-estimation/tables/method-comparison.csv`, `choice/consideration-set-estimation/tables/ranking-recovery.csv`
- Seed audit: `choice/consideration-set-estimation/bullshit-detector_consideration-set-estimation_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (independent recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | `p(a,{a,b})=0.444`, `p(b,{b,c})=0.500`, `p(a,{a,c})=0.444` in Example 2 replication | HOLDS | - | no |
| 2 | "Method 1 recovers the full ranking; Method 2 gets 9 of 10 pairs right" | HOLDS | - | no |
| 3 | KL divergence: M1=0.0042, M2=1.4908, M2 is ~351x M1 | HOLDS | - | no |
| 4 | True-DGP LL disclosed: -15075.5 in method-comparison table | HOLDS | - | no |
| 5 | "within one bootstrap SE on 3 of 5 alternatives" (fig-1 block, conditional) | HOLDS | - | no |
| 6 | "bracket the true values" (fig-4 block, hardcoded, no conditional) | DILUTED | LOW | no — contradicts the fig-1 block but does not change any table number |
| 7 | Method 2 score pseudocode sign matches code sign | HOLDS | - | no |
| 8 | Closed-form choice probability formula | HOLDS | - | no |
| 9 | gamma MLE closed form given ranking | HOLDS | - | no |
| 10 | Method 2 attention from singleton-with-default frequencies | HOLDS | - | no |
| 11 | Method-comparison LL values match committed CSV | HOLDS | - | no |
| 12 | "Method 1 within 2 LL units of true-DGP" | HOLDS | - | no |

## Findings

### Finding 1 (residual): "bracket the true values" in fig-4 results block is hardcoded and contradicts the dynamically generated "3 of 5" claim in the fig-1 block

- **Claim source (verbatim):** "Method 1 attention bars carry bootstrap standard errors that bracket the true values." — `README.md:147`

- **Code evidence (verbatim):**
  ```python
  report.add_results(
      "Both methods recover the attention parameters across all five alternatives. "
      "Method 1 attention bars carry bootstrap standard errors that bracket the true values. "
      "Method 2 attention is read directly off singleton-with-default frequencies and matches Method 1 within sampling noise. "
      ...
  )
  ```
  `run.py:699-705` — no conditional gate; fixed string.

- **Contradicting code evidence (verbatim):**
  ```python
  within_one_se = bool(np.all(np.abs(gamma_m1 - gamma_true) <= gamma_se_m1))
  n_within_se = int(np.sum(np.abs(gamma_m1 - gamma_true) <= gamma_se_m1))
  if within_one_se:
      attention_sentence = (
          "and the recovered attention parameters lie within one bootstrap "
          "standard error of the truth on every alternative."
      )
  else:
      attention_sentence = (
          f"and the recovered attention parameters lie within one bootstrap "
          f"standard error of the truth on {n_within_se} of {J} "
          "alternatives."
      )
  ```
  `run.py:562-574` — correctly conditional; produces "3 of 5 alternatives" in the generated README (line 135).

- **Data evidence:** `README.md:135` reads "the recovered attention parameters lie within one bootstrap standard error of the truth on 3 of 5 alternatives." This is produced by the conditional block at `run.py:562-574` and is correct. `README.md:147` then reads "bracket the true values" (unconditional, implying all 5), produced by the hardcoded block at `run.py:699`. The same README contains both statements. A reader who reads both lines sees an internal contradiction: 3 of 5 within SE (line 135) vs. SE bars bracket the true values (line 147).

- **Category:** DILUTED — the conditional check exists and its output is correct; the second prose block simply fails to mirror it.

- **Severity:** LOW — the contradiction does not change any table number and no reader would recompute from it. Both statements are about a figure visual.

- **Result-changing:** no — all tables are correct; the only effect is an inconsistent sentence in the fig-4 results paragraph.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "bracket the true values" in open("README.md").read() and "3 of 5" in open("README.md").read()  # both contradictory claims coexist
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "bracket the true values" not in open("README.md").read() or "3 of 5" not in open("README.md").read()  # contradiction resolved
  ```

---

## Resolution of original findings

| Original # | Claim | Original verdict | New verdict |
|------------|-------|-----------------|-------------|
| F1 | `p(a,{a,b})=0.000` bar-chart menu-index bug | FALSE / HIGH | RESOLVED — `run.py:354-365` uses `menu_row()` mask lookup; values are 0.444, 0.500, 0.444 by direct computation |
| F2 | "Both methods recover the full ranking" false for Method 2 | FALSE / MED | RESOLVED — `run.py:737-753` conditional branch; README says "Method 1 recovers the full ranking; Method 2 gets 9 of 10" |
| F3 | KL "essentially zero" for Method 2 (actual 1.4908) | FALSE / MED | RESOLVED — prose replaced with `{kl_m2:.4f}, about {kl_m2/kl_m1:.0f} times Method 1's` |
| F4 | True-DGP LL never disclosed | DILUTED / MED | RESOLVED — `run.py:334` computes `ll_true`; CSV row "True DGP,-15075.5" present |
| F5 | "within one SE" hardcoded in fig-1 block | DILUTED / LOW | RESOLVED — `run.py:562-574` conditional; README says "3 of 5 alternatives" |
| F6 | Method-2 pseudocode sign reversed | MISLABELED / LOW | RESOLVED — `run.py:530` pseudocode reads `score[j] <- sum_i impact[j, i] - sum_i impact[i, j]`; README line 127 matches |

**New residual finding:** F6b — hardcoded "bracket the true values" prose at `run.py:699` / `README.md:147` contradicts the dynamically generated "3 of 5 alternatives" at `README.md:135`. This is DILUTED/LOW. Not present in the original audit.

## Cross-cutting patterns

- All six original findings are resolved. The fix agent addressed F1 (correct menu indices via mask lookup), F2 (conditional ranking-recovery sentence), F3 (actual KL values in prose), F4 (true-DGP LL row in CSV), F5 (conditional SE sentence in fig-1 block), and F6 (pseudocode sign alignment).
- One residual disease from the same family as F5 survives in the fig-4 results block (`run.py:699-705`): the "bracket the true values" sentence was not gated on the same `within_one_se` conditional that correctly handles the fig-1 block. The fix agent solved F5 at one call site but missed the second call site in the same function.
- The pattern: `run.py:main()` has two separate `report.add_results()` calls that both describe attention parameter recovery. The F5 fix was applied to the first call (fig-1 block, lines 575-579) but not the second (fig-4 block, lines 699-705).

## TDD execution sequence (for the next agent)

0. **Bullshit score 15% (< 25%). Safe to proceed.**
1. Write a test asserting both contradictory claims coexist:
   `assert "bracket the true values" in readme and "3 of 5" in readme` (PASSES on current code).
2. Fix `run.py:700-701`: replace the hardcoded "bracket the true values" sentence with a conditional that mirrors `run.py:564-573` — either reuse `attention_sentence` or introduce a new `bracket_sentence` conditioned on `within_one_se`.
3. Regenerate README (`python run.py`). Confirm "bracket the true values" is gone or qualified.
4. Run `scripts/validate_catalog.py`.
5. Re-run `python -m pytest tests/ -q`. All 6 honest-fix tests must pass; the 6 violated-invariant tests must fail (all 12 correct after fix).
