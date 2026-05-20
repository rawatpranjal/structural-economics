# bullshit-detector — global-search-multistart — 2026-05-20

**Bullshit score: 75%** — Three FALSE/HIGH findings: the tutorial's central pedagogical claim (single-start misses the global peak) is contradicted by the committed CSV and by the code's own output; the basin-boundary claim is refuted by committed data; and the hardcoded "30 percent improvement" is arithmetically impossible given the computed gap of 0.000.

## Header
- Claim sources: `numerical-methods/global-search-multistart/README.md` (Overview, Equations, Results, Model Setup)
- Code / artifact root: `numerical-methods/global-search-multistart/run.py`
- Data artifacts: `tables/method_comparison.csv`, `tables/multistart_results.csv`, `tables/basin_summary.csv`
- Seed audit (if any): none
- Run by: bullshit-detector subagent (Claude Sonnet 4.6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Both single-start methods land at the low-price local peak from p0=1.0" | FALSE | HIGH | yes — the tutorial's core lesson collapses |
| 2 | "Starts below the kink at p_kink=2.00 converge to the low peak" | FALSE | HIGH | yes — basin boundary characterisation wrong by ~0.5 in price |
| 3 | "gap is -0.000 … a 30 percent profit improvement that single-start methods miss silently" | FALSE | HIGH | yes — 0.000 gap ≠ 30%; the 30% is hardcoded dead prose |
| 4 | Equations section labels Method 1 as Multi-start; Solution Method section labels Method 1 as Single-start | DATA DRIFT | LOW | no — labelling inconsistency only |

## Findings

### Finding 1: Single-start L-BFGS-B from p0=1.0 finds the global peak, not the local

- **Claim source (verbatim):** "Both single-start methods land at the low-price local peak from $p_0 = 1.0$. Their reported profits are $\pi = 5.625$ for L-BFGS-B and $\pi = 4.136$ for Nelder-Mead." — `README.md:203`

- **Code evidence (verbatim):**
  ```python
  p0_single = 1.0
  res_single = minimize(
      neg_profit, x0=np.array([p0_single]),
      method='L-BFGS-B', bounds=[(p_lo, p_hi)],
  )
  p_single = float(res_single.x[0])
  profit_single = float(profit(p_single))
  ```
  `run.py:81-89`

  ```python
  "Found global?": [
      "no" if profit_single < profit_global - 1e-3 else "yes",
  ```
  `run.py:597`

- **Data evidence:** `tables/method_comparison.csv:2`:
  ```
  Single-start L-BFGS-B,Starting price 1.0,4.2500,5.6250,8,yes
  ```
  The CSV shows `p=4.2500`, `profit=5.6250`, `Found global?=yes`. The code's own logic at `run.py:597` evaluates `"no" if profit_single < 5.625 - 0.001 else "yes"` and prints `yes`. The claim that L-BFGS-B "lands at the low-price local peak" producing `pi=5.625` is internally inconsistent: `pi=5.625` is the **global** peak, not the local.

  Corroborating evidence from `tables/multistart_results.csv`: start 17 at `p0=0.9796` converges to `p_final=4.2500` (high-price basin). Start 27 at `p0=0.8295` similarly. p0=1.0 falls in the same sub-basin.

- **Category:** FALSE — the prose says single-start L-BFGS-B misses the global; the committed CSV says it finds it.
- **Severity:** HIGH
- **Result-changing:** yes — the tutorial's entire pedagogical point (single-start blindly reports a local maximum) is inverted for L-BFGS-B on this calibration with this start. A reader following the tutorial is told the wrong lesson about which methods fail.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert float(pd.read_csv("tables/method_comparison.csv").iloc[0]["Found global?"]) == "no"
  # PASSES on current buggy output (actually reads "yes" so assertion FAILS — proves the bug)
  # Correction: the test that PASSES on current buggy code is:
  assert pd.read_csv("tables/method_comparison.csv").iloc[0]["Found global?"] == "yes"
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert pd.read_csv("tables/method_comparison.csv").iloc[0]["Found global?"] == "no"
  # PASSES only after p0_single is changed to a start that genuinely lands in the low basin (e.g., p0=1.7)
  ```

---

### Finding 2: Basin boundary is ~1.5, not p_kink=2.0

- **Claim source (verbatim):** "Starts below the kink at $p_L^{\max} = 2.00$ converge to the low peak. Starts above the kink converge to the high peak. The basin volumes are 6.5 percent low and 93.5 percent high on this bracket." — `README.md:195`

- **Code evidence (verbatim):**
  ```python
  def basin_label(p_final):
      return "low-price" if p_final < (p_low_peak + p_high_peak) / 2 else "high-price"
  ```
  `run.py:113-114`

  The basin label function classifies by converged price, not by starting price. It makes no claim about the kink. The convergence mapping from start to basin is an empirical outcome of L-BFGS-B dynamics.

- **Data evidence:** `tables/multistart_results.csv` — starts below `p_kink=2.0` that land in the **high-price** basin:

  | Start id | Starting price | Basin |
  |----------|---------------|-------|
  | 27 | 0.8295 | high-price |
  | 17 | 0.9796 | high-price |
  | 4 | 1.2072 | high-price |
  | 8 | 1.4617 | high-price |
  | 36 | 1.4753 | high-price |

  Five committed data rows falsify the claim. The actual L-BFGS-B basin boundary lies near `p ≈ 1.5`, not at `p_kink=2.0`. The 6.5%/93.5% volumes (from the dense 200-start scan) are consistent with a boundary at ~1.5: `(2.0-1.5)/(8.0-0.501)*200 ≈ 13` starts in `[1.5, 2.0]`, matching `6.5% × 200 = 13`. They are inconsistent with all 40 below-kink starts going low.

- **Category:** FALSE — the committed CSV directly contradicts the stated basin boundary.
- **Severity:** HIGH
- **Result-changing:** yes — the tutorial uses this claim to explain why single-start misses the global. The correct picture (basin boundary ~1.5, not kink=2.0; gradient overshoots the low peak for most low starts) is the opposite of the stated characterisation.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert all(pd.read_csv("tables/multistart_results.csv").query("`Starting price` < 2.0")["Basin"] == "low-price")
  # FAILS on current data (5 rows below 2.0 have Basin=high-price) — proves the bug
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert pd.read_csv("tables/multistart_results.csv").query("`Starting price` < 2.0 and Basin == 'high-price'").empty
  # PASSES only after the calibration or start is changed so the kink genuinely separates basins
  ```

---

### Finding 3: "30 percent profit improvement" is hardcoded dead prose; computed gap is 0.000

- **Claim source (verbatim):** "The gap between local and global on this calibration is $-0.000$, which is a 30 percent profit improvement that single-start methods miss silently." — `README.md:203`

- **Code evidence (verbatim):**
  ```python
  f"The gap between local and global on this calibration is ${profit_global - profit_single:.3f}$, "
  f"which is a 30 percent profit improvement that single-start methods miss silently."
  ```
  `run.py:549`

  `profit_global - profit_single` evaluates to `5.625 - 5.625 = 0.000` because `profit_single` is the result of the single-start L-BFGS-B run that finds the global peak (Finding 1). The string `"30 percent"` is a literal hardcoded in the f-string; it is not computed from the gap. The two parts of the sentence contradict each other in the same generated line: a gap of `$-0.000$` cannot be a `30 percent` improvement.

- **Data evidence:** `tables/method_comparison.csv:2`: `Profit=5.6250` for Single-start. `Profit=5.6250` for the global. Difference = 0.

- **Category:** FALSE — the sentence contains a computed gap of 0.000 and a hardcoded percentage that contradicts it. The 30% was presumably intended to describe `(5.625 - 4.136)/4.136 ≈ 36%`, but that requires single-start to have found the low peak, which it did not.
- **Severity:** HIGH
- **Result-changing:** yes — the quantitative lesson of the tutorial ("a 30 percent profit loss from using single-start") is arithmetically impossible given the computed result. A reader computing `(5.625 - 5.625)/5.625` gets 0%, not 30%.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "30 percent" in open("README.md").read() and "$-0.000$" in open("README.md").read()
  # PASSES on current file (both strings present in same sentence) — proves the contradiction
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  gap_str = re.search(r"\$([0-9.]+)\$, which is a ([0-9]+) percent", open("README.md").read())
  assert float(gap_str.group(1)) / 4.136 * 100 == pytest.approx(float(gap_str.group(2)), abs=5)
  # PASSES only when the computed gap and the stated percentage are consistent
  ```

---

### Finding 4: Method numbering differs between Equations and Solution Method sections

- **Claim source (verbatim):**
  - Equations section `README.md:60`: `### Method 1: Multi-start L-BFGS-B`
  - Solution Method section `README.md:112`: `### Method 1: Single-start L-BFGS-B`

- **Code evidence (verbatim):**
  ```python
  # Equations section in run.py:264
  "### Method 1: Multi-start L-BFGS-B\n\n"
  # Solution Method section in run.py:318
  "### Method 1: Single-start L-BFGS-B\n\n"
  ```
  `run.py:264` vs `run.py:318`

- **Data evidence:** n/a — labelling only.
- **Category:** DATA DRIFT — two sections of the same document use conflicting method numbering.
- **Severity:** LOW
- **Result-changing:** no — no numeric result depends on the label.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert open("README.md").read().count("### Method 1: Multi-start") == 0
  # PASSES on current file (the Equations section has it) — proves the drift
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert open("README.md").read().count("### Method 1:") == 1
  # PASSES when exactly one section uses Method 1, eliminating the conflict
  ```

---

## Cross-cutting patterns

- All three HIGH findings are causally linked. The root cause is a single decision: `p0_single = 1.0` at `run.py:81` sends L-BFGS-B into the global basin rather than the local one. This invalidates the narrative (Finding 1), the basin-boundary prose (Finding 2), and the hardcoded improvement figure (Finding 3) simultaneously. Fixing the start price (e.g., to `1.7`) would likely repair all three without touching any other logic.

- The basin boundary for L-BFGS-B is an L-BFGS-B artifact: the gradient at `p=1.0` is `4.1` (strongly positive), so the quasi-Newton solver overshoots the low peak at `p≈1.6029` and descends into the global basin. The committed CSV proves the true boundary lies near `p≈1.5`. The prose equates the L-BFGS-B basin boundary with the economic kink (`p_kink=2.0`), which is wrong; these are different objects.

- The hardcoded `"30 percent"` in `run.py:549` is the canonical dilution-marker failure: a literal string that was written when `profit_single` was expected to equal `4.136`, not validated when the code produced `5.625`. Any time a `run.py` f-string mixes a computed gap with a hardcoded prose interpretation of that gap, the interpretation will rot independently of the code.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 75%.** Halt forward code work. Surface these three FALSE/HIGH findings to the user before proposing fixes. The tutorial's pedagogical core is inverted.

1. For each non-HOLDS finding, turn the violated invariant into a pytest test under `tests/`. Confirm the test PASSES on current code (proves the bug is real).

2. Convert the honest-fix pass condition into a second pytest test that FAILS on current code. The pair is the red/green spec.

3. Root-cause fix to investigate (do not implement without user sign-off):
   - Change `p0_single` at `run.py:81` from `1.0` to a value in `[1.5, 2.0]` (e.g., `1.75`) so single-start L-BFGS-B genuinely finds the local peak.
   - After fixing, the 30% figure should be computed from the gap, not hardcoded (replace `run.py:549` string literal `"30 percent"` with `f"{(profit_global - profit_single)/abs(profit_single)*100:.0f} percent"`).
   - The basin boundary prose should cite the empirical boundary (~1.5) rather than equating it to the economic kink (2.0).

4. After fixes, re-run `python run.py`, inspect generated `README.md` and CSV artifacts, then re-run this skill to confirm all findings read HOLDS and bullshit score drops to ≤25%.
