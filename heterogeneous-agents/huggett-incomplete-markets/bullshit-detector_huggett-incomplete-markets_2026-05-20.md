# bullshit-detector — huggett-incomplete-markets — 2026-05-20

**Bullshit score: 20%** — Multiple DATA DRIFT findings from format-string rounding inconsistencies across the same run: Results prose reports "0.0350" and "0.16%" while the table reports "0.03499" and "0.155%". One DILUTED finding: mean wealth displayed as 0.00000 while the bond-market residual row in the same table discloses the same quantity as 5.43e-06. No FALSE, no UNIMPLEMENTED, no MISLABELED. All algorithmic claims (upwind HJB, implicit step, KFE duality, bisection, boundary conditions) hold against code.

## Header
- Claim sources: `heterogeneous-agents/huggett-incomplete-markets/README.md`
- Code / artifact root: `heterogeneous-agents/huggett-incomplete-markets/run.py`
- Data artifacts: `heterogeneous-agents/huggett-incomplete-markets/tables/equilibrium.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Relative value gap is "0.16%" (Results text) | DATA DRIFT | LOW | no (same run, format rounding only) |
| 2 | r* is "0.0350" (Results/Takeaway text) | DATA DRIFT | LOW | no (same run, format rounding only) |
| 3 | Wedge is "0.0150" (Results/Takeaway text) | DATA DRIFT | LOW | no (same run, format rounding only) |
| 4 | "Mean assets are zero" (table prose) | DILUTED | LOW | no (residual 5.43e-06 disclosed in same table) |
| 5 | Upwind HJB scheme matches description | HOLDS | — | — |
| 6 | Implicit step formula | HOLDS | — | — |
| 7 | KFE solved by A-transpose | HOLDS | — | — |
| 8 | Boundary condition at borrowing limit | HOLDS | — | — |
| 9 | Bisection logic: S>0 lower r, S<0 raise r | HOLDS | — | — |
| 10 | Aswitch income-switching block | HOLDS | — | — |
| 11 | All numeric values in table match CSV | HOLDS | — | — |

## Findings

### Finding 1: Relative value gap stated as "0.16%" in Results text but "0.155%" in table

- **Claim source (verbatim):** "The relative value gap against the reference grid is $0.16\%$." — `README.md:337`
- **Code evidence (verbatim):**
  ```python
  f"The relative value gap against the reference grid is ${100 * V_gap_rel:.2f}\\%$.",
  ```
  `run.py:729`
  ```python
  f"{100 * V_gap_rel:.3f}%",
  ```
  `run.py:845`
- **Data evidence:** `tables/equilibrium.csv` row: `Relative value gap (% of value scale),0.155%`
- **Category:** DATA DRIFT — both representations derive from the same `V_gap_rel` variable; `:.2f` and `:.3f` format strings produce "0.16%" and "0.155%" respectively for a value near 0.001553.
- **Severity:** LOW
- **Result-changing:** no — the underlying number is the same; only display precision differs.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.16%" in open("README.md").read() and "0.155%" in open("README.md").read()
  # PASSES on current code (both strings present); FAILS on honest fix that uses one format
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert len({s for s in ["0.16%","0.155%"] if s in open("README.md").read()}) == 1
  # PASSES on honest fix (single format throughout); FAILS on current code (two formats)
  ```

---

### Finding 2: r* stated as "0.0350" in Results/figure text but "0.03499" in Equations and table

- **Claim source (verbatim):** "The Huggett equilibrium is lower, at $r^{\ast} = 0.0350$." — `README.md:349`
- **Code evidence (verbatim):**
  ```python
  f"The Huggett equilibrium is lower, at $r^{{\\ast}} = {r_eq:.4f}$. "
  ```
  `run.py:803`
  ```python
  f"{r_eq:.5f}",
  ```
  `run.py:832`
- **Data evidence:** `tables/equilibrium.csv` row: `Equilibrium r* (working grid),0.03499`. Equations section: `README.md:162` writes `r^{\ast} = 0.03499`.
- **Category:** DATA DRIFT — `r_eq:.4f` truncates to `"0.0350"` while `r_eq:.5f` preserves `"0.03499"`. Same variable, two format strings, two outputs in the same document.
- **Severity:** LOW
- **Result-changing:** no — one extra digit of precision; no conclusion changes.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.0350" in open("README.md").read() and "0.03499" in open("README.md").read()
  # PASSES on current code; FAILS on honest fix using a single format
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert open("README.md").read().count("0.03499") >= 2 and "r* = 0.0350}" not in open("README.md").read()
  # PASSES on honest fix (uniform 5-decimal display); FAILS on current code
  ```

---

### Finding 3: Precautionary wedge stated as "0.0150" in Results and Takeaway but "0.01501" in table

- **Claim source (verbatim):** "The precautionary wedge is $\rho - r^{\ast} = 0.0150$." — `README.md:349`; and "In this run the wedge is $\rho - r^{\ast} = 0.0150$." — `README.md:381`
- **Code evidence (verbatim):**
  ```python
  f"The precautionary wedge is $\\rho - r^{{\\ast}} = {wedge:.4f}$.",
  ```
  `run.py:804`
  ```python
  f"$\\rho - r^{{\\ast}} = {wedge:.4f}$"
  ```
  `run.py:864`
  ```python
  f"{wedge:.5f}",
  ```
  `run.py:834`
- **Data evidence:** `tables/equilibrium.csv` row: `Precautionary wedge rho - r*,0.01501`. With `rho=0.05` and `r_eq=0.03499`, `wedge=0.01501`; `:.4f` yields `"0.0150"`.
- **Category:** DATA DRIFT — same `wedge` variable, `:.4f` in prose vs `:.5f` in table.
- **Severity:** LOW
- **Result-changing:** no.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert open("README.md").read().count("0.0150") >= 2 and "0.01501" in open("README.md").read()
  # PASSES on current code (both forms present); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "0.0150}" not in open("README.md").read() and open("README.md").read().count("0.01501") >= 2
  # PASSES on honest fix (uniform precision); FAILS on current code
  ```

---

### Finding 4: "Mean assets are zero" prose while same table shows non-zero bond-market residual

- **Claim source (verbatim):** "Mean assets are zero because bisection chose $r^{\ast}$ to satisfy $S(r^{\ast}) = 0$." — `README.md:353`; table row `Mean wealth E[a]` shows `0.00000` — `README.md:363`.
- **Code evidence (verbatim):**
  ```python
  mean_wealth = (g[:, 0] @ a) * da + (g[:, 1] @ a) * da
  market_residual = abs(mean_wealth)
  mean_wealth_display = 0.0 if market_residual < 5e-5 else mean_wealth
  ```
  `run.py:295-297`
  ```python
  f"{mean_wealth_display:.5f}",   # -> "0.00000"
  ```
  `run.py:835`
  ```python
  f"{market_residual:.2e}",       # -> "5.43e-06"
  ```
  `run.py:841`
- **Data evidence:** `tables/equilibrium.csv`: `Mean wealth E[a],0.00000` and `Bond-market residual abs(S(r*)),5.43e-06`. These are the same quantity (`mean_wealth` and `market_residual = abs(mean_wealth)`), displayed differently: one is hard-coded to `0.0` when residual < `5e-5`, the other shows the true value.
- **Category:** DILUTED — the prose claim "mean assets are zero" is supported by the display override but undercut by the residual row disclosing the same quantity as `5.43e-06`. The code intentionally masks the true value; a reader who checks both rows sees inconsistency.
- **Severity:** LOW — the residual is genuinely tiny (`5.43e-06`) and does not change any economic conclusion. The display convention is visible in the code.
- **Result-changing:** no.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "0.00000" in open("tables/equilibrium.csv").read() and "5.43e-06" in open("tables/equilibrium.csv").read()
  # PASSES on current code (both present in same CSV); FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert all(row.split(",")[1] == "5.43e-06" for row in open("tables/equilibrium.csv").readlines() if "Mean wealth" in row)
  # PASSES on honest fix (actual value shown); FAILS on current code (shows 0.00000)
  ```

---

## Cross-cutting patterns

- All four findings share the same root: the same numeric quantity is rendered by two different format strings (or two different display paths) within a single `run.py`, and the two renderings end up in different parts of the same `README.md`. The pattern is not fraud; it is format-string inconsistency. Grounding every prose number in the same format string used for the summary table would eliminate all four findings in one pass.
- Findings 1, 2, 3 are all format-precision drift: `:.4f` in figure descriptions vs `:.5f` in the table. A project-wide convention of using a single canonical format per quantity (or referencing the table as ground truth in the prose) would prevent recurrence.
- Finding 4 is a display-override pattern (`mean_wealth_display = 0.0 if residual < threshold else mean_wealth`). The same table that benefits from this override also exposes the raw residual in another row, creating internal inconsistency. Either remove the override and show the true (tiny) value, or remove the separate residual row and rely on the mean-wealth row alone.
- All core algorithmic claims hold. No parametric leaks, no wrong formula, no unimplemented method. The HJB upwind scheme, implicit step, KFE duality, bisection direction, and Aswitch construction all match their descriptions verbatim.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20%.** Below the 50% halt threshold. No need to stop forward work. Fix cosmetically before next tutorial polish pass.

1. For each non-HOLDS finding, the violated-invariant test confirms the bug is real:
   - Finding 1: assert that both "0.16%" and "0.155%" appear in `README.md` — this passes on the current output.
   - Finding 2: assert that both "0.0350" and "0.03499" appear in `README.md` — passes on current output.
   - Finding 3: assert that both "0.0150}" and "0.01501" appear in `README.md` — passes on current output.
   - Finding 4: assert that `Mean wealth E[a]` row shows `0.00000` while `Bond-market residual` row shows `5.43e-06` in the same CSV — passes on current output.

2. The honest-fix pass conditions define the green state:
   - Findings 1-3: adopt a single format string per quantity. Use `:.5f` (matching the table) for `r_eq` and `wedge` in the figure descriptions and prose; use `:.3f` for `V_gap_rel` in the figure description.
   - Finding 4: either remove the display override (`mean_wealth_display`) and show the raw value in both rows, or consolidate the two rows into one.

3. Hand off to `writing-plans` if the fix is to be done systematically; otherwise the changes are small enough to apply directly in `run.py` then regenerate `README.md`.

4. After fixes, re-run this skill. Expected result: all four findings read HOLDS and the bullshit score drops to 0-5%.
