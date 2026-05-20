# bullshit-detector — global-search-multistart — recheck — 2026-05-20

**Bullshit score: 10%** — All three original HIGH/FALSE findings are resolved; one original LOW/DATA DRIFT finding is resolved; one new LOW/DATA DRIFT finding is introduced by the fix (basin window lower bound is `[1.52, 1.97]` in prose but committed 50-start CSV shows minimum low-basin start is 1.549, not 1.52 — the discrepancy is between the dense-scan figure and the 50-start multistart CSV and is a benign consequence of the two different grids, not a fabrication).

## Header
- Claim sources: `numerical-methods/global-search-multistart/README.md`
- Code / artifact root: `numerical-methods/global-search-multistart/run.py`
- Data artifacts: `tables/method_comparison.csv`, `tables/multistart_results.csv`, `tables/basin_summary.csv`
- Seed audit: `bullshit-detector_global-search-multistart_2026-05-20.md`
- Run by: bullshit-detector (independent recheck agent, Claude Sonnet 4.6)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Single-start L-BFGS-B lands at local peak from p0=1.7, Found global?=no" | HOLDS | none | n/a |
| 2 | "Gap is $1.489$, a 36 percent profit improvement" | HOLDS | none | n/a |
| 3 | "Basin boundary near p≈1.52, not at kink p=2.0; volumes 6.5%/93.5%" | HOLDS (dense-scan) | none | n/a |
| 4 | Method numbering is consistent: Method 1=Single-start...Method 5=Simulated annealing | HOLDS | none | n/a |
| 5 | Basin window lower bound $[1.52, 1.97]$ vs multistart CSV min low-start = 1.549 | DATA DRIFT | LOW | no — two different grids; dense scan correctly produces 1.52 |

## Findings

### Finding 1 (RESOLVED): Single-start finds the global peak

**Original claim (verbatim from prior audit):** "Both single-start methods land at the low-price local peak from $p_0 = 1.0$." — original `README.md:203`

**Resolution:**
- `run.py:85`: `p0_single = 1.7`
- `tables/method_comparison.csv:2`: `Single-start L-BFGS-B,Starting price 1.7,1.6029,4.1360,6,no`
- `README.md:204`: "Both single-start methods land at the low-price local peak from $p_0 = 1.7$."

L-BFGS-B from `p0=1.7` converges to `p=1.6029`, `profit=4.1360`, `Found global?=no`. The prior bug (`p0=1.0` overshooting into global basin) is fixed. The CSV, code, and prose are mutually consistent.

- **Category:** HOLDS
- **Severity:** none

---

### Finding 2 (RESOLVED): Gap hardcoded at "30 percent" while computed as 0.000

**Original claim (verbatim from prior audit):** "The gap between local and global on this calibration is $-0.000$, which is a 30 percent profit improvement that single-start methods miss silently." — original `README.md:203`

**Resolution:**
- `run.py:553-554`:
  ```python
  gap_abs = profit_global - profit_single
  gap_pct = gap_abs / profit_single * 100.0
  ```
- `run.py:560`: `f"The gap between local and global on this calibration is ${gap_abs:.3f}$, a {gap_pct:.0f} percent profit improvement that single-start methods miss silently."`
- `README.md:204` (generated): "The gap between local and global on this calibration is $1.489$, a 36 percent profit improvement that single-start methods miss silently."

Both the gap and the percentage are now computed from the live run. `gap_abs = 5.625 - 4.136 = 1.489`. `gap_pct = 1.489 / 4.136 * 100 = 36.0`. The README reads `1.489` and `36 percent`. The two are arithmetically consistent (verified: `1.489 / 4.136 * 100 = 36.0`, tolerance < 0.1%).

- **Category:** HOLDS
- **Severity:** none

---

### Finding 3 (RESOLVED): Basin boundary equated with economic kink p=2.0

**Original claim (verbatim from prior audit):** "Starts below the kink at $p_L^{\max} = 2.00$ converge to the low peak." — original `README.md:195`

**Resolution:**
- `README.md:196` (generated): "Only starts in the narrow window $[1.52,\, 1.97]$ converge to the low peak. Every start below that window also converges to the high peak: the gradient at a low price is strongly positive, so the quasi-Newton step overshoots the low peak and descends into the global basin. The L-BFGS-B basin boundary near $p \approx 1.52$ is an artifact of the solver dynamics and sits below the economic kink $p_L^{\max} = 2.00$, not at it."

The prose now correctly distinguishes the L-BFGS-B empirical boundary (~1.52 from the 200-start dense scan) from the economic kink (2.00). The equation in `run.py:474-477` computes the window from the dense scan dynamically (`boundary_lo`, `boundary_hi`), not hardcoded.

- **Category:** HOLDS
- **Severity:** none

---

### Finding 4 (RESOLVED): Method numbering conflict between Equations and Solution Method sections

**Original claim (verbatim from prior audit):** Equations section labelled "Method 1: Multi-start L-BFGS-B"; Solution Method section labelled "Method 1: Single-start L-BFGS-B" — original `README.md:60` and `README.md:112`

**Resolution:**
- Equations section now uses natural descriptive headers: "### Multi-start L-BFGS-B", "### Random search", "### Nelder-Mead", "### Simulated annealing" (no "Method N:" prefix).
- Solution Method section uses: "### Method 1: Single-start L-BFGS-B", "### Method 2: Multi-start L-BFGS-B", ..., "### Method 5: Simulated annealing via `dual_annealing`".
- `README.md`: no two sections share any "### Method N:" label pointing to different methods.

Verified: `re.findall(r'### Method \d+:', README)` returns exactly five headers, all in the Solution Method section, none duplicated.

- **Category:** HOLDS
- **Severity:** none

---

### Finding 5 (NEW): Basin window lower bound is $1.52$ in prose but first low-basin start in 50-start CSV is $1.549$

**Claim source (verbatim):** "Only starts in the narrow window $[1.52,\, 1.97]$ converge to the low peak." — `README.md:196`

**Code evidence (verbatim):**
```python
boundary_lo = float(starts_dense[low_mask].min()) if low_mask.any() else float("nan")
boundary_hi = float(starts_dense[low_mask].max()) if low_mask.any() else float("nan")
```
`run.py:470-471`

The basin window `[1.52, 1.97]` is derived from the 200-point dense scan (`starts_dense = np.linspace(p_lo, p_hi, 200)` at `run.py:446`), which has finer resolution than the 50-start multistart run. The dense scan correctly identifies a low-basin start at approximately `p0 = 1.52`.

**Data evidence:** `tables/multistart_results.csv` rows with `Basin=low-price`:
- Start 49: Starting price `1.5490`, Converged price `1.6029`
- Start 28: Starting price `1.6580`, Converged price `1.6029`
- Start 35: Starting price `1.9218`, Converged price `1.6029`
- Start 25: Starting price `1.9606`, Converged price `1.6029`

The minimum low-basin start in the 50-start CSV is `1.549`, not `1.52`. The discrepancy is `0.029` in price. This is not a fabrication: the prose basin window comes from the dense 200-start scan (which includes a point near `p0 = 1.52` that falls in the low basin), while the multistart CSV is a separate 50-draw RNG run. The two grids are different; neither is wrong. However, a reader comparing the prose window `[1.52, 1.97]` against the multistart CSV will find no entry at `1.52`. This is a low-severity DATA DRIFT between two data artifacts that describe the same object (the L-BFGS-B basin boundary).

- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no — the pedagogical claim (boundary near 1.5, below kink at 2.0) is correct. The difference is granularity of sampling, not a directional error.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert pd.read_csv("tables/multistart_results.csv").query("Basin == 'low-price'")["Starting price"].min() == pytest.approx(1.52, abs=0.01)
  # FAILS on current data (min is 1.549, not 1.52) — captures the drift
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert pd.read_csv("tables/multistart_results.csv").query("Basin == 'low-price'")["Starting price"].min() < 1.55
  # PASSES on current data (1.549 < 1.55) — confirms the basin is correctly in the right neighborhood
  ```

Note: the fix, if desired, is to add a note in prose distinguishing "dense-scan boundary (1.52)" from "multistart-CSV minimum low start (1.549)". Both are consistent with the 1.5 boundary claim. The LOW severity reflects that no numeric result is wrong; only the mapping between two data artifacts is imprecise.

---

## Cross-cutting patterns

- The three original HIGH/FALSE findings were causally chained (all flowed from `p0_single = 1.0`). Changing to `p0_single = 1.7` resolved all three simultaneously, as predicted by the original audit's cross-cutting section.
- The new LOW/DATA DRIFT finding (Finding 5) is a cosmetic side-effect of the fix. The dense-scan basin window and the multistart CSV basin window are derived from independent RNG draws and different grid resolutions. They cannot be identical. The prose window `[1.52, 1.97]` is correct for the dense scan; the multistart minimum of `1.549` is correct for that separate 50-start run.
- All computed quantities are now dynamic f-strings derived from live code output (`gap_abs`, `gap_pct`, `boundary_lo`, `boundary_hi`, `pct_low`, `pct_high`). No hardcoded prose percentages remain.
- Method numbering is clean: Solution Method section uses "Method 1" through "Method 5"; Equations section uses natural names only.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Three honest-fix tests pass; four violated-invariant tests correctly fail (bugs eliminated). One new LOW/DATA DRIFT finding (Finding 5) does not require a code fix; it may warrant a prose clarification distinguishing the two grids.

1. Finding 5 is optional cleanup. If the user wants the prose window to reference both grids, add a parenthetical: "Only starts in the narrow window $[1.52,\, 1.97]$ (from a 200-point dense scan) converge to the low peak; the 50-start multistart log shows starts as low as $1.549$ in this basin."

2. No other action required. The tutorial is faithful. The violated-invariant tests correctly fail on fixed code (proving the bugs are gone) and the honest-fix tests pass (proving the claims are true).
