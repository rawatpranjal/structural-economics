# bullshit-detector — ramsey-growth — 2026-05-20

**Bullshit score: 20%** — two low-severity cosmetic gaps (DATA DRIFT + MISLABELED); all equations, parameters, ODE signs, algorithm description, and numeric results hold against code and CSV.

## Header
- Claim sources: `optimal-control/ramsey-growth/README.md`
- Code / artifact root: `optimal-control/ramsey-growth/run.py`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | c0 values in README table match CSV to 6 decimal places | DATA DRIFT | LOW | no (max diff 4e-06) |
| 2 | "Terminal capital gap" column is in units of capital | MISLABELED | LOW | no (reader interpretation only) |
| 3 | k* = 8.2898 | HOLDS | - | - |
| 4 | c* = 1.5952 | HOLDS | - | - |
| 5 | Euler equation sign: c_dot/c = [f'(k)-delta-rho]/sigma | HOLDS | - | - |
| 6 | ODE system in code matches README equations | HOLDS | - | - |
| 7 | dot{k}=0 isocline = net output f(k)-delta*k | HOLDS | - | - |
| 8 | dot{c}=0 locus = vertical line at k* | HOLDS | - | - |
| 9 | Shooting brackets c0 with opposite-sign terminal gaps | HOLDS | - | - |
| 10 | Positive G -> c0 too low; negative G -> c0 too high | HOLDS | - | - |
| 11 | Brent's method finds the zero | HOLDS | - | - |
| 12 | lambda_stable is the negative eigenvalue of the Jacobian | HOLDS | - | - |

## Findings

### Finding 1: c0 values in README table truncated vs CSV (DATA DRIFT)

- **Claim source (verbatim):** `README.md:121-124` — four rows of the Shooting Diagnostics table list c0 values: `1.16845`, `1.39947`, `1.92645`, `2.20938`.
- **Code evidence (verbatim):**
  ```python
  "$c_0$ from shooting": [f"{c0:.6f}" for c0 in initial_consumption],
  ```
  `run.py:366` — format string requests 6 decimal places.
- **Data evidence (verbatim):** `tables/shooting-results.csv:3-6`:
  ```
  0.50,1.168448,...
  0.75,1.399473,...
  1.50,1.926453,...
  2.00,2.209384,...
  ```
  The CSV records 6 decimal places. The README records 5 decimal places for those four rows. Differences: 2e-06, 3e-06, 3e-06, 4e-06. The first row (k0/k*=0.25, c0=0.867114) matches exactly.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no — differences are O(1e-06), far below any interpretive threshold.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert float(readme_table["c0"][1]) == pytest.approx(float(csv_table["$c_0$ from shooting"][1]), abs=1e-7)
  # PASSES on current state (diff=2e-06 > 1e-7 tolerance), FAILS after README regeneration
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert float(readme_table["c0"][1]) == pytest.approx(float(csv_table["$c_0$ from shooting"][1]), abs=1e-7)
  # PASSES after README is regenerated from run.py (both come from same f"{c0:.6f}" format)
  ```

### Finding 2: "Terminal capital gap" column is a relative (normalized) gap, not absolute capital units (MISLABELED)

- **Claim source (verbatim):** "The last column reports the terminal capital gap." — `README.md:114`. Column header in table: `Terminal capital gap` — `README.md:118`.
- **Code evidence (verbatim):**
  ```python
  terminal_residuals = [abs(sol.y[0, -1] - k_star) / k_star for sol in solutions]
  ```
  `run.py:363-364` — divides by `k_star`, producing a dimensionless ratio `|k(T)-k*|/k*`.
- **Data evidence:** CSV column header is `Terminal capital gap`; values are `2.75e-07`, `3.70e-07`, `3.17e-06`, `7.69e-08`, `4.36e-10`. With k*=8.2898, an absolute gap of 2.75e-07 would be negligible in capital units; a relative gap of 2.75e-07 means |k(T)-k*| = 2.28e-06 in capital units. Both readings yield tiny residuals, but the label "capital gap" implies absolute units of capital while the code computes a dimensionless ratio.
- **Category:** MISLABELED
- **Severity:** LOW
- **Result-changing:** no — the numeric values are unambiguous diagnostics of solver accuracy regardless of label interpretation.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "/ k_star" not in inspect.getsource(main)  # PASSES on current buggy code; FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "Relative terminal capital gap" in open("tables/shooting-results.csv").readline()  # PASSES after fix
  ```

## Cross-cutting patterns

- Both findings are cosmetic: one is a stale README (DATA DRIFT from truncated formatting), one is an imprecise column label (MISLABELED). Neither corrupts equations, parameter values, ODE signs, or numeric results.
- The core claim-vs-code alignment is strong throughout: every equation in `Equations` maps directly to a line in `ode_system` or `find_saddle_path_c0`; every parameter in `Model Setup` is set at the exact value stated; the shooting algorithm pseudocode accurately describes `bracket_saddle_consumption` + `brentq`.
- The README was likely generated from an earlier run whose c0 values were formatted at 5 decimal places (or the README was manually edited), while the committed CSV reflects the 6-decimal-place format string at `run.py:366`. Regenerating README from `python run.py` would resolve Finding 1.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 20% (< 50%).** Safe to proceed with minor fixes without halting.
1. For Finding 1 (DATA DRIFT): turn the violated invariant into a pytest test that reads both README table and CSV and asserts c0 values match to 1e-07. Confirm it PASSES on current state (proves the drift).
2. Convert honest-fix pass condition into a second pytest test. It FAILS currently because README has truncated values.
3. Fix: run `python run.py` inside `optimal-control/ramsey-growth/` to regenerate README with 6-decimal-place c0 values that match the CSV.
4. For Finding 2 (MISLABELED): rename the column at `run.py:367` from `"Terminal capital gap"` to `"Relative terminal capital gap"` (or `"|k(T)-k*|/k*"`). Update the matching description string at `run.py:384` and `README.md:114`.
5. After fixes, re-run this skill to confirm all findings read HOLDS and score drops to 0-10%.
