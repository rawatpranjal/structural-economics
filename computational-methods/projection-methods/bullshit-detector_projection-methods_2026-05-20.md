# bullshit-detector - projection-methods - 2026-05-20

**Bullshit score: 10%** - one DILUTED/LOW prose imprecision ("between collocation nodes") that does not touch any result; all numeric, algorithmic, and formula claims HOLD against code and CSV.

## Header
- Claim sources: `computational-methods/projection-methods/README.md`
- Code / artifact root: `computational-methods/projection-methods/run.py`
- Data artifacts: `computational-methods/projection-methods/tables/projection-accuracy.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "dense grid between collocation nodes" | DILUTED | LOW | no |
| 2 | steady-state capital = 0.1870 | HOLDS | - | - |
| 3 | interval [0.0468, 0.3273] | HOLDS | - | - |
| 4 | exact policy g*(k) = alpha*beta*A*k^alpha | HOLDS | - | - |
| 5 | Euler residual R_i formula | HOLDS | - | - |
| 6 | log-linear Chebyshev parameterization | HOLDS | - | - |
| 7 | table values (all 4 basis sizes) | HOLDS | - | - |
| 8 | prose result "max Euler error 5.57e-04" | HOLDS | - | - |
| 9 | pseudocode steps 1-7 vs code | HOLDS | - | - |
| 10 | full-depreciation technology | HOLDS | - | - |

## Findings

### Finding 1: "dense grid between collocation nodes"

- **Claim source (verbatim):** "The table evaluates errors on a dense grid between collocation nodes." - `README.md:102`
- **Code evidence (verbatim):**
  ```python
  eval_grid = np.linspace(lower, upper, 320)
  ```
  `run.py:191`
- **Data evidence (if applicable):** None - this is a prose description of the grid, not a numeric output.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no - the grid itself is fine; the description is inaccurate. Chebyshev collocation nodes are interior roots of T_n (not endpoints), computed at `run.py:127-129` as `cos((2j-1)*pi/(2n))` mapped to `[lower, upper]`. The eval grid is `linspace(lower, upper, 320)` which covers the full interval including regions at and beyond the node positions - not restricted to sub-intervals strictly between nodes.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert not np.all((eval_grid > min(nodes)) & (eval_grid < max(nodes)))  # grid extends outside node-to-node interior
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "full interval" in readme_prose or "dense grid" in readme_prose and "between" not in readme_prose[readme_prose.index("dense grid"):readme_prose.index("dense grid")+60]
  ```

## Cross-cutting patterns

- No systematic mislabeling or parameter leak found. The tutorial is a closed-form benchmark where the exact solution is known; this structure makes fabrication straightforward to detect and none was found.
- The one DILUTED finding is a standard prose shorthand ("between collocation nodes" meaning "covering the approximation interval") that appears in numerical methods pedagogy, though it is technically imprecise.
- All four numeric rows in the table (`n = 2, 3, 5, 8`) match the committed CSV exactly to the printed significant figures. No DATA DRIFT between `README.md` table, `README.md` prose, and `tables/projection-accuracy.csv`.
- Euler residual formula in code (`run.py:86-95`) matches the displayed equation in `README.md:44-48` symbol-for-symbol.
- Chebyshev basis indexing (`chebvander(x, n_basis - 1)` producing columns `T_0...T_{n-1}`) matches the sum `sum_{j=0}^{n-1}` in the README equation.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%. Safe to proceed.** No halt required.
1. The one non-HOLDS finding (Finding 1) is a prose fix, not a code fix. No test needed against code behavior.
2. Fix: replace "between collocation nodes" with "over the full approximation interval" in `run.py:390` (the `description=` argument to `report.add_table`) and re-run `python run.py` to regenerate `README.md`.
3. After fix, re-run this skill to confirm Finding 1 now reads HOLDS and the bullshit score drops to 0-5%.
4. No simulation re-run needed; no data artifacts change.
