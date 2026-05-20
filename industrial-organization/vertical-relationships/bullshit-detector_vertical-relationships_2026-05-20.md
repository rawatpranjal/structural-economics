# bullshit-detector — vertical-relationships — 2026-05-20

**Bullshit score: 20%** — one MISLABELED finding (MED): Solution Method pseudocode uses loop/search syntax for what is a single closed-form analytical derivation; all numeric claims and formula claims HOLD.

## Header
- Claim sources: `industrial-organization/vertical-relationships/README.md` (Overview, Equations, Model Setup, Solution Method, Results, table)
- Code / artifact root: `industrial-organization/vertical-relationships/run.py`
- Data artifacts: `industrial-organization/vertical-relationships/tables/vertical-contracts.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | `p^I = (p_bar + c_M + c_R)/2` | HOLDS | - | no |
| 2 | Retailer best response `p_R(w) = (p_bar + w + c_R)/2` | HOLDS | - | no |
| 3 | `w^DM = (p_bar - c_R + c_M)/2` | HOLDS | - | no |
| 4 | Two-part tariff fee `F = (p^I - c_M - c_R)*q(p^I)` | HOLDS | - | no |
| 5 | Integrated: price=6.5, qty=7.0 | HOLDS | - | no |
| 6 | Linear wholesale: price=8.25, qty=3.5 | HOLDS | - | no |
| 7 | TPT restores price=6.5, qty=7.0 | HOLDS | - | no |
| 8 | All table values match CSV and computed outcomes | HOLDS | - | no |
| 9 | Pseudocode step 2 implies loop/search over candidate w; code uses single closed-form | MISLABELED | MED | no |

## Findings

### Finding 1: Pseudocode implies optimization loop; code uses direct closed-form

- **Claim source (verbatim):** "For a candidate wholesale price w: / retailer sets p_R(w) = (a/b + w + c_R) / 2 / Manufacturer chooses w_DM to maximize (w-c_M) q(p_R(w))" — `README.md:68-71`
- **Code evidence (verbatim):**
  ```python
  wholesale_dm = (a / b - cr + cm) / 2
  retail_dm = (a + b * (wholesale_dm + cr)) / (2 * b)
  ```
  `run.py:49-50`
- **Data evidence (if applicable):** None. Numeric result is identical regardless: `wholesale_dm=5.5`, `retail_dm=8.25` match all table entries.
- **Category:** MISLABELED
- **Severity:** MED
- **Result-changing:** no (analytical solution = numerical optimizer solution for this linear-quadratic problem; all published numbers match)
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "optimize" not in inspect.getsource(solve_contracts) and "linspace" not in inspect.getsource(solve_contracts) and "for w" not in inspect.getsource(solve_contracts)
  # PASSES on current code (no loop/optimizer found); FAILS if an honest optimizer were added
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "For a candidate wholesale price" not in open("README.md").read() or "closed-form" in open("README.md").read()
  # PASSES once pseudocode is corrected to say 'analytically solve FOC'; FAILS on current README
  ```

## Cross-cutting patterns

- All formula derivations (`p^I`, `p_R(w)`, `w^DM`, `F`) are algebraically equivalent between Equations section and `run.py`. No formula drift.
- All 10 numeric cells in the Results table match `tables/vertical-contracts.csv` exactly (rounding consistent: `.2f` Python format matches stated 2-decimal values; Python banker rounding of `6.125 -> 6.12` matches both CSV and README).
- The single MISLABELED finding is the gap between "for a candidate w" loop framing in pseudocode and the direct closed-form in code. The Equations section correctly presents `w^DM` as a formula, so the Equations section is self-consistent. Only the Solution Method pseudocode mismatches.
- No numeric claim requires re-run to verify; all are grounded in `tables/vertical-contracts.csv` and algebraic computation in this session.

## TDD execution sequence (for the next agent)

0. **Bullshit score = 20%.** Below the 50% halt threshold. Safe to proceed. Single low-stakes fix: correct the pseudocode framing.
1. Turn **violated invariant** into a test confirming the current README contains "For a candidate wholesale price" without a "closed-form" qualifier. Confirm it PASSES on current README.
2. Turn **honest-fix pass condition** into a second test. It FAILS on current README.
3. Fix: in `run.py` `add_solution_method(...)` string, replace the "For a candidate wholesale price w:" pseudocode block with language that names the closed-form solution explicitly (e.g., "The manufacturer's FOC yields the closed-form solution w_DM = (a/b - c_R + c_M)/2 directly.").
4. Regenerate `README.md` with `python run.py`. Re-run tests. Violated-invariant test now FAILS; pass-condition test now PASSES.
5. Re-run this skill. Expected score: 0-10%.
