# bullshit-detector -- rbc-irreversible-investment -- recheck -- 2026-05-20

**Bullshit score: 10%** -- prior MISLABELED finding (pseudocode said boundary only added to A_irr) resolved; three low-severity DATA DRIFT findings remain (iteration counts, 9.5% grid binding, 13.3% stress binding -- runtime scalars not in any committed CSV artifact).

## Header
- Claim sources: `global-dsge/rbc-irreversible-investment/README.md`
- Code / artifact root: `global-dsge/rbc-irreversible-investment/run.py`
- Data artifacts: `global-dsge/rbc-irreversible-investment/tables/stationary-moments.csv`
- Seed audit (if any): `bullshit-detector_rbc-irreversible-investment_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, recheck)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Pseudocode: boundary added to both A_std and A_irr | HOLDS | -- | -- |
| 2 | Binding indicator only recorded for irreversible model | HOLDS | -- | -- |
| 3 | VFI convergence 49/49 iters | DATA DRIFT | LOW | no |
| 4 | Boundary covers 9.5% of grid states | DATA DRIFT | LOW | no |
| 5 | Boundary binds 13.3% of stress path | DATA DRIFT | LOW | no |
| 6 | Stationary 0.42% binding matches CSV | HOLDS | -- | -- |
| 7 | Table values (mean K, std(Y), std(C)/std(Y), mean I/Y) | HOLDS | -- | -- |
| 8 | Steady-state K_ss=37.989, Y_ss=3.704, C_ss=2.754, I_ss=0.950 | HOLDS | -- | -- |
| 9 | Grid 72 K pts, 7 Tauchen z states | HOLDS | -- | -- |
| 10 | Stress K0=1.25*K_ss, common productivity draws | HOLDS | -- | -- |

## Findings

### Prior finding 1 (MISLABELED MED: pseudocode said boundary only to A_irr): RESOLVED

- **Original state:** Pseudocode said `"add the exact boundary K'=(1-delta)K_m to A_irr"`, while code computed the boundary unconditionally for both models.
- **Current state:** `README.md:75`: `"add the exact off-grid boundary K'=(1-delta)K_m to both A_std and A_irr"`. `README.md:76`: `"(for A_std it can only win when the optimum is at or above I=0)"`. Code at `run.py:121-131` computes `ev_boundary` unconditionally; the binding indicator assignment `is_binding = use_boundary` is guarded by `if constrained:` at line 130. Pseudocode and code now agree.
- **Category:** HOLDS

### Finding 1: VFI iteration counts not in committed CSV (DATA DRIFT, LOW)

- **Claim source (verbatim):** "The irreversible model converged in **49** VFI iterations. The standard comparison converged in **49**." -- `README.md:84`
- **Code evidence (verbatim):**
  ```python
  f"The irreversible model converged in **{sol_irr['iterations']}** VFI iterations. "
  f"The standard comparison converged in **{sol_std['iterations']}**."
  ```
  `run.py:421-422`
- **Data evidence:** `tables/stationary-moments.csv` has no `iterations` column. The value "49" in README is from the run that last generated the file. Not verifiable from committed artifacts.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no -- iteration count is a convergence diagnostic, not a result table value.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "iterations" not in open("tables/stationary-moments.csv").read()
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "iterations" in pd.read_csv("tables/stationary-moments.csv").columns or "irr_iterations" in open("tables/stationary-moments.csv").read()
  ```

### Finding 2: Grid binding share 9.5% not in committed CSV (DATA DRIFT, LOW)

- **Claim source (verbatim):** "the boundary covers 9.5% of grid states" -- `README.md:113`
- **Code evidence (verbatim):**
  ```python
  binding_share_states = float(sol_irr["binding"].mean())
  ```
  `run.py:303`
- **Data evidence:** `tables/stationary-moments.csv` has no column for this value. README prose assertion not backed by any committed artifact.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no -- descriptive statistic about the grid, not a published comparison number.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "grid_binding_pct" not in open("tables/stationary-moments.csv").read()
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(float(pd.read_csv("tables/stationary-moments.csv")["grid_binding_pct"].iloc[0]) - 9.5) < 0.5
  ```

### Finding 3: Stress-path binding 13.3% not in committed CSV (DATA DRIFT, LOW)

- **Claim source (verbatim):** "It binds for 13.3% of the stress path" -- `README.md:113`
- **Code evidence (verbatim):**
  ```python
  binding_share_stress = float(stress_irr["binding"].mean())
  ```
  `run.py:304`
- **Data evidence:** Not in `tables/stationary-moments.csv`. Runtime scalar embedded from prior run.
- **Category:** DATA DRIFT
- **Severity:** LOW
- **Result-changing:** no -- stress-path result, not in stationary-moments table.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "stress_binding_pct" not in open("tables/stationary-moments.csv").read()
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert abs(float(pd.read_csv("tables/stationary-moments.csv")["stress_binding_pct"].iloc[0]) - 13.3) < 1.0
  ```

## Cross-cutting patterns

- The three DATA DRIFT findings (iteration counts, 9.5%, 13.3%) share the same root cause: runtime scalars embedded in README prose from the generating run, with no committed CSV column to verify them. The stationary-moments CSV archives the result table but not these diagnostic scalars.
- The 0.42% stationary binding rate is the exception: it appears in both `I=0 frequency` column of `tables/stationary-moments.csv` and in README prose. That value is fully grounded.
- The main finding from the original audit (pseudocode said A_irr only) is resolved. The code behavior was always correct; the prose was corrected to match.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10%.** Below the 25% threshold. Safe to ship; the DATA DRIFT findings are low-severity prose annotations, not result-table errors.
1. The violated-invariant test for Finding 1 of original audit (`"boundary K'=(1-delta)K_m to A_irr" in SRC`) now FAILS, confirming the pseudocode fix removed the false claim.
2. The honest-fix test (`"to both A_std and A_irr" in SRC`) now PASSES.
3. Optional improvement for Findings 1-3 here: add `grid_binding_pct`, `stress_binding_pct`, and iteration counts to `tables/stationary-moments.csv` so all README prose numbers are verifiable without re-run.
4. No code changes needed to correct any result.
