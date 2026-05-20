# bullshit-detector -- phase-diagrams -- 2026-05-20

**Bullshit score: 25%** -- one FALSE: README.md:109 and run.py:358-359 claim "Arrows explain motion" in the path-selection figure; the figure code (run.py:334-361) contains no quiver call, no annotate with arrowprops, and no arrow of any kind. Reviewer 2 catches it; substance survives revision. Diagram-only cap (25%) applies but the caption-figure contradiction fires the cap exception; score lands at cap ceiling, not above it.

## Header
- Claim sources: `optimal-control/phase-diagrams/README.md`
- Code / artifact root: `optimal-control/phase-diagrams/run.py`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: yes (cap ceiling = 25%; caption-figure contradiction exception fires, score stays at 25% not below)

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Arrows explain motion" in path-selection figure | FALSE | MED | no |
| 2 | k* = 5.5843 | HOLDS | -- | -- |
| 3 | c* = 1.3961 | HOLDS | -- | -- |
| 4 | lambda_s = -0.0710, lambda_u = 0.1110 | HOLDS | -- | -- |
| 5 | Stable eigenvector slope dc/dk = 0.1110 | HOLDS | -- | -- |
| 6 | Jacobian form J = [[f'(k*)-delta, -1],[c*f''(k*)/sigma, 0]] | HOLDS | -- | -- |
| 7 | Capital nullcline: c = f(k) - delta*k (blue curve) | HOLDS | -- | -- |
| 8 | Consumption nullcline: vertical at k* (red line) | HOLDS | -- | -- |
| 9 | Euler equation: dc/c = (f'(k)-delta-rho)/sigma | HOLDS | -- | -- |
| 10 | Backward integration: code integrates -F(k,c) | HOLDS | -- | -- |
| 11 | "Below blue curve, capital rises" | HOLDS | -- | -- |
| 12 | "Left of k*, consumption rises" | HOLDS | -- | -- |
| 13 | Stable arm is black curve | HOLDS | -- | -- |
| 14 | Dashed line is local linear approximation | HOLDS | -- | -- |

## Findings

### Finding 1: "Arrows explain motion" in the path-selection figure

- **Claim source (verbatim):** "Arrows explain motion, but the stable arm selects the path." -- `README.md:109`
- **Code evidence (verbatim):**
  ```python
  fig3, ax3 = plt.subplots(figsize=(9, 6.5))
  ax3.plot(k_range, c_nullcline, color="#1f77b4", linestyle="--", linewidth=1.8,
           label="$\\dot{k}=0$")
  ax3.axvline(k_ss, color="#c44e52", linestyle="--", linewidth=1.8,
              label="$\\dot{c}=0$")
  ax3.plot(k_stable, c_stable, color="0.55", linewidth=2.2, label="stable arm")
  for label, k_path, c_path, color in traced_selection:
      ax3.plot(k_path, c_path, color=color, linewidth=2.3, label=label)
      ax3.plot(k_path[0], c_path[0], marker="o", color=color, markersize=7)
  ax3.plot(k_ss, c_ss, "ko", markersize=9)
  ```
  `run.py:334-343`
- **Data evidence (if applicable):** The figure description passed to `report.add_figure` at `run.py:355-360` repeats the same claim verbatim: "Arrows explain motion, but the stable arm selects the path." No quiver call exists anywhere in the fig3 block (lines 334-361). The only quiver call in the file is at `run.py:290-300` inside the `fig1` (phase-diagram.png) block. The path-selection figure contains only `ax3.plot` and `ax3.axvline` calls -- no arrows of any form.
- **Category:** FALSE -- the figure code contains no arrows; the claim asserts arrows are present.
- **Severity:** MED -- non-quantitative finding; no number or result is changed, but the reader is told to look at a visual element that does not exist in the figure they are viewing.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "quiver" not in inspect.getsource(run.main).split("fig3")[1].split("fig3.tight_layout")[0]
  # PASSES on current buggy code (no quiver in fig3 block); FAILS on honest fix (quiver added)
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "quiver" in inspect.getsource(run.main).split("fig3")[1].split("fig3.tight_layout")[0]
  # PASSES on honest fix (quiver added to fig3); FAILS on current buggy code
  ```

## Cross-cutting patterns

- All numeric claims (k*, c*, eigenvalues, slope, Jacobian entries) are generated at runtime and embedded via f-strings. They cannot drift between README and code. This is a structural strength of the ModelReport pattern.
- The single FALSE finding is a documentation-generation gap: the figure description string at `run.py:355-360` was written to describe a richer figure (one with quiver arrows) but the fig3 code block was not updated to match. The quiver grid (K, C, DK_norm, DC_norm) computed at lines 127-134 is available and used only in fig1; it was not plumbed into fig3.
- No tables directory exists; the tutorial is diagram-only. Diagram-only cap (25%) applies.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Non-blocking. One finding only. Safe to proceed with a touch-up fix after surfacing to the user.
1. Turn the violated invariant into a pytest test under `tests/`. Confirm it PASSES on current code (proves the bug is real -- no quiver in fig3 block).
2. Convert the honest-fix pass condition into a second test that FAILS on current code.
3. Two fixes are possible; surface to user before choosing:
   - **Fix A (add arrows):** add a quiver call to fig3 using the already-computed DK_norm / DC_norm grid, matching the fig1 quiver call. The figure then matches the description.
   - **Fix B (remove arrow claim):** remove "Arrows explain motion, but the stable arm selects the path." from the `description=` argument at `run.py:355-360` (and regenerate README.md). The description then matches the figure.
4. After fix, violated-invariant test FAILS, pass-condition test PASSES. Regenerate README.md. Re-run this skill to confirm all findings read HOLDS and score drops to 0%.
