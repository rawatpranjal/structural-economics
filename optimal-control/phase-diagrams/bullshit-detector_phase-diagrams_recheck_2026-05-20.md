# bullshit-detector -- phase-diagrams_recheck -- 2026-05-20

**Bullshit score: 0%** -- all claims HOLDS; diagram-only cap applies (cap ceiling 25%); no caption-figure contradiction found in post-fix code; score lands at 0%, below cap.

## Header
- Claim sources: `optimal-control/phase-diagrams/README.md`
- Code / artifact root: `optimal-control/phase-diagrams/run.py`
- Seed audit (if any): `optimal-control/phase-diagrams/bullshit-detector_phase-diagrams_2026-05-20.md`
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, independent recheck)
- Date: 2026-05-20
- Diagram-only cap applied: yes (cap ceiling = 25%; no exception fires; score 0%)

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | k* = 5.5843 | HOLDS | -- | -- |
| 2 | c* = 1.3961 | HOLDS | -- | -- |
| 3 | lambda_s = -0.0710 | HOLDS | -- | -- |
| 4 | lambda_u = 0.1110 | HOLDS | -- | -- |
| 5 | Stable eigenvector slope dc/dk = 0.1110 | HOLDS | -- | -- |
| 6 | Jacobian J = [[f'(k*)-delta, -1],[c*f''(k*)/sigma, 0]] | HOLDS | -- | -- |
| 7 | Capital nullcline: c = f(k) - delta*k (blue curve) | HOLDS | -- | -- |
| 8 | Consumption nullcline: vertical at k* (red line) | HOLDS | -- | -- |
| 9 | Euler equation: dc/c = (f'(k)-delta-rho)/sigma | HOLDS | -- | -- |
| 10 | Backward integration: code integrates -F(k,c) | HOLDS | -- | -- |
| 11 | "Below blue curve, capital rises" | HOLDS | -- | -- |
| 12 | "Left of k*, consumption rises" | HOLDS | -- | -- |
| 13 | Stable arm is black curve | HOLDS | -- | -- |
| 14 | Dashed line is local linear approximation | HOLDS | -- | -- |
| 15 | "Arrows explain motion" in path-selection figure | HOLDS | -- | -- |

## Findings

None. All 15 claims ground to verbatim code or independently verified numeric output. See cross-cutting patterns for evidence citations.

### Claim 15 resolution (the original FALSE finding)

- **Claim source (verbatim):** "Arrows explain motion, but the stable arm selects the path." -- `README.md:109`
- **Code evidence (verbatim):**
  ```python
  fig3, ax3 = plt.subplots(figsize=(9, 6.5))
  ax3.quiver(
      K,
      C,
      DK_norm,
      DC_norm,
      speed,
      cmap="viridis",
      alpha=0.35,
      scale=32,
      width=0.003,
  )
  ```
  `run.py:334-345`
- **Category:** HOLDS -- `ax3.quiver(...)` is present in the fig3 block. The caption claim is now truthful.
- **Severity:** none
- **Result-changing:** no

## Cross-cutting patterns

- All numeric claims (k*, c*, eigenvalues, slope) are runtime-embedded via f-strings. Independently re-derived: k*=5.5843, c*=1.3961, lambda_s=-0.0710, lambda_u=0.1110, slope=0.1110. Zero drift possible between README and code.
- The quiver grid (K, C, DK_norm, DC_norm, speed) is computed at `run.py:127-134` and is now used in both fig1 (`run.py:290-300`) and fig3 (`run.py:335-345`). The structural gap from the original audit (grid computed but not plumbed into fig3) is closed.
- fig3 stable arm plotted at `run.py:350` uses `color="0.55"` (gray), not black. The README prose at `README.md:105` refers to "the black curve is the stable arm" in the context of fig1 (phase-diagram.png), where `run.py:305` uses `color="black"`. No contradiction.
- No tables directory exists. Tutorial is diagram-only. Diagram-only cap (25%) applies; no findings fire the cap exception.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 0%.** All findings HOLDS. No fixes required.
1. The three existing tests in `tests/test_run.py` reflect the corrected state:
   - `test_fig3_block_isolated` -- PASSES (infrastructure check)
   - `test_finding1_violated_invariant__fig3_has_no_quiver` -- FAILS (expected: violated-invariant test correctly fails on fixed code)
   - `test_finding1_honest_fix__fig3_draws_quiver` -- PASSES (honest-fix condition satisfied)
   - `test_finding1_arrow_claim_backed_by_arrows` -- PASSES
2. No further action needed. Re-running this skill after any future README or run.py change is the correct verification gate.
