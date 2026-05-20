# bullshit-detector — mixed-logit-simulation — 2026-05-20

**Bullshit score: 10%** — one DILUTED finding (Model Setup SD bounds row says "Applied to both random-coefficient standard deviations" but bounds are on log-sigma, not sigma; Solution Method prose repairs this one section later). All numeric claims match committed CSVs exactly. All algorithmic claims grounded line-by-line in code.

## Header

- Claim sources: `choice/mixed-logit-simulation/README.md`
- Code / artifact root: `choice/mixed-logit-simulation/run.py`
- Data artifacts: `tables/parameter-recovery.csv`, `tables/share-fit.csv`, `tables/price-substitution-matrix.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, six-pass)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | SD bounds [0.03, 1.30] "applied to random-coefficient standard deviations" | DILUTED | LOW | no |
| 2 | n_consumers=1500, J=4, R=120 | HOLDS | — | no |
| 3 | True theta=(-1.00,1.10,0.36,0.55) | HOLDS | — | no |
| 4 | L-BFGS-B, maxiter=220, start=(-0.75,0.85,log0.25,log0.35) | HOLDS | — | no |
| 5 | Price-taste bound [-3.00,-0.05]; quality-taste bound [0.05,2.50] | HOLDS | — | no |
| 6 | Probability floor 1e-14 | HOLDS | — | no |
| 7 | Profile grid 21x21 | HOLDS | — | no |
| 8 | Draws made once, held fixed (common random numbers) | HOLDS | — | no |
| 9 | P_hat=(1/R) sum P_ij(theta,nu_r) | HOLDS | — | no |
| 10 | SML objective = argmax sum log P_hat_{i,yi} | HOLDS | — | no |
| 11 | sigma=exp(log_sigma) inside estimator | HOLDS | — | no |
| 12 | Q_R=-ell_R/N (division by N) | HOLDS | — | no |
| 13 | D_jk formula (numerator/denominator) | HOLDS | — | no |
| 14 | D_kk=-1 | HOLDS | — | no |
| 15 | Price step Delta_p=0.10 | HOLDS | — | no |
| 16 | Parameter recovery table numbers vs CSV | HOLDS | — | no |
| 17 | Share fit table numbers vs CSV | HOLDS | — | no |
| 18 | Price substitution table numbers vs CSV | HOLDS | — | no |

## Findings

### Finding 1: SD bounds table row omits log-space qualifier

- **Claim source (verbatim):** "| SD bounds | [0.03, 1.30] | Applied to both random-coefficient standard deviations |" — `README.md:92`

- **Code evidence (verbatim):**
  ```python
  MIXED_BOUNDS = [
      (-3.0, -0.05),
      (0.05, 2.5),
      (np.log(0.03), np.log(1.3)),
      (np.log(0.03), np.log(1.3)),
  ]
  ```
  `run.py:19-24`

- **Data evidence (if applicable):** None. Bounds are not recorded in committed CSVs.

- **Category:** DILUTED

- **Severity:** LOW

- **Result-changing:** no

- **Explanation:** The third and fourth entries of `MIXED_BOUNDS` are `(np.log(0.03), np.log(1.3))` = `(-3.507, 0.262)`. These bounds are applied to the log-sigma parameters `log_sigma_alpha` and `log_sigma_beta`. The table row says "Applied to both random-coefficient standard deviations," implying the constraint sits on sigma directly. The effective sigma range is correctly stated as [0.03, 1.30] because `exp(-3.507) ≈ 0.03` and `exp(0.262) ≈ 1.30`, so the table value is accurate in effect. The gap is that the table row is silent about the log transform that mediates the constraint. The Solution Method section at `README.md:101-102` says "The standard deviations are optimized in logs. The optimizer can move freely over log standard deviations, while the model sees positive values after exponentiation." A reader who reads only the Model Setup table sees bounds applied to sigmas; the log parameterization is clarified one section later. No result changes because the optimizer behavior is unchanged.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "log" not in "Applied to both random-coefficient standard deviations"  # PASSES on current text (log qualifier absent); FAILS on honest fix
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(ph in row_text for ph in ["log-sigma", "log standard deviation", "log scale", "log SD"])  # PASSES if row names the log-space qualifier; FAILS on current text
  ```

## Cross-cutting patterns

- All numeric claims in README tables are byte-exact copies of committed CSV rows. No DATA DRIFT detected.
- All algorithmic claims (SML objective, simulated probability averaging, substitution matrix formula, sigma reparameterization, common draws) are implemented exactly as stated in pseudocode. Grounded `file:line` for each in summary table above.
- The single DILUTED finding is display-only: the Model Setup table row omits a qualifier that the Solution Method section supplies. No algorithmic or numeric claim is wrong.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 10%.** Below the 50% halt threshold. Proceed.
1. Finding 1 only. Write one pytest test under `tests/` that reads the committed `README.md` and asserts the SD bounds row contains a log-space qualifier. Test PASSES on current text (qualifier absent) — proves the gap. Test FAILS after the fix.
2. Fix: edit `run.py:488` (the SD bounds row string in `add_model_setup`) to say e.g. "Applied to log-sigma for both random coefficients (effective sigma range)". Regenerate README with `python run.py`.
3. Confirm pytest test now FAILS on the updated README (qualifier present). Run `scripts/validate_catalog.py` to confirm no math rendering regressions.
4. Re-run this skill on the updated README to confirm Finding 1 reads HOLDS and score drops to 0-5%.
