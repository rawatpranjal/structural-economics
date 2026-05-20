# bullshit-detector — huggett-aggregate-risk-srl-recheck — 2026-05-20

**Bullshit score: 15%** — one DILUTED/LOW finding: Results-section preamble asserts "Aggregate consumption should move with aggregate income but be smoother" without a run-conditional qualification; results tables and Takeaway are fully honest. All five original HIGH/MED findings are now HOLDS.

## Header

- Claim sources: `heterogeneous-agents/huggett-aggregate-risk-srl/README.md`
- Code / artifact root: `heterogeneous-agents/huggett-aggregate-risk-srl/run.py`
- Data artifacts: `tables/diagnostics.csv`, `tables/hyperparameters.csv`, `tables/paper_benchmark.csv`, `tables/calibration.csv`
- Seed audit: `heterogeneous-agents/huggett-aggregate-risk-srl/bullshit-detector_huggett-aggregate-risk-srl_2026-05-20.md`
- Run by: bullshit-detector skill (Claude Sonnet 4.6), independent re-audit
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Takeaway: "smoother aggregate consumption" conditional on ratio < 1 | HOLDS | - | resolved |
| 2 | Benchmark table: grid row profile-conditional, "Not matched" for quick | HOLDS | - | resolved |
| 3 | diagnostics_table uses profile.convergence_tol, not FULL_PROFILE | HOLDS | - | resolved |
| 4 | Market-clearing residual labeled "Zero by construction" | HOLDS | - | resolved |
| 5 | Equations section documents L2 regularization with kappa=1e-5 | HOLDS | - | resolved |
| 6 | Results preamble: "Aggregate consumption should... be smoother" unqualified | DILUTED | LOW | no |

## Findings

### Finding 1 (original): Takeaway smoothing claim — HOLDS

- **Claim source (verbatim):** "an aggregate consumption volatility ratio of 1.000, so this short run does not yet reproduce the consumption-smoothing result" — `README.md:182`
- **Code evidence (verbatim):**
  ```python
  if round(volatility_ratio, 3) < 1.0:
      smoothing_clause = (
          "aggregate consumption smoother than income "
          f"(volatility ratio {volatility_ratio:.3f})"
      )
  else:
      smoothing_clause = (
          "an aggregate consumption volatility ratio of "
          f"{volatility_ratio:.3f}, so this short run does not yet reproduce "
          "the consumption-smoothing result"
      )
  ```
  `run.py:1251-1261`
- **Data evidence:** `tables/diagnostics.csv:14` — `Aggregate consumption volatility divided by income volatility,1`. Ratio = 1.000 >= 1.0 triggers the honest else-branch.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no — fix confirmed

---

### Finding 2 (original): Grid and training row profile-conditional — HOLDS

- **Claim source (verbatim):** "Reduced quick grid: 56 bond points, b_max=12, 7 aggregate states, 9 rate points, T=44, 90 epochs, batch=14" / "Not matched" — `README.md:175`
- **Code evidence (verbatim):**
  ```python
  grid_matches_benchmark = profile == FULL_PROFILE
  if grid_matches_benchmark:
      grid_tutorial_run = (
          "Uses the published grid, horizon, learning-rate schedule, and "
          "batch size"
      )
      grid_status = "Matched"
  else:
      grid_tutorial_run = (
          f"Reduced quick grid: {profile.asset_points} bond points, "
          f"b_max={profile.asset_upper:g}, {profile.aggregate_states} "
          f"aggregate states, {profile.rate_points} rate points, "
          f"T={profile.horizon}, {profile.epochs} epochs, "
          f"batch={profile.batch_size}"
      )
      grid_status = "Not matched"
  ```
  `run.py:774-789`
- **Data evidence:** `tables/paper_benchmark.csv:3` — Assessment = `Not matched`; Tutorial run = `Reduced quick grid: 56 bond points, b_max=12, 7 aggregate states, 9 rate points, T=44, 90 epochs, batch=14`. All grid numbers verified against `tables/hyperparameters.csv`.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no — fix confirmed

---

### Finding 3 (original): diagnostics_table uses profile.convergence_tol — HOLDS

- **Claim source (verbatim):** "Convergence threshold | 0.0001" — `README.md:157`
- **Code evidence (verbatim):**
  ```python
  ("Convergence threshold", f"{profile.convergence_tol:.6g}"),
  ```
  `run.py:714` — uses the active `profile` argument, not `FULL_PROFILE`. Confirmed by grep: `FULL_PROFILE.convergence_tol` does not appear in run.py.
- **Data evidence:** `tables/diagnostics.csv:5` — `Convergence threshold,0.0001`. `tables/hyperparameters.csv:15` — Tutorial setting = `0.0001`. Both match `QUICK_PROFILE.convergence_tol = 1.0e-4`.
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no — fix confirmed

---

### Finding 4 (original): Market-clearing residual labeled "Zero by construction" — HOLDS

- **Claim source (verbatim):** "Zero by construction" — `README.md:177` (Assessment column)
- **Code evidence (verbatim):**
  ```python
  residual_status = "Zero by construction" if bracketing_share > 0.95 else "Mixed"
  ```
  `run.py:769` — runtime-computed, not hardcoded "Matched".
- **Data evidence:** `tables/paper_benchmark.csv:5` — Assessment = `Zero by construction`. `tables/diagnostics.csv:10` — `Share of periods with a bracketing root,1.000` (= 1.000 > 0.95).
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no — fix confirmed

---

### Finding 5 (original): L2 regularization documented in Equations — HOLDS

- **Claim source (verbatim):** "$$J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T-1}\beta^t u(c_t)\right] - \kappa\,\overline{\theta^2}, \qquad \kappa = 10^{-5}.$$" and "The penalty coefficient $\kappa$ is small relative to per-period utility, so it regularizes the parameters without materially distorting the policy." — `README.md:33-36`
- **Code evidence (verbatim):**
  ```python
  objective_value = jnp.mean(returns) - 1.0e-5 * jnp.mean(theta**2)
  ```
  `run.py:336`
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no — fix confirmed

---

### Finding 6 (new): Results preamble asserts smoothing without run-conditional qualification

- **Claim source (verbatim):** "Aggregate consumption should move with aggregate income but be smoother." — `README.md:90`
- **Code evidence (verbatim):**
  ```python
  report.add_results(
      "The figures below follow the objects emphasized in the published SRL "
      "Huggett exercise. The consumption policy should rise with bond holdings "
      "and flatten out for wealthier households. Aggregate consumption should "
      "move with aggregate income but be smoother. The interest rate is "
      "endogenous, and the saving schedule should cross zero where the bond "
      "market clears."
  )
  ```
  `run.py:1217-1224` — this string is unconditional; it does not branch on `volatility_ratio` or `profile`.
- **Data evidence:** `tables/diagnostics.csv:14` — C/Y volatility ratio = 1.000, meaning this run does NOT show aggregate consumption smoother than income. `README.md:178` correctly reports `C/Y volatility ratio 1.000; Assessment: Mixed`. `README.md:182` (Takeaway) correctly states "does not yet reproduce the consumption-smoothing result." The preamble and the results/takeaway disagree on whether smoothing was achieved.
- **Category:** DILUTED — the Results section design-goal preamble lists smoothing as a target without noting that this particular run may fail it; the actual results and takeaway correctly walk it back. A reader skimming only the preamble prose at README.md:90 before the tables forms the wrong impression.
- **Severity:** LOW — the tables and Takeaway are unambiguous; no reader who reads past the preamble is misled.
- **Result-changing:** no — results tables and Takeaway are correct; only the introductory framing prose is unqualified.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "Aggregate consumption should move with aggregate income but be smoother" in open("README.md").read()  # PASSES on current code; FAILS on honest fix
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "Aggregate consumption should move with aggregate income but be smoother" not in open("README.md").read() or "but this run" in open("README.md").read()  # PASSES on honest fix (either phrase removed or qualified)
  ```

---

## Cross-cutting patterns

- All five original HIGH/MED findings (two FALSE/HIGH, one DATA DRIFT/MED, one MISLABELED/MED, one DILUTED/LOW) are now HOLDS. The opus agent's fixes were faithful and complete for those findings.
- The one remaining gap (Finding 6, DILUTED/LOW) is a cosmetic framing issue: the Results section preamble describes design goals of the SRL exercise unconditionally, while the actual results and Takeaway are fully honest about the run's failure to reproduce consumption smoothing. This is a documentation polish gap, not a result-changing error.
- The `add_results` preamble at `run.py:1217-1224` is not profile-conditional. It would need to branch on `volatility_ratio` or `profile` to be fully consistent with the rest of the honest-fix logic. The same pattern that fixed the Takeaway (conditional on `round(volatility_ratio, 3) < 1.0`) could be applied here.
- No cross-artifact inconsistency detected between diagnostics.csv, hyperparameters.csv, paper_benchmark.csv, and README.md for any numeric claim.

## TDD execution sequence (for the next agent)

0. **Bullshit score = 15%. All original result-changing findings resolved. One LOW-severity preamble gap remains.**

1. For Finding 6: turn the violated invariant into a pytest test under `tests/`. Confirm it PASSES on current code.
2. Convert the honest-fix pass condition into a second test that FAILS on current code.
3. Fix: make `add_results` preamble at `run.py:1217-1224` conditional on `volatility_ratio`. When `round(volatility_ratio, 3) >= 1.0`, replace "be smoother" with "be smoother in the full run (this quick run does not yet achieve it)" or restructure the sentence to frame it as a design goal of the exercise rather than a result claim.
4. Re-run `python run.py` (QUICK profile). Confirm `README.md:90` no longer asserts unconditional smoothing. Re-run this skill to confirm Finding 6 becomes HOLDS and score drops to 0-10%.
