# bullshit-detector - particle-filter - 2026-05-20

**Bullshit score: 25%** - One DILUTED finding: the measurement-noise sweep silently uses 350 particles / 20 runs while the Model Setup table advertises 500 / 50, leaving the reader with an incorrect mental model of what produced the sweep table.

## Header

- Claim sources: `computational-methods/particle-filter/README.md`
- Code / artifact root: `computational-methods/particle-filter/run.py`
- Data artifacts: `computational-methods/particle-filter/tables/filter-summary.csv`, `computational-methods/particle-filter/tables/measurement-noise-sweep.csv`
- Seed audit (if any): None
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Sweep uses same 500 particles / 50 runs as baseline | DILUTED | MED | no (sweep numbers correct; config undisclosed) |
| 2 | PSI = [1.0, 0.9] | HOLDS | - | - |
| 3 | PHI = diag(0.4, 0.5) | HOLDS | - | - |
| 4 | Measurement std = 0.10 | HOLDS | - | - |
| 5 | Process std = (0.30, 0.25) | HOLDS | - | - |
| 6 | Baseline particles = 500 | HOLDS | - | - |
| 7 | Repeated runs = 50 | HOLDS | - | - |
| 8 | Bootstrap PF RMSE = 0.0273 | HOLDS | - | - |
| 9 | Optimal PF RMSE = 0.0100 | HOLDS | - | - |
| 10 | Bootstrap Mean ESS = 121.829 | HOLDS | - | - |
| 11 | Optimal Mean ESS = 492.636 | HOLDS | - | - |
| 12 | Bootstrap Loglike sd = 0.6914 | HOLDS | - | - |
| 13 | Optimal Loglike sd = 0.0397 | HOLDS | - | - |
| 14 | All 8 sweep table rows match CSV | HOLDS | - | - |
| 15 | Bootstrap weights = observation likelihood only | HOLDS | - | - |
| 16 | Optimal proposal conditions on y_t before drawing state | HOLDS | - | - |
| 17 | ESS = 1 / sum_i (w_i)^2 on normalized weights | HOLDS | - | - |

## Findings

### Finding 1: Measurement-noise sweep uses undisclosed particle/run counts

- **Claim source (verbatim):** "The baseline repeats each filter 50 times with 500 particles. It compares each particle mean with the Kalman mean." - `README.md:149`, immediately preceding the measurement-noise sweep table.

  Model Setup table also states:
  "| Baseline particles | 500 |" - `README.md:107`
  "| Repeated runs | 50 |" - `README.md:108`

- **Code evidence (verbatim):**
  ```python
  measurement_table = measurement_noise_sweep(n_periods, n_particles=350, n_runs=20)
  ```
  `run.py:284`

  The function signature:
  ```python
  def measurement_noise_sweep(
      n_periods: int,
      n_particles: int,
      n_runs: int,
  ) -> pd.DataFrame:
      """Compare particle accuracy as observation noise shrinks."""
      rows = []
      for measurement_std in [0.25, 0.15, 0.10, 0.05]:
          states, observations = simulate_state_space(n_periods, measurement_std, seed=777)
          kalman = kalman_filter(observations, measurement_std)
          for method in ["bootstrap", "optimal"]:
              mse, loglikes, ess = repeated_filter_mse(
  ```
  `run.py:207-218`

- **Data evidence:** `tables/measurement-noise-sweep.csv` has no `Particles` column; the committed CSV does not record the 350/20 config used to produce it.

- **Category:** DILUTED - the code does compute a genuine measurement-noise sweep, but the load-bearing particle count and run count for that sweep are neither disclosed in the README nor present in the CSV artifact. The Model Setup table entries (500, 50) correctly describe the baseline run only, but their placement and the prose sentence at README.md:149 lead a careful reader to infer both experiments share the same config.

- **Severity:** MED - no numeric result in the README or CSV is wrong; the config difference is real and materially affects ESS and RMSE levels in the sweep table (350 particles vs 500 = ~30% fewer particles; 20 runs vs 50 = noisier estimates), but the directional conclusions hold.

- **Result-changing:** no - the qualitative lesson (bootstrap ESS collapses as measurement_std falls; optimal ESS stays high) is robust to the config difference. However, a reader trying to reproduce the sweep table with the advertised 500/50 will get different numbers.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "n_particles=350" in open("/Users/pranjal/Code/computational-economics/computational-methods/particle-filter/run.py").read()
  # PASSES on current code (proves the undisclosed config); FAILS on honest fix that uses 500
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "n_particles=500" in open("/Users/pranjal/Code/computational-economics/computational-methods/particle-filter/run.py").read() or "Particles" in open("/Users/pranjal/Code/computational-economics/computational-methods/particle-filter/tables/measurement-noise-sweep.csv").read()
  # PASSES on honest fix (either uses 500 particles or adds Particles column to sweep CSV); FAILS on current code
  ```

## Cross-cutting patterns

- The only gap is a silent config inconsistency between two experimental blocks in `main()`. The baseline block (lines 264-283) uses `n_particles=500, n_runs=50`. The sweep call (line 284) uses `n_particles=350, n_runs=20`. Neither the README nor the CSV artifact discloses the sweep-specific values. This is the canonical "quick sweep with lighter settings that never got documented" pattern.
- All algorithmic claims (bootstrap weight formula, optimal proposal derivation, ESS formula, normalization, resampling) are faithfully implemented. No parametric claim is wrong. This is a documentation gap, not an implementation gap.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25%.** Below the 50% halt threshold. Surface to the user as a documentation fix, not a code fix.
1. Turn the violated invariant into a test under `tests/`:
   ```python
   # test_particle_filter_config.py
   def test_sweep_uses_documented_particle_count():
       src = open("computational-methods/particle-filter/run.py").read()
       assert "n_particles=350" in src  # PASSES now; proves the undisclosed config
   ```
2. Convert the honest-fix condition into a second test that FAILS now:
   ```python
   def test_sweep_config_matches_readme():
       src = open("computational-methods/particle-filter/run.py").read()
       # After fix: either sweep uses 500, or README explicitly documents 350
       assert "n_particles=500" in src or "350" in open("computational-methods/particle-filter/README.md").read()
   ```
3. Fix options (user's choice):
   a. Change `run.py:284` to `measurement_noise_sweep(n_periods, n_particles=500, n_runs=50)` and regenerate tables.
   b. Add a "Sweep particles | 350" and "Sweep runs | 20" row to the Model Setup table and regenerate README.
   c. Add a `Particles` column to `measurement-noise-sweep.csv` (regenerate) so the artifact is self-documenting.
4. Re-run this skill after the fix to confirm Finding 1 reads HOLDS and score drops to 0-10%.
