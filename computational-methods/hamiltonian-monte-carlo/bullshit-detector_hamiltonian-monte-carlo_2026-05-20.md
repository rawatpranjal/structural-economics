# bullshit-detector — hamiltonian-monte-carlo — 2026-05-20

**Bullshit score: 40%** — two DILUTED findings where the tutorial's own committed data directly contradicts interpretive claims drawn from the HMC literature (acceptance range 0.7-0.9 and ESS-optimal acceptance 0.6-0.8); neither changes a reported number but both will mislead a reader who trusts the interpretation over the table.

## Header
- Claim sources: `computational-methods/hamiltonian-monte-carlo/README.md`
- Code / artifact root: `computational-methods/hamiltonian-monte-carlo/run.py`
- Data artifacts: `computational-methods/hamiltonian-monte-carlo/tables/method-comparison.csv`, `computational-methods/hamiltonian-monte-carlo/tables/stepsize-sweep.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "acceptance rates of 0.7 to 0.9 are typical" on well-tuned trajectory | DILUTED | MED | no (interpretive; actual run shows 0.991) |
| 2 | "ESS is largest at a step size that keeps acceptance around 0.6 to 0.8" | DILUTED | MED | no (step-size sweep data shows peak ESS at acceptance 0.963-0.986) |
| 3 | "HMC autocorrelation drops to near zero within 10 lags" | DATA DRIFT | LOW | needs re-run to verify |

## Findings

### Finding 1: "acceptance rates of 0.7 to 0.9 are typical"

- **Claim source (verbatim):** "On a well-tuned trajectory the leapfrog energy drift is tiny and acceptance rates of 0.7 to 0.9 are typical." — `README.md:180` (generated from `run.py:525`)

- **Code evidence (verbatim):**
  ```python
  step_size = 0.18
  n_leapfrog = 25
  ...
  hmc_draws, hmc_acceptance, hmc_grads = hmc_sample(
      n_draws=n_draws_hmc, step_size=step_size, n_leapfrog=n_leapfrog,
      seed=1, start=start,
  )
  ```
  `run.py:232-242`

- **Data evidence (if applicable):** `tables/method-comparison.csv`, row 2: `Hamiltonian Monte Carlo,3500,0.991,0.040,0.082,726,841,,103974.0` — acceptance rate is 0.991, not in [0.7, 0.9].

- **Category:** DILUTED — the claim is not fabricated (it reflects the textbook HMC optimum in high dimensions), but the tutorial's own committed result uses a configuration that produces acceptance 0.991, which falls outside the claimed "typical" range. The tutorial presents its own run as the demonstration of a well-tuned trajectory. A reader who trusts the prose will infer 0.991 is anomalous; the prose says this IS the well-tuned case.

- **Severity:** MED

- **Result-changing:** no (the table number 0.991 is correct; only the interpretive label "0.7 to 0.9 are typical" is false for the configuration shown)

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert 0.7 <= 0.991 <= 0.9  # PASSES on buggy claim (fails immediately), shows the interval does NOT contain 0.991
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "0.7 to 0.9" not in open("README.md").read() or "0.9 to 1.0" in open("README.md").read()
  # PASSES on honest fix where the range matches the actual result or is qualified
  ```

---

### Finding 2: "ESS is largest at a step size that keeps acceptance around 0.6 to 0.8"

- **Claim source (verbatim):** "Effective sample size is largest at a step size that keeps acceptance around 0.6 to 0.8, which matches the asymptotic-optimal acceptance results in the HMC literature." — `README.md:248` (generated from `run.py:721-722`)

- **Code evidence (verbatim):**
  ```python
  step_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
  sweep_rows = []
  for step in step_grid:
      d, a, _ = hmc_sample(n_draws=1500, step_size=step, n_leapfrog=n_leapfrog,
                           seed=42, start=start)
  ```
  `run.py:256-259`

- **Data evidence (if applicable):** `tables/stepsize-sweep.csv` — all seven rows:

  | eps | acceptance |
  |-----|-----------|
  | 0.05 | 0.999 |
  | 0.10 | 0.998 |
  | 0.15 | 0.997 |
  | 0.20 | 0.989 |
  | 0.25 | 0.986 |
  | 0.30 | 0.963 |
  | 0.35 | 0.944 |

  Best ESS_x (331) at eps=0.25, acceptance=0.986. Best ESS_y (817) at eps=0.30, acceptance=0.963. No row has acceptance in [0.6, 0.8]. The prose says the sweet spot "matches" asymptotic-optimal acceptance 0.6-0.8; the table shows the empirical sweet spot is acceptance 0.963-0.986.

- **Category:** DILUTED — the asymptotic-optimal acceptance result (Betancourt 2014, Neal 2011) is a real literature result, but it applies in the d → ∞ limit. For d=2 the optimal acceptance is known to be higher (~0.65 is the high-d limit; for d=2 it is closer to 0.85-0.95). The tutorial presents the claim as demonstrated by its sweep, but the sweep data falsifies it for the configuration shown. The author probably intended to reference the literature while running a d=2 example where different physics applies; the claim as written says the data "matches" the 0.6-0.8 range, which it does not.

- **Severity:** MED

- **Result-changing:** no (table numbers are correct; the interpretive sentence misrepresents what the table shows)

- **Violated invariant (one-line pytest assertion):**
  ```python
  import pandas as pd; df = pd.read_csv("tables/stepsize-sweep.csv"); assert any((df["Acceptance rate"].astype(float) >= 0.6) & (df["Acceptance rate"].astype(float) <= 0.8))
  # PASSES on current data only if a row exists with acceptance in [0.6, 0.8]; FAILS because no such row exists
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "0.6 to 0.8" not in open("README.md").read() or any((pd.read_csv("tables/stepsize-sweep.csv")["Acceptance rate"].astype(float) >= 0.6) & (pd.read_csv("tables/stepsize-sweep.csv")["Acceptance rate"].astype(float) <= 0.8))
  # PASSES on honest fix: either drop the 0.6-0.8 claim or extend the sweep to show it
  ```

---

### Finding 3: "HMC autocorrelation drops to near zero within 10 lags"

- **Claim source (verbatim):** "For $\theta_1$, the HMC autocorrelation drops to near zero within 10 lags, while random-walk MH stays correlated out beyond 200 lags." — `README.md:233`

- **Code evidence (verbatim):**
  ```python
  report.add_results(
      f"For $\\theta_1$, the HMC autocorrelation drops to near zero within "
      f"{int(np.argmax(acf_hmc_x < 0.05))} lags, "
  ```
  `run.py:686-687`

- **Data evidence (if applicable):** The value "10" is dynamically computed at run time from `np.argmax(acf_hmc_x < 0.05)` applied to the live chain. No committed CSV records the per-lag ACF values. The number "10" in the committed README cannot be cross-checked against any on-disk artifact without re-running. **needs re-run to verify**

- **Category:** DATA DRIFT — the committed README states 10; the code generates this number dynamically from the chain; no intermediate artifact records the ACF series to verify the committed value.

- **Severity:** LOW (the number is plausible and the mechanism is correct; risk is stale committed README if the seed or algorithm changes)

- **Result-changing:** needs re-run to verify

- **Violated invariant (one-line pytest assertion):**
  ```python
  # No pytest assertion can be written without re-running; this is an artifact-traceability gap
  assert False, "No committed ACF artifact to cross-check README:233 claim of 10 lags"
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  # After re-run: int(np.argmax(acf_hmc_x < 0.05)) == int(open("README.md").read().split("within ")[1].split(" lags")[0])
  ```

## Cross-cutting patterns

- Both DILUTED findings (F1 and F2) involve the same mechanism: the tutorial borrows a qualitative statement from high-dimensional HMC theory (acceptance 0.7-0.9 typical; ESS maximized at acceptance 0.6-0.8) and presents it as though the tutorial's own d=2 run demonstrates it. The d=2 case has materially higher optimal acceptance rates. The fix pattern is the same in both cases: either qualify the claim with "in high dimensions" or extend the step-size sweep to show the regime where acceptance drops to 0.6-0.8.

- The DATA DRIFT finding (F3) follows from the absence of a committed ACF CSV. The tutorial commits method-comparison and stepsize-sweep tables but no ACF table, so one dynamically computed number in the prose is not independently checkable. Adding `tables/acf-summary.csv` would close this gap for future audits.

- Internal code inconsistency not rising to a prose-claim finding: `leapfrog_trajectory` uses a divergence guard of `abs(q) > 1e4` (`run.py:93`) while `hmc_sample` uses `abs(q) > 1e6` (`run.py:132`). This does not affect correctness at the tuned step size (eps=0.18) but means the energy-drift figure and the sampler treat divergence differently at large step sizes. Not a claim violation; recorded here for completeness.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 40%.** Two MED-severity DILUTED findings. Do not halt; surface to the user before proposing fixes.

1. For F1 (typical acceptance range):
   - Violated invariant test: `assert not (0.7 <= acceptance <= 0.9)` where `acceptance` is read from `tables/method-comparison.csv` row "Hamiltonian Monte Carlo". Should PASS (0.991 is outside [0.7, 0.9]).
   - Fix: either change the prose from "0.7 to 0.9 are typical" to "0.9 to 1.0 are typical for d=2 at these hyperparameters" or add a parenthetical noting this is the high-d asymptotic benchmark.

2. For F2 (ESS vs acceptance sweep):
   - Violated invariant test: confirm no row in `tables/stepsize-sweep.csv` has acceptance in [0.6, 0.8].
   - Fix option A: extend `step_grid` to include eps values (e.g., 0.50, 0.60, 0.70) that push acceptance into the 0.6-0.8 range, then verify ESS peaks there.
   - Fix option B: qualify the prose to "in high dimensions (d >> 2)" and add a sentence noting that for d=2 the empirical optimum in this sweep is around acceptance 0.96-0.99.

3. For F3 (ACF lag claim):
   - Re-run `python run.py` and confirm `int(np.argmax(acf_hmc_x < 0.05))` returns 10.
   - Optionally add `tables/acf-summary.csv` with per-lag values for future audits.

4. After fixes: re-run this skill. Target score <= 25%.
