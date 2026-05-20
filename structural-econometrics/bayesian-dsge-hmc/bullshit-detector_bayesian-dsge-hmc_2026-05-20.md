# bullshit-detector — bayesian-dsge-hmc — 2026-05-20

**Bullshit score: 35%** — one MISLABELED (sequential chains called "parallel") and one DILUTED HIGH (ESS comparison does not equalize compute time; per-second NUTS loses on 5/8 parameters while the Takeaway claims "order of magnitude more effective draws per compute unit").

## Header
- Claim sources: `structural-econometrics/bayesian-dsge-hmc/README.md`
- Code / artifact root: `structural-econometrics/bayesian-dsge-hmc/run.py`, `lib/perturbation_jax.py`, `lib/kalman_jax.py`
- Data artifacts: `structural-econometrics/bayesian-dsge-hmc/tables/posterior_summary.csv`
- Seed audit (if any): none
- Run by: bullshit-detector skill (Claude Sonnet)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Klein A, B matrices match code | HOLDS | - | no |
| 2 | 2x2 worked-example math (psi=4/3, gradients 8/9, 4/3, 8/9) | HOLDS | - | no |
| 3 | Kalman log-likelihood formula matches code | HOLDS | - | no |
| 4 | Prior hyperparameters and means match code | HOLDS | - | no |
| 5 | Jacobian log-correction formula matches code | HOLDS | - | no |
| 6 | Table numbers match posterior_summary.csv | HOLDS | - | no |
| 7 | RW-MH draws = 8000 (single chain) | HOLDS | - | no |
| 8 | Four chains run in parallel | MISLABELED | MED | no (ESS/posterior unaffected; wall-time is sequential sum) |
| 9 | "order of magnitude more effective draws per compute unit" | DILUTED | HIGH | yes (NUTS loses per-second on 5/8 params when compute unit = wall time) |
| 10 | "ESS is on a log scale" (after table, before figure) | DILUTED | LOW | no (figure has log axis; table values are raw; sentence placement is ambiguous) |
| 11 | Runtime numbers (520.2s NUTS, 3.0s RW-MH, 0.91 accept, 0.02 accept) | DATA DRIFT | LOW | needs re-run to verify |

## Findings

### Finding 1 (HOLDS batch): Structural equations, matrices, Kalman, priors, Jacobian, table numbers

- **Claim source (verbatim):** Multiple assertions in `README.md:16-75` covering A/B matrices, 2x2 example, Kalman formula, prior hyperparameters, Jacobian, posterior table.
- **Code evidence:** `run.py:83-95` (build_AB), `run.py:154-168` (transform_z_to_theta Jacobian), `run.py:171-179` (log_prior), `lib/kalman_jax.py:76-94` (Kalman step), `lib/perturbation_jax.py:120-139` (_klein_core primal).
- **Data evidence:** `tables/posterior_summary.csv` — all eight rows match `README.md:126-134` to full floating-point precision (e.g. sigma post mean: 0.877018645922225 in CSV matches 0.877019 in README).
- **Category:** HOLDS
- **Severity:** none
- **Result-changing:** no

### Finding 2: "Four chains run in parallel"

- **Claim source (verbatim):** "Four chains run in parallel from near-zero unconstrained starting points." — `README.md:85`
- **Code evidence (verbatim):**
  ```python
  for c in range(num_chains):
      pos, accept = jax.jit(run_chain)(keys[c], init_z + 0.01 * jax.random.normal(keys[c], init_z.shape))
      positions.append(np.asarray(pos))
      accept_rates.append(float(accept.mean()))
  ```
  `run.py:280-283`
- **Data evidence:** None directly, but the reported wall time of 520.2 seconds is consistent with sequential execution of four independent chains. If chains ran in parallel (e.g. via `jax.vmap` or `jax.pmap`), wall time would be roughly one-quarter of the sequential sum.
- **Category:** MISLABELED — code runs chains in a sequential Python `for`-loop, one after the other. The word "parallel" describes the conceptual independence of chains, not the execution model. The ESS numbers, posterior, and R-hat are unaffected.
- **Severity:** MED
- **Result-changing:** no — ESS and posterior samples are identical whether chains run sequentially or in parallel; the only affected quantity is wall time, and the 520.2 second figure is the sequential total. The figure caption says "NUTS (4 chains x 2000 draws)" which is accurate; only the word "parallel" is wrong.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "jax.vmap" not in inspect.getsource(run_nuts) and "pmap" not in inspect.getsource(run_nuts)
  # PASSES on current code (no parallel dispatch); FAILS if an honest fix adds vmap/pmap
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert any(kw in inspect.getsource(run_nuts) for kw in ("jax.vmap", "pmap", "lax.map"))
  # PASSES on honest parallel fix; FAILS on current sequential for-loop
  ```

### Finding 3: "order of magnitude more effective draws per compute unit"

- **Claim source (verbatim):** "gradient-based sampling delivers an order of magnitude more effective draws per compute unit than the random-walk baseline." — `README.md:142` (Takeaway)
- **Code evidence (verbatim):**
  ```python
  rwmh_pos, rwmh_accept = run_rwmh(log_density, init_z,
                                   jax.random.PRNGKey(7),
                                   NUM_SAMPLES * NUM_CHAINS, step_size=0.1)
  ```
  `run.py:342-344` — RW-MH receives `NUM_SAMPLES * NUM_CHAINS = 8000` draws, not a wall-time-equalized budget. NUTS also runs 8000 draws (4 chains x 2000) plus 4000 warmup, consuming 520.2 seconds. RW-MH runs 8000 draws in 3.0 seconds.
- **Data evidence:** From `tables/posterior_summary.csv` and reported runtimes (needs re-run to verify exact seconds):
  - NUTS ESS per second (520.2s total): sigma=4.21, phi_pi=5.77, phi_y=7.04, kappa=4.04, sigma_v=7.41, rho_v=6.98, sigma_d=4.79, rho_d=5.17
  - RW-MH ESS per second (3.0s): sigma=1.26, phi_pi=17.54, phi_y=3.38, kappa=9.96, sigma_v=14.07, rho_v=1.42, sigma_d=13.81, rho_d=6.06
  - Per wall-clock second, NUTS **loses** on 5 of 8 parameters (phi_pi: 0.33x, kappa: 0.41x, sigma_v: 0.53x, sigma_d: 0.35x, rho_d: 0.85x).
  - Raw ESS ratio (equal sample count): sigma=578x, phi_pi=57x, phi_y=361x, kappa=70x, sigma_v=91x, rho_v=850x, sigma_d=60x, rho_d=148x — these ratios are real but reflect equal sample count, not equal wall time.
- **Category:** DILUTED — the raw ESS ratios (57x-850x) are real and in the table, but "per compute unit" is undefined and the comparison is sample-count-equalized, not wall-time-equalized. The Takeaway presents a favorable framing that does not survive "compute unit = wall-clock second." The NUTS startup cost (warmup + JIT compilation) makes 520.2s vs 3.0s a 173x wall-time disadvantage; ESS must overcome this to win on throughput.
- **Severity:** HIGH — the Takeaway is the final pedagogical lesson readers carry away. It presents NUTS as unambiguously superior "per compute unit" when the table itself, with reported runtimes, shows RW-MH is faster per second on most parameters.
- **Result-changing:** yes — a reader computing ESS-per-second from the reported data would conclude RW-MH wins on 5 of 8 parameters, the opposite of the Takeaway's claim.
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert all(ne/520.2 > re/3.0 for ne, re in zip(nuts_ess_list, rwmh_ess_list))
  # FAILS on current data (5 of 8 violate it); PASSES only if NUTS truly wins per-second on all params
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "per sample" in takeaway_text or "equal sample count" in takeaway_text or "wall time" not in [w.lower() for w in takeaway_text.split()]
  # PASSES on honest fix that qualifies the comparison; FAILS on current unqualified "per compute unit"
  ```

### Finding 4: "ESS is on a log scale" sentence placement

- **Claim source (verbatim):** "ESS is on a log scale." — `README.md:136`
- **Code evidence (verbatim):**
  ```python
  ax.set_yscale("log")
  ```
  `run.py:458` — the ESS comparison *figure* uses a log y-axis. The *table* above `README.md:136` contains raw ESS values (2192, 3000, etc.), not log-transformed values.
- **Data evidence:** `tables/posterior_summary.csv` column `NUTS ESS` = 2192.42 for sigma. `log(2192.42) = 7.69`. The table value is 2192.42, confirming it is raw, not log-scaled.
- **Category:** DILUTED — sentence is placed directly after the summary table and before the ESS figure, making it ambiguous whether it describes the table or the figure. The figure correctly uses a log axis; the table is raw. A reader parsing the sentence as describing the table would misread every ESS value by a factor of ~3-12 (e.g., interpreting 2192 as e^2192).
- **Severity:** LOW — the actual figure caption says "ESS per parameter: NUTS vs. random-walk Metropolis" and the figure visually uses log axis; a careful reader recovers the correct interpretation. No number in any artifact is wrong.
- **Result-changing:** no
- **Violated invariant (one-line pytest assertion):**
  ```python
  assert readme_line_136 == "ESS is on a log scale." and table_line < readme_line_136 < figure_line
  # PASSES on current layout (ambiguous placement); FAILS after honest fix moves sentence
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "figure" in readme_line_136.lower() or "y-axis" in readme_line_136.lower()
  # PASSES if sentence says "the figure y-axis is on a log scale"; FAILS on current bare sentence
  ```

### Finding 5: Runtime numbers (520.2s, 3.0s, 0.91, 0.02 acceptance)

- **Claim source (verbatim):** "NUTS ran 4 chains of 1000 warm-up plus 2000 kept draws in 520.2 seconds wall time. Random-walk Metropolis ran a single chain of 8000 draws in 3.0 seconds. NUTS reached an average acceptance of 0.91; RW-MH at the chosen step size reached 0.02." — `README.md:113`
- **Code evidence:** `run.py:628-634` — these values are interpolated at report-generation time from live variables `t_nuts`, `t_rwmh`, `nuts_accept.mean()`, `rwmh_accept`. They are not hardcoded.
- **Data evidence:** No `tables/` file stores runtime or acceptance. Values cannot be grounded against a committed artifact.
- **Category:** DATA DRIFT — the numbers in `README.md` are from a past run and cannot be verified against any committed data artifact without re-running. `needs re-run to verify`.
- **Severity:** LOW — the chain configuration (4x1000+2000 NUTS, 1x8000 RW-MH) is correctly stated and matches code constants `NUM_WARMUP=1000`, `NUM_SAMPLES=2000`, `NUM_CHAINS=4`.
- **Result-changing:** no (runtimes are informational, not used in any downstream computation in the report)
- **Violated invariant:** N/A — cannot assert without re-run.
- **Honest-fix pass condition:** N/A — `needs re-run to verify`.

## Cross-cutting patterns

- The report conflates "same sample count" with "same computational budget." The ESS comparison and the Takeaway both treat 8000 NUTS draws and 8000 RW-MH draws as equivalent effort, but NUTS requires JIT compilation and warm-up, making wall time 173x larger. The word "compute unit" in the Takeaway is the same ambiguity as "per compute unit" in the ESS figure. Both should be tightened to "per raw sample."
- The word "parallel" is used aspirationally in two places (Solution Method and the `run_nuts` docstring at `run.py:259`). Neither site implements data-parallel execution. If a future maintainer adds `jax.vmap`, the claim becomes true without any prose change, but the current codebase does not support it.
- All numeric claims that can be grounded against committed artifacts (matrix entries, prior parameters, posterior table, ESS ratios) are exact. The only unverifiable numbers are run-time-dependent (wall seconds, acceptance rates).

## TDD execution sequence (for the next agent)

0. **Bullshit score is 35%.** Under the 50% threshold; surface findings to user but do not halt forward work. Fix Finding 3 (Takeaway prose) first as it is the highest-severity finding and result-changing.

1. **Finding 3 (DILUTED HIGH) — Takeaway prose fix:**
   - Write a pytest that reads `README.md` and asserts the Takeaway does not contain the unqualified phrase "per compute unit" alongside ESS numbers without a qualifier like "per sample" or "equal sample count."
   - Confirm test fails on current `README.md`.
   - The fix is a prose change in `run.py:688` to qualify the comparison: "per raw sample drawn" or "for equal numbers of log-density evaluations."
   - Re-run `python run.py` (or patch `report.add_takeaway()`) and confirm the test passes.

2. **Finding 2 (MISLABELED MED) — "parallel" chain claim:**
   - Write a pytest: `assert "jax.vmap" not in inspect.getsource(run_nuts)` — confirms the mislabel is real.
   - Fix option A: change "run in parallel" to "run independently" in `run.py:584`. Cosmetic only.
   - Fix option B: implement actual parallel dispatch with `jax.vmap` over chains. Functional change; re-time.
   - If choosing fix A, the mislabel disappears and the MISLABELED finding becomes HOLDS.

3. **Finding 4 (DILUTED LOW) — "ESS is on a log scale" sentence:**
   - Cosmetic prose fix in `run.py` report body: change to "The figure y-axis uses a log scale."
   - No test strictly required; a regex check on the generated README suffices.

4. **Findings 1, 5 (HOLDS / DATA DRIFT):** No action. Runtime numbers refresh on every `python run.py`.

5. After prose fixes, re-run `python run.py` to regenerate `README.md`. Re-run this skill on the new README to confirm findings 2, 3, 4 now read HOLDS and the new bullshit score is <= 25%.
