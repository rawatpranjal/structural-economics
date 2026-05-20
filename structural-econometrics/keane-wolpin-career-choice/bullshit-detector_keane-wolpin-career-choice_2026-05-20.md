# bullshit-detector - keane-wolpin-career-choice - 2026-05-20

**Bullshit score: 35%** - One FALSE/HIGH finding (inverted direction on state-count prose) and one DILUTED/MED finding (undisclosed feature normalization in regression basis) combine to place this squarely in the 30-50% band; no FALSE claim changes a published number, but a reader trusting the table description or the basis formula gets a wrong mental model.

## Header

- Claim sources: `structural-econometrics/keane-wolpin-career-choice/README.md`
- Code / artifact root: `structural-econometrics/keane-wolpin-career-choice/run.py`
- Data artifacts: `tables/emax-diagnostics.csv`, `tables/emax-fit.csv`, `tables/lifecycle-moments.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | "Later ages have fewer continuation states, so approximation becomes nearly exact" | FALSE | HIGH | no (table shows truth; prose description is wrong) |
| 2 | "Approximation errors are largest at young ages" | DILUTED | MED | no (true for normalized RMSE; false for absolute RMSE shown in figure) |
| 3 | phi(s,t) = (1, E, Xb, Xw, ...) raw state variables | DILUTED | MED | no (regression still fits; documented formula uninterpretable against raw state) |
| 4 | np.empty fallback shape (0, 11) vs actual 12 columns | DATA DRIFT | LOW | no (empty path never triggered in main()) |
| 5 | All wage/utility formulas | HOLDS | - | - |
| 6 | Emax formula, Euler constant | HOLDS | - | - |
| 7 | N_s = 2,310 pre-terminal states | HOLDS | - | - |
| 8 | max normalized RMSE = 12.3%, min policy agreement = 90.0% | HOLDS | - | - |
| 9 | All calibration parameters (beta, sigma_e, horizon, etc.) | HOLDS | - | - |
| 10 | Terminal value formula | HOLDS | - | - |

## Findings

### Finding 1: "Later ages have fewer continuation states, so approximation becomes nearly exact"

- **Claim source (verbatim):** "The Emax regression uses at most the sample cap at each age. Later ages have fewer continuation states, so the approximation becomes nearly exact." - `README.md:312` (generated from `run.py:836-837`)

- **Code evidence (verbatim):**
  ```python
  description=(
      "The Emax regression uses at most the sample cap at each age. Later ages "
      "have fewer continuation states, so the approximation becomes nearly exact."
  ),
  ```
  `run.py:835-838`

- **Data evidence:** `tables/emax-fit.csv` shows the exact opposite: age 16 has 1 state (sampled=1, in_sample_rmse=0.0); age 29 has 525 states (sampled=260, in_sample_rmse=0.087). The sample cap of M=260 binds only at ages 26-29 (282, 354, 435, 525 states). The in-sample RMSE at age 29 (0.087) is 145x the in-sample RMSE at age 18 (0.0006). The approximation is LEAST exact at later ages, not most exact. The correct direction: EARLY ages have fewer states, all are sampled exactly, RMSE = 0 or near 0.

- **Category:** FALSE - the direction is inverted relative to what the data artifact shows.

- **Severity:** HIGH - this is the description of the table it directly accompanies. A reader using the table to understand when the approximation is reliable will be told the opposite of the truth.

- **Result-changing:** no - the table itself (`tables/emax-fit.csv`) contains the correct numbers; only the prose interpretation is wrong.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert fit_df.loc[fit_df["Age"] == 29, "Sampled states"].values[0] > fit_df.loc[fit_df["Age"] == 16, "Sampled states"].values[0]  # later ages have MORE states, not fewer
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "early ages" in description_text and "fewer" in description_text  # description must say early ages have fewer states
  ```

---

### Finding 2: "Approximation errors are largest at young ages"

- **Claim source (verbatim):** "Approximation errors are largest at young ages because early states carry many future option values." - `README.md:287` (generated from `run.py:774-775`)

- **Code evidence (verbatim):**
  ```python
  "The exact solve provides a benchmark. Approximation errors are largest at "
  "young ages because early states carry many future option values. Policy "
  "agreement is the share of states where exact and approximate deterministic "
  "argmax choices match. In this run, the largest age-specific RMSE is "
  f"{max_normalized_rmse:.1%} of the exact Emax standard deviation, and "
  ```
  `run.py:774-779`

- **Data evidence:** `tables/emax-diagnostics.csv` shows absolute RMSE increasing monotonically from 0.0175 (age 16) to 0.0862 (age 29). The figure `figures/emax-accuracy.png` plots absolute RMSE and 90th percentile absolute error, both of which grow with age. The normalized RMSE (RMSE / exact Emax sd) IS largest at age 17 (0.123), so the claim is true for the normalized metric. The sentence cites `max_normalized_rmse` (12.3%) immediately after, which is the normalized RMSE. However, the figure being described shows absolute errors, not normalized errors, and a reader looking at the figure sees errors growing with age, directly contradicting "largest at young ages."

- **Category:** DILUTED - the claim is true for normalized RMSE (which is what the inline number cites) but false for absolute RMSE (which is what the figure shows). The claim is ambiguous about which metric, and the figure contradicts the prose for a reader who looks at both.

- **Severity:** MED - does not change any reported number; misleads figure interpretation.

- **Result-changing:** no - the numbers in the table are correct; only the prose-figure alignment is broken.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert diag_df["rmse"].is_monotonic_increasing  # absolute RMSE grows with age, not with youth
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "normalized" in figure_description_text or "young ages" not in figure_description_text  # prose must specify normalized metric or remove the directional claim
  ```

---

### Finding 3: phi(s,t) documented as raw state variables, code uses normalized features

- **Claim source (verbatim):** "In this tutorial the basis vector is phi(s,t) = (1, E, Xb, Xw, Xb+Xw, E^2, (Xb)^2, (Xw)^2, E*Xb, E*Xw, Xb*Xw, t)." - `README.md:129-131`

- **Code evidence (verbatim):**
  ```python
  arr = np.asarray(states, dtype=float)
  schooling = (arr[:, 0] - p.initial_schooling) / (p.max_schooling - p.initial_schooling)
  blue = arr[:, 1] / max(p.horizon, 1)
  white = arr[:, 2] / max(p.horizon, 1)
  total_exp = blue + white
  age = np.full_like(schooling, age_at(t, p) / (p.start_age + p.horizon))
  return np.column_stack(
      [
          np.ones(len(states)),
          schooling,
          blue,
          white,
          total_exp,
          schooling**2,
          blue**2,
          white**2,
          schooling * blue,
          schooling * white,
          blue * white,
          age,
      ]
  )
  ```
  `run.py:187-208`

- **Data evidence:** n/a (structural claim).

- **Category:** DILUTED - the Equations section documents phi with raw state variables (E, Xb, Xw, t) and the pseudocode in Solution Method uses the same notation, but the actual features are (E - E_0)/(E_max - E_0), Xb/horizon, Xw/horizon, and age_t/(start_age + horizon). The regression is solved on normalized inputs; the coefficient vector b_hat_t corresponds to normalized phi, not the documented raw-variable phi. A reader attempting to evaluate the fitted Emax surface using the documented formula with raw state values would obtain wrong predictions.

- **Severity:** MED - the regression still fits and the approximation diagnostics are valid; the documentation makes b_hat_t uninterpretable in raw-state units.

- **Result-changing:** no - the in-code regression uses consistent normalized inputs throughout; only external reproducibility is broken.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert feature_matrix([(10, 0, 0)], 0, CareerPrimitives())[0, 1] == 10  # fails: code returns 0.0 (normalized), not raw E=10
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert "normalized" in readme_basis_text or feature_matrix([(10, 0, 0)], 0, CareerPrimitives())[0, 1] == 10  # either disclose normalization or use raw inputs
  ```

---

### Finding 4: np.empty fallback shape (0, 11) inconsistent with 12-column output

- **Claim source (verbatim):** "phi(s,t) = (1, E, Xb, Xw, Xb+Xw, E^2, (Xb)^2, (Xw)^2, E*Xb, E*Xw, Xb*Xw, t)" - 12 terms. `README.md:129-131`

- **Code evidence (verbatim):**
  ```python
  def feature_matrix(states: list[State], t: int, p: CareerPrimitives) -> np.ndarray:
      """Polynomial state features for the Emax approximation."""
      if not states:
          return np.empty((0, 11))
  ```
  `run.py:183-186`

- **Data evidence:** `tables/emax-fit.csv` rows for ages 16-29 all show valid regression output, confirming the empty path was never triggered. The column_stack at `run.py:193-207` assembles 12 items (ones, schooling, blue, white, total_exp, schooling**2, blue**2, white**2, schooling*blue, schooling*white, blue*white, age). The guard `return np.empty((0, 11))` is off by one column.

- **Category:** DATA DRIFT - the code self-contradicts on column count: the empty fallback says 11, the populated path produces 12. The README and the populated path agree on 12.

- **Severity:** LOW - the empty path is never triggered in main(); no results are affected.

- **Result-changing:** no.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert feature_matrix([], 0, CareerPrimitives()).shape == (0, 12)  # fails: returns (0, 11)
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert feature_matrix([], 0, CareerPrimitives()).shape[1] == feature_matrix([(10, 0, 0)], 0, CareerPrimitives()).shape[1]  # empty and populated paths must agree on column count
  ```

---

## Cross-cutting patterns

- Findings 1 and 2 both involve prose descriptions of the emax-fit table and the emax-accuracy figure that state directional claims (fewer states at later ages; larger errors at younger ages) that are contradicted by the data artifacts they accompany. Both errors favor the pedagogically appealing story (KW approximation saves computation at the ages where it is most needed) over what the numbers actually show.
- Finding 3 is a documentation-code gap in the regression basis that is invisible to the in-sample diagnostics but would break any external attempt to evaluate the fitted continuation surface.
- Finding 4 is an internal code inconsistency in the empty-path guard, a latent bug triggered only when an age has zero reachable states (which cannot happen in the current calibration but would fail silently if the calibration changed).

## TDD execution sequence (for the next agent)

0. **Bullshit score is 35%.** Below the 50% halt threshold; surface to the user before fixing but do not stop forward work. Findings 1 and 2 require prose fixes only; Finding 3 requires either a prose disclosure or a code change; Finding 4 requires a one-line code fix.

1. **Finding 1 (FALSE/HIGH):** Write `assert fit_df.loc[fit_df["Age"] == 29, "Sampled states"].values[0] > fit_df.loc[fit_df["Age"] == 16, "Sampled states"].values[0]` - this PASSES on current code (the data is correct). The bug is in the description string at `run.py:836`. Fix: change "Later ages have fewer continuation states" to "Early ages have fewer states and are sampled exactly; later ages exceed the sample cap."

2. **Finding 2 (DILUTED/MED):** Write `assert diag_df["rmse"].is_monotonic_increasing` - this PASSES on current code (proving the figure shows rising errors). The bug is in the description string at `run.py:774`. Fix: replace "Approximation errors are largest at young ages" with "The normalized approximation error (RMSE relative to Emax standard deviation) is largest at young ages" and ensure the figure adds a normalized RMSE series or the prose description specifies which metric it is describing.

3. **Finding 3 (DILUTED/MED):** Write `assert feature_matrix([(10, 0, 0)], 0, CareerPrimitives())[0, 1] == 10` - this FAILS on current code (returns 0.0). Fix options: (a) add a note to the Equations section that phi uses normalized inputs; or (b) change the code to use raw inputs and ensure ridge remains numerically stable.

4. **Finding 4 (DATA DRIFT/LOW):** Write `assert feature_matrix([], 0, CareerPrimitives()).shape == (0, 12)` - this FAILS on current code (returns (0, 11)). Fix: change `run.py:186` from `return np.empty((0, 11))` to `return np.empty((0, 12))`.

5. After fixes, re-run `python run.py` inside the tutorial folder to regenerate README.md and tables. Re-run this skill on the new code; expected score: 0-10% (all HOLDS or DATA DRIFT/LOW only).
