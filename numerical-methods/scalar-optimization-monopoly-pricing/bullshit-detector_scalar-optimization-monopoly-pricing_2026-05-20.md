# bullshit-detector — scalar-optimization-monopoly-pricing — 2026-05-20

**Bullshit score: 25%** — one DATA DRIFT finding (Equations says random-search error O(1/N), four other locations in the same README say O(1/√N)); all code-vs-claim checks HOLD; no FALSE or UNIMPLEMENTED findings; cap at 25% does not apply (artifact has Monte Carlo and numeric results).

## Header
- Claim sources: `numerical-methods/scalar-optimization-monopoly-pricing/README.md`
- Code / artifact root: `numerical-methods/scalar-optimization-monopoly-pricing/run.py`
- Data artifacts: `tables/method_comparison.csv`, `tables/elasticity_sensitivity.csv`, `tables/newton_sensitivity.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | Random-search error scales as 1/N (Equations) vs 1/√N (rest of README) | DATA DRIFT | MED | no (does not change computed numbers, changes the stated pedagogical rate claim) |
| 2 | π'(p) formula | HOLDS | — | — |
| 3 | π''(p) formula | HOLDS | — | — |
| 4 | p* = ε/(ε−1)·c closed form | HOLDS | — | — |
| 5 | p_inflect = (ε+1)/(ε−1)·c | HOLDS | — | — |
| 6 | p* < p_inflect (concave region claim) | HOLDS | — | — |
| 7 | φ = (√5−1)/2 ≈ 0.618 | HOLDS | — | — |
| 8 | Bracket shrinks by factor φ per step | HOLDS | — | — |
| 9 | Safeguard pseudocode matches code | HOLDS | — | — |
| 10 | Bad start x_1 = 5.4, diverged after 1 iteration | HOLDS | — | — |
| 11 | Safeguarded Newton: 9 iterations | HOLDS | — | — |
| 12 | 3 of 9 starts diverge | HOLDS | — | — |
| 13 | Newton good start: 6 iterations | HOLDS | — | — |
| 14 | Method comparison table numbers match CSV | HOLDS | — | — |
| 15 | Elasticity sensitivity table numbers match CSV | HOLDS | — | — |

## Findings

### Finding 1: Random-search convergence rate stated as 1/N in Equations, 1/√N everywhere else

- **Claim source (verbatim):**
  > "The expected error scales as $1/N$ in one dimension, the same order as the grid but with stochastic noise on each draw."
  >
  > — `README.md:75`

  Contradicted by the same document at four locations:

  > "The expected distance from the closest draw to $p^{\ast}$ scales as $1/\sqrt{N}$. That is slower than grid search in one dimension."
  >
  > — `README.md:140`

  > "Averaging across seeds at each $N$ recovers the smooth $1/\sqrt{N}$ rate."
  >
  > — `README.md:152`

  > "Random-search error scales as $1/\sqrt{N}$ on average across seeds."
  >
  > — `README.md:213`

  > "Random search trades the deterministic mesh for stochastic error that scales as $1/\sqrt{N}$."
  >
  > — `README.md:265`

- **Code evidence (verbatim):**
  ```python
  ref_sqrt = (p_high - p_low) / np.sqrt(n_arr)
  ax4b.loglog(n_arr, ref_sqrt, ":", color="tab:purple", linewidth=1.0,
              label=r"Reference $\propto 1/\sqrt{N}$")
  ```
  `run.py:659-661`

  The code's add_results call:
  ```python
  "Random-search error scales as $1/\\sqrt{N}$ on average across seeds. "
  "Grid is faster than random in one dimension. "
  ```
  `run.py:673-674`

- **Data evidence:** Not applicable — the rate claim is pedagogical, not a number in the CSV tables. The CSV method_comparison.csv row for random search shows absolute error `2.67e-03` at N=1001; this alone does not distinguish 1/N from 1/√N (both predict small errors at N=1001).

- **Category:** DATA DRIFT — two claim sources within the same README disagree on the same rate. The code and three of the four README locations agree on `1/√N`; the Equations section alone says `1/N`.

- **Severity:** MED — no computed number is wrong. The stated pedagogical rate claim in the Equations section contradicts the rest of the document and the code. A reader following Equations would conclude random search and grid search have the same 1D rate, contradicting the lesson of the tutorial (that grid dominates random in 1D, with random's advantage being dimensional).

- **Result-changing:** no — the computed table values and figures are unaffected. The lesson's framing is affected: the Equations section wrongly says random search matches grid at O(1/N) in 1D, while the rest of the tutorial correctly identifies 1/√N and uses it to motivate the dimensional argument.

  Note on mathematical ground truth: for N uniform draws on [a,b], the expected error of the argmax (gap from nearest draw to the true maximum in 1D) is O(1/N) — specifically (b-a)/(N+1). The 1/√N rate cited in Bergstra-Bengio (2012) applies to the probability of covering a fraction of the domain within tolerance δ in high dimensions, not to the 1D argmax error. Both rates have defensible interpretations; the document picks different interpretations in different sections without noting the distinction. The Equations section (1/N) is mathematically closer to what the code actually computes (argmax of N draws), while the Solution Method and Takeaway (1/√N) import the dimension-free framing from the literature. The intra-document inconsistency is the finding regardless of which rate is "more correct."

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "scales as $1/N$ in one dimension, the same order as the grid" in open("README.md").read()
  # PASSES on current (buggy) README; FAILS once Equations is corrected to 1/√N
  ```

- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert README.count("1/\\sqrt{N}") >= 5 and "1/N in one dimension, the same order as the grid" not in README
  # PASSES once all five locations agree on 1/√N (or a note distinguishes the two rates)
  ```

## Cross-cutting patterns

- All code-vs-claim checks HOLD. The `run.py` faithfully implements every formula stated in Equations (profit, FOC, second derivative, closed-form p*, p_inflect, φ, Newton update rule, safeguard pseudocode). The tables and figures are generated directly from the computed variables with no hardcoding.
- The single DATA DRIFT finding is entirely within the README prose: the Equations section was written with a different rate claim than the Solution Method, Results, and Takeaway sections. The code is not the source of the inconsistency.
- All six headline rows in `method_comparison.csv` match the README table verbatim (checked row by row). All six elasticity rows and all nine Newton sensitivity rows match. No stale numbers were detected.
- The safeguard bisection direction (`p_hi + x)/2` when π'>0, `(p_lo + x)/2` when π'<0) matches the pseudocode exactly at `run.py:148`.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 25% (MED threshold).** One finding. Safe to proceed after fixing the inconsistency; no need to halt forward work.

1. Turn Finding 1's violated invariant into a pytest test:
   ```python
   def test_equations_rate_claim():
       text = open("README.md").read()
       # This test PASSES now (proves the drift is real):
       assert "scales as $1/N$ in one dimension, the same order as the grid" in text
   ```

2. The honest-fix pass condition as a second test:
   ```python
   def test_equations_rate_fixed():
       text = open("README.md").read()
       # This test FAILS now (proves fix is needed):
       assert "scales as $1/N$ in one dimension, the same order as the grid" not in text
   ```

3. Fix: in `run.py`'s `add_equations` call, change the random-search rate description at the line corresponding to `README.md:75` from `1/N` to `1/√N` (or add a note distinguishing the 1D argmax rate O(1/N) from the dimension-free O(1/√N) rate, whichever the author intends). Regenerate README via `python run.py`.

4. After fix, `test_equations_rate_claim` should FAIL and `test_equations_rate_fixed` should PASS.

5. Re-run this skill to confirm all findings read HOLDS and score drops to ≤10%.
