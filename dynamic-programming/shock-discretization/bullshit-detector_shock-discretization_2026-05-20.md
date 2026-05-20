# bullshit-detector — shock-discretization — 2026-05-20

**Bullshit score: 10%** — all substance holds; one pseudocode description is imprecise ("interior rows") but is mathematically a no-op distinction; diagram-only cap does not apply (results table and numeric comparisons present).

## Header
- Claim sources: `dynamic-programming/shock-discretization/README.md`
- Code / artifact root: `dynamic-programming/shock-discretization/run.py`, `lib/discretize.py`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | sigma_z = 0.0641 for rho=0.95, sigma_eps=0.02 | HOLDS | — | no |
| 2 | Half-life approx 14 periods | HOLDS | — | no |
| 3 | Tauchen pseudocode formula (midpoint CDF differences) | HOLDS | — | no |
| 4 | Rouwenhorst base uses p=(1+rho)/2 | HOLDS | — | no |
| 5 | Rouwenhorst grid formula: z_j = sigma_z * sqrt(N-1) * (2*(j-1)/(N-1) - 1) | HOLDS | — | no |
| 6 | Rouwenhorst invariant dist is binomial: pi_j = Bin(N-1,j-1)/2^{N-1} | HOLDS | — | no |
| 7 | Pseudocode: "row-normalize interior rows" vs lib normalizing all rows | DILUTED | LOW | no |
| 8 | Rouwenhorst matches rho and sigma_z^2 for any N>=2 (zero error) | HOLDS | — | no |
| 9 | Tauchen N=7 persistence = 0.9622, target = 0.95 | HOLDS | — | no |
| 10 | Tauchen N=3 "almost absorbing" / persistence near one | HOLDS | — | no |
| 11 | Finite chains use "same innovation ranks" as AR(1) (common random numbers) | HOLDS | — | no |
| 12 | Invariant pi satisfies pi = pi P (left eigenvector) | HOLDS | — | no |
| 13 | Tauchen puts "extra mass in outer states" creating positive variance error | HOLDS | — | no |
| 14 | All README table numbers match tables/moment-comparison.csv | HOLDS | — | no |

## Findings

### Finding 1: "row-normalize interior rows of P_n" — pseudocode says interior; lib normalizes all rows

- **Claim source (verbatim):** "row-normalize interior rows of P_n     (they receive two contributions)" — `README.md:111`
- **Code evidence (verbatim):**
  ```python
  trans = trans / trans.sum(axis=1, keepdims=True)
  ```
  `lib/discretize.py:41`
- **Data evidence (if applicable):** None — this is a procedural description, not a numeric claim.
- **Category:** DILUTED
- **Severity:** LOW
- **Result-changing:** no

  Endpoint rows of the Rouwenhorst recursion always sum exactly to 1.0 before normalization (verified numerically: endpoint row sums = 1.0 for N=3,5,7,9). Dividing by 1.0 is a no-op. Normalizing all rows and normalizing only interior rows produce transition matrices identical to floating-point precision (max diff 2.22e-16 at N=7). The claim is imprecise — it implies normalization is applied selectively — but the output is correct. The only reader harm is that a reader implementing the pseudocode who normalizes only interior rows gets the same answer; the discrepancy is in the description, not the result.

- **Violated invariant (one-line pytest assertion):**
  ```python
  assert "interior" in readme_pseudocode and lib_trans_all_rows == lib_trans_interior_only  # passes on current code — both normalize identically
  ```
- **Honest-fix pass condition (one-line pytest assertion):**
  ```python
  assert pseudocode_says_all_rows_or_adds_note_about_endpoint_rows_summing_to_one  # pseudocode updated to "normalize all rows (interior rows receive two contributions; endpoint rows already sum to 1)"
  ```

## Cross-cutting patterns

- No systematic mislabeling found. All five economic/algorithmic claims (Tauchen CDF formula, Rouwenhorst recursion, moment-matching property, invariant distribution, common-random-numbers coupling) are grounded in matching code.
- The one DILUTED finding (interior-rows description) is a pseudocode precision issue, not an algorithmic bug. No similar imprecision was found in the Tauchen pseudocode.
- All numeric values in the README (sigma_z, half-life, Tauchen N=7 persistence 0.9622, all 10 table rows) match the committed CSV artifact and are reproducible from the formulas in `run.py` and `lib/discretize.py` without re-running.
- The covariance formula in `markov_moments` (`run.py:42`) correctly implements Cov(z_t, z_{t+1}) / Var(z_t) via `sum(pi[:,None] * P * centered[:,None] * centered[None,:])` — the outer-product broadcasting is correct for one-step autocorrelation under stationarity.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 10% — below 50% threshold. No halt required.**
1. The single non-HOLDS finding (Finding 1, LOW severity) warrants only a pseudocode clarification in `run.py`'s string passed to `report.add_solution_method`. No code change needed in `lib/discretize.py`.
2. Violated-invariant test: confirm the endpoint row sums of the intermediate (pre-normalization) Rouwenhorst matrix equal 1.0 for N >= 3, confirming that normalizing all rows vs interior rows is identical.
3. Honest-fix pass condition: update the pseudocode line in `run.py` from "row-normalize interior rows of P_n" to "normalize all rows of P_n (interior rows received two contributions and sum to 2; endpoint rows already sum to 1)".
4. Re-run `scripts/validate_catalog.py` after the prose change.
5. No re-run of `run.py` needed for numeric verification — all numbers confirmed against committed artifacts.
