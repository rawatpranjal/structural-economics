# bullshit-detector -- money-pump-index -- 2026-05-20

**Bullshit score: 0%** -- Every claim in the README is grounded verbatim in run.py. All numeric values cross-check against tables/mpi-summary.csv. Karp DP is mathematically correct for maximum mean cycle. The prior 20% score from the existing file was itself a false finding: "max_v min_k" in the pseudocode is a literal description of `max(best, min(ratios))` in code -- the quantifier order is identical, not misleading. No FALSE, DILUTED, MISLABELED, DATA DRIFT, or UNIMPLEMENTED findings.

## Header
- Claim sources: `choice/money-pump-index/README.md` (prose, Equations, Model Setup, Solution Method, Results)
- Code / artifact root: `choice/money-pump-index/run.py`
- Data artifacts: `choice/money-pump-index/tables/mpi-summary.csv`
- Seed audit (if any): prior `bullshit-detector_money-pump-index_2026-05-20.md` (20% score, now superseded)
- Run by: Claude Sonnet 4.6 (bullshit-detector skill, six-pass method, caveman ultra mode)
- Date: 2026-05-20
- Diagram-only cap applied: no

## Summary table

| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | MPI = max-mean cycle via Karp DP | HOLDS | -- | no |
| 2 | w_ij = (E_ii - E_ij) / E_ii | HOLDS | -- | no |
| 3 | Edges added when w_ij > 0 | HOLDS | -- | no |
| 4 | Own expenditure = 1.00 | HOLDS | -- | no |
| 5 | Severe slack: 18%, 24%, 8% | HOLDS | -- | no |
| 6 | Severe MPI = 0.167 | HOLDS | -- | no |
| 7 | Small cycle MPI = 0.030 | HOLDS | -- | no |
| 8 | Medium cycle MPI = 0.103 | HOLDS | -- | no |
| 9 | No-cycle: MPI = 0.000, GARP passes | HOLDS | -- | no |
| 10 | Pseudocode step 6 = max_v min_k [D_T(v)-D_k(v)]/(T-k) | HOLDS | -- | no |
| 11 | Best cycle displayed as 1->2->3->1 | HOLDS | -- | no |
| 12 | MPI range 0.030 to 0.167 across inconsistent datasets | HOLDS | -- | no |

## Findings

All claims HOLD. Detailed grounding below.

### Finding 1: MPI = max-mean cycle, Karp DP

- **Claim source (verbatim):** "The Money Pump Index is the largest average slack over all directed cycles in the revealed-preference graph: MPI = max_C w-bar(C)." -- `README.md:31-34`
- **Code evidence (verbatim):**
  ```python
  def karp_maximum_mean_cycle(edges: np.ndarray, weights: np.ndarray) -> float:
      """Karp dynamic program for the maximum mean-weight directed cycle."""
      if not simple_cycles(edges):
          return 0.0
      ...
      best = 0.0
      for v in range(n):
          if dp[n, v] <= neg_inf / 2:
              continue
          ratios = [
              (dp[n, v] - dp[k, v]) / (n - k)
              for k in range(n)
              if dp[k, v] > neg_inf / 2
          ]
          if ratios:
              best = max(best, min(ratios))
      return max(0.0, best)
  ```
  `run.py:103-136`
- **Data evidence:** CSV `tables/mpi-summary.csv:5`: `Severe cycle,yes,1 -> 2 -> 3 -> 1,"18%, 24%, 8%",0.167`. Arithmetic: (0.18+0.24+0.08)/3 = 0.16666... -> 0.167 at 3 d.p.
- **Category:** HOLDS
- **Note on prior finding:** The prior audit rated this DILUTED (20%) claiming "max_v min_k order potentially misleading." This is incorrect. The pseudocode `max_v min_{0<=k<T} [D_T(v)-D_k(v)]/(T-k)` is a literal description of `max(best, min(ratios))` -- outer max over v, inner min over k. No ambiguity exists. Stress-tested with a two-cycle graph (2-cycle mean 0.075, 3-cycle mean 0.200); code returned 0.200 correctly.

### Finding 2: Edge weight formula w_ij = (E_ii - E_ij) / E_ii

- **Claim source (verbatim):** "define the relative budget slack on a direct revealed-preference edge as w_ij = (E_ii - E_ij) / E_ii" -- `README.md:23-24`
- **Code evidence (verbatim):**
  ```python
  own = np.diag(costs)
  savings = (own[:, None] - costs) / own[:, None]
  ```
  `run.py:35-36`
- **Category:** HOLDS

### Finding 3: Edges added when w_ij > 0

- **Claim source (verbatim):** "The graph keeps edges with w_ij > 0." -- `README.md:26`
- **Code evidence (verbatim):**
  ```python
  TOL = 1e-10
  edges = (savings > TOL) & ~np.eye(len(prices), dtype=bool)
  ```
  `run.py:22, 37`
- **Category:** HOLDS (TOL = 1e-10 is numerically zero)

### Finding 4: Own expenditure = 1.00

- **Claim source (verbatim):** "Own expenditure | 1.00 | Each chosen bundle costs one at its own prices" -- `README.md:41`
- **Code evidence (verbatim):**
  ```python
  quantities = np.eye(3)
  prices = np.array([
      [1.00, 1.00 - s12, 1.20],
      [1.20, 1.00,       1.00 - s23],
      [1.00 - s31, 1.20, 1.00],
  ])
  ```
  `run.py:141-148`. E_ii = prices[i,i] = 1.00 for all i, both `make_cycle_case` and `make_rational_case`.
- **Category:** HOLDS

### Finding 5: Severe slack 18%, 24%, 8% -> MPI 0.167

- **Claim source (verbatim):** "Severe-cycle slack | 18%, 24%, 8% | Slack on A over B, B over C, and C over A" -- `README.md:43`; "Severe MPI | 0.167" -- `README.md:44`
- **Code evidence (verbatim):** `run.py:294`: `make_cycle_case("Severe cycle", (0.18, 0.24, 0.08))`; `run.py:367`: `f"| Severe MPI | {float(severe['mpi']):.3f} | ..."`
- **Data evidence:** CSV `mpi-summary.csv:5`: `"18%, 24%, 8%",0.167`. Arithmetic: (0.18+0.24+0.08)/3 = 0.16666... -> 0.167.
- **Category:** HOLDS

### Finding 6: Small MPI = 0.030, Medium MPI = 0.103

- **Claim source (verbatim):** "Small cycle | ... | 3%, 4%, 2% | 0.03" and "Medium cycle | ... | 10%, 13%, 8% | 0.103" -- `README.md:71-72`
- **Code evidence:** `run.py:292-293`: `make_cycle_case("Small cycle", (0.03, 0.04, 0.02))` and `make_cycle_case("Medium cycle", (0.10, 0.13, 0.08))`
- **Data evidence:** CSV `mpi-summary.csv:3`: `0.030`; CSV `mpi-summary.csv:4`: `0.103`. Arithmetic: (0.03+0.04+0.02)/3 = 0.0300; (0.10+0.13+0.08)/3 = 0.10333... -> 0.103 at 3 d.p.
- **Category:** HOLDS

### Finding 7: No-cycle passes GARP, MPI = 0.000

- **Claim source (verbatim):** "No cycle | no | none | none | 0" -- `README.md:69`
- **Code evidence (verbatim):**
  ```python
  prices = np.array([
      [1.00, 1.08, 1.20],
      [1.18, 1.00, 1.10],
      [1.12, 1.15, 1.00],
  ])
  ```
  `run.py:171-176`. All off-diagonal prices exceed own price -> savings matrix has no positive off-diagonal entry -> edges all False -> Karp returns 0.0; garp_violations returns [].
- **Data evidence:** CSV `mpi-summary.csv:2`: `No cycle,no,none,none,0.000`
- **Category:** HOLDS

### Finding 8: Pseudocode step 6 = code exactly

- **Claim source (verbatim):** "6. Return max_v min_{0 <= k < T} [D_T(v) - D_k(v)] / (T - k)." -- `README.md:57`
- **Code evidence (verbatim):**
  ```python
  ratios = [
      (dp[n, v] - dp[k, v]) / (n - k)
      for k in range(n)
      if dp[k, v] > neg_inf / 2
  ]
  if ratios:
      best = max(best, min(ratios))
  ```
  `run.py:129-135`. n = T = 3. range(n) = 0..2 = 0 <= k < T. dp[n,v] = D_T(v). min(ratios) then max(best,...) = max_v min_k. Exact match.
- **Category:** HOLDS

### Finding 9: Best cycle displayed as 1->2->3->1

- **Claim source (verbatim):** "Best cycle | 1 -> 2 -> 3 -> 1" -- `README.md:71-73`
- **Code evidence (verbatim):** `run.py:306`: `" -> ".join(str(i + 1) for i in cycle + [cycle[0]])`. cycle=[0,1,2] -> "1 -> 2 -> 3 -> 1".
- **Category:** HOLDS

### Finding 10: MPI range 0.030 to 0.167

- **Claim source (verbatim):** "Their MPI values range from 0.030 to 0.167." -- `README.md:63`
- **Data evidence:** CSV rows for GARP-rejecting datasets: 0.030, 0.103, 0.167. Range 0.030 to 0.167.
- **Category:** HOLDS

## Cross-cutting patterns

None. All claims hold. The tutorial is a closed, small-scale worked example (T=3, G=3) where every number derives analytically from the designed slack parameters. The prior audit's 20% score was based on a false DILUTED finding about pseudocode quantifier order. That finding was wrong: `max_v min_k` in pseudocode is the literal algebraic description of `max(best, min(ratios))` in code. Re-audit confirms 0%.

## TDD execution sequence (for the next agent)

0. **Bullshit score: 0%.** No non-HOLDS findings. Forward work unblocked.
1. No violated-invariant tests to write.
2. No honest-fix pass conditions to write.
3. No fixes needed.
4. If code changes in future, re-check: MPI(small)=0.030, MPI(medium)=0.103, MPI(severe)=0.167, MPI(no-cycle)=0.000, and that the two-cycle stress test returns the higher-mean cycle (0.200 not 0.075).
