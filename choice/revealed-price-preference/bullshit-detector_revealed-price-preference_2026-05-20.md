# bullshit-detector - revealed-price-preference - 2026-05-20

**Bullshit score: 5%** - All six audited claims HOLDS. Every equation grounds verbatim in code; every result-table number re-derives from committed seeds. No FALSE, DILUTED, MISLABELED, DATA DRIFT, or UNIMPLEMENTED findings.

## Header
- Claim sources: `choice/revealed-price-preference/README.md` (prose, Equations lines 13-54, Model Setup lines 58-65, Results lines 88-97)
- Code / artifact root: `choice/revealed-price-preference/run.py`
- Data artifacts: `choice/revealed-price-preference/tables/garp-gapp-examples.csv`
- Seed audit (if any): none
- Run by: Claude Sonnet 4.6 (bullshit-detector skill)
- Date: 2026-05-20
- Diagram-only cap applied: no (numeric results table with 4 rows x 4 value columns present)

## Summary table
| # | Claim (short) | Category | Severity | Result-changing? |
|---|---------------|----------|----------|------------------|
| 1 | C_st = p^s dot x^t | HOLDS | - | no |
| 2 | sR_p^D t iff C_st <= C_tt | HOLDS | - | no |
| 3 | sP_p^D t iff C_st < C_tt | HOLDS | - | no |
| 4 | GAPP violation: sR_p t AND tP_p^D s | HOLDS | - | no |
| 5 | Transitive closure by Boolean Warshall | HOLDS | - | no |
| 6 | Table numbers A(0,2) B(0,0) C(4,0) D(2,2) match CSV and code | HOLDS | - | no |

## Findings

### Finding 1: C_st = p^s dot x^t

- **Claim source (verbatim):** "define the cross-cost matrix $C_{st}=p^s\cdot x^t$" - `README.md:21`
- **Code evidence (verbatim):**
  ```python
  costs = prices @ quantities.T
  ```
  `run.py:49` (GAPP path); `run.py:30` (GARP path)
- **Data evidence:** Not applicable (architectural claim).
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

`costs[s,t] = sum_l prices[s,l] * quantities[t,l] = p^s . x^t`. Exact match.

### Finding 2: sR_p^D t iff C_st <= C_tt

- **Claim source (verbatim):** "$sR_p^D t \quad\Longleftrightarrow\quad C_{st}\le C_{tt}=m_t$" - `README.md:27-29`
- **Code evidence (verbatim):**
  ```python
  own_bundle_cost = np.diag(costs)
  weak = costs <= own_bundle_cost[None, :] + TOL
  ```
  `run.py:49-50`
- **Data evidence:** Not applicable.
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

`weak[s,t] = costs[s,t] <= own_bundle_cost[t] + TOL = C_st <= C_tt + TOL`. The TOL = 1e-10 guard is a numerical stability measure, not a semantic deviation. Exact match.

### Finding 3: sP_p^D t iff C_st < C_tt

- **Claim source (verbatim):** "$sP_p^D t \quad\Longleftrightarrow\quad C_{st}<C_{tt}$" - `README.md:37-39`
- **Code evidence (verbatim):**
  ```python
  strict = costs < own_bundle_cost[None, :] - TOL
  ```
  `run.py:51`
- **Data evidence:** Not applicable.
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

`strict[s,t] = C_st < C_tt - TOL`. Symmetric TOL usage. Exact match.

### Finding 4: GAPP violation condition sR_p t AND tP_p^D s

- **Claim source (verbatim):** "GAPP holds when there is no pair $(s,t)$ such that $sR_p t$ and $tP_p^D s$." - `README.md:42-49`
- **Code evidence (verbatim):**
  ```python
  reach = transitive_closure(weak)
  violations = [
      (s, t)
      for s in range(len(prices))
      for t in range(len(prices))
      if s != t and reach[s, t] and strict[t, s]
  ]
  ```
  `run.py:52-58`
- **Data evidence:** Not applicable.
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

`reach[s,t]` encodes `sR_p t` (transitive closure of weak). `strict[t,s] = C_ts < C_ss - TOL` encodes `tP_p^D s`. The `s!=t` guard prevents spurious self-comparisons (diagonal of weak and reach is always True since `C_tt <= C_tt`). Violation list matches definition exactly.

### Finding 5: Transitive closure by Boolean Warshall

- **Claim source (verbatim):** "A Boolean transitive closure gives exact reachability on the finite data." - `README.md:68`
- **Code evidence (verbatim):**
  ```python
  def transitive_closure(relation: np.ndarray) -> np.ndarray:
      """Compute Boolean transitive closure."""
      reach = relation.copy()
      for k in range(reach.shape[0]):
          reach |= reach[:, [k]] & reach[[k], :]
      return reach
  ```
  `run.py:20-25`
- **Data evidence:** Not applicable.
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

`reach[:,[k]] & reach[[k],:]` broadcasts to shape `(n,n)` computing `reach[i,k] & reach[k,j]` for all `(i,j)`. This is Floyd-Warshall Boolean transitive closure. The weak matrix has `weak[t,t]=True` for all `t` (since `C_tt <= C_tt`), so self-loops are present from initialization; the algorithm is correct. Exact match.

### Finding 6: Results table A(0,2) B(0,0) C(4,0) D(2,2)

- **Claim source (verbatim):** Results table rows `README.md:92-97`.
- **Code evidence (verbatim):**
  ```python
  ("A", "Bundle-rational, price-inconsistent", 0),
  ("B", "Both restrictions pass", 2),
  ("C", "Bundle-inconsistent, price-rational", 5),
  ("D", "Both restrictions fail", 16),
  ```
  `run.py:73-76` (seeds driving synthetic_case)
- **Data evidence (verbatim):**
  ```
  A,"Bundle-rational, price-inconsistent",pass,fail,0,2
  B,Both restrictions pass,pass,pass,0,0
  C,"Bundle-inconsistent, price-rational",fail,pass,4,0
  D,Both restrictions fail,fail,fail,2,2
  ```
  `tables/garp-gapp-examples.csv:2-5`
- **Category:** HOLDS
- **Severity:** -
- **Result-changing:** no

Re-derived from code using committed seeds via independent execution: Case A seed=0 yields (0 bundle, 2 price); Case B seed=2 yields (0,0); Case C seed=5 yields (4,0); Case D seed=16 yields (2,2). All match README.md and CSV exactly. The `len(focal['gapp_violations'])` live-injection at `run.py:307` also guarantees the Model Setup "Main GAPP violations | 2" cell is runtime-computed, not hardcoded.

## Cross-cutting patterns

- Zero claim-vs-code gaps across all six load-bearing assertions. Every equation in `Equations` drives a directly verifiable code path; every number in `Results` is re-derivable from committed seeds without re-running `run.py`.
- The Overview phrase "GAPP closes the graph and checks whether a strict reverse edge creates a cycle" (`README.md:9`) is a permissible pedagogical compression. The full condition requires the forward direction via transitive closure `R_p` rather than only `R_p^D`, but for the cycle framing described, the prose is accurate enough that it does not mislead. The precise definition follows immediately in `Equations`. Not promoted to a finding.
- TOL = 1e-10 (`run.py:17`) is applied symmetrically: `+TOL` on weak, `-TOL` on strict. This prevents boundary points from satisfying both simultaneously. Sound numerical design.
- `plot_price_preference_graph` colors an edge red when `(s,t) OR (t,s)` is a violation pair (`run.py:145`), correctly highlighting both legs of each violating cycle. Consistent with caption.

## TDD execution sequence (for the next agent)

0. **Bullshit score is 5% (<50%). No halt required.** Proceed normally.

No non-HOLDS findings exist. No failing tests need to be written. The tutorial is faithful across all six passes.

Regression-guard assertions (optional, for future refactors):

```python
# Invariant 1: cross-cost matrix definition
assert costs[s, t] == pytest.approx(prices[s] @ quantities[t])

# Invariant 2: weak relation matches definition C_st <= C_tt
assert weak[s, t] == (costs[s, t] <= costs[t, t] + TOL)

# Invariant 3: every flagged violation satisfies both conditions
for (s, t) in violations:
    assert reach[s, t] and strict[t, s]

# Invariant 4: numeric results stable under committed seeds
assert len(price_preference(synthetic_case(0)[0], synthetic_case(0)[1])[1]) == 2  # Case A
assert len(bundle_garp(synthetic_case(5)[0], synthetic_case(5)[1])[1]) == 4        # Case C
```
