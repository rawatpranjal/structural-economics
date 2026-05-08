# Price-Regime Revealed Preference

> Use observed bundles to test whether price schedules can be ranked consistently.

## Overview

A researcher may observe one household, city, or market under several price schedules. The schedules could be tariffs, taxes, subsidies, or insurance menus. Each schedule produces one chosen bundle.

The object is the ranking of price schedules. Revealed price preference asks whether the observed bundles can rank those schedules consistently.

The computation compares every chosen bundle under every observed price vector. Those cross-cost comparisons form a directed graph. GAPP closes the graph and checks whether a strict reverse edge creates a cycle.

## Equations

The data are $\mathcal D=\{(p^t,x^t)\}_{t=1}^{T}$, where
$p^t\in\mathbb{R}_{++}^{L}$ is the observed price vector and
$x^t\in\mathbb{R}_{+}^{L}$ is the chosen bundle. Own expenditure is
$m_t=p^t\cdot x^t$.

For price-regime comparisons, define the cross-cost matrix
$$
C_{st}=p^s\cdot x^t .
$$
Use this matrix to define direct weak preference between price vectors:
$$
sR_p^D t
\quad\Longleftrightarrow\quad
C_{st}\le C_{tt}=m_t .
$$
This means schedule $s$ makes bundle $t$ no more expensive than schedule $t$ did.

The strict relation is
$$
sP_p^D t
\quad\Longleftrightarrow\quad
C_{st}<C_{tt}.
$$
Let $R_p$ be the transitive closure of $R_p^D$. GAPP holds when there is no
pair $(s,t)$ such that
$$
sR_p t
\quad\text{and}\quad
tP_p^D s .
$$
The first relation says the data rank schedule $s$ at least as good as schedule
$t$ after allowing indirect comparisons. The second relation says the direct
reverse comparison strictly favors $t$ over $s$. Together they form the
price-regime analogue of a revealed-preference cycle.

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Observations $T$ | 3 | Each case has three price-quantity observations |
| Goods $L$ | 3 | Bundles are finite consumption vectors |
| Deterministic cases | 4 | Examples separate bundle GARP from price GAPP |
| Main example | Case A | Bundle GARP passes while price GAPP fails |
| Main GAPP violations | 2 | Strict reverse edges close a price-schedule cycle |

## Solution Method

The computational object is a directed graph. Each node is an observed price vector. An edge from $s$ to $t$ means schedule $s$ made bundle $t$ weakly cheaper. Direct edges are not enough because indirect comparisons can matter. A Boolean transitive closure gives exact reachability on the finite data.

```text
Algorithm: GAPP test for price-regime rankings
Input: price vectors p^t and chosen bundles x^t for t=1,...,T
Output: pass/fail GAPP decision and violating price-vector pairs

1. Form C_st = p^s . x^t for every pair of observations (s,t).
2. Draw a weak edge s -> t when C_st <= C_tt.
3. Mark the edge strict when C_st < C_tt.
4. Compute reachability R_p by transitive closure of the weak edges.
5. For each pair (s,t), flag a violation if R_p[s,t] = 1 and the reverse
   direct edge t -> s is strict.
6. Accept GAPP when no violating pair remains.
```

The script also runs ordinary bundle GARP on the same observations. The two tests answer different questions. Stable bundle choices need not imply a stable ranking of the price schedules.

## Results

Case A is the main example. Bundle choices pass GARP there, but the price schedules fail GAPP.

**Bundle GARP and Price GAPP Tests**

| Case   | Economic comparison                 | GARP   | GAPP   |   Bundle violations |   Price violations |
|:-------|:------------------------------------|:-------|:-------|--------------------:|-------------------:|
| A      | Bundle-rational, price-inconsistent | pass   | fail   |                   0 |                  2 |
| B      | Both restrictions pass              | pass   | pass   |                   0 |                  0 |
| C      | Bundle-inconsistent, price-rational | fail   | pass   |                   4 |                  0 |
| D      | Both restrictions fail              | fail   | fail   |                   2 |                  2 |

The heat map shows the cross-cost ratio $C_{st}/C_{tt}$. Rows are candidate price vectors. Columns are observed bundles. Entries below one mark cheaper counterfactual prices for the same bundle.

<img src="figures/price-cost-ratios.png" alt="Cost ratios used to reveal preferences over price vectors." width="80%">

The graph turns cost comparisons into revealed preferences over price schedules. Each arrow keeps the chosen bundle fixed.

<img src="figures/price-preference-graph.png" alt="A cycle in the price-preference graph rejects GAPP." width="80%">

Across the four panels, GARP and GAPP separate cleanly. The same data can pass one test and fail the other.

<img src="figures/garp-vs-gapp-cases.png" alt="GARP and GAPP classify the same datasets differently." width="80%">

## Takeaway

Revealed price preference fits welfare exercises where schedules are the objects being compared. GARP asks whether one utility ordering rationalizes the bundles. GAPP asks whether observed price vectors have a consistent ranking. The tests can disagree on the same finite data.

## References

- Deb, R., Kitamura, Y., Quah, J. K. H., & Stoye, J. (2023). Revealed price preference: Theory and empirical analysis. Review of Economic Studies, 90(2), 707-743.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
- Chambers, C. P., & Echenique, F. (2016). Revealed Preference Theory. Cambridge University Press.
