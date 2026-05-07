# Price-Regime Revealed Preference

> Use observed bundles to test whether price schedules can be ranked consistently.

## Overview

A researcher may observe the same household, city, or market under several price schedules: a baseline tariff, a subsidy, a tax change, or a different insurance menu. The bundles chosen under those schedules can support a familiar GARP test of utility maximization. The same observations can also support a different welfare question: do they rank the price schedules themselves in a consistent way?

Revealed price preference keeps the observed price-quantity pairs $(p^t,x^t)$ but changes the object being ranked. For each bundle that was actually chosen, the calculation asks which price vectors would have made that same bundle weakly cheaper. Those pairwise cost comparisons form a directed graph over price vectors. The GAPP test closes that graph transitively and looks for a strict reverse comparison. The examples below show why this extra computation matters: a finite dataset can pass bundle GARP while failing to support a consistent ranking of price regimes.

## Equations

The data are $\mathcal D=\{(p^t,x^t)\}_{t=1}^{T}$, where
$p^t\in\mathbb R_{++}^{L}$ is the observed price vector and
$x^t\in\mathbb R_{+}^{L}$ is the chosen bundle. Own expenditure is
$m_t=p^t\cdot x^t$.

For price-regime comparisons, define the cross-cost matrix
$$
C_{st}=p^s\cdot x^t .
$$
Price vector $s$ is directly revealed weakly preferred to price vector $t$ when
schedule $s$ would have made the bundle chosen under schedule $t$ no more
expensive than it actually was:
$$
sR_p^D t
\quad\Longleftrightarrow\quad
C_{st}\le C_{tt}=m_t .
$$

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
| Deterministic cases | 4 | The examples cover every GARP/GAPP pass-fail cell |
| Focal example | Case A | Bundle GARP passes while price GAPP fails |
| Focal GAPP violations | 2 | Strict reverse edges close a price-schedule cycle |

## Solution Method

The computational object is a directed graph. Each node is an observed price vector. An edge from $s$ to $t$ means price schedule $s$ would have made the bundle chosen under $t$ affordable at weakly lower expenditure. As in the [Afriat revealed-preference test](../revealed-preference-afriat/), the direct edges are not enough because indirect comparisons can matter. A Boolean transitive closure gives exact reachability on the finite observation set, with cost $O(T^3)$.

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

The script also runs ordinary bundle GARP on the same observations. Comparing the two diagnostics keeps the economic object clear. A dataset can be consistent with stable preferences over bundles and still reject a stable ranking of the price schedules that generated those bundles.

## Results

The four synthetic panels occupy all four pass-fail cells. Case A is the focal example because bundle choices look rational there, but the price schedules cannot be ranked consistently.

**Bundle GARP and Price GAPP Diagnostics**

| Case   | Economic comparison                 | GARP   | GAPP   |   Bundle violations |   Price violations |
|:-------|:------------------------------------|:-------|:-------|--------------------:|-------------------:|
| A      | Bundle-rational, price-inconsistent | pass   | fail   |                   0 |                  2 |
| B      | Both restrictions pass              | pass   | pass   |                   0 |                  0 |
| C      | Bundle-inconsistent, price-rational | fail   | pass   |                   4 |                  0 |
| D      | Both restrictions fail              | fail   | fail   |                   2 |                  2 |

The heat map shows the cross-cost ratio $C_{st}/C_{tt}$. Rows are candidate price vectors and columns are observed bundles. Entries below one mean that the row price vector would have made the column's chosen bundle cheaper than the observed price vector did.

<img src="figures/price-cost-ratios.png" alt="Cost ratios used to reveal preferences over price vectors." width="80%">

The graph translates those cost comparisons into revealed preferences over price schedules. Each arrow holds a chosen bundle fixed and asks which price vector made that bundle cheaper.

<img src="figures/price-preference-graph.png" alt="A cycle in the price-preference graph rejects GAPP." width="80%">

Across the four deterministic panels, GARP and GAPP separate cleanly. The same price-quantity data can support utility maximization over bundles while rejecting a consistent ordering of price regimes. Another panel can do the reverse.

<img src="figures/garp-vs-gapp-cases.png" alt="GARP and GAPP classify the same datasets differently." width="80%">

## Takeaway

Revealed price preference fits welfare exercises where price schedules, tariffs, or menus are the objects being compared. GARP asks whether one stable utility ordering can rationalize the chosen bundles. GAPP asks whether the observed price vectors can be ranked consistently by the bundles they made affordable. Because the two tests can disagree on the same finite data, a researcher should choose the diagnostic that matches the object of comparison. After a standard [Afriat test](../revealed-preference-afriat/), this page gives the dual check for applications where the price regime itself carries the welfare comparison.

## References

- Deb, R., Kitamura, Y., Quah, J. K. H., & Stoye, J. (2023). Revealed price preference: Theory and empirical analysis. Review of Economic Studies, 90(2), 707-743.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
- Chambers, C. P., & Echenique, F. (2016). Revealed Preference Theory. Cambridge University Press.
