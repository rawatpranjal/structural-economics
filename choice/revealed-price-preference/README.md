# Revealed Price Preference

> When the object being compared is a price regime, not a chosen bundle.

## Overview

Many revealed-preference exercises ask whether observed bundles could have come from one stable utility ordering. Sometimes the empirical object is different. A researcher may want to compare tax schedules, tariffs, insurance menus, or other price regimes and ask whether the data rank those regimes consistently.

Revealed price preference keeps the same price-quantity observations $(p^t,x^t)$, but it reverses the object of comparison. GARP asks whether the chosen bundles can be ordered. GAPP asks whether the observed price vectors can be ordered by the bundles they make cheap. A price vector $p^s$ is better than $p^t$ for observation $t$ if it would have made the bundle actually chosen under $p^t$ weakly cheaper. The tutorial uses small deterministic panels to show that bundle rationalizability and price-regime rationalizability are distinct restrictions, not two descriptions of the same test.

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
it would have made the bundle chosen at $t$ no more expensive than it actually
was:
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
The first statement says the data indirectly rank price vector $s$ at least as
good as price vector $t$. The second says the data strictly rank $t$ above $s$
in the direct reverse comparison. Together they form the price-regime analogue
of a revealed-preference cycle.

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Observations $T$ | 3 | Each case has three price-quantity observations |
| Goods $L$ | 3 | Bundles are finite consumption vectors |
| Deterministic cases | 4 | The examples cover every GARP/GAPP pass-fail cell |
| Focal example | Case A | Bundle GARP passes while price GAPP fails |
| Focal GAPP violations | 2 | Strict reverse comparisons closing a price cycle |

## Solution Method

The computation is the same kind of finite graph exercise used in the [Afriat revealed-preference test](../revealed-preference-afriat/), but the nodes are price vectors rather than bundles. The only costly step is the transitive closure, which is $O(T^3)$ and exact for these finite panels.

```text
Algorithm: GAPP test for revealed price preference
Input: price vectors p^t and chosen bundles x^t for t=1,...,T
Output: pass/fail GAPP decision and violating price-vector pairs

1. Form C_st = p^s . x^t for every pair of observations (s,t).
2. Set R_p^D[s,t] = 1 if C_st <= C_tt.
3. Set P_p^D[s,t] = 1 if C_st < C_tt.
4. Compute the transitive closure R_p of R_p^D.
5. For every pair (s,t), flag a violation if R_p[s,t] = 1 and P_p^D[t,s] = 1.
6. The data pass GAPP exactly when the violation set is empty.
```

The script also runs ordinary bundle GARP on the same observations. That side-by-side comparison is deliberate: if the empirical question is about price regimes, a clean bundle-rationalizability test can miss the restriction that matters.

## Results

The four synthetic panels occupy all four pass-fail cells. Case A is the focal example because it looks rational when the bundles are tested, but it fails once the price vectors themselves are treated as the objects being ranked.

**Bundle GARP and Price GAPP Diagnostics**

| Case   | Economic comparison                 | GARP   | GAPP   |   Bundle violations |   Price violations |
|:-------|:------------------------------------|:-------|:-------|--------------------:|-------------------:|
| A      | Bundle-rational, price-inconsistent | pass   | fail   |                   0 |                  2 |
| B      | Both restrictions pass              | pass   | pass   |                   0 |                  0 |
| C      | Bundle-inconsistent, price-rational | fail   | pass   |                   4 |                  0 |
| D      | Both restrictions fail              | fail   | fail   |                   2 |                  2 |

The heat map shows the cross-cost ratio $C_{st}/C_{tt}$. Rows are candidate price vectors and columns are observed bundles. Entries below one mean that the row price vector would have made the column's bundle cheaper than the price vector under which that bundle was actually chosen.

<img src="figures/price-cost-ratios.png" alt="Cost ratios used to reveal preferences over price vectors." width="80%">

The graph translates those cost comparisons into revealed preferences over price vectors. The arrows do not compare bundles. They say which price vector is revealed to be at least as attractive after holding a chosen bundle fixed.

<img src="figures/price-preference-graph.png" alt="A cycle in the price-preference graph rejects GAPP." width="80%">

Across the four deterministic panels, GARP and GAPP separate cleanly. The same price-quantity data can support utility maximization over bundles while rejecting a consistent ordering of price regimes, or vice versa.

<img src="figures/garp-vs-gapp-cases.png" alt="GARP and GAPP classify the same datasets differently." width="80%">

## Takeaway

Revealed price preference is useful when the economic question is about price regimes rather than about the chosen bundles themselves. GARP asks whether a stable utility ordering can rationalize choices over bundles. GAPP asks whether observed price vectors can be ranked consistently by the bundles they make affordable. Since the two tests can disagree on the same finite data, the right diagnostic depends on which object the empirical exercise is trying to compare. After a standard [Afriat test](../revealed-preference-afriat/), this tutorial is the natural dual check for applications where prices, tariffs, or schedules are the object of welfare comparison.

## References

- Deb, R., Kitamura, Y., Quah, J. K. H., & Stoye, J. (2023). Revealed price preference: Theory and empirical analysis. Review of Economic Studies, 90(2), 707-743.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
- Chambers, C. P., & Echenique, F. (2016). Revealed Preference Theory. Cambridge University Press.
