# Houtman-Maks Rational Subsets

> Finding the largest rationalizable core after a revealed-preference rejection.

## Overview

A GARP rejection is not automatically a rejection of economic choice theory. In household scanner data, lab choices, or administrative purchase records, the failure could come from a different decision problem, a transcription error, or one receipt that should not be pooled with the rest. The Houtman-Maks index asks how much of the sample can still be read as utility-maximizing behavior under one stable preference ordering.

This tutorial uses a small synthetic demand panel where the uncorrupted choices come from Cobb-Douglas budget shares. Two chosen bundles are then swapped across receipts. The full dataset fails GARP, but the largest rationalizable core keeps 11 of 12 observations. The example is useful because the simulation gives an oracle label for the swapped rows, so the exact Houtman-Maks answer and a greedy large-sample diagnostic can be compared to a known source of contamination.

## Equations

There are $T$ observations. Observation $t$ contains prices $p_t \in \mathbb{R}_{+}^{J}$ and the chosen bundle $x_t \in \mathbb{R}_{+}^{J}$. Expenditure is $m_t=p_t \cdot x_t$.

Choice $t$ directly weakly reveals $x_t$ preferred to $x_s$ when $x_s$ was affordable at prices $p_t$:

$$x_t R^D x_s \quad \Longleftrightarrow \quad p_t \cdot x_t \geq p_t \cdot x_s.$$

The direct relation is strict when the inequality is strict. Let $R$ be the transitive closure of $R^D$. GARP holds on a subset $S$ if there is no pair $t,s \in S$ such that

$$x_t R x_s \quad \text{and} \quad p_s \cdot x_s > p_s \cdot x_t.$$

For any subset $S$, let $\operatorname{GARP}(S)=1$ when these restrictions hold after keeping only observations in $S$. The Houtman-Maks index is

$$HM = \max_{S \subseteq \{1,\ldots,T\}} |S| \quad \text{s.t.} \quad \operatorname{GARP}(S)=1.$$

The minimum number of observations needed to restore GARP is

$$T - HM.$$

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Observations $T$ | 12 | Shopping trips with prices and chosen bundles |
| Goods $J$ | 3 | Small multi-good demand environment |
| Data-generating preferences | Cobb-Douglas shares $(0.45,0.35,0.20)$ | Rational benchmark before corruption |
| Synthetic corruption | rows 3 and 4 swapped | Oracle label available only because this is simulated data |
| Full-sample GARP violations | 9 | Contradictions after taking transitive closure |
| Exact Houtman-Maks index | 11 | Largest rationalizable subset size |
| Greedy deletion | observation 4 | Same deletion selected by the heuristic |

## Solution Method

The exact calculation treats Houtman-Maks as a finite combinatorial problem. For $T=12$, exhaustive subset search is small enough to be the benchmark.

```text
Algorithm: exact Houtman-Maks core
Inputs: observations {(p_t, x_t)}_{t=1}^T
Output: largest subset S* satisfying GARP

for k = T, T-1, ..., 1:
    for each subset S with |S| = k:
        build R^D on S using p_t dot x_t >= p_t dot x_s
        compute the transitive closure R of R^D
        if no strict budget cycle remains:
            return S* = S and HM = k
```

That search is transparent but not scalable. The second calculation keeps the same revealed-preference object and uses the graph structure to choose deletions. A violating strongly connected component is a set of observations tied together by revealed-preference paths, with at least one strict comparison closing the cycle.

```text
Algorithm: SCC greedy Houtman-Maks diagnosis
Inputs: observations {(p_t, x_t)}_{t=1}^T
Output: a GARP-consistent retained set S

initialize S = {1, ..., T}
while GARP(S) fails:
    compute weak arcs, strict arcs, and violating pairs on S
    find strongly connected components of the weak graph
    restrict attention to components containing a strict internal arc
    remove the observation with the most violation participation
return S
```

In this run, the greedy rule removes observation 4, the same receipt removed by exact search. The exact result is the benchmark; the greedy result is a scalable diagnostic for larger panels.

## Results

The table separates three objects that are easy to conflate. The synthetic swap column is the oracle label from the simulation. The exact Houtman-Maks action is the maximum rationalizable subset. The greedy action is the approximation one would use when exact subset enumeration is too expensive.

**Which Receipts Carry the Rejection**

|   Observation |   Violation participation | Synthetic swap row   | Exact HM action   | Greedy action   |   Observed good 1 |   Pre-swap good 1 |
|--------------:|--------------------------:|:---------------------|:------------------|:----------------|------------------:|------------------:|
|             1 |                         0 | no                   | keep              | keep            |              2.64 |              2.64 |
|             2 |                         2 | no                   | keep              | keep            |              2.6  |              2.6  |
|             3 |                         5 | yes                  | keep              | keep            |              5.94 |              2.73 |
|             4 |                         6 | yes                  | remove            | remove          |              2.73 |              5.94 |
|             5 |                         0 | no                   | keep              | keep            |              5.24 |              5.24 |
|             6 |                         0 | no                   | keep              | keep            |              5.03 |              5.03 |
|             7 |                         0 | no                   | keep              | keep            |              6.2  |              6.2  |
|             8 |                         0 | no                   | keep              | keep            |              4.33 |              4.33 |
|             9 |                         2 | no                   | keep              | keep            |              3.1  |              3.1  |
|            10 |                         0 | no                   | keep              | keep            |              5.21 |              5.21 |
|            11 |                         0 | no                   | keep              | keep            |              3.82 |              3.82 |
|            12 |                         3 | no                   | keep              | keep            |              3.42 |              3.42 |

The revealed-preference graph makes the comparison concrete. Red fill marks the exact Houtman-Maks deletion, the black x marks the greedy deletion, and gold rings mark the two receipts whose bundles were swapped in the simulation. The method does not need to remove both swapped rows: dropping one side of the conflict is enough to recover a GARP-consistent core.

<img src="figures/conflict-graph.png" alt="Preference conflict graph comparing exact, greedy, and synthetic corruption markers." width="80%">

The heat maps are the certificate. The full sample has strict budget-cycle contradictions. After removing the exact Houtman-Maks deletion, the retained core has none.

<img src="figures/violations-before-after.png" alt="GARP violations before and after removing the Houtman-Maks outlier." width="80%">

## Takeaway

Houtman-Maks turns a binary revealed-preference rejection into a question about the size of the rationalizable core. Here the rejection is concentrated: one deletion restores GARP for 11 of 12 observations. The synthetic oracle also shows the main interpretive limit. The index is not a causal label for every corrupted row; it is a finite-data robustness measure for how much of the sample can still support utility maximization.

## References

- Houtman, M., & Maks, J. A. H. (1985). Determining all maximal data subsets consistent with revealed preference. Kwantitatieve Methoden, 19, 89-104.
- Heufer, J., & Hjertstrand, P. (2015). Consistent subsets: Computationally feasible methods to compute the Houtman-Maks-index. Economics Letters, 128, 87-89.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
