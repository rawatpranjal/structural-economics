# Houtman-Maks Rational Subsets

> Diagnosing whether revealed-preference failures are global or outlier-driven.

## Overview

A failed GARP test need not mean that the whole demand model is useless. It may mean that one receipt was miscoded, one shopping trip was unusual, or one observation was drawn from a different decision problem. The Houtman-Maks question is therefore economic before it is computational: what is the largest subset of choices that can still be read as utility-maximizing behavior?

The example starts from Cobb-Douglas choices and corrupts the revealed-preference pattern by swapping two receipts' chosen bundles. The full dataset fails GARP. The Houtman-Maks subset keeps 11 of 12 observations and removes observation 4, restoring rationalizability.

## Equations

For any subset $S$ of observations, let $\operatorname{GARP}(S)=1$ if that subset satisfies GARP.
The Houtman-Maks index is

$$HM = \max_{S \subseteq \{1,\ldots,T\}} |S| \quad \text{s.t.} \quad \operatorname{GARP}(S)=1.$$

Equivalently, the minimum number of observations to discard is

$$T - HM.$$

The exact search solves this objective directly for a small sample. The greedy diagnostic removes high-conflict nodes from violating strongly connected components until GARP holds.

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Observations | 12 | Shopping trips with prices and chosen bundles |
| Goods | 3 | A small multi-good demand environment |
| Corruption | 2 swapped rows | Swapping two chosen bundles creates the failure |
| Exact Houtman-Maks index | 11 | Largest rationalizable subset size |
| Minimum deletions | 1 | Observations needed to restore GARP |

## Solution Method

The script first builds the direct revealed-preference graph from budget comparisons and checks GARP by transitive closure. For the exact Houtman-Maks index, it enumerates subsets from largest to smallest and stops at the first GARP-consistent subset. This is feasible here because $T=12$.

The larger-sample diagnostic uses the same economic object but a cheaper rule: find violating strongly connected components, count each observation's participation in GARP conflicts, remove the highest-conflict observation, and repeat. In this example the greedy heuristic removes the same observation as the exact search: 4.

## Results

The excluded observation participates in most GARP conflicts. Removing it restores a large rationalizable core rather than discarding the whole dataset.

**Observation-Level Houtman-Maks Diagnosis**

|   Observation |   Violation participation | Exact HM action   | Greedy action   |   Chosen q1 |   Rational q1 before corruption |
|--------------:|--------------------------:|:------------------|:----------------|------------:|--------------------------------:|
|             1 |                         0 | keep              | keep            |        2.64 |                            2.64 |
|             2 |                         2 | keep              | keep            |        2.6  |                            2.6  |
|             3 |                         5 | keep              | keep            |        5.94 |                            2.73 |
|             4 |                         6 | remove            | remove          |        2.73 |                            5.94 |
|             5 |                         0 | keep              | keep            |        5.24 |                            5.24 |
|             6 |                         0 | keep              | keep            |        5.03 |                            5.03 |
|             7 |                         0 | keep              | keep            |        6.2  |                            6.2  |
|             8 |                         0 | keep              | keep            |        4.33 |                            4.33 |
|             9 |                         2 | keep              | keep            |        3.1  |                            3.1  |
|            10 |                         0 | keep              | keep            |        5.21 |                            5.21 |
|            11 |                         0 | keep              | keep            |        3.82 |                            3.82 |
|            12 |                         3 | keep              | keep            |        3.42 |                            3.42 |

The red node is the observation excluded by the maximum rational subset. Red arcs mark strict revealed-preference comparisons inside the conflict region.

<img src="figures/conflict-graph.png" alt="Preference conflict graph with the exact Houtman-Maks removal highlighted." width="80%">
*Preference conflict graph with the exact Houtman-Maks removal highlighted.*

The left panel shows the full dataset's contradictions. The right panel shows that the retained subset has no remaining GARP violations.

<img src="figures/violations-before-after.png" alt="GARP violations before and after removing the Houtman-Maks outlier." width="80%">
*GARP violations before and after removing the Houtman-Maks outlier.*

## Takeaway

Houtman-Maks reframes a revealed-preference rejection as an outlier diagnosis. The important empirical question is not only whether GARP fails, but whether the failure is diffuse or concentrated. Here the consumer has a large rationalizable core, and one high-conflict observation explains the rejection. Exact search gives the benchmark for small samples; SCC-aware greedy removal gives a transparent diagnostic when exhaustive subset search is too expensive.

## Reproduce

```bash
python run.py
```

## References

- Houtman, M., & Maks, J. A. H. (1985). Determining all maximal data subsets consistent with revealed preference. Kwantitatieve Methoden, 19, 89-104.
- Heufer, J., & Hjertstrand, P. (2015). Consistent subsets: Computationally feasible methods to compute the Houtman-Maks-index. Economics Letters, 128, 87-89.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
