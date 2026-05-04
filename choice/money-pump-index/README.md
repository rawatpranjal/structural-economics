# Money Pump Index for Revealed Preference

> Measuring the welfare cost of revealed-preference cycles.

## Overview

A GARP failure says that the consumer's choices cannot all come from one stable utility function. But the binary rejection does not say whether the contradiction is tiny or economically important. The money-pump question is sharper: if an outside trader followed the revealed preference cycle, how much budget slack could be extracted from the consumer on each trade?

This tutorial uses three chosen bundles, labeled A, B, and C. In the severe case, the consumer chooses A when B was 18 percent cheaper, chooses B when C was 24 percent cheaper, and chooses C when A was 8 percent cheaper. The cycle is a GARP violation, but the index reports its economic size rather than only its existence.

## Equations

For observation $i$, let $E_{ij} = p_i \cdot x_j$ be the cost of bundle $j$ at prices $i$.
Choosing $x_i$ reveals $x_i \succeq x_j$ when $E_{ii} \ge E_{ij}$.

The relative budget slack on a direct revealed-preference edge is

$$w_{ij} = \frac{E_{ii} - E_{ij}}{E_{ii}}.$$

The Money Pump Index is the largest average slack over all directed revealed-preference cycles:

$$\operatorname{MPI} = \max_C \frac{1}{|C|} \sum_{(i,j) \in C} w_{ij}.$$

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Bundles | 3 | A small choice cycle among A, B, and C |
| Own expenditure | 1.00 | Each chosen bundle is normalized to cost one |
| Severe-cycle savings | 18%, 24%, 8% | Slack on A over B, B over C, and C over A |
| Severe MPI | 0.167 | Average extractable slack per step |

## Solution Method

The economics comes first: construct the revealed-preference graph from budget comparisons, then ask which cycles contain budget slack. The computation uses Karp's dynamic program for the maximum mean-weight directed cycle. For the small teaching graph, the script also enumerates the winning cycle so the figure can label the exact trades.

In the severe example, GARP rejects and Karp's algorithm returns $\operatorname{MPI} = 0.167$.

## Results

The same binary GARP rejection can hide very different welfare stakes. The MPI column reports the average budget slack in the most exploitable cycle.

**GARP Rejection and Money Pump Severity**

| Dataset      | GARP rejects   | Best cycle       |   MPI |   Violating pairs |
|:-------------|:---------------|:-----------------|------:|------------------:|
| No cycle     | no             | none             | 0     |                 0 |
| Small cycle  | yes            | 1 -> 2 -> 3 -> 1 | 0.03  |                 3 |
| Medium cycle | yes            | 1 -> 2 -> 3 -> 1 | 0.103 |                 3 |
| Severe cycle | yes            | 1 -> 2 -> 3 -> 1 | 0.167 |                 3 |

Each arrow points from the chosen bundle to another bundle that was strictly cheaper at the same prices. Following the red cycle repeatedly would let a trader extract the displayed slack from the consumer.

<img src="figures/money-pump-cycle.png" alt="The severe revealed-preference cycle and its edge-level budget slack." width="80%">
*The severe revealed-preference cycle and its edge-level budget slack.*

The three inconsistent datasets all fail GARP, but they are not equally costly. MPI separates a near-miss from a large revealed-preference contradiction.

<img src="figures/mpi-severity-comparison.png" alt="GARP is binary, while the Money Pump Index ranks the severity of failures." width="80%">
*GARP is binary, while the Money Pump Index ranks the severity of failures.*

## Takeaway

The Money Pump Index turns a revealed-preference rejection into an economic loss measure. GARP answers whether the data can be rationalized. MPI asks how much money is available in the most exploitable cycle. That distinction matters in empirical work: a tiny violation and a large cyclic arbitrage should not receive the same interpretation.

## Reproduce

```bash
python run.py
```

## References

- Echenique, F., Lee, S., & Shum, M. (2011). The money pump as a measure of revealed preference violations. Journal of Political Economy, 119(6), 1201-1223.
- Karp, R. M. (1978). A characterization of the minimum cycle mean in a digraph. Discrete Mathematics, 23(3), 309-311.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
