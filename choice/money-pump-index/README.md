# Revealed-Preference Cycles and the Money Pump Index

> Measuring the expenditure exposed by inconsistent choices.

## Overview

Suppose a consumer is observed choosing bundle A at one price vector, bundle B at another, and bundle C at a third. A GARP test tells us whether these choices could come from one stable utility ordering. It does not tell us whether the failure is a near-tie or whether an observer could make money by cycling the consumer through trades.

This tutorial puts a price on that failure. Own expenditure is normalized to one, so each comparison can be read as a share of the consumer's budget. In the severe case the consumer chooses A when B was 18 percent cheaper, chooses B when C was 24 percent cheaper, and chooses C when A was 8 percent cheaper. Those comparisons form a directed revealed-preference cycle. The computation assigns each edge its budget slack and finds the cycle with the largest average slack, which is 16.7% of expenditure per trade.

## Equations

There are $T$ observations. Observation $i$ records a price vector
$p_i \in \mathbb{R}^G_+$ and chosen bundle $x_i \in \mathbb{R}^G_+$.
Let

$$E_{ij}=p_i \cdot x_j$$

be the cost of bundle $j$ at observation $i$ prices. Choosing $x_i$ when
$x_j$ was affordable, $E_{ii} \ge E_{ij}$, directly reveals
$x_i \succeq^D x_j$. For strict comparisons, define the relative budget slack
on a direct revealed-preference edge as

$$w_{ij} = \frac{E_{ii} - E_{ij}}{E_{ii}}.$$

The graph keeps edges with $w_{ij}>0$. For a directed cycle
$C=(i_1,\ldots,i_m,i_1)$, average slack is

$$\bar w(C)=\frac{1}{m}\sum_{\ell=1}^{m} w_{i_\ell,i_{\ell+1}}.$$

The Money Pump Index is the largest average slack over all directed cycles in
the revealed-preference graph:

$$\operatorname{MPI} = \max_C \bar w(C).$$

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Observations | 3 | One price vector and one chosen bundle in each row |
| Bundles | 3 | A, B, and C are the only candidate bundles |
| Own expenditure | 1.00 | Each chosen bundle costs one at its own prices |
| Severe-cycle slack | 18%, 24%, 8% | Slack on A over B, B over C, and C over A |
| Severe MPI | 0.167 | Average extractable slack per trade |
| Nearby tutorials | Afriat, Houtman-Maks | Afriat checks rationalizability; Houtman-Maks searches for rows to drop |

## Solution Method

After the budget comparisons, the problem becomes a graph problem. Each observation is a node. A directed edge points from observation $i$ to observation $j$ when bundle $j$ was strictly cheaper at prices $i$, even though the consumer chose $i$. The edge weight is the saved budget share. Karp's dynamic program computes the maximum mean weight over all directed cycles in this finite graph. For this three-node example, direct enumeration gives a visible check on the plotted cycle.

```text
Inputs: prices p_i, bundles x_i, tolerance eps
1. Form E_ij = p_i . x_j for all observations i,j.
2. Add arc i -> j when (E_ii - E_ij) / E_ii > eps.
3. Attach weight w_ij = (E_ii - E_ij) / E_ii to each arc.
4. Let D_k(v) be the largest total weight of a k-arc path ending at v.
5. Update D_k(v) = max_{u -> v} D_{k-1}(u) + w_uv for k = 1,...,T.
6. Return max_v min_{0 <= k < T} [D_T(v) - D_k(v)] / (T - k).
Output: MPI, the maximum average budget slack in a cycle.
```

For the severe example, the enumerated cycle mean and Karp's MPI both equal $\operatorname{MPI}=0.167$.

## Results

The table separates the logical rejection from the expenditure at stake. All three inconsistent datasets reject GARP, but their MPI values range from three cents to nearly seventeen cents per dollar of expenditure. In this small graph, direct enumeration gives the same cycle mean as Karp's dynamic program.

**GARP Rejection and Money Pump Severity**

| Dataset      | GARP rejects   | Best cycle       | Designed slack   |   Enumerated mean |   Karp MPI |   Violating pairs |
|:-------------|:---------------|:-----------------|:-----------------|------------------:|-----------:|------------------:|
| No cycle     | no             | none             | none             |             0     |      0     |                 0 |
| Small cycle  | yes            | 1 -> 2 -> 3 -> 1 | 3%, 4%, 2%       |             0.03  |      0.03  |                 3 |
| Medium cycle | yes            | 1 -> 2 -> 3 -> 1 | 10%, 13%, 8%     |             0.103 |      0.103 |                 3 |
| Severe cycle | yes            | 1 -> 2 -> 3 -> 1 | 18%, 24%, 8%     |             0.167 |      0.167 |                 3 |

Each arrow points from an observed choice to another bundle that was strictly cheaper at the same prices. Cycling through the red arrows moves the consumer through trades with 16.7 percent average budget slack.

<img src="figures/money-pump-cycle.png" alt="The severe revealed-preference cycle and its edge-level budget slack." width="80%">

The left panel records pass/fail GARP. The right panel keeps the expenditure magnitude, so small, medium, and severe cycles line up by average slack. The black diamonds are the exact cycle means obtained by enumeration in this three-node example.

<img src="figures/mpi-severity-comparison.png" alt="GARP is binary, while the Money Pump Index ranks the severity of failures." width="80%">

## Takeaway

Revealed-preference tests are useful because they turn finite choices into sharp restrictions. The Money Pump Index adds an economic scale after a rejection. It reports the average expenditure a cycle exposes, so an applied researcher can distinguish a harmless near-tie from a large inconsistency. Use it beside the [Afriat test](../revealed-preference-afriat/) and [Houtman-Maks deletion diagnostics](../houtman-maks-rational-subsets/): after rejection, one can ask how much expenditure the worst cycle exposes.

## References

- Echenique, F., Lee, S., & Shum, M. (2011). The money pump as a measure of revealed preference violations. Journal of Political Economy, 119(6), 1201-1223.
- Karp, R. M. (1978). A characterization of the minimum cycle mean in a digraph. Discrete Mathematics, 23(3), 309-311.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
