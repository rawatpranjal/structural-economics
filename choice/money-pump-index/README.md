# Money Pump Index for Revealed Preference

> Putting economic size on a revealed-preference violation.

## Overview

A GARP rejection is a statement about internal consistency, not about economic magnitude. Two datasets can both violate revealed preference, but one violation may be a nearly indifferent accounting error while another creates a large cyclic arbitrage. The money-pump interpretation asks how much budget slack an outside trader could extract by repeatedly moving the consumer around the revealed-preference cycle.

The example keeps the economic object deliberately small: three observed choices, three bundles, and own expenditure normalized to one. In the severe case the consumer chooses A when B was 18 percent cheaper, chooses B when C was 24 percent cheaper, and chooses C when A was 8 percent cheaper. GARP says this cannot come from one stable utility ordering. The Money Pump Index says the exploitable cycle is worth 16.7% of expenditure per trade.

## Equations

There are $T$ observations. Observation $i$ records a price vector $p_i \in \mathbb{R}^G_+$
and chosen bundle $x_i \in \mathbb{R}^G_+$. Let

$$E_{ij}=p_i \cdot x_j$$

be the cost of bundle $j$ at prices $i$. Choosing $x_i$ directly reveals
$x_i \succeq^D x_j$ when $E_{ii} \ge E_{ij}$. For strict comparisons, define
the relative budget slack on a direct revealed-preference edge as

$$w_{ij} = \frac{E_{ii} - E_{ij}}{E_{ii}}.$$

In this tutorial the graph keeps edges with $w_{ij}>0$. For a directed cycle
$C=(i_1,\ldots,i_m,i_1)$, the average slack is

$$\bar w(C)=\frac{1}{m}\sum_{\ell=1}^{m} w_{i_\ell,i_{\ell+1}}.$$

The Money Pump Index is the largest average slack over all directed cycles in
the revealed-preference graph:

$$\operatorname{MPI} = \max_C \bar w(C).$$

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Observations | 3 | One price vector and one chosen bundle per observation |
| Bundles | 3 | A, B, and C are the only candidate bundles |
| Own expenditure | 1.00 | Each chosen bundle is normalized to cost one |
| Severe-cycle slack | 18%, 24%, 8% | Slack on A over B, B over C, and C over A |
| Severe MPI | 0.167 | Average extractable slack per trade |
| Nearby tutorials | Afriat, Houtman-Maks | Afriat asks whether choices are rationalizable; Houtman-Maks asks which rows to drop |

## Solution Method

The first step is still the revealed-preference test: compare every chosen bundle with every other bundle at the same prices. The extra step is to put weights on strict revealed-preference arcs and solve a maximum mean-cycle problem. Karp's dynamic program is exact for this finite graph. The script also enumerates cycles only because the graph has three nodes, which gives a transparent benchmark for the plotted examples.

```text
Inputs: prices p_i, bundles x_i, tolerance eps
1. Form E_ij = p_i . x_j for all observations i,j.
2. Add arc i -> j when (E_ii - E_ij) / E_ii > eps.
3. Attach weight w_ij = (E_ii - E_ij) / E_ii to each arc.
4. Let D_k(v) be the largest total weight of any k-arc path ending at v.
5. Iterate D_k(v) = max_{u -> v} D_{k-1}(u) + w_uv for k = 1,...,T.
6. Return max_v min_{0 <= k < T} [D_T(v) - D_k(v)] / (T-k).
Output: MPI, the maximum average budget slack in a cycle.
```

For the severe example, the enumerated cycle mean and Karp's MPI both equal $\operatorname{MPI}=0.167$.

## Results

The first column of interest is the GARP rejection: it is the same yes/no answer for all three inconsistent datasets. The severity columns show why that is not enough. In this small graph, direct enumeration gives the same cycle mean as Karp's dynamic program.

**GARP Rejection and Money Pump Severity**

| Dataset      | GARP rejects   | Best cycle       | Designed slack   |   Enumerated mean |   Karp MPI |   Violating pairs |
|:-------------|:---------------|:-----------------|:-----------------|------------------:|-----------:|------------------:|
| No cycle     | no             | none             | none             |             0     |      0     |                 0 |
| Small cycle  | yes            | 1 -> 2 -> 3 -> 1 | 3%, 4%, 2%       |             0.03  |      0.03  |                 3 |
| Medium cycle | yes            | 1 -> 2 -> 3 -> 1 | 10%, 13%, 8%     |             0.103 |      0.103 |                 3 |
| Severe cycle | yes            | 1 -> 2 -> 3 -> 1 | 18%, 24%, 8%     |             0.167 |      0.167 |                 3 |

Each arrow points from a chosen bundle to another bundle that was strictly cheaper at the same prices. The red cycle is not just a graph-theoretic oddity: it is the sequence of trades that creates the money pump.

<img src="figures/money-pump-cycle.png" alt="The severe revealed-preference cycle and its edge-level budget slack." width="80%">

The left panel deliberately throws away magnitude. The right panel keeps it: small, medium, and severe cycles all fail the same rationalizability test, but the average slack available to a trader is very different. The black diamonds are the exact cycle means obtained by enumeration in this three-node example.

<img src="figures/mpi-severity-comparison.png" alt="GARP is binary, while the Money Pump Index ranks the severity of failures." width="80%">

## Takeaway

The Money Pump Index is useful because it keeps the economic content of a revealed-preference failure in view. GARP asks whether the finite dataset is rationalizable. MPI asks how much expenditure is exposed by the worst cycle. That makes it a complement to the [Afriat test](../revealed-preference-afriat/) and to [Houtman-Maks deletion diagnostics](../houtman-maks-rational-subsets/): after rejection, one can ask whether the problem is small, large, or concentrated in a few observations.

## References

- Echenique, F., Lee, S., & Shum, M. (2011). The money pump as a measure of revealed preference violations. Journal of Political Economy, 119(6), 1201-1223.
- Karp, R. M. (1978). A characterization of the minimum cycle mean in a digraph. Discrete Mathematics, 23(3), 309-311.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
