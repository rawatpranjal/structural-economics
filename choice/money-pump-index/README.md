# Revealed-Preference Cycles and the Money Pump Index

## Overview

A consumer buys bundle A at one price vector, bundle B at another, and bundle C at a third. These choices can violate GARP because each chosen bundle makes another affordable bundle look strictly better.

The Money Pump Index prices the violation. It measures the average budget slack along the worst revealed-preference cycle.

The task becomes a finite graph problem. Nodes are observations, edges carry budget slack, and Karp's dynamic program finds the maximum mean cycle.

## Equations

There are $`T`$ observations. Observation $`i`$ records a price vector
$`p_i \in \mathbb{R}^G_+`$ and chosen bundle $`x_i \in \mathbb{R}^G_+`$.
Let

```math
E_{ij}=p_i \cdot x_j
```

be the cost of bundle $`j`$ at observation $`i`$ prices. Choosing $`x_i`$ when
$`x_j`$ was affordable, $`E_{ii} \ge E_{ij}`$, directly reveals
$`x_i \succeq^D x_j`$. For strict comparisons, define the relative budget slack
on a direct revealed-preference edge as

```math
w_{ij} = \frac{E_{ii} - E_{ij}}{E_{ii}}.
```

The graph keeps edges with $`w_{ij}>0`$. For a directed cycle
$`C=(i_1,\ldots,i_m,i_1)`$, average slack is

```math
\bar w(C)=\frac{1}{m}\sum_{\ell=1}^{m} w_{i_\ell,i_{\ell+1}}.
```

The Money Pump Index is the largest average slack over all directed cycles in
the revealed-preference graph:

```math
\mathrm{MPI} = \max_C \bar w(C).
```

## Worked Numerical Example

Take three observations. Each bundle is a unit good, so $`x_1 = (1,0,0)`$, $`x_2 = (0,1,0)`$, $`x_3 = (0,0,1)`$. Prices are

```math
p_1 = (1.00,\ 0.82,\ 1.20), \quad
p_2 = (1.20,\ 1.00,\ 0.76), \quad
p_3 = (0.92,\ 1.20,\ 1.00).
```

Because each bundle is a unit good, $`E_{ij} = p_i \cdot x_j`$ equals the $`j`$-th component of $`p_i`$. The own-expenditure diagonal is $`E_{11} = E_{22} = E_{33} = 1`$.

Check the cycle $`1 \to 2 \to 3 \to 1`$. For each arc, the cross-cost is strictly below own expenditure, so the consumer's earlier choice directly reveals the new bundle was affordable and foregone.

Edge $`1 \to 2`$: bundle $`x_2`$ cost $`E_{12} = 0.82`$ at prices $`p_1`$, while the consumer spent $`E_{11} = 1.00`$. The relative slack is

```math
w_{12} = \frac{E_{11} - E_{12}}{E_{11}} = \frac{1.00 - 0.82}{1.00} = 0.18.
```

Edge $`2 \to 3`$: bundle $`x_3`$ cost $`E_{23} = 0.76`$ at prices $`p_2`$, while the consumer spent $`E_{22} = 1.00`$.

```math
w_{23} = \frac{E_{22} - E_{23}}{E_{22}} = \frac{1.00 - 0.76}{1.00} = 0.24.
```

Edge $`3 \to 1`$: bundle $`x_1`$ cost $`E_{31} = 0.92`$ at prices $`p_3`$, while the consumer spent $`E_{33} = 1.00`$.

```math
w_{31} = \frac{E_{33} - E_{31}}{E_{33}} = \frac{1.00 - 0.92}{1.00} = 0.08.
```

All three weights are positive, confirming a GARP-violating cycle. Average the three edge weights:

```math
\mathrm{MPI} = \frac{w_{12} + w_{23} + w_{31}}{3}
             = \frac{0.18 + 0.24 + 0.08}{3}
             = \frac{0.50}{3}
             = \boxed{0.1\overline{6}}.
```

Each trade around the cycle extracts on average 16.7 percent of the consumer's budget. A money pump that runs the consumer once through this three-step sequence captures one-sixth of expenditure.

## Model Setup

| Object | Value | Interpretation |
|---|---:|---|
| Observations | 3 | One price vector and one chosen bundle in each row |
| Bundles | 3 | A, B, and C are the only candidate bundles |
| Own expenditure | 1.00 | Each chosen bundle costs one at its own prices |
| Severe-cycle slack | 18%, 24%, 8% | Slack on A over B, B over C, and C over A |
| Severe MPI | 0.167 | Average extractable slack per trade |

## Solution Method

After the budget comparisons, the data are a directed graph. Each observation is a node. An edge $`i \to j`$ exists when bundle $`j`$ was strictly cheaper at prices $`i`$. The edge weight is the saved budget share. Karp's dynamic program computes the maximum mean weight cycle.

```text
Inputs: prices p_i, bundles x_i, tolerance eps
1. Form E_ij = p_i . x_j for all observations i,j.
2. Add arc i -> j when (E_ii - E_ij) / E_ii > eps.
3. Attach weight w_ij = (E_ii - E_ij) / E_ii to each arc.
4. Let D_k(v) be the largest total weight of a k-arc path ending at v.
5. Update D_k(v) = max[u -> v] D[k-1](u) + w_uv for k = 1,...,T.
6. Return max_v min[0 <= k < T] [D_T(v) - D_k(v)] / (T - k).
Output: MPI, the maximum average budget slack in a cycle.
```

## Results

The table separates a logical rejection from expenditure at stake. All three inconsistent datasets reject GARP. Their MPI values range from 0.030 to 0.167.

**GARP Rejection and Money Pump Severity**

| Dataset      | GARP rejects   | Best cycle       | Designed slack   |   Karp MPI |
|:-------------|:---------------|:-----------------|:-----------------|-----------:|
| No cycle     | no             | none             | none             |      0     |
| Small cycle  | yes            | 1 -> 2 -> 3 -> 1 | 3%, 4%, 2%       |      0.03  |
| Medium cycle | yes            | 1 -> 2 -> 3 -> 1 | 10%, 13%, 8%     |      0.103 |
| Severe cycle | yes            | 1 -> 2 -> 3 -> 1 | 18%, 24%, 8%     |      0.167 |

Each arrow points from an observed choice to a cheaper bundle at the same prices. The red cycle exposes 16.7 percent average budget slack.

<img src="figures/money-pump-cycle.png" alt="The severe revealed-preference cycle and its edge-level budget slack." width="80%">

The left panel records pass/fail GARP. The right panel keeps the expenditure scale across small, medium, and severe cycles.

<img src="figures/mpi-severity-comparison.png" alt="GARP is binary, while the Money Pump Index ranks the severity of failures." width="80%">

## Takeaway

Binary GARP tests say whether choices pass the axioms. The Money Pump Index says how much expenditure the worst cycle exposes. That scale separates small inconsistencies from large money-pump opportunities.

## References

- Echenique, F., Lee, S., & Shum, M. (2011). The money pump as a measure of revealed preference violations. Journal of Political Economy, 119(6), 1201-1223.
- Karp, R. M. (1978). A characterization of the minimum cycle mean in a digraph. Discrete Mathematics, 23(3), 309-311.
- Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.
