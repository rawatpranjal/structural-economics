# Consumer Search with Sequential Inspection Costs

> Estimate search costs from observed search paths and purchases.

## Overview

A consumer does not see every product match value before choosing. She can inspect products one at a time, pay a search cost, remember what she has seen, and stop when further inspection is no longer worth it.

The tutorial implements the sequential search logic used in the empirical framework surveyed by Ursu, Seiler, and Honka. Search paths matter because they reveal more than final purchases. They help separate preference for a product from the cost of learning about it.

The primitive object is a consideration process. A product can be valuable if inspected, but still rarely bought because consumers do not pay to inspect it. That is why the data include both searched products and final purchases.

## Equations

Product $j$ has known mean match value

$$
\begin{aligned}
\mu_j
&=
\beta q_j - \alpha p_j,
\end{aligned}
$$

and an uncertain realized match value

$$
\begin{aligned}
u_{ij}
&=
\mu_j + \sigma \varepsilon_{ij},
\qquad \varepsilon_{ij}\sim N(0,1).
\end{aligned}
$$

Inspecting product $j$ costs

$$
\begin{aligned}
c_j
&=
c_0 \exp(\gamma x_j),
\end{aligned}
$$

where $x_j$ is product complexity. With perfect recall, the Weitzman reservation
value $z_j$ solves

$$
\begin{aligned}
c_j
&=
E[\max(u_{ij}-z_j,0)].
\end{aligned}
$$

The consumer keeps an inspected set $S_i$ and a current best value

$$
\begin{aligned}
b_i
&=
\max\{0,\max_{j\in S_i} u_{ij}\}.
\end{aligned}
$$

She searches the uninspected product with the highest $z_j$ if that reservation
value exceeds $b_i$. Otherwise she stops and buys the best inspected product, or
the outside option when all inspected values are below zero.

The simulated-moments estimator chooses

$$
\begin{aligned}
\hat\theta
&=
\arg\min_\theta
\left[m_{sim}(\theta)-m_{obs}\right]^{\top}
W
\left[m_{sim}(\theta)-m_{obs}\right].
\end{aligned}
$$

where moments include product search rates, purchase shares, average searches,
and the probability of stopping after one search.

In this exercise, $\gamma$ is fixed and the estimator recovers the quality taste
and the base search-cost level.

## Model Setup

| Object | Value | Role |
|--------|-------|------|
| Products | 5 | Alternatives that can be inspected |
| Observed consumers | 4,000 | Synthetic search paths and purchases |
| Simulation consumers | 9,000 | Fixed draws for simulated moments |
| True quality taste | 1.18 | Preference weight on product quality |
| True base search cost | 0.080 | Cost level before complexity adjustment |
| Complexity slope | 0.48 | Fixed search-cost increase with product complexity |
| Match-value sd | 0.85 | Fixed uncertainty about product fit |

**Numerical settings**

| Setting | Value | Role |
|---------|-------|------|
| Optimizer | Nelder-Mead | Derivative-free search over quality taste and log base cost |
| Start | (0.95, log 0.09) | Initial quality taste and base search cost |
| Moment scale floor | 0.08 | Prevents tiny moments from dominating the criterion |
| Reservation bracket | [-8, 8] | Root-search bracket for the normal reservation equation |
| Max iterations | 220 | Nelder-Mead iteration cap |
| Tolerances | xatol=1e-04, fatol=1e-06 | Stopping rule for parameter moves and criterion changes |
| Counterfactual grid | [0.5, 0.75, 1.0, 1.5, 2.0] | Search-cost multipliers used in the policy experiment |

## Solution Method

The search rule has two layers. First, reservation values rank products before the consumer knows her idiosyncratic match values. Second, after each search, the realized match value updates the current best option. Search continues only when the best remaining reservation value is above that current best value.

For a normal match distribution, the reservation equation can be solved as a one-dimensional root. A high search cost lowers $z_j$ because the product must offer more option value before inspection is worthwhile. A high mean utility raises $z_j$ because the product is likely to be useful if inspected.

```text
Algorithm 1: Weitzman reservation values
Input: mean utilities mu_j, search costs c_j, match-value sd sigma
Output: reservation values z_j and search order
for each product j:
    solve for k_j in E[max(Z - k_j, 0)] = c_j / sigma, with Z ~ N(0,1)
    set z_j = mu_j + sigma * k_j
sort products from highest z_j to lowest z_j
```

```text
Algorithm 2: simulate one consumer's search path
Input: reservation values z_j, realized match values u_ij, outside value 0
Output: searched set S_i, search count, purchase y_i
initialize S_i = empty, best value b_i = 0, best product = outside
while there is an unsearched product with max z_j > b_i:
    choose the unsearched product j with the highest z_j
    add j to S_i and observe u_ij
    if u_ij > b_i:
        set b_i = u_ij and best product = j
stop and purchase the best product, or outside if no inspected product beats 0
```

The estimator simulates the full path for many consumers at each parameter vector. It matches search rates and purchase shares, so the same product can be identified as hard to discover rather than simply low quality.

```text
Algorithm 3: simulated-moments estimation
Input: observed search indicators, purchases, product data, fixed shocks eps_i
Parameters: theta = (beta, log c_0), with gamma and sigma fixed
Observed targets: product search rates, purchase shares, mean searches, one-search rate
for each candidate theta proposed by Nelder-Mead:
    compute mu_j(theta) and c_j(theta)
    solve reservation values z_j(theta)
    for each simulated consumer i:
        build realized match values u_ij(theta, eps_i)
        simulate the sequential search path using Algorithm 2
    compute simulated moments m_sim(theta)
    scale moment errors by max(abs(m_obs), moment scale floor)
    evaluate the quadratic criterion
choose theta with the smallest scaled moment distance
```

Purchase data alone confound low demand with high search costs. Search paths help because a product can be attractive among consumers who inspect it but rarely inspected when its search cost is high.

## Results

The fitted model matches both product inspection and final choice. This matters because high-quality products can have low purchase shares either because they are unattractive or because few consumers pay to learn about them.

<img src="figures/search-and-choice-fit.png" alt="Search and purchase fit" width="80%">

Full-information demand is the benchmark where every match value is observed for free. Sequential search shifts demand because some products never enter a consumer's consideration set.

<img src="figures/consideration-demand.png" alt="Sequential-search versus full-information demand" width="80%">

Increasing search costs lowers the amount of inspection and pushes some consumers to stop earlier. The inside purchase share falls because consumers are less likely to discover a product match that beats the outside option.

<img src="figures/search-cost-counterfactual.png" alt="Search-cost counterfactual" width="80%">

Known-truth recovery is approximate because the estimator matches moments, not the exact likelihood. The residual table shows which observed search and purchase summaries drive the fit.

**Known-truth parameter recovery**

| Parameter             |   True |   Estimate |   Error | Status    |
|:----------------------|-------:|-----------:|--------:|:----------|
| Quality taste         |   1.18 |     1.193  |  0.013  | estimated |
| Base search cost      |   0.08 |     0.0791 | -0.0009 | estimated |
| Complexity cost slope |   0.48 |     0.48   |  0      | fixed     |

**Search and purchase moment fit**

| Moment                   |   Observed target |   Simulated at estimate |   Difference |
|:-------------------------|------------------:|------------------------:|-------------:|
| Search rate: Basic       |            0.0965 |                  0.0903 |      -0.0062 |
| Search rate: Comfort     |            0.3718 |                  0.3838 |       0.012  |
| Search rate: Sport       |            0.6645 |                  0.6787 |       0.0142 |
| Search rate: Premium     |            1      |                  1      |       0      |
| Search rate: Boutique    |            0.242  |                  0.2426 |       0.0006 |
| Purchase share: Basic    |            0.0268 |                  0.0259 |      -0.0009 |
| Purchase share: Comfort  |            0.1152 |                  0.1233 |       0.0081 |
| Purchase share: Sport    |            0.2735 |                  0.2733 |      -0.0002 |
| Purchase share: Premium  |            0.4895 |                  0.4811 |      -0.0084 |
| Purchase share: Boutique |            0.095  |                  0.0963 |       0.0013 |
| Purchase share: Outside  |            0      |                  0      |       0      |
| Average searches         |            2.3748 |                  2.3953 |       0.0206 |
| One-search rate          |            0.3355 |                  0.3213 |      -0.0142 |

**Search-cost counterfactuals**

|   Search cost multiplier |   Average searches |   Inside purchase share |   Outside share |
|-------------------------:|-------------------:|------------------------:|----------------:|
|                     0.5  |             3.0218 |                       1 |               0 |
|                     0.75 |             2.6519 |                       1 |               0 |
|                     1    |             2.3953 |                       1 |               0 |
|                     1.5  |             2.0396 |                       1 |               0 |
|                     2    |             1.818  |                       1 |               0 |

## Takeaway

Sequential search turns observed demand into a joint outcome of preferences and information acquisition. Search-path data are valuable because they show which products entered consideration before the purchase. That is the key empirical distinction between a search model and a full-information discrete choice model.

## References

- [Ursu, R., Seiler, S., and Honka, E. (2025). The sequential search model: A framework for empirical research. *Quantitative Marketing and Economics*, 23, 165-213.](https://doi.org/10.1007/s11129-024-09291-2)
- [Weitzman, M. L. (1979). Optimal Search for the Best Alternative. *Econometrica*, 47(3), 641-654.](https://doi.org/10.2307/1910412)
