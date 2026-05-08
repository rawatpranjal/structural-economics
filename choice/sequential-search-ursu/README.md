# Consumer Search with Sequential Inspection Costs

> Estimate search costs from observed search paths and purchases.

## Overview

A consumer does not see every product match value before choosing. She can inspect products one at a time, pay a search cost, remember what she has seen, and stop when further inspection is no longer worth it.

The tutorial implements the sequential search logic used in the empirical framework surveyed by Ursu, Seiler, and Honka. Search paths matter because they reveal more than final purchases. They help separate preference for a product from the cost of learning about it.

## Equations

Product $j$ has known mean match value

$$
\mu_j = \beta q_j - \alpha p_j,
$$

and an uncertain realized match value

$$
u_{ij} = \mu_j + \sigma \varepsilon_{ij},
\qquad \varepsilon_{ij}\sim N(0,1).
$$

Inspecting product $j$ costs

$$
c_j = c_0 \exp(\gamma x_j),
$$

where $x_j$ is product complexity. With perfect recall, the Weitzman reservation
value $z_j$ solves

$$
c_j = E[\max(u_{ij}-z_j,0)].
$$

The consumer searches the uninspected product with the highest $z_j$ if that
reservation value exceeds the best inspected value so far. Otherwise she stops
and buys the best inspected product, or the outside option with value zero.

The simulated-moments estimator chooses

$$
\hat\theta
=
\arg\min_\theta
\left[m_{sim}(\theta)-m_{obs}\right]'
W
\left[m_{sim}(\theta)-m_{obs}\right],
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

## Solution Method

The search rule is deterministic once reservation values and match-value draws are fixed. The estimator therefore simulates full search paths, not just final choices. Fixed draws make the moment criterion comparable across parameter vectors.

```text
Algorithm: estimate sequential search by simulated moments
Input: search paths, purchases, product qualities, prices, complexities
For each candidate theta:
  Compute mean utilities and product-specific search costs
  Solve one reservation-value equation for each product
  Search products in descending reservation-value order
  Stop when the best remaining reservation value is below current best value
  Record searched products, search counts, and purchases
  Compare simulated moments with observed search and purchase moments
Choose theta with the smallest scaled moment distance
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
