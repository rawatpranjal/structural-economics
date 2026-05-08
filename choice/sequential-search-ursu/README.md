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
Algorithm 1: compute Weitzman reservation values
Inputs   product primitives {q_j,p_j,x_j}_{j=1..J};
         theta=(beta,ell_c); fixed alpha, gamma, sigma
Outputs  reservation values z_j and priority order pi

c_0 <- exp(ell_c)
for j = 1,...,J:
    mu_j <- beta q_j - alpha p_j
    c_j  <- c_0 exp(gamma x_j)
    solve for k_j in G(k_j) = c_j / sigma
        where G(k) = phi(k) - k[1 - Phi(k)]
    z_j <- mu_j + sigma k_j

pi <- products sorted so z_{pi_1} >= z_{pi_2} >= ... >= z_{pi_J}
return {z_j}, pi
```

```text
Algorithm 2: simulate consumer i's search path
Inputs   reservation values {z_j}; order pi; shocks eps_ij;
         mean utilities {mu_j}; match-value sd sigma
Outputs  searched set S_i; search count n_i; purchase y_i

S_i <- empty set
A_i <- {1,...,J}
b_i <- 0
y_i <- outside option

for h = 1,...,J following pi_h:
    j <- pi_h
    if z_j <= b_i:
        break
    S_i <- S_i union {j}
    A_i <- A_i \ {j}
    u_ij <- mu_j + sigma eps_ij
    if u_ij > b_i:
        b_i <- u_ij
        y_i <- j

n_i <- |S_i|
return S_i, n_i, y_i
```

The estimator simulates the full path for many consumers at each parameter vector. It matches search rates and purchase shares, so the same product can be identified as hard to discover rather than simply low quality.

```text
Algorithm 3: estimate theta by simulated moments
Inputs   observed moments m_obs; fixed simulation shocks eps_ij^s;
         starting value theta_0; moment scale floor a_min
Output   theta_hat

for each moment ell:
    a_ell <- max(|m_obs,ell|, a_min)

repeat until Nelder-Mead stops:
    choose a candidate theta^m=(beta^m,ell_c^m)
    compute {mu_j(theta^m), c_j(theta^m), z_j(theta^m)} using Algorithm 1
    for simulated consumer s = 1,...,S:
        simulate S_s(theta^m), n_s(theta^m), y_s(theta^m) using Algorithm 2
    m_sim(theta^m) <- search rates, purchase shares, mean n_s, and Pr(n_s=1)
    e_ell(theta^m) <- [m_sim,ell(theta^m) - m_obs,ell] / a_ell
    Q_S(theta^m) <- sum_ell e_ell(theta^m)^2

theta_hat <- argmin_theta Q_S(theta)
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

**Computational tricks used here**

- The same shocks $\varepsilon_{ij}^{s}$ are reused for every trial $\theta$.
- The reservation equation is solved in standardized units $k_j=(z_j-\mu_j)/\sigma$.
- The fixed bracket keeps each scalar root solve away from unstable tails.
- The optimizer uses $\ell_c=\log c_0$ so the base search cost stays positive.
- Moment scaling prevents a tiny search rate from dominating the criterion.
- Counterfactual cost multipliers reuse $\hat\theta$ and isolate the cost channel.

## References

- [Ursu, R., Seiler, S., and Honka, E. (2025). The sequential search model: A framework for empirical research. *Quantitative Marketing and Economics*, 23, 165-213.](https://doi.org/10.1007/s11129-024-09291-2)
- [Weitzman, M. L. (1979). Optimal Search for the Best Alternative. *Econometrica*, 47(3), 641-654.](https://doi.org/10.2307/1910412)
