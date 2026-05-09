# Cereal Demand and Markup Recovery from Prices

## Overview

A cereal market has differentiated products, observed shares, observed prices, and multi-product owners.

The target object is the marginal cost vector behind those prices. Markups are the wedge between observed prices and recovered costs.

Accounting costs are missing, so the model infers costs from demand and firm optimality. Berry inversion turns shares into mean utilities. IV/2SLS estimates the price slope using excluded cost variation. Bertrand-Nash FOCs then map demand derivatives and ownership into markups.

## Equations

Markets are indexed by $t$. Products are indexed by $j$.
Mean utility collects characteristics, price, and unobserved quality:
$$\delta_{jt} =\beta_0+\beta_{\text{sugar}}x^{\text{sugar}}_{jt} +\beta_{\text{fiber}}x^{\text{fiber}}_{jt} -\alpha p_{jt}+\xi_{jt}.$$

Simple logit shares satisfy
$$s_{jt}=\frac{\exp(\delta_{jt})}{1+\sum_k \exp(\delta_{kt})}, \qquad s_{0t}=\frac{1}{1+\sum_k \exp(\delta_{kt})}.$$
Berry's inversion turns observed shares into a linear estimating equation:
$$\log s_{jt}-\log s_{0t} =\beta_0+\beta_{\text{sugar}}x^{\text{sugar}}_{jt} +\beta_{\text{fiber}}x^{\text{fiber}}_{jt} -\alpha p_{jt}+\xi_{jt}.$$
Identification needs price variation excluded from $\xi_{jt}$. The cost shifter
plays that role in the simulation.

The supply inversion uses the logit derivative matrix:
$$\frac{\partial s_k}{\partial p_j} =\begin{cases} {}-\alpha s_j(1-s_j), & k=j,\\ \alpha s_k s_j, & k\neq j. \end{cases}$$

In the supply equations, $p$ and $s$ are vectors collecting prices and shares across all products in a market.
Firm $f$ chooses prices for its products. Product $j$'s FOC is
$$0=s_j(p)+\sum_k \mathbf 1[f(j)=f(k)](p_k-c_k)\frac{\partial s_k(p)}{\partial p_j}.$$
Here $c_k$ denotes the marginal cost of product $k$.
Let $O_{jk}=1$ when products $j$ and $k$ share an owner. Define the pricing
matrix
$$\Omega_{jk}=-O_{jk}\frac{\partial s_k}{\partial p_j}.$$
The markup vector $m=p-c$ solves
$$\Omega m=s.$$
The recovered cost vector is then $c=p-m$. Ownership matters because a firm
internalizes lost sales across its own products.

## Model Setup

The simulation fixes true demand parameters and marginal costs. This makes demand bias and cost-recovery error observable.

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\alpha$ | 1.5 | Price sensitivity |
| $\beta_{\text{sugar}}$ | 0.3 | Sugar taste |
| $\beta_{\text{fiber}}$ | 0.5 | Fiber taste |
| $\beta_0$ | 1.0 | Base utility |
| Products | 5 | Choco-Bombs, Fiber-Bran, Store-Frosted, Honey-Os, Nutri-Crunch |
| Markets | 100 | Cross-sectional variation in costs |
| Firms | 3 | Firms 1 and 2 own 2 products each (multi-product) |

## Solution Method

The calculation has two stages. Demand estimation recovers the price slope that drives substitution. The supply inversion asks which costs make observed prices optimal.

```text
Inputs: product characteristics, prices, shares, instruments, firm labels
Outputs: demand estimates, elasticities, markups, recovered marginal costs

1. Convert shares to mean utilities: delta_jt = log(s_jt) - log(s_0t).
2. Estimate linear logit demand by OLS as a biased benchmark.
3. Re-estimate by IV/2SLS using excluded cost shifters for price.
4. For one market, compute the logit derivative matrix Delta.
5. Combine Delta with firm ownership to form Omega.
6. Solve Omega m = s for markups, then set c = p - m.
7. Compare recovered costs with the simulated marginal costs.
```

## Results

OLS estimates alpha at 1.009, below the true value 1.500. Unobserved quality raises both demand and price. IV/2SLS estimates alpha at 1.465.

In market 0, recovered marginal costs have mean absolute error 0.455 dollars. These outputs show how demand bias moves recovered costs.

OLS misses alpha because unobserved quality raises demand and price together. IV uses cost-driven price variation and moves toward the truth.

<img src="figures/estimation-comparison.png" alt="Parameter estimates: true, OLS, and IV/2SLS. OLS attenuates price sensitivity because high-xi products command higher prices." width="80%">

The heatmap shows the demand curvature used by the FOC. Logit cross-elasticities depend on rival shares, not product similarity.

<img src="figures/elasticity-heatmap.png" alt="Elasticity matrix from the estimated logit demand system." width="80%">

Demand estimates and ownership turn prices into costs plus markups. Product-level errors remain because the estimated price slope is not exact. Multi-product firms internalize cannibalization across their own products.

<img src="figures/price-decomposition.png" alt="Price = marginal cost + markup. Estimated MC (green, from Bertrand-Nash FOC) compared with true MC (blue)." width="80%">

The main demand error is alpha, the endogenous price coefficient. That error carries into the supply inversion.

**Estimation Results: True vs OLS vs IV/2SLS**

| Parameter   |   True |    OLS |   IV/2SLS |   IV s.e. |
|:------------|-------:|-------:|----------:|----------:|
| alpha       |    1.5 |  1.009 |     1.465 |     0.058 |
| beta_sugar  |    0.3 |  0.376 |     0.409 |     0.022 |
| beta_fiber  |    0.5 |  0.519 |     0.63  |     0.027 |
| beta_const  |    1   | -0.5   |    -0.1   |     0.23  |

## Takeaway

Markup recovery is only as credible as the estimated demand slope. Berry inversion and IV/2SLS estimate that slope from shares, prices, and instruments. The Bertrand-Nash FOC then converts demand derivatives and ownership into marginal costs. Simple logit makes the inversion clear but imposes rigid substitution.

## References

- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics* 25(2), 242-262.
- Nevo, A. (2001). "Measuring Market Power in the Ready-to-Eat Cereal Industry." *Econometrica* 69(2), 307-342.
- Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition, Ch. 3.
