# Cereal Demand and Markup Recovery from Prices

> Berry inversion, IV/2SLS, and Bertrand-Nash pricing FOCs.

## Overview

Think of a cereal aisle where a researcher sees product characteristics, prices, market shares, and which firm owns each box. The researcher wants markups and marginal costs, because those objects drive counterfactual prices and market-power measurement. Accounting marginal costs are usually missing, so the costs have to be inferred from an economic pricing model.

The example uses five cereal products sold by three firms. Firms charge more for products with high unobserved quality, which makes price endogenous in the demand equation. Berry inversion turns shares into mean utilities, IV/2SLS uses cost shifters and rival characteristics to estimate price sensitivity, and the Bertrand-Nash first-order condition converts demand curvature into markups and marginal costs. The logit model keeps the mechanics transparent, while the elasticity figures show the cost of that transparency: substitution follows IIA.

## Equations

There are markets $t$, products $j$, and an outside option. Mean utility collects
observed characteristics, price, and unobserved quality:
$$
\delta_{jt}
=\beta_0+\beta_{\text{sugar}}x^{\text{sugar}}_{jt}
+\beta_{\text{fiber}}x^{\text{fiber}}_{jt}
-\alpha p_{jt}+\xi_{jt}.
$$

Simple logit shares satisfy
$$
s_{jt}=\frac{\exp(\delta_{jt})}{1+\sum_k \exp(\delta_{kt})},
\qquad
s_{0t}=\frac{1}{1+\sum_k \exp(\delta_{kt})}.
$$
Berry's inversion turns observed shares into a linear estimating equation:
$$
\log s_{jt}-\log s_{0t}
=\beta_0+\beta_{\text{sugar}}x^{\text{sugar}}_{jt}
+\beta_{\text{fiber}}x^{\text{fiber}}_{jt}
-\alpha p_{jt}+\xi_{jt}.
$$
The price coefficient is identified from price variation that is excluded from
$\xi_{jt}$. In the simulated data, cost shifters and rival characteristics play
that role.

The logit elasticity matrix is
$$
\eta_{jj}=-\alpha p_j(1-s_j), \qquad
\eta_{jk}=\alpha p_k s_k, \quad j\neq k.
$$
The cross-elasticity $\eta_{jk}$ depends on product $k$'s price and share, but
not on how close products $j$ and $k$ are. That restriction is what makes simple
logit easy to invert and too rigid for many product-space applications.

On the supply side, firm $f$ chooses prices for its products. Product $j$'s FOC is
$$
0=s_j(p)+\sum_k
\mathbf 1[f(j)=f(k)](p_k-c_k)\frac{\partial s_k(p)}{\partial p_j}.
$$
Let $O_{jk}=1$ when products $j$ and $k$ are owned by the same firm, and define
the pricing matrix
$$
\Omega_{jk}=-O_{jk}\frac{\partial s_k}{\partial p_j}.
$$
The markup vector $m=p-c$ solves
$$
\Omega m=s.
$$
The recovered cost vector is then $c=p-m$. Multi-product firms internalize
business stolen from their own products, so ownership enters directly into the
cost recovery.

## Model Setup

The data are simulated so the true demand parameters and marginal costs are known. This lets the run separate two mistakes that real data often mix together: estimating the wrong price slope because price is endogenous, and recovering the wrong marginal costs because the demand curvature is wrong.

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

The computation separates the demand step from the supply inversion. Demand estimation recovers the price slope that governs substitution. The supply step then treats observed prices as firm choices and asks which marginal costs make those choices optimal under Bertrand-Nash pricing.

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

The first-stage F-statistic is 303.1, so the instrument set is strong in this synthetic design. The strength is built in because the lesson is the chain from demand-side IV to supply-side markup recovery, rather than weak-instrument diagnostics.

## Results

OLS recovers a price-sensitivity estimate of 1.009, far below the true value 1.500, because high-$\xi$ products are more popular and more expensive. IV/2SLS moves the estimate to 1.465. In market 0, the recovered marginal costs have mean absolute error 0.455 dollars. The figures connect those numbers to the economic objects: biased demand slopes, rigid substitution, and the price decomposition into markup and cost.

The red OLS bar misses the true alpha because unobserved quality raises demand and price at the same time. The blue IV/2SLS bar moves back toward the truth by using cost-driven price variation.

<img src="figures/estimation-comparison.png" alt="Parameter estimates: true, OLS, and IV/2SLS. OLS attenuates price sensitivity because high-xi products command higher prices." width="80%">

Each column of cross-elasticities is identical because the logit model forces all products to be equally substitutable. When a sugary cereal raises its price, the model sends consumers to a similar sugary cereal and to a dissimilar fiber cereal in the same proportional way.

<img src="figures/elasticity-heatmap.png" alt="Elasticity matrix. Cross-elasticities in each column are identical, the IIA limitation of the simple logit." width="80%">

The decomposition is the supply-side accounting exercise: demand estimates plus the Bertrand-Nash FOC turn observed prices into markup and marginal-cost components. Estimated costs are imperfect product by product because the estimated demand slope is not exactly the true slope, but the exercise recovers the economic object needed for counterfactual pricing. Multi-product firms (Choco-Bombs and Store-Frosted, both owned by Firm 1) charge higher markups because they internalize cannibalization across their own products.

<img src="figures/price-decomposition.png" alt="Price = marginal cost + markup. Estimated MC (green, from Bertrand-Nash FOC) compared with true MC (blue)." width="80%">

Within each panel, all bars have the same height because every rival gains the same cross-elasticity regardless of product similarity. That is the IIA property. The BLP random coefficients model (see blp-random-coefficients/) breaks the restriction by allowing consumer heterogeneity.

<img src="figures/iia-demonstration.png" alt="IIA demonstration. When any product raises its price, substitution to each rival is proportional to that rival's market share, rather than how similar the products are." width="80%">

Compare the OLS and IV/2SLS columns: the bias is concentrated in alpha (price sensitivity) because price is the endogenous variable. The characteristic coefficients (sugar, fiber) are less affected because product attributes have weaker correlation with the unobserved quality term.

**Estimation Results: True vs OLS vs IV/2SLS**

| Parameter   |   True |    OLS |   IV/2SLS |   IV s.e. |
|:------------|-------:|-------:|----------:|----------:|
| alpha       |    1.5 |  1.009 |     1.465 |     0.058 |
| beta_sugar  |    0.3 |  0.376 |     0.409 |     0.022 |
| beta_fiber  |    0.5 |  0.519 |     0.63  |     0.027 |
| beta_const  |    1   | -0.5   |    -0.1   |     0.23  |

## Takeaway

The recovered supply-side object is the marginal cost vector that rationalizes observed prices under the estimated demand system and ownership matrix. A biased price coefficient therefore changes more than a demand table: it changes markups and costs. Simple logit makes Berry inversion and markup recovery easy to see, but its IIA substitution pattern is too rigid for many merger and product-space applications. Random-coefficients demand in [BLP](../blp-random-coefficients/) lets substitution vary with consumer heterogeneity and product characteristics.

## References

- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics* 25(2), 242-262.
- Berry, S., Levinsohn, J., and Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica* 63(4), 841-890.
- Nevo, A. (2001). "Measuring Market Power in the Ready-to-Eat Cereal Industry." *Econometrica* 69(2), 307-342.
- Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition, Ch. 3.
