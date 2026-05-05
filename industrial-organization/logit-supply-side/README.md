# Logit Demand and Markup Recovery

> Berry inversion, price endogeneity, and the supply-side recovery of marginal costs.

## Overview

A differentiated-products demand estimate becomes economically useful when it can say something about markups and marginal costs. In many IO applications, the researcher observes prices, shares, product characteristics, and firm ownership, but not accounting marginal cost. The supply side uses the firm's pricing first-order condition to recover those costs from demand curvature.

The tutorial builds a synthetic cereal market with five products and three firms. Prices are endogenous because firms charge more for products with high unobserved quality. OLS therefore understates price sensitivity. IV/2SLS uses cost shifters and rival-characteristic instruments to recover demand, then the Bertrand-Nash FOC decomposes each observed price into a markup and marginal cost. Simple logit is not a rich substitution model; the elasticity figures make its IIA restriction visible.

## Equations

There are markets $t$, products $j$, and an outside option. Mean utility is
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
$\xi_{jt}$, here cost shifters and rival characteristics.

The logit elasticity matrix is
$$
\eta_{jj}=-\alpha p_j(1-s_j), \qquad
\eta_{jk}=\alpha p_k s_k, \quad j\neq k.
$$
The cross-elasticity $\eta_{jk}$ depends on product $k$'s price and share, but
not on how close products $j$ and $k$ are. That is the IIA restriction.

On the supply side, firm $f$ chooses prices for its products. Product $j$'s FOC is
$$
0=s_j(p)+\sum_k
\mathbf 1[f(j)=f(k)](p_k-c_k)\frac{\partial s_k(p)}{\partial p_j}.
$$
Let $\Delta_{jk}=\partial s_j/\partial p_k$ and let $O_{jk}=1$ when products
$j$ and $k$ are owned by the same firm. The markup equation is
$$
p-c=-(O\circ \Delta')^{-1}s.
$$
Multi-product firms internalize business stolen from their own products, so the
ownership matrix is part of the cost recovery exercise.

## Model Setup

The data are simulated so the true demand parameters and marginal costs are known. This separates two errors that are often mixed in real applications: demand bias from endogenous prices and cost-recovery error from using the wrong demand curvature.

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

The computation keeps demand estimation and supply inversion separate. First recover mean utilities from shares and estimate demand. Then take the estimated price coefficient into the Bertrand-Nash FOC for one market.

```text
Inputs: product characteristics, prices, shares, instruments, firm labels
Outputs: demand estimates, elasticities, markups, recovered marginal costs

1. Compute delta_jt = log(s_jt) - log(s_0t).
2. Regress delta_jt on characteristics and price by OLS.
3. Re-estimate by IV/2SLS using excluded cost shifters for price.
4. In a representative market, form the logit derivative matrix Delta.
5. Build ownership O from firm labels.
6. Recover markups from p - c = -[(O .* Delta')]^{-1}s.
7. Compare recovered marginal costs with the simulated truth.
```

The first-stage F-statistic is 303.1, so the instrument set is strong in this synthetic design. That strength is deliberately built in; the exercise is about the mechanics of demand-side IV and supply-side markup recovery, not weak-instrument diagnostics.

## Results

OLS recovers a price-sensitivity estimate of 1.009, far below the true value 1.500, because high-$\xi$ products are both more popular and more expensive. IV/2SLS moves the estimate to 1.465. In market 0, the recovered marginal costs have mean absolute error 0.455 dollars. The remaining figures put the recovery in context and indicate where simple logit remains restrictive.

The gap between OLS (red) and the true value (green) for alpha is the endogeneity bias: OLS understates price sensitivity because unobserved quality raises both demand and price simultaneously. IV/2SLS (blue) recovers the parameter by isolating exogenous cost-driven price variation.

<img src="figures/estimation-comparison.png" alt="Parameter estimates: True vs OLS (biased) vs IV/2SLS (consistent). OLS attenuates price sensitivity because high-xi products command higher prices." width="80%">

Each column of cross-elasticities is identical because the logit model forces all products to be equally substitutable. When a sugary cereal raises its price, the model predicts equal substitution to a similar sugary cereal and to a dissimilar fiber cereal -- an unrealistic restriction.

<img src="figures/elasticity-heatmap.png" alt="Elasticity matrix. Cross-elasticities in each column are identical -- the IIA limitation of the simple logit." width="80%">

The decomposition is the supply-side accounting exercise: demand estimates plus the Bertrand-Nash FOC turn observed prices into markup and marginal-cost components. Estimated costs are imperfect product by product because the estimated demand slope is not exactly the true slope, but the exercise recovers the economic object needed for counterfactual pricing. Multi-product firms (Choco-Bombs and Store-Frosted, both owned by Firm 1) charge higher markups because they internalize cannibalization across their own products.

<img src="figures/price-decomposition.png" alt="Price = marginal cost + markup. Estimated MC (green, from Bertrand-Nash FOC) compared with true MC (blue). No accounting data required." width="80%">

Within each panel, all bars have the same height -- every rival gains the same cross-elasticity regardless of product similarity. That is the IIA property. The BLP random coefficients model (see blp-random-coefficients/) breaks the restriction by allowing consumer heterogeneity.

<img src="figures/iia-demonstration.png" alt="IIA demonstration. When any product raises its price, substitution to each rival is proportional to that rival's market share -- not to how similar the products are." width="80%">

Compare the OLS and IV/2SLS columns: the bias is concentrated in alpha (price sensitivity) because price is the endogenous variable. The characteristic coefficients (sugar, fiber) are less affected because product attributes have weaker correlation with the unobserved quality term.

**Estimation Results: True vs OLS vs IV/2SLS**

| Parameter   |   True |    OLS |   IV/2SLS |   IV s.e. |
|:------------|-------:|-------:|----------:|----------:|
| alpha       |    1.5 |  1.009 |     1.465 |     0.058 |
| beta_sugar  |    0.3 |  0.376 |     0.409 |     0.022 |
| beta_fiber  |    0.5 |  0.519 |     0.63  |     0.027 |
| beta_const  |    1   | -0.5   |    -0.1   |     0.23  |

## Takeaway

The supply-side object is not observed cost; it is the marginal cost that rationalizes observed prices under the estimated demand system and an ownership matrix. The demand estimate is therefore consequential: attenuating price sensitivity also distorts markups and recovered costs. Simple logit is a useful benchmark because Berry inversion and the markup equation are transparent, but its IIA substitution pattern is too rigid for many merger and product-space applications. Random-coefficients demand in [BLP](../blp-random-coefficients/) lets substitution vary with consumer heterogeneity and product characteristics.

## References

- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics* 25(2), 242-262.
- Berry, S., Levinsohn, J., and Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica* 63(4), 841-890.
- Nevo, A. (2001). "Measuring Market Power in the Ready-to-Eat Cereal Industry." *Econometrica* 69(2), 307-342.
- Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition, Ch. 3.
