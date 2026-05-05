# Plain Logit Demand and IIA

> Estimate a baseline product-choice model and read the substitution restriction it imposes.

## Overview

A product-choice model is a disciplined way to turn prices, characteristics, and observed purchases into demand primitives. The market here is small: five products differ only in price and quality, and each consumer buys the product with the highest utility draw.

Plain logit is the benchmark because the Type I extreme-value taste shock gives closed-form choice probabilities. That convenience also gives the model its sharp economic restriction: substitution depends on existing market shares, not on product similarity. The tutorial estimates the coefficients, compares the fitted shares with the known data-generating model, and then makes the Independence of Irrelevant Alternatives (IIA) visible.

## Equations

Consumers $i=1,\ldots,N$ choose one alternative $j\in\{1,\ldots,J\}$.
Product $j$ has price $p_j$ and quality $q_j$.

**Utility:**
$$U_{ij}=V_j+\varepsilon_{ij}, \qquad
V_j=\beta_p p_j+\beta_q q_j,$$

with $\varepsilon_{ij}$ i.i.d. Type I extreme value. The expected sign is
$\beta_p<0$ and $\beta_q>0$.

**Choice probability:**
$$P_j(\beta)=\Pr(y_i=j\mid p,q;\beta)
=\frac{\exp(V_j)}{\sum_{k=1}^J \exp(V_k)}.$$

If $d_{ij}=1\{y_i=j\}$, the sample log-likelihood is
$$\ell(\beta)=\sum_{i=1}^N\sum_{j=1}^J d_{ij}\log P_j(\beta).$$

Because this one-market example has no individual covariates, fitted market
shares are just $s_j=P_j(\hat\beta)$. The implied price elasticities are
$$\eta_{jj}=\beta_p p_j(1-s_j), \qquad
\eta_{jk}=-\beta_p p_k s_k \quad (j\neq k).$$

IIA follows from the odds ratio
$$\frac{P_j}{P_k}=\exp(V_j-V_k),$$
which does not depend on any third product.

## Model Setup

The sample is synthetic, so the true coefficients and population logit shares are available for comparison. That makes the exercise about the estimator and the model restriction, not about data cleaning.

| Object | Value | Role |
|-----------|-------|-------------|
| Consumers | 5000 | Independent choice draws |
| Products | 5 | Fixed alternatives in one market |
| Prices | 2.0, 3.5, 5.0, 7.0, 10.0 | Utility shifter with negative coefficient |
| Quality | 1.0, 2.0, 3.5, 4.0, 5.0 | Utility shifter with positive coefficient |
| True $\beta_p$ | -0.5 | Price coefficient used to simulate choices |
| True $\beta_q$ | 1.2 | Quality coefficient used to simulate choices |

## Solution Method

The estimator chooses the coefficients that make the realized product choices most likely. The code minimizes the negative log-likelihood, but the economic object is the same maximizer:

$$\hat\beta=\arg\max_\beta \ell(\beta).$$

```text
Inputs: prices p_j, qualities q_j, choices y_i, starting value beta^(0)
Repeat inside the optimizer:
    1. Form V_j(beta) = beta_p p_j + beta_q q_j for every product j.
    2. Convert V into logit probabilities P_j(beta).
    3. Evaluate ell(beta) = sum_i log P_{y_i}(beta).
Choose beta_hat that maximizes ell(beta).
At beta_hat: compute fitted shares, elasticities, and IIA share ratios.
```

For this plain logit the likelihood is globally concave, so the two-parameter surface has a single peak. BFGS converged in **9 iterations** with log-likelihood **-7493.95**. The inverse Hessian approximation supplies the standard errors reported below.

## Results

The contour plot shows the whole estimation problem in two dimensions. The likelihood has one peak, and the MLE sits close to the true coefficients used to generate the data. Sampling noise keeps the estimate from landing exactly on the star, but the gap is small at 5,000 choices.

<img src="figures/log-likelihood-surface.png" alt="Log-likelihood surface with true and estimated coefficients marked" width="80%">

Observed shares are finite-sample frequencies; the green bars are the population shares from the data-generating logit. The fitted shares mostly sit between those two objects, which is what correct specification predicts.

<img src="figures/market-shares.png" alt="Observed, fitted, and true logit market shares" width="80%">

The own-price elasticities combine the estimated price coefficient with each product's price and share. The expensive products have larger elasticities in absolute value, so a one percent price increase costs them a larger fraction of demand.

<img src="figures/own-price-elasticities.png" alt="Own-price elasticities implied by the estimated logit" width="80%">

When Product 3 is removed, the remaining products do not become closer or farther apart in the model. Their probabilities are renormalized, so the pairwise odds ratios in the right panel stay fixed. This is the IIA restriction that the nested-logit tutorial relaxes by grouping similar products.

<img src="figures/iia-illustration.png" alt="IIA reallocation after removing one alternative" width="80%">

The signs are economically sensible and match the simulation: consumers dislike price and value quality. The estimates are close to the true coefficients because the likelihood is correctly specified and the sample is large.

**MLE estimates and true coefficients**

| Parameter   |   True |   Estimate |   Std. error |   t-stat | p-value   |
|:------------|-------:|-----------:|-------------:|---------:|:----------|
| beta_p      |   -0.5 |    -0.4913 |       0.0174 |   -28.21 | <0.001    |
| beta_q      |    1.2 |     1.1559 |       0.0362 |    31.94 | <0.001    |

Rows are the products whose shares change; columns are the products whose prices change. In every off-diagonal column the cross-elasticities are identical, so the model sends lost buyers to rivals in proportion to their shares rather than to the most similar product.

**Price elasticity matrix**

| Product   |   Product 1 |   Product 2 |   Product 3 |   Product 4 |   Product 5 |
|:----------|------------:|------------:|------------:|------------:|------------:|
| Product 1 |      -0.896 |        0.23 |       0.889 |       0.83  |       0.863 |
| Product 2 |       0.086 |       -1.49 |       0.889 |       0.83  |       0.863 |
| Product 3 |       0.086 |        0.23 |      -1.568 |       0.83  |       0.863 |
| Product 4 |       0.086 |        0.23 |       0.889 |      -2.609 |       0.863 |
| Product 5 |       0.086 |        0.23 |       0.889 |       0.83  |      -4.051 |

## Takeaway

Plain logit is a useful first demand model because the mapping from utility coefficients to shares is transparent and the likelihood is easy to optimize. The same structure makes its substitution patterns too rigid: once shares are known, cross-price responses are pinned down by IIA rather than by economic similarity. Nested logit and mixed logit are the natural next models, not cosmetic complications.

## References

- McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice Behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics*. Academic Press.
- Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition.
