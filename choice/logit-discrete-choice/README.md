# Product Demand with Plain Logit and IIA

> Estimate product-choice demand by maximum likelihood and read the substitution pattern implied by IIA.

## Overview

Suppose a researcher observes purchases in a product category where five items differ in price and quality. The demand object is practical: recover price and quality tastes, predict market shares, and ask where buyers go when one product leaves the shelf.

Plain logit gives a transparent first demand system for that problem. The Type I extreme-value taste shock turns utility indexes into closed-form choice probabilities, so the code can fit the taste coefficients by maximum likelihood. After estimation, the same probabilities deliver fitted shares, elasticities, and an immediate counterfactual. They also reveal the model's restriction: IIA reallocates lost buyers in proportion to existing shares, even when the remaining products differ in economic closeness.

## Equations

Consumers $i=1,\ldots,N$ choose one product $j\in\{1,\ldots,J\}$.
Product $j$ has price $p_j$ and quality $q_j$. The deterministic part of
utility is common across consumers in this simple market.

**Utility:**
$$U_{ij}=V_j+\varepsilon_{ij}, \qquad
V_j=\beta_p p_j+\beta_q q_j,$$

with $\varepsilon_{ij}$ i.i.d. Type I extreme value. The expected sign is
$\beta_p<0$ and $\beta_q>0$.

**Choice probability:**
$$P_j(\beta)=\Pr(y_i=j\mid p,q;\beta)
=\frac{\exp(V_j)}{\sum_{k=1}^J \exp(V_k)}.$$

If $d_{ij}=1\{y_i=j\}$ records the observed purchase, the sample
log-likelihood is
$$\ell(\beta)=\sum_{i=1}^N\sum_{j=1}^J d_{ij}\log P_j(\beta).$$

Because this one-market example has no individual covariates, fitted market
shares are $s_j=P_j(\hat\beta)$. The implied price elasticities are
$$\eta_{jj}=\beta_p p_j(1-s_j), \qquad
\eta_{jk}=-\beta_p p_k s_k \quad (j\neq k).$$

IIA follows from the odds ratio
$$\frac{P_j}{P_k}=\exp(V_j-V_k),$$
which does not depend on any third product in the choice set.

## Model Setup

The example keeps the market small enough to see the economics of the estimates. Five fixed products trade off price against quality. The sample is synthetic, so the true coefficients and population shares are available for comparison after estimation.

| Object | Value | Role |
|-----------|-------|-------------|
| Consumers | 5000 | Independent choice draws |
| Products | 5 | Fixed alternatives in one market |
| Prices | 2.0, 3.5, 5.0, 7.0, 10.0 | Utility shifter with negative coefficient |
| Quality | 1.0, 2.0, 3.5, 4.0, 5.0 | Utility shifter with positive coefficient |
| True $\beta_p$ | -0.5 | Price coefficient used to simulate choices |
| True $\beta_q$ | 1.2 | Quality coefficient used to simulate choices |

## Solution Method

The likelihood turns the demand model into a two-parameter optimization problem. Each candidate $\beta=(\beta_p,\beta_q)$ implies utilities, utilities imply choice probabilities, and the observed choices score that candidate through the log-likelihood:

$$\hat\beta=\arg\max_\beta \ell(\beta).$$

```text
Inputs: prices p_j, qualities q_j, choices y_i, starting value beta^(0)
For each trial beta proposed by the optimizer:
    1. Form V_j(beta) = beta_p p_j + beta_q q_j for every product j.
    2. Convert V into logit probabilities P_j(beta).
    3. Evaluate ell(beta) = sum_i log P_{y_i}(beta).
Choose beta_hat that maximizes ell(beta).
At beta_hat: compute fitted shares, elasticities, and IIA share ratios.
```

The code evaluates probabilities with a stable softmax calculation, which subtracts the largest utility before exponentiating and leaves odds ratios unchanged. For this plain logit the likelihood is globally concave, so the two-parameter surface has a single peak. BFGS converged in **9 iterations** with log-likelihood **-7493.95**. The inverse Hessian approximation supplies the standard errors reported below.

## Results

The contour plot shows the estimation problem in two dimensions. The likelihood has one peak, and the MLE sits close to the coefficients used to generate the data. Sampling noise keeps the estimate from landing exactly on the star, but the gap is small with 5,000 choices.

<img src="figures/log-likelihood-surface.png" alt="Log-likelihood surface with true and estimated coefficients marked" width="80%">

Observed shares are finite-sample purchase frequencies. The green bars are the population shares from the data-generating logit, and the fitted shares mostly sit between the realized sample and that population target.

<img src="figures/market-shares.png" alt="Observed, fitted, and true logit market shares" width="80%">

The own-price elasticities combine the estimated price coefficient with each product's price and fitted share. Higher-priced products are more elastic in absolute value here, so a one percent price increase costs them a larger fraction of demand.

<img src="figures/own-price-elasticities.png" alt="Own-price elasticities implied by the estimated logit" width="80%">

When Product 3 is removed, the remaining products do not become closer or farther apart in the model. Their probabilities are renormalized, so the pairwise odds ratios in the right panel stay fixed. This is the IIA restriction that nested logit relaxes by grouping similar products.

<img src="figures/iia-illustration.png" alt="IIA reallocation after removing one alternative" width="80%">

The estimated signs match the simulation: consumers dislike price and value quality. The estimates are close to the true coefficients because the likelihood is correctly specified and the sample is large.

**MLE estimates and true coefficients**

| Parameter   |   True |   Estimate |   Std. error |   t-stat | p-value   |
|:------------|-------:|-----------:|-------------:|---------:|:----------|
| beta_p      |   -0.5 |    -0.4913 |       0.0174 |   -28.21 | <0.001    |
| beta_q      |    1.2 |     1.1559 |       0.0362 |    31.94 | <0.001    |

Rows are the products whose shares change; columns are the products whose prices change. In every off-diagonal column the cross-elasticities are identical, so the model sends lost buyers to rivals in proportion to their shares rather than to the closest product.

**Price elasticity matrix**

| Product   |   Product 1 |   Product 2 |   Product 3 |   Product 4 |   Product 5 |
|:----------|------------:|------------:|------------:|------------:|------------:|
| Product 1 |      -0.896 |        0.23 |       0.889 |       0.83  |       0.863 |
| Product 2 |       0.086 |       -1.49 |       0.889 |       0.83  |       0.863 |
| Product 3 |       0.086 |        0.23 |      -1.568 |       0.83  |       0.863 |
| Product 4 |       0.086 |        0.23 |       0.889 |      -2.609 |       0.863 |
| Product 5 |       0.086 |        0.23 |       0.889 |       0.83  |      -4.051 |

## Takeaway

Plain logit gives an executable first pass at product demand: utility coefficients map directly into shares, elasticities, and removal counterfactuals. Its cost is rigid substitution. Once shares are known, IIA pins down cross-price responses without using product closeness. A richer demand model has to change the taste-shock structure, through nests or random coefficients, before that economics can appear.

## References

- McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice Behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics*. Academic Press.
- Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition.
