# BLP Random Coefficients Demand

> Differentiated-products demand when substitution depends on consumer tastes.

## Overview

In differentiated-products IO, demand is not only about fitting market shares. The substitution matrix is what turns demand estimates into merger effects, markups, and welfare calculations. The simple logit model in [logit demand and markup recovery](../logit-supply-side/) is useful because Berry inversion is transparent, but it also forces the IIA restriction: when one product changes price, all rivals gain share in proportion to their existing shares.

BLP replaces that representative-consumer substitution pattern with random coefficients. Consumers differ in their taste for the observed characteristic and in price sensitivity, so products that attract similar consumers become closer substitutes. This tutorial uses a synthetic market where the true parameters are known, estimates the nonlinear taste dispersion by GMM, and then compares the implied elasticities with both the true DGP and a plain-logit benchmark.

## Equations

Consumer $i$ in market $t$ chooses among $J$ inside goods and an outside good.
The indirect utility from inside product $j$ is

$$u_{ijt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt} + \sigma_x \nu_{i1} x_{jt} + \sigma_p \nu_{i2} p_{jt} + \varepsilon_{ijt}$$

where $x_{jt}$ is an observed product characteristic, $p_{jt}$ is price,
$\xi_{jt}$ is unobserved quality, $\nu_i \sim N(0,I)$, and
$\varepsilon_{ijt}$ is Type-I extreme value. The outside good has utility
normalized to zero.

It is useful to separate mean utility from the individual-specific part:

$$\delta_{jt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt}, \qquad \mu_{ijt} = \sigma_x \nu_{i1} x_{jt} + \sigma_p \nu_{i2} p_{jt}$$

For a candidate $\sigma=(\sigma_x,\sigma_p)$, simulated market shares are

$$s_{jt} = \frac{1}{ns} \sum_{i=1}^{ns} \frac{\exp(\delta_{jt} + \mu_{ijt})}{1 + \sum_{k=1}^{J} \exp(\delta_{kt} + \mu_{ikt})}$$

The BLP contraction finds the mean utilities that rationalize observed shares:

$$\delta^{(r+1)}_{jt} = \delta^{(r)}_{jt} + \log s^{\text{obs}}_{jt} - \log s^{\text{pred}}_{jt}(\delta^{(r)}, \sigma)$$

Given $\delta(\sigma)$, the linear demand equation is

$$\delta_{jt} = X_{jt}\theta_1 + \xi_{jt}, \qquad X_{jt}=(1,x_{jt},p_{jt})$$

and the identifying moments are $E[Z_{jt}\xi_{jt}]=0$. The instruments include
a cost shifter and sums of rival characteristics, so price can be endogenous
through $\operatorname{Cov}(p_{jt},\xi_{jt}) \ne 0$.

## Model Setup

The data-generating process has 100 independent markets with five products per market. Prices are deliberately correlated with unobserved quality, so the IV step is doing real work rather than decorating an exogenous logit regression.

| Object | Value | Role |
|-----------|-------|-------------|
| $T$ | 100 | Markets |
| $J$ | 5 | Products per market |
| $ns$ | 200 | Simulation draws used for shares |
| $\beta_0$ | 2.0 | Mean inside-good utility |
| $\beta_x$ | 1.5 | Mean taste for $x$ |
| $\alpha$ | -0.8 | Mean price coefficient |
| $\sigma_x$ | 0.8 | Dispersion in taste for $x$ |
| $\sigma_p$ | 0.3 | Dispersion in price sensitivity |

## Solution Method

The estimator is nested fixed point with GMM. The outer problem chooses the taste-dispersion parameters $\sigma=(\sigma_x,\sigma_p)$. The inner problem inverts market shares for the mean utilities $\delta(\sigma)$.

```text
Inputs: observed shares s_obs, characteristics x, prices p, instruments Z, draws nu
Choose trial nonlinear parameters sigma = (sigma_x, sigma_p)
Initialize delta with the simple-logit inversion log(s_jt) - log(s_0t)
Repeat until the share residual is small:
    predict shares s_pred(delta, sigma) by averaging over taste draws nu
    update delta <- delta + log(s_obs) - log(s_pred)
Run 2SLS of delta(sigma) on (1, x, p) using Z
Compute xi(sigma) and Q(sigma) = n g(sigma)' W g(sigma), where g = Z' xi / n
Search over sigma and keep the minimizer
Output: sigma_hat, theta_1_hat, xi_hat, elasticities
```

The contraction is the economic inversion: it asks what common product utility must be present for the model to match the observed shares after integrating over consumer heterogeneity. The GMM step then asks whether those recovered unobserved qualities are orthogonal to excluded cost and rival-characteristic instruments.

At the true nonlinear parameters, the contraction converged in **627 iterations** with max $|\delta^{\mathrm{recovered}}-\delta^{\mathrm{true}}|=2.45e-11$. The GMM search used Nelder-Mead after a coarse starting grid and evaluated the objective 46 times.

## Results

The estimated model matches the simulated market shares closely, which is expected because the data come from the same random-coefficients family. The more useful checks are the parameter table and the elasticity comparison: they show whether the estimator recovers the DGP objects that matter for counterfactual IO work.

The share fit sits on the 45-degree line because the BLP contraction forces the model to rationalize observed shares for the chosen nonlinear parameters. This is why a good-looking share plot is not, by itself, evidence that the substitution pattern is right.

<img src="figures/observed-vs-predicted-shares.png" alt="Observed and predicted market shares at estimated parameters." width="80%">

The true-DGP bars are available because this is a simulation. Estimated BLP tracks the product-level pattern, with a maximum own-elasticity error of 0.315 in this market. Plain logit has no consumer-specific price coefficient, so its elasticities mostly inherit price and share differences rather than the composition of buyers.

<img src="figures/own-price-elasticities.png" alt="Own-price elasticities in market 1 under the true DGP, estimated BLP model, and plain logit benchmark." width="80%">

The inner fixed point is slow but stable. On the log scale, the update norm falls almost linearly, which is the practical reason the BLP inversion can be embedded inside an outer GMM search.

<img src="figures/contraction-convergence.png" alt="Convergence of the BLP contraction mapping." width="80%">

The cross-elasticity matrix is the main economic object. In the plain-logit panel, every off-diagonal entry in a column is identical, so a price increase for product k sends the same proportional demand response to each rival. In the BLP panel, off-diagonal entries vary by row because products draw different mixtures of consumer tastes.

<img src="figures/cross-price-elasticity-matrix.png" alt="Cross-price elasticity matrices for estimated BLP and plain logit in market 1." width="80%">

Because the data are synthetic, the parameter table is an actual truth check rather than a loose calibration summary. The nonlinear parameters are harder to pin down than the linear taste and price coefficients, but they are the parameters that move substitution away from IIA.

**Estimated vs True Parameters**

| Parameter                  |   True |   Estimated |
|:---------------------------|-------:|------------:|
| $\beta_0$ (intercept)      |    2   |       1.969 |
| $\beta_x$ (characteristic) |    1.5 |       1.576 |
| $\alpha$ (price)           |   -0.8 |      -0.835 |
| $\sigma_x$ (RC on $x$)     |    0.8 |       0.951 |
| $\sigma_p$ (RC on price)   |    0.3 |       0.196 |

## Takeaway

BLP is valuable because it changes the counterfactual object, not because it adds a more complicated optimizer. The contraction lets each candidate $\sigma$ fit observed shares, while the IV/GMM moments choose the amount of heterogeneity that makes recovered unobserved quality orthogonal to excluded instruments. Once heterogeneity is present, substitution is no longer forced to follow existing shares. That is why the model is the natural next step after simple logit demand, especially before using demand estimates for mergers, markups, or welfare.

## References

- Berry, S., Levinsohn, J., and Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica*, 63(4), 841-890.
- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics*, 25(2), 242-262.
- Nevo, A. (2000). "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand." *Journal of Economics & Management Strategy*, 9(4), 513-548.
