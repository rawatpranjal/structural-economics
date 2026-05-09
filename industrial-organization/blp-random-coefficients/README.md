# Differentiated-Products Demand with BLP

## Overview

An antitrust analyst needs to know where demand goes after one product raises price. Market shares alone do not answer that question.

The object is differentiated-products demand with heterogeneous consumers. Products are closer substitutes when they attract similar buyers.

The computation estimates those tastes from market shares. A BLP contraction recovers mean utility for each trial dispersion, and IV/GMM chooses dispersion.

## Equations

Consumer $i$ in market $t$ chooses among $J$ inside goods and an outside good.

Think of $x_{jt}$ as a product attribute such as quality, style, or size.

The indirect utility from inside product $j$ is:

$$u_{ijt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt} + \sigma_x \nu_{i1} x_{jt} + \sigma_p \nu_{i2} p_{jt} + \varepsilon_{ijt}$$

Here $x_{jt}$ is an observed product characteristic, $p_{jt}$ is price,
$\xi_{jt}$ is unobserved quality, $\nu_i \sim N(0,I)$, and
$\varepsilon_{ijt}$ is Type-I extreme value.

The outside good has utility
normalized to zero.

Mean utility and individual taste enter separately:

$$\delta_{jt} = \beta_0 + \beta_x x_{jt} + \alpha p_{jt} + \xi_{jt}, \qquad \mu_{ijt} = \sigma_x \nu_{i1} x_{jt} + \sigma_p \nu_{i2} p_{jt}$$

For a candidate $\sigma=(\sigma_x,\sigma_p)$, simulated market shares are:

$$s_{jt} = \frac{1}{ns} \sum_{i=1}^{ns} \frac{\exp(\delta_{jt} + \mu_{ijt})}{1 + \sum_{k=1}^{J} \exp(\delta_{kt} + \mu_{ikt})}$$

The BLP contraction finds the mean utilities that make predicted shares equal
observed shares:

$$\delta^{(r+1)}_{jt} = \delta^{(r)}_{jt} + \log s^{\text{obs}}_{jt} - \log s^{\text{pred}}_{jt}(\delta^{(r)}, \sigma)$$

Given $\delta(\sigma)$, the linear demand equation is:

$$\delta_{jt} = X_{jt}\theta_1 + \xi_{jt}, \qquad X_{jt}=(1,x_{jt},p_{jt})$$

The identifying moments are $E[Z_{jt}\xi_{jt}]=0$. The instruments include
a cost shifter and sums of rival characteristics, so price can be endogenous
through $\mathrm{Cov}(p_{jt},\xi_{jt}) \ne 0$.

## Model Setup

The example has 100 independent markets with five products per market. Each product has an observed characteristic, an unobserved quality draw, a cost shifter, and a price. Price loads on both cost and unobserved quality, so the IV step has an actual endogeneity problem to solve.

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

The estimator is a nested fixed point with GMM. The outer search chooses the taste-dispersion parameters $\sigma=(\sigma_x,\sigma_p)$. For each trial $\sigma$, the inner contraction finds the mean utilities $\delta(\sigma)$ that reproduce the observed shares.

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

The contraction is the share inversion. It asks what common product utility must be present for the model to match observed shares after averaging over consumer heterogeneity.

The GMM step then checks whether the recovered unobserved qualities are orthogonal to excluded cost and rival-characteristic instruments.

At the true nonlinear parameters, the contraction converged in **627 iterations** with max $|\delta^{\mathrm{recovered}}-\delta^{\mathrm{true}}|=2.45e-11$.

The GMM search used Nelder-Mead after a coarse starting grid and evaluated the objective 46 times.

## Results

The estimated model matches the simulated market shares closely. The elasticity comparison is the harder check. It asks whether estimated heterogeneity changes substitution in the right direction.

The share fit lies on the 45-degree line. The contraction makes predicted shares match observed shares at the chosen dispersion. This plot checks the inversion, not the substitution pattern.

<img src="figures/observed-vs-predicted-shares.png" alt="Observed and predicted market shares at estimated parameters." width="80%">

The true-model bars are available because the data are simulated. Estimated BLP follows the product-level pattern. The largest own-elasticity error is 0.315 in this market. Plain logit has no consumer-specific price coefficient.

<img src="figures/own-price-elasticities.png" alt="Own-price elasticities in market 1 under the true model, estimated BLP model, and plain logit benchmark." width="80%">

The inner fixed point is stable. The update norm falls steadily on the log scale, so the inversion can sit inside GMM.

<img src="figures/contraction-convergence.png" alt="Convergence of the BLP contraction mapping." width="80%">

The cross-elasticity matrix is the main economic object. In plain logit, each column has identical off-diagonal entries. A price increase sends proportional demand to each rival. In BLP, off-diagonal entries vary because products attract different buyers.

<img src="figures/cross-price-elasticity-matrix.png" alt="Cross-price elasticity matrices for estimated BLP and plain logit in market 1." width="80%">

The parameter table checks the simulation truth. The nonlinear dispersion estimates are less exact than the linear coefficients. They are also what break IIA.

**Estimated vs True Parameters**

| Parameter                  |   True |   Estimated |
|:---------------------------|-------:|------------:|
| $\beta_0$ (intercept)      |    2   |       1.969 |
| $\beta_x$ (characteristic) |    1.5 |       1.576 |
| $\alpha$ (price)           |   -0.8 |      -0.835 |
| $\sigma_x$ (RC on $x$)     |    0.8 |       0.951 |
| $\sigma_p$ (RC on price)   |    0.3 |       0.196 |

## Takeaway

BLP changes the estimated substitution object. The contraction lets each candidate $\sigma$ fit observed shares. IV/GMM chooses heterogeneity using moments for recovered unobserved quality. With heterogeneity, substitution no longer has to follow existing shares.

## References

- Berry, S., Levinsohn, J., and Pakes, A. (1995). "Automobile Prices in Market Equilibrium." *Econometrica*, 63(4), 841-890.
- Berry, S. (1994). "Estimating Discrete-Choice Models of Product Differentiation." *RAND Journal of Economics*, 25(2), 242-262.
- Nevo, A. (2000). "A Practitioner's Guide to Estimation of Random-Coefficients Logit Models of Demand." *Journal of Economics & Management Strategy*, 9(4), 513-548.
