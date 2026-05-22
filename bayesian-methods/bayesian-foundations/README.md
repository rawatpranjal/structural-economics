# Bayesian Foundations: Priors, Likelihoods, and Conjugate Posteriors

## Overview

Bayesian inference replaces a single estimate with a full posterior distribution over the unknown parameter. The posterior is built from two pieces. The prior says what the analyst believes before seeing the data. The likelihood says how the data depend on the parameter. Bayes rule combines them into a posterior that updates the prior toward the data without throwing the prior away.

This tutorial is the entry point for the bayesian-methods view of the catalog. It teaches the rule on two cases where the posterior is available in closed form. The Beta-Binomial gives a scalar conjugate update for a probability of success. The Gaussian-Gaussian conjugate regression gives a vector update for linear-regression coefficients whose posterior mean is the precision-weighted average of the prior mean and the ordinary-least-squares estimate.

A reader who has finished this prelim can read [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/) as an algorithm tutorial about what to do when conjugacy fails. The Gaussian-Gaussian derivation that ends this prelim is the same update that drives the Minnesota-prior BVAR in [`time-series/minnesota-svar/`](../../time-series/minnesota-svar/). The function-space version is the Gaussian-process posterior in [`numerical-methods/bayesian-optimization/`](../../numerical-methods/bayesian-optimization/).

Three runnable examples earn the lesson. Posterior contraction under three priors as the sample grows. Posterior bands on a regression line shrinking with sample size. A prior-sensitivity sweep that interpolates from a prior-dominated posterior to a data-dominated one.

## Equations

Let $`\theta`$ denote the unknown parameter and let $`y`$ denote the observed data.
Bayes rule combines a prior density $`p(\theta)`$ with a likelihood $`p(y \mid \theta)`$ into a posterior density:

```math
p(\theta \mid y) = \frac{p(y \mid \theta)  p(\theta)}{\int p(y \mid \theta')  p(\theta')  d\theta'}
\propto p(y \mid \theta)  p(\theta).
```

The integral in the denominator is the marginal likelihood. It is a constant in $`\theta`$, so the posterior is proportional to the prior-times-likelihood kernel. Closed-form posteriors are available when the prior is conjugate to the likelihood, meaning prior and posterior live in the same parametric family.

### Beta-Binomial conjugate update

The Beta-Binomial model has scalar parameter $`\theta \in (0, 1)`$ interpreted as a probability of success. The prior is a Beta distribution with shape parameters $`\alpha > 0`$ and $`\beta > 0`$:

```math
\theta \sim \mathrm{Beta}(\alpha, \beta),
\qquad
p(\theta) \propto \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}.
```

The data are $`n`$ independent Bernoulli trials with $`s`$ successes. The likelihood is the binomial mass function and depends on $`\theta`$ only through $`\theta^s (1 - \theta)^{n - s}`$.

Multiplying prior and likelihood and collecting $`\theta`$-dependent factors gives the posterior kernel:

```math
p(\theta \mid y) \propto \theta^{\alpha - 1} (1 - \theta)^{\beta - 1} \cdot \theta^s (1 - \theta)^{n - s}
= \theta^{\alpha + s - 1} (1 - \theta)^{\beta + n - s - 1}.
```

The kernel has Beta form, so the posterior is itself Beta with updated shape parameters:

```math
\theta \mid y \sim \mathrm{Beta}(\alpha + s,  \beta + n - s).
```

The posterior mean is a convex combination of the prior mean and the sample fraction:

```math
\mathbb{E}[\theta \mid y] = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \frac{\alpha}{\alpha + \beta} + \frac{n}{\alpha + \beta + n} \cdot \frac{s}{n}.
```

The prior weight $`(\alpha + \beta) / (\alpha + \beta + n)`$ shrinks to zero as $`n`$ grows. A reader with a flat prior and a large dataset reports essentially the sample fraction.

### Gaussian-Gaussian conjugate linear regression

The second model is a linear regression with a known residual variance $`\sigma^2`$. The data are $`y \in \mathbb{R}^n`$ and a design matrix $`X \in \mathbb{R}^{n \times p}`$ with coefficient vector $`\beta \in \mathbb{R}^p`$. The likelihood is

```math
p(y \mid \beta) \propto \exp\left[-\frac{1}{2 \sigma^2} (y - X \beta)^{\top} (y - X \beta)\right].
```

The conjugate prior is a multivariate Gaussian with mean vector $`b_0 \in \mathbb{R}^p`$ and prior precision matrix $`V_0^{-1}`$:

```math
\beta \sim \mathcal{N}(b_0,  V_0).
```

Completing the square in the prior-times-likelihood exponent shows the posterior is Gaussian with precision and mean

```math
V^{-1} = V_0^{-1} + \frac{X^{\top} X}{\sigma^2},
\qquad
b = V \left[V_0^{-1} b_0 + \frac{X^{\top} y}{\sigma^2}\right].
```

The posterior precision is the sum of prior precision and data precision. The posterior mean is the precision-weighted average of the prior mean and the data-only ordinary-least-squares estimate. Write $`\hat\beta_{\mathrm{OLS}} = (X^{\top} X)^{-1} X^{\top} y`$ to make the reading explicit:

```math
b = V \left[V_0^{-1} b_0 + \frac{X^{\top} X}{\sigma^2} \hat\beta_{\mathrm{OLS}}\right].
```

As $`n`$ grows the data-precision matrix $`X^{\top} X / \sigma^2`$ dominates $`V_0^{-1}`$ and the posterior mean approaches the OLS estimate. With a tight prior or a small sample the prior anchors the posterior.

### Posterior predictive density

The posterior predictive density for a new observation $`\tilde y`$ averages the data-generating density over the posterior:

```math
p(\tilde y \mid y) = \int p(\tilde y \mid \theta)  p(\theta \mid y)  d\theta.
```

For the Beta-Binomial coin-flip model with one new trial the posterior predictive probability of success is the posterior mean:

```math
\Pr(\tilde y = 1 \mid y) = \mathbb{E}[\theta \mid y] = \frac{\alpha + s}{\alpha + \beta + n}.
```

The predictive density carries parameter uncertainty into the forecast. It is wider than the plug-in density that fixes $`\theta`$ at its posterior mean.

## Model Setup

| Object | Symbol | Role |
|---|---|---|
| Parameter | $`\theta`$ | Unknown probability or coefficient vector [convention shared with `metropolis-hastings`; that tutorial writes $`\theta \mid D`$ where this prelim writes $`\theta \mid y`$, with $`D`$ and $`y`$ both naming the observed data] |
| Data | $`y`$ | Observed outcomes [from `minnesota-svar`] |
| Beta prior shapes | $`\alpha, \beta`$ | Pseudo-counts in the Beta-Binomial prior [from `metropolis-hastings`] |
| Sample size | $`n`$ | Number of Bernoulli trials |
| Successes | $`s`$ | Trial successes [prelim introduces this; the MH refactor renames its $`k`$ to $`s`$ on the cut for alignment] |
| True coin probability | $`p^{\ast}`$ | Data-generating probability in Example A |
| Sample sizes | $`\{0, 10, 100, 1000\}`$ | Posterior snapshots in Example A |
| Three priors | $`\mathrm{Beta}(1, 1)`$, $`\mathrm{Beta}(20, 20)`$, $`\mathrm{Beta}(40, 10)`$ | Uniform, conservative-correct, confident-wrong |
| Regression coefficient | $`\beta`$ | Coefficient vector in Example B |
| Regression noise scale | $`\sigma`$ | Known Gaussian residual standard deviation |
| Prior mean vector | $`b_0`$ | Vector prior mean [from `minnesota-svar` notation $`b_i^0`$] |
| Prior precision | $`V_0^{-1}`$ | Inverse prior covariance [from `minnesota-svar` notation $`(V_i^0)^{-1}`$] |
| Posterior precision | $`V^{-1}`$ | $`V_0^{-1} + X^{\top} X / \sigma^2`$ |
| Posterior mean | $`b`$ | Precision-weighted average of $`b_0`$ and $`\hat\beta_{\mathrm{OLS}}`$ |
| Sample-size grid (B) | $`\{10, 50, 200, 1000\}`$ | Bands shrink with $`n`$ |
| Prior strengths (C) | $`\{0.5, 1, 2, \ldots, 1000\}`$ | Sweep of $`\alpha_0 + \beta_0`$ at a fixed prior mean |

## Solution Method

The three examples share one machinery. Form the prior, write the likelihood, multiply, recognize the family, read off the posterior parameters. No simulation is needed and the answer is exact.

### Beta-Binomial conjugacy

```text
Algorithm: Beta-Binomial conjugate update
Input : prior shapes alpha, beta; data n trials with s successes
Output: posterior shapes and moments
  alpha_post = alpha + s
  beta_post  = beta  + n - s
  mean       = alpha_post / (alpha_post + beta_post)
  variance   = alpha_post * beta_post /
               ((alpha_post + beta_post)^2 * (alpha_post + beta_post + 1))
```

Three priors are run against draws from a coin with true probability $`p^{\ast} = 0.55`$: a flat $`\mathrm{Beta}(1, 1)`$, a conservative symmetric $`\mathrm{Beta}(20, 20)`$, and a confident-wrong $`\mathrm{Beta}(40, 10)`$. The same draws update each prior. Posterior shapes are recorded after 0, 10, 100, and 1000 observations.

### Gaussian-Gaussian conjugate linear regression

```text
Algorithm: Gaussian conjugate regression
Input : design matrix X, response y, noise scale sigma,
        prior mean b0, prior precision V0_inv
Output: posterior mean b and posterior covariance V
  data_precision   = X' X / sigma^2
  posterior_prec   = V0_inv + data_precision
  posterior_cov    = inverse(posterior_prec)
  posterior_rhs    = V0_inv b0 + X' y / sigma^2
  posterior_mean   = posterior_cov * posterior_rhs
```

A synthetic dataset is drawn with intercept and one covariate, true coefficients $`(1.5, -0.8)`$, and Gaussian noise. The weakly informative prior puts mean zero on both coefficients and prior standard deviation 5 on each. The posterior is computed at sample sizes 10, 50, 200, and 1000.

### Prior sensitivity

```text
Algorithm: Beta-Binomial prior-sensitivity sweep
Input : data n trials with s successes; fixed prior mean p0;
        grid of prior strengths kappa in {0.5, 1, ..., 1000}
Output: posterior mean and credible bands as kappa grows
  for each kappa in grid:
    alpha0 = p0 * kappa
    beta0  = (1 - p0) * kappa
    alpha_post = alpha0 + s
    beta_post  = beta0  + n - s
    posterior_mean = alpha_post / (alpha_post + beta_post)
    record 2.5% and 97.5% Beta quantiles
```

Data are fixed at 50 trials with 32 successes. The prior anchor is $`p_0 = 0.3`$, far below the sample fraction $`0.64`$. Sweeping the prior strength $`\kappa = \alpha_0 + \beta_0`$ shows how the posterior mean slides from the data-dominated regime (small $`\kappa`$) to the prior-dominated regime (large $`\kappa`$).

## Results

Posterior contraction is visible across all three priors as the sample size grows. The flat prior peaks early at the noisy 10-observation sample fraction and then converges. The conservative $`\mathrm{Beta}(20, 20)`$ posterior moves more slowly because the prior carries weight equivalent to forty pseudo-observations. The confident-wrong $`\mathrm{Beta}(40, 10)`$ posterior is pinned near $`0.80`$ for the first ten observations and only by $`n = 1000`$ does the data drag the posterior mean to $`0.55`$. Across panels the dashed red line at $`p^{\ast} = 0.55`$ is the contraction target. By $`n = 1000`$ all three posteriors are tight bumps centred near it.

<img src="figures/beta-posteriors.png" alt="Beta posteriors after 0, 10, 100, and 1000 observations under three priors" width="90%">

In the regression panel the four sample sizes are read left to right. At $`n = 10`$ the posterior band over the regression line is wide and the posterior mean is shifted from the truth by sampling noise. By $`n = 50`$ the band has tightened by a factor of two and the posterior mean matches the OLS estimate to three decimals. At $`n = 200`$ and $`n = 1000`$ the band is barely visible and the posterior intercept and slope sit on top of the dashed truth line. The shrinkage is the precision identity in action: the posterior precision grows linearly with $`n`$, so the posterior standard deviation falls as $`1 / \sqrt{n}`$.

<img src="figures/gaussian-regression-bands.png" alt="Gaussian-Gaussian conjugate regression posterior bands across sample sizes" width="90%">

The prior-sensitivity fan separates the data-dominated regime from the prior-dominated regime. With prior strength below the sample size of 50 the posterior mean stays near the sample fraction of $`0.64`$. At prior strength 50 the prior weight equals the data weight and the posterior mean sits halfway between the two anchors. As the prior strength grows the posterior mean drifts toward the prior anchor of $`0.3`$. The 95% credible band shrinks throughout the sweep because the total information $`\alpha + \beta + n`$ keeps growing.

<img src="figures/prior-sensitivity-fan.png" alt="Beta-Binomial prior sensitivity fan over prior strength" width="80%">

The posterior predictive probability of success on the next trial is the posterior mean itself, $`(\alpha + s) / (\alpha + \beta + n)`$. On the confident-wrong prior at $`n = 1000`$ this evaluates to $`0.55`$, identical to the posterior mean and to the analytic Bernoulli predictive. The closed-form match is what makes the Beta-Binomial worth carrying through every later chapter: it is the smallest non-trivial Bayesian model whose posterior predictive can be checked by hand.

## Takeaway

Conjugate Bayes is the right starting point. The Beta-Binomial gives a closed-form scalar update and a closed-form posterior predictive probability. The Gaussian-Gaussian conjugate regression gives a closed-form vector update whose posterior mean is the precision-weighted average of the prior mean and the OLS estimate. Both updates are read off the prior-times-likelihood kernel without simulation. They are the right tool whenever the model permits and the right baseline against which any sampler should be checked.

Most structural posteriors are not conjugate. When the likelihood is non-Gaussian, the parameter space is bounded, or latent variables intervene, the posterior is only known up to a normalizing constant. The algorithmic response is sampling. Random-walk Metropolis-Hastings in [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/) is the gradient-free option. Hamiltonian Monte Carlo in [`computational-methods/hamiltonian-monte-carlo/`](../../computational-methods/hamiltonian-monte-carlo/) is the gradient-based option for curved posteriors. Once a chain is run, the diagnostics in [`computational-methods/mcmc-diagnostics/`](../../computational-methods/mcmc-diagnostics/) tell the reader whether to trust its averages.

## References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd edition. Chapman and Hall / CRC. Chapters 1-2 on foundations and single-parameter conjugacy; Chapter 14 on the normal linear regression conjugate update.
- Koop, G. (2003). *Bayesian Econometrics*. Wiley, Chapter 2. Field-specific exposition linking conjugacy to econometric practice.
- Robert, C. P. (2007). *The Bayesian Choice*, 2nd edition. Springer, Chapter 3. Decision-theoretic framing for conjugate priors and prior selection.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press, Chapter 12. Bridges scalar Bayes to Bayesian vector autoregressions.
- **See also.** The Beta-Binomial conjugate update derived above is the closed-form baseline the random-walk sampler is checked against in [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/). The Gaussian-Gaussian conjugate regression update is the building block of the Minnesota-prior BVAR in [`time-series/minnesota-svar/`](../../time-series/minnesota-svar/). The chain diagnostics that decide whether to trust an MCMC posterior mean are in [`computational-methods/mcmc-diagnostics/`](../../computational-methods/mcmc-diagnostics/).
