# MCMC Chain Diagnostics: ESS, R-hat, and Integrated Autocorrelation Time

## Overview

A chain can look converged on a trace plot and still be reporting the average of one slice of the posterior. Diagnostics decide whether to trust the averages.

Three random-walk Metropolis-Hastings chains run on a correlated bivariate Gaussian target. The integrated autocorrelation time and effective sample size measure how much information the chain has carried. The classical Gelman-Rubin R-hat and its rank-normalised split variant ask whether independent chains have agreed.

Two step sizes bracket the Roberts-Gelman-Gilks acceptance band. R-hat separates them without looking at the trace.

The sampler lives in [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/). The same R-hat column governs convergence in [`structural-econometrics/bayesian-dsge-hmc/`](../../structural-econometrics/bayesian-dsge-hmc/), where any chain above 1.01 is unconverged.

## Preliminary readings

- [`bayesian-methods/bayesian-foundations/`](../../bayesian-methods/bayesian-foundations/)
- [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/)

## Equations

Let $`\theta`$ denote a single coordinate of the chain. Let $`(\theta_1, \ldots, \theta_T)`$ be the post-burn-in draws. Assume stationarity: any joint distribution depends only on lag, not time index.

The lag-$`t`$ autocorrelation is the correlation between draws $`t`$ steps apart:

```math
\rho_t = \mathrm{Corr}(\theta_s,  \theta_{s + t}).
```

Independent draws give $`\rho_t = 0`$ for $`t \ge 1`$. A slow-mixing chain has $`\rho_t`$ close to one for many lags.

The integrated autocorrelation time sums all positive lags:

```math
\tau = 1 + 2 \sum_{t \ge 1} \rho_t.
```

A chain of $`\tau`$ correlated draws carries the information of one independent draw. Empirical autocorrelations grow noisy at large lags, so the sum is truncated. Geyer's (1992) monotone-positive estimator pairs adjacent lags into sums $`\rho_{2i + 1} + \rho_{2i + 2}`$, walks forward until the first nonpositive pair, then enforces a running minimum on the survivors.

The effective sample size inverts the inflation factor:

```math
\mathrm{ESS} = \frac{T}{\tau}.
```

The Gelman-Rubin diagnostic compares within-chain to between-chain variance. Run $`M`$ chains of equal length $`T`$ from dispersed starts. Let $`\bar\theta_{m}`$ and $`s_m^2`$ denote chain $`m`$'s sample mean and variance. The within-chain variance is

```math
W = \frac{1}{M} \sum_{m = 1}^{M} s_m^2.
```

The between-chain variance is

```math
B = \frac{T}{M - 1} \sum_{m = 1}^{M} (\bar\theta_m - \bar{\bar\theta})^2,
```

where $`\bar{\bar\theta}`$ is the grand mean. The classical Gelman-Rubin potential scale reduction factor is

```math
\hat R = \sqrt{\frac{T - 1}{T} + \frac{B}{T  W}}.
```

At $`\hat R = 1`$ within-chain and between-chain variance estimate the same posterior variance. Values above 1.01 mean the chains have not agreed.

Vehtari et al. (2021) sharpen the diagnostic two ways. Splitting each chain in half lets within-chain variance pick up drift between halves. Rank-normalising the pooled draws makes $`\hat R`$ robust to heavy tails. Let $`r_{m, t}`$ be the rank of $`\theta_{m, t}`$ in the pooled sample and let

```math
z_{m, t} = \Phi^{-1}\left(\frac{r_{m, t} - 3/8}{M T + 1/4}\right),
```

where $`\Phi^{-1}`$ is the standard normal quantile. Apply classical $`\hat R`$ to the half-split $`z`$ values. This is the default in modern probabilistic-programming libraries.

## Model Setup

| Object | Symbol | Role |
|---|---|---|
| Parameter coordinate | $`\theta`$ | One component of the chain [from `metropolis-hastings`] |
| Chain length | $`T`$ | Post-burn-in draws [from `metropolis-hastings`] |
| Number of chains | $`M`$ | Independent chains from dispersed starts |
| Lag autocorrelation | $`\rho_t`$ | Correlation between draws $`t`$ steps apart [prelim introduces this; dense tutorials adopt on cut] |
| Integrated autocorrelation time | $`\tau`$ | $`1 + 2 \sum_{t \ge 1} \rho_t`$ [prelim introduces this; dense tutorials adopt on cut] |
| Effective sample size | $`\mathrm{ESS}`$ | $`T / \tau`$ [prelim introduces this; dense tutorials adopt on cut] |
| Within-chain variance | $`W`$ | Mean of per-chain sample variances [from Gelman-Rubin] |
| Between-chain variance | $`B`$ | $`T`$ times variance of chain means [from Gelman-Rubin] |
| Potential scale reduction | $`\hat R`$ | $`\sqrt{(T - 1)/T + B / (T W)}`$ [from Gelman-Rubin; convention shared with `bayesian-dsge-hmc`] |
| Target correlation | $`\rho`$ | 0.95 between $`\theta_1`$ and $`\theta_2`$ |
| Proposal step (tiny) | $`s_{\mathrm{tiny}}`$ | 0.05, deliberately too small |
| Proposal step (optimal) | $`s_{\mathrm{opt}}`$ | 0.9, near the Roberts-Gelman-Gilks acceptance band |
| Chain draws | 8,000 each | First 1,000 discarded as burn-in |
| Starts | $`\{(3, -3), (-3, 3), (0, 0)\}`$ | Dispersed initial points |

## Solution Method

Three diagnostics run on the same chains.

### Integrated autocorrelation time and effective sample size

```text
Algorithm: Geyer monotone-positive IAT
Input : a scalar chain coordinate of length T, maximum lag L
Output: tau and the truncation cutoff
  acf <- sample autocorrelation at lags 0, 1, ..., L
  pair_sums <- (acf[2i+1] + acf[2i+2]) for i = 0, 1, ...
  truncate pair_sums at the first nonpositive entry
  enforce a running minimum across the surviving pair sums
  tau <- 1 + 2 * sum of the resulting monotone sequence
  ESS <- T / tau
```

For a reversible Markov chain, adjacent lag pairs have positive autocovariance even when individual lags fluctuate around zero. The monotone projection stabilises the tail. The cutoff is reported with the IAT.

### Classical Gelman-Rubin R-hat

```text
Algorithm: classical R-hat across M chains
Input : an M-by-T array of one coordinate of M independent chains
Output: R-hat
  per-chain mean and sample variance
  W <- mean of per-chain variances
  B <- T * variance of chain means (sample variance, ddof = 1)
  var_hat <- (T - 1)/T * W + B / T
  R-hat <- sqrt(var_hat / W)
```

Three chains run from $`(3, -3)`$, $`(-3, 3)`$, and $`(0, 0)`$. Dispersed starts let the diagnostic detect chains stuck in different posterior regions. After burn-in the chains should overlap so $`B`$ stays small relative to $`W`$.

### Rank-normalised split-R-hat

```text
Algorithm: rank-normalised split-R-hat
Input : an M-by-T array
Output: R-hat
  split each chain in half to give a 2M-by-(T/2) array
  pool the draws, compute ranks
  z <- normal quantile of (rank - 3/8) / (total + 1/4)
  return classical R-hat on the z array
```

Splitting catches drift between halves. Rank normalisation handles heavy tails. Vehtari treats any chain with split-rank-$`\hat R`$ above $`1.01`$ as unconverged.

The Roberts-Gelman-Gilks asymptotic acceptance optimum for random-walk Metropolis is $`0.234`$ as $`d \to \infty`$. On this $`d = 2`$ target the practical band is $`0.25`$ to $`0.40`$. The two step sizes bracket it.

## Results

The trace plots show the tuning story. Under the tiny step the three chains barely leave their starts; each explores a different slice of the posterior over 8000 draws and the coloured paths never overlap. Under the near-optimal step the chains mix within a few hundred draws and then overlap. Acceptance rates are 0.92 for the tiny step and 0.26 for the near-optimal step, inside the asymptotic band.

<img src="figures/trace-plots.png" alt="Trace plots: tiny step versus near-optimal step across three chains" width="90%">

The autocorrelation decay panel quantifies the traces. Under the tiny step, lag-200 autocorrelation stays above 0.6 and Geyer's truncation pushes deep into the tail. Under the near-optimal step, autocorrelation crosses zero before lag 100 and the cutoff lands an order of magnitude earlier. The IAT falls from roughly 540 to 40; for 7000 retained draws ESS rises from about 13 to 170.

<img src="figures/autocorrelation-decay.png" alt="Autocorrelation decay with Geyer monotone-positive truncation under each step size" width="90%">

Only one step size passes $`\hat R`$. Under the near-optimal step both classical and rank-normalised diagnostics fall below 1.01 after a few thousand draws. Under the tiny step both stay near 1.8 because the chains still explore different posterior slices. The dotted line at 1.01 is Vehtari's threshold; anything above it should be re-run with more draws or a different proposal. Trace plots alone cannot tell the two apart.

<img src="figures/r-hat-trajectory.png" alt="R-hat trajectory versus chain length under the tiny and near-optimal step sizes" width="80%">

## Takeaway

Three diagnostics catch three failures. IAT and ESS flag chains that look fine but carry few independent draws. Classical $`\hat R`$ flags chains that have not agreed. Rank-normalised split-$`\hat R`$ sharpens the test and handles heavy tails. Trace plots are the cheapest visual cross-check. Any structural posterior with ESS below a few hundred or split-$`\hat R`$ above $`1.01`$ should be re-tuned and re-run.

These diagnostics are gradient-free. Hamiltonian Monte Carlo adds sampler-specific divergence counts, introduced in [`computational-methods/hamiltonian-monte-carlo/`](../../computational-methods/hamiltonian-monte-carlo/) and reported in the posterior summary of [`structural-econometrics/bayesian-dsge-hmc/`](../../structural-econometrics/bayesian-dsge-hmc/).

## References

- Gelman, A. and Rubin, D. B. (1992). "Inference from Iterative Simulation Using Multiple Sequences." *Statistical Science*, 7(4), 457-472. Sections 2-3 define the potential scale reduction factor.
- Geyer, C. J. (1992). "Practical Markov Chain Monte Carlo." *Statistical Science*, 7(4), 473-483. Sections 2-3 derive the IAT from the Kipnis-Varadhan central limit theorem and connect it to the effective sample size.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. (2021). "Rank-Normalization, Folding, and Localization: An Improved $`\hat R`$ for Assessing Convergence of MCMC." *Bayesian Analysis*, 16(2), 667-718. Sections 2-3 present the rank-normalised split-$`\hat R`$ used as the default modern threshold.
- Robert, C. P. and Casella, G. (2004). *Monte Carlo Statistical Methods*, 2nd edition. Springer. Chapter 12 on convergence diagnostics.
- **See also.** The sampler diagnosed here is [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/). The gradient-based alternative is [`computational-methods/hamiltonian-monte-carlo/`](../../computational-methods/hamiltonian-monte-carlo/), which adds divergence counts. The same R-hat column governs convergence in [`structural-econometrics/bayesian-dsge-hmc/`](../../structural-econometrics/bayesian-dsge-hmc/). The conjugate posteriors that motivate sampling are in [`bayesian-methods/bayesian-foundations/`](../../bayesian-methods/bayesian-foundations/).
