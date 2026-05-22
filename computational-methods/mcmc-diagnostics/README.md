# MCMC Chain Diagnostics: ESS, R-hat, and Integrated Autocorrelation Time

## Overview

A Markov-chain sampler returns a sequence of correlated draws. Posterior averages computed from those draws look like simple sample means, but each draw carries less information than an independent draw and consecutive draws may not have explored the same region of the posterior. A chain that looks reasonable on a trace plot can still be reporting the average of one slice of the posterior. Diagnostics decide whether to trust the averages.

This tutorial runs three random-walk Metropolis-Hastings chains on a correlated bivariate Gaussian target and demonstrates the three diagnostics that catch different pathologies. The integrated autocorrelation time and the effective sample size measure how much information the chain has actually carried. The classical Gelman-Rubin R-hat and its modern rank-normalised split variant compare across chains and ask whether they have agreed on the posterior. Trace plots show the same story visually.

The two step-size settings expose the diagnostics. A step that is too small produces a chain whose autocorrelation barely decays over hundreds of lags. A step near the Roberts-Gelman-Gilks optimum mixes well across the ridge. R-hat tells the reader which is which without looking at the trace.

The algorithm that produces the chains lives in [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/). The same R-hat column appears in the posterior summary of [`structural-econometrics/bayesian-dsge-hmc/`](../../structural-econometrics/bayesian-dsge-hmc/), where any chain whose R-hat exceeds 1.01 is considered unconverged.

## Preliminary readings

- [`bayesian-methods/bayesian-foundations/`](../../bayesian-methods/bayesian-foundations/)
- [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/)

## Equations

Let $`\theta`$ denote a single coordinate of the chain. Let $`(\theta_1, \theta_2, \ldots, \theta_T)`$ denote the post-burn-in draws of length $`T`$. Assume the chain is at stationarity, so that any joint distribution depends only on the lag, not on the time index.

The lag-$`t`$ autocorrelation is the correlation between draws that are $`t`$ steps apart:

```math
\rho_t = \mathrm{Corr}(\theta_s,  \theta_{s + t}).
```

Independent draws would give $`\rho_t = 0`$ for $`t \ge 1`$. A slow-mixing chain has $`\rho_t`$ close to one for many lags.

The integrated autocorrelation time aggregates the autocorrelation across all positive lags:

```math
\tau = 1 + 2 \sum_{t \ge 1} \rho_t.
```

The interpretation is that $`\tau`$ draws of the chain carry the same information as one independent draw. The series in practice has to be truncated because empirical autocorrelations become noisy at large lags. This tutorial uses Geyer's (1992) initial monotone-positive estimator: pair adjacent lags into sums $`\rho_{2i + 1} + \rho_{2i + 2}`$, walk forward until the first nonpositive pair, then enforce a running minimum on the surviving pair sums.

The effective sample size is the inverse of the inflation factor:

```math
\mathrm{ESS} = \frac{T}{\tau}.
```

The Gelman-Rubin diagnostic compares the variance within chains to the variance between chains. Run $`M`$ independent chains of equal length $`T`$ from dispersed starts. Let $`\bar\theta_{m}`$ denote the sample mean of chain $`m`$ and let $`s_m^2`$ denote its sample variance. The within-chain variance is

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

A value of $`\hat R = 1`$ would say that within-chain and between-chain variance estimate the same posterior variance. Values above 1.01 say the chains have not yet agreed.

The rank-normalised split-$`\hat R`$ of Vehtari et al. (2021) sharpens the diagnostic in two ways. Each chain is split in half so that within-chain variance includes the chain's autocorrelation between halves. The pooled draws are then transformed by rank-normalisation so that $`\hat R`$ is not sensitive to heavy tails. Let $`r_{m, t}`$ denote the rank of draw $`\theta_{m, t}`$ in the pooled sample and let

```math
z_{m, t} = \Phi^{-1}\left(\frac{r_{m, t} - 3/8}{M T + 1/4}\right),
```

where $`\Phi^{-1}`$ is the standard normal quantile. Compute classical $`\hat R`$ on the half-split rank-normalised $`z`$ values. This is the default diagnostic in modern probabilistic-programming libraries.

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

Three diagnostics are computed in sequence on the same chains.

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

Geyer's pairing exploits the fact that the autocovariance of a reversible Markov chain is positive in adjacent pairs even when individual lags fluctuate around zero. The monotone projection stabilises the tail. The cutoff is reported alongside the IAT so the reader can see how many lags the estimator actually used.

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

Three chains are run from $`(3, -3)`$, $`(-3, 3)`$, and $`(0, 0)`$. The dispersed starts let the diagnostic detect chains stuck in different parts of the posterior. After burn-in the chains should overlap so that $`B`$ stays small relative to $`W`$.

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

Splitting in half makes the within-chain variance include drift between the chain's first and second halves, which catches slow-moving non-stationary chains. Rank normalisation makes the diagnostic robust to heavy tails. The Vehtari recommendation is to treat the chain as unconverged whenever split-rank-$`\hat R`$ exceeds $`1.01`$.

The Roberts-Gelman-Gilks asymptotic optimum for random-walk Metropolis on a $`d`$-dimensional Gaussian is an acceptance rate of about $`0.234`$ (the limit as $`d \to \infty`$; on this $`d = 2`$ target the practical band is closer to $`0.25\text{-}0.40`$). The two step sizes here bracket that band on either side so the diagnostics can separate them.

## Results

The trace plots show the tuning story directly. Under the tiny step the three chains start at their dispersed initial points and barely move. Each chain explores a different slice of the posterior over 8000 draws. The three coloured paths in the left column do not visit the same region at all. Under the near-optimal step the three chains mix into the posterior within a few hundred draws and from then on overlap. The acceptance rates printed in the legend are 0.92 for the tiny step (almost every proposal is taken because the step is small) and 0.26 for the near-optimal step (within the asymptotic acceptance band).

<img src="figures/trace-plots.png" alt="Trace plots: tiny step versus near-optimal step across three chains" width="90%">

The autocorrelation decay panel quantifies what the traces show. Under the tiny step the autocorrelation at lag 200 is still above 0.6 and Geyer's truncation pushes deep into the tail. Under the near-optimal step the autocorrelation crosses zero before lag 100 and the Geyer cutoff lands an order of magnitude earlier. The integrated autocorrelation time falls from roughly 540 to roughly 40, so the effective sample size for the same 7000 retained draws rises from about 13 to about 170.

<img src="figures/autocorrelation-decay.png" alt="Autocorrelation decay with Geyer monotone-positive truncation under each step size" width="90%">

The $`\hat R`$ trajectory makes the cross-chain story explicit. Under the near-optimal step the classical and rank-normalised diagnostics both fall below 1.01 once the chains have run a few thousand draws, certifying mixing. Under the tiny step both diagnostics stay near 1.8 for the whole run because the three chains are still exploring different posterior slices. The black dotted line at 1.01 is the Vehtari threshold; anything above it should be re-run with more draws or a different proposal. Both step sizes have identical acceptance-rate signs of life on a trace plot, but only one passes $`\hat R`$.

<img src="figures/r-hat-trajectory.png" alt="R-hat trajectory versus chain length under the tiny and near-optimal step sizes" width="80%">

## Takeaway

Three diagnostics catch three different failures of an MCMC sampler. The integrated autocorrelation time and the effective sample size detect chains that look fine but carry few independent draws. The classical Gelman-Rubin $`\hat R`$ detects chains that have not agreed on the posterior. The rank-normalised split-$`\hat R`$ sharpens the diagnostic and is robust to heavy tails. Trace plots are the cheapest visual cross-check on all three. Any structural posterior reported with an effective sample size below a few hundred or a rank-normalised split-$`\hat R`$ above $`1.01`$ should be re-tuned and re-run.

These diagnostics are gradient-free and apply to any MCMC sampler. Hamiltonian Monte Carlo carries additional sampler-specific diagnostics, in particular divergence counts, that are introduced in [`computational-methods/hamiltonian-monte-carlo/`](../../computational-methods/hamiltonian-monte-carlo/) and reported in the posterior summary table in [`structural-econometrics/bayesian-dsge-hmc/`](../../structural-econometrics/bayesian-dsge-hmc/).

## References

- Gelman, A. and Rubin, D. B. (1992). "Inference from Iterative Simulation Using Multiple Sequences." *Statistical Science*, 7(4), 457-472. Sections 2-3 define the potential scale reduction factor.
- Geyer, C. J. (1992). "Practical Markov Chain Monte Carlo." *Statistical Science*, 7(4), 473-483. Sections 2-3 derive the IAT from the Kipnis-Varadhan central limit theorem and connect it to the effective sample size.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. (2021). "Rank-Normalization, Folding, and Localization: An Improved $`\hat R`$ for Assessing Convergence of MCMC." *Bayesian Analysis*, 16(2), 667-718. Sections 2-3 present the rank-normalised split-$`\hat R`$ used as the default modern threshold.
- Robert, C. P. and Casella, G. (2004). *Monte Carlo Statistical Methods*, 2nd edition. Springer. Chapter 12 on convergence diagnostics.
- **See also.** The chains diagnosed here are produced by the algorithm in [`computational-methods/metropolis-hastings/`](../../computational-methods/metropolis-hastings/). The gradient-based alternative is in [`computational-methods/hamiltonian-monte-carlo/`](../../computational-methods/hamiltonian-monte-carlo/), where divergences add a sampler-specific diagnostic to the three covered here. The same R-hat column governs convergence judgement in [`structural-econometrics/bayesian-dsge-hmc/`](../../structural-econometrics/bayesian-dsge-hmc/). The conjugate posteriors that motivate sampling when conjugacy fails are in [`bayesian-methods/bayesian-foundations/`](../../bayesian-methods/bayesian-foundations/).
