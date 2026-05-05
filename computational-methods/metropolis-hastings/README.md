# Metropolis-Hastings for a Bimodal Posterior

> Random-walk MCMC diagnostics for structural parameters with two posterior modes.

## Overview

Bayesian structural work often has a posterior density that can be evaluated pointwise but not integrated analytically. Metropolis-Hastings turns that density into dependent draws. The draws are useful only if the chain explores the economically relevant parts of the posterior in the finite run actually used for inference.

The target below is a two-mode posterior over a parameter vector $\theta$. It is small enough to plot, but it carries the same practical problem as larger models: a proposal scale can look acceptable by one diagnostic and fail by another. Small proposals accept often but move slowly. Large proposals jump farther but spend time being rejected. The point is not to chase a universal acceptance rate; it is to diagnose mixing for the posterior at hand.

## Equations

Let $\theta=(\theta_1,\theta_2)$ be a structural parameter vector. The posterior
kernel used in the tutorial is a two-component mixture:

$$
\begin{aligned}
p(\theta)
&= \omega \phi(\theta; \mu_1, \Sigma) \\
&\quad + (1-\omega)\phi(\theta; \mu_2, \Sigma).
\end{aligned}
$$

Given current draw $\theta_t$, a random-walk proposal draws:

$$
\theta^\star = \theta_t + s \eta_t, \qquad \eta_t \sim N(0, I).
$$

Because the proposal is symmetric, the Metropolis-Hastings acceptance probability is:

$$
\alpha(\theta_t,\theta^\star)
= \min\bigl[1,\frac{p(\theta^\star)}{p(\theta_t)}\bigr].
$$

## Model Setup

| Object | Value |
|--------|-------|
| $\mu_1$ | (1.5, 1.5) |
| $\mu_2$ | (-1.5, -1.5) |
| $\Sigma$ | [[1.0, 0.5], [0.5, 1.0]] |
| Mixing probability $\omega$ | 0.5 |
| Draws | 12,000 |
| Burn-in | 1,000 |
| Starting point | (10.0, -10.0) |
| Proposal steps | [0.15, 0.6, 2.0] |

## Solution Method

The script evaluates the log posterior kernel directly and runs Gaussian random-walk chains at three proposal scales. All acceptance decisions are made in log space to avoid numerical underflow.

```text
Algorithm: random-walk Metropolis-Hastings
Input: log posterior ell(theta), proposal scale s, initial theta_0, draws T
Output: Markov chain theta_1, ..., theta_T and diagnostics
1. Set current state theta = theta_0 and current log density ell(theta)
2. For t = 1, ..., T:
       propose theta_star = theta + s * eta_t, eta_t ~ N(0, I)
       compute log alpha = ell(theta_star) - ell(theta)
       accept theta_star with probability min(1, exp(log alpha))
       otherwise repeat the current theta
3. Drop burn-in draws
4. Report acceptance, mode switches, posterior mean error, and ESS
```

The true mixture mean is known here, so the posterior mean error is a ground-truth diagnostic. In empirical applications, trace plots, multiple chains, posterior moments, and economically meaningful functionals serve the same role.

## Results

With proposal step 0.6, the chain explores both modes while still accepting 69.9% of proposed moves.

<img src="figures/mh-walk.png" alt="Metropolis-Hastings walk over target-density contours" width="80%">

Trace plots show whether the chain has left its starting point, whether it moves between modes, and whether the retained draws are still highly persistent.

<img src="figures/trace-plots.png" alt="Trace plots for the middle-step random-walk chain" width="80%">

A tiny proposal can have a comfortable acceptance rate but still move too slowly. A large proposal can switch modes, but rejection creates persistence. The best choice is empirical and target-specific.

<img src="figures/tuning-diagnostics.png" alt="Proposal tuning changes bias and persistence" width="80%">

The true mixture mean is zero. The true marginal variance of each coordinate is 3.25, which is much larger than the within-component variance because the two modes are far apart.

**Proposal-scale diagnostics**

|   Proposal step |   Acceptance rate |   Mode switches |   Mean error |   ESS theta1 |   ESS theta2 |
|----------------:|------------------:|----------------:|-------------:|-------------:|-------------:|
|            0.15 |             0.918 |              71 |        0.374 |           24 |           23 |
|            0.6  |             0.699 |             319 |        0.255 |          120 |          118 |
|            2    |             0.304 |             689 |        0.048 |          467 |          494 |

The middle proposal step, 0.6, is used in the path and trace plots; it gives acceptance 69.9% and visible movement between modes. The small proposal accepts more often but has more persistent draws. The largest proposal is useful for jumping across the low-density middle region, but many jumps are rejected. The table shows why no single diagnostic is sufficient: acceptance, mode switching, mean error, and effective sample size rank the proposal scales differently.

## Takeaway

Metropolis-Hastings is correct asymptotically under weak conditions, but finite-run Bayesian inference depends on mixing. Acceptance rates, trace plots, cumulative means, mode switching, and autocorrelation diagnose different failures. A sampler can target the right posterior in theory and still produce weak empirical inference if it explores that posterior too slowly.

## References

- [Metropolis, N. et al. (1953). Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 21(6), 1087-1092.](https://doi.org/10.1063/1.1699114)
- [Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97-109.](https://doi.org/10.1093/biomet/57.1.97)
- [Chib, S. and Greenberg, E. (1995). Understanding the Metropolis-Hastings Algorithm. *The American Statistician*, 49(4), 327-335.](https://doi.org/10.1080/00031305.1995.10476177)
