# Sampling a Two-Regime Structural Posterior

> Random-walk Metropolis-Hastings for structural parameters with two posterior modes.

## Overview

Suppose a structural model leaves the researcher with two plausible stories about the same data. In a demand application, one parameter region might explain observed choices through strong price sensitivity, while another explains them through stronger latent taste heterogeneity. The posterior over the structural parameter vector matters because counterfactual prices, welfare, and elasticities can differ across those regions.

The posterior density can often be evaluated up to a constant, but its moments and counterfactual averages require integration. Metropolis-Hastings replaces that hard integral with a Markov chain whose stationary distribution is the posterior. This page uses a small two-mode target so the failure modes are visible: a finite chain may accept many moves and still spend too long in one structural region.

## Equations

Let $D$ denote the data and let $\theta=(\theta_1,\theta_2)$ collect two
structural parameters. A Bayesian analysis targets the posterior kernel

$$
\pi(\theta \mid D) \propto L(D \mid \theta) p_0(\theta).
$$

The tutorial uses a two-component mixture as a transparent stand-in for this
posterior:

$$
\begin{aligned}
\pi(\theta \mid D)
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
= \min[1, \pi(\theta^\star \mid D) / \pi(\theta_t \mid D)].
$$

For any counterfactual object $g(\theta)$, the retained draws approximate
$E[g(\theta) \mid D]$. That approximation is weak when the chain rarely crosses
between posterior modes.

## Model Setup

| Object | Value |
|--------|-------|
| Posterior interpretation | Two empirically plausible structural regimes |
| $\mu_1$ | (1.5, 1.5) |
| $\mu_2$ | (-1.5, -1.5) |
| $\Sigma$ | [[1.0, 0.5], [0.5, 1.0]] |
| Mixing probability $\omega$ | 0.5 |
| Draws | 12,000 |
| Burn-in | 1,000 |
| Starting point | (10.0, -10.0) |
| Proposal steps | [0.15, 0.6, 2.0] |

## Solution Method

Random-walk Metropolis-Hastings only needs the posterior kernel at proposed and current parameter values. The normalizing constant cancels from the acceptance ratio, which is why the algorithm is useful in structural Bayesian work where likelihood evaluation is available but integration is not. The script runs three chains with different proposal scales to show the tuning tradeoff.

```text
Algorithm: random-walk Metropolis-Hastings
Input: log posterior kernel ell(theta), proposal scale s, initial theta_0, draws T
Output: draws from pi(theta | D), plus mixing diagnostics
1. Set theta = theta_0 and compute ell(theta)
2. For t = 1, ..., T:
       propose theta_star = theta + s * eta_t, eta_t ~ N(0, I)
       compute log alpha = ell(theta_star) - ell(theta)
       accept theta_star with probability min(1, exp(log alpha))
       otherwise repeat the current theta
3. Drop burn-in draws
4. Report acceptance, mode switches, posterior mean error, and ESS
```

The proposal scale $s$ controls the size of local moves. Tiny proposals accept often but crawl through the posterior. Large proposals can cross the low-density region between modes, but they are rejected more often. The toy target has a known mean, so the code can compare each chain with ground truth. In an empirical model, researchers would instead compare traces, multiple chains, posterior moments, and policy-relevant functions of $\theta$.

## Results

With proposal step 0.6, the chain visits both structural regimes while still accepting 69.9% of proposed moves.

<img src="figures/mh-walk.png" alt="Metropolis-Hastings walk over structural-posterior contours" width="80%">

The traces show whether the sampler has left its distant starting value, how often it moves between posterior regimes, and whether retained draws remain persistent.

<img src="figures/trace-plots.png" alt="Trace plots for the middle-step random-walk chain" width="80%">

The running mean and autocorrelation make the finite-chain problem concrete. A sampler that stays too long in one mode gives the wrong weight to that structural regime, even when every acceptance decision uses the correct posterior kernel.

<img src="figures/tuning-diagnostics.png" alt="Proposal tuning changes posterior bias and persistence" width="80%">

The true posterior mean is zero. The true marginal variance of each coordinate is 3.25, which is much larger than the within-regime variance because the two posterior regimes are far apart.

**Proposal-scale diagnostics**

|   Proposal step |   Acceptance rate |   Mode switches |   Mean error |   ESS theta1 |   ESS theta2 |
|----------------:|------------------:|----------------:|-------------:|-------------:|-------------:|
|            0.15 |             0.918 |              71 |        0.374 |           24 |           23 |
|            0.6  |             0.699 |             319 |        0.255 |          120 |          118 |
|            2    |             0.304 |             689 |        0.048 |          467 |          494 |

The middle proposal step, 0.6, is used in the path and trace plots. It gives acceptance 69.9% and visible movement between regimes. The smallest proposal accepts most often, but its posterior mean remains far from truth because the chain moves slowly. The largest proposal crosses modes more often and has the smallest mean error in this run, despite a lower acceptance rate. A useful Bayesian computation therefore asks whether the retained draws weight the economically relevant regions well enough for the moments or counterfactuals being reported.

## Takeaway

Metropolis-Hastings turns a pointwise posterior kernel into draws that can summarize structural uncertainty and counterfactual objects. The asymptotic guarantee is not the same as a useful finite run. Acceptance rates, traces, cumulative means, mode switches, and autocorrelation tell the researcher whether the chain has actually explored the posterior regions that matter for the economic conclusion.

## References

- [Metropolis, N. et al. (1953). Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 21(6), 1087-1092.](https://doi.org/10.1063/1.1699114)
- [Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97-109.](https://doi.org/10.1093/biomet/57.1.97)
- [Chib, S. and Greenberg, E. (1995). Understanding the Metropolis-Hastings Algorithm. *The American Statistician*, 49(4), 327-335.](https://doi.org/10.1080/00031305.1995.10476177)
