# Sampling a Two-Regime Structural Posterior

## Overview

Suppose a structural model leaves two parameter regions that fit the same data. Counterfactual prices or welfare can depend on which region receives posterior weight.

The object is the posterior over $\theta=(\theta_1,\theta_2)$. This tutorial represents it with two Gaussian modes, which stand for two plausible structural regimes.

We can evaluate the posterior density up to a constant, but averages require integration. Random-walk Metropolis-Hastings replaces the integral with draws from a Markov chain. The finite run is useful only when it crosses modes often enough.

## Equations

Let $D$ denote the data, and let $\theta=(\theta_1,\theta_2)$ collect two
structural parameters. The posterior kernel is

$$
\pi(\theta \mid D) \propto L(D \mid \theta) p_0(\theta).
$$

The normalizing constant $\int L(D\mid\theta) p_0(\theta)\,d\theta$ is the integral we cannot compute. Sampling avoids it.

The target used here is a two-component mixture:

$$
\begin{aligned}
\pi(\theta \mid D)
&= \omega \phi(\theta; \mu_1, \Sigma) \\
&\quad + (1-\omega)\phi(\theta; \mu_2, \Sigma).
\end{aligned}
$$

Here $\phi(\cdot;\mu,\Sigma)$ denotes the bivariate normal density with mean $\mu$ and covariance $\Sigma$.

A random-walk proposal does not need to know the target's shape. It is the natural default when only the kernel can be evaluated. The acceptance step is what filters out moves into low-density regions.

Given current draw $\theta_t$, a random-walk proposal draws:

$$
\theta^\star = \theta_t + s \eta_t, \qquad \eta_t \sim N(0, I).
$$

Because the proposal is symmetric, the Metropolis-Hastings acceptance probability is:

$$
\alpha(\theta_t,\theta^\star) =
\min[1, \pi(\theta^\star \mid D) / \pi(\theta_t \mid D)].
$$

The acceptance ratio depends only on the kernel ratio. The unknown normalizing constant cancels, which is the load-bearing reason MH works without a partition function.

Retained draws approximate posterior averages for any counterfactual object
$g(\theta)$. The approximation is weak when the chain rarely crosses modes.

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

Random-walk Metropolis-Hastings needs the posterior kernel at current and proposed parameter values. The normalizing constant cancels from the acceptance ratio. The script runs three proposal scales to show the tuning tradeoff.

```text
Algorithm: random-walk Metropolis-Hastings
Input: log posterior kernel ell(theta), proposal scale s, initial theta_0, draws T
Output: draws from pi(theta | D), plus mode-crossing summaries
1. Set theta = theta_0 and compute ell(theta)
2. For t = 1, ..., T:
       propose theta_star = theta + s * eta_t, eta_t ~ N(0, I)
       compute log alpha = ell(theta_star) - ell(theta)
       accept theta_star with probability min(1, exp(log alpha))
       otherwise repeat the current theta
3. Drop burn-in draws
4. Report acceptance, mode switches, posterior mean error, and ESS
```

Proposal scale $s$ controls local move size. Tiny steps accept often but cross modes slowly. Large steps cross low-density regions more often, but many proposals are rejected. The known mixture mean lets the code measure finite-chain error.

For high-dimensional Gaussian targets the asymptotically optimal acceptance rate is roughly 0.23 (Roberts, Gelman, and Gilks 1997). That result is why tuning advice for $s$ usually targets acceptance between 0.2 and 0.5. On bimodal targets like this one, the rule is a guide but not a guarantee, because what limits the chain is mode-jumping rather than local mixing.

Two diagnostics measure these chain qualities. Effective sample size turns the autocorrelated chain into an equivalent count of independent draws. Mode switches count how often the chain crosses between regimes. Together these checks say whether posterior averages weight the structural regimes correctly or report a regime artifact.

## Results

With proposal step 0.6, the chain visits both regimes and accepts 69.9% of proposed moves.

<img src="figures/mh-walk.png" alt="Metropolis-Hastings walk over structural-posterior contours" width="80%">

The traces show burn-in, mode crossing, and persistence in the retained draws.

<img src="figures/trace-plots.png" alt="Trace plots for the middle-step random-walk chain" width="80%">

The running mean and autocorrelation show how proposal scale changes finite-chain error.

<img src="figures/tuning-diagnostics.png" alt="Proposal tuning changes posterior bias and persistence" width="80%">

The true posterior mean is zero. Each coordinate has marginal variance 3.25 because the modes are far apart.

**Proposal-scale diagnostics**

|   Proposal step |   Acceptance rate |   Mode switches |   Mean error |   ESS theta1 |   ESS theta2 |
|----------------:|------------------:|----------------:|-------------:|-------------:|-------------:|
|            0.15 |             0.918 |              71 |        0.374 |           24 |           23 |
|            0.6  |             0.699 |             319 |        0.255 |          120 |          118 |
|            2    |             0.304 |             689 |        0.048 |          467 |          494 |

The middle proposal step, 0.6, is used in the path and trace plots. It gives acceptance 69.9% and moves between regimes. The smallest proposal accepts most often but crosses modes slowly. The largest proposal has lower acceptance, more mode switches, and the smallest mean error. The table shows why acceptance rate alone is not enough.

## Takeaway

Metropolis-Hastings turns a posterior kernel into draws for structural uncertainty and counterfactual averages. A finite run can still weight regimes incorrectly. Use traces, cumulative means, mode switches, and autocorrelation to check whether the chain supports the economic conclusion.

## References

- [Metropolis, N. et al. (1953). Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 21(6), 1087-1092.](https://doi.org/10.1063/1.1699114)
- [Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97-109.](https://doi.org/10.1093/biomet/57.1.97)
- [Chib, S. and Greenberg, E. (1995). Understanding the Metropolis-Hastings Algorithm. *The American Statistician*, 49(4), 327-335.](https://doi.org/10.1080/00031305.1995.10476177)
