# Metropolis-Hastings Sampling Diagnostics

> Random-walk MCMC, proposal tuning, and mixing diagnostics on a bimodal target.

## Overview

Metropolis-Hastings turns a density that is easy to evaluate into draws from that density. The algorithm is simple: propose a move, compare the target density at the new and old locations, and sometimes accept a worse move so the chain keeps exploring.

This tutorial uses the same bimodal mixture as the optimization example. It is intentionally small enough to plot. Small proposals have high acceptance but move slowly. Large proposals jump farther but are often rejected. Useful MCMC lives between those extremes.

## Equations

The target is the same two-component mixture used in the optimization tutorial:

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
= \min\left\{1,\frac{p(\theta^\star)}{p(\theta_t)}\right\}.
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

The script evaluates the log target directly and runs a Gaussian random-walk chain. All acceptance decisions are made in log space to avoid numerical underflow.

The diagnostics compare three proposal step sizes. For each chain, the code reports the acceptance rate, number of switches between modes, posterior mean error, and a simple effective-sample-size estimate from autocorrelations.

## Results

With proposal step 0.6, the chain explores both modes while still accepting 69.9% of proposed moves.

![Metropolis-Hastings walk over target-density contours](figures/mh-walk.png)
*Metropolis-Hastings walk over target-density contours*

Trace plots reveal whether the chain has left its starting point, whether it moves between modes, and whether the retained draws are still highly persistent.

![Trace plots for the middle-step random-walk chain](figures/trace-plots.png)
*Trace plots for the middle-step random-walk chain*

A tiny proposal can have a comfortable acceptance rate but still move too slowly. A large proposal can switch modes, but rejection creates persistence. The best choice is empirical and target-specific.

![Proposal tuning changes bias and persistence](figures/tuning-diagnostics.png)
*Proposal tuning changes bias and persistence*

The true mixture mean is zero. The true marginal variance of each coordinate is 3.25, which is much larger than the within-component variance because the two modes are far apart.

**Proposal-scale diagnostics**

|   Proposal step |   Acceptance rate |   Mode switches |   Mean error |   ESS theta1 |   ESS theta2 |
|----------------:|------------------:|----------------:|-------------:|-------------:|-------------:|
|            0.15 |             0.918 |              71 |        0.374 |           24 |           23 |
|            0.6  |             0.699 |             319 |        0.255 |          120 |          118 |
|            2    |             0.304 |             689 |        0.048 |          467 |          494 |

The middle proposal step, 0.6, is used in the path and trace plots; it gives acceptance 69.9% and visible movement between modes. The small proposal accepts more often but has more persistent draws. The largest proposal is useful for jumping across the low-density middle region, but many jumps are rejected.

## Takeaway

Metropolis-Hastings is easy to implement, but not automatic. Acceptance rates, trace plots, cumulative means, mode switching, and autocorrelation diagnose different failure modes. The key lesson is general: a sampler can be correct in theory and still be weak for a finite computation if it explores the target too slowly.

## Reproduce

```bash
python run.py
```

## References

- [Metropolis, N. et al. (1953). Equation of State Calculations by Fast Computing Machines. *Journal of Chemical Physics*, 21(6), 1087-1092.](https://doi.org/10.1063/1.1699114)
- [Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications. *Biometrika*, 57(1), 97-109.](https://doi.org/10.1093/biomet/57.1.97)
- [Chib, S. and Greenberg, E. (1995). Understanding the Metropolis-Hastings Algorithm. *The American Statistician*, 49(4), 327-335.](https://doi.org/10.1080/00031305.1995.10476177)
