# Latent-Regime Likelihoods and Optimizer Basins

> How optimization choices shape estimates when a likelihood has more than one basin.

## Overview

Suppose an economist estimates a model with two latent regimes, such as high and low productivity states, or two consumer types that generate similar choice patterns. The object of interest is the parameter vector that explains the data. The difficulty is that the likelihood may have more than one region that fits well.

This tutorial builds that situation in two dimensions. A mixture of two Gaussian regimes creates a negative log likelihood with two equally good basins. The example is small enough to draw, but it carries the empirical lesson: one optimizer run reports where its starting value led, while multi-start and global search checks tell us whether the estimate is stable across the criterion surface.

## Equations

Let $\theta=(\theta_1,\theta_2)$ denote the structural parameter vector. The
target density represents a stylized latent-regime likelihood. Each regime has
its own mean, and the researcher observes only the mixture:

$$
\begin{aligned}
p(\theta)
&= \omega \phi(\theta; \mu_1, \Sigma) \\
&\quad + (1-\omega)\phi(\theta; \mu_2, \Sigma).
\end{aligned}
$$

The estimator minimizes the negative log likelihood:

$$
\min_{\theta \in \mathbb{R}^2} f(\theta),
\qquad
f(\theta) = -\log p(\theta).
$$

Newton's method uses local curvature around the current parameter guess:

$$
\theta_{n+1} = \theta_n - H_f(\theta_n)^{-1}\nabla f(\theta_n).
$$

BFGS approximates the Hessian from gradient changes. Nelder-Mead moves a
simplex using only function values. Simulated annealing searches more broadly by
allowing occasional uphill moves, then polishes the best point locally.

## Model Setup

| Object | Value |
|--------|-------|
| $\mu_1$ | (1.50, 1.50) |
| $\mu_2$ | (-1.50, -1.50) |
| $\Sigma$ | [[1.0, 0.5], [0.5, 1.0]] |
| Mixing probability $\omega$ | 0.5 |
| Local-method start | (3.00, -2.50) |
| Global search box | $[-5,5]^2$ |

## Solution Method

All methods see the same likelihood criterion $f(\theta)$. The local optimizers start from the same off-diagonal point, so their paths show how curvature and simplex moves choose a basin. Dual annealing searches over the full box before a local polish. A separate BFGS restart grid maps which starting values lead to which mode.

```text
Algorithm: optimizer diagnostics for a latent-regime likelihood
Input: objective f(theta), starting value theta_0, search box B
Output: candidate estimates, paths, basin diagnostics
1. Run local optimizers from theta_0:
       Newton: update with a regularized Hessian and backtracking line search
       BFGS: update an inverse-Hessian approximation from gradient changes
       Nelder-Mead: move a simplex using only objective values
2. Run global search over B with stochastic uphill moves and local polishing
3. For each candidate theta_hat, record f(theta_hat) and its nearest regime mean
4. Restart BFGS on a grid of initial values to map basins of attraction
5. Read instability across starts as information about the likelihood surface
```

The known component means provide a diagnostic in this teaching example. In empirical work, the same role usually falls to multi-start checks, profile likelihoods, moment-residual plots, and restrictions that rule out economically irrelevant labels.

## Results

The contour plot mimics an estimation problem where two latent-regime parameter vectors fit about equally well. Newton and BFGS follow local curvature toward the upper-right basin, while Nelder-Mead and dual annealing settle in the lower-left basin.

<img src="figures/optimizer-paths.png" alt="Optimizer paths over negative log-density contours" width="80%">

Iteration counts measure local progress after a basin has effectively been chosen. A flat objective gap near zero does not tell us whether the optimizer inspected the other basin.

<img src="figures/convergence.png" alt="Objective gaps along recorded optimizer paths" width="80%">

Each dot is a BFGS starting value. The color records which regime mean the optimizer reaches, so the map shows where initialization changes the reported estimate.

<img src="figures/basin-map.png" alt="BFGS solutions from different starting points" width="80%">

The local methods share one starting value. Dual annealing searches over a box before returning a polished estimate.

**Optimizer outcomes**

| Method         | Start         | Solution       | Mode        |   Objective |   Distance to mode |   Iterations | Success   |
|:---------------|:--------------|:---------------|:------------|------------:|-------------------:|-------------:|:----------|
| Newton         | (3.00, -2.50) | (1.49, 1.49)   | upper-right |     2.38467 |            0.01082 |           38 | True      |
| BFGS           | (3.00, -2.50) | (1.49, 1.49)   | upper-right |     2.38467 |            0.01082 |            7 | True      |
| Nelder-Mead    | (3.00, -2.50) | (-1.49, -1.49) | lower-left  |     2.38467 |            0.01082 |           74 | True      |
| Dual annealing | box [-5,5]^2  | (-1.49, -1.49) | lower-left  |     2.38467 |            0.01082 |           80 | True      |

The best objective found is 2.38467. Because the mixture weights are equal, the two regime labels produce equally good estimates near their component means. The table therefore does not rank labels. It shows how much the reported estimate depends on local geometry and initialization. BFGS reaches a mode quickly from the common start, yet the restart grid shows that nearby starts can point to a different regime.

## Takeaway

Treat optimization as part of identification work. A point estimate from a local optimizer is credible only after the researcher has checked the criterion around it and tested sensitivity to starting values. Smooth, well-identified likelihoods reward derivative-based methods. Multimodal likelihoods call for restart grids, diagnostics, and sometimes a global search pass before the estimate gets an economic interpretation.

## References

- [Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.](https://doi.org/10.1007/978-0-387-40065-5)
- [Goffe, W. L., Ferrier, G. D., and Rogers, J. (1994). Global Optimization of Statistical Functions with Simulated Annealing. *Journal of Econometrics*, 60(1-2), 65-99.](https://doi.org/10.1016/0304-4076(94)90038-8)
- [Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261-272.](https://doi.org/10.1038/s41592-019-0686-2)
