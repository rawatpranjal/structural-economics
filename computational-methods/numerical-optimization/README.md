# Optimization Methods for Economic Objectives

> Local, derivative-free, and stochastic search on a multimodal objective.

## Overview

Optimization is the computational core of estimation, calibration, design, and many machine-learning workflows. This tutorial uses one two-dimensional objective so the algorithms can be seen rather than treated as black boxes.

The surface is the negative log density of a bimodal Gaussian mixture. It has two equally good modes. That simple feature makes the main lesson visible: local methods are fast once they are in the right basin, while global methods spend more computation to reduce dependence on the starting point.

## Equations

The target density is a mixture of two bivariate normals:

$$
\begin{aligned}
p(\theta)
&= \omega \phi(\theta; \mu_1, \Sigma) \\
&\quad + (1-\omega)\phi(\theta; \mu_2, \Sigma).
\end{aligned}
$$

The numerical problem is:

$$
\min_{\theta \in \mathbb{R}^2} f(\theta),
\qquad
f(\theta) = -\log p(\theta).
$$

Newton's method uses local curvature:

$$
\theta_{n+1} = \theta_n - H_f(\theta_n)^{-1}\nabla f(\theta_n).
$$

BFGS approximates the Hessian from gradient changes, Nelder-Mead moves a simplex without
derivatives, and simulated annealing accepts occasional uphill moves to search more globally.

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

**Newton:** compute finite-difference gradients and Hessians, then use backtracking to reject steps that raise the objective.

**BFGS:** use SciPy's quasi-Newton optimizer from the same starting point.

**Nelder-Mead:** use a derivative-free simplex search from the same starting point.

**Dual annealing:** search over a bounded box with stochastic global exploration and a final local polish.

## Results

The same objective can make algorithms look very different. Local methods move quickly once their local model is useful. Global search explores the box before settling near one of the modes.

<img src="figures/optimizer-paths.png" alt="Optimizer paths over negative log-density contours" width="80%">
*Optimizer paths over negative log-density contours*

Fast convergence is conditional on being in a good basin and having a stable local approximation. A low iteration count is not the same thing as a global guarantee.

<img src="figures/convergence.png" alt="Objective gaps along recorded optimizer paths" width="80%">
*Objective gaps along recorded optimizer paths*

A local optimizer does not just solve an objective; it solves an objective from a starting point. Multi-start checks are a cheap diagnostic for multimodality.

<img src="figures/basin-map.png" alt="BFGS solutions from different starting points" width="80%">
*BFGS solutions from different starting points*

All local methods start at the same point; dual annealing searches over a box.

**Optimizer outcomes**

| Method         | Start         | Solution       | Mode        |   Objective |   Distance to mode |   Iterations | Success   |
|:---------------|:--------------|:---------------|:------------|------------:|-------------------:|-------------:|:----------|
| Newton         | (3.00, -2.50) | (1.49, 1.49)   | upper-right |     2.38467 |            0.01082 |           38 | True      |
| BFGS           | (3.00, -2.50) | (1.49, 1.49)   | upper-right |     2.38467 |            0.01082 |            7 | True      |
| Nelder-Mead    | (3.00, -2.50) | (-1.49, -1.49) | lower-left  |     2.38467 |            0.01082 |           74 | True      |
| Dual annealing | box [-5,5]^2  | (-1.49, -1.49) | lower-left  |     2.38467 |            0.01082 |           80 | True      |

The best objective found is 2.38467. Because the two mixture weights are equal, the objective has two equally good modes near the two component means. The important comparison is not which side wins, but how much each method depends on local geometry and initialization.

## Takeaway

Optimization is a modeling choice as well as a numerical routine. For smooth unimodal problems, derivative-based methods are usually efficient. For rough, flat, or multimodal surfaces, it is safer to combine local methods with multi-start runs, diagnostic plots, or a global search pass. The plots here are small enough to inspect, but the same logic applies when the objective has hundreds of parameters.

## Reproduce

```bash
python run.py
```

## References

- [Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.](https://doi.org/10.1007/978-0-387-40065-5)
- [Goffe, W. L., Ferrier, G. D., and Rogers, J. (1994). Global Optimization of Statistical Functions with Simulated Annealing. *Journal of Econometrics*, 60(1-2), 65-99.](https://doi.org/10.1016/0304-4076(94)90038-8)
- [Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261-272.](https://doi.org/10.1038/s41592-019-0686-2)
