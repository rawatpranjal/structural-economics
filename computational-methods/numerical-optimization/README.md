# Latent-Regime Likelihoods and Optimizer Basins

## Overview

An economist may estimate a model with two latent regimes. Each regime can explain the same data with a different parameter vector.

The object is the parameter vector that minimizes a negative log likelihood. The tutorial uses a two-dimensional Gaussian mixture so the full criterion can be drawn.

A single local optimizer can report the basin reached from its starting value. Restart grids and global search check whether the estimate changes across basins.

## Equations

Let $\theta=(\theta_1,\theta_2)$ denote the structural parameter vector. The
target density is a stylized latent-regime likelihood. Each regime has its own
mean, while the researcher observes only the mixture:

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

Newton's method uses local curvature around the current guess:

$$
\theta_{n+1} = \theta_n - H_f(\theta_n)^{-1}\nabla f(\theta_n).
$$

Here $H_f(\theta)$ denotes the Hessian matrix of $f$ at $\theta$.

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

All methods minimize the same criterion $f(\theta)$. The local methods start from the same off-diagonal value. Their paths show how different search rules choose a basin. Dual annealing searches the full box before local polishing. The BFGS restart grid maps the basin reached from each start.

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

The known component means make basin labels observable in this teaching example. In empirical work, the restart grid is the main diagnostic.

## Results

The contour plot shows two parameter vectors with the same likelihood value. Newton and BFGS reach the upper-right basin. Nelder-Mead and dual annealing reach the lower-left basin.

<img src="figures/optimizer-paths.png" alt="Optimizer paths over negative log-density contours" width="80%">

Iteration counts measure progress after a basin has been chosen. A zero objective gap does not show that the other basin was checked.

<img src="figures/convergence.png" alt="Objective gaps along recorded optimizer paths" width="80%">

Each dot is a BFGS starting value. The color records the regime mean reached by the optimizer.

<img src="figures/basin-map.png" alt="BFGS solutions from different starting points" width="80%">

The local methods share one starting value. Dual annealing searches the box first.

**Optimizer outcomes**

| Method         | Start         | Solution       | Mode        |   Objective |   Distance to mode |   Iterations | Success   |
|:---------------|:--------------|:---------------|:------------|------------:|-------------------:|-------------:|:----------|
| Newton         | (3.00, -2.50) | (1.49, 1.49)   | upper-right |     2.38467 |            0.01082 |           38 | True      |
| BFGS           | (3.00, -2.50) | (1.49, 1.49)   | upper-right |     2.38467 |            0.01082 |            7 | True      |
| Nelder-Mead    | (3.00, -2.50) | (-1.49, -1.49) | lower-left  |     2.38467 |            0.01082 |           74 | True      |
| Dual annealing | box [-5,5]^2  | (-1.49, -1.49) | lower-left  |     2.38467 |            0.01082 |           80 | True      |

The best objective found is 2.38467. The equal mixture weights make both regime labels fit equally well. The table therefore does not rank labels. It shows that the reported estimate can depend on initialization. The restart grid shows that nearby starts can point to different regimes.

## Takeaway

A local optimum is not enough in a latent-regime likelihood. Check the criterion around the estimate and rerun from many starts. Use a global pass when local starts find different basins. Interpret the estimate only after those checks.

## References

- [Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.](https://doi.org/10.1007/978-0-387-40065-5)
- [Goffe, W. L., Ferrier, G. D., and Rogers, J. (1994). Global Optimization of Statistical Functions with Simulated Annealing. *Journal of Econometrics*, 60(1-2), 65-99.](https://doi.org/10.1016/0304-4076(94)90038-8)
- [Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261-272.](https://doi.org/10.1038/s41592-019-0686-2)
