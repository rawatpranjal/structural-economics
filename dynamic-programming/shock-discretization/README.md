# Discretizing Persistent Shocks

> Why persistent shocks need a finite-state approximation before they enter Bellman equations.

## Overview

Suppose a household receives low income this year. Saving depends on whether low income is likely to persist. The same issue appears when productivity shocks guide investment.

The object is a persistent log income or productivity shock. We model it as a Gaussian AR(1) with persistence $\rho$ and innovation scale $\sigma_\epsilon$.

A Bellman equation needs finite shock states. It also needs a transition matrix for next-period expectations. The tutorial compares Tauchen and Rouwenhorst by their variance and persistence errors.

## Equations

A household with assets $a$ faces shock $z_i$. It chooses next assets $a'$.
The continuation value averages over next-period shock states:

$$V(a,z_i) = \max_{a' \in \mathcal{A}}
[ u(Ra+\exp(z_i)-a') + \beta \sum_{j=1}^N P_{ij} V(a',z_j) ].$$

Here $R$ is the gross return factor, $\beta \in (0,1)$ is the discount factor,
and $u(\cdot)$ is a concave increasing utility function. $\mathcal{A}$ is the
feasible asset set.

The finite object is the grid $\{z_1,\dots,z_N\}$ and transition matrix $P$.
The continuous target is the Gaussian AR(1)

$$z_{t+1} = \rho\, z_t + \sigma_\epsilon\, \varepsilon_{t+1},
\qquad \varepsilon_{t+1} \sim \mathcal{N}(0,1).$$

The AR(1) has unconditional law $z_t \sim \mathcal{N}(0,\sigma_z^2)$ with

$$\sigma_z^2 = \frac{\sigma_\epsilon^2}{1-\rho^2},
\qquad \rho_k \equiv \mathrm{Corr}(z_t, z_{t+k}) = \rho^k.$$

For $\rho=0.95$ and $\sigma_\epsilon=0.02$, the standard deviation is
$\sigma_z = 0.0641$. The shock half-life is
$\ln 2 / (-\ln \rho) \approx 14$ periods.

A finite chain replaces the conditional Gaussian law with
$P\in\mathbb{R}^{N\times N}$. Each row gives probabilities
$P_{ij}=\Pr(z_{t+1}=z_j\mid z_t=z_i)$. The conditional expectation becomes

$$\mathbb{E}[V(a',z_{t+1})\mid z_t=z_i]
= \sum_{j=1}^N P_{ij} V(a', z_j).$$

The chain has an invariant distribution $\pi$ satisfying $\pi=\pi P$ and
$\sum_i \pi_i = 1$. Two diagnostics matter:

1. Does the chain match $\sigma_z$?
2. Does the chain match $\rho$?

Variance controls risk exposure. Persistence controls expected continuation
values after good and bad shocks.

## Model Setup

The calibration is a small annual log-income or log-productivity process. It is designed for dynamic programming, not forecasting.

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\rho$ | 0.95 | AR(1) persistence |
| $\sigma_\epsilon$ | 0.02 | Innovation standard deviation |
| $\sigma_z$ | 0.0641 | Implied unconditional standard deviation |
| $N$ | 7 | Main comparison grid size |
| Grid sweep | [3, 5, 7, 9, 15] | Grid sizes for moment checks |
| Tauchen half-width $m$ | 3 | Grid bound in unconditional std deviations |
| $T_{sim}$ | 180 | Simulation horizon |

## Solution Method

Choose a small grid and transition matrix that keep variance and persistence. Tauchen and Rouwenhorst make different compromises.

### Tauchen (1986): integrate Gaussian mass between cell midpoints

Tauchen places an evenly spaced grid over $[-m\sigma_z,\, m\sigma_z]$. For each $z_i$, $z_{t+1}$ is normal with mean $\rho z_i$. $P_{ij}$ is the conditional mass assigned to the cell around $z_j$, computed using the standard normal CDF $\Phi(\cdot)$. Endpoint cells collect remaining tail mass.

```text
Algorithm 1: Tauchen
Input:  rho, sigma_eps, N, half-width m
Output: grid {z_j}, transition P, invariant pi
  sigma_z = sigma_eps / sqrt(1 - rho^2)
  z_j     = -m*sigma_z + (j-1) * 2*m*sigma_z/(N-1)         for j = 1..N
  c_j     = (z_{j-1} + z_j) / 2                             (cell midpoints)
  c_1     = -inf,   c_{N+1} = +inf
  for i = 1..N, j = 1..N:
      P[i,j] = Phi((c_{j+1} - rho*z_i) / sigma_eps)
             - Phi((c_j     - rho*z_i) / sigma_eps)
  solve pi = pi P,   sum_j pi_j = 1
```

The benefit is transparency. The grid support is visible. The cost appears with high $\rho$ and small $N$. Mass from near-tail states spills past endpoints. The last cell absorbs that mass and becomes too sticky. A wider support protects tails. A narrower support protects the center. Neither choice fixes a coarse grid.

### Rouwenhorst (1995): match the moments by construction

Rouwenhorst builds $P_N$ from a two-state base. The base uses $p=(1+\rho)/2$. The recursion preserves autocorrelation as states are added. The grid is scaled to match $\sigma_z^2$.

By construction, the chain matches $\rho$ and $\sigma_z^2$ for any $N \ge 2$. It has no quadrature error in those moments. The tradeoff is distributional shape. On small grids, its invariant distribution is binomial, not Gaussian.

```text
Algorithm 2: Rouwenhorst
Input:  rho, sigma_eps, N
Output: grid {z_j}, transition P_N, invariant pi
  p   = (1 + rho) / 2
  P_2 = [[p, 1-p],
         [1-p, p]]
  for n = 3..N:
      A_TL = embed P_{n-1} in top-left of n x n zero matrix
      A_TR = embed P_{n-1} in top-right of n x n zero matrix
      A_BL = embed P_{n-1} in bottom-left of n x n zero matrix
      A_BR = embed P_{n-1} in bottom-right of n x n zero matrix
      P_n  = p*A_TL + (1-p)*A_TR + (1-p)*A_BL + p*A_BR
      row-normalize interior rows of P_n     (they receive two contributions)
  sigma_z = sigma_eps / sqrt(1 - rho^2)
  z_j     = sigma_z * sqrt(N-1) * (2*(j-1)/(N-1) - 1)        for j = 1..N
  pi_j    = binomial(N-1, j-1) / 2^{N-1}
```

For highly persistent shocks on coarse grids, this is usually safer. It protects the moments that enter continuation values.

## Results

Stationary mass shows where each chain puts probability. The dashed curve is the AR(1)'s long-run Gaussian density, scaled by the Tauchen cell width. Tauchen follows the curve near the center, but it puts extra mass in outer states. That extra tail mass creates the variance error. Rouwenhorst has a binomial invariant distribution, so its center is heavier and its tails are thinner.

<img src="figures/stationary-mass.png" alt="Stationary mass for Tauchen and Rouwenhorst" width="80%">

Moment errors show the main diagnostic. The zero line is the AR(1) target. Rouwenhorst stays on zero because the recursion enforces variance and persistence. Tauchen approaches the targets as $N$ grows. With $N=3$, Tauchen is almost absorbing, so persistence is near one. At $\rho=0.95$, small persistence errors affect each continuation value.

<img src="figures/moment-accuracy.png" alt="Moment errors by grid size" width="80%">

Simulated paths make the transition matrix visible. The finite chains receive the same innovation ranks as the AR(1) path. They move on coarse grids, so they cannot match the continuous path point by point. The useful check is the rhythm of persistence. Rouwenhorst tracks slow drift more cleanly at $N=7$.

<img src="figures/simulated-paths.png" alt="Simulated AR(1) and finite-chain paths" width="80%">

Numerical detail behind the moment-error figure. The 7-state Tauchen chain implies persistence 0.9622 against a target of 0.95. Rouwenhorst matches the target at every $N$.

**Moment accuracy across discretization methods**

| Method      |   States |     Std |   Std error |   Persistence |   Persistence error |
|:------------|---------:|--------:|------------:|--------------:|--------------------:|
| Tauchen     |        3 | 0.13344 |     0.06939 |       0.99999 |             0.04999 |
| Rouwenhorst |        3 | 0.06405 |     0       |       0.95    |             0       |
| Tauchen     |        5 | 0.08414 |     0.02009 |       0.98787 |             0.03787 |
| Rouwenhorst |        5 | 0.06405 |    -0       |       0.95    |            -0       |
| Tauchen     |        7 | 0.07918 |     0.01513 |       0.9622  |             0.0122  |
| Rouwenhorst |        7 | 0.06405 |     0       |       0.95    |             0       |
| Tauchen     |        9 | 0.07509 |     0.01103 |       0.95128 |             0.00128 |
| Rouwenhorst |        9 | 0.06405 |    -0       |       0.95    |            -0       |
| Tauchen     |       15 | 0.06805 |     0.004   |       0.94889 |            -0.00111 |
| Rouwenhorst |       15 | 0.06405 |     0       |       0.95    |             0       |

## Takeaway

Discretization is part of the economic model. With persistent shocks and small $N$, Rouwenhorst is the safer default. It matches $\sigma_z$ and $\rho$ by construction. Tauchen is transparent and can approximate the Gaussian shape well on finer grids. At $\rho=0.95$ and $N=7$, Tauchen overstates persistence enough to change continuation values. Choose the chain by the moments that matter in the Bellman equation.

## References

- [Tauchen, G. (1986). Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions. *Economics Letters*, 20(2), 177-181.](https://doi.org/10.1016/0165-1765%2886%2990168-0)
- [Rouwenhorst, K. G. (1995). Asset Pricing Implications of Equilibrium Business Cycle Models. In T. Cooley (ed.), *Frontiers of Business Cycle Research*. Princeton University Press.](https://doi.org/10.1515/9780691218052-014)
- [Kopecky, K. A. and Suen, R. M. H. (2010). Finite State Markov-Chain Approximations to Highly Persistent Processes. *Review of Economic Dynamics*, 13(3), 701-714.](https://doi.org/10.1016/j.red.2010.02.002)
