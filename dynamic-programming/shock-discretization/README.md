# Discretizing Persistent Shocks

> Finite Markov approximations to persistent productivity and income risk.

## Overview

Persistent productivity or income risk usually enters a Bellman equation through a finite Markov chain. That chain is not just a numerical convenience. Its grid points and transition probabilities determine continuation values, so they can change precautionary behavior, asset prices, equilibrium distributions, and the curvature of policy functions.

This tutorial compares Tauchen, Rouwenhorst, and a discrete-normal quadrature. The point is to see how one target AR(1) can lead to different grids, transition matrices, and stationary distributions. The same issue appears downstream in the consumption-savings, RBC, and Aiyagari tutorials, where the shock process feeds directly into household choices or equilibrium objects.

## Equations

Let $z_t$ denote the continuous shock state. The target process is the AR(1)

$$z_{t+1} = \rho z_t + \sigma_\epsilon \epsilon_{t+1}, \qquad
\epsilon_{t+1} \sim N(0,1).$$

Here $\rho$ is persistence and $\sigma_\epsilon$ is the innovation standard
deviation. The continuous process has unconditional variance

$$\operatorname{Var}(z_t) = \frac{\sigma_\epsilon^2}{1-\rho^2}.$$

A finite-state approximation replaces the continuous state with grid points
$z_1,\ldots,z_N$ and transition probabilities

$$P_{ij} = \Pr(z_{t+1} = z_j \mid z_t = z_i).$$

If a Bellman equation contains expected continuation value, the approximation
turns the integral over next period's shock into

$$\mathbb{E}[V(x_{t+1},z_{t+1})\mid z_t=z_i]
  \approx \sum_{j=1}^N P_{ij} V(x_{t+1},z_j).$$

The invariant distribution $\pi$ of the finite chain satisfies
$\pi = \pi P$ and $\sum_i \pi_i = 1$. With $\rho=0.95$ and
$\sigma_\epsilon=0.02$, the target unconditional standard deviation is
**0.0641**.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\rho$ | 0.95 | AR(1) persistence |
| $\sigma_\epsilon$ | 0.02 | Innovation standard deviation |
| States | 7 | Main comparison grid size |
| Grid sizes tested | [3, 5, 7, 9, 15] | Moment-accuracy sweep |
| Simulation periods | 180 | Path-comparison horizon |
| Path benchmark | Continuous AR(1) | Ground-truth history for transition plot |

## Solution Method

The pseudocode below uses the main comparison values $\rho=0.95$, $\sigma_\epsilon=0.02$, and $N=7$.

**Tauchen** is transparent and interval-based. It chooses an evenly spaced grid over a fixed number of unconditional standard deviations, then assigns transition probabilities by integrating normal mass between grid-cell cutoffs.

```text
Algorithm: Tauchen AR(1) discretization
Input: rho=0.95, sigma_epsilon=0.02, N=7, width m=3
Output: grid z_1,...,z_N, transition matrix P, invariant distribution pi
sigma_z = sigma_epsilon / sqrt(1 - rho^2)
Delta = 2 * m * sigma_z / (N - 1)
z_j = -m * sigma_z + (j - 1) * Delta,  j=1,...,N
c_1 = -infinity, c_{N+1} = +infinity
c_j = (z_{j-1} + z_j) / 2,  j=2,...,N
for current state i=1,...,N:
    for next state j=1,...,N:
        P_ij = Phi((c_{j+1} - rho * z_i) / sigma_epsilon)
               - Phi((c_j - rho * z_i) / sigma_epsilon)
pi solves pi = pi P and sum_j pi_j = 1
```

**Rouwenhorst** is built for persistent processes on coarse grids. The recursive construction preserves the target autocorrelation especially well when $\rho$ is close to one.

```text
Algorithm: Rouwenhorst AR(1) discretization
Input: rho=0.95, sigma_epsilon=0.02, N=7
Output: grid z_1,...,z_N, transition matrix P_N, invariant distribution pi
p = (1 + rho) / 2
P_2 = [[p, 1 - p], [1 - p, p]]
for n = 3,...,N:
    P_n = p       * upper_left(P_{n-1})
        + (1 - p) * upper_right(P_{n-1})
        + (1 - p) * lower_left(P_{n-1})
        + p       * lower_right(P_{n-1})
row-normalize P_N so each row sums to one
sigma_z = sigma_epsilon / sqrt(1 - rho^2)
z_j = sigma_z * sqrt(N - 1) * (2*(j - 1)/(N - 1) - 1)
pi solves pi = pi P_N and sum_j pi_j = 1
```

**Discrete normal** approximates an IID normal random variable directly. It is useful for quadrature and independent shocks, but it should not be confused with a Markov approximation to a persistent AR(1).

```text
Algorithm: discrete-normal quadrature
Input: mu=0, sigma_z=sigma_epsilon/sqrt(1-rho^2), N=7, width m=3
Output: grid z_1,...,z_N and one-period probabilities p_1,...,p_N
z_j = evenly spaced points from mu - m*sigma_z to mu + m*sigma_z
d_j = (z_j + z_{j+1}) / 2,  j=1,...,N-1
p_1 = Phi((d_1 - mu) / sigma_z)
for j=2,...,N-1:
    p_j = Phi((d_j - mu) / sigma_z) - Phi((d_{j-1} - mu) / sigma_z)
p_N = 1 - sum_{j=1}^{N-1} p_j
return grid z and probabilities p; there is no transition matrix
```

## Results

Start with the stationary probabilities. They show where the finite chain spends time in the long run. Even when methods target the same continuous process, they can place mass on different support points, which changes the states used inside continuation-value calculations.

Rouwenhorst places more mass near the center while preserving the target variance. Tauchen uses the wider evenly spaced support implied by the chosen width. The discrete-normal grid matches a one-period normal distribution rather than a persistent transition law.

<img src="figures/stationary-mass.png" alt="Stationary mass across discretization methods" width="80%">

The moment errors separate two questions: whether the finite chain has the right unconditional dispersion, and whether it carries shocks forward at the right rate. For persistent income or productivity risk, autocorrelation errors are often the more damaging error because they directly enter expected continuation values.

For highly persistent shocks, Tauchen can distort persistence on coarse grids. Rouwenhorst is designed to keep the autocorrelation close to the target even with few states.

<img src="figures/moment-accuracy.png" alt="Coarse Tauchen grids can overstate persistence" width="80%">

Sample paths make the transition matrix concrete. The black line is the continuous AR(1) ground truth. The finite chains start from the nearest grid point and use the same sequence of shock ranks, so the comparison shows how a coarse Markov chain turns a continuous history into discrete transitions.

The finite paths should not match the continuous path point by point. They show which movements the solved finite model can represent when the ground-truth process is compressed to seven states.

<img src="figures/simulated-paths.png" alt="Transition histories against the continuous AR(1)" width="80%">

The table reports the moments implied by each finite Markov chain. The 7-state Tauchen chain implies persistence 0.9622, while the 7-state Rouwenhorst chain implies persistence 0.9500. The discrete-normal standard-deviation error is 2.5132e-03.

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

These differences matter in models such as the consumption-savings, RBC, and Aiyagari examples. In each case, the shock process is inside an expectation operator, so the Markov chain affects choices before any simulation is run.

## Takeaway

The Markov chain is part of the economic model, not preprocessing. Tauchen is transparent and often adequate for moderate persistence. Rouwenhorst is safer for persistent income or productivity shocks on small grids because it preserves autocorrelation. Discrete-normal grids are useful for IID quadrature, but they do not replace a transition matrix when persistence matters.

## References

- [Tauchen, G. (1986). Finite State Markov-Chain Approximations to Univariate and Vector Autoregressions. *Economics Letters*, 20(2), 177-181.](https://doi.org/10.1016/0165-1765%2886%2990168-0)
- [Rouwenhorst, K. G. (1995). Asset Pricing Implications of Equilibrium Business Cycle Models. In T. Cooley (ed.), *Frontiers of Business Cycle Research*. Princeton University Press.](https://doi.org/10.1515/9780691218052-014)
- [Kopecky, K. A. and Suen, R. M. H. (2010). Finite State Markov-Chain Approximations to Highly Persistent Processes. *Review of Economic Dynamics*, 13(3), 701-714.](https://doi.org/10.1016/j.red.2009.07.002)
