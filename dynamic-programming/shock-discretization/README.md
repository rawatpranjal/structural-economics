# Shock Discretization

> Finite-state approximations to continuous productivity and income shocks.

## Overview

Many dynamic economic models begin with a continuous shock process but solve on a finite grid. This tutorial compares three workhorse approximations: Tauchen's normal grid for AR(1) processes, Rouwenhorst's highly persistent Markov chain, and a simple discrete-normal quadrature for one-period shocks.

The goal is not to pick one method forever. The goal is to understand what each method preserves: unconditional variance, persistence, tail support, and transition probabilities.

## Equations

We start from the continuous AR(1):

$$z_{t+1} = \rho z_t + \sigma_\epsilon \epsilon_{t+1}, \qquad
\epsilon_{t+1} \sim N(0,1).$$

The continuous process has unconditional variance:

$$\operatorname{Var}(z_t) = \frac{\sigma_\epsilon^2}{1-\rho^2}.$$

With $\rho=0.95$ and $\sigma_\epsilon=0.02$, the true unconditional
standard deviation is **0.0641**.

## Model Setup

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\rho$ | 0.95 | AR(1) persistence |
| $\sigma_\epsilon$ | 0.02 | Innovation standard deviation |
| States | 7 | Main comparison grid size |
| Grid sizes tested | [3, 5, 7, 9, 15] | Moment-accuracy sweep |

## Solution Method

**Tauchen** chooses an evenly spaced grid over a fixed number of unconditional standard deviations and assigns transition probabilities by integrating normal mass between grid-cell cutoffs.

**Rouwenhorst** builds the transition matrix recursively. It is especially useful when shocks are persistent because it preserves persistence well with few states.

**Discrete normal** approximates a one-period normal distribution directly. It is not a Markov approximation to persistence, but it is useful for quadrature and independent shocks.

## Results

Rouwenhorst places more mass near the center but keeps wider tail support. Tauchen uses evenly spaced cutoffs over the chosen width. The discrete-normal grid matches a one-period normal distribution rather than an AR(1) transition.

![Stationary distributions implied by 7-state discretizations](figures/stationary-mass.png)
*Stationary distributions implied by 7-state discretizations*

For highly persistent shocks, Tauchen can miss persistence on coarse grids. Rouwenhorst is designed to preserve the persistence parameter more tightly.

![Moment errors relative to the continuous AR(1)](figures/moment-accuracy.png)
*Moment errors relative to the continuous AR(1)*

The two chains can share the same target process but generate visibly different finite-state dynamics on coarse grids. This matters for policy functions near borrowing constraints or other nonlinear regions.

![Sample paths from Tauchen and Rouwenhorst chains](figures/simulated-paths.png)
*Sample paths from Tauchen and Rouwenhorst chains*

The table reports moments implied by the finite Markov chains.

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

The 7-state Tauchen chain implies persistence 0.9622, while the 7-state Rouwenhorst chain implies persistence 0.9500. The discrete-normal standard-deviation error is 2.5132e-03.

## Economic Takeaway

Discretization is part of the model, not a harmless preprocessing step. Tauchen is transparent and often adequate for moderate persistence. Rouwenhorst is safer for persistent income or productivity shocks because it preserves autocorrelation on small grids. Discrete-normal grids are useful for independent quadrature but do not replace a Markov transition matrix when persistence matters.

## Reproduce

```bash
python run.py
```

## References

- Tauchen, G. (1986). Finite state Markov-chain approximations to univariate and vector autoregressions. Economics Letters.
- Rouwenhorst, K. G. (1995). Asset pricing implications of equilibrium business cycle models. In Frontiers of Business Cycle Research.
- Kopecky, K. A. and Suen, R. M. H. (2010). Finite state Markov-chain approximations to highly persistent processes. Review of Economic Dynamics.
