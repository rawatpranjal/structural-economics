# Particle Filtering Latent Economic States

> Sequential Monte Carlo, particle degeneracy, and proposal design in signal extraction.

## Overview

When latent economic states evolve nonlinearly or shocks are non-Gaussian, the Kalman filter is no longer available in closed form. Particle filters keep the same filtering question but represent the posterior distribution with weighted simulated states. The economic object is still the filtered state distribution; the computational issue is whether the particles are placed where the likelihood is informative.

This tutorial deliberately keeps the model linear and Gaussian, so the Kalman filter provides a benchmark. That lets us isolate particle error without changing the state space model. The bootstrap filter simulates from the transition equation and then weights by the observation density. The conditionally optimal filter uses the current observation inside the proposal, reducing weight degeneracy when measurements are sharp.

## Equations

The hidden-state model is:

$$
y_t = \Psi s_t + u_t, \qquad s_t = \Phi s_{t-1} + \epsilon_t.
$$

The bootstrap particle filter propagates particles from:

$$
s_t^{(i)} \sim p(s_t \mid s_{t-1}^{(i)})
$$

and weights them by the observation likelihood:

$$
w_t^{(i)} \propto p(y_t \mid s_t^{(i)}).
$$

The conditionally optimal proposal uses the current observation:

$$
s_t^{(i)} \sim p(s_t \mid s_{t-1}^{(i)}, y_t).
$$

Effective sample size summarizes weight concentration:

$$
ESS_t = \frac{1}{\sum_i (w_t^{(i)})^2}.
$$

## Model Setup

| Object | Value |
|--------|-------|
| Observation matrix $\Psi$ | [1.0, 0.9] |
| Transition matrix $\Phi$ | diag(0.4, 0.5) |
| Measurement std | 0.10 |
| Process std | (0.30, 0.25) |
| Baseline particles | 500 |
| Repeated runs | 50 |
| Benchmark | Kalman filtered mean |

## Solution Method

**Bootstrap filter:** propagate particles with the transition equation, weight by the Gaussian observation density, estimate the state mean, then resample every period.

**Conditionally optimal filter:** for each particle, combine the transition density and the current observation to sample from $p(s_t \mid s_{t-1}, y_t)$. The remaining weights are the one-step predictive likelihoods $p(y_t \mid s_{t-1})$.

```text
Algorithm: particle filtering with resampling
Input: observations y_t, particles s_{0}^{(i)}, proposal q, particle count N
Output: filtered state means, ESS, likelihood estimate
for t = 1, ..., T:
    for each particle i:
        draw proposed state s_t^{(i)} from q(s_t | s_{t-1}^{(i)}, y_t)
        compute importance weight w_t^{(i)} from target / proposal density
    normalize weights and estimate E[s_t | y_{1:t}]
    compute ESS_t = 1 / sum_i (w_t^{(i)})^2
    resample particles according to normalized weights
    accumulate the likelihood increment
```

The code repeats each filter many times and reports Monte Carlo error relative to the Kalman filtered mean. Because the benchmark is exact in this linear Gaussian model, the tables measure particle approximation error rather than model misspecification.

## Results

With 500 particles, both particle filters track the Kalman benchmark. The difference between the two methods is easier to see in repeated-run error and effective sample size.

<img src="figures/filter-comparison.png" alt="Particle filter state estimates compared with the Kalman filter" width="80%">

Effective sample size falls when a few particles receive most of the weight. The conditionally optimal proposal usually preserves more useful particles because it looks at the observation before drawing the new state.

<img src="figures/mse-and-ess.png" alt="Repeated-run Monte Carlo error and effective sample size" width="80%">

Low measurement noise makes the likelihood sharply peaked. Bootstrap particles drawn from the transition can miss that peak, creating weight degeneracy.

<img src="figures/measurement-noise.png" alt="Particle accuracy as measurement noise falls" width="80%">

Outliers are hard for likelihood-weighted simulation. A single extreme observation can concentrate weights and pull the filtered state sharply.

<img src="figures/outlier-stress.png" alt="Filtering after multiplying observation 25 by ten" width="80%">

The baseline repeats each filter 50 times with 500 particles.

**Baseline repeated-run comparison**

| Method    |   Particles |   PF RMSE vs Kalman |   Mean ESS |   Loglike sd |
|:----------|------------:|--------------------:|-----------:|-------------:|
| bootstrap |         500 |              0.0273 |    121.829 |       0.6914 |
| optimal   |         500 |              0.01   |    492.636 |       0.0397 |

Lower measurement noise makes the observation likelihood sharper.

**Measurement-noise sensitivity**

|   Measurement std | Method    |   PF RMSE vs Kalman |   Mean ESS |   Loglike sd |   Kalman RMSE vs truth |
|------------------:|:----------|--------------------:|-----------:|-------------:|-----------------------:|
|              0.25 | bootstrap |              0.024  |   172.396  |       0.4286 |                 0.2361 |
|              0.25 | optimal   |              0.0133 |   334.07   |       0.0851 |                 0.2361 |
|              0.15 | bootstrap |              0.0315 |   116.293  |       0.8892 |                 0.2188 |
|              0.15 | optimal   |              0.0123 |   339.868  |       0.074  |                 0.2188 |
|              0.1  | bootstrap |              0.0364 |    81.3175 |       1.2776 |                 0.2114 |
|              0.1  | optimal   |              0.0119 |   343.872  |       0.0511 |                 0.2114 |
|              0.05 | bootstrap |              0.051  |    42.0492 |       1.4592 |                 0.2062 |
|              0.05 | optimal   |              0.0117 |   347.331  |       0.0345 |                 0.2062 |

The optimal proposal can often match bootstrap accuracy with fewer particles.

**Particle-count sensitivity**

|   Particles | Method    |   PF RMSE vs Kalman |   Mean ESS |   Loglike sd |
|------------:|:----------|--------------------:|-----------:|-------------:|
|         100 | bootstrap |              0.0637 |    24.5238 |       1.6359 |
|         100 | optimal   |              0.0224 |    98.5483 |       0.1097 |
|         250 | bootstrap |              0.0375 |    60.757  |       1.1285 |
|         250 | optimal   |              0.0142 |   246.333  |       0.0547 |
|         500 | bootstrap |              0.0276 |   121.579  |       0.6529 |
|         500 | optimal   |              0.0099 |   492.635  |       0.0406 |
|        1000 | bootstrap |              0.02   |   244.002  |       0.7239 |
|        1000 | optimal   |              0.0069 |   985.287  |       0.0324 |

With 500 particles, the bootstrap filter has RMSE 0.0273 relative to the Kalman filtered mean, while the conditionally optimal filter has RMSE 0.0100. The main difference is not the model; it is the proposal distribution used to place particles. The measurement-noise sweep shows the mechanism: as observations become sharper, bootstrap particles drawn before seeing the signal are more likely to receive near-zero weight.

## Takeaway

Particle filters are flexible because they replace analytic filtering distributions with weighted simulations. That flexibility has a cost: particle placement matters. When observations are very informative or contaminated by outliers, naive bootstrap particles can collapse onto a few high-weight draws. Better proposals, more particles, and outlier-robust measurement models are practical responses, but the first warning usually appears in ESS and repeated-run Monte Carlo error.

## References

- [Gordon, N. J., Salmond, D. J., and Smith, A. F. M. (1993). Novel Approach to Nonlinear/non-Gaussian Bayesian State Estimation. *IEE Proceedings F*, 140(2), 107-113.](https://doi.org/10.1049/ip-f-2.1993.0015)
- [Doucet, A., de Freitas, N., and Gordon, N. (eds.) (2001). *Sequential Monte Carlo Methods in Practice*. Springer.](https://doi.org/10.1007/978-1-4757-3437-9)
