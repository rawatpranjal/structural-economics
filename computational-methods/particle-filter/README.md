# Nowcasting Hidden Economic States with Particle Filters

> Sequential Monte Carlo for noisy signal extraction and latent-state nowcasting.

## Overview

An economist often observes a noisy indicator, such as an activity index, inflation signal, or demand measure, and wants to infer the state that generated it. In a linear Gaussian state-space model, the Kalman filter gives that filtered state distribution exactly. Many structural and empirical models add nonlinear transitions, discrete regime changes, or non-Gaussian shocks. The hidden state still matters for nowcasts, likelihood evaluation, and counterfactual paths, but the closed-form recursion is gone.

This tutorial uses a two-state signal-extraction model. We keep the model linear and Gaussian so the Kalman filter gives a clean benchmark, then solve the same filtering problem with sequential Monte Carlo. A particle filter carries many simulated candidates for the hidden state, weights them by how well they explain the new observation, and resamples the useful candidates. The comparison shows where the computation can fail: if particles are drawn before seeing an informative signal, most receive tiny weights and Monte Carlo error rises.

## Equations

Let $s_t$ collect latent economic states, such as persistent activity and demand
pressure, and let $y_t$ be the observed signal. The state-space model is:

$$
y_t = \Psi s_t + u_t, \qquad s_t = \Phi s_{t-1} + \epsilon_t.
$$

After observing data through date $t$, the object of interest is the filtered
distribution $p(s_t \mid y_{1:t})$ or a moment such as $E[s_t \mid y_{1:t}]$.
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
| Hidden state $s_t$ | Two persistent economic components |
| Observed signal $y_t$ | Noisy linear indicator of the state |
| Observation matrix $\Psi$ | [1.0, 0.9] |
| Transition matrix $\Phi$ | diag(0.4, 0.5) |
| Measurement std | 0.10 |
| Process std | (0.30, 0.25) |
| Baseline particles | 500 |
| Repeated runs | 50 |
| Benchmark | Kalman filtered mean |

## Solution Method

The code compares two ways to place particles before computing the filtered mean. The bootstrap filter draws each candidate state from the transition equation and then asks whether that draw explains the observed signal. The conditionally optimal filter uses the same model but draws from $p(s_t \mid s_{t-1}, y_t)$, so the new signal helps place particles before weights are assigned.

```text
Algorithm: particle filtering with resampling
Input: observations y_t, particles s_0^(i), proposal q, particle count N
Output: filtered state means, ESS, likelihood estimate
for t = 1, ..., T:
    for each particle i:
        draw proposed state s_t^(i) from q(s_t | s_{t-1}^(i), y_t)
        compute importance weight w_t^{(i)} from target / proposal density
    normalize weights and estimate E[s_t | y_{1:t}]
    compute ESS_t = 1 / sum_i (w_t^{(i)})^2
    resample particles according to normalized weights
    accumulate the likelihood increment
```

The experiment repeats each filter many times and reports Monte Carlo error relative to the Kalman filtered mean. The benchmark is exact in this linear Gaussian model, so the tables measure particle approximation error rather than model misspecification.

## Results

With 500 particles, both particle filters track the latent state estimated by the Kalman benchmark. The main differences appear in repeated-run error and effective sample size.

<img src="figures/filter-comparison.png" alt="Particle filter state estimates compared with the Kalman filter" width="80%">

Effective sample size falls when a few particles receive most of the weight. The conditionally optimal proposal retains more useful particles because it looks at the signal before drawing the new state.

<img src="figures/mse-and-ess.png" alt="Repeated-run Monte Carlo error and effective sample size" width="80%">

Low measurement noise makes the likelihood sharply peaked. Bootstrap particles drawn from the transition can miss that peak, so the filtered estimate depends on a small set of high-weight draws.

<img src="figures/measurement-noise.png" alt="Particle accuracy as measurement noise falls" width="80%">

This stress test treats period 25 as an extreme data release. Likelihood weighting can concentrate particles on that observation and pull the filtered state sharply.

<img src="figures/outlier-stress.png" alt="Filtering after multiplying observation 25 by ten" width="80%">

The baseline repeats each filter 50 times with 500 particles and compares the filtered state mean with the Kalman benchmark.

**Baseline repeated-run comparison**

| Method    |   Particles |   PF RMSE vs Kalman |   Mean ESS |   Loglike sd |
|:----------|------------:|--------------------:|-----------:|-------------:|
| bootstrap |         500 |              0.0273 |    121.829 |       0.6914 |
| optimal   |         500 |              0.01   |    492.636 |       0.0397 |

Lower measurement noise makes the observed signal more informative.

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

Using the signal inside the proposal often matches bootstrap accuracy with fewer particles.

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

With 500 particles, the bootstrap filter has RMSE 0.0273 relative to the Kalman filtered mean, while the conditionally optimal filter has RMSE 0.0100. Both filters target the same latent economic state. The gap comes from the proposal distribution used to place particles before the weights are normalized. The measurement-noise sweep shows the mechanism: as observations become more informative, bootstrap particles drawn before seeing the signal are more likely to receive near-zero weight.

## Takeaway

Particle filters let economists carry latent-state models beyond the linear Gaussian case by replacing analytic recursions with weighted simulations. The method is only as useful as the particle cloud it maintains. Informative signals and extreme observations expose weak proposal distributions, so ESS, repeated-run error, and likelihood variability should be read before filtered states enter a structural estimate or policy exercise.

## References

- [Gordon, N. J., Salmond, D. J., and Smith, A. F. M. (1993). Novel Approach to Nonlinear/non-Gaussian Bayesian State Estimation. *IEE Proceedings F*, 140(2), 107-113.](https://doi.org/10.1049/ip-f-2.1993.0015)
- [Doucet, A., de Freitas, N., and Gordon, N. (eds.) (2001). *Sequential Monte Carlo Methods in Practice*. Springer.](https://doi.org/10.1007/978-1-4757-3437-9)
