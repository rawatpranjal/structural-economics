# Nowcasting Hidden Economic States by Particle Filtering

## Overview

A policy analyst observes a noisy activity indicator and wants a current estimate of the hidden state. The state may combine persistent demand pressure and real activity.

The object is the filtered distribution $p(s_t \mid y_{1:t})$. Its mean is the nowcast used in later likelihood or policy calculations.

A Kalman filter is exact in this linear Gaussian example. We use it as a benchmark for a particle filter. The particle filter is needed when analytic filtering is unavailable.

## Equations

Let $s_t$ collect two latent economic states, and let $y_t$ be the observed
signal. The state-space model is:

$$
y_t = \Psi s_t + u_t, \qquad s_t = \Phi s_{t-1} + \epsilon_t.
$$

Here $u_t$ is measurement noise and $\epsilon_t$ is process noise.

Particles approximate the filtered distribution with weighted simulated states.
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

The bootstrap filter first draws each particle from the state transition. It then weights the draw by the signal likelihood.

The optimal proposal in this example uses the signal before drawing. That timing keeps particles near states that can explain the observation.

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

We repeat each filter and compare its mean with the Kalman mean. ESS records how many particles carry meaningful weight.

## Results

Both filters track the Kalman mean in the baseline run. The visible paths look similar, so repeated-run diagnostics are needed.

<img src="figures/filter-comparison.png" alt="Particle filter state estimates compared with the Kalman filter" width="80%">

ESS falls when weights concentrate on a few particles. The optimal proposal keeps ESS close to the particle count.

<img src="figures/mse-and-ess.png" alt="Repeated-run Monte Carlo error and effective sample size" width="80%">

Sharper signals reduce bootstrap ESS and raise error. The optimal proposal is less sensitive because it conditions on the signal.

<img src="figures/measurement-noise.png" alt="Particle accuracy as measurement noise falls" width="80%">

The baseline repeats each filter 50 times with 500 particles. It compares each particle mean with the Kalman mean.

**Baseline repeated-run comparison**

| Method    |   Particles |   PF RMSE vs Kalman |   Mean ESS |   Loglike sd |
|:----------|------------:|--------------------:|-----------:|-------------:|
| bootstrap |         500 |              0.0273 |    121.829 |       0.6914 |
| optimal   |         500 |              0.01   |    492.636 |       0.0397 |

Lower measurement noise makes the signal more informative. That setting reveals weight collapse in the bootstrap filter.

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

With 500 particles, bootstrap RMSE is 0.0273. The optimal proposal lowers RMSE to 0.0100. The tables show the reason. Bootstrap ESS falls when the signal is sharp.

## Takeaway

Particle filters nowcast hidden economic states with weighted simulations. The main diagnostic is whether the weights collapse. Use ESS and repeated-run error before treating filtered states as inputs to estimation.

## References

- [Gordon, N. J., Salmond, D. J., and Smith, A. F. M. (1993). Novel Approach to Nonlinear/non-Gaussian Bayesian State Estimation. *IEE Proceedings F*, 140(2), 107-113.](https://doi.org/10.1049/ip-f-2.1993.0015)
- [Doucet, A., de Freitas, N., and Gordon, N. (eds.) (2001). *Sequential Monte Carlo Methods in Practice*. Springer.](https://doi.org/10.1007/978-1-4757-3437-9)
