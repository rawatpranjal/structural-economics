# Kalman Filtering a Latent Economic State

> Recursive signal extraction, uncertainty, and likelihood in a linear Gaussian model.

## Overview

Many economic states are not directly observed. A policymaker may see noisy indicators of the business cycle, an econometrician may observe prices but not latent demand, and a forecaster may want the persistent component of a volatile series. The Kalman filter is the canonical linear-Gaussian answer to that signal-extraction problem.

This tutorial uses a two-state model observed through one noisy scalar signal. Each period has two economically distinct steps: predict the latent state using the law of motion, then update that prediction using the surprise in the new observation. The same recursion gives filtered states, posterior uncertainty, forecast innovations, Kalman gains, and the likelihood.

## Equations

The state-space model is:

$$
y_t = \Psi s_t + u_t, \qquad u_t \sim N(0, R).
$$

$$
s_t = \Phi s_{t-1} + \epsilon_t, \qquad \epsilon_t \sim N(0, Q).
$$

The Kalman prediction and update are:

$$
\begin{aligned}
\hat{s}_{t|t-1} &= \Phi \hat{s}_{t-1|t-1}, \\
P_{t|t-1} &= \Phi P_{t-1|t-1}\Phi' + Q, \\
K_t &= P_{t|t-1}\Psi'(\Psi P_{t|t-1}\Psi' + R)^{-1}, \\
\hat{s}_{t|t} &= \hat{s}_{t|t-1} + K_t(y_t - \Psi\hat{s}_{t|t-1}).
\end{aligned}
$$

The innovation $\nu_t=y_t-\Psi\hat{s}_{t|t-1}$ has variance
$S_t=\Psi P_{t|t-1}\Psi' + R$, so the likelihood contribution is the Gaussian
density of $\nu_t$ under variance $S_t$.

## Model Setup

| Object | Value |
|--------|-------|
| Observation matrix $\Psi$ | [1.0, 0.9] |
| Transition matrix $\Phi$ | diag(0.4, 0.5) |
| Measurement std | 0.10 |
| Process std | (0.30, 0.25) |
| Periods | 50 |
| Initial state | $s_0 = (0,0)$ |

## Solution Method

The code simulates the hidden state and observed signal, then runs the Kalman filter from an initial state known to be zero. At each date it stores the one-step-ahead prediction, the filtered state mean, posterior covariance, Kalman gain, innovation, and log likelihood increment.

```text
Algorithm: Kalman filtering in a linear Gaussian state-space model
Input: observations y_t, transition Phi, loading Psi, covariances Q and R
Output: filtered means, filtered covariances, innovations, likelihood
Initialize s_hat_{0|0} and P_{0|0}
for t = 1, ..., T:
    predict state:      s_hat_{t|t-1} = Phi s_hat_{t-1|t-1}
    predict covariance: P_{t|t-1} = Phi P_{t-1|t-1} Phi' + Q
    innovation:         nu_t = y_t - Psi s_hat_{t|t-1}
    innovation var:     S_t = Psi P_{t|t-1} Psi' + R
    gain:               K_t = P_{t|t-1} Psi' S_t^{-1}
    update state:       s_hat_{t|t} = s_hat_{t|t-1} + K_t nu_t
    update covariance:  P_{t|t} = P_{t|t-1} - K_t Psi P_{t|t-1}
    add log p(nu_t; 0, S_t) to the likelihood
```

The figures make the recursion concrete: the raw observation is noisy, the filtered states are smoother than the signal, and uncertainty changes with the information in the state equation and measurement equation.

## Results

The scalar observation combines both hidden states and measurement error. The filter uses the transition law to separate persistent state movements from observation noise.

<img src="figures/simulated-signal.png" alt="Observed signal and hidden state paths" width="80%">

The posterior covariance is not a side product. It tells us how uncertain the filter is about each latent state after seeing data through period t.

<img src="figures/filter-bands.png" alt="Kalman filtered states with credible bands" width="80%">

The innovation is the surprise in the new observation. The Kalman gain converts that surprise into a state update, with weights pinned down by signal and state noise.

<img src="figures/innovations-gain.png" alt="Forecast innovations and Kalman gains" width="80%">

The table compares filtered state means to the simulated hidden states.

**Filter diagnostics**

| State          | RMSE   | Mean abs error   | Mean posterior std   |   90% band coverage |
|:---------------|:-------|:-----------------|:---------------------|--------------------:|
| s1             | 0.2297 | 0.1754           | 0.2114               |                0.86 |
| s2             | 0.2447 | 0.2106           | 0.2286               |                0.92 |
| log likelihood |        |                  |                      |              -25.73 |

The total log likelihood for the simulated sample is -25.73. The filter tracks both states well despite observing only one noisy scalar signal because the transition equation supplies dynamic discipline. The Kalman gain stabilizes once the filter learns the relative precision of the transition equation and the measurement equation.

## Takeaway

The Kalman filter is more than a smoother. It is a disciplined accounting system for uncertainty: prior state uncertainty, measurement noise, forecast surprises, posterior uncertainty, and likelihood update together. That is why the same recursion is useful for nowcasting, forecasting, latent-state estimation, and maximum-likelihood estimation of linear Gaussian models.

## References

- [Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35-45.](https://doi.org/10.1115/1.3662552)
- [Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press.](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)
