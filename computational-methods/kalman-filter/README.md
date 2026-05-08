# Nowcasting a Latent Business-Cycle State

> Kalman filtering turns noisy economic indicators into state estimates, uncertainty bands, and a likelihood.

## Overview

Suppose a policy team tracks current economic activity before the full national accounts arrive. It sees a noisy indicator, perhaps a survey or high-frequency spending series, but the object of interest is the latent state that the indicator only partly reveals. The filter below treats that object as a state vector, lets it evolve with a simple law of motion, and asks how each new signal should change the nowcast.

That task creates a numerical object: the distribution of the hidden state conditional on the observations seen so far. In a linear Gaussian model, the Kalman filter carries that distribution with two objects, a mean and a covariance. The same recursion supplies the forecast error, the weight placed on the new signal, and the likelihood used when the state-space parameters are estimated rather than fixed.

## Equations

Let $s_t$ collect two latent components of economic activity. The researcher
observes a scalar indicator $y_t$ that loads on both components and adds
measurement noise:

$$
y_t = \Psi s_t + u_t, \qquad u_t \sim N(0, R).
$$

The hidden state follows a linear transition equation:

$$
s_t = \Phi s_{t-1} + \epsilon_t, \qquad \epsilon_t \sim N(0, Q).
$$

Given data through $t-1$, the filter predicts the next state and its covariance:

$$
\begin{aligned}
\hat{s}_{t|t-1} &= \Phi \hat{s}_{t-1|t-1}, \\
P_{t|t-1} &= \Phi P_{t-1|t-1}\Phi' + Q.
\end{aligned}
$$

The new signal produces a forecast surprise, a signal variance, and a Kalman
gain:

$$
\begin{aligned}
\nu_t &= y_t - \Psi\hat{s}_{t|t-1}, \\
S_t &= \Psi P_{t|t-1}\Psi' + R, \\
K_t &= P_{t|t-1}\Psi'(\Psi P_{t|t-1}\Psi' + R)^{-1}, \\
\hat{s}_{t|t} &= \hat{s}_{t|t-1} + K_t\nu_t, \\
P_{t|t} &= P_{t|t-1} - K_t\Psi P_{t|t-1}.
\end{aligned}
$$

The likelihood contribution is the Gaussian density of $\nu_t$ under variance
$S_t$. The same scalar density is what maximum-likelihood estimation uses when
the state-space parameters are unknown.

## Model Setup

| Object | Value |
|--------|-------|
| Latent state $s_t=(s_{1t}, s_{2t})$ | two activity components |
| Observed signal $y_t$ | noisy indicator of current activity |
| Loading matrix $\Psi$ | [1.0, 0.9] |
| Transition matrix $\Phi$ | diag(0.4, 0.5) |
| Measurement std | 0.10 |
| Process std | (0.30, 0.25) |
| Periods | 50 |
| Initial state | $s_0 = (0,0)$ |

## Solution Method

The simulation draws the true latent path and the noisy observed indicator. The filter starts from zero, makes a one-period forecast, compares the forecasted indicator with observed $y_t$, and moves the state estimate toward the surprise. The Kalman gain is larger when the signal is precise relative to prior state uncertainty, and smaller when measurement noise is high.

```text
Algorithm: nowcasting a latent state with the Kalman filter
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

The stored arrays let the report connect the economic object to the computation: the raw indicator is noisy, the filtered states summarize current conditions, and posterior uncertainty records how much the data have pinned down the hidden state.

## Results

The observed indicator combines both hidden activity components and measurement error. The filter uses the transition law to separate persistent state movements from observation noise.

<img src="figures/simulated-signal.png" alt="Observed signal and hidden state paths" width="80%">

The posterior covariance reports how uncertain the nowcast is about each latent component after seeing data through period t.

<img src="figures/filter-bands.png" alt="Kalman filtered states with credible bands" width="80%">

Forecast errors drive the updates. After a short initialization phase, the Kalman gain settles near constants set by signal noise and state noise.

<img src="figures/innovations-gain.png" alt="Forecast innovations and Kalman gains" width="80%">

The table compares filtered state means with the simulated hidden states.

**Filter diagnostics**

| State          | RMSE   | Mean abs error   | Mean posterior std   |   90% band coverage |
|:---------------|:-------|:-----------------|:---------------------|--------------------:|
| s1             | 0.2297 | 0.1754           | 0.2114               |                0.86 |
| s2             | 0.2447 | 0.2106           | 0.2286               |                0.92 |
| log likelihood |        |                  |                      |              -25.73 |

The total log likelihood for the simulated sample is -25.73. The estimated path follows the two hidden components even though the researcher observes only one noisy scalar indicator. That is the economic payoff: a noisy data release becomes a current estimate of the latent state, with uncertainty attached. The computation matters because the covariance and gain decide how much the new data should move the nowcast.

## Takeaway

When an economic object is hidden, smoothing the raw series is not enough. A state-space model states what can move over time and how noisy the indicators are. The Kalman filter then computes the conditional state distribution one observation at a time, which makes the same machinery useful for nowcasting, forecasting, and maximum-likelihood estimation of linear Gaussian models.

## References

- [Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35-45.](https://doi.org/10.1115/1.3662552)
- [Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press.](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)
