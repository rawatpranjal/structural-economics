# Kalman Filtering Hidden States

> Recursive signal extraction and likelihood evaluation in a linear Gaussian model.

## Overview

The Kalman filter is the workhorse algorithm for tracking hidden states from noisy measurements. It appears in macroeconomics, engineering, robotics, finance, and any setting where a latent system evolves over time and observations arrive sequentially.

This tutorial uses a two-state linear Gaussian model. Each period has two steps: predict the hidden state from the transition equation, then update that prediction using the new observation. The same recursion also produces the likelihood.

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

The code simulates the hidden state and observed signal, then runs the Kalman filter with an initial state known to be zero. At each date it stores the one-step-ahead prediction, the filtered state mean, posterior covariance, Kalman gain, innovation, and log likelihood increment.

The plots are meant to make the recursion concrete: the data are noisy, the filtered states are smoother than the raw observation, and uncertainty bands narrow when the signal is informative.

## Results

The scalar observation combines both hidden states and measurement error. The filter uses the transition law to separate persistent state movements from observation noise.

![Observed signal and hidden state paths](figures/simulated-signal.png)
*Observed signal and hidden state paths*

The posterior covariance is not a side product. It tells us how uncertain the filter is about each latent state after seeing data through period t.

![Kalman filtered states with credible bands](figures/filter-bands.png)
*Kalman filtered states with credible bands*

The innovation is the surprise in the new observation. The Kalman gain converts that surprise into a state update, with weights pinned down by signal and state noise.

![Forecast innovations and Kalman gains](figures/innovations-gain.png)
*Forecast innovations and Kalman gains*

The table compares filtered state means to the simulated hidden states.

**Filter diagnostics**

| State          | RMSE   | Mean abs error   | Mean posterior std   |   90% band coverage |
|:---------------|:-------|:-----------------|:---------------------|--------------------:|
| s1             | 0.2297 | 0.1754           | 0.2114               |                0.86 |
| s2             | 0.2447 | 0.2106           | 0.2286               |                0.92 |
| log likelihood |        |                  |                      |              -25.73 |

The total log likelihood for the simulated sample is -25.73. The filter tracks both states well despite observing only one noisy scalar signal because the transition equation supplies dynamic discipline.

## Economic Takeaway

The Kalman filter is more than a smoother. It is a disciplined accounting system for uncertainty: prior state uncertainty, measurement noise, forecast surprises, posterior uncertainty, and likelihood all update together. That is why the same recursion is used for forecasting, nowcasting, state estimation, and maximum-likelihood estimation.

## Reproduce

```bash
python run.py
```

## References

- [Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35-45.](https://doi.org/10.1115/1.3662552)
- [Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press.](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)
