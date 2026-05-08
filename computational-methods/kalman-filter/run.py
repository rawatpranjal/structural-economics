#!/usr/bin/env python3
"""Kalman filtering for a latent economic indicator.

A policymaker observes a noisy activity signal and wants the hidden state that
moves with the economy. The tutorial shows how recursive prediction, updating,
uncertainty, and likelihood evaluation work in a linear Gaussian state-space
model.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


PSI = np.array([[1.0, 0.9]])
PHI = np.array([[0.4, 0.0], [0.0, 0.5]])
MEASUREMENT_STD = 0.1
PROCESS_STD = np.array([0.3, 0.25])
R = np.array([[MEASUREMENT_STD**2]])
Q = np.diag(PROCESS_STD**2)


def simulate_state_space(
    n_periods: int,
    seed: int = 609,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate hidden states and observations from the state-space model."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n_periods, 2), dtype=float)
    observations = np.zeros(n_periods, dtype=float)
    previous_state = np.zeros(2, dtype=float)

    for t in range(n_periods):
        state = PHI @ previous_state + rng.multivariate_normal(np.zeros(2), Q)
        observation = float((PSI @ state).item() + rng.normal(scale=MEASUREMENT_STD))
        states[t] = state
        observations[t] = observation
        previous_state = state

    return states, observations


def normal_logpdf(value: float, mean: float, variance: float) -> float:
    """Evaluate a scalar normal log density."""
    return float(-0.5 * (np.log(2.0 * np.pi * variance) + (value - mean) ** 2 / variance))


def kalman_filter(
    observations: np.ndarray,
    initial_mean: np.ndarray | None = None,
    initial_cov: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the Kalman filter and return prediction, filtering, and likelihood arrays."""
    y = np.asarray(observations, dtype=float)
    n_periods = len(y)
    state_dim = PHI.shape[0]

    filtered_mean = np.zeros((n_periods, state_dim), dtype=float)
    predicted_mean = np.zeros((n_periods, state_dim), dtype=float)
    filtered_cov = np.zeros((n_periods, state_dim, state_dim), dtype=float)
    predicted_cov = np.zeros((n_periods, state_dim, state_dim), dtype=float)
    gains = np.zeros((n_periods, state_dim), dtype=float)
    innovations = np.zeros(n_periods, dtype=float)
    innovation_variance = np.zeros(n_periods, dtype=float)
    loglike_increment = np.zeros(n_periods, dtype=float)

    mean = np.zeros(state_dim, dtype=float) if initial_mean is None else np.asarray(initial_mean, dtype=float)
    cov = np.zeros((state_dim, state_dim), dtype=float) if initial_cov is None else np.asarray(initial_cov, dtype=float)

    for t, observation in enumerate(y):
        pred_mean = PHI @ mean
        pred_cov = PHI @ cov @ PHI.T + Q
        pred_y = float((PSI @ pred_mean).item())
        pred_y_var = float((PSI @ pred_cov @ PSI.T).item() + R[0, 0])
        innovation = observation - pred_y
        gain = (pred_cov @ PSI.T / pred_y_var).ravel()

        mean = pred_mean + gain * innovation
        cov = pred_cov - np.outer(gain, PSI @ pred_cov)
        cov = 0.5 * (cov + cov.T)

        predicted_mean[t] = pred_mean
        predicted_cov[t] = pred_cov
        filtered_mean[t] = mean
        filtered_cov[t] = cov
        gains[t] = gain
        innovations[t] = innovation
        innovation_variance[t] = pred_y_var
        loglike_increment[t] = normal_logpdf(observation, pred_y, pred_y_var)

    return {
        "predicted_mean": predicted_mean,
        "predicted_cov": predicted_cov,
        "filtered_mean": filtered_mean,
        "filtered_cov": filtered_cov,
        "gains": gains,
        "innovations": innovations,
        "innovation_variance": innovation_variance,
        "loglike_increment": loglike_increment,
    }


def diagnostic_table(states: np.ndarray, filtered: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute filter accuracy and uncertainty diagnostics."""
    means = filtered["filtered_mean"]
    covs = filtered["filtered_cov"]
    rows = []
    z90 = 1.645
    for dim in range(states.shape[1]):
        error = means[:, dim] - states[:, dim]
        std = np.sqrt(covs[:, dim, dim])
        coverage = np.mean(np.abs(error) <= z90 * std)
        rows.append(
            {
                "State": f"s{dim + 1}",
                "RMSE": f"{np.sqrt(np.mean(error**2)):.4f}",
                "Mean abs error": f"{np.mean(np.abs(error)):.4f}",
                "Mean posterior std": f"{np.mean(std):.4f}",
                "90% band coverage": f"{coverage:.3f}",
            }
        )
    rows.append(
        {
            "State": "log likelihood",
            "RMSE": "",
            "Mean abs error": "",
            "Mean posterior std": "",
            "90% band coverage": f"{filtered['loglike_increment'].sum():.2f}",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    n_periods = 50
    states, observations = simulate_state_space(n_periods=n_periods)
    filtered = kalman_filter(observations)
    table = diagnostic_table(states, filtered)
    time = np.arange(1, n_periods + 1)

    print("Kalman filter state-space tutorial")
    print(f"  periods={n_periods}")
    print(f"  total log likelihood={filtered['loglike_increment'].sum():.2f}")
    print(
        "  state RMSE="
        f"{np.sqrt(np.mean((filtered['filtered_mean'] - states) ** 2, axis=0))[0]:.4f}, "
        f"{np.sqrt(np.mean((filtered['filtered_mean'] - states) ** 2, axis=0))[1]:.4f}"
    )

    report = ModelReport(
        "Nowcasting a Latent Business-Cycle State",
        "Kalman filtering turns noisy economic indicators into state estimates, uncertainty bands, and a likelihood.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A policy team tracks current activity before the full national accounts arrive. "
        "It sees a noisy indicator, such as a survey or spending series.\n\n"
        "The target is a latent business-cycle state. The indicator reveals part of that "
        "state, but it also includes measurement noise.\n\n"
        "The computational need is recursive inference. After each signal, the filter "
        "updates the nowcast, its covariance, and the likelihood."
    )

    report.add_equations(
        r"""
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
"""
    )

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        "| Latent state $s_t=(s_{1t}, s_{2t})$ | two activity components |\n"
        "| Observed signal $y_t$ | noisy indicator of current activity |\n"
        "| Loading matrix $\\Psi$ | [1.0, 0.9] |\n"
        "| Transition matrix $\\Phi$ | diag(0.4, 0.5) |\n"
        f"| Measurement std | {MEASUREMENT_STD:.2f} |\n"
        f"| Process std | ({PROCESS_STD[0]:.2f}, {PROCESS_STD[1]:.2f}) |\n"
        f"| Periods | {n_periods} |\n"
        "| Initial state | $s_0 = (0,0)$ |"
    )

    report.add_solution_method(
        "The simulation draws the true latent path and the noisy observed indicator. "
        "The filter starts from zero and makes a one-period forecast. "
        "It compares the forecasted indicator with observed $y_t$. "
        "The Kalman gain moves the state estimate toward that surprise.\n\n"
        "```text\n"
        "Algorithm: nowcasting a latent state with the Kalman filter\n"
        "Input: observations y_t, transition Phi, loading Psi, covariances Q and R\n"
        "Output: filtered means, filtered covariances, innovations, likelihood\n"
        "Initialize s_hat_{0|0} and P_{0|0}\n"
        "for t = 1, ..., T:\n"
        "    predict state:      s_hat_{t|t-1} = Phi s_hat_{t-1|t-1}\n"
        "    predict covariance: P_{t|t-1} = Phi P_{t-1|t-1} Phi' + Q\n"
        "    innovation:         nu_t = y_t - Psi s_hat_{t|t-1}\n"
        "    innovation var:     S_t = Psi P_{t|t-1} Psi' + R\n"
        "    gain:               K_t = P_{t|t-1} Psi' S_t^{-1}\n"
        "    update state:       s_hat_{t|t} = s_hat_{t|t-1} + K_t nu_t\n"
        "    update covariance:  P_{t|t} = P_{t|t-1} - K_t Psi P_{t|t-1}\n"
        "    add log p(nu_t; 0, S_t) to the likelihood\n"
        "```"
    )

    fig1, axes1 = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True)
    axes1[0].plot(time, observations, color="black", marker="o", markersize=3.0, label="observed y")
    axes1[0].plot(
        time,
        (PSI @ states.T).ravel(),
        color="tab:blue",
        linewidth=1.5,
        label="signal without measurement noise",
    )
    axes1[0].set_ylabel("Observation")
    axes1[0].set_title("Noisy Indicator of a Hidden Activity State")
    axes1[0].legend(loc="upper right")
    for dim, ax in enumerate(axes1[1:]):
        ax.plot(time, states[:, dim], color="black", label=f"true s{dim + 1}")
        ax.plot(time, filtered["filtered_mean"][:, dim], color="tab:blue", label=f"filtered s{dim + 1}")
        ax.set_ylabel(f"s{dim + 1}")
        ax.legend(loc="upper right")
    axes1[-1].set_xlabel("Period")
    report.add_figure(
        "figures/simulated-signal.png",
        "Observed signal and hidden state paths",
        fig1,
        description=(
            "The observed indicator mixes hidden activity with measurement error. "
            "The filter separates persistent state movements from noise."
        ),
    )

    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    z90 = 1.645
    for dim, ax in enumerate(axes2):
        mean = filtered["filtered_mean"][:, dim]
        std = np.sqrt(filtered["filtered_cov"][:, dim, dim])
        ax.fill_between(time, mean - z90 * std, mean + z90 * std, color="tab:blue", alpha=0.18)
        ax.plot(time, mean, color="tab:blue", label="filtered mean")
        ax.plot(time, states[:, dim], color="black", linestyle="--", label="true state")
        ax.set_ylabel(f"s{dim + 1}")
        ax.legend(loc="upper right")
    axes2[0].set_title("Filtered Activity States and 90% Credible Bands")
    axes2[-1].set_xlabel("Period")
    report.add_figure(
        "figures/filter-bands.png",
        "Kalman filtered states with credible bands",
        fig2,
        description=(
            "The posterior covariance shows uncertainty about each latent component "
            "after period t."
        ),
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.5))
    axes3[0].plot(time, filtered["innovations"], color="tab:orange", marker="o", markersize=3.0)
    axes3[0].axhline(0.0, color="black", linewidth=0.8)
    axes3[0].set_xlabel("Period")
    axes3[0].set_ylabel("Innovation")
    axes3[0].set_title("One-Period Forecast Errors")
    axes3[1].plot(time, filtered["gains"][:, 0], label="gain on s1")
    axes3[1].plot(time, filtered["gains"][:, 1], label="gain on s2")
    axes3[1].set_xlabel("Period")
    axes3[1].set_ylabel("Kalman gain")
    axes3[1].set_title("Kalman Gain Weights the New Signal")
    axes3[1].legend()
    fig3.tight_layout()
    report.add_figure(
        "figures/innovations-gain.png",
        "Forecast innovations and Kalman gains",
        fig3,
        description=(
            "Forecast errors drive the updates. "
            "The Kalman gain settles near constants set by signal and state noise."
        ),
    )

    report.add_table(
        "tables/filter-diagnostics.csv",
        "Filter diagnostics",
        table,
        description="The table compares filtered state means with the simulated hidden states.",
    )

    report.add_results(
        f"The total log likelihood for the simulated sample is "
        f"{filtered['loglike_increment'].sum():.2f}. The estimated path follows the two hidden "
        "components from one noisy scalar indicator. "
        "The covariance and gain decide how much each signal moves the nowcast."
    )

    report.add_takeaway(
        "When an economic state is hidden, smoothing the raw series is not enough. "
        "A state-space model says how the state moves and how noisy signals are. "
        "The Kalman filter updates the conditional state distribution one observation at a time."
    )

    report.add_references(
        [
            "[Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35-45.](https://doi.org/10.1115/1.3662552)",
            "[Durbin, J. and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press.](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)",
        ]
    )

    report.write("README.md")


if __name__ == "__main__":
    main()
