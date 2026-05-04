#!/usr/bin/env python3
"""Kalman filtering for a linear Gaussian state-space model.

The tutorial uses a two-state signal extraction model. The hidden state is
observed only through a noisy scalar signal, so the Kalman filter shows how
prediction, updating, uncertainty, and likelihood evaluation fit together.
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
        "Kalman Filtering Hidden States",
        "Recursive signal extraction and likelihood evaluation in a linear Gaussian model.",
    )

    report.add_overview(
        "The Kalman filter is the workhorse algorithm for tracking hidden states from noisy "
        "measurements. It appears in macroeconomics, engineering, robotics, finance, and any "
        "setting where a latent system evolves over time and observations arrive sequentially.\n\n"
        "This tutorial uses a two-state linear Gaussian model. Each period has two steps: "
        "predict the hidden state from the transition equation, then update that prediction "
        "using the new observation. The same recursion also produces the likelihood."
    )

    report.add_equations(
        r"""
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
"""
    )

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        "| Observation matrix $\\Psi$ | [1.0, 0.9] |\n"
        "| Transition matrix $\\Phi$ | diag(0.4, 0.5) |\n"
        f"| Measurement std | {MEASUREMENT_STD:.2f} |\n"
        f"| Process std | ({PROCESS_STD[0]:.2f}, {PROCESS_STD[1]:.2f}) |\n"
        f"| Periods | {n_periods} |\n"
        "| Initial state | $s_0 = (0,0)$ |"
    )

    report.add_solution_method(
        "The code simulates the hidden state and observed signal, then runs the Kalman filter "
        "with an initial state known to be zero. At each date it stores the one-step-ahead "
        "prediction, the filtered state mean, posterior covariance, Kalman gain, innovation, "
        "and log likelihood increment.\n\n"
        "The plots are meant to make the recursion concrete: the data are noisy, the filtered "
        "states are smoother than the raw observation, and uncertainty bands narrow when the "
        "signal is informative."
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
    axes1[0].set_title("Noisy Observations of a Hidden Two-State System")
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
            "The scalar observation combines both hidden states and measurement error. The filter "
            "uses the transition law to separate persistent state movements from observation noise."
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
    axes2[0].set_title("Filtered Means and 90% Credible Bands")
    axes2[-1].set_xlabel("Period")
    report.add_figure(
        "figures/filter-bands.png",
        "Kalman filtered states with credible bands",
        fig2,
        description=(
            "The posterior covariance is not a side product. It tells us how uncertain the "
            "filter is about each latent state after seeing data through period t."
        ),
    )

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.5))
    axes3[0].plot(time, filtered["innovations"], color="tab:orange", marker="o", markersize=3.0)
    axes3[0].axhline(0.0, color="black", linewidth=0.8)
    axes3[0].set_xlabel("Period")
    axes3[0].set_ylabel("Innovation")
    axes3[0].set_title("One-Step Forecast Errors")
    axes3[1].plot(time, filtered["gains"][:, 0], label="gain on s1")
    axes3[1].plot(time, filtered["gains"][:, 1], label="gain on s2")
    axes3[1].set_xlabel("Period")
    axes3[1].set_ylabel("Kalman gain")
    axes3[1].set_title("Kalman Gain Stabilizes")
    axes3[1].legend()
    fig3.tight_layout()
    report.add_figure(
        "figures/innovations-gain.png",
        "Forecast innovations and Kalman gains",
        fig3,
        description=(
            "The innovation is the surprise in the new observation. The Kalman gain converts "
            "that surprise into a state update, with weights pinned down by signal and state noise."
        ),
    )

    report.add_table(
        "tables/filter-diagnostics.csv",
        "Filter diagnostics",
        table,
        description="The table compares filtered state means to the simulated hidden states.",
    )

    report.add_results(
        f"The total log likelihood for the simulated sample is "
        f"{filtered['loglike_increment'].sum():.2f}. The filter tracks both states well despite "
        "observing only one noisy scalar signal because the transition equation supplies dynamic "
        "discipline."
    )

    report.add_takeaway(
        "The Kalman filter is more than a smoother. It is a disciplined accounting system for "
        "uncertainty: prior state uncertainty, measurement noise, forecast surprises, posterior "
        "uncertainty, and likelihood all update together. That is why the same recursion is used "
        "for forecasting, nowcasting, state estimation, and maximum-likelihood estimation."
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
