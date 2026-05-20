#!/usr/bin/env python3
"""Particle filtering for hidden economic states.

The tutorial compares simulation-based filters against the Kalman filter in a
small signal-extraction model. The Kalman answer is available, so particle
approximation error is visible without changing the economic state-space model.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


PSI = np.array([[1.0, 0.9]])
PHI = np.array([[0.4, 0.0], [0.0, 0.5]])
PROCESS_STD = np.array([0.3, 0.25])
Q = np.diag(PROCESS_STD**2)
STATE_DIM = 2


def simulate_state_space(
    n_periods: int,
    measurement_std: float,
    seed: int = 609,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate hidden states and observations."""
    rng = np.random.default_rng(seed)
    states = np.zeros((n_periods, STATE_DIM), dtype=float)
    observations = np.zeros(n_periods, dtype=float)
    previous_state = np.zeros(STATE_DIM, dtype=float)

    for t in range(n_periods):
        state = PHI @ previous_state + rng.multivariate_normal(np.zeros(STATE_DIM), Q)
        observation = float((PSI @ state).item() + rng.normal(scale=measurement_std))
        states[t] = state
        observations[t] = observation
        previous_state = state

    return states, observations


def normal_logpdf(value: np.ndarray | float, mean: np.ndarray | float, variance: float) -> np.ndarray:
    """Evaluate scalar normal log density for scalar or vector inputs."""
    return -0.5 * (np.log(2.0 * np.pi * variance) + (value - mean) ** 2 / variance)


def normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Return normalized weights and log mean exp of unnormalized log weights."""
    log_norm = float(logsumexp(log_weights))
    weights = np.exp(log_weights - log_norm)
    log_increment = log_norm - np.log(len(log_weights))
    return weights, float(log_increment)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling indices for normalized particle weights."""
    n_particles = len(weights)
    positions = (rng.uniform() + np.arange(n_particles)) / n_particles
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="right")


def kalman_filter(observations: np.ndarray, measurement_std: float) -> dict[str, np.ndarray]:
    """Kalman filter used as the linear-Gaussian benchmark."""
    y = np.asarray(observations, dtype=float)
    r = measurement_std**2
    n_periods = len(y)
    mean = np.zeros(STATE_DIM, dtype=float)
    cov = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    filtered_mean = np.zeros((n_periods, STATE_DIM), dtype=float)
    filtered_cov = np.zeros((n_periods, STATE_DIM, STATE_DIM), dtype=float)
    loglike_increment = np.zeros(n_periods, dtype=float)

    for t, observation in enumerate(y):
        pred_mean = PHI @ mean
        pred_cov = PHI @ cov @ PHI.T + Q
        pred_y = float((PSI @ pred_mean).item())
        pred_var = float((PSI @ pred_cov @ PSI.T).item() + r)
        innovation = observation - pred_y
        gain = (pred_cov @ PSI.T / pred_var).ravel()
        mean = pred_mean + gain * innovation
        cov = pred_cov - np.outer(gain, PSI @ pred_cov)
        cov = 0.5 * (cov + cov.T)
        filtered_mean[t] = mean
        filtered_cov[t] = cov
        loglike_increment[t] = normal_logpdf(observation, pred_y, pred_var)

    return {
        "filtered_mean": filtered_mean,
        "filtered_cov": filtered_cov,
        "loglike_increment": loglike_increment,
    }


def bootstrap_particle_filter(
    observations: np.ndarray,
    measurement_std: float,
    n_particles: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run a bootstrap particle filter with resampling every period."""
    rng = np.random.default_rng(seed)
    y = np.asarray(observations, dtype=float)
    r = measurement_std**2
    particles = np.zeros((n_particles, STATE_DIM), dtype=float)
    means = np.zeros((len(y), STATE_DIM), dtype=float)
    ess = np.zeros(len(y), dtype=float)
    loglike_increment = np.zeros(len(y), dtype=float)

    for t, observation in enumerate(y):
        noise = rng.multivariate_normal(np.zeros(STATE_DIM), Q, size=n_particles)
        particles = particles @ PHI.T + noise
        predicted_y = particles @ PSI.ravel()
        log_weights = normal_logpdf(observation, predicted_y, r)
        weights, loglike_increment[t] = normalize_log_weights(log_weights)
        means[t] = weights @ particles
        ess[t] = 1.0 / np.sum(weights**2)
        particles = particles[systematic_resample(weights, rng)]

    return {
        "mean": means,
        "ess": ess,
        "loglike_increment": loglike_increment,
    }


def optimal_particle_filter(
    observations: np.ndarray,
    measurement_std: float,
    n_particles: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Run the conditionally optimal particle filter for this linear Gaussian model."""
    rng = np.random.default_rng(seed)
    y = np.asarray(observations, dtype=float)
    r = measurement_std**2
    particles = np.zeros((n_particles, STATE_DIM), dtype=float)
    means = np.zeros((len(y), STATE_DIM), dtype=float)
    ess = np.zeros(len(y), dtype=float)
    loglike_increment = np.zeros(len(y), dtype=float)

    predictive_variance = float((PSI @ Q @ PSI.T).item() + r)
    proposal_gain = (Q @ PSI.T / predictive_variance).ravel()
    proposal_cov = Q - np.outer(proposal_gain, PSI @ Q)
    proposal_cov = 0.5 * (proposal_cov + proposal_cov.T)

    for t, observation in enumerate(y):
        prior_mean = particles @ PHI.T
        predictive_mean = prior_mean @ PSI.ravel()
        log_weights = normal_logpdf(observation, predictive_mean, predictive_variance)
        weights, loglike_increment[t] = normalize_log_weights(log_weights)
        proposal_mean = prior_mean + (observation - predictive_mean)[:, None] * proposal_gain
        particles_proposed = proposal_mean + rng.multivariate_normal(
            np.zeros(STATE_DIM),
            proposal_cov,
            size=n_particles,
        )
        means[t] = weights @ particles_proposed
        ess[t] = 1.0 / np.sum(weights**2)
        particles = particles_proposed[systematic_resample(weights, rng)]

    return {
        "mean": means,
        "ess": ess,
        "loglike_increment": loglike_increment,
    }


def repeated_filter_mse(
    observations: np.ndarray,
    kalman_mean: np.ndarray,
    measurement_std: float,
    n_particles: int,
    n_runs: int,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a particle filter many times and summarize MSE, log likelihood, and ESS."""
    mse_by_time = np.zeros((n_runs, len(observations)), dtype=float)
    loglikes = np.zeros(n_runs, dtype=float)
    mean_ess = np.zeros(n_runs, dtype=float)
    filter_fn = bootstrap_particle_filter if method == "bootstrap" else optimal_particle_filter

    for run in range(n_runs):
        result = filter_fn(
            observations,
            measurement_std=measurement_std,
            n_particles=n_particles,
            seed=seed + run,
        )
        error = result["mean"] - kalman_mean
        mse_by_time[run] = np.mean(error**2, axis=1)
        loglikes[run] = result["loglike_increment"].sum()
        mean_ess[run] = result["ess"].mean()

    return mse_by_time, loglikes, mean_ess


def measurement_noise_sweep(
    n_periods: int,
    n_particles: int,
    n_runs: int,
) -> pd.DataFrame:
    """Compare particle accuracy as observation noise shrinks."""
    rows = []
    for measurement_std in [0.25, 0.15, 0.10, 0.05]:
        states, observations = simulate_state_space(n_periods, measurement_std, seed=777)
        kalman = kalman_filter(observations, measurement_std)
        for method in ["bootstrap", "optimal"]:
            mse, loglikes, ess = repeated_filter_mse(
                observations,
                kalman["filtered_mean"],
                measurement_std,
                n_particles,
                n_runs,
                method,
                seed=9_000 + int(measurement_std * 1000),
            )
            state_rmse = float(
                np.sqrt(np.mean((kalman["filtered_mean"] - states) ** 2))
            )
            rows.append(
                {
                    "Measurement std": measurement_std,
                    "Method": method,
                    "PF RMSE vs Kalman": np.sqrt(np.mean(mse)),
                    "Mean ESS": np.mean(ess),
                    "Loglike sd": np.std(loglikes),
                    "Kalman RMSE vs truth": state_rmse,
                }
            )
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for readable Markdown tables."""
    out = df.copy()
    for col in out.columns:
        if col == "Method":
            continue
        if col == "Particles":
            out[col] = out[col].map(lambda x: f"{int(x)}")
        else:
            out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    return out


def main() -> None:
    setup_style()
    n_periods = 50
    measurement_std = 0.1
    n_particles = 500
    n_runs = 50
    states, observations = simulate_state_space(n_periods, measurement_std)
    kalman = kalman_filter(observations, measurement_std)
    bootstrap_one = bootstrap_particle_filter(observations, measurement_std, n_particles, seed=610)
    optimal_one = optimal_particle_filter(observations, measurement_std, n_particles, seed=611)
    mse_boot, log_boot, ess_boot = repeated_filter_mse(
        observations,
        kalman["filtered_mean"],
        measurement_std,
        n_particles,
        n_runs,
        "bootstrap",
        seed=1_000,
    )
    mse_opt, log_opt, ess_opt = repeated_filter_mse(
        observations,
        kalman["filtered_mean"],
        measurement_std,
        n_particles,
        n_runs,
        "optimal",
        seed=2_000,
    )
    measurement_table = measurement_noise_sweep(n_periods, n_particles=500, n_runs=50)
    time = np.arange(1, n_periods + 1)

    print("Particle filter tutorial")
    print(f"  particles={n_particles}, repeated runs={n_runs}")
    print(f"  bootstrap RMSE vs Kalman={np.sqrt(np.mean(mse_boot)):.4f}")
    print(f"  optimal RMSE vs Kalman={np.sqrt(np.mean(mse_opt)):.4f}")

    fig1, axes1 = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for dim, ax in enumerate(axes1):
        ax.plot(time, kalman["filtered_mean"][:, dim], color="black", label="Kalman")
        ax.plot(time, bootstrap_one["mean"][:, dim], color="tab:blue", alpha=0.9, label="bootstrap PF")
        ax.plot(time, optimal_one["mean"][:, dim], color="tab:orange", alpha=0.9, label="optimal PF")
        ax.set_ylabel(f"s{dim + 1}")
        ax.legend(loc="upper right")
    axes1[0].set_title("Hidden-State Estimates Against Kalman Benchmark")
    axes1[-1].set_xlabel("Period")
    save_figure(fig1, "figures/filter-comparison.png", dpi=150)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.6))
    axes2[0].plot(time, mse_boot.mean(axis=0), label="bootstrap PF")
    axes2[0].plot(time, mse_opt.mean(axis=0), label="optimal PF")
    axes2[0].set_xlabel("Period")
    axes2[0].set_ylabel("Mean squared error")
    axes2[0].set_title("Monte Carlo Error in Filtered Means")
    axes2[0].legend()
    axes2[1].plot(time, bootstrap_one["ess"], label="bootstrap PF")
    axes2[1].plot(time, optimal_one["ess"], label="optimal PF")
    axes2[1].axhline(n_particles / 2.0, color="black", linestyle=":", linewidth=1.0)
    axes2[1].set_xlabel("Period")
    axes2[1].set_ylabel("Effective sample size")
    axes2[1].set_title("Effective Sample Size by Period")
    axes2[1].legend()
    fig2.tight_layout()
    save_figure(fig2, "figures/mse-and-ess.png", dpi=150)

    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4.6))
    for method, group in measurement_table.groupby("Method"):
        axes3[0].plot(
            group["Measurement std"],
            group["PF RMSE vs Kalman"],
            marker="o",
            label=method,
        )
        axes3[1].plot(group["Measurement std"], group["Mean ESS"], marker="o", label=method)
    axes3[0].invert_xaxis()
    axes3[1].invert_xaxis()
    axes3[0].set_xlabel("Measurement std")
    axes3[0].set_ylabel("PF RMSE vs Kalman")
    axes3[0].set_title("Informative Signals Raise Bootstrap Error")
    axes3[1].set_xlabel("Measurement std")
    axes3[1].set_ylabel("Mean ESS")
    axes3[1].set_title("Informative Signals Lower ESS")
    axes3[1].legend()
    fig3.tight_layout()
    save_figure(fig3, "figures/measurement-noise.png", dpi=150)

    summary_rows = [
        {
            "Method": "bootstrap",
            "Particles": n_particles,
            "PF RMSE vs Kalman": np.sqrt(np.mean(mse_boot)),
            "Mean ESS": np.mean(ess_boot),
            "Loglike sd": np.std(log_boot),
        },
        {
            "Method": "optimal",
            "Particles": n_particles,
            "PF RMSE vs Kalman": np.sqrt(np.mean(mse_opt)),
            "Mean ESS": np.mean(ess_opt),
            "Loglike sd": np.std(log_opt),
        },
    ]
    Path("tables").mkdir(parents=True, exist_ok=True)
    format_table(pd.DataFrame(summary_rows)).to_csv("tables/filter-summary.csv", index=False)
    format_table(measurement_table).to_csv("tables/measurement-noise-sweep.csv", index=False)

    save_thumbnail("figures/filter-comparison.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
