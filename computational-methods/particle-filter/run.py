#!/usr/bin/env python3
"""Particle filtering for a linear Gaussian state-space model.

The tutorial compares simulation-based filters against the Kalman filter in a
model where the Kalman answer is available. That makes particle approximation
error visible without changing the economic or statistical model.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


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


def particle_count_sweep(
    observations: np.ndarray,
    kalman_mean: np.ndarray,
    measurement_std: float,
    n_runs: int,
) -> pd.DataFrame:
    """Compare bootstrap and optimal filters over particle counts."""
    rows = []
    for n_particles in [100, 250, 500, 1_000]:
        for method in ["bootstrap", "optimal"]:
            mse, loglikes, ess = repeated_filter_mse(
                observations,
                kalman_mean,
                measurement_std,
                n_particles,
                n_runs,
                method,
                seed=12_000 + n_particles,
            )
            rows.append(
                {
                    "Particles": n_particles,
                    "Method": method,
                    "PF RMSE vs Kalman": np.sqrt(np.mean(mse)),
                    "Mean ESS": np.mean(ess),
                    "Loglike sd": np.std(loglikes),
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
    measurement_table = measurement_noise_sweep(n_periods, n_particles=350, n_runs=20)
    particle_table = particle_count_sweep(
        observations,
        kalman["filtered_mean"],
        measurement_std,
        n_runs=25,
    )
    outlier_observations = observations.copy()
    outlier_observations[24] *= 10.0
    outlier_kalman = kalman_filter(outlier_observations, measurement_std)
    outlier_boot = bootstrap_particle_filter(outlier_observations, measurement_std, n_particles, seed=800)
    outlier_opt = optimal_particle_filter(outlier_observations, measurement_std, n_particles, seed=801)
    time = np.arange(1, n_periods + 1)

    print("Particle filter tutorial")
    print(f"  particles={n_particles}, repeated runs={n_runs}")
    print(f"  bootstrap RMSE vs Kalman={np.sqrt(np.mean(mse_boot)):.4f}")
    print(f"  optimal RMSE vs Kalman={np.sqrt(np.mean(mse_opt)):.4f}")

    report = ModelReport(
        "Particle Filtering and Degeneracy",
        "Simulation-based filtering, particle degeneracy, and proposal design.",
    )

    report.add_overview(
        "Particle filters approximate a filtering distribution with simulated states. They are "
        "useful when nonlinearities or non-Gaussian shocks make the Kalman filter unavailable. "
        "Here we deliberately use a linear Gaussian model so the Kalman filter supplies a truth "
        "benchmark for particle error.\n\n"
        "The tutorial compares a bootstrap particle filter with a conditionally optimal particle "
        "filter. The bootstrap filter simulates from the transition equation and then weights by "
        "the observation density. The conditionally optimal filter uses the current observation "
        "inside the proposal, which reduces weight degeneracy when measurements are informative."
    )

    report.add_equations(
        r"""
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
"""
    )

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        "| Observation matrix $\\Psi$ | [1.0, 0.9] |\n"
        "| Transition matrix $\\Phi$ | diag(0.4, 0.5) |\n"
        f"| Measurement std | {measurement_std:.2f} |\n"
        f"| Process std | ({PROCESS_STD[0]:.2f}, {PROCESS_STD[1]:.2f}) |\n"
        f"| Baseline particles | {n_particles:,} |\n"
        f"| Repeated runs | {n_runs} |\n"
        "| Benchmark | Kalman filtered mean |"
    )

    report.add_solution_method(
        "**Bootstrap filter:** propagate particles with the transition equation, weight by "
        "the Gaussian observation density, estimate the state mean, then resample every period.\n\n"
        "**Conditionally optimal filter:** for each particle, combine the transition density "
        "and the current observation to sample from $p(s_t \\mid s_{t-1}, y_t)$. The remaining "
        "weights are the one-step predictive likelihoods $p(y_t \\mid s_{t-1})$.\n\n"
        "The code repeats each filter many times and reports Monte Carlo error relative to the "
        "Kalman filtered mean."
    )

    fig1, axes1 = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for dim, ax in enumerate(axes1):
        ax.plot(time, kalman["filtered_mean"][:, dim], color="black", label="Kalman")
        ax.plot(time, bootstrap_one["mean"][:, dim], color="tab:blue", alpha=0.9, label="bootstrap PF")
        ax.plot(time, optimal_one["mean"][:, dim], color="tab:orange", alpha=0.9, label="optimal PF")
        ax.set_ylabel(f"s{dim + 1}")
        ax.legend(loc="upper right")
    axes1[0].set_title("Particle Filter Means Against Kalman Benchmark")
    axes1[-1].set_xlabel("Period")
    report.add_figure(
        "figures/filter-comparison.png",
        "Particle filter state estimates compared with the Kalman filter",
        fig1,
        description=(
            "With 500 particles, both particle filters track the Kalman benchmark. The difference "
            "between the two methods is easier to see in repeated-run error and effective sample size."
        ),
    )

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.6))
    axes2[0].plot(time, mse_boot.mean(axis=0), label="bootstrap PF")
    axes2[0].plot(time, mse_opt.mean(axis=0), label="optimal PF")
    axes2[0].set_xlabel("Period")
    axes2[0].set_ylabel("Mean squared error")
    axes2[0].set_title("Monte Carlo Error vs Kalman")
    axes2[0].legend()
    axes2[1].plot(time, bootstrap_one["ess"], label="bootstrap PF")
    axes2[1].plot(time, optimal_one["ess"], label="optimal PF")
    axes2[1].axhline(n_particles / 2.0, color="black", linestyle=":", linewidth=1.0)
    axes2[1].set_xlabel("Period")
    axes2[1].set_ylabel("Effective sample size")
    axes2[1].set_title("Weight Degeneracy Diagnostic")
    axes2[1].legend()
    fig2.tight_layout()
    report.add_figure(
        "figures/mse-and-ess.png",
        "Repeated-run Monte Carlo error and effective sample size",
        fig2,
        description=(
            "Effective sample size falls when a few particles receive most of the weight. The "
            "conditionally optimal proposal usually preserves more useful particles because it "
            "looks at the observation before drawing the new state."
        ),
    )

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
    axes3[0].set_title("Sharper Signals Raise Bootstrap Error")
    axes3[1].set_xlabel("Measurement std")
    axes3[1].set_ylabel("Mean ESS")
    axes3[1].set_title("Sharper Signals Lower ESS")
    axes3[1].legend()
    fig3.tight_layout()
    report.add_figure(
        "figures/measurement-noise.png",
        "Particle accuracy as measurement noise falls",
        fig3,
        description=(
            "Low measurement noise makes the likelihood sharply peaked. Bootstrap particles "
            "drawn from the transition can miss that peak, creating weight degeneracy."
        ),
    )

    fig4, axes4 = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    for dim, ax in enumerate(axes4):
        ax.plot(time, outlier_kalman["filtered_mean"][:, dim], color="black", label="Kalman with outlier")
        ax.plot(time, outlier_boot["mean"][:, dim], color="tab:blue", label="bootstrap PF")
        ax.plot(time, outlier_opt["mean"][:, dim], color="tab:orange", label="optimal PF")
        ax.axvline(25, color="crimson", linestyle="--", linewidth=1.0, label="outlier" if dim == 0 else None)
        ax.set_ylabel(f"s{dim + 1}")
        ax.legend(loc="upper right")
    axes4[0].set_title("Outlier Stress Test")
    axes4[-1].set_xlabel("Period")
    report.add_figure(
        "figures/outlier-stress.png",
        "Filtering after multiplying observation 25 by ten",
        fig4,
        description=(
            "Outliers are hard for likelihood-weighted simulation. A single extreme observation "
            "can concentrate weights and pull the filtered state sharply."
        ),
    )

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
    report.add_table(
        "tables/filter-summary.csv",
        "Baseline repeated-run comparison",
        format_table(pd.DataFrame(summary_rows)),
        description="The baseline repeats each filter 50 times with 500 particles.",
    )
    report.add_table(
        "tables/measurement-noise-sweep.csv",
        "Measurement-noise sensitivity",
        format_table(measurement_table),
        description="Lower measurement noise makes the observation likelihood sharper.",
    )
    report.add_table(
        "tables/particle-count-sweep.csv",
        "Particle-count sensitivity",
        format_table(particle_table),
        description="The optimal proposal can often match bootstrap accuracy with fewer particles.",
    )

    report.add_results(
        f"With {n_particles} particles, the bootstrap filter has RMSE "
        f"{np.sqrt(np.mean(mse_boot)):.4f} relative to the Kalman filtered mean, while the "
        f"conditionally optimal filter has RMSE {np.sqrt(np.mean(mse_opt)):.4f}. The main "
        "difference is not the model; it is the proposal distribution used to place particles."
    )

    report.add_takeaway(
        "Particle filters are flexible because they replace analytic filtering distributions "
        "with weighted simulations. That flexibility has a cost: particle placement matters. "
        "When observations are very informative or contaminated by outliers, naive bootstrap "
        "particles can collapse onto a few high-weight draws. Better proposals, more particles, "
        "and outlier-robust measurement models are practical ways to respond."
    )

    report.add_references(
        [
            "Chang, M. ECON 609 Problem Set 3: Kalman Filter vs. Particle Filter.",
            "Gordon, N. J., Salmond, D. J., and Smith, A. F. M. (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. IEE Proceedings F.",
            "Doucet, A., de Freitas, N., and Gordon, N. (2001). Sequential Monte Carlo Methods in Practice. Springer.",
        ]
    )

    report.write("README.md")


if __name__ == "__main__":
    main()
