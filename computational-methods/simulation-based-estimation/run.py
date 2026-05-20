#!/usr/bin/env python3
"""Simulation-based estimation of a search acceptance rule."""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


def simulate_search_panel(theta: np.ndarray, draws: dict[str, np.ndarray], choice_scale: float) -> dict[str, np.ndarray]:
    """Simulate wage offers and stochastic acceptance decisions."""
    mu, sigma, reservation = theta
    log_wage = mu + sigma * draws["z"]
    accept_prob = expit((log_wage - reservation) / choice_scale)
    accept = draws["u"] < accept_prob
    return {"log_wage": log_wage, "accept": accept, "accept_prob": accept_prob}


def economic_moments(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Moments used by the direct MSM estimator."""
    log_wage = sample["log_wage"]
    accept = sample["accept"]
    accepted_wages = log_wage[accept]
    return np.array(
        [
            accept.mean(),
            log_wage.mean(),
            log_wage.std(ddof=0),
            accepted_wages.mean(),
            accepted_wages.std(ddof=0),
        ]
    )


def auxiliary_statistics(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Auxiliary linear probability model coefficients plus wage distribution moments."""
    log_wage = sample["log_wage"]
    accept = sample["accept"].astype(float)
    X = np.column_stack([np.ones_like(log_wage), log_wage])
    intercept, slope = np.linalg.lstsq(X, accept, rcond=None)[0]
    accepted_wages = log_wage[sample["accept"]]
    return np.array(
        [
            intercept,
            slope,
            log_wage.mean(),
            log_wage.std(ddof=0),
            accept.mean(),
            accepted_wages.mean(),
        ]
    )


def criterion(
    theta: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
    statistic_fn,
) -> float:
    """Scaled quadratic simulation criterion."""
    if theta[0] < 2.4 or theta[0] > 3.6 or theta[1] < 0.2 or theta[1] > 0.8 or theta[2] < 2.5 or theta[2] > 3.8:
        return 1e6
    sample = simulate_search_panel(theta, draws, choice_scale)
    diff = (statistic_fn(sample) - target) / scale
    return float(diff @ diff)


def estimate_by_simulation(
    target: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
    statistic_fn,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate parameters by minimizing a simulation criterion."""
    scale = np.maximum(np.abs(target), 0.1)
    objective = lambda theta: criterion(theta, target, scale, draws, choice_scale, statistic_fn)
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 350, "xatol": 1e-4, "fatol": 1e-6, "disp": False},
    )
    theta_hat = np.asarray(result.x, dtype=float)
    sample_hat = simulate_search_panel(theta_hat, draws, choice_scale)
    simulated_stats = statistic_fn(sample_hat)
    residual = (simulated_stats - target) / scale
    return {
        "theta": theta_hat,
        "criterion": float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
        "simulated_stats": simulated_stats,
        "residual": residual,
    }


def make_draws(seed: int, n: int) -> dict[str, np.ndarray]:
    """Common random numbers for simulation objectives."""
    rng = np.random.default_rng(seed)
    return {"z": rng.normal(size=n), "u": rng.uniform(size=n)}


def criterion_grid(
    mu_grid: np.ndarray,
    reservation_grid: np.ndarray,
    sigma_fixed: float,
    target: np.ndarray,
    scale: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
    statistic_fn,
) -> np.ndarray:
    """Evaluate a two-dimensional criterion surface."""
    values = np.zeros((len(reservation_grid), len(mu_grid)))
    for i, reservation in enumerate(reservation_grid):
        for j, mu in enumerate(mu_grid):
            theta = np.array([mu, sigma_fixed, reservation])
            values[i, j] = criterion(theta, target, scale, draws, choice_scale, statistic_fn)
    return values


def sample_prior(prior_bounds: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n particles uniformly from the rectangle defined by prior_bounds (d, 2)."""
    lo = prior_bounds[:, 0]
    hi = prior_bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, lo.size))


def in_bounds(theta: np.ndarray, prior_bounds: np.ndarray) -> bool:
    """Check that every coordinate of theta lies inside prior_bounds."""
    return bool(np.all(theta >= prior_bounds[:, 0]) and np.all(theta <= prior_bounds[:, 1]))


def perturb(
    parent: np.ndarray,
    cov: np.ndarray,
    prior_bounds: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> np.ndarray:
    """Gaussian random-walk perturbation; resample if the proposal exits the prior box."""
    for _ in range(max_tries):
        proposal = rng.multivariate_normal(parent, cov)
        if in_bounds(proposal, prior_bounds):
            return proposal
    return parent


def simulated_distance(
    theta: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
) -> float:
    """Scaled Euclidean distance between simulated and observed economic moments."""
    sample = simulate_search_panel(theta, draws, choice_scale)
    diff = (economic_moments(sample) - target) / scale
    return float(np.sqrt(diff @ diff))


def smc_weights(
    particles: np.ndarray,
    prev_particles: np.ndarray,
    prev_weights: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Importance weights for a uniform prior: w_i propto 1 / sum_j w_{t-1,j} K(theta_i | theta_{t-1,j})."""
    n, d = particles.shape
    inv_cov = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    log_norm = -0.5 * (logdet + d * np.log(2 * np.pi))
    log_prev = np.log(np.maximum(prev_weights, 1e-300))
    log_w = np.zeros(n)
    for i in range(n):
        diff = particles[i] - prev_particles
        quad = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        log_k = log_norm - 0.5 * quad
        terms = log_prev + log_k
        m = terms.max()
        log_w[i] = -(m + np.log(np.exp(terms - m).sum()))
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def posterior_summary(particles: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted posterior mean and 5/95 percentile credible interval per parameter."""
    mean = (weights[:, None] * particles).sum(axis=0)
    d = particles.shape[1]
    ci = np.zeros((d, 2))
    for j in range(d):
        idx = np.argsort(particles[:, j])
        sorted_vals = particles[idx, j]
        sorted_w = weights[idx]
        cum = np.cumsum(sorted_w) / sorted_w.sum()
        ci[j, 0] = sorted_vals[np.searchsorted(cum, 0.05)]
        ci[j, 1] = sorted_vals[np.searchsorted(cum, 0.95)]
    return mean, ci


def abc_smc(
    target: np.ndarray,
    draws: dict[str, np.ndarray],
    choice_scale: float,
    prior_bounds: np.ndarray,
    n_particles: int = 1000,
    n_rounds: int = 6,
    alpha: float = 0.5,
    seed: int = 789,
) -> dict[str, object]:
    """Run ABC-SMC with adaptive tolerance schedule on the economic moments."""
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(target), 0.1)
    d = prior_bounds.shape[0]
    n_round0 = int(np.ceil(n_particles / alpha))

    particles_hist = np.zeros((n_rounds, n_particles, d))
    weights_hist = np.zeros((n_rounds, n_particles))
    distances_hist = np.zeros((n_rounds, n_particles))
    tolerances = np.zeros(n_rounds)
    accept_rates = np.zeros(n_rounds)
    ess = np.zeros(n_rounds)
    proposals = np.zeros(n_rounds, dtype=int)

    raw = sample_prior(prior_bounds, n_round0, rng)
    raw_d = np.array([simulated_distance(p, target, scale, draws, choice_scale) for p in raw])
    order = np.argsort(raw_d)[:n_particles]
    particles_hist[0] = raw[order]
    distances_hist[0] = raw_d[order]
    weights_hist[0] = 1.0 / n_particles
    tolerances[0] = float(raw_d[order].max())
    proposals[0] = n_round0
    accept_rates[0] = n_particles / n_round0
    ess[0] = n_particles

    for t in range(1, n_rounds):
        eps_t = float(np.quantile(distances_hist[t - 1], alpha))
        tolerances[t] = eps_t

        w_prev = weights_hist[t - 1]
        mean_prev = (w_prev[:, None] * particles_hist[t - 1]).sum(axis=0)
        centered = particles_hist[t - 1] - mean_prev
        cov_prev = (w_prev[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(axis=0)
        cov_t = 2.0 * cov_prev + 1e-10 * np.eye(d)

        cum_weights = np.cumsum(w_prev)
        cum_weights /= cum_weights[-1]

        new_particles = np.zeros((n_particles, d))
        new_distances = np.zeros(n_particles)
        n_proposals_t = 0

        for i in range(n_particles):
            while True:
                u = rng.uniform()
                parent_idx = int(np.searchsorted(cum_weights, u))
                parent = particles_hist[t - 1, parent_idx]
                proposal = perturb(parent, cov_t, prior_bounds, rng)
                n_proposals_t += 1
                d_proposal = simulated_distance(proposal, target, scale, draws, choice_scale)
                if d_proposal <= eps_t:
                    new_particles[i] = proposal
                    new_distances[i] = d_proposal
                    break

        new_weights = smc_weights(new_particles, particles_hist[t - 1], w_prev, cov_t)
        ess_t = 1.0 / float((new_weights ** 2).sum())
        if ess_t < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=new_weights)
            new_particles = new_particles[idx]
            new_distances = new_distances[idx]
            new_weights = np.ones(n_particles) / n_particles
            ess_t = float(n_particles)

        particles_hist[t] = new_particles
        weights_hist[t] = new_weights
        distances_hist[t] = new_distances
        proposals[t] = n_proposals_t
        accept_rates[t] = n_particles / n_proposals_t
        ess[t] = ess_t

    post_mean, post_ci = posterior_summary(particles_hist[-1], weights_hist[-1])
    criterion_at_mean = float(simulated_distance(post_mean, target, scale, draws, choice_scale) ** 2)
    return {
        "particles": particles_hist,
        "weights": weights_hist,
        "distances": distances_hist,
        "tolerances": tolerances,
        "accept_rates": accept_rates,
        "ess": ess,
        "proposals": proposals,
        "theta_post_mean": post_mean,
        "theta_post_ci": post_ci,
        "theta": post_mean,
        "criterion": criterion_at_mean,
        "final_tolerance": float(tolerances[-1]),
        "total_proposals": int(proposals.sum()),
    }


def parameter_table(
    theta_true: np.ndarray,
    msm: dict[str, object],
    ii: dict[str, object],
    abc: dict[str, object],
) -> pd.DataFrame:
    """Parameter recovery table including the ABC posterior mean and 5/95 credible interval."""
    abc_mean = np.asarray(abc["theta_post_mean"], dtype=float)
    abc_ci = np.asarray(abc["theta_post_ci"], dtype=float)
    return pd.DataFrame(
        {
            "Parameter": ["Offer mean mu", "Offer sd sigma", "Reservation log wage"],
            "True": theta_true,
            "MSM estimate": np.asarray(msm["theta"], dtype=float),
            "MSM error": np.asarray(msm["theta"], dtype=float) - theta_true,
            "Indirect inference estimate": np.asarray(ii["theta"], dtype=float),
            "Indirect inference error": np.asarray(ii["theta"], dtype=float) - theta_true,
            "ABC posterior mean": abc_mean,
            "ABC error": abc_mean - theta_true,
            "ABC 5%": abc_ci[:, 0],
            "ABC 95%": abc_ci[:, 1],
        }
    )


def method_comparison_table(
    msm: dict[str, object],
    ii: dict[str, object],
    abc: dict[str, object],
    times: dict[str, float],
) -> pd.DataFrame:
    """One-row-per-method comparison of estimates, loss, work, and wall time."""
    msm_theta = np.asarray(msm["theta"], dtype=float)
    ii_theta = np.asarray(ii["theta"], dtype=float)
    abc_theta = np.asarray(abc["theta_post_mean"], dtype=float)
    return pd.DataFrame(
        {
            "Method": ["MSM", "Indirect inference", "ABC-SMC"],
            "Offer mean mu": [msm_theta[0], ii_theta[0], abc_theta[0]],
            "Offer sd sigma": [msm_theta[1], ii_theta[1], abc_theta[1]],
            "Reservation log wage": [msm_theta[2], ii_theta[2], abc_theta[2]],
            "Criterion at point estimate or posterior mean": [
                float(msm["criterion"]),
                float(ii["criterion"]),
                float(abc["criterion"]),
            ],
            "Iterations or proposals": [
                int(msm["iterations"]),
                int(ii["iterations"]),
                int(abc["total_proposals"]),
            ],
            "Wall time (s)": [times["msm"], times["ii"], times["abc"]],
        }
    )


def residual_table(names: list[str], target: np.ndarray, msm: dict[str, object], ii: dict[str, object]) -> pd.DataFrame:
    """Stack target and simulated residual diagnostics."""
    rows = []
    for estimator, result in [("MSM", msm), ("Indirect inference", ii)]:
        simulated = np.asarray(result["simulated_stats"], dtype=float)
        residual = np.asarray(result["residual"], dtype=float)
        for name, target_value, simulated_value, scaled_error in zip(names, target, simulated, residual):
            rows.append(
                {
                    "Estimator": estimator,
                    "Statistic": name,
                    "Observed target": target_value,
                    "Simulated at estimate": simulated_value,
                    "Scaled residual": scaled_error,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    theta_true = np.array([3.00, 0.45, 3.15])
    choice_scale = 0.18
    n_observed = 5_000
    n_simulated = 30_000
    observed_draws = make_draws(123, n_observed)
    simulation_draws = make_draws(456, n_simulated)

    observed = simulate_search_panel(theta_true, observed_draws, choice_scale)
    target_moments = economic_moments(observed)
    target_aux = auxiliary_statistics(observed)
    start = np.array([2.85, 0.36, 3.00])

    t0 = time.perf_counter()
    msm = estimate_by_simulation(target_moments, simulation_draws, choice_scale, economic_moments, start)
    t_msm = time.perf_counter() - t0

    t0 = time.perf_counter()
    ii = estimate_by_simulation(target_aux, simulation_draws, choice_scale, auxiliary_statistics, start)
    t_ii = time.perf_counter() - t0

    prior_bounds = np.array([[2.4, 3.6], [0.2, 0.8], [2.5, 3.8]])
    n_particles = 1000
    n_rounds = 6
    alpha_quantile = 0.5
    t0 = time.perf_counter()
    abc = abc_smc(
        target_moments,
        simulation_draws,
        choice_scale,
        prior_bounds,
        n_particles=n_particles,
        n_rounds=n_rounds,
        alpha=alpha_quantile,
        seed=789,
    )
    t_abc = time.perf_counter() - t0
    times = {"msm": t_msm, "ii": t_ii, "abc": t_abc}

    msm_scale = np.maximum(np.abs(target_moments), 0.1)
    ii_scale = np.maximum(np.abs(target_aux), 0.1)
    mu_grid = np.linspace(2.82, 3.18, 32)
    reservation_grid = np.linspace(2.95, 3.35, 32)
    msm_surface = criterion_grid(
        mu_grid,
        reservation_grid,
        theta_true[1],
        target_moments,
        msm_scale,
        simulation_draws,
        choice_scale,
        economic_moments,
    )
    ii_surface = criterion_grid(
        mu_grid,
        reservation_grid,
        theta_true[1],
        target_aux,
        ii_scale,
        simulation_draws,
        choice_scale,
        auxiliary_statistics,
    )

    print("Simulation-based estimation tutorial")
    print(f"  True theta: {theta_true}")
    print(f"  MSM theta: {np.asarray(msm['theta'])}")
    print(f"  Indirect inference theta: {np.asarray(ii['theta'])}")
    print(f"  ABC posterior mean: {np.asarray(abc['theta_post_mean'])}")
    print(f"  MSM criterion: {float(msm['criterion']):.4e}")
    print(f"  II criterion: {float(ii['criterion']):.4e}")
    print(f"  ABC criterion at posterior mean: {float(abc['criterion']):.4e}")
    print(f"  ABC final tolerance: {float(abc['final_tolerance']):.4e}")
    print(f"  Wall time (s): MSM {t_msm:.2f}, II {t_ii:.2f}, ABC {t_abc:.2f}")

    setup_style()

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    extent = [mu_grid.min(), mu_grid.max(), reservation_grid.min(), reservation_grid.max()]
    im0 = axes[0].imshow(msm_surface, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    axes[0].scatter(theta_true[0], theta_true[2], color="white", edgecolor="black", label="True")
    axes[0].scatter(np.asarray(msm["theta"])[0], np.asarray(msm["theta"])[2], color="tab:red", label="Estimate")
    axes[0].set_title("MSM Criterion")
    axes[0].set_xlabel("Offer mean mu")
    axes[0].set_ylabel("Reservation log wage")
    axes[0].legend(loc="upper left")
    fig1.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(ii_surface, origin="lower", extent=extent, aspect="auto", cmap="magma")
    axes[1].scatter(theta_true[0], theta_true[2], color="white", edgecolor="black", label="True")
    axes[1].scatter(np.asarray(ii["theta"])[0], np.asarray(ii["theta"])[2], color="tab:cyan", edgecolor="black", label="Estimate")
    axes[1].set_title("Indirect-Inference Criterion")
    axes[1].set_xlabel("Offer mean mu")
    axes[1].legend(loc="upper left")
    fig1.colorbar(im1, ax=axes[1], fraction=0.046)
    save_figure(fig1, "figures/criterion-surfaces.png", dpi=150)

    observed_accept = observed["accept"]
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
    bins = np.linspace(1.5, 4.7, 34)
    ax2.hist(observed["log_wage"], bins=bins, alpha=0.45, label="All offers", density=True)
    ax2.hist(observed["log_wage"][observed_accept], bins=bins, alpha=0.65, label="Accepted offers", density=True)
    ax2.axvline(theta_true[2], color="black", linestyle="--", linewidth=1.3, label="True reservation")
    ax2.axvline(np.asarray(msm["theta"])[2], color="tab:red", linestyle=":", linewidth=2, label="MSM estimate")
    ax2.axvline(np.asarray(ii["theta"])[2], color="tab:blue", linestyle="-.", linewidth=2, label="II estimate")
    ax2.axvline(np.asarray(abc["theta_post_mean"])[2], color="tab:green", linestyle="-", linewidth=2, alpha=0.7, label="ABC posterior mean")
    ax2.set_xlabel("Log wage")
    ax2.set_ylabel("Density")
    ax2.set_title("Observed Search Data")
    ax2.legend()
    save_figure(fig2, "figures/search-data.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    wage_grid = np.linspace(1.8, 4.4, 200)
    true_prob = expit((wage_grid - theta_true[2]) / choice_scale)
    msm_prob = expit((wage_grid - np.asarray(msm["theta"])[2]) / choice_scale)
    ii_prob = expit((wage_grid - np.asarray(ii["theta"])[2]) / choice_scale)
    abc_prob = expit((wage_grid - np.asarray(abc["theta_post_mean"])[2]) / choice_scale)
    ax3.plot(wage_grid, true_prob, label="True")
    ax3.plot(wage_grid, msm_prob, "--", label="MSM")
    ax3.plot(wage_grid, ii_prob, ":", label="Indirect inference")
    ax3.plot(wage_grid, abc_prob, "-.", label="ABC posterior mean")
    ax3.set_xlabel("Log wage offer")
    ax3.set_ylabel("Acceptance probability")
    ax3.set_title("Recovered Acceptance Rule")
    ax3.legend()
    save_figure(fig3, "figures/acceptance-rule.png", dpi=150)

    final_particles = np.asarray(abc["particles"])[-1]
    final_weights = np.asarray(abc["weights"])[-1]
    abc_post_mean = np.asarray(abc["theta_post_mean"])
    abc_post_ci = np.asarray(abc["theta_post_ci"])
    fig4, axes4 = plt.subplots(1, 3, figsize=(13.5, 4.2))
    param_names = [r"Offer mean $\mu$", r"Offer sd $\sigma$", r"Reservation $r$"]
    grids = [
        np.linspace(prior_bounds[j, 0], prior_bounds[j, 1], 400) for j in range(3)
    ]
    for j, ax in enumerate(axes4):
        kde = gaussian_kde(final_particles[:, j], weights=final_weights, bw_method=0.35)
        density = kde(grids[j])
        ax.plot(grids[j], density, color="tab:green", linewidth=2, label="ABC posterior")
        mask = (grids[j] >= abc_post_ci[j, 0]) & (grids[j] <= abc_post_ci[j, 1])
        ax.fill_between(grids[j], 0, density, where=mask, color="tab:green", alpha=0.18, label="90% CI")
        ax.axvline(theta_true[j], color="black", linestyle="-", linewidth=1.5, label="True")
        ax.axvline(np.asarray(msm["theta"])[j], color="tab:red", linestyle="--", linewidth=1.4, label="MSM")
        ax.axvline(np.asarray(ii["theta"])[j], color="tab:blue", linestyle=":", linewidth=1.6, label="II")
        ax.axvline(abc_post_mean[j], color="tab:green", linestyle="-.", linewidth=1.4, label="ABC mean")
        ax.set_xlabel(param_names[j])
        ax.set_ylabel("Posterior density")
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig4.suptitle("ABC-SMC Posterior Marginals")
    fig4.tight_layout()
    save_figure(fig4, "figures/abc-posteriors.png", dpi=150)

    abc_tolerances = np.asarray(abc["tolerances"])
    abc_accept = np.asarray(abc["accept_rates"])
    rounds_axis = np.arange(len(abc_tolerances))
    fig5, ax5 = plt.subplots(figsize=(7, 4.2))
    color_tol = "tab:purple"
    ax5.plot(rounds_axis, abc_tolerances, color=color_tol, marker="o", label=r"Tolerance $\varepsilon_t$")
    ax5.set_yscale("log")
    ax5.set_xlabel("Round $t$")
    ax5.set_ylabel(r"Tolerance $\varepsilon_t$ (log scale)", color=color_tol)
    ax5.tick_params(axis="y", labelcolor=color_tol)
    ax5b = ax5.twinx()
    color_acc = "tab:orange"
    ax5b.plot(rounds_axis, abc_accept, color=color_acc, marker="s", linestyle="--", label="Acceptance rate")
    ax5b.set_ylabel("Acceptance rate", color=color_acc)
    ax5b.tick_params(axis="y", labelcolor=color_acc)
    ax5b.set_ylim(0, 1.05)
    ax5.set_title("ABC-SMC Tolerance Schedule")
    fig5.tight_layout()
    save_figure(fig5, "figures/abc-tolerance-schedule.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    parameter_table(theta_true, msm, ii, abc).round(5).to_csv("tables/parameter-recovery.csv", index=False)

    residual_table(
        ["Acceptance rate", "Mean log wage", "SD log wage", "Mean accepted log wage", "SD accepted log wage"],
        target_moments,
        msm,
        msm,
    ).query("Estimator == 'MSM'").drop(columns=["Estimator"]).round(5).to_csv("tables/msm-residuals.csv", index=False)

    residual_table(
        ["LPM intercept", "LPM slope", "Mean log wage", "SD log wage", "Acceptance rate", "Mean accepted log wage"],
        target_aux,
        msm,
        ii,
    ).query("Estimator == 'Indirect inference'").drop(columns=["Estimator"]).round(5).to_csv(
        "tables/indirect-inference-residuals.csv", index=False
    )

    method_comparison_table(msm, ii, abc, times).round(5).to_csv("tables/method-comparison.csv", index=False)

    abc_summary = pd.DataFrame(
        {
            "Round": np.arange(len(abc["tolerances"])),
            "Tolerance": np.asarray(abc["tolerances"], dtype=float),
            "Proposals": np.asarray(abc["proposals"], dtype=int),
            "Acceptance rate": np.asarray(abc["accept_rates"], dtype=float),
            "ESS": np.asarray(abc["ess"], dtype=float),
            "Mean distance": np.asarray(abc["distances"]).mean(axis=1),
        }
    ).round({"Tolerance": 5, "Acceptance rate": 4, "ESS": 2, "Mean distance": 5})
    abc_summary.to_csv("tables/abc-summary.csv", index=False)

    save_thumbnail("figures/criterion-surfaces.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
