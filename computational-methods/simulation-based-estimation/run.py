#!/usr/bin/env python3
"""Simulation-based estimation of a search acceptance rule."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


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


def parameter_table(theta_true: np.ndarray, msm: dict[str, object], ii: dict[str, object]) -> pd.DataFrame:
    """Parameter recovery table."""
    return pd.DataFrame(
        {
            "Parameter": ["Offer mean mu", "Offer sd sigma", "Reservation log wage"],
            "True": theta_true,
            "MSM estimate": np.asarray(msm["theta"], dtype=float),
            "MSM error": np.asarray(msm["theta"], dtype=float) - theta_true,
            "Indirect inference estimate": np.asarray(ii["theta"], dtype=float),
            "Indirect inference error": np.asarray(ii["theta"], dtype=float) - theta_true,
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

    msm = estimate_by_simulation(target_moments, simulation_draws, choice_scale, economic_moments, start)
    ii = estimate_by_simulation(target_aux, simulation_draws, choice_scale, auxiliary_statistics, start)

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
    print(f"  MSM criterion: {float(msm['criterion']):.4e}")
    print(f"  II criterion: {float(ii['criterion']):.4e}")

    setup_style()
    report = ModelReport(
        "Estimating a Search Acceptance Rule by Simulation",
        "Estimate offer distribution and reservation wage with simulated summaries.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A researcher observes wage offers and whether workers accept them. The "
        "reservation wage is hidden.\n\n"
        "The object is a rule that maps offers into acceptance probabilities. It "
        "depends on offer mean, offer dispersion, and reservation log wage.\n\n"
        "The model is easy to simulate for any parameter vector. Simulation-based "
        "estimation searches for parameters whose simulated data match observed "
        "summaries."
    )

    report.add_equations(
        r"""
Worker $i$ receives a log wage offer,

$$
\log w_i = \mu + \sigma z_i,\qquad z_i\sim N(0,1),
$$

and accepts with probability

$$
\Pr(d_i=1\mid w_i;\theta)
= \frac{1}{1+\exp[-(\log w_i-r)/s]}.
$$

The parameter vector is $\theta=(\mu,\sigma,r)$. Parameters $\mu$ and $\sigma$
set the offer distribution. The reservation log wage is $r$. The scale $s$
fixes how sharply acceptance changes near $r$.

MSM chooses $\theta$ to match a vector of economic moments:

$$
\hat\theta_{MSM}
= \arg\min_\theta
\left[m_{sim}(\theta)-m_{obs}\right]'
W_m
\left[m_{sim}(\theta)-m_{obs}\right].
$$

Indirect inference fits an auxiliary model $a(d_i,\log w_i)$ and matches its
estimated statistics:

$$
\hat\theta_{II}
= \arg\min_\theta
\left[b_{sim}(\theta)-b_{obs}\right]'
W_b
\left[b_{sim}(\theta)-b_{obs}\right].
$$

Here the auxiliary model is a linear probability regression of acceptance on
log wages. It also includes offer-distribution and acceptance summaries. It is
not the structural model. The simulator must reproduce its fitted statistics.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| True $\\mu$ | {theta_true[0]:.2f} | Mean of the latent log offer distribution |\n"
        f"| True $\\sigma$ | {theta_true[1]:.2f} | Dispersion of latent log offers |\n"
        f"| True $r$ | {theta_true[2]:.2f} | Latent reservation log wage |\n"
        f"| Choice scale $s$ | {choice_scale:.2f} | Smoothness of acceptance rule |\n"
        f"| Observed sample | {n_observed:,} | Synthetic data generated once from the model |\n"
        f"| Simulation draws | {n_simulated:,} | Common random numbers used in both criteria |\n"
        f"| MSM targets | {len(target_moments)} | Acceptance rate and offer-wage moments |\n"
        f"| II targets | {len(target_aux)} | Auxiliary acceptance coefficients and moments |"
    )

    report.add_solution_method(
        "The computation uses one simulator and two target vectors. For each candidate "
        "theta, the code simulates a search panel with fixed draws. It computes "
        "summaries, scales errors by target magnitudes, and minimizes the quadratic "
        "distance. Fixed draws keep the objective from changing because of fresh Monte "
        "Carlo noise.\n\n"
        "```text\n"
        "Algorithm: estimate the search rule by simulation\n"
        "Input: observed offers and decisions, simulator S(theta, eps), fixed shocks eps_sim\n"
        "Observed targets\n"
        "  MSM: m_obs, acceptance and wage moments\n"
        "  II: b_obs, auxiliary acceptance-model statistics\n"
        "For each candidate theta:\n"
        "  Draw simulated offers and acceptances using eps_sim\n"
        "  Compute m_sim(theta) for MSM or b_sim(theta) for indirect inference\n"
        "  Evaluate the scaled quadratic distance from the observed targets\n"
        "Choose the theta with the smallest distance\n"
        "Output: estimated offer distribution, reservation wage, residuals, and surfaces\n"
        "```\n\n"
        "MSM puts economic moments directly in the criterion. Indirect inference uses "
        "the auxiliary regression slope to summarize how acceptance changes with wages. "
        "Both criteria can identify the reservation rule when their targets use "
        "variation near the threshold."
    )

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
    report.add_results(
        "The criterion surfaces show how each estimator trades off offer mean and "
        "reservation wage. Both plots fix offer dispersion at its true value. The valley "
        "tilts because a higher mean can offset a higher reservation wage."
    )
    report.add_figure(
        "figures/criterion-surfaces.png",
        "Criterion surfaces for MSM and indirect inference",
        fig1,
    )

    observed_accept = observed["accept"]
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
    bins = np.linspace(1.5, 4.7, 34)
    ax2.hist(observed["log_wage"], bins=bins, alpha=0.45, label="All offers", density=True)
    ax2.hist(observed["log_wage"][observed_accept], bins=bins, alpha=0.65, label="Accepted offers", density=True)
    ax2.axvline(theta_true[2], color="black", linestyle="--", linewidth=1.3, label="True reservation")
    ax2.axvline(np.asarray(msm["theta"])[2], color="tab:red", linestyle=":", linewidth=2, label="MSM estimate")
    ax2.axvline(np.asarray(ii["theta"])[2], color="tab:blue", linestyle="-.", linewidth=2, label="II estimate")
    ax2.set_xlabel("Log wage")
    ax2.set_ylabel("Density")
    ax2.set_title("Observed Search Data")
    ax2.legend()
    report.add_results(
        f"The observed acceptance rate is **{observed_accept.mean():.3f}**. Accepted "
        "wages mostly come from the upper tail of offers. Stochastic choice leaves "
        "overlap near the reservation wage. That overlap helps locate the latent "
        "threshold."
    )
    report.add_figure(
        "figures/search-data.png",
        "Offer and accepted-wage distributions",
        fig2,
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    wage_grid = np.linspace(1.8, 4.4, 200)
    true_prob = expit((wage_grid - theta_true[2]) / choice_scale)
    msm_prob = expit((wage_grid - np.asarray(msm["theta"])[2]) / choice_scale)
    ii_prob = expit((wage_grid - np.asarray(ii["theta"])[2]) / choice_scale)
    ax3.plot(wage_grid, true_prob, label="True")
    ax3.plot(wage_grid, msm_prob, "--", label="MSM")
    ax3.plot(wage_grid, ii_prob, ":", label="Indirect inference")
    ax3.set_xlabel("Log wage offer")
    ax3.set_ylabel("Acceptance probability")
    ax3.set_title("Recovered Acceptance Rule")
    ax3.legend()
    report.add_results(
        "Both estimators recover the acceptance curve closely. The small gaps reflect "
        "the observed sample and finite simulation. They do not come from different "
        "search models."
    )
    report.add_figure(
        "figures/acceptance-rule.png",
        "True and estimated acceptance probabilities",
        fig3,
    )

    report.add_results(
        "Parameter estimates and residuals give a compact diagnostic. Small scaled "
        "residuals show that each target vector is matched closely."
    )

    report.add_table(
        "tables/parameter-recovery.csv",
        "Known-truth parameter recovery",
        parameter_table(theta_true, msm, ii).round(5),
    )

    report.add_table(
        "tables/msm-residuals.csv",
        "MSM moment residuals",
        residual_table(
            ["Acceptance rate", "Mean log wage", "SD log wage", "Mean accepted log wage", "SD accepted log wage"],
            target_moments,
            msm,
            msm,
        ).query("Estimator == 'MSM'").drop(columns=["Estimator"]).round(5),
    )

    report.add_table(
        "tables/indirect-inference-residuals.csv",
        "Indirect-inference auxiliary residuals",
        residual_table(
            ["LPM intercept", "LPM slope", "Mean log wage", "SD log wage", "Acceptance rate", "Mean accepted log wage"],
            target_aux,
            ii,
            ii,
        ).query("Estimator == 'MSM'").drop(columns=["Estimator"]).round(5),
    )

    report.add_takeaway(
        "Simulation-based estimation is useful when the structural model is easier to "
        "simulate than to evaluate by likelihood. MSM matches economic moments chosen "
        "by the researcher. Indirect inference matches fitted statistics from an "
        "auxiliary acceptance model. Here both identify the offer distribution and "
        "reservation wage because they use threshold variation."
    )

    report.add_references(
        [
            "[McFadden, D. (1989). A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. *Econometrica*, 57(5), 995-1026.](https://doi.org/10.2307/1913621)",
            "[Gourieroux, C., Monfort, A., and Renault, E. (1993). Indirect Inference. *Journal of Applied Econometrics*, 8(S1), S85-S118.](https://doi.org/10.1002/jae.3950080507)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
