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
        "Recover offer-distribution and reservation-wage parameters with MSM and indirect inference.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Suppose a researcher observes wage offers and worker acceptances but does not "
        "observe the reservation wage that turns an offer into a job. The object of "
        "interest is an acceptance rule: how the distribution of offers and the latent "
        "reservation wage shape employment decisions.\n\n"
        "The model is easy to simulate. Draw an offer, pass it through a smooth "
        "acceptance rule, and record whether the worker takes the job. Writing and "
        "maximizing the exact likelihood is not the point here. The computation replaces "
        "that likelihood with simulated data and asks which parameter values make the "
        "simulated search economy look like the observed one.\n\n"
        "The tutorial estimates the same search environment two ways. Method of "
        "simulated moments (MSM) matches economic summaries such as acceptance rates and "
        "accepted wages. Indirect inference fits a simple auxiliary acceptance model to "
        "observed and simulated samples, then matches the fitted coefficients. Fixed "
        "simulation draws make the criterion surfaces readable, so the reader can see "
        "where the parameters are identified and where tradeoffs remain."
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

The parameter vector is $\theta=(\mu,\sigma,r)$. The mean and dispersion of
offers are $\mu$ and $\sigma$, the reservation log wage is $r$, and $s$ fixes
how sharply acceptance changes around the reservation wage.

MSM chooses $\theta$ to match a vector of economic moments:

$$
\hat\theta_{MSM}
= \arg\min_\theta
\left[m_{sim}(\theta)-m_{obs}\right]'
W_m
\left[m_{sim}(\theta)-m_{obs}\right].
$$

Indirect inference takes a different route. It fits an auxiliary model
$a(d_i,\log w_i)$ and matches its estimated statistics:

$$
\hat\theta_{II}
= \arg\min_\theta
\left[b_{sim}(\theta)-b_{obs}\right]'
W_b
\left[b_{sim}(\theta)-b_{obs}\right].
$$

Here the auxiliary model is a linear probability regression of acceptance on
log wages, augmented with offer-distribution and acceptance statistics. It is
not the structural model. It is a low-dimensional description of the acceptance
pattern that the structural simulator has to reproduce.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| True $\\mu$ | {theta_true[0]:.2f} | Mean of the latent log offer distribution |\n"
        f"| True $\\sigma$ | {theta_true[1]:.2f} | Dispersion of latent log offers |\n"
        f"| True $r$ | {theta_true[2]:.2f} | Latent reservation log wage |\n"
        f"| Choice scale $s$ | {choice_scale:.2f} | Smoothness of acceptance rule |\n"
        f"| Observed sample | {n_observed:,} | Synthetic data generated once from the DGP |\n"
        f"| Simulation draws | {n_simulated:,} | Common random numbers used in both criteria |\n"
        f"| MSM targets | {len(target_moments)} | Acceptance rate and offer-wage moments |\n"
        f"| II targets | {len(target_aux)} | Auxiliary acceptance coefficients and moments |"
    )

    report.add_solution_method(
        "The computation uses one simulator and two choices of summary statistics. For "
        "a candidate parameter vector, the code simulates a large search panel with fixed "
        "random draws, computes the requested summaries, scales their differences from "
        "the observed targets, and minimizes the resulting quadratic distance. Reusing "
        "the same draws at every trial value keeps changes in the objective tied to "
        "parameters rather than fresh Monte Carlo noise.\n\n"
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
        "MSM keeps the economist's moment choice in the foreground. Indirect inference "
        "moves some of that choice into an auxiliary regression. If the regression slope "
        "captures how acceptance changes with wages, matching it gives the structural "
        "estimator information about the reservation rule."
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
        "The criterion surfaces show how the simulated search economy maps parameters "
        "into target statistics. Both surfaces hold offer dispersion at its true value. "
        "The valleys tilt because a higher offer mean and a higher reservation wage can "
        "partly offset each other in the acceptance rate."
    )
    report.add_figure(
        "figures/criterion-surfaces.png",
        "MSM and indirect-inference criterion surfaces",
        fig1,
        description="Common random numbers make the surfaces interpretable rather than dominated by fresh simulation noise.",
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
        "The observed sample has an acceptance rate of "
        f"**{observed_accept.mean():.3f}**. Accepted wages come mostly from the upper "
        "tail of the offer distribution, but stochastic choice leaves overlap around "
        "the reservation wage. That overlap is the evidence the estimator uses to locate "
        "the latent threshold."
    )
    report.add_figure(
        "figures/search-data.png",
        "Offer and accepted-wage distributions",
        fig2,
        description="The reservation wage is not observed directly; it is inferred from the acceptance pattern around the offer distribution.",
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
        "Both simulation estimators recover the acceptance rule because their targets "
        "use variation near the threshold. The remaining differences come from the "
        "observed sample and the finite simulation, not from a different search model."
    )
    report.add_figure(
        "figures/acceptance-rule.png",
        "True and estimated acceptance probabilities",
        fig3,
        description="The estimated reservation wages map directly into the smooth acceptance curves.",
    )

    report.add_table(
        "tables/parameter-recovery.csv",
        "Known-truth parameter recovery",
        parameter_table(theta_true, msm, ii).round(5),
        description="Both estimators use the same simulated model but target different summaries of the data.",
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
        description="MSM residuals are scaled by the target magnitudes used in the quadratic criterion.",
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
        description="Indirect inference matches auxiliary regression coefficients and auxiliary distribution statistics.",
    )

    diagnostics = pd.DataFrame(
        {
            "Estimator": ["MSM", "Indirect inference"],
            "Criterion": [float(msm["criterion"]), float(ii["criterion"])],
            "Success": [bool(msm["success"]), bool(ii["success"])],
            "Iterations": [int(msm["iterations"]), int(ii["iterations"])],
            "Parameter RMSE": [
                float(np.sqrt(np.mean((np.asarray(msm["theta"]) - theta_true) ** 2))),
                float(np.sqrt(np.mean((np.asarray(ii["theta"]) - theta_true) ** 2))),
            ],
        }
    )
    report.add_table(
        "tables/estimator-diagnostics.csv",
        "Solver and estimator diagnostics",
        diagnostics,
        description="The RMSE is computed against the known DGP parameters, which are available only in the simulation exercise.",
    )

    report.add_takeaway(
        "Simulation-based estimation lets a researcher estimate a structural search rule "
        "when simulation is easier than likelihood evaluation. MSM asks the researcher "
        "to choose economic moments directly. Indirect inference asks for an auxiliary "
        "model whose fitted coefficients summarize the same behavior. In this example, "
        "both routes recover the offer distribution and reservation wage because their "
        "targets discipline the acceptance pattern around the latent threshold."
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
