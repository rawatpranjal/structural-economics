#!/usr/bin/env python3
"""Neural posterior estimation for the Brock-Hommes asset-pricing ABM."""
from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("sbi").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.brock_hommes import Params, average_moments, simulate, summary_statistics
from lib.plotting import save_figure, save_thumbnail, setup_style


PARAM_NAMES = ["beta", "g", "sigma_eps", "c_T"]
PARAM_LATEX = [r"$\beta$", r"$g$", r"$\sigma_\epsilon$", r"$c_T$"]
PARAM_DESC = [
    "intensity of choice",
    "trend gain",
    "shock scale",
    "trend cost",
]
TRUE_THETA = np.array([30.0, 1.40, 0.02, 0.001], dtype=np.float64)
PRIOR_LOW = np.array([1.0, 0.5, 0.005, 0.0], dtype=np.float64)
PRIOR_HIGH = np.array([60.0, 2.0, 0.05, 0.005], dtype=np.float64)
SUMMARY_NAMES = [
    "std of returns",
    "abs-return autocorr (lag 1)",
    "excess kurtosis",
    "return autocorr (lag 1)",
    "abs-return autocorr (lag 5)",
]
N_TRAIN = 10_000
N_OBSERVATIONS = 4
N_POSTERIOR_DRAWS = 5_000


def params_from_theta(theta: np.ndarray, base: Params) -> Params:
    """Wrap a 4-d sample (beta, g, sigma_eps, c_T) in a Params dataclass."""
    return replace(
        base,
        trend_gain=float(theta[1]),
        shock_sigma=float(theta[2]),
        trend_cost=float(theta[3]),
    )


def simulate_summary(theta: np.ndarray, base: Params, seed: int) -> np.ndarray:
    """One draw: parameter vector to 5-d summary statistic vector."""
    params = params_from_theta(theta, base)
    run = simulate(beta=float(theta[0]), params=params, seed=seed)
    return summary_statistics(run.x, params.burn)


def build_training_set(
    prior: BoxUniform,
    base: Params,
    n: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample theta from prior, simulate summary statistics."""
    theta = prior.sample((n,)).numpy()
    seeds = rng.integers(low=0, high=2**31 - 1, size=n).astype(np.int64)
    x = np.empty((n, len(SUMMARY_NAMES)), dtype=np.float64)
    for i in range(n):
        x[i] = simulate_summary(theta[i], base, int(seeds[i]))
        if (i + 1) % 1000 == 0:
            print(f"  simulated {i + 1}/{n}")
    finite = np.all(np.isfinite(x), axis=1)
    theta = theta[finite]
    x = x[finite]
    return torch.from_numpy(theta).float(), torch.from_numpy(x).float()


def observed_summaries(base: Params, rng: np.random.Generator, count: int) -> np.ndarray:
    """Generate count independent observed summary vectors at TRUE_THETA."""
    seeds = rng.integers(low=10**8, high=2**31 - 1, size=count).astype(np.int64)
    rows = [simulate_summary(TRUE_THETA, base, int(s)) for s in seeds]
    return np.stack(rows, axis=0)


def run_laplace_demo(seed: int = 7) -> plt.Figure:
    """Train NPE on a toy Laplace-mean problem and overlay the analytic posterior.

    A one-dimensional sanity check: x = mu + eps with eps ~ Laplace(0, 1) and
    mu ~ U(-5, 5). The likelihood is closed form, so the analytic posterior
    is also closed form and the flow approximation can be checked exactly.
    The same trained flow is evaluated at three different observations to
    show amortization without retraining.
    """
    torch.manual_seed(seed)
    prior = BoxUniform(low=torch.tensor([-5.0]), high=torch.tensor([5.0]))

    n_train = 4_000
    theta = prior.sample((n_train,))
    eps = torch.distributions.Laplace(0.0, 1.0).sample(theta.shape)
    x = theta + eps

    inference = NPE(prior=prior, density_estimator="maf", show_progress_bars=False)
    inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    x_obs_values = [-2.0, 0.5, 2.5]
    grid = np.linspace(-5.0, 5.0, 1001)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    for ax, x_obs in zip(panel_axes, x_obs_values):
        analytic = np.exp(-np.abs(grid - x_obs))
        analytic /= analytic.sum() * (grid[1] - grid[0])
        samples = posterior.sample(
            (5_000,),
            x=torch.tensor([x_obs]),
            show_progress_bars=False,
        ).numpy().flatten()
        ax.hist(samples, bins=40, range=(-5.0, 5.0), density=True, color="C0", alpha=0.65, label="NPE flow")
        ax.plot(grid, analytic, color="black", linestyle="--", linewidth=1.4, label="analytic")
        ax.axvline(x_obs, color="C3", linestyle=":", linewidth=1.0, label=fr"$x_{{obs}} = {x_obs}$")
        ax.set_xlim(-5.0, 5.0)
        ax.set_xlabel(r"$\mu$")
        ax.set_yticks([])
        ax.set_title(fr"observation $x_{{obs}} = {x_obs}$")
        if ax is panel_axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    ax_scatter = axes[1, 1]
    theta_np = theta.numpy().flatten()
    x_np = x.numpy().flatten()
    ax_scatter.scatter(theta_np, x_np, s=4, alpha=0.2, color="C0")
    for x_obs in x_obs_values:
        ax_scatter.axhline(x_obs, color="C3", linestyle=":", linewidth=0.9)
    ax_scatter.set_xlabel(r"prior draw $\mu$")
    ax_scatter.set_ylabel(r"simulator output $x = \mu + \epsilon$")
    ax_scatter.set_title(fr"training pairs $(\mu_i, x_i)$, $N = {n_train}$")
    fig.suptitle("Toy Laplace-mean problem: NPE posterior against the analytic posterior")
    fig.tight_layout()
    return fig


def plot_posterior_marginals(
    posterior_samples: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    truth: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        ax.hist(
            posterior_samples[:, j],
            bins=40,
            range=(prior_low[j], prior_high[j]),
            color="C0",
            alpha=0.85,
            density=True,
        )
        prior_density = 1.0 / (prior_high[j] - prior_low[j])
        ax.hlines(prior_density, prior_low[j], prior_high[j], color="grey", linestyles=":", label="prior")
        ax.axvline(truth[j], color="black", linestyle="--", linewidth=1.2, label="truth")
        ax.set_xlim(prior_low[j], prior_high[j])
        ax.set_xlabel(PARAM_LATEX[j])
        ax.set_yticks([])
        ax.set_title(PARAM_DESC[j])
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("NPE posterior marginals against a uniform prior")
    fig.tight_layout()
    return fig


def plot_posterior_pairs(posterior_samples: np.ndarray, truth: np.ndarray) -> plt.Figure:
    n_params = posterior_samples.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(9.5, 9.5))
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                ax.hist(posterior_samples[:, i], bins=30, color="C0", alpha=0.85, density=True)
                ax.axvline(truth[i], color="black", linestyle="--", linewidth=1.0)
                ax.set_yticks([])
            elif i > j:
                ax.scatter(
                    posterior_samples[:, j],
                    posterior_samples[:, i],
                    s=2,
                    alpha=0.15,
                    color="C0",
                )
                ax.axvline(truth[j], color="black", linestyle="--", linewidth=0.8)
                ax.axhline(truth[i], color="black", linestyle="--", linewidth=0.8)
            else:
                ax.axis("off")
            if i == n_params - 1:
                ax.set_xlabel(PARAM_LATEX[j])
            if j == 0 and i != j:
                ax.set_ylabel(PARAM_LATEX[i])
    fig.suptitle("Pairwise posterior draws")
    fig.tight_layout()
    return fig


def plot_posterior_predictive(
    predictive_summaries: np.ndarray,
    observed: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for j, ax in enumerate(axes):
        ax.hist(predictive_summaries[:, j], bins=30, color="C0", alpha=0.85, density=True)
        ax.axvline(observed[j], color="black", linestyle="--", linewidth=1.2, label="observed")
        ax.set_xlabel(SUMMARY_NAMES[j], fontsize=9)
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("density")
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Posterior-predictive summary statistics against the observed value")
    fig.tight_layout()
    return fig


def plot_smm_vs_npe(
    smm_objective: pd.DataFrame,
    posterior_beta: np.ndarray,
    smm_beta_hat: float,
    true_beta: float,
) -> plt.Figure:
    fig, ax_left = plt.subplots(figsize=(9, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        smm_objective["intensity beta"],
        smm_objective["objective"],
        color="C1",
        marker="o",
        markersize=4,
        label="SMM objective (grid search)",
    )
    ax_left.set_xlabel(r"intensity of choice $\beta$")
    ax_left.set_ylabel("SMM weighted moment distance", color="C1")
    ax_left.tick_params(axis="y", labelcolor="C1")

    ax_right.hist(
        posterior_beta,
        bins=40,
        range=(PRIOR_LOW[0], PRIOR_HIGH[0]),
        color="C0",
        alpha=0.5,
        density=True,
        label=r"NPE marginal posterior over $\beta$",
    )
    ax_right.set_ylabel("NPE posterior density", color="C0")
    ax_right.tick_params(axis="y", labelcolor="C0")

    ax_left.axvline(true_beta, color="black", linestyle=":", linewidth=1.2, label="truth")
    ax_left.axvline(smm_beta_hat, color="C3", linestyle="--", linewidth=1.2, label=fr"SMM $\hat\beta = {smm_beta_hat:.0f}$")
    ax_left.set_xlim(PRIOR_LOW[0], PRIOR_HIGH[0])
    handles_l, labels_l = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_l + handles_r, labels_l + labels_r, loc="upper right", fontsize=9)
    ax_left.set_title("SMM grid-search objective and NPE posterior over the intensity parameter")
    fig.tight_layout()
    return fig


def smm_grid_for_beta(base: Params, true_beta: float) -> tuple[pd.DataFrame, float]:
    """Reproduce the SMM grid search from the Brock-Hommes ABM tutorial."""
    candidate_betas = np.arange(2.0, 62.0, 2.0)
    data_rng = np.random.default_rng(2028)
    sim_rng = np.random.default_rng(2029)
    pseudo_data_shocks = data_rng.normal(0.0, base.shock_sigma, size=(8, base.periods))
    smm_shocks = sim_rng.normal(0.0, base.shock_sigma, size=(8, base.periods))
    target = average_moments(true_beta, base, pseudo_data_shocks)
    scale = {
        "volatility": max(abs(target["volatility"]), 0.01),
        "abs return autocorrelation": max(abs(target["abs return autocorrelation"]), 0.05),
        "excess kurtosis": max(abs(target["excess kurtosis"]), 0.25),
    }
    rows = []
    for beta in candidate_betas:
        fitted = average_moments(float(beta), base, smm_shocks)
        objective = sum(((fitted[k] - target[k]) / scale[k]) ** 2 for k in target)
        rows.append({"intensity beta": float(beta), "objective": float(objective)})
    grid = pd.DataFrame(rows)
    beta_hat = float(grid.loc[grid["objective"].idxmin(), "intensity beta"])
    return grid, beta_hat


def main() -> None:
    setup_style()
    torch.manual_seed(0)
    base = Params()

    print("Neural Posterior Estimation for Brock-Hommes")
    print(f"  prior box: low={PRIOR_LOW}, high={PRIOR_HIGH}")
    print(f"  truth: {TRUE_THETA}")

    prior = BoxUniform(
        low=torch.from_numpy(PRIOR_LOW).float(),
        high=torch.from_numpy(PRIOR_HIGH).float(),
    )

    rng = np.random.default_rng(2030)
    print(f"\nSimulating training set: {N_TRAIN} draws...")
    theta_train, x_train = build_training_set(prior, base, N_TRAIN, rng)
    print(f"  kept {theta_train.shape[0]} finite simulations")

    rng_obs = np.random.default_rng(20260510)
    x_obs_array = observed_summaries(base, rng_obs, N_OBSERVATIONS)
    x_obs_mean = x_obs_array.mean(axis=0)
    print(f"  observed summaries (mean over {N_OBSERVATIONS} runs): {x_obs_mean}")

    print("\nTraining NPE with masked autoregressive flow...")
    inference = NPE(prior=prior, density_estimator="maf", show_progress_bars=False)
    inference.append_simulations(theta_train, x_train)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    posterior.set_default_x(torch.from_numpy(x_obs_mean).float())

    print(f"\nSampling {N_POSTERIOR_DRAWS} posterior draws...")
    posterior_samples = posterior.sample((N_POSTERIOR_DRAWS,), show_progress_bars=False).numpy()
    posterior_mean = posterior_samples.mean(axis=0)
    posterior_std = posterior_samples.std(axis=0)
    posterior_q025 = np.quantile(posterior_samples, 0.025, axis=0)
    posterior_q975 = np.quantile(posterior_samples, 0.975, axis=0)

    rng_pred = np.random.default_rng(2031)
    pred_idx = rng_pred.integers(0, posterior_samples.shape[0], size=500)
    pred_seeds = rng_pred.integers(low=0, high=2**31 - 1, size=500).astype(np.int64)
    predictive_summaries = np.stack(
        [simulate_summary(posterior_samples[i], base, int(s)) for i, s in zip(pred_idx, pred_seeds)],
        axis=0,
    )

    print("\nRunning SMM grid search for the beta comparison panel...")
    smm_grid, smm_beta_hat = smm_grid_for_beta(base, true_beta=float(TRUE_THETA[0]))
    print(f"  SMM beta_hat = {smm_beta_hat:.1f}")

    print("\nPosterior summary:")
    for j, name in enumerate(PARAM_NAMES):
        print(
            f"  {name:<10}  truth={TRUE_THETA[j]:.4f}  "
            f"mean={posterior_mean[j]:.4f}  sd={posterior_std[j]:.4f}  "
            f"95%CI=[{posterior_q025[j]:.4f}, {posterior_q975[j]:.4f}]"
        )

    posterior_table = pd.DataFrame({
        "parameter": PARAM_NAMES,
        "truth": TRUE_THETA,
        "posterior mean": posterior_mean,
        "posterior sd": posterior_std,
        "ci 2.5%": posterior_q025,
        "ci 97.5%": posterior_q975,
    })

    summary_table = pd.DataFrame({
        "summary statistic": SUMMARY_NAMES,
        "observed": x_obs_mean,
        "posterior-predictive mean": predictive_summaries.mean(axis=0),
        "posterior-predictive sd": predictive_summaries.std(axis=0),
    })

    # --- Figures and tables ---
    save_figure(run_laplace_demo(), "figures/laplace-demo.png")

    save_figure(plot_posterior_marginals(posterior_samples, PRIOR_LOW, PRIOR_HIGH, TRUE_THETA),
                "figures/posterior-marginals.png", dpi=150)
    save_figure(plot_posterior_pairs(posterior_samples, TRUE_THETA),
                "figures/posterior-pairs.png", dpi=150)
    save_figure(plot_posterior_predictive(predictive_summaries, x_obs_mean),
                "figures/posterior-predictive.png", dpi=150)
    save_figure(plot_smm_vs_npe(smm_grid, posterior_samples[:, 0], smm_beta_hat, float(TRUE_THETA[0])),
                "figures/smm-vs-npe.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    posterior_table.to_csv("tables/posterior-summary.csv", index=False)
    summary_table.to_csv("tables/posterior-predictive.csv", index=False)

    save_thumbnail("figures/posterior-marginals.png", "figures/thumb.png")
    print("\nDone: 5 figures, 2 tables, thumb reproduced.")


if __name__ == "__main__":
    main()
