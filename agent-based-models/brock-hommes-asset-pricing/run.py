#!/usr/bin/env python3
"""Brock-Hommes asset pricing with strategy switching."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.brock_hommes import Params, Run, average_moments, moments, simulate
from lib.plotting import save_figure, save_thumbnail, setup_style


def estimate_smm(params: Params, true_beta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grid-search SMM estimator for the logit intensity parameter."""
    candidate_betas = np.arange(2.0, 62.0, 2.0)
    data_rng = np.random.default_rng(2028)
    sim_rng = np.random.default_rng(2029)
    pseudo_data_shocks = data_rng.normal(0.0, params.shock_sigma, size=(8, params.periods))
    smm_shocks = sim_rng.normal(0.0, params.shock_sigma, size=(8, params.periods))
    target = average_moments(true_beta, params, pseudo_data_shocks)
    scale = {
        "volatility": max(abs(target["volatility"]), 0.01),
        "abs return autocorrelation": max(abs(target["abs return autocorrelation"]), 0.05),
        "excess kurtosis": max(abs(target["excess kurtosis"]), 0.25),
    }

    rows = []
    for beta in candidate_betas:
        fitted = average_moments(float(beta), params, smm_shocks)
        objective = sum(((fitted[key] - target[key]) / scale[key]) ** 2 for key in target)
        rows.append({
            "intensity beta": float(beta),
            "objective": float(objective),
            "volatility": fitted["volatility"],
            "abs return autocorrelation": fitted["abs return autocorrelation"],
            "excess kurtosis": fitted["excess kurtosis"],
        })

    objective = pd.DataFrame(rows)
    beta_hat = float(objective.loc[objective["objective"].idxmin(), "intensity beta"])
    fitted = average_moments(beta_hat, params, smm_shocks)
    fit_table = pd.DataFrame([
        {"quantity": "intensity beta", "target": true_beta, "fit": beta_hat, "difference": beta_hat - true_beta},
        {"quantity": "volatility", "target": target["volatility"], "fit": fitted["volatility"], "difference": fitted["volatility"] - target["volatility"]},
        {"quantity": "abs return autocorrelation", "target": target["abs return autocorrelation"], "fit": fitted["abs return autocorrelation"], "difference": fitted["abs return autocorrelation"] - target["abs return autocorrelation"]},
        {"quantity": "excess kurtosis", "target": target["excess kurtosis"], "fit": fitted["excess kurtosis"], "difference": fitted["excess kurtosis"] - target["excess kurtosis"]},
    ])
    return objective, fit_table


def plot_price_paths(runs: list[Run], params: Params) -> plt.Figure:
    """Price deviations from the rational-expectations fundamental."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, sharey=True)
    time = np.arange(params.periods)
    labels = ["Low intensity", "Medium intensity", "High intensity"]
    for ax, run, label in zip(axes, runs, labels):
        ax.plot(time, run.x, color="C0")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.1, label="RE fundamental")
        ax.fill_between(time, -0.05, 0.05, color="grey", alpha=0.12, label="near fundamental")
        ax.set_ylabel("$x_t = p_t - p^{\\ast}$")
        ax.set_title(f"{label}: $\\beta = {run.beta:.0f}$")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Period $t$")
    fig.suptitle("Brock-Hommes price deviations from the dividend fundamental")
    fig.tight_layout()
    return fig


def plot_strategy_shares(runs: list[Run], params: Params) -> plt.Figure:
    """Trend-follower shares under low, medium, and high intensity."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    time = np.arange(params.periods)
    for run, color in zip(runs, ["C0", "C1", "C3"]):
        ax.plot(time, run.shares[:, 1], color=color, label=rf"$\beta = {run.beta:.0f}$")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Trend-follower share")
    ax.set_title("Logit switching responds to lagged realized forecasting profits")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_moment_fit(objective: pd.DataFrame, fit_table: pd.DataFrame) -> plt.Figure:
    """SMM objective over candidate intensity values."""
    true_beta = float(fit_table.loc[fit_table["quantity"] == "intensity beta", "target"].iloc[0])
    beta_hat = float(fit_table.loc[fit_table["quantity"] == "intensity beta", "fit"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(objective["intensity beta"], objective["objective"], color="C0", marker="o", markersize=4)
    ax.axvline(true_beta, color="black", linestyle=":", linewidth=1.2, label=f"true $\\beta = {true_beta:.0f}$")
    ax.axvline(beta_hat, color="C3", linestyle="--", linewidth=1.2, label=f"SMM $\\hat\\beta = {beta_hat:.0f}$")
    ax.set_xlabel("Candidate intensity of choice $\\beta$")
    ax.set_ylabel("Weighted moment distance")
    ax.set_title("SMM objective for the strategy-switching intensity")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()
    params = Params()
    betas = [2.0, 20.0, 50.0]
    path_shocks = np.random.default_rng(2026).normal(0.0, params.shock_sigma, params.periods)
    runs = [simulate(beta, params, seed=2026, shocks=path_shocks) for beta in betas]
    true_beta = 30.0
    objective, fit_table = estimate_smm(params, true_beta)
    beta_hat = float(fit_table.loc[fit_table["quantity"] == "intensity beta", "fit"].iloc[0])
    high_moments = moments(runs[-1].x, params.burn)

    print("Brock-Hommes asset pricing")
    print(f"  Fundamental price p* = {params.p_star:.2f}")
    print(f"  High-intensity return volatility = {high_moments['volatility']:.4f}")
    print(f"  SMM beta estimate = {beta_hat:.1f} (true {true_beta:.1f})")

    fig_paths = plot_price_paths(runs, params)
    save_figure(fig_paths, "figures/price-paths.png", dpi=150)

    fig_shares = plot_strategy_shares(runs, params)
    save_figure(fig_shares, "figures/strategy-shares.png", dpi=150)

    fig_fit = plot_moment_fit(objective, fit_table)
    save_figure(fig_fit, "figures/moment-fit.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    fit_table.to_csv("tables/smm-fit.csv", index=False)

    save_thumbnail("figures/price-paths.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
