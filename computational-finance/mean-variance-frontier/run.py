#!/usr/bin/env python3
"""Mean-variance portfolio frontier."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


ASSETS = ["Bills", "Bonds", "Equity", "Real assets"]
EXPECTED_RETURNS = np.array([0.025, 0.045, 0.085, 0.065])
VOLATILITIES = np.array([0.010, 0.060, 0.160, 0.120])
CORRELATION = np.array(
    [
        [1.00, 0.20, 0.05, 0.00],
        [0.20, 1.00, 0.25, 0.15],
        [0.05, 0.25, 1.00, 0.45],
        [0.00, 0.15, 0.45, 1.00],
    ]
)
RISK_FREE_RATE = 0.02


def covariance_matrix() -> np.ndarray:
    """Return asset covariance matrix from volatilities and correlations."""
    return np.outer(VOLATILITIES, VOLATILITIES) * CORRELATION


def portfolio_moments(weights: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return portfolio expected returns and standard deviations."""
    returns = weights @ mean
    variances = np.einsum("ij,jk,ik->i", weights, cov, weights)
    return returns, np.sqrt(variances)


def random_long_only_portfolios(n: int = 5000, seed: int = 2026) -> pd.DataFrame:
    """Simulate random long-only portfolio weights and moments."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(ASSETS)), size=n)
    mean = EXPECTED_RETURNS
    cov = covariance_matrix()
    returns, risks = portfolio_moments(weights, mean, cov)
    sharpe = (returns - RISK_FREE_RATE) / risks
    df = pd.DataFrame(weights, columns=ASSETS)
    df["Return"] = returns
    df["Risk"] = risks
    df["Sharpe"] = sharpe
    return df


def frontier(target_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute unconstrained minimum-variance risk for target returns."""
    cov = covariance_matrix()
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(ASSETS))
    mean = EXPECTED_RETURNS
    a = float(ones @ inv_cov @ ones)
    b = float(ones @ inv_cov @ mean)
    c = float(mean @ inv_cov @ mean)
    d = a * c - b**2
    variances = (a * target_returns**2 - 2.0 * b * target_returns + c) / d
    return target_returns, np.sqrt(variances)


def global_minimum_variance_weights() -> np.ndarray:
    """Return unconstrained global minimum-variance weights."""
    cov = covariance_matrix()
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(ASSETS))
    return (inv_cov @ ones) / float(ones @ inv_cov @ ones)


def tangency_weights() -> np.ndarray:
    """Return unconstrained tangency portfolio weights."""
    cov = covariance_matrix()
    inv_cov = np.linalg.inv(cov)
    excess = EXPECTED_RETURNS - RISK_FREE_RATE
    raw = inv_cov @ excess
    return raw / raw.sum()


def portfolio_row(name: str, weights: np.ndarray) -> dict[str, str]:
    """Format portfolio weights and moments for reporting."""
    mean = EXPECTED_RETURNS
    cov = covariance_matrix()
    ret, risk = portfolio_moments(weights[None, :], mean, cov)
    row = {
        "Portfolio": name,
        "Return": f"{100.0 * ret[0]:.2f}%",
        "Risk": f"{100.0 * risk[0]:.2f}%",
        "Sharpe": f"{(ret[0] - RISK_FREE_RATE) / risk[0]:.2f}",
    }
    for asset, weight in zip(ASSETS, weights):
        row[asset] = f"{100.0 * weight:.1f}%"
    return row


def main() -> None:
    setup_style()
    simulated = random_long_only_portfolios()
    gmv = global_minimum_variance_weights()
    tangency = tangency_weights()
    best_long_only = simulated.loc[simulated["Sharpe"].idxmax(), ASSETS].to_numpy()
    target_returns = np.linspace(0.02, 0.10, 160)
    frontier_returns, frontier_risks = frontier(target_returns)
    summary = pd.DataFrame(
        [
            portfolio_row("Global min variance", gmv),
            portfolio_row("Tangency", tangency),
            portfolio_row("Best simulated long-only Sharpe", best_long_only),
        ]
    )

    print("Mean-variance frontier")
    print(summary[["Portfolio", "Return", "Risk", "Sharpe"]].to_string(index=False))

    report = ModelReport(
        "Mean-Variance Portfolio Frontier",
        "Diversification, covariance, and the efficient frontier in a small portfolio model.",
    )

    report.add_overview(
        "The source notebook simulated random portfolio weights. This tutorial keeps that "
        "intuition but adds the analytic Markowitz frontier so the geometry is clear.\n\n"
        "The inputs are synthetic annual expected returns, volatilities, and correlations. "
        "They are chosen for teaching, not investment advice. The important lesson is that "
        "risk depends on covariances and that the frontier is highly sensitive to estimated "
        "means and covariances."
    )

    report.add_equations(
        r"""
For portfolio weights $w$, expected return is

$$
\mu_p = w^\top \mu.
$$

Portfolio variance is

$$
\sigma_p^2 = w^\top \Sigma w.
$$

The efficient frontier solves the minimum-variance problem for each target
return:

$$
\min_w w^\top \Sigma w
\quad \text{subject to} \quad
w^\top \mu = \mu_p,\quad w^\top \mathbf{1} = 1.
$$
"""
    )

    report.add_model_setup(
        "| Asset | Expected return | Volatility |\n"
        "|-------|-----------------|------------|\n"
        + "\n".join(
            f"| {asset} | {100.0 * ret:.1f}% | {100.0 * vol:.1f}% |"
            for asset, ret, vol in zip(ASSETS, EXPECTED_RETURNS, VOLATILITIES)
        )
        + f"\n| Risk-free rate | {100.0 * RISK_FREE_RATE:.1f}% | Used for Sharpe ratios |"
    )

    report.add_solution_method(
        "The script simulates random long-only portfolios from a Dirichlet distribution and "
        "computes their mean, variance, and Sharpe ratio. It also solves the unconstrained "
        "Markowitz formulas for the global minimum-variance portfolio, the tangency portfolio, "
        "and the continuous frontier."
    )

    fig1, ax1 = plt.subplots(figsize=(7.4, 5.4))
    scatter = ax1.scatter(
        100.0 * simulated["Risk"],
        100.0 * simulated["Return"],
        c=simulated["Sharpe"],
        cmap="viridis",
        s=12,
        alpha=0.7,
    )
    ax1.plot(100.0 * frontier_risks, 100.0 * frontier_returns, color="black", linewidth=2.0, label="Analytic frontier")
    for label, weights, color in [
        ("GMV", gmv, "tab:red"),
        ("Tangency", tangency, "tab:orange"),
        ("Best long-only", best_long_only, "tab:purple"),
    ]:
        ret, risk = portfolio_moments(weights[None, :], EXPECTED_RETURNS, covariance_matrix())
        ax1.scatter(100.0 * risk, 100.0 * ret, color=color, s=70, label=label)
    ax1.set_xlabel("Portfolio risk, standard deviation (%)")
    ax1.set_ylabel("Expected return (%)")
    ax1.set_title("Mean-Variance Frontier")
    ax1.legend()
    fig1.colorbar(scatter, ax=ax1, label="Sharpe ratio")
    report.add_figure(
        "figures/frontier.png",
        "Simulated portfolios and analytic frontier",
        fig1,
        description=(
            "Random portfolios fill the feasible long-only region. The analytic frontier shows "
            "the best risk-return tradeoff when short positions are allowed by the formula."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))
    weight_df = pd.DataFrame(
        [gmv, tangency, best_long_only],
        columns=ASSETS,
        index=["GMV", "Tangency", "Best long-only"],
    )
    x = np.arange(len(weight_df.index))
    bottom = np.zeros(len(weight_df.index))
    for asset in ASSETS:
        values = weight_df[asset].to_numpy()
        ax2.bar(x, values, bottom=bottom, label=asset)
        bottom += values
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(weight_df.index)
    ax2.set_ylabel("Portfolio weight")
    ax2.set_title("Portfolio Weights")
    ax2.legend(ncol=2)
    report.add_figure(
        "figures/portfolio-weights.png",
        "Weights for selected portfolios",
        fig2,
        description=(
            "The unconstrained tangency portfolio can use negative or levered positions. That "
            "is a mathematical frontier object, not a recommendation."
        ),
    )

    report.add_table(
        "tables/portfolio-summary.csv",
        "Selected portfolio summaries",
        summary,
    )

    report.add_takeaway(
        "Markowitz's key insight is covariance. A portfolio is not just a weighted average "
        "of standalone risks, because assets move together. The practical caveat is equally "
        "important: frontiers are input-sensitive, especially to expected returns."
    )
    report.add_references(
        [
            "[Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.](https://doi.org/10.2307/2975974)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
