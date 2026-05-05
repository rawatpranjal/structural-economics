#!/usr/bin/env python3
"""Mean-variance portfolio frontier."""
from __future__ import annotations

import itertools
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


def _restricted_min_variance(target_return: float, active_assets: tuple[int, ...]) -> tuple[np.ndarray, float] | None:
    """Solve the minimum-variance problem on one active asset set."""
    idx = np.array(active_assets)
    cov = covariance_matrix()[np.ix_(idx, idx)]
    mean = EXPECTED_RETURNS[idx]
    if len(idx) == 1:
        if not np.isclose(target_return, mean[0], atol=1e-10):
            return None
        weights = np.zeros(len(ASSETS))
        weights[idx[0]] = 1.0
        return weights, float(cov[0, 0])

    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(idx))
    a = float(ones @ inv_cov @ ones)
    b = float(ones @ inv_cov @ mean)
    c = float(mean @ inv_cov @ mean)
    d = a * c - b**2
    if d <= 1e-14:
        return None

    lambda_budget = (c - b * target_return) / d
    lambda_return = (a * target_return - b) / d
    restricted_weights = inv_cov @ (lambda_budget * ones + lambda_return * mean)
    if np.any(restricted_weights < -1e-10):
        return None

    restricted_weights = np.maximum(restricted_weights, 0.0)
    restricted_weights = restricted_weights / restricted_weights.sum()
    weights = np.zeros(len(ASSETS))
    weights[idx] = restricted_weights
    variance = float(weights @ covariance_matrix() @ weights)
    return weights, variance


def exact_long_only_frontier(target_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact long-only frontier by enumerating active sets."""
    risks = np.full_like(target_returns, np.nan, dtype=float)
    asset_indices = range(len(ASSETS))
    active_sets = [
        active
        for size in range(1, len(ASSETS) + 1)
        for active in itertools.combinations(asset_indices, size)
    ]

    for i, target_return in enumerate(target_returns):
        best_variance = np.inf
        for active in active_sets:
            candidate = _restricted_min_variance(float(target_return), active)
            if candidate is None:
                continue
            _, variance = candidate
            best_variance = min(best_variance, variance)
        if np.isfinite(best_variance):
            risks[i] = np.sqrt(best_variance)
    return target_returns, risks


def global_minimum_variance_weights() -> np.ndarray:
    """Return unconstrained global minimum-variance weights."""
    cov = covariance_matrix()
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(len(ASSETS))
    return (inv_cov @ ones) / float(ones @ inv_cov @ ones)


def long_only_minimum_variance_weights() -> np.ndarray:
    """Return the minimum-variance portfolio with no short sales."""
    cov = covariance_matrix()
    best_weights = np.eye(len(ASSETS))[0]
    best_variance = float(best_weights @ cov @ best_weights)
    asset_indices = range(len(ASSETS))

    for size in range(2, len(ASSETS) + 1):
        for active in itertools.combinations(asset_indices, size):
            idx = np.array(active)
            sub_cov = cov[np.ix_(idx, idx)]
            inv_cov = np.linalg.inv(sub_cov)
            ones = np.ones(len(idx))
            restricted_weights = (inv_cov @ ones) / float(ones @ inv_cov @ ones)
            if np.any(restricted_weights < -1e-10):
                continue
            weights = np.zeros(len(ASSETS))
            weights[idx] = np.maximum(restricted_weights, 0.0)
            weights = weights / weights.sum()
            variance = float(weights @ cov @ weights)
            if variance < best_variance:
                best_weights = weights
                best_variance = variance
    return best_weights


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
    gmv_unconstrained = global_minimum_variance_weights()
    gmv_long_only = long_only_minimum_variance_weights()
    tangency = tangency_weights()
    best_long_only = simulated.loc[simulated["Sharpe"].idxmax(), ASSETS].to_numpy()
    target_returns = np.linspace(0.02, 0.10, 160)
    frontier_returns, frontier_risks = frontier(target_returns)
    long_only_targets = np.linspace(float(EXPECTED_RETURNS.min()), float(EXPECTED_RETURNS.max()), 180)
    long_only_returns, long_only_risks = exact_long_only_frontier(long_only_targets)
    summary = pd.DataFrame(
        [
            portfolio_row("Unconstrained global min variance", gmv_unconstrained),
            portfolio_row("Exact long-only global min variance", gmv_long_only),
            portfolio_row("Tangency portfolio", tangency),
            portfolio_row("Best random long-only Sharpe", best_long_only),
        ]
    )

    print("Mean-variance frontier")
    print(summary[["Portfolio", "Return", "Risk", "Sharpe"]].to_string(index=False))

    report = ModelReport(
        "Mean-Variance Portfolio Frontier",
        "How covariance and short-sale constraints shape the portfolio frontier.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The Markowitz problem is a compact way to ask what diversification actually buys. "
        "A portfolio's mean return is linear in the weights, but its risk is not: covariance "
        "decides whether combining assets smooths payoffs or simply repackages the same "
        "aggregate risk.\n\n"
        "The example uses four stylized annual asset classes. Random long-only portfolios "
        "make the feasible set visible, but they are only a simulation. The tutorial also "
        "computes the exact long-only frontier and the unconstrained analytic frontier, so "
        "the reader can see which features are economic restrictions and which are sampling "
        "noise from drawing many portfolios. The usual empirical warning still applies: a "
        "frontier is only as credible as the expected returns and covariance matrix fed "
        "into it."
    )

    report.add_equations(
        r"""
Let $i=1,\ldots,N$ index risky assets. A portfolio is a vector of weights
$w=(w_1,\ldots,w_N)^\top$ with budget constraint $\mathbf{1}^\top w=1$. Let
$\mu$ collect expected risky-asset returns and let $\Sigma$ be the positive
definite covariance matrix of returns. The portfolio mean and variance are

$$
\mu_p(w)=w^\top \mu,
\qquad
\sigma_p^2(w)=w^\top \Sigma w.
$$

For a target mean return $m$, the unconstrained Markowitz frontier solves

$$
\min_w w^\top \Sigma w
\quad \text{subject to} \quad
w^\top \mu=m,\quad \mathbf{1}^\top w=1.
$$

With

$$
A=\mathbf{1}^\top\Sigma^{-1}\mathbf{1},\quad
B=\mathbf{1}^\top\Sigma^{-1}\mu,\quad
C=\mu^\top\Sigma^{-1}\mu,\quad
D=AC-B^2,
$$

the minimum variance at target $m$ is

$$
\sigma^2(m)=\frac{A m^2-2 B m+C}{D}.
$$

Adding no-short-sale constraints gives the long-only problem:

$$
\min_w w^\top\Sigma w
\quad\text{subject to}\quad
w^\top\mu=m,\quad \mathbf{1}^\top w=1,\quad w_i\geq 0.
$$

The tangency portfolio for risk-free rate $r_f$ maximizes the Sharpe ratio,

$$
\max_w \frac{w^\top\mu-r_f}{\sqrt{w^\top\Sigma w}},
$$

and, without short-sale constraints, has weights proportional to
$\Sigma^{-1}(\mu-r_f\mathbf{1})$ and normalized to sum to one.
"""
    )

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|--------|-------|------|\n"
        + "\n".join(
            f"| {asset} | mean {100.0 * ret:.1f}%, volatility {100.0 * vol:.1f}% | risky asset class |"
            for asset, ret, vol in zip(ASSETS, EXPECTED_RETURNS, VOLATILITIES)
        )
        + f"\n| Risk-free rate $r_f$ | {100.0 * RISK_FREE_RATE:.1f}% | Sharpe-ratio benchmark |"
        + "\n| Correlations | fixed $4\\times4$ matrix | determines diversification value |"
        + f"\n| Random portfolios | {len(simulated):,} Dirichlet draws | visual approximation to the long-only set |"
        + "\n| Exact long-only frontier | active-set enumeration | benchmark for the random cloud |"
    )

    report.add_solution_method(
        "The numerical work separates three objects that are often conflated. Random "
        "Dirichlet weights give a picture of the long-only feasible set. The "
        "unconstrained frontier comes from the Lagrange-multiplier formula above. The exact "
        "long-only frontier is computed by enumerating active asset sets; for each candidate "
        "set, the same two-constraint Markowitz problem is solved on that face of the simplex "
        "and discarded if any weight is negative.\n\n"
        "```text\n"
        "Algorithm: Markowitz frontier with a long-only benchmark\n"
        "Input: expected returns mu, covariance matrix Sigma, risk-free rate r_f, target grid M\n"
        "Output: random portfolios, unconstrained frontier, exact long-only frontier, selected weights\n"
        "Draw many long-only portfolios w from a symmetric Dirichlet distribution\n"
        "For each draw, compute mu_p(w), sigma_p(w), and the Sharpe ratio\n"
        "For each target return m in M:\n"
        "    compute the unconstrained frontier variance from A, B, C, and D\n"
        "    initialize the long-only variance at infinity\n"
        "    for each nonempty active asset set S:\n"
        "        solve the Markowitz problem using only assets in S\n"
        "        if all restricted weights are nonnegative:\n"
        "            keep the candidate if it has the lowest variance so far\n"
        "Compute the global minimum-variance and tangency portfolios\n"
        "Compare the best random long-only Sharpe portfolio with the exact frontier\n"
        "```\n\n"
        "Because there are only four assets, active-set enumeration is cleaner than adding a "
        "quadratic-programming dependency. It also makes the economic constraint explicit. "
        "The long-only frontier is the lower envelope of feasible faces of the portfolio "
        "simplex."
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
    ax1.plot(
        100.0 * frontier_risks,
        100.0 * frontier_returns,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Unconstrained frontier",
    )
    ax1.plot(
        100.0 * long_only_risks,
        100.0 * long_only_returns,
        color="tab:blue",
        linewidth=2.4,
        label="Exact long-only frontier",
    )
    for label, weights, color in [
        ("Unconstrained GMV", gmv_unconstrained, "tab:red"),
        ("Long-only GMV", gmv_long_only, "tab:green"),
        ("Tangency", tangency, "tab:orange"),
        ("Best random Sharpe", best_long_only, "tab:purple"),
    ]:
        ret, risk = portfolio_moments(weights[None, :], EXPECTED_RETURNS, covariance_matrix())
        ax1.scatter(100.0 * risk, 100.0 * ret, color=color, s=70, label=label)
    ax1.set_xlabel("Portfolio risk, standard deviation (%)")
    ax1.set_ylabel("Expected return (%)")
    ax1.set_title("Mean-Variance Frontier")
    ax1.legend()
    fig1.colorbar(scatter, ax=ax1, label="Sharpe ratio")
    report.add_results(
        "The random cloud is a Monte Carlo picture of the long-only simplex. Its lower-left "
        "edge is close to, but not identical to, the exact long-only frontier. The dashed "
        "unconstrained frontier extends beyond that curve because it can use short positions "
        "or leverage. That distinction is substantive; allowing negative weights changes the "
        "choice set, not just the algorithm."
    )
    report.add_figure(
        "figures/frontier.png",
        "Mean-variance frontier comparison",
        fig1,
    )

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))
    weight_df = pd.DataFrame(
        [gmv_unconstrained, gmv_long_only, tangency, best_long_only],
        columns=ASSETS,
        index=["Unconstrained GMV", "Long-only GMV", "Tangency", "Best random Sharpe"],
    )
    weight_df.plot(kind="bar", ax=ax2, width=0.78)
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("")
    ax2.set_ylabel("Portfolio weight")
    ax2.set_title("Portfolio Weights")
    ax2.tick_params(axis="x", rotation=18)
    ax2.legend(ncol=2)
    report.add_results(
        "The weight plot shows why the frontier comparison matters. The unconstrained global "
        "minimum-variance portfolio uses a small short position, while the long-only version "
        "moves to the nearest feasible allocation. In this calibration the tangency portfolio "
        "is already long-only, so the best random Sharpe draw lands close to the analytic "
        "tangency weights."
    )
    report.add_figure(
        "figures/portfolio-weights.png",
        "Weights for selected portfolios",
        fig2,
    )

    report.add_table(
        "tables/portfolio-summary.csv",
        "Selected portfolio summaries",
        summary,
        description=(
            "The table reports annualized moments. The tiny difference between the analytic "
            "tangency portfolio and the best random long-only portfolio is simulation error, "
            "not a different economic optimum."
        ),
    )

    report.add_takeaway(
        "The mean-variance frontier makes covariance and constraints visible in the same "
        "object. It is also fragile. Small changes in expected returns or covariances can "
        "move the tangency portfolio sharply, so the computation should be read as a "
        "disciplined mapping from inputs to portfolio tradeoffs, not as a standalone "
        "investment rule."
    )
    report.add_references(
        [
            "[Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.](https://doi.org/10.2307/2975974)",
        ]
    )
    report.write()


if __name__ == "__main__":
    main()
