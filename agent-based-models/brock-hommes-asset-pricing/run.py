#!/usr/bin/env python3
"""Brock-Hommes asset pricing with strategy switching."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


@dataclass(frozen=True)
class Params:
    """Primitive parameters for the two-rule asset market."""

    R: float = 1.01
    dividend: float = 0.20
    trend_gain: float = 1.40
    forecast_bound: float = 0.35
    shock_sigma: float = 0.02
    memory: float = 0.80
    risk_scale: float = 0.04
    trend_cost: float = 0.001
    periods: int = 700
    burn: int = 100
    x_lag: float = 0.10
    x0: float = 0.12

    @property
    def p_star(self) -> float:
        return self.dividend / (self.R - 1.0)


@dataclass(frozen=True)
class Run:
    """Output from one Brock-Hommes simulation."""

    beta: float
    x: np.ndarray
    shares: np.ndarray
    scores: np.ndarray
    profits: np.ndarray


def softmax(scores: np.ndarray, beta: float) -> np.ndarray:
    """Stable logit probabilities for the two forecasting rules."""
    z = beta * (scores - np.max(scores))
    weights = np.exp(np.clip(z, -60.0, 60.0))
    return weights / weights.sum()


def rule_forecasts(x_lag: float, x_lag2: float, params: Params) -> np.ndarray:
    """Fundamentalist and bounded trend forecasts of next deviation."""
    raw_trend = x_lag + params.trend_gain * (x_lag - x_lag2)
    trend = params.forecast_bound * np.tanh(raw_trend / params.forecast_bound)
    return np.array([0.0, trend])


def simulate(beta: float, params: Params, seed: int, shocks: np.ndarray | None = None) -> Run:
    """Simulate price deviations and logit rule shares."""
    rng = np.random.default_rng(seed)
    if shocks is None:
        shocks = rng.normal(0.0, params.shock_sigma, params.periods)
    else:
        shocks = np.asarray(shocks, dtype=float)
        if shocks.size < params.periods:
            raise ValueError("shock array is shorter than the simulation horizon")

    x = np.empty(params.periods)
    x[0] = params.x_lag
    x[1] = params.x0
    shares = np.empty((params.periods, 2))
    scores = np.zeros((params.periods, 2))
    profits = np.zeros((params.periods, 2))
    current_scores = np.zeros(2)
    costs = np.array([0.0, params.trend_cost])

    shares[0] = softmax(current_scores, beta)
    shares[1] = shares[0]

    for t in range(2, params.periods):
        forecasts = rule_forecasts(x[t - 1], x[t - 2], params)
        x[t] = float(shares[t - 1] @ forecasts / params.R + shocks[t])

        realized_excess = x[t] - params.R * x[t - 1]
        forecast_excess = forecasts - params.R * x[t - 1]
        period_profit = realized_excess * forecast_excess / params.risk_scale - costs
        current_scores = params.memory * current_scores + (1.0 - params.memory) * period_profit

        profits[t] = period_profit
        scores[t] = current_scores
        shares[t] = softmax(current_scores, beta)

    return Run(beta=beta, x=x, shares=shares, scores=scores, profits=profits)


def moments(x: np.ndarray, burn: int) -> dict[str, float]:
    """Return moments used by the SMM exercise."""
    returns = np.diff(x)[burn:]
    abs_returns = np.abs(returns)
    z = (returns - returns.mean()) / returns.std()
    return {
        "volatility": float(np.std(returns)),
        "abs return autocorrelation": float(np.corrcoef(abs_returns[1:], abs_returns[:-1])[0, 1]),
        "excess kurtosis": float(np.mean(z**4) - 3.0),
    }


def average_moments(beta: float, params: Params, shock_bank: np.ndarray) -> dict[str, float]:
    """Average moments over common random numbers."""
    rows = [
        moments(simulate(beta, params, seed=20_000 + i, shocks=shocks).x, params.burn)
        for i, shocks in enumerate(shock_bank)
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


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

    report = ModelReport(
        "Brock-Hommes Asset Pricing with Strategy Switching",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Forward-looking asset prices depend on beliefs about future prices. In "
        "the Brock-Hommes model, traders do not all hold the same belief. Some "
        "expect prices to return to the dividend fundamental. Others extrapolate "
        "recent price movements.\n\n"
        "Extrapolative beliefs can move prices away from fundamentals because "
        "forecasting rules gain followers after they make money. When trend "
        "followers earn high recent profits, logit choice shifts more traders "
        "toward the trend rule. The shift can amplify the original movement and "
        "make volatility persistent.\n\n"
        "The simulation compares low, medium, and high intensity of choice. A "
        "small SMM exercise then estimates the switching intensity from simulated "
        "return moments."
    )

    report.add_equations(
        rf"""
The risky asset pays a constant dividend $d$. The gross risk-free return is
$R = 1+r$. If dividends are constant and all traders expect the same future
price, the no-arbitrage price is also constant. It equals the present value of
the dividend stream:

$$p^{{\ast}} = \frac{{d}}{{R-1}}.$$

The model studies deviations from that benchmark. Let
$x_t = p_t - p^{{\ast}}$. A positive $x_t$ means the asset is priced above the
dividend fundamental. A negative $x_t$ means it is priced below it.

Fundamentalists forecast that the deviation will disappear:

$$f_{{F,t}} = 0,$$

Trend followers forecast that recent price movements will continue:

$$\tilde f_{{T,t}} = x_{{t-1}} + g(x_{{t-1}} - x_{{t-2}}),$$

$$f_{{T,t}} = \bar x \tanh(\tilde f_{{T,t}} / \bar x).$$

The hyperbolic tangent bounds the trend forecast by $\bar x$. This keeps the
experiment focused on bounded bubbles rather than numerical explosion.

Market clearing sets today's deviation equal to a weighted average of beliefs,
scaled by the risk-free return, plus noise-trader supply:

$$x_t = \frac{{n_{{F,t-1}} f_{{F,t}} + n_{{T,t-1}} f_{{T,t}}}}{{R}} + \epsilon_t.$$

The realized excess return in deviation form is $e_t = x_t - R x_{{t-1}}$.
Rule $h$ forecasted excess return $f_{{h,t}} - R x_{{t-1}}$. A rule earns a
high profit score when its forecasted position has the same sign as the
realized excess return:

$$\pi_{{h,t}} =
\frac{{e_t (f_{{h,t}} - R x_{{t-1}})}}{{a\sigma^2}} - c_h.$$

Scores are smoothed,

$$U_{{h,t}} = \lambda U_{{h,t-1}} + (1-\lambda)\pi_{{h,t}},$$

and next-period rule shares follow logit choice:

$$n_{{h,t}} =
\frac{{\exp(\beta U_{{h,t}})}}{{\exp(\beta U_{{F,t}}) + \exp(\beta U_{{T,t}})}}.$$

The parameter $\beta$ is the intensity of choice. As $\beta \to 0$, shares stay
near one half. As $\beta$ rises, small score gaps produce large reallocations
across forecasting rules.
"""
    )

    report.add_model_setup(
        "The calibration is intentionally small. The point is to make strategy "
        "switching visible, not to match a particular stock market.\n\n"
        f"| Object | Symbol | Value | Role |\n"
        f"|---|---:|---:|---|\n"
        f"| Gross risk-free return | $R$ | {params.R:.2f} | Discounting benchmark |\n"
        f"| Dividend | $d$ | {params.dividend:.2f} | Constant cash flow |\n"
        f"| Fundamental price | $p^{{\\ast}}$ | {params.p_star:.2f} | RE steady state |\n"
        f"| Trend gain | $g$ | {params.trend_gain:.2f} | Extrapolation strength |\n"
        f"| Forecast bound | $\\bar x$ | {params.forecast_bound:.2f} | Finite trend forecast |\n"
        f"| Shock scale | $\\sigma_\\epsilon$ | {params.shock_sigma:.2f} | Noise-trader supply shock |\n"
        f"| Score memory | $\\lambda$ | {params.memory:.2f} | Profit-score persistence |\n"
        f"| Trend cost | $c_T$ | {params.trend_cost:.3f} | Small information or trading cost |\n"
        f"| Simulation horizon | $T$ | {params.periods} | Price periods |\n"
        f"| Burn-in | $T_0$ | {params.burn} | Moments discard early periods |\n\n"
        f"The plotted intensity values are $\\beta = {betas[0]:.0f}$, "
        f"$\\beta = {betas[1]:.0f}$, and $\\beta = {betas[2]:.0f}$. The SMM "
        f"exercise sets the true value to $\\beta_0 = {true_beta:.0f}$ and "
        "searches over even candidates from 2 to 60."
    )

    report.add_solution_method(
        "The solution is simulation plus a moment-matching outer loop. There is "
        "no representative-agent Euler equation after beliefs become endogenous, "
        "because today's shares are state variables created by past forecast "
        "profits.\n\n"
        "```text\n"
        "Algorithm: Brock-Hommes simulation and SMM\n"
        "Input: R, d, g, xbar, lambda, beta, shocks epsilon_t\n"
        "Output: price deviations x_t, strategy shares n_ht, return moments\n\n"
        "1. Set p* = d / (R - 1), x_t = p_t - p*, and initialize x_0, x_1.\n"
        "   Set initial scores U_F = U_T = 0 and shares n_F = n_T = 1/2.\n"
        "2. For t = 2 to T:\n"
        "   2a. Forecast deviations:\n"
        "       f_F,t = 0\n"
        "       f_T,t = xbar * tanh((x_{t-1} + g * (x_{t-1} - x_{t-2})) / xbar)\n"
        "   2b. Clear the asset market:\n"
        "       x_t = (n_F,t-1 * f_F,t + n_T,t-1 * f_T,t) / R + epsilon_t\n"
        "   2c. Compute excess return:\n"
        "       e_t = x_t - R * x_{t-1}\n"
        "   2d. Score each rule h by realized forecast profit:\n"
        "       pi_h,t = e_t * (f_h,t - R * x_{t-1}) / (a * sigma^2) - c_h\n"
        "   2e. Smooth scores:\n"
        "       U_h,t = lambda * U_h,t-1 + (1 - lambda) * pi_h,t\n"
        "   2f. Update rule shares with logit choice:\n"
        "       n_h,t = exp(beta * U_h,t) / sum_j exp(beta * U_j,t)\n"
        "3. For SMM, compute m(beta): volatility, autocorrelation of absolute\n"
        "   returns, and excess kurtosis after burn-in.\n"
        "4. Choose beta_hat = argmin_beta ||W * (m(beta) - m_data)||^2.\n"
        "```\n\n"
        "The pseudo-data moments come from one deterministic shock bank. The "
        "candidate simulations use a separate deterministic shock bank. Within "
        "the grid, every candidate intensity sees the same candidate shocks, so "
        "the objective mostly reflects strategy switching rather than Monte "
        "Carlo noise."
    )

    report.add_results(
        "With low intensity, deviations are short-lived and remain close to zero. "
        "At medium intensity, trend followers gain share after profitable runs, "
        "so deviations last longer. At high intensity, the rule that just earned "
        "profits can briefly dominate the market, creating bubble-like departures "
        "before reversal profits pull agents back toward fundamentalists."
    )
    report.add_figure(
        "figures/price-paths.png",
        "Price deviations under low, medium, and high intensity of choice",
        plot_price_paths(runs, params),
    )

    report.add_results(
        "The share plot shows the channel behind the price paths. Low intensity "
        "keeps the trend-follower share near one half. High intensity turns "
        "small score gaps into near-corner allocations, so the same shock process "
        "generates much larger price movements."
    )
    report.add_figure(
        "figures/strategy-shares.png",
        "Trend-follower share under logit strategy switching",
        plot_strategy_shares(runs, params),
    )

    report.add_results(
        "The original paper pushes this logic further with local bifurcation "
        "theory and numerical diagnostics. It emphasizes three lessons. First, "
        "the homogeneous rational-expectations benchmark with IID dividends is "
        "a constant fundamental price. Second, realized-profit switching can "
        "endogenously move the market between near-fundamental, optimistic, and "
        "pessimistic phases. Third, the route to instability depends on the "
        "belief types. Trend chasers generate pitchfork bifurcations, "
        "contrarians generate period-doubling, and opposite biased predictors "
        "generate Hopf-style quasi-periodic fluctuations as the intensity of "
        "choice rises."
    )

    report.add_results(
        "The SMM block treats a simulated high-switching economy as pseudo-data. "
        "It matches volatility, autocorrelation of absolute returns, and excess "
        "kurtosis of price-deviation returns. The fit is approximate because "
        "the pseudo-data moments and candidate moments use separate shock banks."
    )
    report.add_figure(
        "figures/moment-fit.png",
        "SMM objective over candidate intensity values",
        plot_moment_fit(objective, fit_table),
    )

    report.add_table(
        "tables/smm-fit.csv",
        "SMM estimate of the intensity parameter and matched moments",
        fit_table,
        "The pseudo-data are generated at the true switching intensity. The SMM "
        "grid uses a separate common-random-number bank, so the selected "
        "intensity matches the target moments only approximately.",
    )

    report.add_takeaway(
        "Brock-Hommes makes a simple point with useful reach: logit choice is not "
        "only a static demand formula. Once agents use it to chase profitable "
        "forecasting rules, the rational-expectations fundamental can become a "
        "locally fragile benchmark. Low intensity leaves the market close to "
        "$p^{\\ast}$; high intensity turns recent forecast profits into endogenous "
        "bubbles, reversals, and clustered volatility."
    )

    report.add_references([
        "[Brock, W. A., and Hommes, C. H. (1998). Heterogeneous beliefs and routes to chaos in a simple asset pricing model. *Journal of Economic Dynamics and Control*, 22(8-9), 1235-1274.](https://doi.org/10.1016/S0165-1889(98)00011-6)",
        "[Hommes, C. H. (2006). Heterogeneous agent models in economics and finance. *Handbook of Computational Economics*, 2, 1109-1186.](https://doi.org/10.1016/S1574-0021(05)02023-X)",
        "[Brock, W. A., and Hommes, C. H. (1997). A rational route to randomness. *Econometrica*, 65(5), 1059-1095.](https://doi.org/10.2307/2171879)",
    ])
    report.write("README.md")


if __name__ == "__main__":
    main()
