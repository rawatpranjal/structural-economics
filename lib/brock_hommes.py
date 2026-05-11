"""Brock-Hommes two-rule asset-pricing simulator.

Shared helper used by:
- agent-based-models/brock-hommes-asset-pricing/ (SMM grid-search)
- structural-econometrics/neural-posterior-brock-hommes/ (NPE posterior)

The model has fundamentalist and trend-following predictors. Rule shares
follow logit choice in a smoothed realized-profit score. The state is the
price deviation x_t = p_t - p^*, where p^* = d / (R - 1) is the
constant-dividend rational-expectations fundamental.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def simulate(
    beta: float,
    params: Params,
    seed: int,
    shocks: np.ndarray | None = None,
) -> Run:
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
    """Return the three moments used by the SMM baseline."""
    returns = np.diff(x)[burn:]
    abs_returns = np.abs(returns)
    z = (returns - returns.mean()) / returns.std()
    return {
        "volatility": float(np.std(returns)),
        "abs return autocorrelation": float(np.corrcoef(abs_returns[1:], abs_returns[:-1])[0, 1]),
        "excess kurtosis": float(np.mean(z**4) - 3.0),
    }


def summary_statistics(x: np.ndarray, burn: int) -> np.ndarray:
    """Five summary statistics for NPE.

    Returns a 1-D float array with:
        s0  std of returns
        s1  lag-1 abs-return autocorrelation
        s2  excess kurtosis of returns
        s3  lag-1 return autocorrelation
        s4  lag-5 abs-return autocorrelation
    """
    returns = np.diff(x)[burn:]
    abs_returns = np.abs(returns)
    z = (returns - returns.mean()) / (returns.std() + 1e-12)
    return np.array([
        float(np.std(returns)),
        float(np.corrcoef(abs_returns[1:], abs_returns[:-1])[0, 1]),
        float(np.mean(z**4) - 3.0),
        float(np.corrcoef(returns[1:], returns[:-1])[0, 1]),
        float(np.corrcoef(abs_returns[5:], abs_returns[:-5])[0, 1]),
    ], dtype=np.float64)


def average_moments(beta: float, params: Params, shock_bank: np.ndarray) -> dict[str, float]:
    """Average the three SMM moments over a common-random-number bank."""
    rows = [
        moments(simulate(beta, params, seed=20_000 + i, shocks=shocks).x, params.burn)
        for i, shocks in enumerate(shock_bank)
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
