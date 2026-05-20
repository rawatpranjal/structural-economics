#!/usr/bin/env python3
"""Maximum score estimation for a semiparametric participation index."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def simulate_binary_choice(n_obs: int, beta_true: float, seed: int) -> dict[str, np.ndarray]:
    """Simulate a binary participation model with median-zero heteroskedastic errors."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n_obs)
    x2 = 0.40 * x1 + rng.normal(size=n_obs)
    scale = 0.55 + 0.65 * np.abs(x2)
    error = scale * rng.logistic(size=n_obs)
    latent = x1 + beta_true * x2 + error
    y = (latent >= 0.0).astype(int)
    return {"x1": x1, "x2": x2, "scale": scale, "error": error, "latent": latent, "y": y}


def max_score(beta: float, x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> float:
    """Fraction correctly classified by the normalized linear index."""
    predicted = (x1 + beta * x2 >= 0.0).astype(int)
    return float(np.mean(predicted == y))


def smoothed_score(beta: float, x1: np.ndarray, x2: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Horowitz-style smoothed classification score."""
    probability = norm.cdf((x1 + beta * x2) / bandwidth)
    return float(np.mean(y * probability + (1.0 - y) * (1.0 - probability)))


def estimate_grid_score(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Maximize the nonsmooth score over a fixed grid."""
    scores = np.array([max_score(float(beta), x1, x2, y) for beta in grid])
    idx = int(np.argmax(scores))
    return {"beta": float(grid[idx]), "score": float(scores[idx]), "scores": scores}


def estimate_smoothed_score(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
) -> dict[str, object]:
    """Estimate beta by maximizing the smoothed score."""
    objective = lambda beta: -smoothed_score(float(beta), x1, x2, y, bandwidth)
    result = minimize_scalar(objective, bounds=(-2.0, 0.50), method="bounded", options={"xatol": 1e-8})
    return {
        "beta": float(result.x),
        "score": -float(result.fun),
        "success": bool(result.success),
        "evaluations": int(result.nfev),
    }


def estimate_logit_ratio(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Misspecified logit, reported as a normalized slope ratio."""
    X = np.column_stack([x1, x2])

    def objective(params: np.ndarray) -> float:
        eta = X @ params
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(objective, np.array([1.0, -0.50]), method="BFGS")
    params = np.asarray(result.x, dtype=float)
    return {
        "params": params,
        "beta_ratio": float(params[1] / params[0]),
        "log_likelihood": -float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
    }


def bootstrap_smoothed_score(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Nonparametric bootstrap distribution for the smoothed score estimate."""
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    estimates = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=n_obs)
        estimate = estimate_smoothed_score(x1[idx], x2[idx], y[idx], bandwidth)
        estimates[b] = float(estimate["beta"])
    return estimates


def classification_table(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    beta_true: float,
    grid_est: dict[str, np.ndarray | float],
    smooth_est: dict[str, object],
    logit: dict[str, object],
) -> pd.DataFrame:
    """Summarize estimates and classification rates."""
    rows = []
    for name, beta in [
        ("True participation index", beta_true),
        ("Grid maximum score", float(grid_est["beta"])),
        ("Smoothed maximum score", float(smooth_est["beta"])),
        ("Misspecified logit ratio", float(logit["beta_ratio"])),
    ]:
        rows.append(
            {
                "Estimator": name,
                "Normalized beta": beta,
                "Error": beta - beta_true,
                "Classification score": max_score(beta, x1, x2, y),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    n_obs = 2_500
    beta_true = -0.85
    bandwidth = 0.25
    data = simulate_binary_choice(n_obs, beta_true, seed=8)
    x1 = data["x1"]
    x2 = data["x2"]
    y = data["y"]
    grid = np.linspace(-2.0, 0.50, 501)

    grid_est = estimate_grid_score(x1, x2, y, grid)
    smooth_est = estimate_smoothed_score(x1, x2, y, bandwidth)
    logit = estimate_logit_ratio(x1, x2, y)
    bootstrap = bootstrap_smoothed_score(x1, x2, y, bandwidth, n_bootstrap=80, seed=19)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    print("Binary participation maximum-score tutorial")
    print(f"  True normalized beta: {beta_true:.3f}")
    print(f"  Grid maximum score beta: {float(grid_est['beta']):.3f}")
    print(f"  Smoothed maximum score beta: {float(smooth_est['beta']):.3f}")
    print(f"  Logit normalized ratio: {float(logit['beta_ratio']):.3f}")

    setup_style()

    raw_scores = np.asarray(grid_est["scores"])
    smooth_scores = np.array([smoothed_score(float(beta), x1, x2, y, bandwidth) for beta in grid])
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(grid, raw_scores, label="Maximum score")
    ax1.plot(grid, smooth_scores, label="Smoothed score")
    ax1.axvline(beta_true, color="black", linestyle="--", linewidth=1.2, label="True")
    ax1.axvline(float(smooth_est["beta"]), color="tab:red", linestyle=":", linewidth=2, label="Smoothed estimate")
    ax1.set_xlabel("Normalized beta")
    ax1.set_ylabel("Classification score")
    ax1.set_title("Nonsmooth and Smoothed Participation Scores")
    ax1.legend()
    save_figure(fig1, "figures/score-objectives.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6.8, 6.2))
    sample = np.linspace(-3.0, 3.0, 200)
    ax2.scatter(x1[y == 0], x2[y == 0], s=10, alpha=0.25, label="Choice 0")
    ax2.scatter(x1[y == 1], x2[y == 1], s=10, alpha=0.25, label="Choice 1")
    ax2.plot(sample, -sample / beta_true, color="black", linewidth=2, label="True boundary")
    ax2.plot(sample, -sample / float(smooth_est["beta"]), color="tab:red", linestyle="--", linewidth=2, label="Estimated boundary")
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_xlabel("Benefit shifter $x^B$")
    ax2.set_ylabel("Cost shifter $x^C$")
    ax2.set_title("Participation Boundary")
    ax2.legend(loc="upper right")
    save_figure(fig2, "figures/classification-boundary.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(7.2, 4.6))
    ax3.hist(bootstrap, bins=18, color="tab:green", alpha=0.85)
    ax3.axvline(beta_true, color="black", linestyle="--", linewidth=1.2, label="True")
    ax3.axvline(float(smooth_est["beta"]), color="tab:red", linewidth=2, label="Estimate")
    ax3.axvline(ci_low, color="tab:gray", linestyle=":", linewidth=1.5)
    ax3.axvline(ci_high, color="tab:gray", linestyle=":", linewidth=1.5, label="95% bootstrap interval")
    ax3.set_xlabel("Bootstrap smoothed-score estimate")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Bootstrap Uncertainty")
    ax3.legend()
    save_figure(fig3, "figures/bootstrap-estimates.png", dpi=150)

    estimates = classification_table(x1, x2, y, beta_true, grid_est, smooth_est, logit)
    Path("tables").mkdir(parents=True, exist_ok=True)
    estimates.round(5).to_csv("tables/estimator-comparison.csv", index=False)

    diagnostics = pd.DataFrame(
        {
            "Diagnostic": [
                "Choice-one share",
                "Grid maximum score",
                "Smoothed score",
                "Smoothed optimizer success",
                "Smoothed optimizer evaluations",
                "Bootstrap mean",
                "Bootstrap standard deviation",
                "Bootstrap lower 95",
                "Bootstrap upper 95",
            ],
            "Value": [
                float(y.mean()),
                float(grid_est["score"]),
                float(smooth_est["score"]),
                float(bool(smooth_est["success"])),
                float(smooth_est["evaluations"]),
                float(np.mean(bootstrap)),
                float(np.std(bootstrap, ddof=1)),
                float(ci_low),
                float(ci_high),
            ],
        }
    )
    Path("tables").mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv("tables/estimator-diagnostics.csv", index=False)

    save_thumbnail("figures/score-objectives.png", "figures/thumb.png")
    print(f"\nDone: figures/ + tables/")


if __name__ == "__main__":
    main()
