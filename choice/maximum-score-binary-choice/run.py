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
from lib.output import ModelReport
from lib.plotting import setup_style


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
    report = ModelReport(
        "Binary Participation with Maximum Score",
        "Recover a binary participation index with a nonsmooth semiparametric estimator.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Suppose an applied microeconomist observes whether people enroll in a job-training "
        "program after seeing wage gains, commuting costs, and other covariates. A binary "
        "choice model describes the sign of latent net surplus, but the researcher may not "
        "want to assume a correctly specified logit error. Maximum score estimates the "
        "direction of that surplus index from the side of the boundary each observation "
        "falls on.\n\n"
        "The computation is awkward because the estimator maximizes the share of choices "
        "classified correctly. A small change in the slope can leave all classifications "
        "unchanged, then another change can flip several observations at once. This tutorial "
        "uses direct grid search for Manski's original score and a smoothed Horowitz-style "
        "score to compute the same median-choice target."
    )

    report.add_equations(
        r"""
The simulated decision is a participation rule:

$$
y_i = 1\{x^B_i+\beta x^C_i+\varepsilon_i \geq 0\}.
$$

Here $x^B_i$ is a benefit shifter, $x^C_i$ is a cost shifter, and
$\beta<0$ makes high costs reduce participation. Only the direction of the
index is identified, so the coefficient on $x^B_i$ is normalized to one.
Manski's maximum-score estimator solves

$$
\hat\beta
= \arg\max_b \frac{1}{n}\sum_i
\left[y_i 1\{x^B_i+b x^C_i\geq 0\} + (1-y_i)1\{x^B_i+b x^C_i<0\}\right].
$$

The smoothed version replaces the hard indicator with a normal CDF:

$$
S_h(b)=\frac{1}{n}\sum_i
\left[y_i \Phi((x^B_i+b x^C_i)/h)
+(1-y_i)\{1-\Phi((x^B_i+b x^C_i)/h)\}\right].
$$
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| Observations | {n_obs:,} | Simulated participation decisions |\n"
        f"| Normalized coefficient | 1 | Benefit shifter weight |\n"
        f"| True $\\beta$ | {beta_true:.2f} | Cost shifter weight |\n"
        f"| Error distribution | heteroskedastic logistic | Median zero, logit likelihood misspecified |\n"
        f"| Grid points | {len(grid)} | Direct search over nonsmooth objective |\n"
        f"| Smoothing bandwidth | {bandwidth:.2f} | Smooth boundary approximation |\n"
        f"| Bootstrap draws | {len(bootstrap)} | Finite-sample check for smoothed estimate |"
    )

    report.add_solution_method(
        "Because the score is a step function, derivative methods do not see the objective's "
        "jumps. The code first evaluates every candidate cost weight on a grid. It then "
        "replaces the indicator with Phi((x^B_i+b x^C_i)/h) and optimizes the smooth "
        "approximation. The bandwidth controls how sharply observations near the boundary "
        "switch classifications.\n\n"
        "```text\n"
        "Algorithm: estimate a binary participation index\n"
        "Input: choices y_i, benefit shifter x^B_i, cost shifter x^C_i, grid B, bandwidth h\n"
        "Normalize the benefit-shifter coefficient to one\n"
        "For each b in B:\n"
        "  classify i as a participant when x^B_i + b x^C_i >= 0\n"
        "  record the share of choices classified correctly\n"
        "Choose the grid value with the largest score\n"
        "For the smoothed estimate:\n"
        "  replace the hard classification rule with Phi((x^B_i+b x^C_i)/h)\n"
        "  maximize the smooth score over b\n"
        "Bootstrap observations and repeat the smoothed estimate\n"
        "Output: normalized cost weight, classification score, and bootstrap interval\n"
        "```\n\n"
        "The normalization matters for interpretation. If every coefficient were multiplied "
        "by a positive constant, the sign of latent surplus would not change. The estimate "
        "is a cost weight relative to the normalized benefit shifter, not an absolute utility "
        "scale."
    )

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
    report.add_results(
        "The raw score is flat across ranges of beta because a boundary that does not cross "
        "any observation classifies the same people. The smoothed curve peaks at "
        f"**{float(smooth_est['beta']):.3f}**, close to the true normalized cost weight "
        f"**{beta_true:.3f}**. In this sample, maximum score recovers a negative cost effect "
        "even though the logit likelihood is misspecified."
    )
    report.add_figure(
        "figures/score-objectives.png",
        "Maximum-score and smoothed-score objective functions",
        fig1,
        description="Both objectives point to a similar participation boundary; smoothing mainly makes the search surface easier to optimize.",
    )

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
    report.add_results(
        "The scatterplot shows why the estimator is a boundary problem. People with larger "
        "benefit shifters are more likely to participate, while large cost shifters push "
        "against participation when beta is negative. Noise means the regions overlap, so "
        "the best boundary cannot classify everyone correctly."
    )
    report.add_figure(
        "figures/classification-boundary.png",
        "Observed participation choices and estimated surplus boundary",
        fig2,
        description="The estimated median-surplus boundary tracks the simulated boundary despite heteroskedastic choice noise.",
    )

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
    report.add_results(
        f"The nonparametric bootstrap interval is **[{ci_low:.3f}, {ci_high:.3f}]**. "
        "It gives a finite-sample read on how stable the smoothed estimate is. It is not a "
        "replacement for the full nonstandard asymptotic theory; here it keeps uncertainty "
        "connected to the computation the tutorial just ran."
    )
    report.add_figure(
        "figures/bootstrap-estimates.png",
        "Bootstrap distribution of smoothed maximum-score estimates",
        fig3,
        description="The bootstrap distribution summarizes finite-sample uncertainty for the smoothed estimator.",
    )

    estimates = classification_table(x1, x2, y, beta_true, grid_est, smooth_est, logit)
    report.add_table(
        "tables/estimator-comparison.csv",
        "Estimator comparison",
        estimates.round(5),
        description="The logit coefficient vector is normalized by the benefit-shifter coefficient so its cost slope can be compared with maximum score.",
    )

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
    report.add_table(
        "tables/estimator-diagnostics.csv",
        "Score and bootstrap diagnostics",
        diagnostics,
        description="The score is the share of choices classified by the sign of the normalized surplus index.",
    )

    report.add_takeaway(
        "For binary participation models, maximum score separates the median-surplus "
        "boundary from a full probability model. The computation maximizes correct "
        "classifications, so the objective is nonsmooth and inference is less automatic "
        "than in logit MLE. Smoothing turns the boundary search into a continuous problem "
        "while preserving the normalized index interpretation."
    )

    report.add_references(
        [
            "[Manski, C. F. (1975). Maximum Score Estimation of the Stochastic Utility Model of Choice. *Journal of Econometrics*, 3(3), 205-228.](https://doi.org/10.1016/0304-4076(75)90032-9)",
            "[Horowitz, J. L. (1992). A Smoothed Maximum Score Estimator for the Binary Response Model. *Econometrica*, 60(3), 505-531.](https://doi.org/10.2307/2951573)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
