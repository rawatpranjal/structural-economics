#!/usr/bin/env python3
"""Conditional logit for fixed effects in short binary panels."""
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def simulate_panel(
    n_agents: int,
    n_periods: int,
    beta_true: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate binary panel choices with agent effects correlated with covariates."""
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.0, 1.0, n_agents)
    x_mean = 0.65 * alpha + rng.normal(0.0, 0.35, n_agents)
    x = x_mean[:, None] + rng.normal(0.0, 0.80, size=(n_agents, n_periods))
    probability = expit(alpha[:, None] + beta_true * x)
    y = rng.binomial(1, probability)
    return {"alpha": alpha, "x": x, "y": y, "probability": probability}


def pooled_logit(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Estimate a pooled logit with one intercept and one slope."""
    X = np.column_stack([np.ones(x.size), x.ravel()])
    target = y.ravel()

    def objective(params: np.ndarray) -> float:
        eta = X @ params
        return float(np.sum(np.logaddexp(0.0, eta) - target * eta))

    result = minimize(objective, np.zeros(2), method="BFGS")
    return {
        "params": np.asarray(result.x, dtype=float),
        "log_likelihood": -float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
    }


def conditional_denominator(x_i: np.ndarray, successes: int, beta: float) -> float:
    """Log denominator of the conditional likelihood for one panel."""
    terms = []
    for combo in itertools.combinations(range(len(x_i)), successes):
        terms.append(beta * float(np.sum(x_i[list(combo)])))
    return float(logsumexp(terms))


def conditional_log_likelihood(beta: float, x: np.ndarray, y: np.ndarray) -> float:
    """Chamberlain conditional log likelihood for switchers."""
    totals = y.sum(axis=1)
    switcher = (totals > 0) & (totals < y.shape[1])
    ll = 0.0
    for x_i, y_i, total in zip(x[switcher], y[switcher], totals[switcher]):
        ll += beta * float(np.dot(y_i, x_i)) - conditional_denominator(x_i, int(total), beta)
    return ll


def conditional_logit(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Estimate the slope after conditioning out agent fixed effects."""
    objective = lambda beta: -conditional_log_likelihood(float(beta), x, y)
    result = minimize_scalar(objective, bounds=(-1.0, 3.0), method="bounded", options={"xatol": 1e-10})
    totals = y.sum(axis=1)
    switcher = (totals > 0) & (totals < y.shape[1])
    return {
        "beta": float(result.x),
        "log_likelihood": -float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nfev),
        "switchers": int(np.sum(switcher)),
        "non_switchers": int(np.sum(~switcher)),
    }


def likelihood_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate the conditional likelihood over a grid of slope values."""
    return np.array([conditional_log_likelihood(float(beta), x, y) for beta in grid])


def bin_by_agent_effect(alpha: np.ndarray, x: np.ndarray, y: np.ndarray, n_bins: int = 8) -> pd.DataFrame:
    """Summarize the correlation that biases pooled logit."""
    quantiles = np.quantile(alpha, np.linspace(0.0, 1.0, n_bins + 1))
    rows = []
    for j in range(n_bins):
        if j == n_bins - 1:
            mask = (alpha >= quantiles[j]) & (alpha <= quantiles[j + 1])
        else:
            mask = (alpha >= quantiles[j]) & (alpha < quantiles[j + 1])
        rows.append(
            {
                "Alpha bin": j + 1,
                "Mean alpha": float(alpha[mask].mean()),
                "Mean covariate": float(x[mask].mean()),
                "Choice rate": float(y[mask].mean()),
                "Agents": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    n_agents = 1_200
    n_periods = 5
    beta_true = 1.20
    panel = simulate_panel(n_agents, n_periods, beta_true, seed=10)
    alpha = panel["alpha"]
    x = panel["x"]
    y = panel["y"]

    pooled = pooled_logit(x, y)
    conditional = conditional_logit(x, y)
    grid = np.linspace(0.25, 2.10, 220)
    ll_grid = likelihood_grid(x, y, grid)
    alpha_summary = bin_by_agent_effect(alpha, x, y)

    pooled_beta = float(np.asarray(pooled["params"])[1])
    conditional_beta = float(conditional["beta"])
    switcher_share = float(conditional["switchers"] / n_agents)

    print("Conditional logit panel tutorial")
    print(f"  True beta: {beta_true:.3f}")
    print(f"  Pooled beta: {pooled_beta:.3f}")
    print(f"  Conditional beta: {conditional_beta:.3f}")
    print(f"  Switcher share: {switcher_share:.3f}")

    setup_style()
    report = ModelReport(
        "Conditional Logit for Fixed Effects Panels",
        "Condition out agent heterogeneity in a short binary-choice panel.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Short panels often contain persistent unobserved heterogeneity. In this example, "
        "some agents have a high baseline taste for the inside option, and those same agents "
        "also tend to face higher values of the observed covariate. A pooled logit confuses "
        "that permanent heterogeneity with the causal slope on the covariate.\n\n"
        "The computational method is the conditional likelihood for fixed-effects logit. "
        "Instead of estimating one intercept per agent, the likelihood conditions on each "
        "agent's total number of successes. That sufficient statistic removes the fixed "
        "effect, so the slope is identified from within-agent changes in the covariate."
    )

    report.add_equations(
        r"""
Agent $i$ is observed for $T$ periods. The binary choice model is

$$
\Pr(y_{it}=1\mid x_{it},\alpha_i)
= \frac{\exp(\alpha_i+\beta x_{it})}{1+\exp(\alpha_i+\beta x_{it})}.
$$

The fixed effect $\alpha_i$ is unrestricted and may be correlated with the
observed covariate. Conditional on $s_i=\sum_t y_{it}$, the probability of the
observed choice sequence no longer contains $\alpha_i$:

$$
\Pr(y_i\mid s_i,x_i;\beta)
=
\frac{\exp\{\beta\sum_t y_{it}x_{it}\}}
{\sum_{A:|A|=s_i}\exp\{\beta\sum_{t\in A}x_{it}\}}.
$$

The conditional log likelihood is the sum of these terms over agents with
$0<s_i<T$. Agents who always choose zero or always choose one identify their
own intercepts but contribute no within-agent slope information.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| Agents | {n_agents} | Short panel units with unobserved heterogeneity |\n"
        f"| Periods | {n_periods} | Repeated binary choices per agent |\n"
        f"| True slope $\\beta$ | {beta_true:.2f} | Effect of the observed covariate |\n"
        f"| Fixed effect | $\\alpha_i\\sim N(0,1)$ | Permanent agent taste for choosing one |\n"
        f"| Covariate correlation | positive | Agent means in $x$ move with $\\alpha_i$ |\n"
        f"| Switchers | {conditional['switchers']} | Agents used by the conditional likelihood |\n"
        f"| Non-switchers | {conditional['non_switchers']} | Agents conditioned out of slope estimation |"
    )

    report.add_solution_method(
        "The denominator is small here because each agent has five observations. The code "
        "enumerates all subsets with the observed number of successes and evaluates the "
        "conditional likelihood directly.\n\n"
        "```text\n"
        "Algorithm: fixed-effects conditional logit\n"
        "Input: panel choices y_it and covariates x_it\n"
        "For each agent i:\n"
        "  Compute s_i = sum_t y_it\n"
        "  If s_i is 0 or T, drop agent from the conditional likelihood\n"
        "  Enumerate all subsets A of periods with cardinality s_i\n"
        "  Add beta sum_t y_it x_it minus log sum_A exp(beta sum_{t in A} x_it)\n"
        "Choose beta that maximizes the summed conditional likelihood\n"
        "Compare against pooled logit and known truth\n"
        "```\n\n"
        "For larger $T$, the denominator can be computed by dynamic programming instead of "
        "literal enumeration. The economic idea is unchanged: use within-agent variation "
        "and avoid estimating thousands of nuisance intercepts."
    )

    fig1, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(grid, ll_grid - ll_grid.max())
    ax1.axvline(beta_true, color="black", linestyle="--", linewidth=1.2, label="True")
    ax1.axvline(pooled_beta, color="tab:red", linestyle=":", linewidth=2, label="Pooled")
    ax1.axvline(conditional_beta, color="tab:blue", linestyle="-.", linewidth=2, label="Conditional")
    ax1.set_xlabel("Slope beta")
    ax1.set_ylabel("Conditional log likelihood relative to maximum")
    ax1.set_title("Conditional Likelihood")
    ax1.legend()
    report.add_results(
        "The conditional likelihood peaks near the true slope. The pooled logit slope is "
        f"**{pooled_beta:.3f}**, while the conditional estimate is **{conditional_beta:.3f}** "
        f"against the true value **{beta_true:.3f}**. The gap is the omitted fixed-effect "
        "problem made visible."
    )
    report.add_figure(
        "figures/conditional-likelihood.png",
        "Conditional likelihood over the slope parameter",
        fig1,
        description="The likelihood is normalized by subtracting its maximum so the curvature is visible.",
    )

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
    ax2.scatter(alpha, x.mean(axis=1), alpha=0.28, s=14)
    ax2.set_xlabel("Agent fixed effect alpha")
    ax2.set_ylabel("Mean observed covariate")
    ax2.set_title("Why Pooled Logit Is Biased")
    report.add_results(
        "The simulation deliberately correlates the agent effect with the mean covariate. "
        "That is the setting where treating every observation as independent logit data "
        "is not innocuous."
    )
    report.add_figure(
        "figures/alpha-covariate-correlation.png",
        "Correlation between fixed effects and observed covariates",
        fig2,
        description="Agents with higher permanent taste also tend to face higher covariate values.",
    )

    fig3, ax3 = plt.subplots(figsize=(7.2, 4.6))
    ax3.plot(alpha_summary["Mean alpha"], alpha_summary["Choice rate"], marker="o", label="Choice rate")
    ax3.plot(alpha_summary["Mean alpha"], alpha_summary["Mean covariate"], marker="s", label="Mean covariate")
    ax3.set_xlabel("Mean alpha in bin")
    ax3.set_title("Between-Agent Variation")
    ax3.legend()
    report.add_results(
        "Between-agent comparisons are contaminated by permanent tastes. Conditional logit "
        "throws away that between-agent intercept information and uses only switchers for "
        "the slope."
    )
    report.add_figure(
        "figures/binned-agent-effects.png",
        "Choice rates and covariates by fixed-effect bin",
        fig3,
        description="The binned pattern shows why between-agent variation cannot be read as the structural slope.",
    )

    estimates = pd.DataFrame(
        {
            "Estimator": ["Pooled logit", "Conditional fixed-effects logit"],
            "Slope estimate": [pooled_beta, conditional_beta],
            "Slope error": [pooled_beta - beta_true, conditional_beta - beta_true],
            "Log likelihood": [float(pooled["log_likelihood"]), float(conditional["log_likelihood"])],
            "Success": [bool(pooled["success"]), bool(conditional["success"])],
            "Iterations or evaluations": [int(pooled["iterations"]), int(conditional["iterations"])],
        }
    )
    report.add_table(
        "tables/estimator-comparison.csv",
        "Pooled versus conditional logit estimates",
        estimates.round(5),
        description="The pooled and conditional likelihoods condition on different information, so their levels are not directly comparable.",
    )

    report.add_table(
        "tables/agent-effect-bins.csv",
        "Agent-effect bin diagnostics",
        alpha_summary.round(5),
        description="The bin table summarizes the source of pooled-logit bias in the simulated panel.",
    )

    diagnostics = pd.DataFrame(
        {
            "Diagnostic": [
                "Choice rate",
                "Switcher share",
                "Correlation alpha and mean x",
                "Pooled beta error",
                "Conditional beta error",
            ],
            "Value": [
                float(y.mean()),
                switcher_share,
                float(np.corrcoef(alpha, x.mean(axis=1))[0, 1]),
                pooled_beta - beta_true,
                conditional_beta - beta_true,
            ],
        }
    )
    report.add_table(
        "tables/panel-diagnostics.csv",
        "Panel and identification diagnostics",
        diagnostics,
        description="The conditional estimator's identifying sample is the switcher group.",
    )

    report.add_takeaway(
        "Conditional logit is a computational way to remove fixed effects rather than "
        "estimate them. Conditioning on each agent's total number of successes eliminates "
        "the nuisance intercept and makes the slope come from within-agent comparisons. "
        "The cost is also clear: agents with no choice variation do not identify the slope."
    )

    report.add_references(
        [
            "[Chamberlain, G. (1980). Analysis of Covariance with Qualitative Data. *Review of Economic Studies*, 47(1), 225-238.](https://doi.org/10.2307/2297110)",
            "[Andersen, E. B. (1970). Asymptotic Properties of Conditional Maximum-Likelihood Estimators. *Journal of the Royal Statistical Society: Series B*, 32(2), 283-301.](https://doi.org/10.1111/j.2517-6161.1970.tb00840.x)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
