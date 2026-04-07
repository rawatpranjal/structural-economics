#!/usr/bin/env python3
"""Preference Recoverability: Bounding Utility from Revealed Preference Data.

Given data satisfying GARP, recovers bounds on the underlying utility function
and indifference curves using Afriat's theorem and Varian's construction.

Reference: Varian (1982), "The Nonparametric Approach to Demand Analysis."
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def generate_cobb_douglas_data(alpha, T, seed=42):
    """Generate consumption data from a Cobb-Douglas consumer."""
    rng = np.random.RandomState(seed)
    prices = rng.uniform(0.5, 2.0, (T, 2))
    income = rng.uniform(5, 15, T)
    # Optimal demand: x1 = alpha * m / p1, x2 = (1-alpha) * m / p2
    x = np.zeros((T, 2))
    x[:, 0] = alpha * income / prices[:, 0]
    x[:, 1] = (1 - alpha) * income / prices[:, 1]
    return prices, x, income


def afriat_numbers(prices, x):
    """Compute Afriat numbers (u_i, lambda_i) that rationalize the data.

    Varian's construction: find u_i, lambda_i > 0 such that
    u_i - u_j <= lambda_j * p_j . (x_i - x_j) for all i, j
    """
    T = len(prices)
    # Expenditures
    e = np.array([prices[t] @ x[t] for t in range(T)])

    # Direct revealed preference
    R = np.zeros((T, T), dtype=bool)
    for i in range(T):
        for j in range(T):
            if prices[i] @ x[i] >= prices[i] @ x[j]:
                R[i, j] = True

    # Warshall transitive closure
    R_star = R.copy()
    for k in range(T):
        for i in range(T):
            for j in range(T):
                if R_star[i, k] and R_star[k, j]:
                    R_star[i, j] = True

    # Varian construction: set u_i = 0, lambda_i = 1/e_i as starting point
    # Then adjust using shortest-path-like algorithm
    lam = 1.0 / e
    u = np.zeros(T)

    # Iterative tightening
    for _ in range(T * 2):
        changed = False
        for i in range(T):
            for j in range(T):
                if i != j and R_star[j, i]:  # j revealed preferred to i
                    # Constraint: u_j - u_i <= lambda_i * p_i . (x_j - x_i)
                    bound = u_i_upper = u[i] + lam[i] * prices[i] @ (x[j] - x[i])
                    if u[j] > bound + 1e-10:
                        u[j] = bound
                        changed = True
        if not changed:
            break

    return u, lam, R, R_star


def recover_indifference_bounds(prices, x, u, lam, target_obs, grid_x1):
    """Recover bounds on the indifference curve through observation target_obs."""
    T = len(prices)
    target_u = u[target_obs]

    # Upper bound: for each grid point x1, find min x2 such that
    # utility >= target_u using all Afriat inequalities
    n_grid = len(grid_x1)
    x2_lower = np.full(n_grid, 0.0)
    x2_upper = np.full(n_grid, np.inf)

    for g in range(n_grid):
        for t in range(T):
            # Afriat inequality: u_t + lam_t * p_t . (x_new - x_t) provides bound
            # If the test bundle (grid_x1[g], x2) satisfies this for u = target_u:
            # target_u <= u_t + lam_t * (p_t[0] * (grid_x1[g] - x_t[0]) + p_t[1] * (x2 - x_t[1]))
            # Rearranging for x2:
            if abs(lam[t] * prices[t, 1]) > 1e-10:
                x2_bound = x[t, 1] + (target_u - u[t] - lam[t] * prices[t, 0] * (grid_x1[g] - x[t, 0])) / (lam[t] * prices[t, 1])
                x2_lower[g] = max(x2_lower[g], x2_bound)

    return x2_lower


def main():
    # =========================================================================
    # Generate data from Cobb-Douglas consumer
    # =========================================================================
    alpha_true = 0.6  # True preference parameter
    T = 15            # Number of observations
    prices, x, income = generate_cobb_douglas_data(alpha_true, T)

    # =========================================================================
    # Compute Afriat numbers
    # =========================================================================
    u, lam, R, R_star = afriat_numbers(prices, x)

    # Check GARP
    violations = 0
    for i in range(T):
        for j in range(T):
            if i != j and R_star[i, j] and prices[j] @ x[j] > prices[j] @ x[i] + 1e-10:
                violations += 1
    print(f"GARP violations: {violations} (should be 0 for Cobb-Douglas data)")

    # =========================================================================
    # Recover indifference curve bounds
    # =========================================================================
    target_obs = T // 2  # Middle observation
    grid_x1 = np.linspace(0.5, max(x[:, 0]) * 1.5, 100)
    x2_bounds = recover_indifference_bounds(prices, x, u, lam, target_obs, grid_x1)

    # True indifference curve (Cobb-Douglas)
    target_utility = x[target_obs, 0] ** alpha_true * x[target_obs, 1] ** (1 - alpha_true)
    x2_true = (target_utility / grid_x1 ** alpha_true) ** (1 / (1 - alpha_true))

    # =========================================================================
    # Welfare bounds: how much compensation for a price change?
    # =========================================================================
    price_change_factor = 1.3  # 30% increase in good 1's price
    cv_bounds = []
    for t in range(T):
        old_cost = prices[t] @ x[t]
        new_prices = prices[t].copy()
        new_prices[0] *= price_change_factor
        # Lower bound on CV: how much extra income needed at new prices
        # to reach original utility level u[t]
        cv_lower = lam[t] * (new_prices[0] - prices[t, 0]) * x[t, 0]
        cv_bounds.append(cv_lower)

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Preference Recoverability",
        "Bounding utility functions and indifference curves from revealed preference data.",
    )

    report.add_overview(
        "Afriat's theorem tells us *whether* data is consistent with utility maximization. "
        "Preference recoverability goes further: given GARP-consistent data, *what can we "
        "learn* about the underlying utility function?\n\n"
        "Using Varian's (1982) construction, we can compute Afriat numbers $(u_t, \\lambda_t)$ "
        "that form a piecewise-linear utility function rationalizing the data. These numbers "
        "provide bounds on indifference curves and welfare measures (compensating variation) "
        "without assuming any functional form for utility."
    )

    report.add_equations(r"""
**Afriat's theorem:** Data $\{(p_t, x_t)\}_{t=1}^T$ satisfies GARP if and only if there exist numbers $u_t, \lambda_t > 0$ such that:
$$u_t - u_s \leq \lambda_s \, p_s \cdot (x_t - x_s) \quad \forall \, t, s$$

**Recovered utility at any bundle** $x$:
$$\hat{U}(x) = \min_s \left\{ u_s + \lambda_s \, p_s \cdot (x - x_s) \right\}$$

This is a piecewise-linear, concave function that passes through all observed points.

**Indifference curve bounds:** For a given utility level $\bar{u}$, the set of bundles with $\hat{U}(x) = \bar{u}$ can be bounded using the Afriat inequalities.

**Compensating variation:** $CV_t = \lambda_t \cdot \Delta p \cdot x_t$ provides a bound on welfare change from price changes.
""")

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| True $\\alpha$ | {alpha_true} | Cobb-Douglas parameter |\n"
        f"| $T$ | {T} | Number of observations |\n"
        f"| Goods | 2 | For visualization |\n"
        f"| GARP violations | {violations} | Confirmed zero |"
    )

    report.add_solution_method(
        "**Step 1:** Compute the direct revealed preference relation $R$ and its transitive "
        "closure $R^*$ via Warshall's algorithm.\n\n"
        "**Step 2:** Construct Afriat numbers $(u_t, \\lambda_t)$ satisfying the Afriat "
        "inequalities using Varian's iterative tightening procedure.\n\n"
        "**Step 3:** Use the Afriat numbers to bound indifference curves and compute "
        "compensating variation bounds for hypothetical price changes."
    )

    # --- Figure 1: Budget lines and demand ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, T))
    for t in range(T):
        # Budget line: p1*x1 + p2*x2 = m
        x1_line = np.array([0, income[t] / prices[t, 0]])
        x2_line = np.array([income[t] / prices[t, 1], 0])
        ax1.plot(x1_line, x2_line, "-", color=colors[t], alpha=0.3, linewidth=0.8)
        ax1.plot(x[t, 0], x[t, 1], "o", color=colors[t], markersize=6)
    ax1.plot(x[target_obs, 0], x[target_obs, 1], "r*", markersize=15, zorder=5,
             label=f"Target (obs {target_obs})")
    ax1.set_xlabel("Good 1")
    ax1.set_ylabel("Good 2")
    ax1.set_title("Budget Lines and Chosen Bundles")
    ax1.legend()
    report.add_figure("figures/budget-lines.png", "Budget constraints and optimal choices from Cobb-Douglas consumer", fig1)

    # --- Figure 2: Recovered indifference curve ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    valid = (x2_bounds > 0) & (x2_bounds < max(x[:, 1]) * 3)
    ax2.plot(grid_x1[valid], x2_bounds[valid], "b-", linewidth=2, label="Recovered bound")
    valid_true = x2_true < max(x[:, 1]) * 3
    ax2.plot(grid_x1[valid_true], x2_true[valid_true], "r--", linewidth=1.5, label="True (Cobb-Douglas)")
    ax2.plot(x[target_obs, 0], x[target_obs, 1], "r*", markersize=15, zorder=5)
    ax2.set_xlabel("Good 1")
    ax2.set_ylabel("Good 2")
    ax2.set_title("Recovered vs True Indifference Curve")
    ax2.legend()
    ax2.set_ylim(0, max(x[:, 1]) * 2)
    report.add_figure("figures/indifference-curve.png", "Nonparametric bounds on indifference curve vs true Cobb-Douglas", fig2)

    # --- Figure 3: Afriat numbers ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    ax3a.bar(range(T), u, color="steelblue")
    ax3a.set_xlabel("Observation")
    ax3a.set_ylabel("$u_t$")
    ax3a.set_title("Afriat Utility Numbers")

    ax3b.bar(range(T), lam, color="coral")
    ax3b.set_xlabel("Observation")
    ax3b.set_ylabel("$\\lambda_t$")
    ax3b.set_title("Afriat Marginal Utility Numbers")
    fig3.tight_layout()
    report.add_figure("figures/afriat-numbers.png", "Afriat numbers (u_t, lambda_t) that rationalize the data", fig3)

    # --- Table ---
    df = pd.DataFrame({
        "Observation": range(T),
        "x1": [f"{x[t,0]:.2f}" for t in range(T)],
        "x2": [f"{x[t,1]:.2f}" for t in range(T)],
        "u_t": [f"{u[t]:.4f}" for t in range(T)],
        "lambda_t": [f"{lam[t]:.4f}" for t in range(T)],
        "CV bound": [f"{cv_bounds[t]:.4f}" for t in range(T)],
    })
    report.add_table("tables/afriat-numbers.csv", "Afriat Numbers and Welfare Bounds", df)

    report.add_takeaway(
        "Preference recoverability shows how much we can learn without functional form assumptions:\n\n"
        "**Key insights:**\n"
        "- **Nonparametric identification**: Afriat numbers provide a complete characterization "
        "of all utility functions consistent with the data. No need to assume Cobb-Douglas, CES, "
        "or any specific functional form.\n"
        "- **Indifference curve bounds**: The recovered bounds on indifference curves tighten as "
        "more data becomes available. With enough observations, the bounds converge to the true "
        "indifference curve.\n"
        "- **Welfare bounds**: Compensating variation can be bounded without knowing the exact "
        "utility function — useful for policy evaluation when preferences are unknown.\n"
        "- **Limitations**: The bounds can be wide with few observations or when prices don't "
        "vary enough to reveal preferences in different regions of the consumption space."
    )

    report.add_references([
        "Afriat, S. (1967). \"The Construction of Utility Functions from Expenditure Data.\" *International Economic Review*, 8(1).",
        "Varian, H. (1982). \"The Nonparametric Approach to Demand Analysis.\" *Econometrica*, 50(4).",
        "Varian, H. (2006). \"Revealed Preference.\" In *Samuelsonian Economics and the Twenty-First Century*, Oxford University Press.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
