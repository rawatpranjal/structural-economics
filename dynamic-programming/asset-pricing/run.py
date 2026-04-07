#!/usr/bin/env python3
"""Lucas Asset Pricing: Equilibrium Prices in a Pure Exchange Economy.

Solves the Lucas (1978) tree model using value function iteration on the
pricing kernel. In equilibrium, consumption equals endowment (no trade),
so asset prices are determined by the representative agent's Euler equation.

Reference: Lucas, R. (1978). "Asset Prices in an Exchange Economy." Econometrica.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.grids import uniform_grid
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    beta = 0.95       # Discount factor
    rho = 0.9         # Persistence of log-endowment AR(1)
    gamma = 2.0       # CRRA risk aversion
    sigma = 0.1       # Std dev of log-endowment shock
    n_grid = 100      # Grid points for endowment y
    n_shocks = 100    # Monte Carlo draws for expectations
    tol = 1e-6        # Convergence tolerance
    max_iter = 500    # Maximum VFI iterations

    # =========================================================================
    # Grid for endowment y
    # =========================================================================
    # Endowment follows log(y') = rho * log(y) + z, z ~ N(0, sigma)
    # Stationary std of log(y) is sigma / sqrt(1 - rho^2)
    stat_std = sigma / np.sqrt(1.0 - rho ** 2)
    y_min = np.exp(-3.0 * stat_std)
    y_max = np.exp(3.0 * stat_std)
    y_grid = np.linspace(y_min, y_max, n_grid)

    # =========================================================================
    # Utility: u(c) = c^(1-gamma) / (1-gamma), u'(c) = c^(-gamma)
    # =========================================================================
    def u_prime(c):
        return c ** (-gamma)

    # =========================================================================
    # Shock draws (fixed seed for reproducibility)
    # =========================================================================
    rng = np.random.default_rng(42)
    z_shocks = rng.normal(0.0, sigma, size=n_shocks)

    # =========================================================================
    # Solve via VFI on the pricing kernel f(y) = u'(y) * p(y)
    #
    # Euler equation in equilibrium (c = y):
    #   f(y) = beta * E[ f(y') + u'(y') * y' ]
    #
    # where y' = exp(rho * log(y) + z), z ~ N(0, sigma)
    # =========================================================================
    f = np.zeros(n_grid)  # Initial guess: f(y) = 0

    for iteration in range(1, max_iter + 1):
        f_new = np.zeros(n_grid)

        for iy in range(n_grid):
            y = y_grid[iy]
            log_y = np.log(y)

            # Next-period endowment for each shock draw
            log_y_prime = rho * log_y + z_shocks
            y_prime = np.exp(log_y_prime)

            # Interpolate f at y' values
            f_y_prime = np.interp(y_prime, y_grid, f)

            # u'(y') * y' for each draw
            uy_prime = u_prime(y_prime) * y_prime

            # Bellman: f(y) = beta * E[f(y') + u'(y') * y']
            f_new[iy] = beta * np.mean(f_y_prime + uy_prime)

        error = np.max(np.abs(f_new - f))
        if iteration % 25 == 0:
            print(f"  VFI iteration {iteration:3d}, error = {error:.2e}")
        f = f_new

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    info = {"iterations": iteration, "converged": error < tol, "error": error}

    # =========================================================================
    # Recover asset price: p(y) = f(y) / u'(y)
    # =========================================================================
    price = f / u_prime(y_grid)

    # Price-dividend ratio: p(y) / y
    pd_ratio = price / y_grid

    # =========================================================================
    # Comparative statics: vary gamma
    # =========================================================================
    gamma_values = [0.5, 1.0, 2.0, 5.0]
    price_by_gamma = {}

    for g in gamma_values:
        up_g = lambda c, gam=g: c ** (-gam)
        f_g = np.zeros(n_grid)

        for it in range(1, max_iter + 1):
            f_g_new = np.zeros(n_grid)
            for iy in range(n_grid):
                y = y_grid[iy]
                log_y_prime = rho * np.log(y) + z_shocks
                y_prime = np.exp(log_y_prime)
                f_interp = np.interp(y_prime, y_grid, f_g)
                uy_prime = up_g(y_prime) * y_prime
                f_g_new[iy] = beta * np.mean(f_interp + uy_prime)
            err_g = np.max(np.abs(f_g_new - f_g))
            f_g = f_g_new
            if err_g < tol:
                break

        price_by_gamma[g] = f_g / up_g(y_grid)
        print(f"  gamma={g:.1f}: converged in {it} iterations")

    # =========================================================================
    # Simulate endowment and price paths
    # =========================================================================
    T_sim = 100
    rng_sim = np.random.default_rng(123)
    z_sim = rng_sim.normal(0.0, sigma, size=T_sim)

    y_path = np.zeros(T_sim)
    y_path[0] = 1.0  # Start at mean endowment

    for t in range(T_sim - 1):
        log_y_next = rho * np.log(y_path[t]) + z_sim[t + 1]
        y_path[t + 1] = np.exp(log_y_next)

    # Interpolate price along the simulated path
    price_path = np.interp(y_path, y_grid, price)

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Lucas Asset Pricing",
        "Equilibrium asset prices in a pure exchange economy with a representative agent.",
    )

    report.add_overview(
        "The Lucas (1978) tree model is the foundational framework for understanding "
        "how asset prices are determined in general equilibrium. A representative agent "
        "owns a tree that produces a stochastic endowment (dividend) each period. Since "
        "the agent cannot trade in equilibrium, consumption must equal the endowment, and "
        "the asset price adjusts to make the agent willing to hold the tree.\n\n"
        "This model shows that asset prices are determined by the marginal rate of "
        "substitution between current and future consumption, weighted by future dividends. "
        "Risk aversion plays a central role: more risk-averse agents demand higher compensation "
        "for holding risky assets, lowering equilibrium prices."
    )

    report.add_equations(
        r"""
$$V(y) = u(y) + \beta \, \mathbb{E}\left[ V(y') \mid y \right]$$

**Euler equation (asset pricing):**
$$p(y) = \beta \, \mathbb{E}\left[ \frac{u'(y')}{u'(y)} \left( p(y') + y' \right) \mid y \right]$$

where $y$ is the endowment (= dividend = consumption in equilibrium).

**Endowment process (AR(1) in logs):**
$$\log y' = \rho \, \log y + z, \qquad z \sim \mathcal{N}(0, \sigma^2)$$

**Pricing kernel transformation:** Define $f(y) = u'(y) \cdot p(y)$, then:
$$f(y) = \beta \, \mathbb{E}\left[ f(y') + u'(y') \cdot y' \right]$$

This is a contraction mapping in $f$, solved by iteration. The asset price is recovered as $p(y) = f(y) / u'(y)$.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $\\rho$   | {rho} | Persistence of log-endowment |\n"
        f"| $\\gamma$ | {gamma} | CRRA risk aversion |\n"
        f"| $\\sigma$ | {sigma} | Std dev of endowment shock |\n"
        f"| Grid points | {n_grid} | Endowment grid |\n"
        f"| MC draws | {n_shocks} | For expectation approximation |"
    )

    report.add_solution_method(
        "**Value Function Iteration on Pricing Kernel:** We iterate on the "
        "transformed Euler equation $f(y) = \\beta \\, \\mathbb{E}[f(y') + u'(y') y']$ "
        f"using Monte Carlo integration with {n_shocks} fixed draws for the expectation. "
        "The operator is a contraction mapping under standard conditions "
        "(bounded, discounted), guaranteeing convergence.\n\n"
        "Starting from $f_0 = 0$, we iterate until "
        f"$\\|f_{{n+1}} - f_n\\|_\\infty < 10^{{-6}}$.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e})."
    )

    # --- Figure 1: Asset Price Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(y_grid, price, "b-", linewidth=2)
    ax1.set_xlabel("Endowment $y$")
    ax1.set_ylabel("Asset price $p(y)$")
    ax1.set_title("Equilibrium Asset Price Function")
    report.add_figure(
        "figures/asset-price-function.png",
        "Equilibrium asset price as a function of endowment",
        fig1,
    )

    # --- Figure 2: Simulated Price and Income Paths (dual y-axis) ---
    fig2, ax2a = plt.subplots(figsize=(10, 5))
    periods = np.arange(T_sim)

    color_y = "steelblue"
    color_p = "firebrick"

    ax2a.plot(periods, y_path, color=color_y, linewidth=1.5, label="Endowment $y_t$")
    ax2a.set_xlabel("Period")
    ax2a.set_ylabel("Endowment $y_t$", color=color_y)
    ax2a.tick_params(axis="y", labelcolor=color_y)

    ax2b = ax2a.twinx()
    ax2b.plot(periods, price_path, color=color_p, linewidth=1.5, label="Price $p_t$")
    ax2b.set_ylabel("Asset price $p_t$", color=color_p)
    ax2b.tick_params(axis="y", labelcolor=color_p)

    ax2a.set_title("Simulated Endowment and Asset Price Paths")

    # Combined legend
    lines_a, labels_a = ax2a.get_legend_handles_labels()
    lines_b, labels_b = ax2b.get_legend_handles_labels()
    ax2a.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")

    fig2.tight_layout()
    report.add_figure(
        "figures/simulation-paths.png",
        "Simulated endowment and asset price over 100 periods (dual y-axis)",
        fig2,
    )

    # --- Figure 3: Comparative Statics (gamma) ---
    fig3, ax3 = plt.subplots()
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]
    for (g, p_g), color in zip(price_by_gamma.items(), colors):
        ax3.plot(y_grid, p_g, linewidth=2, color=color, label=f"$\\gamma = {g}$")
    ax3.set_xlabel("Endowment $y$")
    ax3.set_ylabel("Asset price $p(y)$")
    ax3.set_title("Asset Price: Comparative Statics over Risk Aversion $\\gamma$")
    ax3.legend()
    report.add_figure(
        "figures/comparative-statics-gamma.png",
        "Asset price function for different levels of risk aversion gamma",
        fig3,
    )

    # --- Table: Price-Dividend Ratio at Selected Endowment Levels ---
    sample_idx = np.linspace(5, n_grid - 1, 8, dtype=int)
    table_data = {
        "y": [f"{y_grid[i]:.3f}" for i in sample_idx],
        "p(y)": [f"{price[i]:.4f}" for i in sample_idx],
        "p(y)/y": [f"{pd_ratio[i]:.4f}" for i in sample_idx],
    }
    # Add price for each gamma
    for g in gamma_values:
        p_g = price_by_gamma[g]
        pd_g = p_g / y_grid
        table_data[f"p/y (gamma={g})"] = [f"{pd_g[i]:.4f}" for i in sample_idx]

    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/price-dividend-ratio.csv",
        "Price-Dividend Ratio at Selected Endowment Levels",
        df,
    )

    report.add_takeaway(
        "The Lucas tree model reveals how equilibrium asset prices emerge from the "
        "interaction of time preference, risk aversion, and the stochastic properties "
        "of dividends.\n\n"
        "**Key insights:**\n"
        "- Asset prices are *increasing* in the endowment level: when income is high, "
        "the agent is wealthy and willing to pay more for the asset (wealth effect).\n"
        "- The price-dividend ratio *varies* with the level of risk aversion $\\gamma$. "
        "Higher risk aversion lowers prices because the agent demands greater compensation "
        "for bearing dividend risk (precautionary effect).\n"
        "- With low risk aversion ($\\gamma < 1$), the substitution effect dominates: "
        "prices are high because the agent is relatively indifferent between consumption "
        "today and tomorrow.\n"
        "- Asset prices comove positively with endowment in the simulation, reflecting "
        "the procyclical nature of asset valuations in this model.\n"
        "- The pricing kernel transformation $f(y) = u'(y) p(y)$ converts the Euler equation "
        "into a standard contraction mapping, making VFI straightforward and guaranteed to converge."
    )

    report.add_references([
        "Lucas, R. (1978). \"Asset Prices in an Exchange Economy.\" *Econometrica*, 46(6), 1429-1445.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 13.",
        "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
