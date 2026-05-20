#!/usr/bin/env python3
"""Lucas tree asset pricing in a representative-agent exchange economy."""

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


@dataclass(frozen=True)
class LucasSolution:
    """Numerical solution of the Lucas tree pricing equation."""

    x_grid: np.ndarray
    y_grid: np.ndarray
    f: np.ndarray
    price: np.ndarray
    pd_ratio: np.ndarray
    gamma: float
    iterations: int
    error: float
    converged: bool


def crra_marginal_utility(c: np.ndarray, gamma: float) -> np.ndarray:
    """Return marginal utility under CRRA preferences."""
    return c ** (-gamma)


def normal_quadrature_nodes(sigma: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights for an N(0, sigma^2) innovation."""
    nodes, weights = hermgauss(n_nodes)
    shocks = np.sqrt(2.0) * sigma * nodes
    probabilities = weights / np.sqrt(np.pi)
    return shocks, probabilities


def solve_price_function(
    *,
    beta: float,
    rho: float,
    gamma: float,
    sigma: float,
    x_grid: np.ndarray,
    n_quad: int,
    tol: float,
    max_iter: int,
) -> LucasSolution:
    """Solve the transformed Lucas pricing equation on a log-endowment grid."""
    y_grid = np.exp(x_grid)
    shocks, weights = normal_quadrature_nodes(sigma, n_quad)
    x_next = rho * x_grid[:, None] + shocks[None, :]
    y_next = np.exp(x_next)
    dividend_term = crra_marginal_utility(y_next, gamma) * y_next

    f = np.zeros_like(x_grid)
    error = np.inf
    iteration = 0

    for iteration in range(1, max_iter + 1):
        continuation = np.interp(
            x_next.ravel(),
            x_grid,
            f,
            left=f[0],
            right=f[-1],
        ).reshape(x_next.shape)
        f_new = beta * np.sum((continuation + dividend_term) * weights[None, :], axis=1)
        error = float(np.max(np.abs(f_new - f)))
        f = f_new
        if error < tol:
            break

    price = f / crra_marginal_utility(y_grid, gamma)
    return LucasSolution(
        x_grid=x_grid,
        y_grid=y_grid,
        f=f,
        price=price,
        pd_ratio=price / y_grid,
        gamma=gamma,
        iterations=iteration,
        error=error,
        converged=error < tol,
    )


def simulate_endowment_path(
    *,
    rho: float,
    sigma: float,
    periods: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate log endowment and endowment levels."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, sigma, size=periods)
    x_path = np.zeros(periods)

    for t in range(periods - 1):
        x_path[t + 1] = rho * x_path[t] + shocks[t + 1]

    return x_path, np.exp(x_path)


def main() -> None:
    beta = 0.95
    rho = 0.90
    gamma = 2.0
    sigma = 0.10
    n_grid = 120
    n_quad = 21
    benchmark_grid = 900
    benchmark_quad = 45
    tol = 1e-9
    benchmark_tol = 2e-11
    max_iter = 1_200
    periods = 120
    tol_label = "10^{-9}"

    stat_std = sigma / np.sqrt(1.0 - rho**2)
    x_grid = np.linspace(-5.0 * stat_std, 5.0 * stat_std, n_grid)
    x_grid_fine = np.linspace(-5.5 * stat_std, 5.5 * stat_std, benchmark_grid)

    solution = solve_price_function(
        beta=beta,
        rho=rho,
        gamma=gamma,
        sigma=sigma,
        x_grid=x_grid,
        n_quad=n_quad,
        tol=tol,
        max_iter=max_iter,
    )
    benchmark = solve_price_function(
        beta=beta,
        rho=rho,
        gamma=gamma,
        sigma=sigma,
        x_grid=x_grid_fine,
        n_quad=benchmark_quad,
        tol=benchmark_tol,
        max_iter=max_iter,
    )

    print(
        "  baseline gamma={:.1f}: {} iterations, error={:.2e}".format(
            gamma, solution.iterations, solution.error
        )
    )
    print(
        "  benchmark gamma={:.1f}: {} iterations, error={:.2e}".format(
            gamma, benchmark.iterations, benchmark.error
        )
    )

    gamma_values = [0.5, 1.0, 2.0, 5.0]
    solutions_by_gamma: dict[float, LucasSolution] = {}
    benchmarks_by_gamma: dict[float, LucasSolution] = {}
    for gamma_value in gamma_values:
        solutions_by_gamma[gamma_value] = solve_price_function(
            beta=beta,
            rho=rho,
            gamma=gamma_value,
            sigma=sigma,
            x_grid=x_grid,
            n_quad=n_quad,
            tol=tol,
            max_iter=max_iter,
        )
        benchmarks_by_gamma[gamma_value] = solve_price_function(
            beta=beta,
            rho=rho,
            gamma=gamma_value,
            sigma=sigma,
            x_grid=x_grid_fine,
            n_quad=benchmark_quad,
            tol=benchmark_tol,
            max_iter=max_iter,
        )
        gamma_solution = solutions_by_gamma[gamma_value]
        print(
            "  gamma={:.1f}: {} iterations, error={:.2e}".format(
                gamma_value, gamma_solution.iterations, gamma_solution.error
            )
        )

    central = np.abs(x_grid) <= 3.0 * stat_std
    y_central = solution.y_grid[central]
    price_central = solution.price[central]
    benchmark_price_central = np.interp(
        x_grid[central], benchmark.x_grid, benchmark.price
    )
    relative_error = (price_central - benchmark_price_central) / benchmark_price_central
    max_relative_error_pct = 100.0 * float(np.max(np.abs(relative_error)))

    pd_log_utility = beta / (1.0 - beta)

    x_path, y_path = simulate_endowment_path(
        rho=rho,
        sigma=sigma,
        periods=periods,
        seed=123,
    )
    price_path = np.interp(x_path, solution.x_grid, solution.price)
    pd_path = price_path / y_path

    setup_style()

    fig1, (ax1, ax1_err) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    ax1.plot(
        y_central,
        price_central,
        color="#1f77b4",
        linewidth=2.2,
        label=f"{n_grid}-node solution",
    )
    ax1.plot(
        y_central,
        benchmark_price_central,
        color="#111111",
        linewidth=1.8,
        linestyle="--",
        label=f"{benchmark_grid}-node benchmark",
    )
    ax1.set_ylabel("Asset price $p(y)$")
    ax1.set_title("Equilibrium price of the Lucas tree")
    ax1.legend(loc="upper left")

    ax1_err.axhline(0.0, color="#777777", linewidth=1.0)
    ax1_err.plot(y_central, 100.0 * relative_error, color="#d62728", linewidth=1.8)
    ax1_err.set_xlabel("Dividend / endowment $y$")
    ax1_err.set_ylabel("Error (%)")
    ax1_err.set_title("Coarse-grid error vs. benchmark")
    fig1.tight_layout()
    save_figure(fig1, "figures/asset-price-function.png", dpi=150)

    fig2, (ax2_top, ax2_bot) = plt.subplots(
        2,
        1,
        figsize=(8.0, 5.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0]},
    )
    periods_axis = np.arange(periods)
    ax2_top.plot(
        periods_axis,
        y_path / y_path[0],
        color="#1f77b4",
        linewidth=1.8,
        label="Dividend $y_t$",
    )
    ax2_top.plot(
        periods_axis,
        price_path / price_path[0],
        color="#b22222",
        linewidth=1.8,
        label="Tree price $p(y_t)$",
    )
    ax2_top.axhline(1.0, color="#777777", linewidth=0.9, linestyle=":")
    ax2_top.set_ylabel("Index, $t=0$ value $=1$")
    ax2_top.set_title("A simulated dividend-price path")
    ax2_top.legend(loc="upper left")

    ax2_bot.plot(
        periods_axis,
        pd_path,
        color="#2ca02c",
        linewidth=1.8,
        label=f"$p(y_t)/y_t$, $\\gamma={gamma:g}$",
    )
    ax2_bot.axhline(
        pd_log_utility,
        color="#111111",
        linewidth=1.2,
        linestyle="--",
        label=f"Log-utility benchmark $\\beta/(1-\\beta)={pd_log_utility:.1f}$",
    )
    ax2_bot.set_xlabel("Period $t$")
    ax2_bot.set_ylabel("$p(y_t)/y_t$")
    ax2_bot.legend(loc="upper left")
    fig2.tight_layout()
    save_figure(fig2, "figures/simulation-paths.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(7.2, 4.8))
    colors = ["#2ca02c", "#111111", "#1f77b4", "#9467bd"]
    for gamma_value, color in zip(gamma_values, colors):
        gamma_solution = solutions_by_gamma[gamma_value]
        gamma_benchmark = benchmarks_by_gamma[gamma_value]
        ax3.plot(
            gamma_solution.y_grid[central],
            gamma_solution.pd_ratio[central],
            linewidth=2.0,
            color=color,
            label=f"$\\gamma={gamma_value:g}$",
        )
        bench_y = np.exp(x_grid[central])
        bench_pd = np.interp(
            x_grid[central], gamma_benchmark.x_grid, gamma_benchmark.pd_ratio
        )
        ax3.plot(
            bench_y,
            bench_pd,
            linewidth=1.0,
            linestyle=":",
            color=color,
        )
    ax3.axhline(
        pd_log_utility,
        color="#777777",
        linewidth=0.9,
        linestyle="--",
    )
    ax3.set_xlabel("Dividend / endowment $y$")
    ax3.set_ylabel("Price-dividend ratio $p(y)/y$")
    ax3.set_title("Risk aversion and state-contingent valuation")
    ax3.legend(loc="upper left")
    fig3.tight_layout()
    save_figure(fig3, "figures/comparative-statics-gamma.png", dpi=150)

    # Thumbnail
    save_thumbnail("figures/asset-price-function.png", "figures/thumb.png")

    # =========================================================================
    # Tables
    # =========================================================================
    Path("tables").mkdir(parents=True, exist_ok=True)

    sample_positions = np.linspace(0, len(y_central) - 1, 7, dtype=int)
    sample_indices = np.flatnonzero(central)[sample_positions]
    table_data = {
        "y": [f"{solution.y_grid[i]:.3f}" for i in sample_indices],
        "p(y), gamma=2": [f"{solution.price[i]:.3f}" for i in sample_indices],
    }
    for gamma_value in gamma_values:
        gamma_solution = solutions_by_gamma[gamma_value]
        table_data[f"p/y, gamma={gamma_value:g}"] = [
            f"{gamma_solution.pd_ratio[i]:.3f}" for i in sample_indices
        ]

    df = pd.DataFrame(table_data)
    df.to_csv("tables/price-dividend-ratio.csv", index=False)

    convergence_df = pd.DataFrame(
        {
            "Quantity": [
                "Baseline iterations",
                "Baseline sup-norm residual",
                "Central max relative error (%)",
            ],
            "Value": [
                f"{solution.iterations}",
                f"{solution.error:.2e}",
                f"{max_relative_error_pct:.3f}",
            ],
        }
    )
    convergence_df.to_csv("tables/convergence.csv", index=False)

    print(
        f"\nGenerated: figures + tables"
    )


if __name__ == "__main__":
    main()
