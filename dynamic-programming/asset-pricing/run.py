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
from lib.output import ModelReport
from lib.plotting import setup_style


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
    """Gauss-Hermite nodes and weights for a N(0, sigma^2) innovation."""
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

    x_path, y_path = simulate_endowment_path(
        rho=rho,
        sigma=sigma,
        periods=periods,
        seed=123,
    )
    price_path = np.interp(x_path, solution.x_grid, solution.price)

    setup_style()

    report = ModelReport(
        "Lucas Tree Asset Prices and the Stochastic Discount Factor",
        "A representative-agent exchange economy where dividend risk is priced by marginal utility.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "This tutorial studies the Lucas tree, the cleanest setting in which asset "
        "prices are equilibrium objects rather than exogenous returns. A representative "
        "household owns a claim to the single tree. The tree pays the stochastic "
        "endowment $y_t$ each period, and because the household is representative, "
        "aggregate consumption must equal the dividend: $c_t=y_t$.\n\n"
        "That market-clearing restriction is what makes the example useful. The "
        "quantity allocation is pinned down, so all of the economics appears in the "
        "state-contingent price of the claim. The price must make the household willing "
        "to hold the tree after valuing tomorrow's payoff with the stochastic discount "
        "factor $\\beta u'(y_{t+1})/u'(y_t)$. Compared with the "
        "[income-risk savings problem](../consumption-savings/), the state is still "
        "a persistent endowment process, but there is no saving policy to choose; prices "
        "absorb the intertemporal risk."
    )

    report.add_equations(
        r"""
Let $x_t=\log y_t$ follow
$$x_{t+1}=\rho x_t+\varepsilon_{t+1}, \qquad
\varepsilon_{t+1}\sim \mathcal{N}(0,\sigma^2),$$
and let the representative household have CRRA utility
$$u(c)=\frac{c^{1-\gamma}}{1-\gamma}, \qquad u'(c)=c^{-\gamma}.$$

In equilibrium $c_t=y_t$. A claim to the tree pays dividend $y_{t+1}$ and resale
value $p(y_{t+1})$ next period, so the Euler equation is
$$p(y_t)=\beta\,\mathbb{E}\left[
\frac{u'(y_{t+1})}{u'(y_t)}
\left(p(y_{t+1})+y_{t+1}\right)
\mid y_t
\right].$$

Define the marginal-utility-scaled price
$$f(y)=u'(y)p(y).$$
Multiplying the Euler equation by $u'(y_t)$ gives the linear fixed point
$$f(y)=\beta\,\mathbb{E}\left[f(y')+u'(y')y'\mid y\right].$$
After solving for $f$, recover the asset price from $p(y)=f(y)/u'(y)$ and the
price-dividend ratio from $p(y)/y$.
"""
    )

    report.add_model_setup(
        f"| Primitive | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| $\\beta$ | {beta:.2f} | Discount factor |\n"
        f"| $\\rho$ | {rho:.2f} | Persistence of log dividends |\n"
        f"| $\\sigma$ | {sigma:.2f} | Innovation standard deviation in log dividends |\n"
        f"| $\\gamma$ | {gamma:.1f} | Baseline CRRA risk aversion |\n"
        f"| Coarse grid | {n_grid} log-endowment nodes | Tutorial solution |\n"
        f"| Quadrature | {n_quad} Gauss-Hermite nodes | Conditional expectation |\n"
        f"| Benchmark | {benchmark_grid} grid nodes, {benchmark_quad} quadrature nodes | Fine-grid comparison |\n"
        f"| Stopping rule | $\\|f_{{n+1}}-f_n\\|_\\infty < {tol_label}$ | Fixed-point tolerance |"
    )

    report.add_solution_method(
        "The computation iterates directly on the scaled price $f$. This is not a "
        "household-choice Bellman equation: there is no policy function because "
        "market clearing sets $c=y$. The fixed point is still a contraction with "
        "modulus $\\beta$, and the interpolation step only approximates the "
        "conditional expectation over next-period dividends.\n\n"
        "```text\n"
        "Inputs: beta, rho, sigma, gamma, log-endowment grid X, quadrature nodes eps_j, weights w_j\n"
        "Initialize f_0(x_i) = 0 on X\n"
        "For n = 0, 1, 2, ...:\n"
        "    For each current state x_i:\n"
        "        For each quadrature shock eps_j:\n"
        "            x_ij' = rho x_i + eps_j\n"
        "            y_ij' = exp(x_ij')\n"
        "            interpolate f_n(x_ij') from the grid X\n"
        "        Set f_{n+1}(x_i) = beta sum_j w_j [f_n(x_ij') + (y_ij')^(1-gamma)]\n"
        "    Stop when max_i |f_{n+1}(x_i)-f_n(x_i)| is below tolerance\n"
        "Output: p(exp(x_i)) = f(x_i) exp(gamma x_i)\n"
        "```\n\n"
        f"The baseline solution converged in **{solution.iterations} iterations** "
        f"with final sup-norm error **{solution.error:.2e}**. A finer grid and more "
        "quadrature nodes provide a numerical benchmark for the plotted state range."
    )

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
        label="fine-grid benchmark",
    )
    ax1.set_ylabel("Asset price $p(y)$")
    ax1.set_title("Price of the Lucas tree")
    ax1.legend(loc="upper left")

    ax1_err.axhline(0.0, color="#777777", linewidth=1.0)
    ax1_err.plot(y_central, 100.0 * relative_error, color="#d62728", linewidth=1.8)
    ax1_err.set_xlabel("Dividend/endowment $y$")
    ax1_err.set_ylabel("Error (%)")
    ax1_err.set_title("Coarse-grid error relative to benchmark")
    fig1.tight_layout()
    report.add_figure(
        "figures/asset-price-function.png",
        "Lucas tree price function compared with a fine-grid benchmark",
        fig1,
        description=(
            f"The equilibrium price is increasing over the central dividend states. "
            f"The dashed line is a fine-grid quadrature benchmark; the lower panel "
            f"shows that the coarse tutorial solution stays within "
            f"{max_relative_error_pct:.3f}% of that benchmark on this range. "
            "The benchmark is numerical rather than analytic, but it is a useful "
            "check that the visible curvature is economic rather than a coarse-grid artifact."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(8.0, 4.4))
    periods_axis = np.arange(periods)
    ax2.plot(
        periods_axis,
        y_path / y_path[0],
        color="#1f77b4",
        linewidth=1.8,
        label="Dividend",
    )
    ax2.plot(
        periods_axis,
        price_path / price_path[0],
        color="#b22222",
        linewidth=1.8,
        label="Tree price",
    )
    ax2.axhline(1.0, color="#777777", linewidth=0.9, linestyle=":")
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Index, initial period = 1")
    ax2.set_title("A simulated dividend-price path")
    ax2.legend(loc="upper left")
    fig2.tight_layout()
    report.add_figure(
        "figures/simulation-paths.png",
        "Simulated dividend and price path normalized to the initial period",
        fig2,
        description=(
            "Along a simulated path, prices move with dividends because persistent "
            "high dividends raise expected future payoffs. The price series is more "
            "forward looking than the dividend itself: it capitalizes not only today's "
            "payment but also the continuation value implied by the Markov process."
        ),
    )

    fig3, ax3 = plt.subplots(figsize=(7.2, 4.8))
    colors = ["#2ca02c", "#111111", "#1f77b4", "#9467bd"]
    for gamma_value, color in zip(gamma_values, colors):
        gamma_solution = solutions_by_gamma[gamma_value]
        ax3.plot(
            gamma_solution.y_grid[central],
            gamma_solution.pd_ratio[central],
            linewidth=2.0,
            color=color,
            label=f"$\\gamma={gamma_value:g}$",
        )
    ax3.set_xlabel("Dividend/endowment $y$")
    ax3.set_ylabel("Price-dividend ratio $p(y)/y$")
    ax3.set_title("Risk aversion and state-contingent valuation")
    ax3.legend(loc="upper left")
    fig3.tight_layout()
    report.add_figure(
        "figures/comparative-statics-gamma.png",
        "Price-dividend ratios under alternative CRRA risk aversion values",
        fig3,
        description=(
            "Risk aversion changes the slope of the price-dividend ratio, not just "
            "its level. With log utility ($\\gamma=1$), $p(y)/y=\\beta/(1-\\beta)$ "
            "is constant. When $\\gamma>1$, high current dividends predict mean "
            "reversion toward lower future consumption, so future payoffs receive "
            "larger marginal-utility weights and the ratio rises with the current state."
        ),
    )

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
    report.add_table(
        "tables/price-dividend-ratio.csv",
        "Selected Price-Dividend Ratios",
        df,
        description=(
            "The selected states make the log-utility benchmark visible: the "
            "$\\gamma=1$ price-dividend ratio is 19 at every $y$. Departures from "
            "log utility tilt this ratio across the state space because the stochastic "
            "discount factor interacts with mean reversion in dividends."
        ),
    )

    report.add_takeaway(
        "The Lucas tree turns a simple market-clearing allocation into a state-price "
        "problem. Once $c_t=y_t$, the asset price is the expected discounted payoff "
        "using marginal utility as the deflator. The useful computational trick is "
        "to solve for $f(y)=u'(y)p(y)$, which makes the Euler equation linear and "
        "contractive. Economically, the exercise shows why the price-dividend ratio "
        "is a valuation object: it records how persistence, discounting, and risk "
        "aversion transform a dividend process into an equilibrium claim price."
    )

    report.add_references(
        [
            'Lucas, R. (1978). "Asset Prices in an Exchange Economy." *Econometrica*, 46(6), 1429-1445.',
            "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 13.",
            "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press.",
        ]
    )

    report.write("README.md")
    print(
        f"\nGenerated: README.md + {len(report._figures)} figures + "
        f"{len(report._tables)} tables"
    )


if __name__ == "__main__":
    main()
