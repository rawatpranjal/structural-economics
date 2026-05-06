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

    report = ModelReport(
        "Lucas Tree Asset Prices and the Stochastic Discount Factor",
        "A representative-agent exchange economy where dividend risk is priced by marginal utility.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A single durable claim, the tree, pays a stochastic dividend $y_t$ each "
        "period. The single representative household holds the claim, eats the "
        "dividend, and prices it. There is no labor income, no outside asset, and "
        "no saving margin: market clearing forces $c_t=y_t$, so the entire "
        "economic content of the model is the equilibrium price function $p(y)$ "
        "that makes the household willing to hold what it already owns.\n\n"
        "What this exercise teaches is how the same Euler equation that governs "
        "consumption-saving choice elsewhere in the catalog turns into a *pricing* "
        "equation when quantities are pinned down. The household's intertemporal "
        "marginal rate of substitution\n\n"
        "$$M_{t+1}=\\beta\\,\\frac{u'(c_{t+1})}{u'(c_t)}=\\beta\\,\\left(\\frac{y_{t+1}}{y_t}\\right)^{-\\gamma}$$\n\n"
        "is the stochastic discount factor (SDF) the model produces "
        "endogenously. Once $c_t=y_t$ is imposed, $M_{t+1}$ is a deterministic "
        "function of next period's dividend, and the price of any payoff is "
        "$\\mathbb{E}_t[M_{t+1}\\,\\text{payoff}_{t+1}]$. The Lucas tree is the "
        "claim whose payoff is the dividend itself plus its resale value.\n\n"
        "Three things make the model a useful baseline. First, persistence in $y$ "
        "couples valuation today with expected payoffs tomorrow: a high dividend "
        "predicts mean reversion, which interacts with risk aversion in the SDF. "
        "Second, log utility ($\\gamma=1$) is a knife-edge benchmark with a "
        "constant price-dividend ratio $p/y=\\beta/(1-\\beta)$, so any state "
        "dependence in $p/y$ is visibly *due to* preferences departing from log. "
        "Third, the recursion is linear in a transformed object, which makes the "
        "fixed point unusually clean to solve.\n\n"
        "Compared with the [income-risk savings problem](../consumption-savings/), "
        "the household's continuation expectation looks identical but the choice "
        "set has collapsed: prices, not policies, do the work. The same Lucas "
        "tree shows up under richer dividend processes and news shocks in the "
        "[news-shock asset-pricing tutorial](../../dsge/assetNews/), and the "
        "log-endowment AR(1) here is the same primitive that drives "
        "[shock discretization](../shock-discretization/) — but with continuous "
        "Gauss-Hermite quadrature instead of a finite Markov chain, because the "
        "operator is linear in $f$ and one-dimensional integration is enough."
    )

    report.add_equations(
        r"""
**Endowment process.** Let $x_t=\log y_t$ follow

$$x_{t+1}=\rho x_t+\varepsilon_{t+1}, \qquad
\varepsilon_{t+1}\sim \mathcal{N}(0,\sigma^2),\qquad |\rho|<1.$$

The stationary log endowment has zero mean and variance $\sigma^2/(1-\rho^2)$.
Persistence $\rho$ controls how fast $y_{t+1}$ regresses to its long-run level.

**Preferences.** The representative household has CRRA utility

$$u(c)=\frac{c^{1-\gamma}}{1-\gamma}, \qquad u'(c)=c^{-\gamma},\qquad \gamma>0,$$

with the log case $u(c)=\log c$ obtained as $\gamma\to 1$.

**Equilibrium pricing equation.** Market clearing imposes $c_t=y_t$. A claim
that pays the dividend $y_{t+1}$ and a resale value $p(y_{t+1})$ next period
must satisfy the Euler condition

$$p(y_t)=\mathbb{E}_t\!\left[M_{t+1}\bigl(y_{t+1}+p(y_{t+1})\bigr)\right],
\qquad M_{t+1}=\beta\left(\frac{y_{t+1}}{y_t}\right)^{-\gamma}.$$

Writing this out and noting that $u'(y_t)=y_t^{-\gamma}$ does not depend on
$y_{t+1}$,

$$p(y_t)=\beta\,\mathbb{E}_t\!\left[
\frac{u'(y_{t+1})}{u'(y_t)}\bigl(p(y_{t+1})+y_{t+1}\bigr)\right].$$

**Linearizing transform.** Multiply both sides by $u'(y_t)$ and define the
*marginal-utility-scaled price*

$$f(y)\equiv u'(y)\,p(y).$$

The Euler equation becomes

$$f(y)=\beta\,\mathbb{E}\!\left[f(y')+u'(y')\,y'\,\big|\,y\right],$$

a linear functional fixed point: the right-hand side has no $f^2$ or $f'$
terms, no $1/f$, just a conditional expectation with a known forcing term
$u'(y')y'$. The price and price-dividend ratio recover from

$$p(y)=\frac{f(y)}{u'(y)},\qquad \frac{p(y)}{y}=\frac{f(y)}{y\,u'(y)}.$$

**Log-utility benchmark.** When $\gamma=1$, $u'(y)y=1$, so the recursion
collapses to $f=\beta(f+1)$ at every $y$, giving $f=\beta/(1-\beta)$ and the
constant ratio

$$\frac{p(y)}{y}=\frac{\beta}{1-\beta}.$$

This is the only calibration in which $p/y$ is state-independent: under log
utility the income and substitution effects on intertemporal valuation cancel
exactly. Any visible curvature in $p/y$ in the figures below is therefore a
direct fingerprint of $\gamma\neq 1$.
"""
    )

    report.add_model_setup(
        f"| Primitive | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| $\\beta$ | {beta:.2f} | Discount factor |\n"
        f"| $\\rho$ | {rho:.2f} | Persistence of log dividends |\n"
        f"| $\\sigma$ | {sigma:.2f} | Innovation standard deviation in log dividends |\n"
        f"| Stationary $\\mathrm{{sd}}(\\log y)$ | {stat_std:.4f} | $\\sigma/\\sqrt{{1-\\rho^2}}$ |\n"
        f"| $\\gamma$ | {gamma:.1f} | Baseline CRRA risk aversion |\n"
        f"| Coarse grid | {n_grid} log-endowment nodes on $[\\pm 5\\,\\mathrm{{sd}}(\\log y)]$ | Tutorial solution |\n"
        f"| Quadrature | {n_quad} Gauss-Hermite nodes for $\\varepsilon$ | Conditional expectation |\n"
        f"| Benchmark | {benchmark_grid} grid nodes, {benchmark_quad} quadrature nodes | Fine-grid ground truth |\n"
        f"| Stopping rule | $\\|f_{{n+1}}-f_n\\|_\\infty < {tol_label}$ | Fixed-point tolerance |"
    )

    report.add_solution_method(
        "**Why the linearizing transform matters.** The raw Euler equation in $p$ "
        "has the marginal-utility ratio $u'(y_{t+1})/u'(y_t)$ inside the expectation, "
        "with the denominator depending on the *current* state. That makes the "
        "operator nonlinear in $p$ even though all the agent does is hold the tree. "
        "Multiplying through by $u'(y_t)$ moves $u'(y_t)$ outside the conditional "
        "expectation and absorbs $u'(y_{t+1})\\,y_{t+1}$ into a known forcing "
        "term. What is left is a vanilla affine recursion in the scaled price "
        "$f(y)=u'(y)p(y)$.\n\n"
        "**Contraction.** The operator\n\n"
        "$$(Tf)(y)=\\beta\\,\\mathbb{E}\\!\\left[f(y')+u'(y')y'\\,\\big|\\,y\\right]$$\n\n"
        "is a $\\beta$-contraction on bounded continuous functions: $|Tf-Tg|\\leq "
        "\\beta\\,\\|f-g\\|_\\infty$ pointwise. Banach gives a unique fixed point and "
        "geometric convergence at rate $\\beta$. There is no Bellman maximization "
        "in sight because the household has no choice once $c=y$ is imposed; the "
        "iteration is purely a recursive expectation.\n\n"
        "**Discretizing the conditional expectation.** The state $x=\\log y$ is "
        "approximated on a uniform grid spanning $\\pm 5$ stationary standard "
        "deviations of $\\log y$. Tail truncation matters because high-$x$ states "
        "carry the largest weight in $u'(y')y'=y'^{1-\\gamma}$ when $\\gamma<1$ "
        "and the smallest when $\\gamma>1$. Conditional on $x_t$, the next "
        "log-endowment is normal with mean $\\rho x_t$ and variance $\\sigma^2$, so "
        "$n_q=" + str(n_quad) + "$-node Gauss-Hermite quadrature integrates exactly "
        "any polynomial of degree $\\leq 2n_q-1$ in the innovation. Linear "
        "interpolation evaluates $f$ at the off-grid points $\\rho x_t+\\varepsilon_j$.\n\n"
        "```text\n"
        "Algorithm  Lucas-tree fixed-point iteration on f = u'(y) p\n"
        "Inputs   beta, rho, sigma, gamma; log-endowment grid X = {x_i};\n"
        "           Gauss-Hermite nodes {eps_j}, weights {w_j};\n"
        "           tolerance epsilon\n"
        "Outputs  scaled price f(x_i), price p(y_i), price-dividend ratio p/y\n"
        "\n"
        "Precompute   x'_{ij} <- rho * x_i + eps_j                  # next-state nodes\n"
        "             y'_{ij} <- exp(x'_{ij})\n"
        "             d_{ij}  <- (y'_{ij})^{1 - gamma}              # forcing term u'(y') y'\n"
        "Initialise   f_0(x_i) <- 0\n"
        "for n = 0, 1, 2, ...:\n"
        "    for each x_i:\n"
        "        f_hat_{ij}  <- interp(f_n, X, x'_{ij})              # off-grid continuation\n"
        "        f_{n+1}(x_i) <- beta * sum_j w_j * (f_hat_{ij} + d_{ij})\n"
        "    err <- max_i | f_{n+1}(x_i) - f_n(x_i) |\n"
        "stop when err < epsilon\n"
        "p(y_i)     <- f(x_i) * (y_i)^{gamma}\n"
        "p(y_i)/y_i <- p(y_i) / y_i\n"
        "```\n\n"
        "**Hyperparameters and trade-offs.** The convergence rate is fixed at "
        f"$\\beta={beta}$ and is independent of $n_{{q}}$ or grid size, because "
        "the contraction modulus comes from the time discount, not the "
        "discretization. Refining the grid mainly reduces interpolation bias near "
        "the tails; refining $n_q$ reduces quadrature bias for higher moments of "
        "$y'^{1-\\gamma}$. Both errors are controlled by comparing the tutorial "
        "solution to a fine-grid benchmark with "
        f"{benchmark_grid} state nodes and {benchmark_quad} quadrature nodes.\n\n"
        f"At $\\gamma={gamma}$ the tutorial solution converges in "
        f"**{solution.iterations} iterations** to sup-norm residual "
        f"**{solution.error:.2e}**. Across the central $\\pm 3\\,\\mathrm{{sd}}(\\log y)$ "
        f"region the coarse-grid price agrees with the benchmark to within "
        f"**{max_relative_error_pct:.3f}%**, which is interpolation noise rather "
        "than economic disagreement."
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
    report.add_figure(
        "figures/asset-price-function.png",
        "Lucas tree price function compared with a fine-grid benchmark",
        fig1,
        description=(
            "The equilibrium price is strictly increasing and convex in the "
            "dividend state, with a slope above one in the right tail: a higher "
            "dividend today predicts persistently higher dividends tomorrow, and "
            "with $\\gamma=2$ the future high-dividend states are weighted *less* "
            "by the SDF, so the price still rises but mostly through the level of "
            "future cash flows rather than through discounting. The "
            "lower panel shows that the "
            f"{n_grid}-node solution stays within "
            f"{max_relative_error_pct:.3f}% of the {benchmark_grid}-node "
            "benchmark across the central $\\pm 3\\,\\mathrm{sd}$ region. The "
            "benchmark is numerical, but the agreement confirms that the visible "
            "curvature is economic, not a coarse-grid artefact."
        ),
    )

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
    report.add_figure(
        "figures/simulation-paths.png",
        "Simulated dividend, tree price, and price-dividend ratio",
        fig2,
        description=(
            "Along a 120-period simulation, prices co-move closely with "
            "dividends because persistence in $y$ raises expected future "
            "cash flows roughly proportionally. The price index is more volatile "
            "than the dividend index when the state is far from its mean: prices "
            "capitalize the entire continuation stream, not just the next "
            "payment. The lower panel isolates the more informative ratio "
            "$p(y_t)/y_t$. Under $\\gamma=2$ it varies systematically with the "
            "state and sits above the log-utility benchmark $\\beta/(1-\\beta)$ "
            "when dividends are high — exactly the comparative-static the "
            "next figure reads off the cross-section of states."
        ),
    )

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
    report.add_figure(
        "figures/comparative-statics-gamma.png",
        "Price-dividend ratios under alternative CRRA risk aversion values",
        fig3,
        description=(
            "Risk aversion changes the *slope* of the price-dividend ratio in "
            "the state, not just its level. With log utility ($\\gamma=1$), the "
            "ratio collapses to $\\beta/(1-\\beta)\\approx "
            f"{pd_log_utility:.1f}$ at every $y$ — the dashed grey line and the "
            "$\\gamma=1$ curve overlap exactly, which is the cleanest possible "
            "audit of the solver. When $\\gamma<1$, the ratio falls in $y$: "
            "high current dividends predict mean reversion downward, and an "
            "agent with low risk aversion values the resulting smoothing in "
            "future $u'(y')y'$ less. When $\\gamma>1$, the ratio rises in $y$ "
            "and steeply so for $\\gamma=5$, because the SDF heavily up-weights "
            "the low-dividend tail of $y_{t+1}\\mid y_t$ that is *more likely* "
            "after a high $y_t$. Dotted same-color lines are the fine-grid "
            "benchmarks; they sit on top of the coarse-grid solutions across "
            "the plotted range."
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
        "Price-dividend ratios at selected dividend states",
        df,
        description=(
            "Reading across rows isolates two effects. At the median state "
            "$y\\approx 1$ all four economies price the tree at roughly the "
            f"log-utility benchmark $\\beta/(1-\\beta)={pd_log_utility:.1f}$, "
            "because $\\mathbb{E}_t[(y_{t+1}/y_t)^{1-\\gamma}]\\approx 1$ in a "
            "neighbourhood of the long-run mean. Away from the mean the SDF "
            "interacts with mean reversion: the $\\gamma=5$ ratio doubles from "
            "the median to the tails, while the $\\gamma=0.5$ ratio compresses. "
            "The $\\gamma=1$ column is the closed-form benchmark $19$ at every "
            "$y$, exact to numerical tolerance."
        ),
    )

    report.add_takeaway(
        "Once market clearing forces $c=y$, the household has nothing to choose "
        "and the entire model lives inside the equilibrium price. The Euler "
        "equation becomes a *valuation* equation, not a household problem. "
        "Multiplying through by $u'(y)$ turns it into a linear, $\\beta$-"
        "contractive recursion in the scaled price, which iterates without any "
        "policy-function machinery. Risk aversion then has a sharp diagnostic "
        "role: log utility gives a flat $p/y=\\beta/(1-\\beta)$, and any "
        "state-dependence in the price-dividend ratio is preferences "
        "interacting with mean reversion in dividends — *the* fingerprint that "
        "asset-pricing puzzles like the equity premium read off the data."
    )

    report.add_references(
        [
            'Lucas, R. (1978). "Asset Prices in an Exchange Economy." *Econometrica*, 46(6), 1429-1445.',
            "Mehra, R. and Prescott, E. (1985). \"The Equity Premium: A Puzzle.\" *Journal of Monetary Economics*, 15(2), 145-161.",
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
