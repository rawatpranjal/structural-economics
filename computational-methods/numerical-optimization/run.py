#!/usr/bin/env python3
"""Optimizer diagnostics for a latent-regime likelihood.

A latent-regime likelihood can have two basins that fit the data.
The tutorial compares local starts, a restart grid, and global search.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing, minimize
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


MU1 = np.array([1.5, 1.5])
MU2 = np.array([-1.5, -1.5])
SIGMA = np.array([[1.0, 0.5], [0.5, 1.0]])
MIXING_PROB = 0.5
INV_SIGMA = np.linalg.inv(SIGMA)
LOG_DET_SIGMA = float(np.linalg.slogdet(SIGMA)[1])
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class OptimizationRun:
    """Container for optimizer output and its recorded path."""

    method: str
    start: str
    solution: np.ndarray
    objective: float
    iterations: int
    success: bool
    path: np.ndarray


def log_mvn(points: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Evaluate a bivariate normal log density at one or many points."""
    x = np.asarray(points, dtype=float)
    diff = x - mean
    quad = np.einsum("...i,ij,...j->...", diff, INV_SIGMA, diff)
    return -0.5 * (2.0 * LOG_2PI + LOG_DET_SIGMA + quad)


def log_target(points: np.ndarray) -> np.ndarray:
    """Evaluate the log density of the two-component Gaussian mixture."""
    log_components = np.stack(
        [
            np.log(MIXING_PROB) + log_mvn(points, MU1),
            np.log(1.0 - MIXING_PROB) + log_mvn(points, MU2),
        ],
        axis=0,
    )
    return logsumexp(log_components, axis=0)


def objective(x: np.ndarray) -> float:
    """Negative log density. Minimizing this finds a mode of the mixture."""
    return float(-log_target(np.asarray(x, dtype=float)))


def finite_gradient(fn, x: np.ndarray, step: float = 1e-5) -> np.ndarray:
    """Central-difference gradient."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        direction = np.zeros_like(x)
        direction[i] = step
        grad[i] = (fn(x + direction) - fn(x - direction)) / (2.0 * step)
    return grad


def finite_hessian(fn, x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian."""
    n = len(x)
    hess = np.zeros((n, n), dtype=float)
    f0 = fn(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step
        hess[i, i] = (fn(x + ei) - 2.0 * f0 + fn(x - ei)) / step**2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step
            hess_ij = (
                fn(x + ei + ej)
                - fn(x + ei - ej)
                - fn(x - ei + ej)
                + fn(x - ei - ej)
            ) / (4.0 * step**2)
            hess[i, j] = hess_ij
            hess[j, i] = hess_ij
    return hess


def newton_with_backtracking(
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 60,
) -> OptimizationRun:
    """Newton iteration with a simple line search and Hessian regularization."""
    x = np.asarray(x0, dtype=float).copy()
    path = [x.copy()]
    success = False

    for iteration in range(1, max_iter + 1):
        grad = finite_gradient(objective, x)
        if np.linalg.norm(grad) < tol:
            success = True
            break

        hess = finite_hessian(objective, x)
        ridge = 1e-8
        for _ in range(8):
            try:
                step = np.linalg.solve(hess + ridge * np.eye(len(x)), grad)
                break
            except np.linalg.LinAlgError:
                ridge *= 10.0
        else:
            step = grad

        current = objective(x)
        alpha = 1.0
        accepted = False
        while alpha > 1e-5:
            candidate = x - alpha * step
            if objective(candidate) < current:
                x = candidate
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # Fall back to a small gradient step when the Newton direction is bad.
            x = x - 0.05 * grad

        path.append(x.copy())
    else:
        iteration = max_iter

    return OptimizationRun(
        method="Newton",
        start=format_point(x0),
        solution=x,
        objective=objective(x),
        iterations=iteration,
        success=success or np.linalg.norm(finite_gradient(objective, x)) < 1e-5,
        path=np.asarray(path),
    )


def run_scipy_optimizer(method: str, x0: np.ndarray) -> OptimizationRun:
    """Run a SciPy local optimizer and retain its path."""
    path = [np.asarray(x0, dtype=float).copy()]

    def callback(xk: np.ndarray) -> None:
        path.append(np.asarray(xk, dtype=float).copy())

    options = {"maxiter": 300}
    if method == "Nelder-Mead":
        options["xatol"] = 1e-8
        options["fatol"] = 1e-8
    result = minimize(objective, x0, method=method, callback=callback, options=options)
    if not np.allclose(path[-1], result.x):
        path.append(np.asarray(result.x, dtype=float).copy())

    return OptimizationRun(
        method=method,
        start=format_point(x0),
        solution=np.asarray(result.x, dtype=float),
        objective=float(result.fun),
        iterations=int(result.nit),
        success=bool(result.success),
        path=np.asarray(path),
    )


def run_dual_annealing(seed: int = 123) -> OptimizationRun:
    """Run stochastic global search over a fixed box."""
    path: list[np.ndarray] = []

    def callback(x: np.ndarray, _fun: float, _context: int) -> bool:
        path.append(np.asarray(x, dtype=float).copy())
        return False

    result = dual_annealing(
        objective,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        maxiter=80,
        seed=seed,
        callback=callback,
        no_local_search=False,
    )
    if not path or not np.allclose(path[-1], result.x):
        path.append(np.asarray(result.x, dtype=float).copy())

    return OptimizationRun(
        method="Dual annealing",
        start="box [-5,5]^2",
        solution=np.asarray(result.x, dtype=float),
        objective=float(result.fun),
        iterations=int(result.nit),
        success=bool(result.success),
        path=np.asarray(path),
    )


def nearest_mode_distance(x: np.ndarray) -> float:
    """Distance from x to the closest analytic component mean."""
    return float(min(np.linalg.norm(x - MU1), np.linalg.norm(x - MU2)))


def assigned_mode(x: np.ndarray) -> str:
    """Label the nearest component mean."""
    return "upper-right" if np.linalg.norm(x - MU1) <= np.linalg.norm(x - MU2) else "lower-left"


def format_point(x: np.ndarray) -> str:
    """Format a two-dimensional point for compact tables."""
    return f"({x[0]:.2f}, {x[1]:.2f})"


def optimizer_summary(runs: list[OptimizationRun]) -> pd.DataFrame:
    """Create a compact summary table for optimizer outcomes."""
    rows = []
    for run in runs:
        rows.append(
            {
                "Method": run.method,
                "Start": run.start,
                "Solution": format_point(run.solution),
                "Mode": assigned_mode(run.solution),
                "Objective": f"{run.objective:.5f}",
                "Distance to mode": f"{nearest_mode_distance(run.solution):.3e}",
                "Iterations": run.iterations,
                "Success": run.success,
            }
        )
    return pd.DataFrame(rows)


def make_objective_grid(limit: float = 4.0, n: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, Z arrays for contour plots."""
    grid = np.linspace(-limit, limit, n)
    x_grid, y_grid = np.meshgrid(grid, grid)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    z_grid = np.asarray([objective(point) for point in points]).reshape(x_grid.shape)
    return x_grid, y_grid, z_grid


def bfgs_basin_grid() -> pd.DataFrame:
    """Run BFGS from a grid of starts to show local basin dependence."""
    starts = np.linspace(-3.0, 3.0, 7)
    rows = []
    for x0 in starts:
        for y0 in starts:
            run = run_scipy_optimizer("BFGS", np.array([x0, y0]))
            rows.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "mode": 1 if assigned_mode(run.solution) == "upper-right" else -1,
                    "objective": run.objective,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    common_start = np.array([3.0, -2.5])

    runs = [
        newton_with_backtracking(common_start),
        run_scipy_optimizer("BFGS", common_start),
        run_scipy_optimizer("Nelder-Mead", common_start),
        run_dual_annealing(seed=609),
    ]
    basin = bfgs_basin_grid()
    summary = optimizer_summary(runs)
    best_objective = min(run.objective for run in runs)

    print("Numerical optimization comparison")
    for run in runs:
        print(
            f"  {run.method:14s} solution={format_point(run.solution)} "
            f"objective={run.objective:.5f} iterations={run.iterations}"
        )

    report = ModelReport(
        "Latent-Regime Likelihoods and Optimizer Basins",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "An economist may estimate a model with two latent regimes. Each regime can explain "
        "the same data with a different parameter vector.\n\n"
        "The object is the parameter vector that minimizes a negative log likelihood. The "
        "tutorial uses a two-dimensional Gaussian mixture so the full criterion can be drawn.\n\n"
        "A single local optimizer can report the basin reached from its starting value. Restart "
        "grids and global search check whether the estimate changes across basins."
    )

    report.add_equations(
        r"""
Let $\theta=(\theta_1,\theta_2)$ denote the structural parameter vector. The
target density is a stylized latent-regime likelihood. Each regime has its own
mean, while the researcher observes only the mixture:

$$
\begin{aligned}
p(\theta)
&= \omega \phi(\theta; \mu_1, \Sigma) \\
&\quad + (1-\omega)\phi(\theta; \mu_2, \Sigma).
\end{aligned}
$$

The estimator minimizes the negative log likelihood:

$$
\min_{\theta \in \mathbb{R}^2} f(\theta),
\qquad
f(\theta) = -\log p(\theta).
$$

Newton's method uses local curvature around the current guess:

$$
\theta_{n+1} = \theta_n - H_f(\theta_n)^{-1}\nabla f(\theta_n).
$$

"""
    )

    report.add_model_setup(
        f"| Object | Value |\n"
        f"|--------|-------|\n"
        f"| $\\mu_1$ | {format_point(MU1)} |\n"
        f"| $\\mu_2$ | {format_point(MU2)} |\n"
        f"| $\\Sigma$ | [[1.0, 0.5], [0.5, 1.0]] |\n"
        f"| Mixing probability $\\omega$ | {MIXING_PROB:.1f} |\n"
        f"| Local-method start | {format_point(common_start)} |\n"
        f"| Global search box | $[-5,5]^2$ |"
    )

    report.add_solution_method(
        "All methods minimize the same criterion $f(\\theta)$. The local methods start from "
        "the same off-diagonal value. Their paths show how different search rules choose a "
        "basin. Dual annealing searches the full box before local polishing. The BFGS restart "
        "grid maps the basin reached from each start.\n\n"
        "```text\n"
        "Algorithm: optimizer diagnostics for a latent-regime likelihood\n"
        "Input: objective f(theta), starting value theta_0, search box B\n"
        "Output: candidate estimates, paths, basin diagnostics\n"
        "1. Run local optimizers from theta_0:\n"
        "       Newton: update with a regularized Hessian and backtracking line search\n"
        "       BFGS: update an inverse-Hessian approximation from gradient changes\n"
        "       Nelder-Mead: move a simplex using only objective values\n"
        "2. Run global search over B with stochastic uphill moves and local polishing\n"
        "3. For each candidate theta_hat, record f(theta_hat) and its nearest regime mean\n"
        "4. Restart BFGS on a grid of initial values to map basins of attraction\n"
        "5. Read instability across starts as information about the likelihood surface\n"
        "```\n\n"
        "The known component means make basin labels observable in this teaching example. In "
        "empirical work, the restart grid is the main diagnostic."
    )

    x_grid, y_grid, z_grid = make_objective_grid()
    fig1, ax1 = plt.subplots(figsize=(7.2, 6.2))
    contour = ax1.contour(x_grid, y_grid, z_grid, levels=22, cmap="viridis")
    ax1.clabel(contour, fontsize=7, inline=True)
    colors = {
        "Newton": "tab:blue",
        "BFGS": "tab:orange",
        "Nelder-Mead": "tab:green",
        "Dual annealing": "crimson",
    }
    for run in runs:
        path = run.path
        ax1.plot(path[:, 0], path[:, 1], marker="o", markersize=3.5, label=run.method, color=colors[run.method])
        ax1.scatter(path[0, 0], path[0, 1], marker="s", s=45, color=colors[run.method])
    ax1.scatter(
        [MU1[0], MU2[0]],
        [MU1[1], MU2[1]],
        marker="*",
        s=180,
        color="black",
        label="Component means",
    )
    ax1.set_xlabel(r"$\theta_1$")
    ax1.set_ylabel(r"$\theta_2$")
    ax1.set_title("Optimizer Paths on a Bimodal Objective")
    ax1.legend(loc="upper left")
    report.add_figure(
        "figures/optimizer-paths.png",
        "Optimizer paths over negative log-density contours",
        fig1,
        description=(
            "The contour plot shows two parameter vectors with the same likelihood value. Newton "
            "and BFGS reach the upper-right basin. Nelder-Mead and dual annealing reach the "
            "lower-left basin."
        ),
    )

    fig2, ax2 = plt.subplots()
    for run in runs:
        values = np.asarray([objective(point) for point in run.path])
        gap = np.maximum(values - best_objective, 1e-10)
        ax2.semilogy(np.arange(len(gap)), gap, marker="o", markersize=3.5, label=run.method)
    ax2.set_xlabel("Recorded iteration")
    ax2.set_ylabel("Objective gap")
    ax2.set_title("Convergence to the Best Found Objective")
    ax2.legend()
    report.add_figure(
        "figures/convergence.png",
        "Objective gaps along recorded optimizer paths",
        fig2,
        description=(
            "Iteration counts measure progress after a basin has been chosen. A zero objective "
            "gap does not show that the other basin was checked."
        ),
    )

    fig3, ax3 = plt.subplots(figsize=(6.2, 5.6))
    mode_colors = basin["mode"].map({1: "tab:blue", -1: "tab:orange"}).to_numpy()
    ax3.scatter(basin["x0"], basin["y0"], c=mode_colors, s=95, edgecolor="black", linewidth=0.6)
    ax3.scatter(
        [MU1[0], MU2[0]],
        [MU1[1], MU2[1]],
        marker="*",
        s=190,
        color="black",
        label="component means",
    )
    ax3.scatter([], [], color="tab:blue", edgecolor="black", label="upper-right mode")
    ax3.scatter([], [], color="tab:orange", edgecolor="black", label="lower-left mode")
    ax3.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax3.axvline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax3.set_xlabel(r"Initial $\theta_1$")
    ax3.set_ylabel(r"Initial $\theta_2$")
    ax3.set_title("BFGS Basins of Attraction")
    ax3.legend(loc="upper left")
    report.add_figure(
        "figures/basin-map.png",
        "BFGS solutions from different starting points",
        fig3,
        description=(
            "Each dot is a BFGS starting value. The color records the regime mean reached by "
            "the optimizer."
        ),
    )

    report.add_table(
        "tables/optimizer-summary.csv",
        "Optimizer outcomes",
        summary,
        description=(
            "The local methods share one starting value. Dual annealing searches the box first."
        ),
    )

    report.add_results(
        f"The best objective found is {best_objective:.5f}. The equal mixture weights make both "
        "regime labels fit equally well. The table therefore does not rank labels. It shows that "
        "the reported estimate can depend on initialization. The restart grid shows that nearby "
        "starts can point to different regimes."
    )

    report.add_takeaway(
        "A local optimum is not enough in a latent-regime likelihood. Check the criterion around "
        "the estimate and rerun from many starts. Use a global pass when local starts find "
        "different basins. Interpret the estimate only after those checks."
    )

    report.add_references(
        [
            "[Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed. Springer.](https://doi.org/10.1007/978-0-387-40065-5)",
            "[Goffe, W. L., Ferrier, G. D., and Rogers, J. (1994). Global Optimization of Statistical Functions with Simulated Annealing. *Journal of Econometrics*, 60(1-2), 65-99.](https://doi.org/10.1016/0304-4076(94)90038-8)",
            "[Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261-272.](https://doi.org/10.1038/s41592-019-0686-2)",
        ]
    )

    report.write("README.md")


if __name__ == "__main__":
    main()
