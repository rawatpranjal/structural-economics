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
from lib.plotting import save_figure, save_thumbnail, setup_style


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
        # Armijo sufficient-decrease constant: accept a step only when it
        # reduces the objective by at least c1 * alpha * (grad . step).
        c1 = 1e-4
        directional_decrease = float(np.dot(grad, step))
        alpha = 1.0
        accepted = False
        while alpha > 1e-5:
            candidate = x - alpha * step
            if objective(candidate) <= current - c1 * alpha * directional_decrease:
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

    # scipy's dual_annealing always runs its fixed annealing schedule to the
    # end, so result.nit == maxiter and result.message == "Maximum number of
    # iteration reached" on every run; result.success only reports that the
    # run finished without error. It is not a true convergence certificate.
    # Report success only when the polished solution actually reached a
    # posterior mode, which is what the Success column in the report claims.
    distance_to_mode = min(
        float(np.linalg.norm(result.x - mode)) for mode in (MU1, MU2)
    )
    reached_mode = distance_to_mode < 0.1
    assert reached_mode, (
        f"dual_annealing finished at {np.round(result.x, 3)} with distance "
        f"{distance_to_mode:.4f} to the nearest mode; it did not reach a "
        f"basin, so Success=True would be a stale flag."
    )

    return OptimizationRun(
        method="Dual annealing",
        start="box [-5,5]^2",
        solution=np.asarray(result.x, dtype=float),
        objective=float(result.fun),
        iterations=int(result.nit),
        success=bool(result.success) and reached_mode,
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
    save_figure(fig1, "figures/optimizer-paths.png", dpi=150)

    fig2, ax2 = plt.subplots()
    for run in runs:
        values = np.asarray([objective(point) for point in run.path])
        gap = np.maximum(values - best_objective, 1e-10)
        ax2.semilogy(np.arange(len(gap)), gap, marker="o", markersize=3.5, label=run.method)
    ax2.set_xlabel("Recorded iteration")
    ax2.set_ylabel("Objective gap")
    ax2.set_title("Convergence to the Best Found Objective")
    ax2.legend()
    save_figure(fig2, "figures/convergence.png", dpi=150)

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
    save_figure(fig3, "figures/basin-map.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv("tables/optimizer-summary.csv", index=False)

    save_thumbnail("figures/optimizer-paths.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
