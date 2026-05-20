#!/usr/bin/env python3
"""Smolyak sparse-grid collocation for a multi-sector growth model.

A planner allocates aggregate output across consumption and N sectoral capitals
under one common productivity shock. The state has dimension d = N + 1, large
enough that a tensor Chebyshev grid grows wastefully. Smolyak sparse grids
deliver near-tensor accuracy with polynomial node growth.

Reference: Judd, Maliar, Maliar, Valero (Journal of Economic Dynamics and
Control, 2014).
"""

from __future__ import annotations

import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.chebyshev import chebvander
from scipy.optimize import brentq
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


# Model parameters
BETA = 0.95
ALPHA = 0.36
A_SECTORS = np.array([1.0, 0.9, 1.1, 0.8])
N_SECTORS = A_SECTORS.size
RHO_Z = 0.95
SIGMA_Z = 0.01

# Numerical settings
DIM = N_SECTORS + 1
MU_LEVELS = [1, 2, 3]
QUAD_NODES = 3
TIME_TOL = 1e-7
TIME_MAX_ITER = 200


# --- 1D nested Chebyshev extrema --------------------------------------------

def m_level(level: int) -> int:
    """Doubling-rule node count m_i: m_1 = 1, m_i = 2^(i-1) + 1 for i >= 2."""
    if level <= 1:
        return 1
    return 2 ** (level - 1) + 1


def cheb_extrema_1d(level: int) -> np.ndarray:
    """Nested Chebyshev extrema nodes on [-1, 1] at the given level."""
    m = m_level(level)
    if m == 1:
        return np.array([0.0])
    j = np.arange(m)
    return -np.cos(np.pi * j / (m - 1))


# --- Smolyak grid construction ---------------------------------------------

def admissible_level_indices(d: int, mu: int) -> list[tuple[int, ...]]:
    """Multi-indices i with i_k >= 1 and mu + 1 <= sum_k i_k <= mu + d.

    Minimum sum is d (all ones), so the lower bound bites only when mu < 0.
    """
    out: list[tuple[int, ...]] = []
    for combo in product(range(1, mu + 2), repeat=d):
        total = sum(combo)
        if total <= mu + d and total >= max(mu + 1, d):
            out.append(combo)
    return out


def smolyak_grid_unit(d: int, mu: int) -> np.ndarray:
    """Smolyak sparse grid H(d, mu) on [-1, 1]^d, deduplicated."""
    raw: list[tuple[float, ...]] = []
    for combo in admissible_level_indices(d, mu):
        axes = [cheb_extrema_1d(level) for level in combo]
        for pt in product(*axes):
            raw.append(pt)
    arr = np.asarray(raw, dtype=float)
    keys = np.round(arr, 12)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return arr[np.sort(idx)]


# --- Smolyak polynomial basis ----------------------------------------------

def degree_level(degree: int) -> int:
    """Smallest 1D level whose polynomial space contains Chebyshev T_degree.

    The 1D polynomial space at level i has degrees 0, ..., m_i - 1.
    m_1 - 1 = 0, m_2 - 1 = 2, m_3 - 1 = 4, m_4 - 1 = 8, m_5 - 1 = 16, ...
    """
    if degree <= 0:
        return 0
    if degree == 1:
        return 1
    return int(np.ceil(np.log2(degree)))


def admissible_basis_degrees(d: int, mu: int) -> np.ndarray:
    """Multi-degrees (a_1, ..., a_d) with sum_k level(a_k) <= mu."""
    max_deg = 2 ** mu if mu >= 1 else 0
    levels = np.array([degree_level(a) for a in range(max_deg + 1)])
    rows: list[tuple[int, ...]] = []
    for combo in product(range(max_deg + 1), repeat=d):
        if int(levels[list(combo)].sum()) <= mu:
            rows.append(combo)
    return np.array(rows, dtype=int)


def smolyak_basis_matrix(points: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    """Evaluate the Smolyak Chebyshev tensor basis at given points.

    Phi[n, k] = prod_j T_{degrees[k, j]}(points[n, j]).
    """
    n_pts, d = points.shape
    max_deg = int(degrees.max()) if degrees.size > 0 else 0
    if max_deg == 0:
        return np.ones((n_pts, max(1, degrees.shape[0])))
    cheb_eval = [chebvander(points[:, k], max_deg) for k in range(d)]
    n_basis = degrees.shape[0]
    Phi = np.ones((n_pts, n_basis))
    for j in range(n_basis):
        col = np.ones(n_pts)
        for k in range(d):
            col = col * cheb_eval[k][:, degrees[j, k]]
        Phi[:, j] = col
    return Phi


def smolyak_count(d: int, mu: int) -> int:
    """Number of Smolyak nodes (and basis polynomials) for (d, mu)."""
    return smolyak_grid_unit(d, mu).shape[0]


# --- Model primitives ------------------------------------------------------

def sector_shares() -> np.ndarray:
    """Closed-form sector saving shares omega_i."""
    weighted = A_SECTORS ** (1.0 / (1.0 - ALPHA))
    return weighted / weighted.sum()


def Z_const() -> float:
    """Z = sum_j A_j^{1/(1-alpha)}."""
    return float(np.sum(A_SECTORS ** (1.0 / (1.0 - ALPHA))))


def steady_state_capital() -> np.ndarray:
    """Deterministic steady-state sectoral capitals."""
    omega = sector_shares()
    Z = Z_const()
    S_ss = (ALPHA * BETA) ** (1.0 / (1.0 - ALPHA)) * Z
    return omega * S_ss


def aggregate_output(k: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Y = exp(z) * sum_i A_i k_i^alpha."""
    return np.exp(z) * np.sum(A_SECTORS * (k ** ALPHA), axis=-1)


# --- Coordinate transforms -------------------------------------------------

def scale_to_unit(states: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map state from [lower, upper] component-wise to [-1, 1]."""
    return 2.0 * (states - lower) / (upper - lower) - 1.0


def unit_to_state(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map [-1, 1]^d coordinates back to state space."""
    return lower + 0.5 * (points + 1.0) * (upper - lower)


# --- Solver ----------------------------------------------------------------

def gauss_hermite_normalized() -> tuple[np.ndarray, np.ndarray]:
    """Probabilist Gauss-Hermite for E[g(eps)] with eps ~ N(0,1)."""
    nodes, weights = np.polynomial.hermite.hermgauss(QUAD_NODES)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def solve_smolyak(
    mu: int,
    bounds_lower: np.ndarray,
    bounds_upper: np.ndarray,
    tol: float = TIME_TOL,
    max_iter: int = TIME_MAX_ITER,
) -> dict:
    """Solve the planner policy by Smolyak time iteration.

    Returns a dict with theta, degrees, grid (in state coords), Phi matrix,
    bounds, iteration count, and wall-clock seconds.
    """
    omega = sector_shares()
    Z = Z_const()

    grid_unit = smolyak_grid_unit(DIM, mu)
    degrees = admissible_basis_degrees(DIM, mu)
    grid_state = unit_to_state(grid_unit, bounds_lower, bounds_upper)
    Phi = smolyak_basis_matrix(grid_unit, degrees)
    Phi_inv = np.linalg.inv(Phi)

    k_grid = grid_state[:, :N_SECTORS]
    z_grid = grid_state[:, N_SECTORS]
    Y_grid = aggregate_output(k_grid, z_grid)

    # Initial guess: constant savings fraction 0.3 (not the closed-form value
    # 0.342, so the iteration is doing real work).
    log_S = np.log(0.30 * Y_grid)
    theta = Phi_inv @ log_S

    shocks, q_weights = gauss_hermite_normalized()
    shocks = SIGMA_Z * shocks

    start = time.perf_counter()
    iterations = 0
    for iteration in range(max_iter):
        theta_old = theta.copy()

        S_old = np.exp(Phi @ theta_old)
        k_next = omega[None, :] * S_old[:, None]

        z_next = RHO_Z * z_grid[:, None] + shocks[None, :]
        n_grid = grid_state.shape[0]

        k_next_q = np.broadcast_to(
            k_next[:, None, :], (n_grid, QUAD_NODES, N_SECTORS)
        )
        next_flat_k = k_next_q.reshape(-1, N_SECTORS)
        next_flat_z = z_next.reshape(-1)
        next_states = np.concatenate(
            [next_flat_k, next_flat_z[:, None]], axis=1
        )

        next_unit = scale_to_unit(next_states, bounds_lower, bounds_upper)
        next_unit = np.clip(next_unit, -1.0, 1.0)
        Phi_next = smolyak_basis_matrix(next_unit, degrees)
        log_S_next = Phi_next @ theta_old
        S_next = np.exp(log_S_next).reshape(n_grid, QUAD_NODES)
        Y_next = aggregate_output(next_flat_k, next_flat_z).reshape(
            n_grid, QUAD_NODES
        )
        c_next = np.maximum(Y_next - S_next, 1e-10)

        z_next_grid = z_next
        E_term = np.sum(
            q_weights[None, :] * np.exp(z_next_grid) / c_next, axis=1
        )
        const = BETA * ALPHA * Z ** (1.0 - ALPHA) * E_term

        new_S = np.empty(n_grid)
        for n in range(n_grid):
            log_const = np.log(const[n])

            def residual(s: float) -> float:
                return (
                    log_const + np.log(Y_grid[n] - s) + (ALPHA - 1.0) * np.log(s)
                )

            lo = max(1e-10, 1e-6 * Y_grid[n])
            hi = (1.0 - 1e-9) * Y_grid[n]
            new_S[n] = brentq(residual, lo, hi, xtol=1e-14)

        theta_new = Phi_inv @ np.log(new_S)
        delta = float(np.max(np.abs(theta_new - theta_old)))
        theta = theta_new
        iterations = iteration + 1
        if delta < tol:
            break

    elapsed = time.perf_counter() - start

    return {
        "mu": mu,
        "theta": theta,
        "degrees": degrees,
        "grid_state": grid_state,
        "Phi": Phi,
        "bounds_lower": bounds_lower,
        "bounds_upper": bounds_upper,
        "iterations": iterations,
        "seconds": elapsed,
    }


def smolyak_savings(
    states: np.ndarray, solution: dict
) -> np.ndarray:
    """Total savings S(state) implied by a Smolyak solution."""
    unit = scale_to_unit(states, solution["bounds_lower"], solution["bounds_upper"])
    unit = np.clip(unit, -1.0, 1.0)
    Phi = smolyak_basis_matrix(unit, solution["degrees"])
    return np.exp(Phi @ solution["theta"])


def closed_form_savings(states: np.ndarray) -> np.ndarray:
    """Closed-form total savings: S = alpha * beta * Y."""
    k = states[:, :N_SECTORS]
    z = states[:, N_SECTORS]
    return ALPHA * BETA * aggregate_output(k, z)


# --- Euler error diagnostic ------------------------------------------------

def euler_errors_at(states: np.ndarray, solution: dict) -> np.ndarray:
    """Relative Euler equation error |lhs/rhs - 1| at given states."""
    omega = sector_shares()
    Z = Z_const()
    bounds_lower = solution["bounds_lower"]
    bounds_upper = solution["bounds_upper"]

    k = states[:, :N_SECTORS]
    z = states[:, N_SECTORS]
    Y = aggregate_output(k, z)
    S = smolyak_savings(states, solution)
    c = np.maximum(Y - S, 1e-10)

    k_next = omega[None, :] * S[:, None]
    shocks, q_weights = gauss_hermite_normalized()
    shocks = SIGMA_Z * shocks

    z_next = RHO_Z * z[:, None] + shocks[None, :]
    n_pts = states.shape[0]
    k_next_q = np.broadcast_to(
        k_next[:, None, :], (n_pts, QUAD_NODES, N_SECTORS)
    )
    next_flat_k = k_next_q.reshape(-1, N_SECTORS)
    next_flat_z = z_next.reshape(-1)
    next_states = np.concatenate(
        [next_flat_k, next_flat_z[:, None]], axis=1
    )

    S_next = smolyak_savings(next_states, solution).reshape(n_pts, QUAD_NODES)
    Y_next = aggregate_output(next_flat_k, next_flat_z).reshape(n_pts, QUAD_NODES)
    c_next = np.maximum(Y_next - S_next, 1e-10)

    E_term = np.sum(q_weights[None, :] * np.exp(z_next) / c_next, axis=1)
    rhs = BETA * ALPHA * Z ** (1.0 - ALPHA) * S ** (ALPHA - 1.0) * E_term
    lhs = 1.0 / c
    return np.abs(lhs / rhs - 1.0)


# --- Grid count comparison -------------------------------------------------

def grid_count_table(d_list: list[int], mu_list: list[int]) -> pd.DataFrame:
    """Tensor m^d vs Smolyak node counts for (d, mu) pairs."""
    rows = []
    for d in d_list:
        for mu in mu_list:
            tensor_m = 2 ** mu + 1  # nodes per dim at extrema level mu+1
            tensor = tensor_m ** d
            smolyak = smolyak_count(d, mu)
            rows.append(
                {
                    "Dimension d": d,
                    "Smolyak level mu": mu,
                    "Tensor nodes": tensor,
                    "Smolyak nodes": smolyak,
                    "Tensor / Smolyak ratio": tensor / smolyak,
                }
            )
    return pd.DataFrame(rows)


def format_count_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Tensor nodes"] = out["Tensor nodes"].map(lambda x: f"{int(x):,}")
    out["Smolyak nodes"] = out["Smolyak nodes"].map(lambda x: f"{int(x):,}")
    out["Tensor / Smolyak ratio"] = out["Tensor / Smolyak ratio"].map(
        lambda x: f"{float(x):.2f}"
    )
    return out


# --- Figures ---------------------------------------------------------------

def figure_grid_2d() -> plt.Figure:
    """Four-panel scatter: tensor 5x5 vs Smolyak mu = 1, 2, 3 in 2D."""
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    j = np.arange(5)
    tensor_axis = -np.cos(np.pi * j / 4)
    xx, yy = np.meshgrid(tensor_axis, tensor_axis)
    axes[0].scatter(xx.flatten(), yy.flatten(), s=24, color="crimson")
    axes[0].set_title(f"Tensor 5x5 ({xx.size} nodes)")
    for ax, mu in zip(axes[1:], [1, 2, 3]):
        pts = smolyak_grid_unit(2, mu)
        ax.scatter(pts[:, 0], pts[:, 1], s=24, color="steelblue")
        ax.set_title(f"Smolyak mu={mu} ({pts.shape[0]} nodes)")
    for ax in axes:
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        ax.set_aspect("equal")
    fig.suptitle("Tensor vs Smolyak Nodes in Two Dimensions", y=1.02)
    fig.tight_layout()
    return fig


def figure_count_scaling(df: pd.DataFrame) -> plt.Figure:
    """Tensor vs Smolyak node count across dimensions."""
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    dims = sorted(df["Dimension d"].unique())
    for mu in sorted(df["Smolyak level mu"].unique()):
        sub = df[df["Smolyak level mu"] == mu].sort_values("Dimension d")
        ax.plot(
            sub["Dimension d"],
            sub["Smolyak nodes"],
            marker="o",
            label=f"Smolyak mu={mu}",
        )
    for mu in sorted(df["Smolyak level mu"].unique()):
        sub = df[df["Smolyak level mu"] == mu].sort_values("Dimension d")
        ax.plot(
            sub["Dimension d"],
            sub["Tensor nodes"],
            marker="s",
            linestyle="--",
            alpha=0.6,
            label=f"Tensor m={2 ** mu + 1}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("State dimension d")
    ax.set_ylabel("Number of collocation nodes")
    ax.set_title("Smolyak Polynomial Growth vs Tensor Exponential Growth")
    ax.legend(ncol=2, fontsize=9)
    return fig


def figure_policy_slice(
    solutions: list[dict], bounds_lower: np.ndarray, bounds_upper: np.ndarray
) -> plt.Figure:
    """Closed-form vs Smolyak total savings along a k_1 slice."""
    k_ss = steady_state_capital()
    grid_k1 = np.linspace(0.6 * k_ss[0], 1.4 * k_ss[0], 200)
    states = np.zeros((grid_k1.size, DIM))
    states[:, 0] = grid_k1
    states[:, 1] = k_ss[1]
    states[:, 2] = k_ss[2]
    states[:, 3] = k_ss[3]
    states[:, 4] = 0.0

    closed = closed_form_savings(states)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.plot(grid_k1, closed, color="black", linestyle="--", label="closed form")
    for sol in solutions:
        smolyak = smolyak_savings(states, sol)
        ax.plot(grid_k1, smolyak, label=f"Smolyak mu={sol['mu']}")
    ax.set_xlabel("Sector 1 capital k_1")
    ax.set_ylabel("Total savings S")
    ax.set_title("Policy Slice with Other Sectors at Steady State")
    ax.legend()
    return fig


def figure_euler_errors(
    solutions: list[dict],
    test_states: np.ndarray,
) -> plt.Figure:
    """Empirical CDF of log10 relative Euler errors per method."""
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for sol in solutions:
        errs = euler_errors_at(test_states, sol)
        log_errs = np.log10(np.maximum(errs, 1e-16))
        log_errs = np.sort(log_errs)
        cdf = np.linspace(0.0, 1.0, log_errs.size, endpoint=False)
        ax.plot(log_errs, cdf, label=f"Smolyak mu={sol['mu']}")
    ax.set_xlabel("log10 relative Euler error")
    ax.set_ylabel("Empirical CDF over test states")
    ax.set_title("Off-Grid Euler Equation Errors")
    ax.legend()
    return fig


# --- Main entry ------------------------------------------------------------

def main() -> None:
    setup_style()

    omega = sector_shares()
    k_ss = steady_state_capital()
    z_sd = SIGMA_Z / np.sqrt(1.0 - RHO_Z ** 2)
    bounds_lower = np.concatenate([0.5 * k_ss, np.array([-3.0 * z_sd])])
    bounds_upper = np.concatenate([1.5 * k_ss, np.array([3.0 * z_sd])])

    print("Smolyak sparse grids tutorial")
    print(f"  state dim d={DIM}")
    print(f"  steady-state capitals={np.array2string(k_ss, precision=4)}")
    print(f"  sector shares={np.array2string(omega, precision=4)}")

    # Solve at each mu
    solutions = []
    for mu in MU_LEVELS:
        sol = solve_smolyak(mu, bounds_lower, bounds_upper)
        sol["nodes"] = sol["grid_state"].shape[0]
        print(
            f"  mu={mu:d}  nodes={sol['nodes']:>4d}  iters={sol['iterations']:>3d}"
            f"  time={sol['seconds']:.2f}s"
        )
        solutions.append(sol)

    # Off-grid test set via Sobol within the interior to keep next-period
    # states inside the bounded box for clean error measurement.
    sobol = qmc.Sobol(d=DIM, seed=0)
    sobol_unit = sobol.random(10_000)
    interior_lower = np.concatenate([0.6 * k_ss, np.array([-2.5 * z_sd])])
    interior_upper = np.concatenate([1.4 * k_ss, np.array([2.5 * z_sd])])
    test_states = interior_lower + sobol_unit * (interior_upper - interior_lower)

    # Accuracy table
    accuracy_rows = []
    for sol in solutions:
        errs = euler_errors_at(test_states, sol)
        S_smolyak = smolyak_savings(test_states, sol)
        S_truth = closed_form_savings(test_states)
        savings_err = np.abs(S_smolyak - S_truth) / S_truth
        accuracy_rows.append(
            {
                "Method": f"Smolyak mu={sol['mu']}",
                "Nodes": sol["nodes"],
                "Max Euler error": float(np.max(errs)),
                "Median Euler error": float(np.median(errs)),
                "Max savings error": float(np.max(savings_err)),
                "Seconds": float(sol["seconds"]),
            }
        )
    accuracy_df = pd.DataFrame(accuracy_rows)

    # Grid count table for the catalog comparison (includes d=5 used here)
    count_df = grid_count_table([2, 4, 5, 6, 8, 10], [1, 2, 3])

    # Figures
    fig_grid = figure_grid_2d()
    fig_count = figure_count_scaling(count_df)
    fig_slice = figure_policy_slice(solutions, bounds_lower, bounds_upper)
    fig_errors = figure_euler_errors(solutions, test_states)

    # Formatted accuracy table for display
    accuracy_display = accuracy_df.copy()
    accuracy_display["Nodes"] = accuracy_display["Nodes"].map(lambda x: f"{int(x)}")
    accuracy_display["Max Euler error"] = accuracy_display["Max Euler error"].map(
        lambda x: f"{float(x):.2e}"
    )
    accuracy_display["Median Euler error"] = accuracy_display["Median Euler error"].map(
        lambda x: f"{float(x):.2e}"
    )
    accuracy_display["Max savings error"] = accuracy_display["Max savings error"].map(
        lambda x: f"{float(x):.2e}"
    )
    accuracy_display["Seconds"] = accuracy_display["Seconds"].map(
        lambda x: f"{float(x):.2f}"
    )

    count_display = format_count_table(count_df)

    # Save figures
    save_figure(fig_count, "figures/grid-size-scaling.png", dpi=150)
    save_figure(fig_grid, "figures/smolyak-grid-2d.png", dpi=150)
    save_figure(fig_slice, "figures/policy-slice.png", dpi=150)
    save_figure(fig_errors, "figures/euler-errors.png", dpi=150)

    # Save tables
    Path("tables").mkdir(parents=True, exist_ok=True)
    count_display.to_csv("tables/grid-counts.csv", index=False)
    accuracy_display.to_csv("tables/accuracy.csv", index=False)

    # Thumbnail
    save_thumbnail("figures/grid-size-scaling.png", "figures/thumb.png")

    print("Figures and tables saved.")


if __name__ == "__main__":
    main()
