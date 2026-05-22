#!/usr/bin/env python3
"""Reduced-form VAR estimation by OLS, recursive Cholesky identification under
two orderings, and IRF Monte-Carlo accuracy versus sample size."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


# ---------------------------------------------------------------------------
# Constants / true DGP
# ---------------------------------------------------------------------------

TRUE_A1 = np.array([[0.55, 0.10],
                    [0.05, 0.60]])
TRUE_A2 = np.array([[0.20, 0.00],
                    [0.00, 0.10]])
TRUE_SIGMA_U = np.array([[1.00, 0.40],
                          [0.40, 0.50]])

T_MAIN = 400
BURN = 200
HORIZON = 20
MC_TRIALS = 200
N_GRID = [100, 200, 400, 800, 1600]
SEED = 12345


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_var2(
    A1: np.ndarray,
    A2: np.ndarray,
    Sigma_u: np.ndarray,
    T: int,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a zero-intercept VAR(2) and return T post-burn observations.

    Returns y of shape (T, 2).
    """
    P = np.linalg.cholesky(Sigma_u)
    total = T + burn_in
    y = np.zeros((total, 2))
    for t in range(2, total):
        eps = P @ rng.standard_normal(2)
        y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + eps
    return y[burn_in:]


# ---------------------------------------------------------------------------
# OLS estimation
# ---------------------------------------------------------------------------

def fit_ols_var2(y: np.ndarray) -> dict:
    """Estimate a VAR(2) with intercept by OLS.

    Design matrix has rows [1, y_{t-1}', y_{t-2}'] of shape (T-2, 5).
    Returns dict with keys: c_hat, A1_hat, A2_hat, Sigma_u_hat, residuals.
    """
    T = y.shape[0]
    # Build stacked design X and targets Y.
    ones = np.ones((T - 2, 1))
    X = np.hstack([ones, y[1:-1], y[:-2]])   # (T-2, 5)
    Y = y[2:]                                  # (T-2, 2)

    beta_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # (5, 2)
    residuals = Y - X @ beta_hat

    dof = T - 2 - 5
    Sigma_u_hat = residuals.T @ residuals / max(dof, 1)
    Sigma_u_hat = 0.5 * (Sigma_u_hat + Sigma_u_hat.T)

    c_hat = beta_hat[0]
    A1_hat = beta_hat[1:3].T   # (2, 2)
    A2_hat = beta_hat[3:5].T   # (2, 2)

    return {
        "c_hat": c_hat,
        "A1_hat": A1_hat,
        "A2_hat": A2_hat,
        "Sigma_u_hat": Sigma_u_hat,
        "residuals": residuals,
    }


# ---------------------------------------------------------------------------
# Companion matrix
# ---------------------------------------------------------------------------

def companion_matrix(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    """Return the 4x4 companion matrix F for a VAR(2).

    F = [[A1, A2],
         [I2, 0 ]]
    """
    F = np.zeros((4, 4))
    F[:2, :2] = A1
    F[:2, 2:] = A2
    F[2:, :2] = np.eye(2)
    return F


# ---------------------------------------------------------------------------
# Cholesky IRF
# ---------------------------------------------------------------------------

def cholesky_irf(
    A1: np.ndarray,
    A2: np.ndarray,
    Sigma_u: np.ndarray,
    horizon: int,
    order: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Compute Cholesky-identified IRFs under a given variable ordering.

    Permutes Sigma_u rows/cols by `order`, takes lower Cholesky, then
    inverts the permutation so all returned IRFs use the original
    variable ordering.

    Returns Phi of shape (horizon+1, 2, 2) where
    Phi[j, i, k] = response of variable i to structural shock k at horizon j.
    """
    perm = np.array(list(order))
    inv_perm = np.argsort(perm)

    # Permute covariance, factor, then un-permute columns.
    Sigma_perm = Sigma_u[np.ix_(perm, perm)]
    P_perm = np.linalg.cholesky(Sigma_perm)
    # P_perm maps structural shocks in permuted order to reduced-form residuals
    # in permuted order. Un-permute rows to get residuals in original order.
    P = P_perm[inv_perm, :]   # (2, 2); columns are structural shocks in permuted order
    # Also un-permute columns so shock k corresponds to original variable k.
    P = P[:, inv_perm]

    F = companion_matrix(A1, A2)
    J = np.zeros((2, 4))
    J[:, :2] = np.eye(2)

    Phi = np.zeros((horizon + 1, 2, 2))
    for j in range(horizon + 1):
        Fj = matrix_power(F, j)
        Phi[j] = J @ Fj @ J.T @ P

    return Phi


# ---------------------------------------------------------------------------
# Monte Carlo RMSE
# ---------------------------------------------------------------------------

def monte_carlo_irf_rmse(
    N_grid: list[int],
    A1: np.ndarray,
    A2: np.ndarray,
    Sigma_u: np.ndarray,
    mc_trials: int,
    horizon: int,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each N in N_grid, estimate mean and sd of IRF RMSE over mc_trials.

    Uses ordering (0, 1) (x first) throughout.
    Returns (n_grid_arr, mean_rmse, sd_rmse).
    """
    true_phi = cholesky_irf(A1, A2, Sigma_u, horizon, order=(0, 1))
    rng = np.random.default_rng(rng_seed)

    n_grid_arr = np.array(N_grid, dtype=float)
    mean_rmse = np.zeros(len(N_grid))
    sd_rmse = np.zeros(len(N_grid))

    for n_idx, N in enumerate(N_grid):
        trial_rmse = np.zeros(mc_trials)
        for trial in range(mc_trials):
            y = simulate_var2(A1, A2, Sigma_u, N, BURN, rng)
            fit = fit_ols_var2(y)
            est_phi = cholesky_irf(
                fit["A1_hat"], fit["A2_hat"], fit["Sigma_u_hat"], horizon, order=(0, 1)
            )
            diff = est_phi - true_phi
            trial_rmse[trial] = float(np.sqrt(np.mean(diff ** 2)))
        mean_rmse[n_idx] = float(np.mean(trial_rmse))
        sd_rmse[n_idx] = float(np.std(trial_rmse, ddof=1))

    return n_grid_arr, mean_rmse, sd_rmse


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_VAR_LABELS = ["Output gap", "Inflation"]
_SHOCK_LABELS = ["Output gap shock", "Inflation shock"]


def plot_irf_by_ordering(
    true_phi_a: np.ndarray,
    est_phi_a: np.ndarray,
    true_phi_b: np.ndarray,
    est_phi_b: np.ndarray,
    horizon: int,
    path: str | Path,
) -> None:
    """2x2 grid: 4 IRFs with true and estimated under each Cholesky ordering."""
    horizons = np.arange(horizon + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    panel_titles = [
        "Output gap to output gap shock",
        "Output gap to inflation shock",
        "Inflation to output gap shock",
        "Inflation to inflation shock",
    ]
    # (response variable i, shock variable k) pairs for the 4 panels.
    panels = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for ax_idx, (ax, (i, k)) in enumerate(zip(axes.flat, panels)):
        l1, = ax.plot(horizons, true_phi_a[:, i, k], color="black", linewidth=2.0,
                      label="True, ordering (x, pi)")
        l2, = ax.plot(horizons, est_phi_a[:, i, k], color="#2166ac", linewidth=1.8,
                      linestyle="--", label="Estimated, ordering (x, pi)")
        l3, = ax.plot(horizons, true_phi_b[:, i, k], color="#d95f02", linewidth=2.0,
                      label="True, ordering (pi, x)")
        l4, = ax.plot(horizons, est_phi_b[:, i, k], color="#d7191c", linewidth=1.8,
                      linestyle="--", label="Estimated, ordering (pi, x)")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_title(panel_titles[ax_idx], fontsize=10)
        if ax_idx >= 2:
            ax.set_xlabel("Horizon")
        if ax_idx % 2 == 0:
            ax.set_ylabel("Response")
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("IRFs under two Cholesky orderings: true vs. estimated (T=400)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


def _confidence_ellipse(
    ax: plt.Axes,
    cov: np.ndarray,
    n_std: float,
    center: np.ndarray,
    **kwargs,
) -> None:
    """Add a covariance confidence ellipse to ax."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(
        xy=center, width=width, height=height, angle=theta, **kwargs
    )
    ax.add_patch(ellipse)


def plot_residual_cholesky(
    residuals: np.ndarray,
    Sigma_u_hat: np.ndarray,
    Sigma_u_true: np.ndarray,
    path: str | Path,
) -> None:
    """Scatter of OLS residuals with covariance ellipses and Cholesky axes."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        residuals[:, 0], residuals[:, 1],
        alpha=0.35, s=12, color="#555555", zorder=1,
    )

    center = np.array([0.0, 0.0])

    # 95% confidence ellipse from estimated Sigma_u.
    _confidence_ellipse(ax, Sigma_u_hat, n_std=2.0, center=center,
                        edgecolor="#2166ac", facecolor="none",
                        linewidth=1.8, linestyle="-", label="95% ellipse (estimated)", zorder=3)

    # 95% confidence ellipse from true Sigma_u.
    _confidence_ellipse(ax, Sigma_u_true, n_std=2.0, center=center,
                        edgecolor="#d95f02", facecolor="none",
                        linewidth=1.8, linestyle="--", label="95% ellipse (true)", zorder=3)

    # Cholesky axes for ordering (x, pi): columns of lower Cholesky factor.
    P_xpi = np.linalg.cholesky(Sigma_u_hat)
    for j in range(2):
        vec = P_xpi[:, j]
        ax.annotate(
            "", xy=vec, xytext=center,
            arrowprops=dict(arrowstyle="->", color="#1a9850", linewidth=2.0),
            zorder=4,
        )

    # Cholesky axes for ordering (pi, x): permute, factor, un-permute.
    perm = np.array([1, 0])
    inv_perm = np.argsort(perm)
    Sigma_perm = Sigma_u_hat[np.ix_(perm, perm)]
    P_perm = np.linalg.cholesky(Sigma_perm)
    P_pix = P_perm[inv_perm, :][:, inv_perm]
    for j in range(2):
        vec = P_pix[:, j]
        ax.annotate(
            "", xy=vec, xytext=center,
            arrowprops=dict(arrowstyle="->", color="#762a83", linewidth=2.0),
            zorder=4,
        )

    # Dummy handles for Cholesky axes in legend.
    from matplotlib.lines import Line2D
    h_xpi = Line2D([0], [0], color="#1a9850", linewidth=2.0, label="Cholesky axes, ordering (x, pi)")
    h_pix = Line2D([0], [0], color="#762a83", linewidth=2.0, label="Cholesky axes, ordering (pi, x)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [h_xpi, h_pix], fontsize=8, loc="upper right")

    ax.axhline(0.0, color="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linewidth=0.4)
    ax.set_xlabel("Output gap residual")
    ax.set_ylabel("Inflation residual")
    ax.set_title("OLS residuals, covariance ellipses, and Cholesky shock axes")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


def plot_irf_rmse_by_n(
    N_grid: np.ndarray,
    mean_rmse: np.ndarray,
    sd_rmse: np.ndarray,
    path: str | Path,
) -> None:
    """Log-log RMSE vs N with shaded band and sqrt(N) reference line."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(N_grid, mean_rmse, color="#2166ac", linewidth=2.2, marker="o",
            markersize=5, label="Mean IRF RMSE")
    ax.fill_between(
        N_grid,
        np.maximum(mean_rmse - sd_rmse, 1e-6),
        mean_rmse + sd_rmse,
        alpha=0.25, color="#2166ac", label="+/- 1 sd",
    )

    # sqrt(N) reference line anchored at the first point.
    ref = mean_rmse[0] * np.sqrt(N_grid[0] / N_grid)
    ax.plot(N_grid, ref, color="black", linewidth=1.4, linestyle=":",
            label="slope -1/2 reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size N")
    ax.set_ylabel("Mean RMSE of estimated IRFs")
    ax.set_title("IRF estimation error shrinks at the sqrt(N) rate")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    setup_style()

    rng = np.random.default_rng(SEED)

    # Verify true DGP stability.
    F_true = companion_matrix(TRUE_A1, TRUE_A2)
    true_radius = float(np.max(np.abs(np.linalg.eigvals(F_true))))
    assert true_radius < 1.0, f"True DGP is unstable: spectral radius = {true_radius:.4f}"
    print(f"True companion spectral radius:      {true_radius:.4f}")

    # Headline simulation.
    y = simulate_var2(TRUE_A1, TRUE_A2, TRUE_SIGMA_U, T_MAIN, BURN, rng)
    fit = fit_ols_var2(y)

    F_est = companion_matrix(fit["A1_hat"], fit["A2_hat"])
    est_radius = float(np.max(np.abs(np.linalg.eigvals(F_est))))
    print(f"Estimated companion spectral radius: {est_radius:.4f}")

    print("\nTrue A1:")
    print(TRUE_A1)
    print("Estimated A1:")
    print(np.round(fit["A1_hat"], 4))
    print("Diff A1:")
    print(np.round(fit["A1_hat"] - TRUE_A1, 4))

    print("\nTrue A2:")
    print(TRUE_A2)
    print("Estimated A2:")
    print(np.round(fit["A2_hat"], 4))
    print("Diff A2:")
    print(np.round(fit["A2_hat"] - TRUE_A2, 4))

    print("\nTrue Sigma_u:")
    print(TRUE_SIGMA_U)
    print("Estimated Sigma_u:")
    print(np.round(fit["Sigma_u_hat"], 4))
    print("Diff Sigma_u:")
    print(np.round(fit["Sigma_u_hat"] - TRUE_SIGMA_U, 4))

    # Cholesky IRFs under two orderings.
    true_phi_a = cholesky_irf(TRUE_A1, TRUE_A2, TRUE_SIGMA_U, HORIZON, order=(0, 1))
    est_phi_a = cholesky_irf(fit["A1_hat"], fit["A2_hat"], fit["Sigma_u_hat"], HORIZON, order=(0, 1))
    true_phi_b = cholesky_irf(TRUE_A1, TRUE_A2, TRUE_SIGMA_U, HORIZON, order=(1, 0))
    est_phi_b = cholesky_irf(fit["A1_hat"], fit["A2_hat"], fit["Sigma_u_hat"], HORIZON, order=(1, 0))

    plot_irf_by_ordering(
        true_phi_a, est_phi_a, true_phi_b, est_phi_b,
        HORIZON, "figures/irf-by-ordering.png",
    )
    print("\nSaved figures/irf-by-ordering.png")

    plot_residual_cholesky(
        fit["residuals"], fit["Sigma_u_hat"], TRUE_SIGMA_U,
        "figures/residual-cholesky.png",
    )
    print("Saved figures/residual-cholesky.png")

    # Monte Carlo RMSE.
    print(f"\nRunning Monte Carlo ({MC_TRIALS} trials x {len(N_GRID)} sample sizes)...")
    n_grid_arr, mean_rmse, sd_rmse = monte_carlo_irf_rmse(
        N_GRID, TRUE_A1, TRUE_A2, TRUE_SIGMA_U, MC_TRIALS, HORIZON, SEED + 1
    )

    plot_irf_rmse_by_n(n_grid_arr, mean_rmse, sd_rmse, "figures/irf-rmse-by-n.png")
    print("Saved figures/irf-rmse-by-n.png")

    save_thumbnail("figures/irf-by-ordering.png", "figures/thumb.png")
    print("Saved figures/thumb.png")

    print("\nIRF RMSE by sample size:")
    print(f"  {'N':>6}  {'Mean RMSE':>10}  {'SD RMSE':>10}")
    for n, m, s in zip(N_GRID, mean_rmse, sd_rmse):
        print(f"  {n:>6}  {m:>10.4f}  {s:>10.4f}")


if __name__ == "__main__":
    main()
