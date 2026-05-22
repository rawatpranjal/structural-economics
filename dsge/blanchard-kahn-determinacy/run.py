#!/usr/bin/env python3
"""Blanchard-Kahn determinacy and saddle-path selection in linear RE models.

A toy three-equation New Keynesian model is the workhorse. The Taylor-rule
inflation response coefficient is swept across the determinacy boundary at
unity. At each value, Klein-style QZ counts stable generalised eigenvalues
against the one predetermined state and reports whether the Blanchard-Kahn
condition holds.

A two-parameter grid over the inflation and output Taylor coefficients
reproduces the classical Bullard-Mitra determinacy region. A sanity check
runs the same Klein solver on the small fixed-labor RBC linearisation
from the sibling RBC tutorial and verifies that the recovered capital
decision rule matches the hand-derived undetermined-coefficients solution.

Outputs:
    figures/eigenvalue-trajectories.png
    figures/phase-plane.png
    figures/bk-heatmap.png
    figures/thumb.png
    tables/eigenvalue-sweep.csv
    tables/rbc-sanity-check.csv
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.perturbation import KleinSolution, solve_klein
from lib.plotting import save_figure, save_thumbnail, setup_style


# =========================================================================
# Toy three-equation New Keynesian model in canonical Klein form
# =========================================================================


@dataclass
class NKPrimitives:
    """Calibration of the three-equation NK toy model."""

    sigma: float = 1.0
    beta: float = 0.99
    kappa: float = 0.30
    phi_pi: float = 1.5
    phi_y: float = 0.0
    rho_v: float = 0.5


def build_nk_matrices(p: NKPrimitives) -> tuple[np.ndarray, np.ndarray]:
    """Assemble A and B for A E_t s_{t+1} = B s_t with s_t = (v, y, pi).

    Equations in order:
        Row 0: Taylor wedge AR(1)     v_{t+1} = rho_v v_t + eps_{t+1}.
        Row 1: IS curve                y_t = E_t y_{t+1} - (1/sigma)(i_t - E_t pi_{t+1}).
        Row 2: NK Phillips curve      pi_t = beta E_t pi_{t+1} + kappa y_t.

    The Taylor rule i_t = phi_pi pi_t + phi_y y_t + v_t is substituted
    into the IS curve before forming A and B. Variable v is predetermined
    (top row); y and pi are jump variables.
    """
    sigma, beta, kappa = p.sigma, p.beta, p.kappa
    phi_pi, phi_y, rho_v = p.phi_pi, p.phi_y, p.rho_v

    # A multiplies E_t s_{t+1}.
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0 / sigma],
            [0.0, 0.0, beta],
        ]
    )
    # B multiplies s_t.
    B = np.array(
        [
            [rho_v, 0.0, 0.0],
            [1.0 / sigma, 1.0 + phi_y / sigma, phi_pi / sigma],
            [0.0, -kappa, 1.0],
        ]
    )
    return A, B


def classify_bk(sol: KleinSolution) -> str:
    """Map a KleinSolution to one of three classification labels.

    Standard Blanchard-Kahn (1980) convention:
    - n_stable == n_predetermined: unique bounded solution (determinate).
    - n_stable >  n_predetermined: a continuum of bounded solutions
      (indeterminacy; sunspot equilibria can exist).
    - n_stable <  n_predetermined: no bounded solution starting from an
      arbitrary predetermined initial condition (explosive).
    """
    if sol.blanchard_kahn_satisfied:
        return "determinate"
    if sol.n_stable > sol.n_predetermined:
        return "indeterminate"
    return "explosive"


# =========================================================================
# Parameter sweep across the Taylor-principle boundary
# =========================================================================


def safe_solve_klein(
    A: np.ndarray, B: np.ndarray, n_predetermined: int,
) -> tuple[KleinSolution | None, np.ndarray, str]:
    """Wrap solve_klein so that BK failures do not raise.

    Returns the solution when recovery succeeds, otherwise None and the
    sorted absolute generalised eigenvalues from a manual QZ pass. The
    eigenvalues are reported either way; the F and P matrices are only
    well defined when the leading predetermined block is invertible.
    """
    from scipy.linalg import ordqz
    try:
        sol = solve_klein(A, B, n_predetermined=n_predetermined)
        return sol, np.abs(sol.eigenvalues), sol.bk_message
    except np.linalg.LinAlgError:
        BB, AA, alpha, beta_qz, _, _ = ordqz(B, A, sort="iuc", output="complex")
        with np.errstate(divide="ignore", invalid="ignore"):
            eigvals = np.where(np.abs(beta_qz) > 0, alpha / beta_qz,
                               np.full_like(alpha, np.inf))
        n_stable = int(np.sum(np.abs(eigvals) < 1.0))
        if n_stable < n_predetermined:
            msg = (f"indeterminacy: only {n_stable} stable eigenvalues for"
                   f" {n_predetermined} predetermined vars (Z11 singular)")
        else:
            msg = (f"no solution: {n_stable} stable eigenvalues exceed"
                   f" {n_predetermined} predetermined vars (Z11 singular)")
        return None, np.abs(eigvals), msg


def sweep_phi_pi(
    phi_pi_grid: np.ndarray, base: NKPrimitives,
) -> pd.DataFrame:
    """Solve Klein at each phi_pi and record diagnostics."""
    records = []
    for phi_pi in phi_pi_grid:
        prim = NKPrimitives(
            sigma=base.sigma, beta=base.beta, kappa=base.kappa,
            phi_pi=float(phi_pi), phi_y=base.phi_y, rho_v=base.rho_v,
        )
        A, B = build_nk_matrices(prim)
        sol, abs_eigs, msg = safe_solve_klein(A, B, n_predetermined=1)
        eigs_abs = np.sort(abs_eigs)
        if sol is None:
            n_stable = int(np.sum(abs_eigs < 1.0))
            n_pre = 1
            bk_ok = False
            if n_stable > n_pre:
                label = "indeterminate"
            elif n_stable < n_pre:
                label = "explosive"
            else:
                label = "determinate"
        else:
            n_stable = sol.n_stable
            n_pre = sol.n_predetermined
            bk_ok = sol.blanchard_kahn_satisfied
            label = classify_bk(sol)
        records.append(
            {
                "phi_pi": float(phi_pi),
                "n_stable": n_stable,
                "n_predetermined": n_pre,
                "bk_satisfied": bk_ok,
                "classification": label,
                "abs_eig_1": float(eigs_abs[0]),
                "abs_eig_2": float(eigs_abs[1]),
                "abs_eig_3": float(eigs_abs[2]),
                "bk_message": msg,
            }
        )
    return pd.DataFrame(records)


def bk_heatmap_grid(
    phi_pi_grid: np.ndarray, phi_y_grid: np.ndarray, base: NKPrimitives,
) -> np.ndarray:
    """Classify every (phi_pi, phi_y) cell as determinate / indeterminate / explosive.

    Encoding: 0 = determinate, 1 = indeterminate, 2 = explosive.
    """
    grid = np.zeros((len(phi_y_grid), len(phi_pi_grid)), dtype=int)
    for i, phi_y in enumerate(phi_y_grid):
        for j, phi_pi in enumerate(phi_pi_grid):
            prim = NKPrimitives(
                sigma=base.sigma, beta=base.beta, kappa=base.kappa,
                phi_pi=float(phi_pi), phi_y=float(phi_y), rho_v=base.rho_v,
            )
            A, B = build_nk_matrices(prim)
            sol, abs_eigs, _ = safe_solve_klein(A, B, n_predetermined=1)
            if sol is not None:
                label = classify_bk(sol)
            else:
                n_stable = int(np.sum(abs_eigs < 1.0))
                if n_stable > 1:
                    label = "indeterminate"
                elif n_stable < 1:
                    label = "explosive"
                else:
                    label = "determinate"
            grid[i, j] = {"determinate": 0, "indeterminate": 1, "explosive": 2}[label]
    return grid


# =========================================================================
# Phase-plane sample paths
# =========================================================================


def simulate_determinate_path(
    F: np.ndarray, P: np.ndarray, v0: float, periods: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate the recovered Klein rules under a one-shot wedge shock."""
    v_path = np.zeros(periods)
    v_path[0] = v0
    for t in range(periods - 1):
        v_path[t + 1] = (F @ np.array([v_path[t]]))[0]
    y_path = P[0, 0] * v_path
    pi_path = P[1, 0] * v_path
    return v_path, y_path, pi_path


def simulate_indeterminate_paths(
    A: np.ndarray, B: np.ndarray, v0: float, periods: int,
    sunspot_amplitudes: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct multiple bounded solution paths under indeterminacy.

    Indeterminacy means the QZ decomposition delivers more stable
    generalised eigenvalues than predetermined states. Bounded
    rational-expectations solutions of the linear system s_{t+1} = M s_t
    are linear combinations of decaying modes:

        s_t = sum_k c_k v_k lambda_k^t,

    where v_k is the right eigenvector and lambda_k is the eigenvalue of
    each stable mode. The predetermined initial condition s_0[0] = v0
    imposes one scalar equation on the coefficients c_k. With two stable
    modes and one constraint there is a one-parameter family of bounded
    solutions; each amplitude in sunspot_amplitudes picks one.
    """
    paths = []
    M = np.linalg.solve(A, B)
    eigvals, eigvecs = np.linalg.eig(M)
    stable_idx = np.where(np.abs(eigvals) < 1.0)[0]
    if len(stable_idx) < 2:
        return paths

    # Re-order so the eigenvector with the largest absolute v-loading
    # comes first. It carries the predetermined initial condition; the
    # remaining stable mode is the sunspot direction.
    v_loadings = np.abs(eigvecs[0, stable_idx])
    order = np.argsort(-v_loadings)
    stable_idx = stable_idx[order]
    v_stable = eigvecs[:, stable_idx]
    lam_stable = eigvals[stable_idx]

    if abs(v_stable[0, 0]) < 1e-10:
        return paths

    for amp in sunspot_amplitudes:
        c2 = complex(amp)
        c1 = (v0 - v_stable[0, 1] * c2) / v_stable[0, 0]
        c_vec = np.array([c1, c2])
        path = np.zeros((periods, 3))
        for t in range(periods):
            s_t = (
                c_vec[0] * v_stable[:, 0] * lam_stable[0] ** t
                + c_vec[1] * v_stable[:, 1] * lam_stable[1] ** t
            )
            path[t] = np.real(s_t)
        paths.append((path[:, 1].copy(), path[:, 2].copy()))
    return paths


# =========================================================================
# Sanity check against the small fixed-labor RBC linearisation
# =========================================================================


def rbc_sanity_check() -> dict[str, float]:
    """Re-solve the fixed-labor RBC system from dsge/rbc/ with solve_klein.

    The matrices below match dsge/rbc/run.py exactly. The recovered
    capital decision rule must agree with the hand-derived (p, q) to
    machine precision.
    """
    alpha, beta_disc, delta = 0.33, 0.99, 0.025
    rho, sigma = 0.95, 1.0

    mpk = 1.0 / beta_disc - 1.0 + delta
    capital = (alpha / mpk) ** (1.0 / (1.0 - alpha))
    output = capital ** alpha
    investment = delta * capital
    consumption = output - investment
    capital_output = capital / output
    consumption_share = consumption / output
    gross_marginal_product_share = beta_disc * alpha / capital_output

    A = np.array(
        [
            [capital_output, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [
                -(alpha - 1.0) * gross_marginal_product_share,
                -gross_marginal_product_share,
                sigma,
            ],
        ]
    )
    B = np.array(
        [
            [alpha + (1.0 - delta) * capital_output, 1.0, -consumption_share],
            [0.0, rho, 0.0],
            [0.0, 0.0, sigma],
        ]
    )
    sol = solve_klein(A, B, n_predetermined=2)

    # Hand-derived undetermined-coefficients solve from dsge/rbc.
    consumption_capital_share = consumption_share
    resource_lag_weight = alpha + capital_output * (1.0 - delta)

    def cons_coefficients(p_val: float, q_val: float) -> tuple[float, float]:
        c_k = (resource_lag_weight - capital_output * p_val) / consumption_capital_share
        c_a = (1.0 - capital_output * q_val) / consumption_capital_share
        return c_k, c_a

    def residual(coef: np.ndarray) -> np.ndarray:
        p_val, q_val = coef
        c_k, c_a = cons_coefficients(p_val, q_val)
        euler_k = c_k - (
            c_k * p_val - (gross_marginal_product_share / sigma) * (alpha - 1.0) * p_val
        )
        euler_a = c_a - (
            c_k * q_val + c_a * rho
            - (gross_marginal_product_share / sigma) * (rho + (alpha - 1.0) * q_val)
        )
        return np.array([euler_k, euler_a])

    sol_root = root(residual, np.array([0.95, 0.08]))
    p_hand, q_hand = sol_root.x

    return {
        "p_qz": float(sol.F[0, 0]),
        "q_qz": float(sol.F[0, 1]),
        "p_hand": float(p_hand),
        "q_hand": float(q_hand),
        "abs_diff_p": float(abs(sol.F[0, 0] - p_hand)),
        "abs_diff_q": float(abs(sol.F[0, 1] - q_hand)),
        "bk_message": sol.bk_message,
        "n_stable": sol.n_stable,
        "n_predetermined": sol.n_predetermined,
    }


# =========================================================================
# Plots
# =========================================================================


def plot_eigenvalue_trajectories(
    sweep: pd.DataFrame, base: NKPrimitives,
) -> plt.Figure:
    """Absolute generalised eigenvalues vs the swept phi_pi."""
    fig, ax = plt.subplots(figsize=(9, 5))
    phi = sweep["phi_pi"].to_numpy()
    eigs = sweep[["abs_eig_1", "abs_eig_2", "abs_eig_3"]].to_numpy()

    # Colour rows by classification.
    color_map = {"determinate": "#1b6ca8", "indeterminate": "#b85c00",
                 "explosive": "#a3007a"}
    for label, color in color_map.items():
        mask = sweep["classification"].to_numpy() == label
        if mask.any():
            for col in range(3):
                ax.plot(phi[mask], eigs[mask, col], "o", color=color,
                        markersize=3.5, alpha=0.7,
                        label=label if col == 0 else None)
    # Overlay the smoothed trajectories so the eye can follow each branch.
    for col, sym in enumerate(["o", "s", "^"]):
        ax.plot(phi, eigs[:, col], color="black", linewidth=0.7, alpha=0.3)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--",
               label="unit circle |lambda| = 1")
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":",
               label="Taylor principle boundary phi_pi = 1")
    ax.set_xlabel("Taylor-rule inflation response phi_pi")
    ax.set_ylabel("|generalised eigenvalue|")
    ax.set_title(
        "Generalised eigenvalues across the Blanchard-Kahn boundary\n"
        f"(kappa={base.kappa}, beta={base.beta}, sigma={base.sigma},"
        f" rho_v={base.rho_v})"
    )
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_phase_plane(
    base: NKPrimitives, periods: int,
) -> plt.Figure:
    """Phase-plane (y_t, pi_t) under determinate vs indeterminate calibrations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=False, sharey=False)

    # Determinate: phi_pi = 1.5
    prim_det = NKPrimitives(
        sigma=base.sigma, beta=base.beta, kappa=base.kappa,
        phi_pi=1.5, phi_y=base.phi_y, rho_v=base.rho_v,
    )
    A_det, B_det = build_nk_matrices(prim_det)
    sol_det = solve_klein(A_det, B_det, n_predetermined=1)
    v_path, y_path, pi_path = simulate_determinate_path(
        sol_det.F, sol_det.P, v0=0.01, periods=periods,
    )
    axes[0].plot(y_path, pi_path, marker="o", color="#1b6ca8",
                 linewidth=1.6, markersize=4, label="saddle-path solution")
    axes[0].plot(y_path[0], pi_path[0], marker="*", color="#a3007a",
                 markersize=14, label="impact at t = 0")
    axes[0].plot(0.0, 0.0, marker="X", color="black", markersize=10,
                 label="steady state")
    axes[0].set_title(f"Determinate (phi_pi = {prim_det.phi_pi})\n"
                      f"{sol_det.bk_message}")
    axes[0].set_xlabel("Output gap y_t")
    axes[0].set_ylabel("Inflation pi_t")
    axes[0].axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    axes[0].axvline(0.0, color="black", linewidth=0.5, alpha=0.5)
    axes[0].legend(frameon=False, loc="best")

    # Indeterminate: phi_pi = 0.6
    prim_ind = NKPrimitives(
        sigma=base.sigma, beta=base.beta, kappa=base.kappa,
        phi_pi=0.6, phi_y=base.phi_y, rho_v=base.rho_v,
    )
    A_ind, B_ind = build_nk_matrices(prim_ind)
    sol_ind, abs_eigs_ind, _ = safe_solve_klein(A_ind, B_ind, n_predetermined=1)
    n_stable_ind = int(np.sum(abs_eigs_ind < 1.0))
    ind_msg = (f"{n_stable_ind} stable eigenvalues for 1 predetermined state"
               f" -> indeterminacy")
    sunspot_grid = np.array([-0.020, 0.0, 0.020])
    paths = simulate_indeterminate_paths(
        A_ind, B_ind, v0=0.01, periods=periods,
        sunspot_amplitudes=sunspot_grid,
    )
    palette = ["#b85c00", "#a3007a", "#4b8f29"]
    for (y_ind, pi_ind), color, amp in zip(paths, palette, sunspot_grid):
        if len(y_ind) > 1:
            axes[1].plot(y_ind, pi_ind, marker="o", color=color,
                         linewidth=1.4, markersize=3.0, alpha=0.85,
                         label=f"sunspot amplitude {amp:.3f}")
    axes[1].plot(0.0, 0.0, marker="X", color="black", markersize=10,
                 label="steady state")
    axes[1].set_title(f"Indeterminate (phi_pi = {prim_ind.phi_pi})\n"
                      f"{ind_msg}")
    axes[1].set_xlabel("Output gap y_t")
    axes[1].set_ylabel("Inflation pi_t")
    axes[1].axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    axes[1].axvline(0.0, color="black", linewidth=0.5, alpha=0.5)
    axes[1].legend(frameon=False, loc="best")

    fig.suptitle("Phase plane of (y_t, pi_t) after a unit Taylor-wedge shock",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def plot_bk_heatmap(
    grid: np.ndarray, phi_pi_grid: np.ndarray, phi_y_grid: np.ndarray,
    base: NKPrimitives,
) -> plt.Figure:
    """Heatmap of the BK classification over (phi_pi, phi_y)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap(["#1b6ca8", "#b85c00", "#a3007a"])
    extent = (phi_pi_grid[0], phi_pi_grid[-1],
              phi_y_grid[0], phi_y_grid[-1])
    im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    # Reference contour for the Taylor principle in NK theory.
    phi_pi_curve = np.linspace(phi_pi_grid[0], phi_pi_grid[-1], 400)
    # Bullard-Mitra (2002) determinacy boundary in the simple NK model.
    boundary = (1.0 - phi_pi_curve) * base.kappa / (1.0 - base.beta)
    ax.plot(phi_pi_curve, boundary, color="white", linewidth=1.8,
            linestyle="--", label="theoretical determinacy frontier")
    ax.set_xlabel("Taylor-rule inflation response phi_pi")
    ax.set_ylabel("Taylor-rule output response phi_y")
    ax.set_title(
        "Blanchard-Kahn classification on the (phi_pi, phi_y) grid\n"
        f"(kappa={base.kappa}, beta={base.beta}, sigma={base.sigma},"
        f" rho_v={base.rho_v})"
    )
    ax.set_xlim(phi_pi_grid[0], phi_pi_grid[-1])
    ax.set_ylim(phi_y_grid[0], phi_y_grid[-1])
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["determinate", "indeterminate", "explosive"])
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


# =========================================================================
# Main
# =========================================================================


def main() -> None:
    tutorial_dir = Path(__file__).resolve().parent
    os.chdir(tutorial_dir)

    setup_style()

    base = NKPrimitives()

    # ------------------------------------------------------------------
    # Parameter sweep across the Taylor-principle boundary.
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Toy NK model: Klein QZ as the Taylor-rule coefficient sweeps")
    print("=" * 72)
    phi_pi_grid = np.linspace(0.0, 2.5, 51)
    sweep = sweep_phi_pi(phi_pi_grid, base)
    sweep.to_csv("tables/eigenvalue-sweep.csv", index=False)

    n_det = int((sweep["classification"] == "determinate").sum())
    n_ind = int((sweep["classification"] == "indeterminate").sum())
    n_exp = int((sweep["classification"] == "explosive").sum())
    print(f"  Sweep size: {len(sweep)} grid points")
    print(f"  determinate: {n_det}, indeterminate: {n_ind}, explosive: {n_exp}")
    boundary_idx = int(np.argmin(np.abs(phi_pi_grid - 1.0)))
    print(f"  At phi_pi = {phi_pi_grid[boundary_idx]:.2f}: "
          f"{sweep['classification'].iloc[boundary_idx]} "
          f"({sweep['bk_message'].iloc[boundary_idx]})")

    fig_eig = plot_eigenvalue_trajectories(sweep, base)
    save_figure(fig_eig, "figures/eigenvalue-trajectories.png", dpi=150)

    # ------------------------------------------------------------------
    # Phase-plane figure: determinate vs indeterminate calibrations.
    # ------------------------------------------------------------------
    print()
    print("Phase-plane figure: determinate (phi_pi=1.5) vs"
          " indeterminate (phi_pi=0.6)")
    fig_phase = plot_phase_plane(base, periods=30)
    save_figure(fig_phase, "figures/phase-plane.png", dpi=150)

    # ------------------------------------------------------------------
    # Two-parameter BK classification heatmap.
    # ------------------------------------------------------------------
    print()
    print("Two-parameter BK heatmap over (phi_pi, phi_y)")
    phi_pi_heat = np.linspace(0.0, 2.5, 81)
    phi_y_heat = np.linspace(-0.5, 1.5, 81)
    grid = bk_heatmap_grid(phi_pi_heat, phi_y_heat, base)
    fig_heat = plot_bk_heatmap(grid, phi_pi_heat, phi_y_heat, base)
    save_figure(fig_heat, "figures/bk-heatmap.png", dpi=150)
    print(f"  Heatmap shape: {grid.shape}")
    n_cells = grid.size
    print(f"  determinate cells: {int((grid == 0).sum())} / {n_cells}")
    print(f"  indeterminate cells: {int((grid == 1).sum())} / {n_cells}")
    print(f"  explosive cells: {int((grid == 2).sum())} / {n_cells}")

    # Thumbnail from the heatmap.
    save_thumbnail("figures/bk-heatmap.png", "figures/thumb.png")

    # ------------------------------------------------------------------
    # Sanity check: re-solve the small fixed-labor RBC with solve_klein.
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Sanity check on the small fixed-labor RBC linearisation")
    print("=" * 72)
    rbc = rbc_sanity_check()
    pd.DataFrame([rbc]).to_csv("tables/rbc-sanity-check.csv", index=False)
    print(f"  BK status: {rbc['bk_message']}")
    print(f"  n_stable = {rbc['n_stable']}, "
          f"n_predetermined = {rbc['n_predetermined']}")
    print(f"  Klein QZ: p = {rbc['p_qz']:.6f}, q = {rbc['q_qz']:.6f}")
    print(f"  Hand:     p = {rbc['p_hand']:.6f}, q = {rbc['q_hand']:.6f}")
    print(f"  Max abs diff: p = {rbc['abs_diff_p']:.2e}, "
          f"q = {rbc['abs_diff_q']:.2e}")

    print()
    print("Saved: 3 figures + 2 tables")


if __name__ == "__main__":
    main()
