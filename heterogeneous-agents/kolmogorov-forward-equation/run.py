#!/usr/bin/env python3
"""Stationary Kolmogorov forward equation by transposing the upwind generator.

This prelim demonstrates the operator duality A vs A^T on three examples.
The upwind generator A is the same matrix the HJB prelim assembles for the
backward equation. Its transpose A^T pushes the cross-sectional density
forward in time. The stationary density solves A^T g = 0 with a
normalisation row that pins the scale.

Example 1: Ornstein-Uhlenbeck on a bounded interval.
    Build the drift block by upwinding dot x = -kappa (x - mu) and add a
    centered second-difference diffusion block sigma^2 / 2 D^2. Solve
    A^T g = 0 by sparse LU with one row replaced by the normalisation,
    then compare against the analytic Gaussian stationary density of the
    OU process with the same kappa, mu, sigma.

Example 2: Reuse of the Ramsey toy A from the upwind-finite-differences
    prelim. The same primitives are re-solved here to avoid importing a
    half-script; the stationary density of A^T concentrates near the
    deterministic steady state.

Example 3: Two-state Poisson income chain combined with a 1D asset drift
    via Kronecker block assembly. The joint generator is block-tridiagonal
    on the asset axis and tridiagonal on the income axis. Stationary
    A_joint^T g = 0 then recovers a (asset, income) density.

Figures saved:
    figures/stationary-density-ou.png  - OU density vs analytic Gaussian.
    figures/sparse-A-pattern.png       - spy plot of A and A^T side by side.
    figures/operator-duality.png       - annotated A (HJB) vs A^T (KFE)
                                         schematic for a small grid.
    figures/joint-density.png          - joint asset-income stationary
                                         density from the block generator.
    figures/thumb.png                  - thumbnail from the OU density figure.

Run from the folder root:
    cd heterogeneous-agents/kolmogorov-forward-equation && python run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.finite_differences import (  # noqa: E402
    backward_difference,
    forward_difference,
    stationary_distribution,
    upwind_generator,
)
from lib.plotting import save_figure, save_thumbnail, setup_style  # noqa: E402


np.random.seed(0)


# ---------------------------------------------------------------------------
# Example 1: Ornstein-Uhlenbeck on a bounded interval
# ---------------------------------------------------------------------------


def ou_drift(x: np.ndarray, *, kappa: float, mu: float) -> np.ndarray:
    """Mean-reverting drift dot x = -kappa (x - mu)."""
    return -kappa * (x - mu)


def diffusion_block(n: int, dx: float, sigma: float) -> sparse.csc_matrix:
    """Centered second-difference operator scaled by sigma^2 / 2.

    Acts on interior nodes only; rows 0 and n-1 are left as zero so the
    drift block governs the boundary behaviour. The resulting matrix is a
    sub-generator on the endpoints and a proper Markov generator on the
    interior: zero row sums, non-positive diagonal, non-negative
    off-diagonals.
    """
    coeff = sigma * sigma / (2.0 * dx * dx)
    main = -2.0 * coeff * np.ones(n)
    side = coeff * np.ones(n - 1)
    # Suppress diffusion at the endpoints: zero out boundary rows.
    main[0] = 0.0
    main[n - 1] = 0.0
    side[0] = 0.0       # row 1 sub-diagonal -> position (1, 0)? No: side is super.
    # The sparse.diags layout below: super-diag k=+1 has length n-1, indexed by
    # source row 0..n-2; sub-diag k=-1 has length n-1, indexed by source row 1..n-1.
    # We want rows 0 and n-1 of the operator to be entirely zero. Adjust both.
    super_diag = coeff * np.ones(n - 1)
    sub_diag = coeff * np.ones(n - 1)
    super_diag[0] = 0.0          # row 0 super-diagonal entry
    sub_diag[n - 2] = 0.0        # row n-1 sub-diagonal entry
    D = (
        sparse.diags(main, 0, shape=(n, n))
        + sparse.diags(super_diag, 1, shape=(n, n))
        + sparse.diags(sub_diag, -1, shape=(n, n))
    )
    return D.tocsc()


def ou_stationary_analytic(x: np.ndarray, *, mu: float, kappa: float, sigma: float) -> np.ndarray:
    """Gaussian stationary density of the OU process on the real line."""
    var = sigma * sigma / (2.0 * kappa)
    z = (x - mu) / np.sqrt(var)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi * var)
    return pdf


def solve_ou_stationary(
    x_min: float,
    x_max: float,
    n: int,
    *,
    kappa: float,
    mu: float,
    sigma: float,
):
    """Build A = drift + diffusion and solve A^T g = 0 with normalisation."""
    x = np.linspace(x_min, x_max, n)
    dx = float(x[1] - x[0])
    drift = ou_drift(x, kappa=kappa, mu=mu)
    A_drift = upwind_generator(x, drift, boundary="reflect")
    A_diff = diffusion_block(n, dx, sigma)
    A = (A_drift + A_diff).tocsc()
    g = stationary_distribution(A, dx=dx, fix_row=0)
    return x, A, g


# ---------------------------------------------------------------------------
# Example 2: stationary density of the upwind-finite-differences toy A
# ---------------------------------------------------------------------------


RHO_R = 0.05
ALPHA_R = 0.36
DELTA_R = 0.05
K_MIN_R = 0.5
K_MAX_R = 15.0


def ramsey_net_output(k: np.ndarray) -> np.ndarray:
    return k ** ALPHA_R - DELTA_R * k


def ramsey_log_utility(c: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(c, 1e-12))


def solve_ramsey_hjb(k: np.ndarray, *, delta_step: float = 1000.0,
                     max_iter: int = 500, tol: float = 1e-7):
    """Same toy HJB as the upwind prelim, re-solved here to recover A.

    Returning the converged sparse upwind generator A is all the KFE
    prelim needs from the upstream HJB solve.
    """
    n = len(k)
    dk = float(k[1] - k[0])
    fk = ramsey_net_output(k)
    v = ramsey_log_utility(np.maximum(fk, 1e-6)) / RHO_R

    for it in range(1, max_iter + 1):
        dvf = forward_difference(v, dk)
        dvb = backward_difference(v, dk)
        cf = 1.0 / np.maximum(dvf, 1e-12)
        cb = 1.0 / np.maximum(dvb, 1e-12)
        sf = fk - cf
        sb = fk - cb
        c0 = fk
        use_fwd = sf > 0.0
        use_bwd = sb < 0.0
        use_zero = ~(use_fwd | use_bwd)
        use_fwd[0] = True
        use_bwd[0] = False
        use_zero[0] = False
        use_fwd[n - 1] = False
        use_bwd[n - 1] = True
        use_zero[n - 1] = False
        c = np.where(use_fwd, cf, np.where(use_bwd, cb, c0))
        s = fk - c
        A = upwind_generator(k, s, boundary="reflect")

        rhs = ramsey_log_utility(c) + v / delta_step
        B = (1.0 / delta_step + RHO_R) * sparse.eye(n, format="csc") - A
        v_new = spsolve(B, rhs)
        change = float(np.max(np.abs(v_new - v)))
        v = v_new
        if change < tol:
            break

    # Final policy and generator.
    dvf = forward_difference(v, dk)
    dvb = backward_difference(v, dk)
    cf = 1.0 / np.maximum(dvf, 1e-12)
    cb = 1.0 / np.maximum(dvb, 1e-12)
    sf = fk - cf
    sb = fk - cb
    c0 = fk
    use_fwd = sf > 0.0
    use_bwd = sb < 0.0
    use_zero = ~(use_fwd | use_bwd)
    use_fwd[0] = True
    use_bwd[0] = False
    use_zero[0] = False
    use_fwd[n - 1] = False
    use_bwd[n - 1] = True
    use_zero[n - 1] = False
    c = np.where(use_fwd, cf, np.where(use_bwd, cb, c0))
    s = fk - c
    A = upwind_generator(k, s, boundary="reflect")
    return v, c, s, A, {"iterations": it, "change": change}


def ramsey_stationary(k: np.ndarray):
    """Solve the toy Ramsey HJB and recover the stationary density of A^T.

    The Ramsey HJB is deterministic, so A is the generator of a purely
    convective transport. The unique invariant measure of a deterministic
    system with a globally stable interior fixed point is a point mass at
    that fixed point. On the discrete grid the stationary solve recovers
    a single-bin spike at the node closest to the steady state.
    """
    v, c, s, A, info = solve_ramsey_hjb(k)
    dk = float(k[1] - k[0])
    # Pick the row to fix as the one where the drift is closest to zero;
    # otherwise the trivial deterministic system can return a delta at the
    # boundary depending on numerical rounding.
    idx_ss = int(np.argmin(np.abs(s)))
    g = stationary_distribution(A, dx=dk, fix_row=idx_ss)
    return v, c, s, A, g, info, idx_ss


# ---------------------------------------------------------------------------
# Example 3: joint asset and two-state Poisson income chain
# ---------------------------------------------------------------------------


def two_state_income_generator(lam_lh: float, lam_hl: float) -> np.ndarray:
    """Continuous-time Markov generator for a two-state income chain.

    Off-diagonal entries are Poisson jump rates and rows sum to zero.
    """
    Q = np.array([
        [-lam_lh,  lam_lh],
        [ lam_hl, -lam_hl],
    ])
    return Q


def joint_generator_asset_income(
    A_low: sparse.spmatrix,
    A_high: sparse.spmatrix,
    Q: np.ndarray,
) -> sparse.csc_matrix:
    """Block-diagonal asset generators plus a Kronecker income switch block.

    Asset blocks A_low and A_high act in their own income state. The
    income switch couples the two blocks via Q kron I_n.
    """
    n = A_low.shape[0]
    block_asset = sparse.bmat([[A_low, None], [None, A_high]], format="csc")
    I_n = sparse.eye(n, format="csc")
    block_income = sparse.bmat(
        [[Q[0, 0] * I_n, Q[0, 1] * I_n],
         [Q[1, 0] * I_n, Q[1, 1] * I_n]],
        format="csc",
    )
    return (block_asset + block_income).tocsc()


def solve_joint_stationary(
    x_min: float = -0.5,
    x_max: float = 3.0,
    n: int = 200,
    *,
    a_target_low: float = 0.3,
    a_target_high: float = 1.5,
    kappa_a: float = 0.6,
    lam_lh: float = 1.2,
    lam_hl: float = 1.2,
    sigma_a: float = 0.15,
):
    """Joint stationary solve over assets and a two-state income chain.

    Each income state has its own mean-reverting saving target on the
    asset axis. Low-income households drift toward a smaller buffer,
    high-income households toward a larger one. Adding a small
    diffusion block on the asset axis keeps the joint density smooth
    without coding a full HJB. The point of the example is the
    block-matrix assembly, not the household policy.
    """
    a = np.linspace(x_min, x_max, n)
    da = float(a[1] - a[0])
    s_low = -kappa_a * (a - a_target_low)
    s_high = -kappa_a * (a - a_target_high)
    A_low_drift = upwind_generator(a, s_low, boundary="reflect")
    A_high_drift = upwind_generator(a, s_high, boundary="reflect")
    D = diffusion_block(n, da, sigma_a)
    A_low = (A_low_drift + D).tocsc()
    A_high = (A_high_drift + D).tocsc()
    Q = two_state_income_generator(lam_lh, lam_hl)
    A_joint = joint_generator_asset_income(A_low, A_high, Q)
    g_flat = stationary_distribution(A_joint, dx=da, fix_row=0)
    g_low = g_flat[:n]
    g_high = g_flat[n:]
    return a, A_joint, g_low, g_high


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_style()

    # ------------------------------------------------------------------
    # Example 1: OU stationary density
    # ------------------------------------------------------------------
    kappa = 1.0
    mu = 0.0
    sigma = 0.5
    x_lo, x_hi = -3.0, 3.0
    n_ou = 401
    x, A_ou, g_ou = solve_ou_stationary(x_lo, x_hi, n_ou, kappa=kappa, mu=mu, sigma=sigma)
    dx = float(x[1] - x[0])
    g_analytic = ou_stationary_analytic(x, mu=mu, kappa=kappa, sigma=sigma)
    # Renormalise the analytic density to the bounded interval so the
    # comparison is on the same support.
    g_analytic = g_analytic / (g_analytic.sum() * dx)
    sup_err = float(np.max(np.abs(g_ou - g_analytic)))
    print(f"OU: grid {n_ou} points on [{x_lo}, {x_hi}], dx = {dx:.4f}")
    print(f"OU: sup-norm gap to analytic Gaussian = {sup_err:.3e}")
    mean_num = float((x * g_ou).sum() * dx)
    var_num = float(((x - mean_num) ** 2 * g_ou).sum() * dx)
    print(f"OU: numerical mean = {mean_num:+.4f}, variance = {var_num:.4f}")
    print(f"OU: analytic mean = {mu:+.4f}, variance = {sigma * sigma / (2 * kappa):.4f}")

    # ------------------------------------------------------------------
    # Example 2: Ramsey toy A stationary density
    # ------------------------------------------------------------------
    n_r = 200
    k = np.linspace(K_MIN_R, K_MAX_R, n_r)
    v_r, c_r, s_r, A_r, g_r, info_r, idx_ss = ramsey_stationary(k)
    k_ss_analytic = (ALPHA_R / (RHO_R + DELTA_R)) ** (1.0 / (1.0 - ALPHA_R))
    k_ss_numeric = float(k[idx_ss])
    print(
        f"Ramsey toy: HJB converged in {info_r['iterations']} iterations,"
        f" sup-norm change {info_r['change']:.2e}"
    )
    print(
        f"Ramsey toy: analytic steady state k = {k_ss_analytic:.4f},"
        f" numeric (fixed normalisation row) k = {k_ss_numeric:.4f}"
    )

    # ------------------------------------------------------------------
    # Example 3: joint asset + two-state Poisson income
    # ------------------------------------------------------------------
    a_joint, A_joint, g_low, g_high = solve_joint_stationary(n=200)
    da_joint = float(a_joint[1] - a_joint[0])
    p_low = float(g_low.sum() * da_joint)
    p_high = float(g_high.sum() * da_joint)
    print(f"Joint: marginal income masses p_low = {p_low:.4f}, p_high = {p_high:.4f}")
    print(f"Joint: joint generator A_joint shape = {A_joint.shape}, nnz = {A_joint.nnz}")

    # ------------------------------------------------------------------
    # Figure (a): OU stationary density vs analytic Gaussian
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, g_ou, color="#1f77b4", linewidth=2.2,
            label="stationary g from A^T g = 0")
    ax.plot(x, g_analytic, color="#d62728", linewidth=1.6, linestyle="--",
            label="analytic Gaussian (renormalised)")
    ax.axvline(mu, color="k", linestyle=":", linewidth=0.9, alpha=0.6,
               label=f"mean mu = {mu:.1f}")
    ax.set_xlabel("State x")
    ax.set_ylabel("Stationary density g(x)")
    ax.set_title(
        "Ornstein-Uhlenbeck stationary density: A^T g = 0 vs analytic"
    )
    ax.legend(loc="upper right")
    ax.text(
        0.02, 0.98,
        f"kappa = {kappa:.1f}, sigma = {sigma:.2f}\n"
        f"sup-norm gap = {sup_err:.2e}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, color="#444444",
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    save_figure(fig, "figures/stationary-density-ou.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure (b): sparse pattern of A and A^T side by side
    # ------------------------------------------------------------------
    fig2, (ax_a, ax_at) = plt.subplots(1, 2, figsize=(11, 5))
    ax_a.spy(A_ou, markersize=1.2, color="#1f77b4")
    ax_a.set_title("A (HJB direction)")
    ax_a.set_xlabel("Column index")
    ax_a.set_ylabel("Row index")
    ax_at.spy(A_ou.T, markersize=1.2, color="#d62728")
    ax_at.set_title("A^T (KFE direction)")
    ax_at.set_xlabel("Column index")
    save_figure(fig2, "figures/sparse-A-pattern.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure (b'): Ramsey toy stationary density from prelim #1's A
    # ------------------------------------------------------------------
    fig_r, (ax_g, ax_drift) = plt.subplots(2, 1, figsize=(7.5, 7),
                                            sharex=True)
    ax_g.plot(k, g_r, color="#1f77b4", linewidth=2.2,
              label="stationary g(k) from A_r^T g = 0")
    ax_g.axvline(k_ss_analytic, color="k", linestyle=":", linewidth=0.9,
                 alpha=0.7,
                 label=f"steady state k = {k_ss_analytic:.2f}")
    ax_g.set_ylabel("Stationary density g(k)")
    ax_g.set_title(
        "Ramsey toy: stationary distribution from the upwind HJB generator"
    )
    ax_g.legend(loc="upper right")
    ax_drift.plot(k, s_r, color="#444444", linewidth=1.5,
                  label="drift s(k) = f(k) - delta*k - c(k)")
    ax_drift.axhline(0.0, color="k", linestyle="--", linewidth=0.7,
                     alpha=0.5)
    ax_drift.axvline(k_ss_analytic, color="k", linestyle=":",
                     linewidth=0.9, alpha=0.7)
    ax_drift.set_xlabel("Capital k")
    ax_drift.set_ylabel("Drift s(k)")
    ax_drift.legend(loc="upper right")
    save_figure(fig_r, "figures/ramsey-stationary.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure (c): operator duality schematic on a small 6-node grid
    # ------------------------------------------------------------------
    n_small = 6
    x_small = np.linspace(-1.0, 1.0, n_small)
    drift_small = ou_drift(x_small, kappa=1.0, mu=0.0)
    A_small = upwind_generator(x_small, drift_small, boundary="reflect").toarray()
    AT_small = A_small.T

    fig3, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 5))
    vmax = float(max(np.abs(A_small).max(), np.abs(AT_small).max()))
    im_l = ax_l.imshow(A_small, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_l.set_title("A: HJB direction (rho I - A) v_{n+1} = u + v_n / Delta")
    ax_l.set_xlabel("Destination node j")
    ax_l.set_ylabel("Source node i")
    for i in range(n_small):
        for j in range(n_small):
            val = A_small[i, j]
            if abs(val) > 1e-12:
                ax_l.text(j, i, f"{val:+.2f}", ha="center", va="center",
                          fontsize=8, color="black")
    fig3.colorbar(im_l, ax=ax_l, fraction=0.046, pad=0.04)

    im_r = ax_r.imshow(AT_small, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_r.set_title("A^T: KFE direction A^T g = 0")
    ax_r.set_xlabel("Destination node j")
    ax_r.set_ylabel("Source node i")
    for i in range(n_small):
        for j in range(n_small):
            val = AT_small[i, j]
            if abs(val) > 1e-12:
                ax_r.text(j, i, f"{val:+.2f}", ha="center", va="center",
                          fontsize=8, color="black")
    fig3.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
    fig3.suptitle(
        "Operator duality: same entries, transposed indices",
        fontsize=12,
    )
    save_figure(fig3, "figures/operator-duality.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure (d): joint asset-income stationary density
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(a_joint, g_low, color="#1f77b4", linewidth=2.0,
             label="low income state")
    ax4.plot(a_joint, g_high, color="#d62728", linewidth=2.0,
             label="high income state")
    ax4.fill_between(a_joint, g_low, alpha=0.18, color="#1f77b4")
    ax4.fill_between(a_joint, g_high, alpha=0.18, color="#d62728")
    ax4.set_xlabel("Assets a")
    ax4.set_ylabel("Joint stationary density g_j(a)")
    ax4.set_title(
        "Joint stationary density from A_joint^T g = 0"
        " (Kronecker block assembly with Q)"
    )
    ax4.legend(loc="upper right")
    ax4.text(
        0.02, 0.98,
        f"income masses: p_low = {p_low:.3f},  p_high = {p_high:.3f}\n"
        f"A_joint shape = {A_joint.shape[0]} x {A_joint.shape[1]}",
        transform=ax4.transAxes, va="top", ha="left",
        fontsize=9, color="#444444",
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    save_figure(fig4, "figures/joint-density.png", dpi=150)

    # ------------------------------------------------------------------
    # Thumbnail
    # ------------------------------------------------------------------
    save_thumbnail("figures/stationary-density-ou.png", "figures/thumb.png")
    print(
        "Saved figures: stationary-density-ou, sparse-A-pattern,"
        " ramsey-stationary, operator-duality, joint-density, thumb."
    )


if __name__ == "__main__":
    main()
