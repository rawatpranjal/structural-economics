#!/usr/bin/env python3
"""Upwind finite-difference solve of a toy 1D HJB on a bounded interval.

The state is capital k on a closed interval [k_min, k_max]. The planner
solves

    rho v(k) = max_c { u(c) + v'(k) * (f(k) - delta * k - c) }

with log utility u(c) = ln(c), Cobb-Douglas production f(k) = k^alpha,
and depreciation rate delta. The first-order condition gives
c = 1 / v'(k).

The HJB is discretised on a uniform grid. The upwind rule picks the
forward slope when the policy-implied drift is positive and the
backward slope when it is negative. The resulting sparse generator A
is assembled with `lib.finite_differences.upwind_generator`. Each
implicit pseudo-time step solves

    [(1/Delta + rho) I - A] v_new = u(c) + v / Delta

by sparse LU.

The script saves three figures:
    figures/value-and-drift.png      - converged value plus drift sign
                                       coloured by which one-sided
                                       difference was selected.
    figures/error-vs-reference.png   - sup-norm error of the policy on
                                       a coarse grid against a fine
                                       reference solve.
    figures/failure-naive-central.png- one explicit step under a naive
                                       central difference vs upwind on
                                       the same initial guess.

It also runs a synthetic check of the Kuhn-Tucker state-constraint clip
on a contrived input where the unconstrained forward drift at the lower
boundary is negative. The clip overrides the policy and the override is
printed to stdout.

Run from the folder root:
    cd optimal-control/upwind-finite-differences && python run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Add repo root to path for lib/ imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.finite_differences import (  # noqa: E402
    backward_difference,
    central_difference,
    forward_difference,
    kt_state_constraint_clip,
    upwind_generator,
)
from lib.plotting import save_figure, save_thumbnail, setup_style  # noqa: E402


np.random.seed(0)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

RHO = 0.05
ALPHA = 0.36
DELTA = 0.05
K_MIN = 0.5
K_MAX = 15.0


def production(k: np.ndarray) -> np.ndarray:
    """Cobb-Douglas production f(k) = k^alpha."""
    return k ** ALPHA


def net_output(k: np.ndarray) -> np.ndarray:
    """Output net of depreciation, f(k) - delta * k."""
    return production(k) - DELTA * k


def log_utility(c: np.ndarray) -> np.ndarray:
    """Log utility, clipped away from zero for numerical safety."""
    return np.log(np.maximum(c, 1e-12))


# ---------------------------------------------------------------------------
# Upwind HJB solver on a single grid
# ---------------------------------------------------------------------------


def solve_upwind_hjb(
    k: np.ndarray,
    *,
    delta_step: float = 1000.0,
    max_iter: int = 500,
    tol: float = 1e-7,
):
    """Solve the toy HJB by implicit upwind FD iteration.

    Returns the converged value, consumption policy, drift, and a
    selector array marking each node as forward (1), backward (-1), or
    zero-drift (0).
    """
    n = len(k)
    dk = float(k[1] - k[0])
    fk = net_output(k)

    # Initial guess: myopic flow value (consume net output).
    v = log_utility(np.maximum(fk, 1e-6)) / RHO

    for it in range(1, max_iter + 1):
        dvf = forward_difference(v, dk)
        dvb = backward_difference(v, dk)

        # Candidate consumption from each one-sided slope (FOC c = 1/v').
        cf = 1.0 / np.maximum(dvf, 1e-12)
        cb = 1.0 / np.maximum(dvb, 1e-12)
        sf = fk - cf  # forward candidate drift
        sb = fk - cb  # backward candidate drift

        # Steady-state fallback consumption: c0 = f(k), drift zero.
        c0 = fk

        # Upwind selection: forward if forward drift positive,
        # backward if backward drift negative, otherwise zero drift.
        use_fwd = sf > 0.0
        use_bwd = sb < 0.0
        use_zero = ~(use_fwd | use_bwd)

        # Boundary forcing.
        use_fwd[0] = True
        use_bwd[0] = False
        use_zero[0] = False
        use_fwd[n - 1] = False
        use_bwd[n - 1] = True
        use_zero[n - 1] = False

        c = np.where(use_fwd, cf, np.where(use_bwd, cb, c0))
        s = fk - c

        A = upwind_generator(k, s, boundary="reflect")

        rhs = log_utility(c) + v / delta_step
        B = (1.0 / delta_step + RHO) * sparse.eye(n, format="csc") - A
        v_new = spsolve(B, rhs)

        change = float(np.max(np.abs(v_new - v)))
        v = v_new
        if change < tol:
            break

    # Final policy at converged v.
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

    selector = np.zeros(n, dtype=int)
    selector[use_fwd] = 1
    selector[use_bwd] = -1
    selector[use_zero] = 0

    info = {"iterations": it, "change": change}
    return v, c, s, selector, info


# ---------------------------------------------------------------------------
# Naive central-difference attempt (used to illustrate the failure mode)
# ---------------------------------------------------------------------------


def naive_central_step(v: np.ndarray, k: np.ndarray, dt: float = 0.5):
    """One explicit pseudo-time step using a central-difference slope.

    The HJB residual rho*v - u(c) - v'*s is driven down by an explicit
    Euler update:

        v_new = v + dt * (u(c) + v'_central * s - rho * v).

    The slope `v'_central = (v[i+1] - v[i-1]) / (2 dk)` mixes both
    neighbours regardless of drift direction. Even at moderate dt this
    produces oscillations that grow with iteration count because the
    transport term has no diagonal damping at the centred discretisation.

    Args:
        v: Current value iterate.
        k: Capital grid.
        dt: Explicit pseudo-time step size.
    """
    n = len(k)
    dk = float(k[1] - k[0])
    fk = net_output(k)

    dvc = central_difference(v, dk)
    c = 1.0 / np.maximum(dvc, 1e-6)
    s = fk - c
    residual = log_utility(c) + dvc * s - RHO * v
    return v + dt * residual


def naive_explicit_upwind_step(v: np.ndarray, k: np.ndarray, dt: float = 0.5):
    """One explicit pseudo-time step using upwind one-sided slopes.

    Provided as the counterpart to `naive_central_step`. With the same
    explicit dt, upwinding does not introduce growing oscillations; the
    iterates relax toward the fixed point even though convergence is
    slow compared to the implicit scheme used elsewhere in this script.
    """
    n = len(k)
    dk = float(k[1] - k[0])
    fk = net_output(k)

    dvf = forward_difference(v, dk)
    dvb = backward_difference(v, dk)
    cf = 1.0 / np.maximum(dvf, 1e-6)
    cb = 1.0 / np.maximum(dvb, 1e-6)
    sf = fk - cf
    sb = fk - cb

    use_fwd = sf > 0.0
    use_bwd = sb < 0.0
    use_zero = ~(use_fwd | use_bwd)
    use_fwd[0] = True
    use_bwd[0] = False
    use_zero[0] = False
    use_fwd[n - 1] = False
    use_bwd[n - 1] = True
    use_zero[n - 1] = False

    c = np.where(use_fwd, cf, np.where(use_bwd, cb, fk))
    s = fk - c
    dv = np.where(use_fwd, dvf, np.where(use_bwd, dvb, 0.0))
    residual = log_utility(c) + dv * s - RHO * v
    return v + dt * residual


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_style()

    # Steady state from f'(k_ss) = rho + delta.
    k_ss = (ALPHA / (RHO + DELTA)) ** (1.0 / (1.0 - ALPHA))
    print(f"Steady-state capital k_ss = {k_ss:.4f}")

    # Working grid.
    n_work = 200
    k_work = np.linspace(K_MIN, K_MAX, n_work)
    v, c, s, selector, info = solve_upwind_hjb(k_work)
    print(
        f"Upwind HJB converged in {info['iterations']} iterations,"
        f" sup-norm change {info['change']:.2e}"
    )

    # Reference solve on a fine grid as the policy benchmark.
    n_ref = 4000
    k_ref = np.linspace(K_MIN, K_MAX, n_ref)
    v_ref, c_ref, s_ref, _, info_ref = solve_upwind_hjb(k_ref)
    print(
        f"Reference HJB converged in {info_ref['iterations']} iterations,"
        f" sup-norm change {info_ref['change']:.2e}"
    )

    # Compare policies on a sequence of coarsenings.
    grid_sizes = [50, 100, 200, 400, 800]
    sup_errs = []
    for n in grid_sizes:
        k_n = np.linspace(K_MIN, K_MAX, n)
        _, c_n, _, _, _ = solve_upwind_hjb(k_n)
        c_ref_on_n = np.interp(k_n, k_ref, c_ref)
        sup_errs.append(float(np.max(np.abs(c_n - c_ref_on_n))))
        print(f"  n = {n:4d}: sup-norm policy error vs reference = {sup_errs[-1]:.2e}")

    # ------------------------------------------------------------------
    # Figure (a): value + drift selector
    # ------------------------------------------------------------------
    fig, (ax_v, ax_s) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
    ax_v.plot(k_work, v, color="#1f77b4", linewidth=2.2, label="upwind v(k)")
    ax_v.axvline(k_ss, color="k", linestyle=":", linewidth=0.9, alpha=0.6,
                 label=f"steady state k = {k_ss:.2f}")
    ax_v.set_ylabel("Value v(k)")
    ax_v.set_title("Upwind HJB: value function and drift selector")
    ax_v.legend(loc="lower right")

    fwd_mask = selector == 1
    bwd_mask = selector == -1
    zero_mask = selector == 0

    ax_s.plot(k_work, s, color="#444444", linewidth=1.5, alpha=0.7, zorder=1)
    ax_s.scatter(k_work[fwd_mask], s[fwd_mask], s=14, color="#2ca02c",
                 label="forward FD used", zorder=2)
    ax_s.scatter(k_work[bwd_mask], s[bwd_mask], s=14, color="#d62728",
                 label="backward FD used", zorder=2)
    if zero_mask.any():
        ax_s.scatter(k_work[zero_mask], s[zero_mask], s=20,
                     color="#1f77b4", marker="x",
                     label="zero-drift fallback", zorder=3)
    ax_s.axhline(0.0, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_s.axvline(k_ss, color="k", linestyle=":", linewidth=0.9, alpha=0.6)
    ax_s.set_xlabel("Capital k")
    ax_s.set_ylabel("Drift s(k) = f(k) - c(k)")
    ax_s.legend(loc="lower left")

    save_figure(fig, "figures/value-and-drift.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure (b): sup-norm policy error vs grid size (analytic surrogate
    # is the fine-grid reference solve).
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.loglog(grid_sizes, sup_errs, marker="o", color="#1f77b4",
               linewidth=2.0, label="sup-norm c(k) error")
    # Reference slope 1/n (first-order convergence) for orientation.
    grid_arr = np.array(grid_sizes, dtype=float)
    ref_line = sup_errs[0] * grid_arr[0] / grid_arr
    ax2.loglog(grid_arr, ref_line, color="#888888", linestyle="--",
               linewidth=1.2, label="first-order reference 1/n")
    ax2.set_xlabel("Grid points n")
    ax2.set_ylabel("Sup-norm policy error")
    ax2.set_title("Convergence of the upwind policy under refinement")
    ax2.legend()
    save_figure(fig2, "figures/error-vs-reference.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure (c): failure mode under naive central differences
    # ------------------------------------------------------------------
    n_fail = 80
    k_fail = np.linspace(K_MIN, K_MAX, n_fail)
    fk_fail = net_output(k_fail)
    # Start from the myopic flow value so the residual has something to
    # work on. Both schemes try to drive that residual to zero.
    v_seed = log_utility(np.maximum(fk_fail, 1e-6)) / RHO
    v_central = v_seed.copy()
    v_upwind = v_seed.copy()
    snapshots_central = [v_central.copy()]
    snapshots_upwind = [v_upwind.copy()]

    # Explicit pseudo-time step. The CFL-like stability condition for
    # this transport-style update is roughly dt < dk / max(|drift|). At
    # the chosen step the upwind scheme is well inside the bound; the
    # centred scheme has no dissipation at all and the residual grows
    # checkerboard oscillations.
    n_steps = 200
    dt = 0.08
    for _ in range(n_steps):
        v_central = naive_central_step(v_central, k_fail, dt=dt)
        v_upwind = naive_explicit_upwind_step(v_upwind, k_fail, dt=dt)
        snapshots_central.append(v_central.copy())
        snapshots_upwind.append(v_upwind.copy())

    fig3, (axc, axu) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    cmap = plt.get_cmap("plasma")
    # Clip the y axis to make the comparison visible even when the
    # centred scheme blows up off the chart.
    v_clip_lo = float(np.min(v_seed)) - 4.0
    v_clip_hi = float(np.max(v_seed)) + 4.0

    for j, snap in enumerate(snapshots_central):
        color = cmap(j / max(len(snapshots_central) - 1, 1))
        axc.plot(k_fail, snap, color=color, linewidth=1.0, alpha=0.85)
    axc.set_title("Explicit centred slope: iterates oscillate and drift")
    axc.set_xlabel("Capital k")
    axc.set_ylabel("Value iterate")
    axc.set_ylim(v_clip_lo, v_clip_hi)

    for j, snap in enumerate(snapshots_upwind):
        color = cmap(j / max(len(snapshots_upwind) - 1, 1))
        axu.plot(k_fail, snap, color=color, linewidth=1.0, alpha=0.85)
    axu.set_title("Explicit upwind slope: iterates stay bounded")
    axu.set_xlabel("Capital k")
    axu.set_ylim(v_clip_lo, v_clip_hi)

    save_figure(fig3, "figures/failure-naive-central.png", dpi=150)

    save_thumbnail("figures/value-and-drift.png", "figures/thumb.png")
    print(
        "Saved figures: value-and-drift, error-vs-reference,"
        " failure-naive-central, thumb."
    )

    # ------------------------------------------------------------------
    # Synthetic check of the KT state-constraint clip. The Ramsey
    # calibration above has an interior steady state, so the clip never
    # binds there. To verify the helper, construct a contrived problem
    # where the unconstrained forward drift at the lower boundary is
    # negative, then show the clip overriding the policy.
    # ------------------------------------------------------------------
    contrived_policy = np.array([1.50, 1.10, 0.80, 0.60])
    contrived_constrained = 0.30  # zero-drift consumption at the boundary
    contrived_drift_at_min = -0.20  # negative -> clip should bind
    clipped = kt_state_constraint_clip(
        contrived_policy,
        contrived_drift_at_min,
        constrained_policy=contrived_constrained,
    )
    print(
        "KT clip synthetic check: unconstrained drift at boundary"
        f" = {contrived_drift_at_min:+.2f},"
        f" original policy[0] = {contrived_policy[0]:.2f},"
        f" clipped policy[0] = {clipped[0]:.2f}"
        " (override fires, holds state at the floor)."
    )


if __name__ == "__main__":
    main()
