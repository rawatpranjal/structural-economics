#!/usr/bin/env python3
"""Continuous-time Aiyagari (HACT) and the mean-field game framing.

Solves a production economy with idiosyncratic income risk using the
HJB/KFE approach of Achdou-Han-Lasry-Lions-Moll (2022). Households pick
consumption to maximise discounted utility; firms hire capital and labor
at competitive prices; the interest rate clears the capital market. The
HJB, the stationary KFE, and the market-clearing condition together
constitute a Lasry-Lions mean-field game on the (a, z) state space.

The same calibration is solved by a discrete-time Aiyagari reference
(VFI plus stationary forward iteration) inside this script. The two
solutions overlay consumption rules, MPCs, and the wealth density so the
reader can see exactly what continuous time buys.

References:
    Aiyagari, S. R. (1994). "Uninsured Idiosyncratic Risk and Aggregate
        Saving." Quarterly Journal of Economics, 109(3), 659-684.
    Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022).
        "Income and Wealth Distribution in Macroeconomics: A
        Continuous-Time Approach." Review of Economic Studies 89(1), 45-86.
    Lasry, J.-M., and Lions, P.-L. (2007). "Mean field games." Japanese
        Journal of Mathematics 2(1), 229-260.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import logm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.discretize import rouwenhorst
from lib.grids import exponential_grid
from lib.plotting import setup_style, save_figure, save_thumbnail


# =============================================================================
# Continuous-time core: HJB, KFE, equilibrium
# =============================================================================

def build_ctmc_generator(P: np.ndarray, dt: float = 1.0) -> tuple[np.ndarray, str]:
    """Convert a discrete-time transition matrix into a CTMC generator Q.

    The continuous-time generator satisfies P = expm(Q * dt) in the
    embeddable case. We try Q = logm(P)/dt first. If logm returns complex
    entries or negative off-diagonals beyond tolerance, fall back to
    Q = (P - I)/dt, which is always a valid CTMC generator.

    Returns:
        Q: (n, n) generator with non-negative off-diagonals and zero row sums.
        method: "logm" or "linear" indicating which branch was used.
    """
    n = P.shape[0]
    Q_log = logm(P) / dt
    if np.max(np.abs(Q_log.imag)) < 1e-10:
        Q_log = Q_log.real
        off = Q_log - np.diag(np.diag(Q_log))
        if off.min() > -1e-10:
            diag_mask = np.eye(n, dtype=bool)
            Q = Q_log.copy()
            Q[~diag_mask] = np.maximum(Q[~diag_mask], 0.0)
            Q[diag_mask] = 0.0
            Q[diag_mask] = -Q.sum(axis=1)
            return Q, "logm"
    Q = (P - np.eye(n)) / dt
    return Q, "linear"


def build_switch_matrix(Q: np.ndarray, n_asset: int) -> sparse.csc_matrix:
    """Block-Kronecker product Q (x) I_n_asset for the income-switching part.

    Index convention: stacked state (j, k) lives at row j * n_asset + k,
    so V_stacked = [V[:, 0]; V[:, 1]; ...; V[:, n_z - 1]].
    """
    return sparse.kron(sparse.csc_matrix(Q), sparse.eye(n_asset, format="csc"), format="csc")


def solve_hjb(r: float, w: float, params: dict) -> tuple:
    """Implicit upwind HJB for the household's value function.

    Generalises the Huggett two-state HJB to an N-state continuous-time
    income chain with generator Q. Wage w enters the budget so the
    drift is s_j(a) = w * z_j + r * a - c_j(a).

    Returns:
        V: (I, n_z) value function.
        c: (I, n_z) consumption policy.
        s: (I, n_z) savings drift.
        A: (n_z * I, n_z * I) sparse upwind generator.
        info: dict with convergence diagnostics.
    """
    rho = params["rho"]
    sigma = params["sigma"]
    z = params["z"]
    a = params["a"]
    I = params["I"]
    da = params["da"]
    n_z = params["n_z"]
    Aswitch = params["Aswitch"]
    Delta = params["Delta"]
    maxit = params["maxit"]
    crit = params["crit"]

    aa = np.broadcast_to(a[:, None], (I, n_z))
    zz = np.broadcast_to(z[None, :], (I, n_z))

    income = w * zz + r * aa
    income_pos = np.maximum(income, 1e-10)
    V = income_pos ** (1 - sigma) / (1 - sigma) / rho

    dVf = np.zeros((I, n_z))
    dVb = np.zeros((I, n_z))

    for n in range(1, maxit + 1):
        dVf[:I - 1, :] = (V[1:I, :] - V[:I - 1, :]) / da
        dVf[I - 1, :] = np.maximum(w * z + r * a[-1], 1e-10) ** (-sigma)

        dVb[1:I, :] = (V[1:I, :] - V[:I - 1, :]) / da
        dVb[0, :] = np.maximum(w * z + r * a[0], 1e-10) ** (-sigma)

        cf = np.maximum(dVf, 1e-10) ** (-1.0 / sigma)
        ssf = w * zz + r * aa - cf
        cb = np.maximum(dVb, 1e-10) ** (-1.0 / sigma)
        ssb = w * zz + r * aa - cb
        c0 = w * zz + r * aa

        If = (ssf > 0).astype(float)
        Ib = (ssb < 0).astype(float)
        I0 = 1.0 - If - Ib

        c = cf * If + cb * Ib + c0 * I0
        c = np.maximum(c, 1e-10)
        u = c ** (1 - sigma) / (1 - sigma)

        X = -np.minimum(ssb, 0) / da
        Y = -np.maximum(ssf, 0) / da + np.minimum(ssb, 0) / da
        Z = np.maximum(ssf, 0) / da

        blocks = []
        for j in range(n_z):
            block = (sparse.diags(Y[:, j], 0, shape=(I, I))
                     + sparse.diags(X[1:I, j], -1, shape=(I, I))
                     + sparse.diags(Z[:I - 1, j], 1, shape=(I, I)))
            blocks.append(block)
        A = sparse.block_diag(blocks, format="csc") + Aswitch

        B = (1.0 / Delta + rho) * sparse.eye(n_z * I, format="csc") - A
        u_stacked = u.T.reshape(-1)
        V_stacked = V.T.reshape(-1)
        rhs = u_stacked + V_stacked / Delta
        V_new_stacked = spsolve(B, rhs)
        V_new = V_new_stacked.reshape(n_z, I).T

        change = np.max(np.abs(V_new - V))
        V = V_new
        if change < crit:
            break

    dVf[:I - 1, :] = (V[1:I, :] - V[:I - 1, :]) / da
    dVf[I - 1, :] = np.maximum(w * z + r * a[-1], 1e-10) ** (-sigma)
    dVb[1:I, :] = (V[1:I, :] - V[:I - 1, :]) / da
    dVb[0, :] = np.maximum(w * z + r * a[0], 1e-10) ** (-sigma)

    cf = np.maximum(dVf, 1e-10) ** (-1.0 / sigma)
    ssf = w * zz + r * aa - cf
    cb = np.maximum(dVb, 1e-10) ** (-1.0 / sigma)
    ssb = w * zz + r * aa - cb
    c0 = w * zz + r * aa
    If = (ssf > 0).astype(float)
    Ib = (ssb < 0).astype(float)
    I0 = 1.0 - If - Ib
    c = cf * If + cb * Ib + c0 * I0
    s = w * zz + r * aa - c

    info = {"iterations": n, "converged": change < crit, "error": change}
    return V, c, s, A, info


def solve_kfe(A: sparse.spmatrix, params: dict) -> np.ndarray:
    """Stationary KFE: solve A^T g = 0 with integral(g) = 1."""
    I = params["I"]
    n_z = params["n_z"]
    da = params["da"]

    AT = A.T.tocsc()
    rhs = np.zeros(n_z * I)
    i_fix = 0
    rhs[i_fix] = 0.1
    AT = AT.tolil()
    AT[i_fix, :] = 0
    AT[i_fix, i_fix] = 1.0
    AT = AT.tocsc()

    gg = spsolve(AT, rhs)
    gg = gg / (gg.sum() * da)
    g = gg.reshape(n_z, I).T
    return g


def capital_demand(r: float, alpha: float, delta: float, L: float = 1.0) -> float:
    """Firm capital demand from the FOC at interest rate r."""
    return L * ((r + delta) / alpha) ** (1.0 / (alpha - 1.0))


def wage_from_K(K: float, alpha: float, L: float = 1.0) -> float:
    """Competitive wage at aggregate capital K."""
    return (1.0 - alpha) * (K / L) ** alpha


def aggregate_supply(r: float, params: dict) -> tuple:
    """Solve HJB+KFE at r and return household capital supply plus diagnostics."""
    K_d = capital_demand(r, params["alpha"], params["delta"], params["L"])
    w = wage_from_K(K_d, params["alpha"], params["L"])
    V, c, s, A, info = solve_hjb(r, w, params)
    g = solve_kfe(A, params)
    a = params["a"]
    da = params["da"]
    K_s = float(np.sum(g.sum(axis=1) * a) * da)
    return K_s, K_d, w, V, c, s, g, info


def find_equilibrium(params: dict, r_min: float, r_max: float,
                     tol: float = 1e-5, max_iter: int = 40, label: str = "") -> dict:
    """Bisection on r until household capital supply equals firm demand."""
    r_lo, r_hi = r_min, r_max
    r_history: list[float] = []
    Ks_history: list[float] = []
    Kd_history: list[float] = []

    r_eq = 0.5 * (r_lo + r_hi)
    K_s = K_d = w = V = c = s = g = info = None

    for it in range(1, max_iter + 1):
        r_mid = 0.5 * (r_lo + r_hi)
        K_s, K_d, w, V, c, s, g, info = aggregate_supply(r_mid, params)

        r_history.append(r_mid)
        Ks_history.append(K_s)
        Kd_history.append(K_d)

        gap = K_s - K_d
        print(f"  {label}Bisection {it:2d}: r = {r_mid:.6f}, "
              f"K_s = {K_s:.4f}, K_d = {K_d:.4f}, gap = {gap:+.4e}")

        if abs(gap) / max(K_d, 1e-6) < tol:
            r_eq = r_mid
            print(f"  {label}Equilibrium: r* = {r_eq:.6f}")
            break
        elif gap > 0:
            r_hi = r_mid
        else:
            r_lo = r_mid
        r_eq = r_mid

    return {
        "r_eq": r_eq, "K_s": K_s, "K_d": K_d, "w": w,
        "V": V, "c": c, "s": s, "g": g, "info": info,
        "r_history": np.array(r_history),
        "Ks_history": np.array(Ks_history),
        "Kd_history": np.array(Kd_history),
        "iterations": it,
    }


# =============================================================================
# Discrete-time reference: VFI replica of dynamic-programming/aiyagari/run.py
# =============================================================================

def crra_utility(c: np.ndarray, sigma: float) -> np.ndarray:
    c_safe = np.maximum(c, 1e-15)
    if sigma == 1.0:
        return np.log(c_safe)
    return c_safe ** (1.0 - sigma) / (1.0 - sigma)


def dt_solve_household(a_grid: np.ndarray, z_grid: np.ndarray, P: np.ndarray,
                       beta: float, sigma: float, r: float, w: float,
                       tol: float = 1e-7, max_iter: int = 4000,
                       value_init: np.ndarray | None = None) -> dict:
    """Vectorised grid-search VFI on (a, z). Mirrors the discrete-time Aiyagari."""
    n_asset = a_grid.size
    cash = (1.0 + r) * a_grid[:, None] + w * z_grid[None, :]
    consumption = cash[:, :, None] - a_grid[None, None, :]
    feasible = consumption > 1e-12
    flow = crra_utility(np.maximum(consumption, 1e-15), sigma)

    if value_init is not None:
        value = value_init.copy()
    else:
        value = crra_utility(np.maximum(cash, 1e-12), sigma) / (1.0 - beta)

    for it in range(1, max_iter + 1):
        expected = value @ P.T
        cand = flow + beta * expected.T[None, :, :]
        cand = np.where(feasible, cand, -np.inf)
        idx = np.argmax(cand, axis=2)
        v_new = np.take_along_axis(cand, idx[:, :, None], axis=2).squeeze(2)
        err = float(np.max(np.abs(v_new - value)))
        value = v_new
        if err < tol:
            break

    asset_policy = a_grid[idx]
    consumption_policy = cash - asset_policy
    return {
        "value": value, "policy_idx": idx,
        "asset_policy": asset_policy, "consumption_policy": consumption_policy,
        "iterations": it, "error": err,
    }


def dt_stationary_distribution(policy_idx: np.ndarray, P: np.ndarray,
                               tol: float = 1e-11, max_iter: int = 10_000) -> np.ndarray:
    """Forward iterate the (a, z) distribution under (policy, P)."""
    n_asset, n_income = policy_idx.shape
    dist = np.full((n_asset, n_income), 1.0 / (n_asset * n_income))
    for _ in range(max_iter):
        new = np.zeros_like(dist)
        for j in range(n_income):
            mass = dist[:, j]
            np.add.at(new, (policy_idx[:, j], slice(None)),
                      mass[:, None] * P[j, None, :])
        if float(np.max(np.abs(new - dist))) < tol:
            dist = new
            break
        dist = new
    return dist


def dt_aggregate_capital(dist: np.ndarray, a_grid: np.ndarray) -> float:
    return float(np.sum(dist * a_grid[:, None]))


def gini_from_density(a: np.ndarray, mass: np.ndarray) -> float:
    """Gini from a non-negative wealth distribution mass on a sorted grid."""
    order = np.argsort(a)
    a_sorted = a[order]
    m_sorted = mass[order]
    cum_pop = np.cumsum(m_sorted)
    cum_wealth = np.cumsum(m_sorted * np.maximum(a_sorted, 0.0))
    if cum_wealth[-1] <= 0:
        return 0.0
    cum_pop = np.concatenate([[0.0], cum_pop / cum_pop[-1]])
    cum_wealth = np.concatenate([[0.0], cum_wealth / cum_wealth[-1]])
    return float(1.0 - 2.0 * np.trapezoid(cum_wealth, cum_pop))


def find_equilibrium_dt(a_grid: np.ndarray, z_grid: np.ndarray, P: np.ndarray,
                        beta: float, sigma: float, alpha: float, delta: float,
                        r_low: float, r_high: float, tol: float = 5e-4,
                        max_iter: int = 60) -> dict:
    """Bisect r in the discrete-time Aiyagari economy."""
    value_warm: np.ndarray | None = None
    sol = None
    dist = None
    K_s = K_d = w = r_trial = None
    for it in range(1, max_iter + 1):
        r_trial = 0.5 * (r_low + r_high)
        K_d = capital_demand(r_trial, alpha, delta, L=1.0)
        w = wage_from_K(K_d, alpha, L=1.0)
        sol = dt_solve_household(a_grid, z_grid, P, beta, sigma, r_trial, w,
                                 value_init=value_warm)
        value_warm = sol["value"]
        dist = dt_stationary_distribution(sol["policy_idx"], P)
        K_s = dt_aggregate_capital(dist, a_grid)
        gap_rel = (K_s - K_d) / K_d
        print(f"  [DT] iter {it:2d}: r = {r_trial:.6f}, K_s = {K_s:.4f}, "
              f"K_d = {K_d:.4f}, rel gap = {gap_rel:+.3e}")
        if abs(gap_rel) < tol:
            break
        if gap_rel > 0:
            r_high = r_trial
        else:
            r_low = r_trial
        if r_high - r_low < 1e-6:
            break
    return {
        "r_eq": r_trial, "K_s": K_s, "K_d": K_d, "w": w,
        "sol": sol, "dist": dist, "iterations": it,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # =========================================================================
    # Calibration (matches dynamic-programming/aiyagari for direct comparison)
    # =========================================================================
    beta = 0.96
    sigma = 2.0
    alpha = 0.36
    delta = 0.08
    rho_z = 0.9
    sigma_eps = 0.2
    n_z = 7

    rho = -np.log(beta)
    L = 1.0

    a_min = 0.0
    a_max = 30.0
    I = 800

    # Income chain: same Rouwenhorst grid as the discrete-time Aiyagari, then
    # mapped to a CTMC generator.
    z_grid_log_jax, P_jax, ergodic_jax = rouwenhorst(
        n=n_z, mu=0.0, sigma=sigma_eps, rho=rho_z,
    )
    z_grid = np.exp(np.asarray(z_grid_log_jax).ravel())
    P = np.asarray(P_jax)
    ergodic = np.asarray(ergodic_jax).ravel()
    norm_z = float(np.dot(ergodic, z_grid))
    z_grid = z_grid / norm_z
    Q, ctmc_method = build_ctmc_generator(P, dt=1.0)
    print(f"CTMC generator built via {ctmc_method}; row sums max abs: "
          f"{float(np.max(np.abs(Q.sum(axis=1)))):.2e}")

    a = np.linspace(a_min, a_max, I)
    da = (a_max - a_min) / (I - 1)
    Aswitch = build_switch_matrix(Q, I)

    params = {
        "rho": rho, "sigma": sigma, "z": z_grid, "a": a, "I": I,
        "da": da, "n_z": n_z, "Aswitch": Aswitch,
        "alpha": alpha, "delta": delta, "L": L,
        "Delta": 1000.0, "maxit": 100, "crit": 1e-6,
    }

    impatience_rate = 1.0 / beta - 1.0
    print(f"\nContinuous-time Aiyagari (HACT): bisecting r on capital market")
    eq = find_equilibrium(params, r_min=0.005, r_max=impatience_rate - 1e-3,
                          tol=5e-4, max_iter=40)
    r_eq = eq["r_eq"]
    K_eq = eq["K_d"]
    w_eq = eq["w"]
    Y_eq = K_eq ** alpha * L ** (1.0 - alpha)
    KY_ratio = K_eq / Y_eq
    market_gap = eq["K_s"] - eq["K_d"]
    market_gap_rel = market_gap / eq["K_d"]
    wedge = impatience_rate - r_eq

    V = eq["V"]; c = eq["c"]; s = eq["s"]; g = eq["g"]
    g_marginal = g.sum(axis=1)
    mean_wealth = float(np.sum(a * g_marginal) * da)
    mass_at_constraint = float(np.sum(g_marginal[a <= a_min + 0.02]) * da)
    hact_gini = gini_from_density(a, g_marginal * da)

    # MPC at the constraint from finite differences of the consumption rule.
    mpc_hact = np.zeros_like(c)
    mpc_hact[:-1, :] = (c[1:, :] - c[:-1, :]) / da
    mpc_hact[-1, :] = mpc_hact[-2, :]
    mpc_hact_at_lim = mpc_hact[0, :]

    # =========================================================================
    # Capital supply curve for the equilibrium picture (HACT side only)
    # =========================================================================
    print("\nHACT: tracing K_s(r) for the capital-market figure")
    n_supply = 9
    r_supply = np.linspace(0.005, impatience_rate - 5e-4, n_supply)
    Ks_curve = np.empty(n_supply)
    Kd_curve = np.empty(n_supply)
    for idx_r, r_val in enumerate(r_supply):
        K_s_local, K_d_local, *_ = aggregate_supply(r_val, params)
        Ks_curve[idx_r] = K_s_local
        Kd_curve[idx_r] = K_d_local
        print(f"  supply sweep: r = {r_val:+.4f}, K_s = {K_s_local:.4f}, "
              f"K_d = {K_d_local:.4f}")

    # =========================================================================
    # Discrete-time reference
    # =========================================================================
    print("\nDiscrete-time Aiyagari reference")
    n_asset_dt = 200
    a_grid_dt = np.asarray(exponential_grid(0.0, 50.0, n_asset_dt, density=3.0))
    dt_eq = find_equilibrium_dt(a_grid_dt, z_grid, P, beta, sigma, alpha, delta,
                                r_low=0.005, r_high=impatience_rate - 1e-3,
                                tol=5e-4, max_iter=60)
    r_eq_dt = dt_eq["r_eq"]
    K_eq_dt = dt_eq["K_d"]
    dist_dt = dt_eq["dist"]
    dt_marginal = dist_dt.sum(axis=1)
    dt_consumption = dt_eq["sol"]["consumption_policy"]
    mean_wealth_dt = dt_aggregate_capital(dist_dt, a_grid_dt)
    dt_gini = gini_from_density(a_grid_dt, dt_marginal)
    # Sum the DT probability mass over the same physical window the HACT side
    # integrates (a within a_min + 0.02). On the exponential DT grid several
    # nodes fall inside this window, so this is the like-for-like analogue of
    # the HACT integral, not the bare node-0 point mass.
    dt_mass_at_constraint = float(np.sum(dt_marginal[a_grid_dt <= a_min + 0.02]))

    # =========================================================================
    # Comparison overlays
    # =========================================================================
    # Consumption rule: interpolate DT policy onto HACT grid for plotting.
    dt_c_on_a = np.column_stack([
        np.interp(a, a_grid_dt, dt_consumption[:, j]) for j in range(n_z)
    ])
    # DT MPC by finite difference on its own grid, then interp onto a for figure.
    dt_mpc_on_grid = np.zeros_like(dt_consumption)
    dt_mpc_on_grid[:-1, :] = (dt_consumption[1:, :] - dt_consumption[:-1, :]) / \
                              (a_grid_dt[1:, None] - a_grid_dt[:-1, None])
    dt_mpc_on_grid[-1, :] = dt_mpc_on_grid[-2, :]
    dt_mpc_on_a = np.column_stack([
        np.interp(a, a_grid_dt, dt_mpc_on_grid[:, j]) for j in range(n_z)
    ])

    # =========================================================================
    # Figures
    # =========================================================================
    setup_style()

    # Figure 1: Capital-market clearing (headline -> thumbnail)
    fig_cap, ax_cap = plt.subplots()
    r_demand_dense = np.linspace(0.005, impatience_rate - 5e-4, 200)
    K_demand_dense = np.array([capital_demand(rv, alpha, delta, L) for rv in r_demand_dense])
    ax_cap.plot(K_demand_dense, r_demand_dense, color="tab:blue", linewidth=2.0,
                label="Firm demand $K^d(r)$")
    ax_cap.plot(Ks_curve, r_supply, color="tab:red", linewidth=2.0,
                marker="o", markersize=5,
                label="Household supply $K^s(r)$ (HACT)")
    ax_cap.axhline(impatience_rate, color="0.45", linestyle=":", linewidth=1.0,
                   label="$1/\\beta - 1$")
    ax_cap.scatter([K_eq], [r_eq], s=140, marker="*", color="black", zorder=5,
                   label=f"HACT ($K^{{\\ast}} = {K_eq:.2f}$, $r^{{\\ast}} = {r_eq:.4f}$)")
    ax_cap.scatter([K_eq_dt], [r_eq_dt], s=110, marker="D", color="0.25", zorder=5,
                   label=f"DT ($K^{{\\ast}}_{{\\rm DT}} = {K_eq_dt:.2f}$, "
                         f"$r^{{\\ast}}_{{\\rm DT}} = {r_eq_dt:.4f}$)")
    x_lo = 0.85 * min(Ks_curve.min(), K_demand_dense.min())
    x_hi = 1.05 * max(Ks_curve.max(), K_demand_dense.max(), K_eq)
    ax_cap.set_xlim(x_lo, x_hi)
    ax_cap.set_ylim(0.0, impatience_rate + 5e-3)
    ax_cap.set_xlabel("Aggregate capital $K$")
    ax_cap.set_ylabel("Interest rate $r$")
    ax_cap.set_title("Capital-Market Clearing")
    ax_cap.legend(loc="upper right", fontsize=9)
    save_figure(fig_cap, "figures/capital-market.png", dpi=150)

    # Figure 2: Consumption rule comparison (HACT vs DT, 3 income states)
    plot_states = [0, n_z // 2, n_z - 1]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_z))
    fig_c, ax_c = plt.subplots()
    a_plot_max = 10.0
    plot_mask = a <= a_plot_max
    for j in plot_states:
        ax_c.plot(a[plot_mask], c[plot_mask, j], color=cmap[j], linewidth=2.0,
                  label=f"HACT, $z_{{{j+1}}} = {z_grid[j]:.2f}$")
        ax_c.plot(a[plot_mask], dt_c_on_a[plot_mask, j], color=cmap[j],
                  linewidth=1.4, linestyle="--",
                  label=f"DT, $z_{{{j+1}}} = {z_grid[j]:.2f}$")
    ax_c.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_c.set_xlabel("Assets $a$")
    ax_c.set_ylabel("Consumption $c_j(a)$")
    ax_c.set_title("Consumption Rule: HACT vs Discrete-time Aiyagari")
    ax_c.legend(loc="lower right", fontsize=8, ncol=2)
    save_figure(fig_c, "figures/consumption-policy-comparison.png", dpi=150)

    # Figure 3: Savings policy (HACT only)
    fig_s, ax_s = plt.subplots()
    plot_states_full = [0, n_z // 4, n_z // 2, 3 * n_z // 4, n_z - 1]
    for j in plot_states_full:
        ax_s.plot(a[plot_mask], s[plot_mask, j], color=cmap[j], linewidth=2.0,
                  label=f"$z_{{{j+1}}} = {z_grid[j]:.2f}$")
    ax_s.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax_s.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_s.set_xlabel("Assets $a$")
    ax_s.set_ylabel("Savings drift $s_j(a) = w z_j + r a - c_j(a)$")
    ax_s.set_title("Savings Drift by Income State")
    ax_s.legend(loc="upper right", fontsize=9)
    save_figure(fig_s, "figures/savings-policy.png", dpi=150)

    # Figure 4: Wealth distribution comparison (HACT density vs DT histogram)
    fig_d, (ax_w, ax_l) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    a_max_plot = float(min(a_max, max(8.0, mean_wealth * 2.8)))
    n_bins = 50
    bin_edges = np.linspace(0.0, a_max_plot, n_bins + 1)
    bin_mass_dt = np.zeros(n_bins)
    in_range = a_grid_dt <= a_max_plot
    bin_idx = np.clip(np.digitize(a_grid_dt[in_range], bin_edges) - 1, 0, n_bins - 1)
    np.add.at(bin_mass_dt, bin_idx, dt_marginal[in_range])
    bin_density_dt = bin_mass_dt / np.diff(bin_edges)
    ax_w.bar(bin_edges[:-1], bin_density_dt, width=np.diff(bin_edges), align="edge",
             color="lightsteelblue", alpha=0.7, edgecolor="steelblue",
             linewidth=0.4, label="DT histogram density")
    ax_w.plot(a, g_marginal, color="firebrick", linewidth=2.2,
              label="HACT density $\\sum_j g_j(a)$")
    ax_w.axvline(mean_wealth, color="black", linestyle="--", linewidth=1.2,
                 label=f"HACT mean $= {mean_wealth:.2f}$")
    ax_w.set_xlim(0, a_max_plot)
    ax_w.set_xlabel("Assets $a$")
    ax_w.set_ylabel("Density")
    ax_w.set_title("Stationary Wealth Distribution")
    ax_w.legend(loc="upper right", fontsize=8)

    # Lorenz curves
    hact_mass = g_marginal * da
    order_hact = np.argsort(a)
    cum_pop_hact = np.cumsum(hact_mass[order_hact])
    cum_w_hact = np.cumsum(hact_mass[order_hact] * np.maximum(a[order_hact], 0))
    cum_pop_hact = np.concatenate([[0], cum_pop_hact / cum_pop_hact[-1]])
    cum_w_hact = np.concatenate([[0], cum_w_hact / max(cum_w_hact[-1], 1e-12)])
    order_dt = np.argsort(a_grid_dt)
    cum_pop_dt = np.cumsum(dt_marginal[order_dt])
    cum_w_dt = np.cumsum(dt_marginal[order_dt] * np.maximum(a_grid_dt[order_dt], 0))
    cum_pop_dt = np.concatenate([[0], cum_pop_dt / cum_pop_dt[-1]])
    cum_w_dt = np.concatenate([[0], cum_w_dt / max(cum_w_dt[-1], 1e-12)])
    ax_l.plot(cum_pop_hact, cum_w_hact, color="firebrick", linewidth=2.2,
              label=f"HACT (Gini $= {hact_gini:.3f}$)")
    ax_l.plot(cum_pop_dt, cum_w_dt, color="steelblue", linewidth=1.6, linestyle="--",
              label=f"DT (Gini $= {dt_gini:.3f}$)")
    ax_l.plot([0, 1], [0, 1], color="0.4", linestyle=":", linewidth=0.9,
              label="Equality")
    ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.set_aspect("equal")
    ax_l.set_xlabel("Population share (sorted by assets)")
    ax_l.set_ylabel("Wealth share")
    ax_l.set_title("Lorenz Curves")
    ax_l.legend(loc="upper left", fontsize=9)
    fig_d.tight_layout()
    save_figure(fig_d, "figures/wealth-distribution-comparison.png", dpi=150)

    # Figure 5: MPC comparison
    fig_m, ax_m = plt.subplots()
    j_show = 0
    mpc_plot_max = 3.0
    mpc_mask = a <= mpc_plot_max
    ax_m.plot(a[mpc_mask], mpc_hact[mpc_mask, j_show], color="firebrick", linewidth=2.2,
              label=f"HACT, $z_{{{j_show+1}}} = {z_grid[j_show]:.2f}$ (lowest)")
    ax_m.plot(a[mpc_mask], dt_mpc_on_a[mpc_mask, j_show], color="steelblue",
              linewidth=1.6, linestyle="--",
              label=f"DT (interp), $z_{{{j_show+1}}} = {z_grid[j_show]:.2f}$")
    ax_m.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_m.annotate(f"HACT MPC at $a = 0$: {mpc_hact_at_lim[j_show]:.3f}",
                  xy=(a_min, mpc_hact_at_lim[j_show]),
                  xytext=(0.4, mpc_hact_at_lim[j_show] * 0.6),
                  arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
                  fontsize=9)
    ax_m.set_xlabel("Assets $a$")
    ax_m.set_ylabel("MPC $\\partial c / \\partial a$")
    ax_m.set_title("MPC by Wealth at the Lowest Income State")
    ax_m.legend(loc="upper right", fontsize=9)
    save_figure(fig_m, "figures/mpc-comparison.png", dpi=150)

    # =========================================================================
    # Tables
    # =========================================================================
    Path("tables").mkdir(parents=True, exist_ok=True)

    eq_table = pd.DataFrame({
        "Statistic": [
            "Interest rate r*",
            "Wage w*",
            "Aggregate capital K*",
            "Output Y*",
            "Capital-output ratio K/Y",
            "Precautionary wedge 1/beta - 1 - r*",
            "Mean wealth E[a]",
            "Wealth Gini",
            "Mass within 0.02 of borrowing limit",
            "HACT MPC at borrowing limit, lowest income",
            "HJB iterations at equilibrium",
            "HJB sup-norm change at equilibrium",
            "Bisection iterations",
            "Relative capital-market gap",
        ],
        "Value": [
            f"{r_eq:.6f}",
            f"{w_eq:.4f}",
            f"{K_eq:.4f}",
            f"{Y_eq:.4f}",
            f"{KY_ratio:.4f}",
            f"{wedge:.5f}",
            f"{mean_wealth:.4f}",
            f"{hact_gini:.4f}",
            f"{mass_at_constraint:.4f}",
            f"{mpc_hact_at_lim[0]:.4f}",
            f"{eq['info']['iterations']}",
            f"{eq['info']['error']:.2e}",
            f"{eq['iterations']}",
            f"{market_gap_rel:+.3e}",
        ],
    })
    eq_table.to_csv("tables/equilibrium.csv", index=False)

    gini_gap = abs(hact_gini - dt_gini)
    mass_gap_pp = 100.0 * abs(mass_at_constraint - dt_mass_at_constraint)
    cmp_table = pd.DataFrame({
        "Object": [
            "Interest rate r*",
            "Aggregate capital K*",
            "Mean wealth",
            "Wealth Gini",
            "Mass at borrowing limit",
            "MPC at borrowing limit, lowest income",
        ],
        "HACT": [
            f"{r_eq:.5f}",
            f"{K_eq:.4f}",
            f"{mean_wealth:.4f}",
            f"{hact_gini:.4f}",
            f"{mass_at_constraint:.4f}",
            f"{mpc_hact_at_lim[0]:.4f}",
        ],
        "Discrete-time": [
            f"{r_eq_dt:.5f}",
            f"{K_eq_dt:.4f}",
            f"{mean_wealth_dt:.4f}",
            f"{dt_gini:.4f}",
            f"{dt_mass_at_constraint:.4f}",
            f"{dt_mpc_on_a[0, 0]:.4f}",
        ],
        "Absolute gap": [
            f"{abs(r_eq - r_eq_dt):.5f}",
            f"{abs(K_eq - K_eq_dt):.4f}",
            f"{abs(mean_wealth - mean_wealth_dt):.4f}",
            f"{gini_gap:.4f}",
            f"{abs(mass_at_constraint - dt_mass_at_constraint):.4f}",
            f"{abs(mpc_hact_at_lim[0] - dt_mpc_on_a[0, 0]):.4f}",
        ],
    })
    cmp_table.to_csv("tables/dt-vs-ct-comparison.csv", index=False)

    save_thumbnail("figures/capital-market.png", "figures/thumb.png")
    print(f"\nDone: 5 figures, 2 tables. r*={r_eq:.5f}, K*={K_eq:.4f}")


if __name__ == "__main__":
    main()
