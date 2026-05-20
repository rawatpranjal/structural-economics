#!/usr/bin/env python3
"""Envelope-equation iteration for an IID income-risk saving problem.

The household saves against labor-income risk under a borrowing limit. The
method iterates the marginal continuation value
$W_a(a) = R\\,\\mathbb{E}_y\\,u'(c(a,y))$. The Euler equation then recovers the
household policy at each asset-income state.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


# -----------------------------------------------------------------------------
# Income discretization (same as the EGP tutorial, kept local for transparency)
# -----------------------------------------------------------------------------
def discrete_normal(
    n: int,
    mu: float,
    sigma: float,
    width: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Equally spaced discretization of N(mu, sigma^2) with cell-midpoint rule."""
    grid = np.linspace(mu - width * sigma, mu + width * sigma, n)
    if n == 2:
        probs = 0.5 * np.ones(n)
    else:
        probs = np.zeros(n)
        half_steps = 0.5 * np.diff(grid)
        probs[0] = norm.cdf(grid[0] + half_steps[0], mu, sigma)
        for i in range(1, n - 1):
            probs[i] = (
                norm.cdf(grid[i] + half_steps[i], mu, sigma)
                - norm.cdf(grid[i] - half_steps[i - 1], mu, sigma)
            )
        probs[-1] = 1.0 - np.sum(probs[:-1])
    mean = float(grid @ probs)
    sd = float(np.sqrt((grid ** 2) @ probs - mean ** 2))
    return sd - sigma, grid, probs


def build_income_grid(
    n_income: int, mean_income: float, sd_income: float
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the support so the discretized standard deviation matches sigma."""
    width = float(fsolve(lambda x: discrete_normal(n_income, mean_income, sd_income, float(x[0]))[0], np.array([2.0]))[0])
    _, grid, probs = discrete_normal(n_income, mean_income, sd_income, width)
    return grid, probs


def lin_interp(x: np.ndarray, y: np.ndarray, xi: np.ndarray | float) -> np.ndarray:
    """Linear interpolation with flat extrapolation. Used inside the Euler step."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    idx = np.clip(np.searchsorted(x, xi) - 1, 0, len(x) - 2)
    x_lo, x_hi = x[idx], x[idx + 1]
    y_lo, y_hi = y[idx], y[idx + 1]
    t = (xi - x_lo) / (x_hi - x_lo + 1e-30)
    return y_lo + t * (y_hi - y_lo)


# -----------------------------------------------------------------------------
# CRRA utility primitives
# -----------------------------------------------------------------------------
def make_crra(gamma: float):
    eps = 1e-15

    def u(c: np.ndarray) -> np.ndarray:
        c = np.maximum(c, eps)
        if gamma == 1.0:
            return np.log(c)
        return (c ** (1.0 - gamma) - 1.0) / (1.0 - gamma)

    def u_prime(c: np.ndarray) -> np.ndarray:
        return np.maximum(c, eps) ** (-gamma)

    def u_prime_inv(m: np.ndarray) -> np.ndarray:
        return np.maximum(m, eps) ** (-1.0 / gamma)

    return u, u_prime, u_prime_inv


# -----------------------------------------------------------------------------
# Solver 1: Envelope-equation iteration on W_a(a)
# -----------------------------------------------------------------------------
def solve_eei(
    asset_grid: np.ndarray,
    income_grid: np.ndarray,
    income_probs: np.ndarray,
    beta: float,
    R: float,
    gamma: float,
    a_min: float,
    tol: float,
    max_iter: int,
    verbose: bool = False,
) -> dict:
    """Iterate $W_a$ via the envelope condition; close the Euler step by bisection."""
    u, u_prime, _ = make_crra(gamma)
    n_a, n_y = asset_grid.shape[0], income_grid.shape[0]

    consumption = (R - 1.0) * asset_grid[:, None] + income_grid[None, :]
    savings = np.zeros((n_a, n_y))
    errors: list[float] = []
    t0 = time.time()

    for iteration in range(1, max_iter + 1):
        consumption_old = consumption.copy()

        # Envelope step: collapse the policy into a 1-D marginal continuation value.
        W_a = R * (u_prime(consumption_old) @ income_probs)

        # Euler step: solve u'(c) = beta * W_a(R*a + y - c) at each (a, y).
        for ia in range(n_a):
            for iy in range(n_y):
                cash = R * asset_grid[ia] + income_grid[iy]
                # Constraint check: at a' = a_min, can the household still want to borrow?
                rhs_constraint = beta * lin_interp(asset_grid, W_a, a_min)
                lhs_constraint = u_prime(cash - a_min)
                if lhs_constraint >= rhs_constraint:
                    consumption[ia, iy] = cash - a_min
                    savings[ia, iy] = a_min
                    continue

                # Interior root: u'(c) - beta * W_a(cash - c) = 0, monotone decreasing in c.
                c_lo, c_hi = 1e-10, cash - a_min - 1e-10
                for _ in range(80):
                    c_mid = 0.5 * (c_lo + c_hi)
                    a_next = cash - c_mid
                    resid = u_prime(c_mid) - beta * lin_interp(asset_grid, W_a, a_next)
                    if resid > 0.0:
                        c_lo = c_mid
                    else:
                        c_hi = c_mid
                    if c_hi - c_lo < 1e-12:
                        break
                c_sol = 0.5 * (c_lo + c_hi)
                consumption[ia, iy] = c_sol
                savings[ia, iy] = cash - c_sol

        err = float(np.max(np.abs(consumption - consumption_old)))
        errors.append(err)
        if verbose and (iteration % 10 == 0 or iteration == 1):
            print(f"  EEI iter {iteration:4d}, sup-norm Δc = {err:.2e}")
        if err < tol:
            break

    W_a_final = R * (u_prime(consumption) @ income_probs)
    return {
        "consumption": consumption,
        "savings": savings,
        "marginal_value": W_a_final,
        "iterations": iteration,
        "errors": errors,
        "elapsed": time.time() - t0,
    }


# -----------------------------------------------------------------------------
# Solver 2: Endogenous grid points (used for the same-grid comparison and as
# the fine-grid reference policy).
# -----------------------------------------------------------------------------
def solve_egp(
    asset_grid: np.ndarray,
    income_grid: np.ndarray,
    income_probs: np.ndarray,
    beta: float,
    R: float,
    gamma: float,
    a_min: float,
    tol: float,
    max_iter: int,
) -> dict:
    """Carroll-style EGP solve, no inner one-dimensional search."""
    u, u_prime, u_prime_inv = make_crra(gamma)
    n_a, n_y = asset_grid.shape[0], income_grid.shape[0]
    consumption = (R - 1.0) * asset_grid[:, None] + income_grid[None, :]
    savings = np.zeros((n_a, n_y))
    errors: list[float] = []
    t0 = time.time()

    for iteration in range(1, max_iter + 1):
        consumption_old = consumption.copy()
        expected_mu = u_prime(consumption_old) @ income_probs
        c_at_next_a = u_prime_inv(beta * R * expected_mu)

        for iy in range(n_y):
            a_endo = (c_at_next_a + asset_grid - income_grid[iy]) / R
            savings[:, iy] = np.where(
                asset_grid < a_endo[0],
                a_min,
                np.interp(asset_grid, a_endo, asset_grid),
            )
            consumption[:, iy] = R * asset_grid + income_grid[iy] - savings[:, iy]

        err = float(np.max(np.abs(consumption - consumption_old)))
        errors.append(err)
        if err < tol:
            break

    return {
        "consumption": consumption,
        "savings": savings,
        "iterations": iteration,
        "errors": errors,
        "elapsed": time.time() - t0,
    }


# -----------------------------------------------------------------------------
# Solver 3: Discrete-choice VFI (grid maximization), kept for the convergence
# comparison plot only.
# -----------------------------------------------------------------------------
def solve_vfi(
    asset_grid: np.ndarray,
    income_grid: np.ndarray,
    income_probs: np.ndarray,
    beta: float,
    R: float,
    gamma: float,
    tol: float,
    max_iter: int,
) -> dict:
    u, u_prime, _ = make_crra(gamma)
    n_a, n_y = asset_grid.shape[0], income_grid.shape[0]
    V = np.zeros((n_a, n_y))
    for iy in range(n_y):
        V[:, iy] = u((R - 1.0) * asset_grid + income_grid[iy]) / (1.0 - beta)
    errors: list[float] = []
    t0 = time.time()

    for iteration in range(1, max_iter + 1):
        V_old = V.copy()
        EV = V_old @ income_probs
        for ia in range(n_a):
            for iy in range(n_y):
                cash = R * asset_grid[ia] + income_grid[iy]
                c = cash - asset_grid
                feasible = c > 1e-10
                vals = np.full(n_a, -1e20)
                vals[feasible] = u(c[feasible]) + beta * EV[feasible]
                V[ia, iy] = vals.max()

        err = float(np.max(np.abs(V - V_old)))
        errors.append(err)
        if err < tol:
            break

    return {"iterations": iteration, "errors": errors, "elapsed": time.time() - t0}


# -----------------------------------------------------------------------------
# Simulation under a saving policy
# -----------------------------------------------------------------------------
def simulate_panel(
    asset_grid: np.ndarray,
    savings: np.ndarray,
    income_grid: np.ndarray,
    income_probs: np.ndarray,
    n_agents: int,
    periods: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(income_probs)
    n_y = income_grid.shape[0]
    sav_interp = [
        interp1d(
            asset_grid,
            savings[:, iy],
            kind="linear",
            bounds_error=False,
            fill_value=(savings[0, iy], savings[-1, iy]),
        )
        for iy in range(n_y)
    ]
    assets = np.zeros(n_agents)
    income_idx = np.zeros(n_agents, dtype=int)
    for t in range(periods):
        income_idx = np.searchsorted(cdf, rng.random(n_agents), side="right")
        if t == periods - 1:
            break
        assets_next = np.empty_like(assets)
        for iy in range(n_y):
            mask = income_idx == iy
            assets_next[mask] = sav_interp[iy](assets[mask])
        assets = assets_next
    return assets, income_idx


def main() -> None:
    # Preferences and returns
    gamma = 2.0
    beta = 0.95
    r = 0.03
    R = 1.0 + r
    beta_R = beta * R

    # Income risk: 5-state IID approximation to N(mu_y, sd_y^2)
    mean_income = 1.0
    sd_income = 0.2
    n_income = 5

    # Asset grid: power-spaced so points pile up near the borrowing limit
    a_min = 0.0
    a_max = 50.0
    n_asset = 50
    grid_curvature = 0.5  # exponent in u**(1/curvature); <1 packs near a_min
    n_asset_ref = 600

    # Solver controls
    tol = 1.0e-6
    max_iter = 1000

    # Simulation
    n_agents = 50_000
    periods = 500
    seed = 2020

    # MPC experiment
    transfer = 0.10

    # ----- Build grids -----
    income_grid, income_probs = build_income_grid(n_income, mean_income, sd_income)
    print(f"Income grid: {income_grid}")
    print(f"Income probs: {income_probs}")

    raw = np.linspace(0.0, 1.0, n_asset)
    asset_grid = a_min + (a_max - a_min) * raw ** (1.0 / grid_curvature)

    raw_ref = np.linspace(0.0, 1.0, n_asset_ref)
    asset_grid_ref = a_min + (a_max - a_min) * raw_ref ** (1.0 / grid_curvature)

    # ----- Solve household problem three ways -----
    print("\nMethod 1: Envelope-equation iteration (EEI)")
    eei = solve_eei(
        asset_grid, income_grid, income_probs, beta, R, gamma, a_min,
        tol=tol, max_iter=max_iter, verbose=True,
    )
    print(f"  EEI: {eei['iterations']} iterations in {eei['elapsed']:.2f}s")

    print("\nMethod 2: Endogenous grid points (same grid, for comparison)")
    egp = solve_egp(
        asset_grid, income_grid, income_probs, beta, R, gamma, a_min,
        tol=tol, max_iter=max_iter,
    )
    print(f"  EGP: {egp['iterations']} iterations in {egp['elapsed']:.2f}s")

    print("\nMethod 3: Grid VFI (same grid, for the contraction comparison)")
    vfi = solve_vfi(
        asset_grid, income_grid, income_probs, beta, R, gamma,
        tol=tol, max_iter=max_iter,
    )
    print(f"  VFI: {vfi['iterations']} iterations in {vfi['elapsed']:.2f}s")

    print(f"\nFine-grid EGP reference ({n_asset_ref} points)")
    egp_ref = solve_egp(
        asset_grid_ref, income_grid, income_probs, beta, R, gamma, a_min,
        tol=tol, max_iter=max_iter,
    )
    print(f"  Reference EGP: {egp_ref['iterations']} iterations in {egp_ref['elapsed']:.2f}s")

    # ----- Discretization audit on the active asset range -----
    audit_max = 20.0
    audit_mask = asset_grid <= audit_max
    consumption_ref_on_main = np.zeros_like(eei["consumption"])
    savings_ref_on_main = np.zeros_like(eei["savings"])
    for iy in range(n_income):
        consumption_ref_on_main[:, iy] = np.interp(
            asset_grid, asset_grid_ref, egp_ref["consumption"][:, iy]
        )
        savings_ref_on_main[:, iy] = np.interp(
            asset_grid, asset_grid_ref, egp_ref["savings"][:, iy]
        )
    consumption_gap = float(
        np.max(np.abs(eei["consumption"][audit_mask] - consumption_ref_on_main[audit_mask]))
    )
    savings_gap = float(
        np.max(np.abs(eei["savings"][audit_mask] - savings_ref_on_main[audit_mask]))
    )
    _, u_prime_local, _ = make_crra(gamma)
    W_a_ref = R * (u_prime_local(egp_ref["consumption"]) @ income_probs)

    # ----- Simulate the EEI policy -----
    print("\nSimulating EEI policy")
    final_assets, final_income_idx = simulate_panel(
        asset_grid, eei["savings"], income_grid, income_probs,
        n_agents=n_agents, periods=periods, seed=seed,
    )

    # MPCs from a 0.10 transfer
    con_interp = [
        interp1d(asset_grid, eei["consumption"][:, iy], kind="linear",
                 bounds_error=False, fill_value="extrapolate")
        for iy in range(n_income)
    ]
    mpc_sim = np.zeros(n_agents)
    for iy in range(n_income):
        mask = final_income_idx == iy
        a_i = final_assets[mask]
        mpc_sim[mask] = (con_interp[iy](a_i + transfer) - con_interp[iy](a_i)) / transfer
    mean_mpc = float(np.mean(mpc_sim))
    mpc_lim = R * (beta_R ** (-1.0 / gamma)) - 1.0

    mean_assets = float(np.mean(final_assets))
    frac_constrained = float(np.mean(final_assets <= a_min + 1e-6) * 100.0)
    p10, p50, p90 = (float(np.quantile(final_assets, q)) for q in (0.10, 0.50, 0.90))

    print(f"  Mean assets: {mean_assets:.3f}")
    print(f"  Fraction at constraint: {frac_constrained:.1f}%")
    print(f"  Mean MPC (0.10 transfer): {mean_mpc:.3f}")
    print(f"  Perfect-foresight MPC limit: {mpc_lim:.4f}")
    print(f"  Coarse vs fine-grid consumption gap (a≤{audit_max:g}): {consumption_gap:.2e}")
    print(f"  Coarse vs fine-grid saving gap (a≤{audit_max:g}): {savings_gap:.2e}")

    # ---------------- Figures ----------------
    setup_style()
    plot_max = 20.0
    low, mid, high = 0, n_income // 2, n_income - 1

    # Figure 1: Consumption policy with fine-grid reference
    fig1, ax1 = plt.subplots()
    ax1.plot(asset_grid, eei["consumption"][:, low], color="steelblue",
             linewidth=2.0, label=f"EEI, low income $y_1={income_grid[low]:.2f}$")
    ax1.plot(asset_grid, eei["consumption"][:, mid], color="seagreen",
             linewidth=2.0, label=f"EEI, mid income $y_{{{mid+1}}}={income_grid[mid]:.2f}$")
    ax1.plot(asset_grid, eei["consumption"][:, high], color="indianred",
             linewidth=2.0, label=f"EEI, high income $y_{{{high+1}}}={income_grid[high]:.2f}$")
    ax1.plot(asset_grid_ref, egp_ref["consumption"][:, low], color="black",
             linewidth=1.2, linestyle="--", alpha=0.85, label=f"Fine-grid ref ($N_a={n_asset_ref}$)")
    ax1.plot(asset_grid_ref, egp_ref["consumption"][:, high], color="black",
             linewidth=1.2, linestyle="--", alpha=0.85)
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("Consumption $c(a, y_j)$")
    ax1.set_title("Consumption Policy")
    ax1.set_xlim(0.0, plot_max)
    ax1.legend()
    save_figure(fig1, "figures/consumption-policy.png", dpi=150)

    # Figure 2: Marginal continuation value W_a
    fig2, ax2 = plt.subplots()
    ax2.plot(asset_grid, eei["marginal_value"], color="navy", linewidth=2.0,
             label=r"$W_a(a)$, EEI")
    ax2.plot(asset_grid_ref, W_a_ref, color="black", linestyle="--",
             linewidth=1.2, alpha=0.85, label=r"$W_a(a)$, fine-grid ref")
    ax2.plot(asset_grid, R * u_prime_local(eei["consumption"][:, low]),
             linestyle=":", linewidth=1.2, color="steelblue", alpha=0.85,
             label=r"$R\,u'(c(a, y_1))$")
    ax2.plot(asset_grid, R * u_prime_local(eei["consumption"][:, high]),
             linestyle=":", linewidth=1.2, color="indianred", alpha=0.85,
             label=r"$R\,u'(c(a, y_{n_y}))$")
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel(r"Marginal continuation value")
    ax2.set_title("The Iterated Object: $W_a(a)$")
    ax2.set_xlim(0.0, plot_max)
    ax2.set_ylim(0.0, float(min(eei["marginal_value"][0] * 1.3, eei["marginal_value"].max() * 1.2)))
    ax2.legend()
    save_figure(fig2, "figures/value-derivative.png", dpi=150)

    # Figure 3: Simulated stationary wealth distribution
    fig3, ax3 = plt.subplots()
    upper_hist = float(np.quantile(final_assets, 0.99) * 1.1)
    ax3.hist(final_assets, bins=60, density=True, color="steelblue",
             edgecolor="navy", linewidth=0.3, alpha=0.85)
    ax3.axvline(mean_assets, color="darkred", linestyle="--", linewidth=1.4,
                label=f"Mean = {mean_assets:.2f}")
    ax3.axvline(p50, color="darkorange", linestyle=":", linewidth=1.4,
                label=f"Median = {p50:.2f}")
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Density")
    ax3.set_title("Simulated Stationary Wealth Distribution")
    ax3.set_xlim(0.0, upper_hist)
    ax3.legend()
    save_figure(fig3, "figures/wealth-distribution.png", dpi=150)

    # Figure 4: Convergence comparison
    fig4, ax4 = plt.subplots()
    ax4.semilogy(range(1, len(eei["errors"]) + 1), eei["errors"],
                 color="navy", linewidth=2.0, label=f"EEI ({eei['iterations']} iter)")
    ax4.semilogy(range(1, len(egp["errors"]) + 1), egp["errors"],
                 color="indianred", linewidth=2.0, label=f"EGP ({egp['iterations']} iter)")
    ax4.semilogy(range(1, len(vfi["errors"]) + 1), vfi["errors"],
                 color="black", alpha=0.7, linewidth=2.0,
                 label=f"VFI ({vfi['iterations']} iter)")
    ax4.axhline(tol, color="gray", linewidth=1.0, linestyle=":",
                label=f"Tolerance = {tol:.0e}")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Sup-norm error (log scale)")
    ax4.set_title("Convergence: EEI vs EGP vs Grid VFI")
    ax4.set_xlim(0, max(len(eei["errors"]), len(egp["errors"]), min(len(vfi["errors"]), 500)))
    ax4.legend()
    save_figure(fig4, "figures/convergence-comparison.png", dpi=150)

    # ---------------- Table ----------------
    table_data = {
        "Statistic": [
            "EEI iterations",
            "Same-grid EGP iterations",
            "Same-grid VFI iterations",
            "Fine-grid reference points",
            "Fine-grid reference iterations",
            f"Max consumption gap vs reference, a ≤ {audit_max:g}",
            f"Max next-asset gap vs reference, a ≤ {audit_max:g}",
            "Mean assets",
            "Fraction at borrowing limit",
            "Mean MPC, 0.10 transfer",
            "Perfect-foresight MPC limit",
            "10th percentile wealth",
            "50th percentile wealth",
            "90th percentile wealth",
        ],
        "Value": [
            f"{eei['iterations']}",
            f"{egp['iterations']}",
            f"{vfi['iterations']}",
            f"{n_asset_ref}",
            f"{egp_ref['iterations']}",
            f"{consumption_gap:.2e}",
            f"{savings_gap:.2e}",
            f"{mean_assets:.4f}",
            f"{frac_constrained:.1f}%",
            f"{mean_mpc:.4f}",
            f"{mpc_lim:.4f}",
            f"{p10:.3f}",
            f"{p50:.3f}",
            f"{p90:.3f}",
        ],
    }
    df = pd.DataFrame(table_data)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/solution-statistics.csv", index=False)

    save_thumbnail("figures/consumption-policy.png", "figures/thumb.png")
    print(
        f"\nDone: 4 figures, 1 table. Mean assets={mean_assets:.4f}, "
        f"MPC={mean_mpc:.4f}, MPC limit={mpc_lim:.4f}"
    )


if __name__ == "__main__":
    main()
