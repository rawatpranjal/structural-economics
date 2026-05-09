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
from lib.output import ModelReport
from lib.plotting import setup_style


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

    # ----- Generate report -----
    setup_style()

    report = ModelReport(
        "Envelope-Equation Iteration for Buffer-Stock Saving",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A CRRA household faces IID labor income and cannot borrow. Wealth is "
        "a buffer against low income. The policy says how assets shape "
        "consumption and saving.\n\n"
        "The object is the marginal continuation value $W_a(a)$. It measures "
        "the value of one more dollar before next period's income draw. EEI "
        "updates this curve directly with the envelope condition.\n\n"
        "The computational need is to update this curve without solving for "
        "the whole value function. The Euler equation then recovers "
        "consumption at each asset-income state."
    )

    report.add_equations(
        r"""
The household enters with assets $a$ and IID income $y_j$.
Income has probabilities $\pi_j$ over $\{y_1,\dots,y_{n_y}\}$.
With gross return $R = 1+r$, the Bellman equation is

$$
V(a,y_j) = \max_{a' \geq \underline a}\,
\{\,u(R a + y_j - a') + \beta\,W(a')\,\},
\qquad
W(a') = \sum_{\ell=1}^{n_y}\pi_\ell\,V(a',y_\ell),
$$

The policy is $g(a,y_j)$.
Consumption is $c(a,y_j) = R a + y_j - g(a,y_j)$.

Preferences are CRRA:

$$
u(c) = \frac{c^{1-\gamma}-1}{1-\gamma},
\qquad
u'(c) = c^{-\gamma},
\qquad
(u')^{-1}(\mu) = \mu^{-1/\gamma}.
$$

At an interior optimum, the Euler equation uses only $W_a(a')$:

$$
u'(c(a,y_j)) = \beta\,W_a(g(a,y_j)).
$$

The envelope condition updates that object from the policy:

$$
W_a(a) = \sum_{\ell=1}^{n_y}\pi_\ell\,V_a(a,y_\ell) =
R\,\sum_{\ell=1}^{n_y}\pi_\ell\,u'(c(a,y_\ell)).
$$

These two equations close the system without using the value level.

The borrowing limit binds when the household wants $a' < \underline a$.
Then $g(a,y_j) = \underline a$ and the Euler inequality is

$$
u'(R a + y_j - \underline a) \geq \beta\,W_a(\underline a),
$$

This case produces high MPCs near zero assets.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| CRRA $\\gamma$ | {gamma:.1f} | Curvature; sets the precautionary motive and the slope of $W_a$ |\n"
        f"| Discount factor $\\beta$ | {beta:.2f} | Annual time preference |\n"
        f"| Net rate $r$ | {r:.2f} | Exogenous risk-free return |\n"
        f"| Patience-return product $\\beta R$ | {beta_R:.4f} | $<1$ rules out an unbounded asset target |\n"
        f"| Income mean $\\mu_y$ | {mean_income:.1f} | Normalisation |\n"
        f"| Income s.d. $\\sigma_y$ | {sd_income:.1f} | Width of the IID labor-income shock |\n"
        f"| Income states $n_y$ | {n_income} | Width-fitted equal-spaced normal grid |\n"
        f"| Borrowing limit $\\underline a$ | {a_min:.1f} | Hard zero; binds with positive mass |\n"
        f"| Upper grid bound $\\bar a$ | {a_max:.1f} | Wide enough to contain the simulated tail |\n"
        f"| EEI asset grid | {n_asset} pts | Power-spaced; denser at $\\underline a$ |\n"
        f"| Reference asset grid | {n_asset_ref} pts | Audit grid for the EEI policy |\n"
        f"| Convergence tolerance | {tol:.0e} | Sup-norm on the consumption iterates |\n"
        f"| Simulation | {n_agents:,} households, {periods} periods | Forward-iterated cross section |"
    )

    report.add_solution_method(
        rf"""
EEI starts from a consumption policy.
The envelope step computes $W_a(a_i)$ by averaging marginal utilities across income states.
The Euler step solves for current consumption at each $(a_i,y_j)$.

The Euler step solves a scalar root at each state.
It finds $c \in (0,\,Ra + y_j - \underline a)$ such that

$$
u'(c) = \beta\,W_a(R a + y_j - c).
$$

The borrowing check comes first.
If the household wants to borrow, the solver sets $a'=\underline a$.
Otherwise bisection solves the interior Euler equation.

This update carries only one curve across iterations.
The policy still depends on assets and income after the Euler step.

```text
Algorithm: EEI for IID-income buffer-stock saving
Inputs    asset grid {{a_i}}, income chain ({{y_j}}, {{pi_j}}),
          primitives (beta, R, gamma), borrowing limit a_min, tolerance eps
Output    consumption policy c(a, y), saving policy g(a, y),
          marginal continuation value W_a(a)

Initialise c_0(a_i, y_j) = (R - 1) a_i + y_j        # consume current resources
repeat n = 0, 1, 2, ...
    # 1. Envelope step: collapse the policy into W_a on the exogenous grid
    W_{{a,n}}(a_i) = R * sum_l pi_l * u'(c_n(a_i, y_l))

    # 2. Euler step at each (a_i, y_j)
    for each i, j:
        cash = R a_i + y_j
        if u'(cash - a_min) >= beta * W_{{a,n}}(a_min):
            g_{{n+1}}(a_i, y_j) = a_min                # constraint binds
            c_{{n+1}}(a_i, y_j) = cash - a_min
        else:
            # Solve u'(c) - beta * W_{{a,n}}(cash - c) = 0 by bisection on c.
            c_star = bisect(lambda c: u'(c) - beta * W_{{a,n}}(cash - c),
                            lo=eps, hi=cash - a_min - eps)
            g_{{n+1}}(a_i, y_j) = cash - c_star
            c_{{n+1}}(a_i, y_j) = c_star

    err = max_{{i,j}} |c_{{n+1}}(a_i, y_j) - c_n(a_i, y_j)|
until err < eps
```

The run keeps a {n_asset}-point EEI grid.
It also solves EGP and grid VFI on that grid.
A {n_asset_ref}-point EGP solve checks the policy on $a \leq {audit_max:g}$.

EEI converged in **{eei['iterations']} iterations**.
The maximum consumption gap against the fine-grid policy is {consumption_gap:.2e}.
The same gap for next assets is {savings_gap:.2e}.
"""
    )

    # ---------------- Figures ----------------
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
    report.add_figure(
        "figures/consumption-policy.png",
        "EEI consumption policy with fine-grid reference",
        fig1,
        description=(
            "The consumption policy is increasing and concave in assets. "
            "Income shifts it because IID income enters cash on hand. "
            "Near the borrowing limit, consumption tracks available cash. "
            f"The fine-grid EGP reference stays within {consumption_gap:.2e} "
            f"on $a \\leq {audit_max:g}$."
        ),
    )

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
    report.add_figure(
        "figures/value-derivative.png",
        "Marginal continuation value with state-specific decomposition",
        fig2,
        description=(
            r"$W_a(a)$ is steep near zero assets. "
            "One more dollar is most valuable when the buffer is empty. "
            "The curve flattens as wealth rises. "
            "The envelope condition averages the state-specific marginal utilities."
        ),
    )

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
    report.add_figure(
        "figures/wealth-distribution.png",
        "Simulated terminal wealth distribution under the EEI policy",
        fig3,
        description=(
            "The simulated asset distribution is right-skewed. "
            f"Mean assets are {mean_assets:.2f}. "
            f"{frac_constrained:.1f}% of households sit at the borrowing limit. "
            "IID income keeps the asset scale modest. "
            f"The borrowing-limit mass raises the average MPC to {mean_mpc:.3f}."
        ),
    )

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
    report.add_figure(
        "figures/convergence-comparison.png",
        "Convergence paths for EEI, EGP, and grid VFI on the same asset grid",
        fig4,
        description=(
            "EEI and EGP converge at nearly the same rate. "
            "Both update policies through the Euler equation. "
            "Grid VFI updates the value level and needs more iterations. "
            "This is a fixed-point comparison, not a timing claim."
        ),
    )

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
    report.add_table(
        "tables/solution-statistics.csv",
        "Solution and Simulation Summary",
        df,
        description=(
            "The table reports the main economic moments and policy checks. "
            "The fine-grid rows show interpolation error, not a new model."
        ),
    )

    report.add_takeaway(
        "EEI is a fixed point for the same buffer-stock household. "
        "It iterates $W_a(a)$ instead of the value level. "
        "Low-wealth households consume more of a transfer. "
        "High-wealth households smooth toward the perfect-foresight MPC "
        f"$\\kappa^{{\\ast}}\\approx{mpc_lim:.3f}$. "
        f"Here $\\kappa^{{\\ast}} = R(\\beta R)^{{-1/\\gamma}}-1$ is the MPC "
        f"in the perfect-foresight limit.\n\n"
        "The computational lesson is simple. "
        "The envelope condition can be an update rule. "
        "EGP is faster here because it uses an analytic inverse. "
        "All three methods agree up to the fine-grid gap."
    )

    report.add_references([
        "Arellano, C., Maliar, L., Maliar, S. and Tsyrennikov, V. (2016). Envelope "
        "Condition Method with an Application to Default Risk Models. *Journal of "
        "Economic Dynamics and Control*, 69, 436-459.",
        "Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving "
        "Dynamic Stochastic Optimization Problems. *Economics Letters*, 91(3), "
        "312-320.",
        "Deaton, A. (1991). Saving and Liquidity Constraints. *Econometrica*, "
        "59(5), 1221-1248.",
        "Carroll, C. D. (1997). Buffer-Stock Saving and the Life Cycle/Permanent "
        "Income Hypothesis. *Quarterly Journal of Economics*, 112(1), 1-55.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. "
        "MIT Press, 4th edition, Ch. 18.",
    ])

    report.write("README.md")
    print(
        f"\nGenerated: README.md + {len(report._figures)} figures + "
        f"{len(report._tables)} tables"
    )


if __name__ == "__main__":
    main()
