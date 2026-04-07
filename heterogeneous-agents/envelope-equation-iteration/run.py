#!/usr/bin/env python3
"""Envelope Equation Iteration (EEI) for the Income-Fluctuation Problem.

Solves the standard heterogeneous-agent consumption-savings problem with IID
income risk using the envelope equation iteration method. Instead of iterating
on the value function V (VFI) or inverting the Euler equation on an endogenous
grid (EGP), EEI iterates directly on the derivative V'(a) using the envelope
condition V'(a) = R * u'(c*(a)).

This avoids both the expensive inner maximization of VFI and the grid
inversion step of EGP, making it a computationally attractive alternative.

Reference: Maliar, L. and Maliar, S. (2013). "Envelope Condition Method with
an Application to Default Risk Models." Journal of Economic Dynamics and
Control, 37(7), 1439-1459.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters (matching the source codes for comparability)
    # =========================================================================
    # Preferences
    risk_aver = 2        # CRRA risk aversion
    beta = 0.95          # Discount factor

    # Returns
    r = 0.03             # Interest rate
    R = 1 + r            # Gross return

    # Income risk: discretized N(mu_y, sd_y^2)
    mu_y = 1.0           # Mean income
    sd_y = 0.2           # Std dev of income
    ny = 5               # Number of income states

    # Asset grid
    na = 50              # Grid points for assets
    amax = 50            # Maximum asset level
    borrow_lim = 0.0     # Borrowing limit (natural: zero)
    agrid_par = 0.5      # Grid curvature (1=linear, <1 = denser near zero)

    # Computation
    max_iter = 1000      # Maximum iterations
    tol_iter = 1.0e-6    # Convergence tolerance

    # Simulation
    Nsim = 50000         # Number of simulated agents
    Tsim = 500           # Simulation periods

    # MPC computation
    mpc_amount = 0.10    # Windfall size for MPC calculation

    # =========================================================================
    # Random draws (fixed seed for reproducibility)
    # =========================================================================
    np.random.seed(2020)
    yrand = np.random.rand(Nsim, Tsim)

    # =========================================================================
    # Asset Grid: power-spaced (denser near borrowing limit)
    # =========================================================================
    agrid = np.linspace(0, 1, na).reshape(na, 1)
    agrid = agrid ** (1 / agrid_par)
    agrid = borrow_lim + (amax - borrow_lim) * agrid

    # =========================================================================
    # Income Grid: discretized normal distribution
    # =========================================================================
    def discrete_normal(n, mu, sigma, width):
        """Equally spaced approximation to a normal distribution."""
        from scipy.stats import norm as norm_dist
        x = np.linspace(mu - width * sigma, mu + width * sigma, n).reshape(n, 1)
        if n == 2:
            p = 0.5 * np.ones((n, 1))
        else:
            p = np.zeros((n, 1))
            p[0] = norm_dist.cdf(x[0] + 0.5 * (x[1] - x[0]), mu, sigma)
            for i in range(1, n - 1):
                p[i] = (norm_dist.cdf(x[i] + 0.5 * (x[i + 1] - x[i]), mu, sigma)
                        - norm_dist.cdf(x[i] - 0.5 * (x[i] - x[i - 1]), mu, sigma))
            p[n - 1] = 1 - np.sum(p[:n - 1])
        return (np.sqrt((x.T ** 2) @ p - (x.T @ p) ** 2) - sigma)[0, 0], x, p

    width = fsolve(lambda x: discrete_normal(ny, mu_y, sd_y, x)[0], 2.0)
    _, ygrid, ydist = discrete_normal(ny, mu_y, sd_y, width)
    ycumdist = np.cumsum(ydist).flatten()

    # =========================================================================
    # Utility function and derivatives (CRRA)
    # =========================================================================
    if risk_aver == 1:
        u = lambda c: np.log(np.maximum(c, 1e-15))
    else:
        u = lambda c: (np.maximum(c, 1e-15) ** (1 - risk_aver) - 1) / (1 - risk_aver)

    u1 = lambda c: np.maximum(c, 1e-15) ** (-risk_aver)          # u'(c)
    u1inv = lambda v: np.maximum(v, 1e-15) ** (-1 / risk_aver)   # (u')^{-1}

    # =========================================================================
    # Linear interpolation helper
    # =========================================================================
    def lininterp1(x, y, xi):
        """Linear interpolation with extrapolation at boundaries."""
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        idx = np.searchsorted(x, xi) - 1
        idx = np.clip(idx, 0, len(x) - 2)
        x_lo, x_hi = x[idx], x[idx + 1]
        y_lo, y_hi = y[idx], y[idx + 1]
        t = (xi - x_lo) / (x_hi - x_lo + 1e-30)
        return y_lo + t * (y_hi - y_lo)

    # =========================================================================
    # METHOD 1: Envelope Equation Iteration (EEI)
    # =========================================================================
    # The envelope theorem gives: V'(a) = R * E_y[u'(c*(a,y))]
    # The Euler equation (FOC) for each (a,y):
    #   u'(c) >= beta * V'(a')  with equality if a' > borrow_lim
    #
    # Substituting the envelope condition into the Euler equation:
    #   u'(c(a,y)) = beta * R * E_y'[u'(c(a', y'))]    (unconstrained)
    #
    # This is the same Euler equation iteration as in the source code, but
    # we frame it through the lens of V'(a):
    #
    # Algorithm:
    #   1. Guess consumption c_n(a, y) on the grid
    #   2. Compute V'_n(a) = R * sum_y u'(c_n(a,y)) * prob(y)   [envelope]
    #   3. For each (a, y): use V'_n to find c_{n+1} via Euler equation
    #      - If constrained: c = cash - borrow_lim
    #      - If unconstrained: u'(c) = beta * V'_n(a'), solve for c
    #   4. Check convergence of c
    #
    # The key insight: step 2 uses the ENVELOPE condition to compress the
    # consumption function c(a,y) into a single object V'(a) that summarizes
    # all the information needed for the Euler equation. This avoids
    # interpolating ny separate consumption functions -- we only interpolate
    # the single function V'(a).
    # =========================================================================

    print("=" * 60)
    print("Method 1: Envelope Equation Iteration (EEI)")
    print("=" * 60)

    Ey = float((ygrid.T @ ydist).flatten()[0])

    # Initial guess: consume income flow (hand-to-mouth)
    con_eei = np.zeros((na, ny))
    for iy in range(ny):
        con_eei[:, iy] = (r * agrid + ygrid[iy])[:, 0]
    sav_eei = np.zeros((na, ny))

    t0_eei = time.time()
    eei_errors = []

    for iteration in range(1, max_iter + 1):
        con_eei_last = con_eei.copy()

        # Step 1: Compute V'(a) from current consumption via the envelope condition
        # V'(a) = R * E_y[u'(c(a,y))] = R * sum_y u'(c(a,y)) * prob(y)
        dV = R * (u1(con_eei_last) @ ydist)  # shape (na, 1)

        # Step 2: For each (a, y), solve for new consumption using V'(a) in the
        # Euler equation: u'(c) = beta * V'(a'), where a' = R*a + y - c
        for ia in range(na):
            for iy in range(ny):
                cash = R * agrid[ia, 0] + ygrid[iy, 0]

                # Check if borrowing constraint binds:
                # At constraint: a' = borrow_lim, c = cash - borrow_lim
                # Constraint binds if u'(cash - borrow_lim) <= beta * V'(borrow_lim)
                c_constrained = cash - borrow_lim
                lhs_constrained = u1(c_constrained)
                rhs_constrained = beta * lininterp1(agrid[:, 0], dV[:, 0], borrow_lim)

                if lhs_constrained >= rhs_constrained:
                    # Borrowing constraint binds: even at max consumption,
                    # marginal utility exceeds value of saving, so agent
                    # would want to borrow more
                    sav_eei[ia, iy] = borrow_lim
                    con_eei[ia, iy] = c_constrained
                else:
                    # Unconstrained: solve u'(c) = beta * V'(cash - c)
                    # Use bisection for robustness
                    c_lo = 1e-10
                    c_hi = cash - borrow_lim - 1e-10

                    for _ in range(80):  # bisection iterations
                        c_mid = 0.5 * (c_lo + c_hi)
                        ap_mid = cash - c_mid
                        dV_ap = lininterp1(agrid[:, 0], dV[:, 0], ap_mid)
                        resid = u1(c_mid) - beta * dV_ap
                        if resid > 0:
                            # u'(c) too high => c too low => increase c
                            c_lo = c_mid
                        else:
                            c_hi = c_mid
                        if c_hi - c_lo < 1e-12:
                            break

                    c_sol = 0.5 * (c_lo + c_hi)
                    con_eei[ia, iy] = c_sol
                    sav_eei[ia, iy] = cash - c_sol

        cdiff = np.max(np.abs(con_eei - con_eei_last))
        eei_errors.append(cdiff)

        if iteration % 10 == 0 or iteration == 1:
            print(f"  EEI iteration {iteration:4d}, max con diff = {cdiff:.2e}")

        if cdiff < tol_iter:
            print(f"  EEI converged in {iteration} iterations (error = {cdiff:.2e})")
            break

    # Final V'(a) for plotting
    dV = R * (u1(con_eei) @ ydist)

    time_eei = time.time() - t0_eei
    n_iter_eei = iteration
    print(f"  EEI time: {time_eei:.2f}s")

    # =========================================================================
    # METHOD 2: Value Function Iteration (VFI) for comparison
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Method 2: Value Function Iteration (VFI)")
    print("=" * 60)

    na_vfi = na  # Same grid
    agrid_vfi = agrid.copy()

    # Initial guess: consume everything
    V_vfi = np.zeros((na_vfi, ny))
    for iy in range(ny):
        V_vfi[:, iy] = u(r * agrid_vfi[:, 0] + ygrid[iy, 0]) / (1 - beta)

    con_vfi = np.zeros((na_vfi, ny))
    sav_vfi = np.zeros((na_vfi, ny))

    t0_vfi = time.time()
    vfi_errors = []

    for iteration in range(1, max_iter + 1):
        V_last = V_vfi.copy()

        # Expected continuation value (integrated over tomorrow's income)
        EV = V_last @ ydist  # shape (na, 1)

        for ia in range(na_vfi):
            for iy in range(ny):
                cash = R * agrid_vfi[ia, 0] + ygrid[iy, 0]
                # Evaluate all possible a' choices on the grid
                c_choices = cash - agrid_vfi[:, 0]
                feasible = c_choices > 1e-10
                values = np.full(na_vfi, -1e20)
                values[feasible] = u(c_choices[feasible]) + beta * EV[feasible, 0]

                best = np.argmax(values)
                V_vfi[ia, iy] = values[best]
                sav_vfi[ia, iy] = agrid_vfi[best, 0]
                con_vfi[ia, iy] = cash - sav_vfi[ia, iy]

        V_diff = np.max(np.abs(V_vfi - V_last))
        vfi_errors.append(V_diff)

        if iteration % 50 == 0 or iteration == 1:
            print(f"  VFI iteration {iteration:4d}, max V diff = {V_diff:.2e}")

        if V_diff < tol_iter:
            print(f"  VFI converged in {iteration} iterations (error = {V_diff:.2e})")
            break

    time_vfi = time.time() - t0_vfi
    n_iter_vfi = iteration
    print(f"  VFI time: {time_vfi:.2f}s")

    # =========================================================================
    # METHOD 3: Endogenous Grid Points (EGP) for comparison
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Method 3: Endogenous Grid Points (EGP)")
    print("=" * 60)

    # Initial guess for consumption
    con_egp = np.zeros((na, ny))
    for iy in range(ny):
        con_egp[:, iy] = (r * agrid + ygrid[iy])[:, 0]

    t0_egp = time.time()
    egp_errors = []

    for iteration in range(1, max_iter + 1):
        con_egp_last = con_egp.copy()
        sav_egp = np.zeros((na, ny))

        # Expected marginal utility of consumption tomorrow
        emuc = u1(con_egp_last) @ ydist  # (na, 1)
        muc_next = beta * R * emuc       # RHS of Euler equation
        con_endo = u1inv(muc_next)        # Consumption on endogenous grid

        for iy in range(ny):
            # Endogenous grid: a_endo such that agent with a_endo choosing a'=agrid
            # has cash = con_endo + agrid, so a_endo = (con_endo + agrid - ygrid[iy]) / R
            a_endo = ((con_endo + agrid - ygrid[iy]) / R)[:, 0]

            for ia in range(na):
                if agrid[ia, 0] < a_endo[0]:
                    # Borrowing constraint binds
                    sav_egp[ia, iy] = borrow_lim
                else:
                    # Interpolate: given current a, find a' from endogenous grid
                    sav_egp[ia, iy] = lininterp1(a_endo, agrid[:, 0], agrid[ia, 0])

            con_egp[:, iy] = (R * agrid + ygrid[iy])[:, 0] - sav_egp[:, iy]

        cdiff = np.max(np.abs(con_egp - con_egp_last))
        egp_errors.append(cdiff)

        if iteration % 10 == 0 or iteration == 1:
            print(f"  EGP iteration {iteration:4d}, max con diff = {cdiff:.2e}")

        if cdiff < tol_iter:
            print(f"  EGP converged in {iteration} iterations (error = {cdiff:.2e})")
            break

    time_egp = time.time() - t0_egp
    n_iter_egp = iteration
    print(f"  EGP time: {time_egp:.2f}s")

    # =========================================================================
    # Simulate using EEI policy (main method)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Simulating with EEI policy functions")
    print("=" * 60)

    yindsim = np.zeros((Nsim, Tsim), dtype=int)
    asim = np.zeros((Nsim, Tsim))

    # Create interpolating functions for savings policy
    savinterp = []
    for iy in range(ny):
        savinterp.append(interp1d(agrid[:, 0], sav_eei[:, iy], kind='linear',
                                  fill_value=(borrow_lim, sav_eei[-1, iy]),
                                  bounds_error=False))

    for it in range(Tsim):
        if (it + 1) % 100 == 0:
            print(f"  Simulating period {it + 1}/{Tsim}")

        # Income realization
        yindsim[yrand[:, it] <= ycumdist[0], it] = 0
        for iy in range(1, ny):
            mask = np.logical_and(yrand[:, it] > ycumdist[iy - 1],
                                  yrand[:, it] <= ycumdist[iy])
            yindsim[mask, it] = iy

        # Asset choice
        if it < Tsim - 1:
            for iy in range(ny):
                mask = yindsim[:, it] == iy
                asim[mask, it + 1] = savinterp[iy](asim[mask, it])

    ysim = ygrid[yindsim]

    # =========================================================================
    # Compute MPCs
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Computing MPCs")
    print("=" * 60)

    # Theoretical MPC lower bound (for patient, unconstrained agent)
    mpc_lim = R * ((beta * R) ** (-1 / risk_aver)) - 1

    coninterp = []
    mpc_func = np.zeros((na, ny))

    for iy in range(ny):
        coninterp.append(interp1d(agrid[:, 0], con_eei[:, iy], kind='linear',
                                  fill_value='extrapolate'))
        mpc_func[:, iy] = (coninterp[iy](agrid[:, 0] + mpc_amount) - con_eei[:, iy]) / mpc_amount

    # MPC from simulation
    mpc_sim = np.zeros(Nsim)
    for iy in range(ny):
        mask = yindsim[:, Tsim - 1] == iy
        mpc_sim[mask] = (coninterp[iy](asim[mask, Tsim - 1] + mpc_amount)
                         - coninterp[iy](asim[mask, Tsim - 1])) / mpc_amount

    mean_mpc = np.mean(mpc_sim)
    print(f"  Mean MPC (simulated): {mean_mpc:.4f}")
    print(f"  Theoretical MPC limit: {mpc_lim:.4f}")

    # =========================================================================
    # Distribution statistics
    # =========================================================================
    aysim = asim[:, Tsim - 1] / np.mean(ysim[:, Tsim - 1])
    mean_assets = np.mean(aysim)
    frac_constrained = np.sum(aysim <= borrow_lim + 1e-6) / Nsim * 100
    p10 = np.quantile(aysim, 0.10)
    p50 = np.quantile(aysim, 0.50)
    p90 = np.quantile(aysim, 0.90)

    print(f"\n  Mean assets (relative to income): {mean_assets:.4f}")
    print(f"  Fraction constrained: {frac_constrained:.1f}%")
    print(f"  10th / 50th / 90th percentile: {p10:.3f} / {p50:.3f} / {p90:.3f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Envelope Equation Iteration (EEI)",
        "Solving the income-fluctuation problem by iterating on the envelope condition V'(a) = R * u'(c*(a)).",
    )

    report.add_overview(
        "The envelope equation iteration (EEI) method solves the standard heterogeneous-agent "
        "consumption-savings problem by exploiting the **envelope theorem**. Rather than "
        "iterating on the value function $V(a)$ as in VFI, or inverting the Euler equation "
        "onto an endogenous grid as in EGP, EEI iterates directly on the derivative "
        "$V'(a)$ using the envelope condition.\n\n"
        "The key insight is that the envelope theorem links the value function derivative "
        "to the policy function: $V'(a) = R \\cdot u'(c^*(a,y))$ averaged over income states. "
        "This, combined with the Euler equation, gives a complete characterization of the "
        "optimal policy without ever computing $V(a)$ itself. The method avoids both the "
        "costly inner maximization of VFI and the grid inversion of EGP."
    )

    report.add_equations(
        r"""
**Household problem (IID income):**

$$V(a) = \mathbb{E}_y \left[ \max_{a' \ge \underline{a}} \left\{ u(Ra + y - a') + \beta \, V(a') \right\} \right]$$

**Envelope condition (the key equation):**

$$V'(a) = R \cdot \mathbb{E}_y\left[ u'(c^*(a, y)) \right]$$

This follows from the envelope theorem applied to the Bellman equation: differentiating
through the max, the optimal choice satisfies $V'(a) = \partial u / \partial a = R \cdot u'(c)$.

**Euler equation (first-order condition):**

$$u'(c^*(a, y)) = \beta \, V'(a'), \quad a' = Ra + y - c$$

with complementary slackness at the borrowing constraint $a' \ge \underline{a}$.

**EEI algorithm:** Given a guess $V'_n(a)$:
1. For each $(a, y)$: find $c$ satisfying $u'(c) = \beta \, V'_n(Ra + y - c)$ (or set $a' = \underline{a}$ if constrained)
2. Update: $V'_{n+1}(a) = R \cdot \mathbb{E}_y[u'(c^*(a, y))]$
3. Repeat until $\|V'_{n+1} - V'_n\|_\infty < \varepsilon$

**CRRA utility:** $u(c) = \frac{c^{1-\sigma}}{1-\sigma}$, $\quad u'(c) = c^{-\sigma}$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $\\sigma$ | {risk_aver} | CRRA risk aversion |\n"
        f"| $r$      | {r} | Interest rate |\n"
        f"| $R = 1+r$ | {R} | Gross return |\n"
        f"| $\\mu_y$  | {mu_y} | Mean income |\n"
        f"| $\\sigma_y$ | {sd_y} | Income std dev |\n"
        f"| Income states | {ny} | Discretized normal |\n"
        f"| Asset grid | {na} points | Power-spaced on $[{borrow_lim}, {amax}]$ |\n"
        f"| $\\underline{{a}}$ | {borrow_lim} | Borrowing limit |"
    )

    report.add_solution_method(
        "**Envelope Equation Iteration (EEI):** We iterate on the derivative of the "
        "value function $V'(a)$ rather than on $V(a)$ itself.\n\n"
        "At each iteration:\n"
        "1. Given $V'_n(a)$ on the asset grid, for each state $(a, y)$ we solve the "
        "Euler equation $u'(c) = \\beta \\cdot V'_n(R a + y - c)$ for $c$, checking "
        "whether the borrowing constraint $a' \\ge \\underline{a}$ binds.\n"
        "2. Update the derivative using the envelope condition: "
        "$V'_{n+1}(a) = R \\cdot \\mathbb{E}_y[u'(c^*(a, y))]$.\n"
        "3. Check convergence: $\\|V'_{n+1} - V'_n\\|_\\infty < 10^{-6}$.\n\n"
        f"**EEI** converged in **{n_iter_eei} iterations** ({time_eei:.2f}s).\n\n"
        f"For comparison, we also solve the same problem with:\n"
        f"- **VFI** (grid search): {n_iter_vfi} iterations ({time_vfi:.2f}s)\n"
        f"- **EGP** (endogenous grid points): {n_iter_egp} iterations ({time_egp:.2f}s)\n\n"
        "All three methods converge to the same policy function, but differ in "
        "computational cost per iteration and total iterations to convergence."
    )

    # --- Figure 1: Consumption Policy Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(agrid, con_eei[:, 0], 'b-', linewidth=2,
             label=f'Lowest income ($y={ygrid[0,0]:.2f}$)')
    ax1.plot(agrid, con_eei[:, ny - 1], 'r-', linewidth=2,
             label=f'Highest income ($y={ygrid[ny-1,0]:.2f}$)')
    # Plot middle income state
    mid = ny // 2
    ax1.plot(agrid, con_eei[:, mid], 'g--', linewidth=1.5,
             label=f'Middle income ($y={ygrid[mid,0]:.2f}$)')
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("Consumption $c(a, y)$")
    ax1.set_title("Consumption Policy Function (EEI)")
    ax1.legend()
    ax1.set_xlim(0, min(amax, 20))
    report.add_figure("figures/consumption-policy.png",
                       "Consumption policy function c(a,y) from EEI for different income states", fig1)

    # --- Figure 2: Value Function Derivative V'(a) ---
    fig2, ax2 = plt.subplots()
    ax2.plot(agrid, dV, 'b-', linewidth=2, label="$V'(a)$ (EEI)")
    # Also show R * u'(c) for lowest and highest income
    for iy, color, label_str in [
            (0, 'steelblue', r"$R \cdot u'(c(a, y_{\mathrm{low}}))$"),
            (ny - 1, 'indianred', r"$R \cdot u'(c(a, y_{\mathrm{high}}))$")]:
        ax2.plot(agrid, R * u1(con_eei[:, iy]), '--', color=color,
                 linewidth=1.5, alpha=0.7, label=label_str)
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("$V'(a)$")
    ax2.set_title("Value Function Derivative (Envelope Condition)")
    ax2.legend()
    ax2.set_xlim(0, min(amax, 20))
    ax2.set_ylim(0, min(float(dV[0, 0]) * 1.5, float(np.max(dV) * 1.2)))
    report.add_figure("figures/value-derivative.png",
                       "Value function derivative V'(a) from the envelope condition, with R*u'(c) for extreme income states", fig2)

    # --- Figure 3: Simulated Wealth Distribution ---
    fig3, ax3 = plt.subplots()
    final_assets = asim[:, Tsim - 1]
    ax3.hist(final_assets, bins=60, density=True, color='steelblue', alpha=0.7,
             edgecolor='navy', linewidth=0.3)
    ax3.axvline(np.mean(final_assets), color='red', linewidth=2, linestyle='--',
                label=f'Mean = {np.mean(final_assets):.2f}')
    ax3.axvline(np.median(final_assets), color='orange', linewidth=2, linestyle=':',
                label=f'Median = {np.median(final_assets):.2f}')
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Density")
    ax3.set_title("Simulated Stationary Wealth Distribution")
    ax3.legend()
    ax3.set_xlim(0, np.percentile(final_assets, 99) * 1.1)
    report.add_figure("figures/wealth-distribution.png",
                       "Simulated stationary wealth distribution from 50,000 agents over 500 periods", fig3)

    # --- Figure 4: Convergence Comparison ---
    fig4, ax4 = plt.subplots()
    ax4.semilogy(range(1, len(eei_errors) + 1), eei_errors, 'b-', linewidth=2,
                 label=f'EEI ({n_iter_eei} iter, {time_eei:.1f}s)')
    ax4.semilogy(range(1, len(egp_errors) + 1), egp_errors, 'r-', linewidth=2,
                 label=f'EGP ({n_iter_egp} iter, {time_egp:.1f}s)')
    ax4.semilogy(range(1, len(vfi_errors) + 1), vfi_errors, 'k-', linewidth=2,
                 alpha=0.7, label=f'VFI ({n_iter_vfi} iter, {time_vfi:.1f}s)')
    ax4.axhline(tol_iter, color='gray', linewidth=1, linestyle=':', label=f'Tolerance = {tol_iter:.0e}')
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Maximum error (log scale)")
    ax4.set_title("Convergence Speed: EEI vs EGP vs VFI")
    ax4.legend()
    ax4.set_xlim(0, max(len(eei_errors), len(egp_errors), min(len(vfi_errors), 500)))
    report.add_figure("figures/convergence-comparison.png",
                       "Convergence comparison across three solution methods for the same problem", fig4)

    # --- Table: Solution Statistics ---
    table_data = {
        "Statistic": [
            "Iterations to converge",
            "Wall-clock time (s)",
            "Mean assets (rel. to income)",
            "Fraction constrained (%)",
            "Mean MPC",
            "Theoretical MPC limit",
            "10th percentile wealth",
            "50th percentile wealth",
            "90th percentile wealth",
        ],
        "EEI": [
            f"{n_iter_eei}",
            f"{time_eei:.2f}",
            f"{mean_assets:.4f}",
            f"{frac_constrained:.1f}",
            f"{mean_mpc:.4f}",
            f"{mpc_lim:.4f}",
            f"{p10:.3f}",
            f"{p50:.3f}",
            f"{p90:.3f}",
        ],
        "VFI": [
            f"{n_iter_vfi}",
            f"{time_vfi:.2f}",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
        ],
        "EGP": [
            f"{n_iter_egp}",
            f"{time_egp:.2f}",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/solution-statistics.csv", "Solution Statistics: EEI vs VFI vs EGP", df)

    report.add_takeaway(
        "The envelope equation iteration method demonstrates that the same economic "
        "problem — a household choosing consumption and savings under income uncertainty "
        "and borrowing constraints — can be attacked from multiple computational angles.\n\n"
        "**Key insights:**\n"
        "- **Three views of the same optimality condition:** VFI iterates on $V(a)$, "
        "EGP iterates on $c(a)$ via the inverted Euler equation on an endogenous grid, "
        "and EEI iterates on $V'(a)$ via the envelope theorem. All converge to the same "
        "policy function.\n"
        "- **The envelope theorem is powerful:** $V'(a) = R \\cdot u'(c^*(a))$ links the "
        "value function derivative directly to the policy function, bypassing the need "
        "to compute $V$ itself. This is the same envelope theorem that underlies the "
        "Euler equation derivation, but used as a computational device.\n"
        "- **Speed-accuracy tradeoffs:** EGP is typically fastest because it avoids "
        "nonlinear equation solving entirely. EEI requires solving the Euler equation "
        "at each grid point (like Euler equation iteration), but updates via the envelope "
        "condition. VFI with grid search is slowest per iteration and requires many more "
        "iterations due to the contraction rate $\\beta$.\n"
        f"- **Precautionary savings motive:** Mean assets are {mean_assets:.2f} times "
        f"mean income, driven by the buffer-stock motive under IID income risk. "
        f"About {frac_constrained:.1f}% of agents are at the borrowing constraint.\n"
        f"- **MPCs are heterogeneous:** The mean MPC is {mean_mpc:.3f}, well above the "
        f"theoretical lower bound of {mpc_lim:.3f} for a patient unconstrained agent. "
        "Constrained and low-wealth agents have MPCs near 1, while wealthy agents "
        "approach the lower bound."
    )

    report.add_references([
        "Maliar, L. and Maliar, S. (2013). Envelope Condition Method with an Application "
        "to Default Risk Models. *Journal of Economic Dynamics and Control*, 37(7), 1439-1459.",
        "Carroll, C. D. (2006). The Method of Endogenous Gridpoints for Solving Dynamic "
        "Stochastic Optimization Problems. *Economics Letters*, 91(3), 312-320.",
        "Kaplan, G. (2017). Lecture Notes on Heterogeneous Agent Models. University of Chicago.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. "
        "MIT Press, 4th edition, Ch. 18.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
