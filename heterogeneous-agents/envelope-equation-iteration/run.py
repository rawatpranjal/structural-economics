#!/usr/bin/env python3
"""Envelope equation iteration for an IID income-risk saving problem.

The economic object is the same partial-equilibrium buffer-stock household
problem solved in neighboring tutorials. This version changes the object
iterated on: rather than updating the value level, it updates the derivative of
the income-integrated continuation value, W_a(a). The envelope condition is
W_a(a) = R * E_y[u'(c(a, y))].

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
from lib.plotting import setup_style
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
    # Let W(a) = E_y[V(a, y)] denote the value before next period's income draw.
    # The envelope theorem gives W_a(a) = R * E_y[u'(c(a, y))].
    # The Euler equation for each current (a, y) is
    #   u'(c) >= beta * W_a(a')  with equality if a' > borrow_lim
    #
    # Substituting the envelope condition into the Euler equation:
    #   u'(c(a,y)) = beta * R * E_y'[u'(c(a', y'))]    (unconstrained)
    #
    # This is the same Euler equation iteration as in the source code, but
    # we frame it through the lens of W_a(a):
    #
    # Algorithm:
    #   1. Guess consumption c_n(a, y) on the grid
    #   2. Compute W_{a,n}(a) = R * sum_y u'(c_n(a,y)) * prob(y)
    #   3. For each (a, y): use W_{a,n} to find c_{n+1} via Euler equation
    #      - If constrained: c = cash - borrow_lim
    #      - If unconstrained: u'(c) = beta * V'_n(a'), solve for c
    #   4. Check convergence of c
    #
    # The key insight: step 2 uses the ENVELOPE condition to compress the
    # consumption function c(a,y) into a single object W_a(a) that summarizes
    # all the information needed for the Euler equation. This avoids
    # interpolating ny separate consumption functions -- we only interpolate
    # the single function V'(a).
    # =========================================================================

    print("=" * 60)
    print("Method 1: Envelope Equation Iteration (EEI)")
    print("=" * 60)
    # Initial guess: consume income flow (hand-to-mouth)
    con_eei = np.zeros((na, ny))
    for iy in range(ny):
        con_eei[:, iy] = (r * agrid + ygrid[iy])[:, 0]
    sav_eei = np.zeros((na, ny))

    t0_eei = time.time()
    eei_errors = []

    for iteration in range(1, max_iter + 1):
        con_eei_last = con_eei.copy()

        # Step 1: compute W_a(a) from current consumption via the envelope condition
        # W_a(a) = R * E_y[u'(c(a,y))] = R * sum_y u'(c(a,y)) * prob(y)
        dV = R * (u1(con_eei_last) @ ydist)  # shape (na, 1)

        # Step 2: For each (a, y), solve for new consumption using W_a(a) in the
        # Euler equation: u'(c) = beta * W_a(a'), where a' = R*a + y - c
        for ia in range(na):
            for iy in range(ny):
                cash = R * agrid[ia, 0] + ygrid[iy, 0]

                # Check if borrowing constraint binds:
                # At constraint: a' = borrow_lim, c = cash - borrow_lim
                # Constraint binds if u'(cash - borrow_lim) >= beta * W_a(borrow_lim)
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
                    # Unconstrained: solve u'(c) = beta * W_a(cash - c)
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

    # Final W_a(a) for plotting
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

    def solve_egp_policy(asset_grid, label, log_every=None):
        """Solve the same household problem by EGP on the supplied asset grid."""
        n_assets = asset_grid.shape[0]
        con = np.zeros((n_assets, ny))
        for iy in range(ny):
            con[:, iy] = (r * asset_grid + ygrid[iy])[:, 0]

        t0 = time.time()
        errors = []
        sav = np.zeros((n_assets, ny))

        for iteration in range(1, max_iter + 1):
            con_last = con.copy()
            sav = np.zeros((n_assets, ny))

            emuc = u1(con_last) @ ydist
            muc_next = beta * R * emuc
            con_endo = u1inv(muc_next)

            for iy in range(ny):
                a_endo = ((con_endo + asset_grid - ygrid[iy]) / R)[:, 0]

                for ia in range(n_assets):
                    if asset_grid[ia, 0] < a_endo[0]:
                        sav[ia, iy] = borrow_lim
                    else:
                        sav[ia, iy] = lininterp1(
                            a_endo, asset_grid[:, 0], asset_grid[ia, 0]
                        )

                con[:, iy] = (R * asset_grid + ygrid[iy])[:, 0] - sav[:, iy]

            cdiff = np.max(np.abs(con - con_last))
            errors.append(cdiff)

            if log_every and (iteration % log_every == 0 or iteration == 1):
                print(f"  {label} iteration {iteration:4d}, max con diff = {cdiff:.2e}")

            if cdiff < tol_iter:
                print(f"  {label} converged in {iteration} iterations (error = {cdiff:.2e})")
                break

        elapsed = time.time() - t0
        print(f"  {label} time: {elapsed:.2f}s")
        return con, sav, iteration, elapsed, errors

    # =========================================================================
    # METHOD 3: Endogenous Grid Points (EGP) for comparison
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Method 3: Endogenous Grid Points (EGP)")
    print("=" * 60)

    con_egp, sav_egp, n_iter_egp, time_egp, egp_errors = solve_egp_policy(
        agrid, "EGP", log_every=10
    )

    # =========================================================================
    # Fine-grid reference policy
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Fine-grid EGP reference policy")
    print("=" * 60)

    na_ref = 600
    agrid_ref = np.linspace(0, 1, na_ref).reshape(na_ref, 1)
    agrid_ref = borrow_lim + (amax - borrow_lim) * agrid_ref ** (1 / agrid_par)
    con_ref, sav_ref, n_iter_ref, time_ref, _ = solve_egp_policy(
        agrid_ref, "Fine-grid EGP reference"
    )

    con_ref_on_main = np.zeros_like(con_eei)
    sav_ref_on_main = np.zeros_like(sav_eei)
    for iy in range(ny):
        con_ref_on_main[:, iy] = lininterp1(agrid_ref[:, 0], con_ref[:, iy], agrid[:, 0])
        sav_ref_on_main[:, iy] = lininterp1(agrid_ref[:, 0], sav_ref[:, iy], agrid[:, 0])

    accuracy_mask = agrid[:, 0] <= min(amax, 20.0)
    ref_c_gap = np.max(np.abs(con_eei[accuracy_mask, :] - con_ref_on_main[accuracy_mask, :]))
    ref_s_gap = np.max(np.abs(sav_eei[accuracy_mask, :] - sav_ref_on_main[accuracy_mask, :]))
    dV_ref = R * (u1(con_ref) @ ydist)

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
        "Envelope-Equation Iteration for Buffer-Stock Saving",
        "A partial-equilibrium income-risk household problem solved from marginal continuation values.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The household problem is the IID version of the buffer-stock saving "
        "environment used in the "
        "[endogenous-grid](../endogenous-grid-points/) tutorial and the broader "
        "[income-risk savings](../../dynamic-programming/consumption-savings/) "
        "benchmark. An impatient household cannot borrow below $\\underline a=0$, "
        "so assets are valuable because they insure consumption against bad income "
        "draws.\n\n"
        "This tutorial changes the computational object. EEI does not update the "
        "value level state by state, and it does not build the endogenous current "
        "asset grid used by EGP. It updates $W_a(a)$, the marginal value of entering "
        "next period with one more unit of assets before the next IID income draw is "
        "realized. Given that marginal continuation value, the Euler equation chooses "
        "current consumption; given the consumption rule, the envelope condition "
        "updates $W_a(a)$.\n\n"
        "The payoff from this example is conceptual as much as computational. Grid "
        "VFI, EGP, and EEI are three routes to the same buffer-stock saving policy. "
        "The run solves all three on the coarse grid and adds a fine-grid EGP "
        "reference so the plotted EEI policy can be checked against a more accurate "
        "Euler-equation solution."
    )

    report.add_equations(
        r"""
At the start of a period the household has assets $a \in A$ and observes income
$y_j$ drawn from an IID discrete distribution with probabilities $\pi_j$. Let
$V(a,y_j)$ be the value after the current income draw and let

$$
W(a)=\sum_{j=1}^{n_y}\pi_j V(a,y_j)
$$

be the income-integrated value used for continuation payoffs. The household
chooses next-period assets $a'=g(a,y_j)$:

$$
V(a,y_j)=
\max_{a' \geq \underline a}
\{u(Ra+y_j-a')+\beta W(a')\}.
$$

The budget identity is

$$
c(a,y_j)=Ra+y_j-g(a,y_j),
\qquad R=1+r.
$$

Preferences are CRRA,

$$
u(c)=\frac{c^{1-\gamma}-1}{1-\gamma},
\qquad
u'(c)=c^{-\gamma}.
$$

For an interior saving choice, the Euler equation is

$$
u'(c(a,y_j))=\beta W_a(g(a,y_j)).
$$

The envelope condition links the marginal continuation value to the policy:

$$
W_a(a)
=\sum_{j=1}^{n_y}\pi_j V_a(a,y_j)
=R\sum_{j=1}^{n_y}\pi_j u'(c(a,y_j)).
$$

At the borrowing limit, the Euler equality becomes the inequality
$u'(Ra+y_j-\underline a)\geq \beta W_a(\underline a)$. This is the same
liquidity constraint that gives low-asset households high marginal propensities
to consume.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| $\\gamma$ | {risk_aver} | CRRA risk aversion |\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $r$ | {r} | Net risk-free return |\n"
        f"| $\\beta R$ | {beta * R:.4f} | Patience-return product |\n"
        f"| $\\mu_y$ | {mu_y} | Mean labor income |\n"
        f"| $\\sigma_y$ | {sd_y} | Income standard deviation |\n"
        f"| $n_y$ | {ny} | IID income states |\n"
        f"| $\\underline{{a}}$ | {borrow_lim} | Borrowing limit |\n"
        f"| $\\bar a$ | {amax} | Upper asset-grid bound |\n"
        f"| EEI asset grid | {na} points | Power spacing near $\\underline a$ |\n"
        f"| Reference asset grid | {na_ref} points | Fine-grid EGP policy check |\n"
        f"| Simulation | {Nsim:,} households, {Tsim} periods | Terminal cross section |"
    )

    report.add_solution_method(
        "EEI keeps the Euler equation visible. The current guess is a consumption "
        "policy. The envelope equation turns that policy into a one-dimensional "
        "marginal continuation value $W_a(a)$, and the next Euler step turns it "
        "back into a consumption policy.\n\n"
        "```text\n"
        "Input: asset grid A, income states y_j, probabilities pi_j,\n"
        "       primitives beta, R, gamma, borrowing limit a_min\n"
        "Initialize c_0(a_i,y_j), for example from current income plus interest\n"
        "For n = 0, 1, 2, ...:\n"
        "    Compute W_{a,n}(a_i) = R sum_j pi_j u'(c_n(a_i,y_j))\n"
        "    For each current asset a_i and income y_j:\n"
        "        Set cash = R a_i + y_j\n"
        "        If u'(cash-a_min) >= beta W_{a,n}(a_min):\n"
        "            Set g_{n+1}(a_i,y_j) = a_min\n"
        "            Set c_{n+1}(a_i,y_j) = cash - a_min\n"
        "        Otherwise find c in (0, cash-a_min) such that\n"
        "            u'(c) = beta W_{a,n}(cash-c)\n"
        "            and set g_{n+1}(a_i,y_j) = cash - c\n"
        "    Stop when max_{i,j} |c_{n+1}(a_i,y_j)-c_n(a_i,y_j)| < epsilon\n"
        "Output: consumption policy c, savings policy g, marginal value W_a\n"
        "```\n\n"
        f"The coarse-grid EEI solve converged in **{n_iter_eei} iterations** "
        f"with a consumption sup-norm error below $10^{{-6}}$. The same grid took "
        f"{n_iter_egp} EGP iterations and {n_iter_vfi} grid-VFI iterations. Those "
        "iteration counts compare fixed-point objects, not optimized library "
        "implementations: this EEI code uses bisection at every state to make the "
        "Euler step transparent, while EGP avoids those one-dimensional solves.\n\n"
        f"A {na_ref}-point EGP solve provides the fine-grid reference. On the "
        f"plotted asset range $a\\leq 20$, the coarse EEI policy is within "
        f"{ref_c_gap:.2e} in consumption and {ref_s_gap:.2e} in next assets. "
        "Those gaps are grid and interpolation errors, not a different economic "
        "mechanism."
    )

    # --- Figure 1: Consumption Policy Function ---
    fig1, ax1 = plt.subplots()
    mid = ny // 2
    ax1.plot(agrid, con_eei[:, 0], color='steelblue', linewidth=2,
             label=f'Lowest income ($y={ygrid[0,0]:.2f}$)')
    ax1.plot(agrid, con_eei[:, mid], color='seagreen', linewidth=2,
             label=f'Middle income ($y={ygrid[mid,0]:.2f}$)')
    ax1.plot(agrid, con_eei[:, ny - 1], color='indianred', linewidth=2,
             label=f'Highest income ($y={ygrid[ny-1,0]:.2f}$)')
    ax1.plot(agrid_ref, con_ref[:, 0], color='steelblue', linewidth=1.3,
             linestyle='--', alpha=0.75, label='Fine-grid ref, low income')
    ax1.plot(agrid_ref, con_ref[:, ny - 1], color='indianred', linewidth=1.3,
             linestyle='--', alpha=0.75, label='Fine-grid ref, high income')
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("Consumption $c(a,y_j)$")
    ax1.set_title("Consumption Policy and Fine-Grid Reference")
    ax1.legend()
    ax1.set_xlim(0, min(amax, 20))
    report.add_figure("figures/consumption-policy.png",
                       "EEI consumption policy with fine-grid EGP reference", fig1,
        description="The consumption policy has the usual buffer-stock shape. Low-asset "
        "households consume a large share of cash on hand, but they still save after "
        "middle and high income draws because tomorrow's draw may be bad. The dashed "
        f"fine-grid reference nearly overlays the coarse EEI solution; the maximum "
        f"consumption gap over $a\\leq 20$ is {ref_c_gap:.2e}.")

    # --- Figure 2: Marginal Continuation Value W_a(a) ---
    fig2, ax2 = plt.subplots()
    ax2.plot(agrid, dV, color='navy', linewidth=2, label="$W_a(a)$, EEI")
    ax2.plot(agrid_ref, dV_ref, color='black', linestyle='--', linewidth=1.5,
             alpha=0.8, label="$W_a(a)$, fine-grid ref")
    for iy, color, label_str in [
            (0, 'steelblue', r"$R \cdot u'(c(a, y_{\mathrm{low}}))$"),
            (ny - 1, 'indianred', r"$R \cdot u'(c(a, y_{\mathrm{high}}))$")]:
        ax2.plot(agrid, R * u1(con_eei[:, iy]), '--', color=color,
                 linewidth=1.2, alpha=0.55, label=label_str)
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("$W_a(a)$")
    ax2.set_title("Marginal Continuation Value")
    ax2.legend()
    ax2.set_xlim(0, min(amax, 20))
    ax2.set_ylim(0, min(float(dV[0, 0]) * 1.5, float(np.max(dV) * 1.2)))
    report.add_figure("figures/value-derivative.png",
                       "Marginal continuation value from EEI and fine-grid reference", fig2,
        description="This is the object EEI updates. $W_a(a)$ is steep near the "
        "borrowing limit because an extra dollar is most valuable when the household "
        "has little self-insurance. The lighter curves show the state-specific "
        "marginal utilities that the envelope condition averages across income states.")

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
                       "Simulated terminal wealth distribution under the EEI policy", fig3,
        description="The terminal cross section is right-skewed, but modest. This is "
        "still the IID income benchmark: households build buffer stocks, many remain "
        "close to the borrowing constraint, and the right tail mostly reflects long "
        "runs of favorable income draws rather than persistent earnings types.")

    # --- Figure 4: Convergence Comparison ---
    fig4, ax4 = plt.subplots()
    ax4.semilogy(range(1, len(eei_errors) + 1), eei_errors, 'b-', linewidth=2,
                 label=f'EEI ({n_iter_eei} iter)')
    ax4.semilogy(range(1, len(egp_errors) + 1), egp_errors, 'r-', linewidth=2,
                 label=f'EGP ({n_iter_egp} iter)')
    ax4.semilogy(range(1, len(vfi_errors) + 1), vfi_errors, 'k-', linewidth=2,
                 alpha=0.7, label=f'VFI ({n_iter_vfi} iter)')
    ax4.axhline(tol_iter, color='gray', linewidth=1, linestyle=':', label=f'Tolerance = {tol_iter:.0e}')
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Maximum error (log scale)")
    ax4.set_title("Convergence Speed: EEI vs EGP vs VFI")
    ax4.legend()
    ax4.set_xlim(0, max(len(eei_errors), len(egp_errors), min(len(vfi_errors), 500)))
    report.add_figure("figures/convergence-comparison.png",
                       "Convergence paths for EEI, EGP, and grid VFI", fig4,
        description="The convergence plot should be read by fixed-point object. VFI "
        "contracts in value levels. EGP and EEI work with Euler-equation objects, so "
        "their errors shrink in consumption-policy space. This transparent EEI "
        "implementation uses bisection at every state; the figure's main point is the "
        "different updating equation, not a timing race.")

    # --- Table: Solution Statistics ---
    table_data = {
        "Statistic": [
            "EEI iterations",
            "Same-grid EGP iterations",
            "Same-grid VFI iterations",
            "Fine-grid reference points",
            "Fine-grid reference iterations",
            "Max consumption gap vs reference, a <= 20",
            "Max next-asset gap vs reference, a <= 20",
            "Mean assets / mean income",
            "Fraction at borrowing limit",
            "Mean MPC, 0.10 transfer",
            "Perfect-foresight MPC limit",
            "10th percentile wealth",
            "50th percentile wealth",
            "90th percentile wealth",
        ],
        "Value": [
            f"{n_iter_eei}",
            f"{n_iter_egp}",
            f"{n_iter_vfi}",
            f"{na_ref}",
            f"{n_iter_ref}",
            f"{ref_c_gap:.2e}",
            f"{ref_s_gap:.2e}",
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
    report.add_table("tables/solution-statistics.csv", "Solution and Simulation Summary", df,
        description="The table separates economic output from numerical diagnostics. The "
        "asset distribution and MPCs come from simulating the EEI policy. The fine-grid "
        "rows compare the coarse EEI policy with a denser Euler-equation reference.")

    report.add_takeaway(
        "EEI is not a new savings model. It is a different fixed point for the same "
        "incomplete-markets household problem. The economic policy is still the "
        "buffer-stock rule: low income draws push households toward the asset floor, "
        "good draws rebuild wealth, and low-wealth households have high MPCs.\n\n"
        "The computational lesson is that the envelope condition can be used "
        "as an updating equation, not only as a theorem behind the Euler equation. "
        "By carrying $W_a(a)$ forward, EEI avoids value levels and keeps the marginal "
        "value of self-insurance in view. In this transparent implementation, EGP is "
        "faster, but EEI gives a clean way to see why the policy is governed by "
        "marginal continuation values rather than by the value function level itself."
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
