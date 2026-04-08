#!/usr/bin/env python3
"""Endogenous Grid Points (EGP) Method with IID Income.

Solves the standard incomplete-markets savings problem using Carroll's (2006)
Endogenous Grid Points method. Instead of iterating on the value function with
an expensive inner maximization, EGP iterates on the Euler equation: given a
grid on SAVINGS (a'), it finds the IMPLIED current assets (a) from the first-
order condition. This avoids root-finding or grid search entirely, making EGP
much faster than standard VFI.

Reference: Carroll, C. D. (2006). "The Method of Endogenous Gridpoints for
Solving Dynamic Stochastic Optimization Problems." Economics Letters, 91(3).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.stats import norm

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    # Preferences
    risk_aver = 2       # CRRA coefficient
    beta = 0.95         # Discount factor

    # Returns
    r = 0.03            # Interest rate
    R = 1 + r           # Gross return

    # Income risk: discretized N(mu, sigma^2)
    mu_y = 1.0          # Mean income
    sd_y = 0.2          # Std dev of income
    ny = 5              # Number of income states

    # Asset grid
    na = 50             # Number of asset grid points
    amax = 50           # Maximum assets
    borrow_lim = 0      # Borrowing limit
    agrid_par = 0.5     # Grid curvature (1 = linear, <1 = more points near 0)

    # Computation
    max_iter = 1000     # Max Euler equation iterations
    tol_iter = 1.0e-6   # Convergence tolerance

    # Simulation
    Nsim = 50000        # Number of simulated agents
    Tsim = 500          # Simulation periods

    # MPC amounts
    mpc_amount_small = 1.0e-10  # Approximate theoretical MPC
    mpc_amount_large = 0.10     # ~$500 transfer

    # =========================================================================
    # Random draws (fixed seed for reproducibility)
    # =========================================================================
    np.random.seed(2020)
    yrand = np.random.rand(Nsim, Tsim)

    # =========================================================================
    # Asset grid (curved: denser near borrowing limit)
    # =========================================================================
    agrid = np.linspace(0, 1, na).reshape(na, 1)
    agrid = agrid ** (1 / agrid_par)
    agrid = borrow_lim + (amax - borrow_lim) * agrid

    # =========================================================================
    # Income grid: discretize normal distribution
    # =========================================================================
    # Inline discrete_normal using numpy/scipy (avoids JAX scalar conversion
    # issues when called inside fsolve).
    def discrete_normal_np(n, mu, sigma, width):
        """Equally spaced approximation to N(mu, sigma^2). Returns (error, grid, probs)."""
        x = np.linspace(mu - width * sigma, mu + width * sigma, n).reshape(n, 1)
        if n == 2:
            p = 0.5 * np.ones((n, 1))
        else:
            p = np.zeros((n, 1))
            p[0] = norm.cdf(x[0] + 0.5 * (x[1] - x[0]), mu, sigma)
            for i in range(1, n - 1):
                p[i] = (norm.cdf(x[i] + 0.5 * (x[i + 1] - x[i]), mu, sigma)
                        - norm.cdf(x[i] - 0.5 * (x[i] - x[i - 1]), mu, sigma))
            p[n - 1] = 1 - np.sum(p[:n - 1])
        ex = x.T @ p
        sdx = np.sqrt((x.T ** 2) @ p - ex ** 2)
        error = float((sdx - sigma).item())
        return error, x, p

    width = fsolve(lambda w: discrete_normal_np(ny, mu_y, sd_y, w)[0], 2.0)[0]
    _, ygrid, ydist = discrete_normal_np(ny, mu_y, sd_y, width)  # (ny,1) each
    ycumdist = np.cumsum(ydist)

    # =========================================================================
    # Utility function (CRRA) and its derivatives
    # =========================================================================
    u = lambda c: (c ** (1 - risk_aver) - 1) / (1 - risk_aver)
    u1 = lambda c: c ** (-risk_aver)               # Marginal utility
    u1inv = lambda mu: mu ** (-1 / risk_aver)       # Inverse marginal utility

    # =========================================================================
    # Initialize consumption function: consume all cash-on-hand
    # =========================================================================
    con = np.zeros((na, ny))
    for iy in range(ny):
        con[:, iy] = (r * agrid + ygrid[iy])[:, 0]

    # =========================================================================
    # Iterate on Euler equation with Endogenous Grid Points
    # =========================================================================
    # The key insight: given a grid on SAVINGS (a'), the Euler equation
    #   u'(c) = beta * R * E[u'(c')]
    # pins down consumption c, and the budget constraint
    #   a = (c + a' - y) / R
    # gives the IMPLIED current assets. No maximization needed!
    print("EGP: Iterating on Euler equation...")
    sav = np.zeros((na, ny))

    for iteration in range(1, max_iter + 1):
        conlast = con.copy()

        # Expected marginal utility tomorrow (using IID income)
        # conlast is c(a', y') for each (a', y'). Since income is IID,
        # E[u'(c')] = sum over y' of u'(c(a', y')) * prob(y')
        emuc = u1(conlast) @ ydist             # (na, 1)
        muc_today = beta * R * emuc             # Euler equation RHS
        con_today = u1inv(muc_today)            # Implied consumption today

        # For each income state, find implied current assets
        for iy in range(ny):
            # Endogenous grid: what assets TODAY imply choosing each a' point?
            # Budget: c + a' = R*a + y  =>  a = (c + a' - y) / R
            ass_implied = ((con_today + agrid - ygrid[iy]) / R)[:, 0]

            # Now interpolate: for each EXOGENOUS grid point a, find savings a'
            for ia in range(na):
                if agrid[ia] < ass_implied[0]:
                    # Borrowing constraint binds: agent wants to borrow
                    # but cannot, so saves at the limit
                    sav[ia, iy] = borrow_lim
                else:
                    # Interpolate endogenous grid -> exogenous grid
                    sav[ia, iy] = np.interp(agrid[ia, 0], ass_implied, agrid[:, 0])

            # Back out consumption from budget constraint
            con[:, iy] = (R * agrid + ygrid[iy])[:, 0] - sav[:, iy]

        cdiff = np.max(np.abs(con - conlast))
        if iteration % 50 == 0 or cdiff <= tol_iter:
            print(f"  Iteration {iteration:4d}, max consumption change = {cdiff:.2e}")
        if cdiff <= tol_iter:
            print(f"  Converged in {iteration} iterations (tol = {tol_iter:.0e})")
            break

    n_iterations = iteration

    # =========================================================================
    # Simulate panel of agents
    # =========================================================================
    print("\nSimulating panel of agents...")
    yindsim = np.zeros((Nsim, Tsim), dtype=int)
    asim = np.zeros((Nsim, Tsim))

    # Build interpolating functions for savings policy
    savinterp = []
    for iy in range(ny):
        savinterp.append(interp1d(agrid[:, 0], sav[:, iy], kind='linear'))

    for it in range(Tsim):
        if (it + 1) % 100 == 0:
            print(f"  Simulating period {it + 1}/{Tsim}")

        # Draw income states from CDF
        yindsim[yrand[:, it] <= ycumdist[0], it] = 0
        for iy in range(1, ny):
            yindsim[
                np.logical_and(yrand[:, it] > ycumdist[iy - 1],
                               yrand[:, it] <= ycumdist[iy]), it
            ] = iy

        # Asset choice for next period
        if it < Tsim - 1:
            for iy in range(ny):
                mask = yindsim[:, it] == iy
                asim[mask, it + 1] = savinterp[iy](asim[mask, it])

    # Actual income values
    ysim = ygrid[yindsim]

    # =========================================================================
    # Compute MPCs
    # =========================================================================
    print("\nComputing MPCs...")
    mpclim = R * ((beta * R) ** (-1 / risk_aver)) - 1  # Theoretical lower bound

    coninterp = []
    mpc_small = np.zeros((na, ny))
    mpc_large = np.zeros((na, ny))

    for iy in range(ny):
        coninterp.append(interp1d(agrid[:, 0], con[:, iy], kind='linear',
                                  fill_value='extrapolate'))
        mpc_small[:, iy] = (coninterp[iy](agrid[:, 0] + mpc_amount_small) - con[:, iy]) / mpc_amount_small
        mpc_large[:, iy] = (coninterp[iy](agrid[:, 0] + mpc_amount_large) - con[:, iy]) / mpc_amount_large

    # Simulated MPCs (at terminal period)
    mpc_sim_small = np.zeros(Nsim)
    mpc_sim_large = np.zeros(Nsim)
    for iy in range(ny):
        mask = yindsim[:, Tsim - 1] == iy
        a_terminal = asim[mask, Tsim - 1]
        mpc_sim_small[mask] = (coninterp[iy](a_terminal + mpc_amount_small)
                               - coninterp[iy](a_terminal)) / mpc_amount_small
        mpc_sim_large[mask] = (coninterp[iy](a_terminal + mpc_amount_large)
                               - coninterp[iy](a_terminal)) / mpc_amount_large

    # =========================================================================
    # Compute summary statistics
    # =========================================================================
    # Use terminal period for cross-sectional statistics
    a_final = asim[:, Tsim - 1]
    y_final = ysim[:, Tsim - 1, 0]  # squeeze the trailing dimension
    c_final = np.zeros(Nsim)
    for iy in range(ny):
        mask = yindsim[:, Tsim - 1] == iy
        c_final[mask] = coninterp[iy](a_final[mask])

    mean_assets = np.mean(a_final)
    mean_cons = np.mean(c_final)
    mean_mpc = np.mean(mpc_sim_large)
    frac_constrained = np.mean(a_final <= borrow_lim + 1e-6) * 100

    # Gini coefficient for wealth
    def gini(x):
        x_sorted = np.sort(x)
        n = len(x_sorted)
        cumx = np.cumsum(x_sorted)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else 0.0

    gini_wealth = gini(a_final)

    print(f"\n  Mean assets:              {mean_assets:.3f}")
    print(f"  Mean consumption:         {mean_cons:.3f}")
    print(f"  Gini (wealth):            {gini_wealth:.3f}")
    print(f"  Average MPC:              {mean_mpc:.3f}")
    print(f"  Fraction constrained:     {frac_constrained:.1f}%")
    print(f"  Theoretical MPC limit:    {mpclim:.4f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Endogenous Grid Points (EGP) with IID Income",
        "Carroll's (2006) endogenous grid points method for solving the "
        "standard incomplete-markets consumption-savings problem.",
    )

    report.add_overview(
        "The Endogenous Grid Points (EGP) method solves the household savings problem "
        "under income uncertainty using a computational trick that eliminates the "
        "expensive inner maximization of standard VFI. Instead of searching for optimal "
        "consumption at each asset grid point, EGP inverts the Euler equation: given a "
        "grid on *savings* $a'$, it computes the *implied* current assets $a$ that are "
        "consistent with choosing that level of savings.\n\n"
        "This is a partial-equilibrium model: the interest rate $r$ is exogenous. Agents "
        "face IID income risk, cannot borrow, and self-insure through precautionary savings."
    )

    report.add_equations(
        r"""
$$V(a, y) = \max_{c \ge 0} \left\{ u(c) + \beta \, \mathbb{E}\left[ V(a', y') \right] \right\}$$

subject to: $c + a' = Ra + y$, $\quad a' \ge 0$, $\quad y \sim F(y)$ IID.

**Euler equation:** $u'(c) = \beta R \, \mathbb{E}\left[ u'(c'(a', y')) \right]$ (with equality when $a' > 0$).

**EGP insight:** Fix a grid $\{a'_j\}$. For each $a'_j$:
1. Compute RHS: $\mu_j = \beta R \sum_{k} u'(c(a'_j, y_k)) \pi_k$
2. Invert: $c_j = (u')^{-1}(\mu_j)$
3. Recover implied assets: $a_j = (c_j + a'_j - y) / R$

This gives pairs $(a_j, a'_j)$ — the **endogenous grid** — from which we interpolate the savings policy on the original exogenous grid.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\gamma$ | {risk_aver} | CRRA risk aversion |\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $r$      | {r} | Interest rate |\n"
        f"| $\\mu_y$  | {mu_y} | Mean income |\n"
        f"| $\\sigma_y$ | {sd_y} | Std dev of income |\n"
        f"| $n_y$    | {ny} | Income grid points |\n"
        f"| $n_a$    | {na} | Asset grid points |\n"
        f"| $a_{{\\max}}$ | {amax} | Maximum assets |\n"
        f"| $N_{{sim}}$ | {Nsim:,} | Simulated agents |\n"
        f"| $T_{{sim}}$ | {Tsim} | Simulation periods |"
    )

    report.add_solution_method(
        "**Endogenous Grid Points (Carroll, 2006):** Instead of iterating on the "
        "value function with an inner maximization, EGP iterates on the *Euler equation*. "
        "At each iteration:\n\n"
        "1. Compute expected marginal utility $\\mathbb{E}[u'(c')]$ using the current "
        "consumption function and the IID income distribution.\n"
        "2. Apply the Euler equation to get today's consumption: "
        "$c = (u')^{-1}(\\beta R \\, \\mathbb{E}[u'(c')])$.\n"
        "3. Use the budget constraint to find the **implied** current assets: "
        "$a = (c + a' - y)/R$.\n"
        "4. Interpolate from the endogenous grid $(a, a')$ back to the exogenous grid.\n\n"
        "This avoids all root-finding and grid search in the inner loop, making EGP "
        "significantly faster than VFI — typically by an order of magnitude.\n\n"
        f"Converged in **{n_iterations} iterations** (tolerance = {tol_iter:.0e})."
    )

    # --- Figure 1: Consumption Policy by Income State ---
    fig1, ax1 = plt.subplots()
    ax1.plot(agrid, con[:, 0], 'b-', linewidth=2, label='Lowest income')
    ax1.plot(agrid, con[:, ny - 1], 'r-', linewidth=2, label='Highest income')
    for iy in range(1, ny - 1):
        ax1.plot(agrid, con[:, iy], color='gray', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('Assets $a$')
    ax1.set_ylabel('Consumption $c$')
    ax1.set_title('Consumption Policy Function')
    ax1.set_xlim(0, amax)
    ax1.legend()
    report.add_figure(
        "figures/consumption-policy.png",
        "Consumption policy by income state: higher income shifts the policy up",
        fig1,
        description="The kink at low asset levels marks where the borrowing constraint binds: "
        "constrained agents consume all available cash-on-hand. Above the kink, the Euler "
        "equation holds with equality and consumption rises smoothly with wealth.",
    )

    # --- Figure 2: Savings Policy ---
    fig2, ax2 = plt.subplots()
    ax2.plot(agrid, sav[:, 0] - agrid[:, 0], 'b-', linewidth=2, label='Lowest income')
    ax2.plot(agrid, sav[:, ny - 1] - agrid[:, 0], 'r-', linewidth=2, label='Highest income')
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_xlabel('Assets $a$')
    ax2.set_ylabel("Net savings $a' - a$")
    ax2.set_title('Savings Policy Function')
    ax2.set_xlim(0, amax)
    ax2.legend()
    report.add_figure(
        "figures/savings-policy.png",
        "Net savings (a'-a) by income state: low-income agents dissave, high-income agents accumulate",
        fig2,
        description="The flat region at the left where net savings equals negative current assets "
        "corresponds to constrained agents choosing the borrowing limit. The smooth curvature "
        "beyond that point is resolved without any root-finding, thanks to the EGP inversion.",
    )

    # --- Figure 3: Wealth Distribution ---
    fig3, ax3 = plt.subplots()
    ax3.hist(a_final, bins=100, density=True, color='steelblue', edgecolor='black',
             linewidth=0.3, alpha=0.8)
    ax3.set_xlabel('Assets $a$')
    ax3.set_ylabel('Density')
    ax3.set_title('Stationary Wealth Distribution')
    ax3.axvline(mean_assets, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {mean_assets:.2f}')
    ax3.legend()
    report.add_figure(
        "figures/wealth-distribution.png",
        "Simulated stationary wealth distribution with right skew and mass at the constraint",
        fig3,
        description="The spike near zero reflects agents who have been hit by bad income draws and "
        "are at or near the borrowing constraint. The long right tail captures agents who have "
        "accumulated a large buffer stock through a run of favorable income realizations.",
    )

    # --- Figure 4: MPC Distribution ---
    fig4, ax4 = plt.subplots()
    ax4.hist(mpc_sim_large, bins=np.linspace(0, 1.5, 76), density=True,
             color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.8)
    ax4.axvline(mpclim, color='red', linestyle=':', linewidth=1.5,
                label=f'Theoretical limit = {mpclim:.3f}')
    ax4.axvline(mean_mpc, color='orange', linestyle='--', linewidth=1.5,
                label=f'Mean MPC = {mean_mpc:.3f}')
    ax4.set_xlabel('MPC')
    ax4.set_ylabel('Density')
    ax4.set_title('Marginal Propensity to Consume Distribution')
    ax4.set_xlim(0, 1.5)
    ax4.legend()
    report.add_figure(
        "figures/mpc-distribution.png",
        "MPC distribution: constrained agents have MPC near 1, wealthy agents approach the theoretical limit",
        fig4,
        description="MPC heterogeneity is the central policy-relevant prediction of incomplete-markets "
        "models. Constrained households spend nearly every additional dollar, while wealthy "
        "households save most of it. This matters for fiscal multipliers and transfer targeting.",
    )

    # --- Table: Summary Statistics ---
    table_data = {
        "Statistic": [
            "Mean assets",
            "Mean consumption",
            "Gini coefficient (wealth)",
            "Average MPC (large transfer)",
            "Average MPC (small transfer)",
            "Fraction constrained",
            "Theoretical MPC limit",
        ],
        "Value": [
            f"{mean_assets:.3f}",
            f"{mean_cons:.3f}",
            f"{gini_wealth:.3f}",
            f"{mean_mpc:.3f}",
            f"{np.mean(mpc_sim_small):.3f}",
            f"{frac_constrained:.1f}%",
            f"{mpclim:.4f}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/summary-statistics.csv",
        "Summary Statistics from Simulated Stationary Distribution",
        df,
        description="The average MPC well above the theoretical lower bound reflects the large "
        "mass of constrained and near-constrained agents. The Gini coefficient captures "
        "the substantial wealth inequality generated by even IID income risk.",
    )

    report.add_takeaway(
        "The EGP method demonstrates that clever reformulation of the optimality "
        "conditions can yield dramatic computational gains without any approximation "
        "error. By inverting the Euler equation, we avoid the costly inner maximization "
        "of standard VFI.\n\n"
        "**Key insights:**\n"
        "- **Speed:** EGP converges in the same number of iterations as VFI but each "
        "iteration is much cheaper — no root-finding or grid search over consumption.\n"
        "- **Precautionary savings:** Under income uncertainty, agents accumulate a "
        "buffer stock of wealth even though $\\beta R < 1$. The borrowing constraint "
        "and prudence motive (convex marginal utility) drive this behavior.\n"
        "- **Wealth inequality:** The stationary distribution is right-skewed with a "
        "mass point at the borrowing constraint. Constrained agents have MPC near 1, "
        "while wealthy agents have MPC near the theoretical lower bound "
        f"$R(\\beta R)^{{-1/\\gamma}} - 1 \\approx {mpclim:.3f}$.\n"
        "- **MPC heterogeneity:** The average MPC is well above the representative-agent "
        "benchmark, driven by the large fraction of constrained or near-constrained "
        "households. This has important implications for fiscal policy: transfers are "
        "more stimulative when targeted at low-wealth households."
    )

    report.add_references([
        "Carroll, C. D. (2006). \"The Method of Endogenous Gridpoints for Solving "
        "Dynamic Stochastic Optimization Problems.\" *Economics Letters*, 91(3), 312-320.",
        "Deaton, A. (1991). \"Saving and Liquidity Constraints.\" *Econometrica*, 59(5), 1221-1248.",
        "Carroll, C. D. (1997). \"Buffer-Stock Saving and the Life Cycle/Permanent Income "
        "Hypothesis.\" *Quarterly Journal of Economics*, 112(1), 1-55.",
        "Kaplan, G. and Violante, G. L. (2022). \"The Marginal Propensity to Consume in "
        "Heterogeneous Agent Models.\" *Annual Review of Economics*, 14, 747-775.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
