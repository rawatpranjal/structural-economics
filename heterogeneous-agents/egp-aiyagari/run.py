#!/usr/bin/env python3
"""EGP-Aiyagari: Endogenous Grid Points with Aiyagari General Equilibrium.

Solves the Aiyagari (1994) heterogeneous-agent model using the Endogenous Grid
Points (EGP) method of Carroll (2006) for the inner household problem, embedded
in an iterative loop over the capital-labor ratio to clear the capital market.

The EGP method inverts the Euler equation to find today's assets as a function
of tomorrow's assets, avoiding the costly root-finding step of VFI. This speed
advantage makes EGP the preferred inner solver for GE loops.

References:
    Aiyagari (1994), "Uninsured Idiosyncratic Risk and Aggregate Saving", QJE.
    Carroll (2006), "The Method of Endogenous Gridpoints", Economics Letters.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


# =============================================================================
# Helper functions
# =============================================================================

def discrete_normal(n, mu, sigma, width):
    """Equally spaced approximation to a normal distribution."""
    x = np.linspace(mu - width * sigma, mu + width * sigma, n).reshape(n, 1)
    if n == 2:
        p = 0.5 * np.ones((n, 1))
    else:
        p = np.zeros((n, 1))
        p[0] = norm.cdf(x[0, 0] + 0.5 * (x[1, 0] - x[0, 0]), mu, sigma)
        for i in range(1, n - 1):
            p[i] = (norm.cdf(x[i, 0] + 0.5 * (x[i + 1, 0] - x[i, 0]), mu, sigma)
                     - norm.cdf(x[i, 0] - 0.5 * (x[i, 0] - x[i - 1, 0]), mu, sigma))
        p[n - 1] = 1 - np.sum(p[:n - 1])
    ex = (x.T @ p)[0, 0]
    sdx = np.sqrt(((x.T ** 2) @ p)[0, 0] - ex ** 2)
    return sdx - sigma, x, p


def gini_coefficient(data):
    """Compute the Gini coefficient for a 1-D array of non-negative values."""
    data = np.sort(np.asarray(data, dtype=float).ravel())
    data = data[~np.isnan(data)]
    n = len(data)
    if n < 2 or np.sum(data) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * data) - (n + 1) * np.sum(data)) / (n * np.sum(data)))


def lorenz_curve(data):
    """Return (cumulative population share, cumulative wealth share)."""
    data = np.sort(np.asarray(data, dtype=float).ravel())
    cum_wealth = np.cumsum(data)
    cum_wealth = cum_wealth / cum_wealth[-1]
    cum_pop = np.arange(1, len(data) + 1) / len(data)
    return np.concatenate([[0], cum_pop]), np.concatenate([[0], cum_wealth])


def solve_egp_inner(agrid_flat, ygrid, ydist, na, ny, beta, R, wage, yscale,
                    risk_aver, borrow_lim, con_init, max_iter, tol_iter):
    """Solve individual problem via EGP. Returns (con, sav, iterations, error)."""
    u1 = lambda c: c ** (-risk_aver)
    u1inv = lambda uc: uc ** (-1.0 / risk_aver)

    con = con_init.copy()
    agrid_col = agrid_flat.reshape(-1, 1)

    for iteration in range(1, max_iter + 1):
        conlast = con.copy()

        # Expected marginal utility (IID)
        emuc = u1(conlast) @ ydist       # (na, 1)
        muc1 = beta * R * emuc
        con1 = u1inv(muc1)               # (na, 1)

        sav = np.zeros((na, ny))
        for iy in range(ny):
            # Endogenous grid
            ass1 = ((con1[:, 0] + agrid_flat - wage * yscale * ygrid[iy, 0]) / R)

            # Vectorized interpolation: for points below ass1[0], constrained
            sav_iy = np.interp(agrid_flat, ass1, agrid_flat)
            constrained = agrid_flat < ass1[0]
            sav_iy[constrained] = borrow_lim
            sav[:, iy] = sav_iy

            con[:, iy] = R * agrid_flat + wage * yscale * ygrid[iy, 0] - sav_iy

        cdiff = np.max(np.abs(con - conlast))
        if cdiff < tol_iter:
            break

    return con, sav, iteration, cdiff


def simulate_economy(sav, agrid_flat, yindsim, ny, borrow_lim, Nsim, Tsim,
                     asim_init=None):
    """Simulate economy using vectorized np.interp. Returns asim array."""
    asim = np.zeros((Nsim, Tsim))
    if asim_init is not None:
        asim[:, 0] = asim_init
    else:
        asim[:, 0] = 0.0

    # Pre-extract savings columns
    sav_cols = [sav[:, iy] for iy in range(ny)]

    for it in range(Tsim - 1):
        yind_t = yindsim[:, it]
        a_t = asim[:, it]
        a_next = np.empty(Nsim)
        for iy in range(ny):
            mask = yind_t == iy
            if np.any(mask):
                a_next[mask] = np.interp(a_t[mask], agrid_flat, sav_cols[iy])
        np.maximum(a_next, borrow_lim, out=a_next)
        asim[:, it + 1] = a_next

    return asim


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    beta = 0.96
    risk_aver = 2
    alpha = 0.36
    delta = 0.08

    mu_y = 1.0
    sd_y = 0.2
    ny = 5

    na = 100
    amax = 50.0
    borrow_lim = 0.0
    agrid_par = 0.4

    max_iter = 1000
    tol_iter = 1.0e-6

    Nsim = 50_000
    Tsim = 300

    max_iter_ge = 80
    tol_ge = 1.0e-4
    step_ge = 0.01

    # =========================================================================
    # Setup grids
    # =========================================================================
    agrid = np.linspace(0, 1, na) ** (1 / agrid_par)
    agrid = borrow_lim + (amax - borrow_lim) * agrid
    agrid_flat = agrid  # 1-D array

    width_guess = 2.0
    width = fsolve(lambda x: discrete_normal(ny, mu_y, sd_y, x)[0], width_guess)[0]
    _, ygrid, ydist = discrete_normal(ny, mu_y, sd_y, width)
    ycumdist = np.cumsum(ydist.ravel())

    # =========================================================================
    # Pre-simulate income draws
    # =========================================================================
    print("Pre-simulating income realizations...")
    np.random.seed(2024)
    yrand = np.random.rand(Nsim, Tsim)

    yindsim = np.zeros((Nsim, Tsim), dtype=int)
    for it in range(Tsim):
        yindsim[yrand[:, it] <= ycumdist[0], it] = 0
        for iy in range(1, ny):
            mask = (yrand[:, it] > ycumdist[iy - 1]) & (yrand[:, it] <= ycumdist[iy])
            yindsim[mask, it] = iy

    ysim_flat = ygrid[yindsim, 0]  # (Nsim, Tsim)

    # =========================================================================
    # Outer GE loop: iterate on K/L ratio
    # =========================================================================
    print("=" * 60)
    print("EGP-Aiyagari: K/L ratio iteration")
    print("=" * 60)

    r_guess = 1.0 / beta - 1.0 - 0.005
    KLratio = ((r_guess + delta) / alpha) ** (1 / (alpha - 1))

    r_trace = []
    Ks_trace = []
    Kd_trace = []

    con = None
    asim_last = None

    for ge_iter in range(1, max_iter_ge + 1):
        r = alpha * (KLratio ** (alpha - 1)) - delta
        R = 1 + r
        wage = (1 - alpha) * (KLratio ** alpha)
        yscale = (KLratio ** (-alpha)) / (ygrid.T @ ydist)[0, 0]

        # Initialize consumption guess
        if con is None:
            con = np.zeros((na, ny))
            for iy in range(ny):
                con[:, iy] = np.maximum(
                    r * agrid_flat + wage * yscale * ygrid[iy, 0], 1e-10)

        # Inner EGP solve
        con, sav, egp_it, egp_err = solve_egp_inner(
            agrid_flat, ygrid, ydist, na, ny, beta, R, wage, yscale,
            risk_aver, borrow_lim, con, max_iter, tol_iter)

        # Simulate
        asim = simulate_economy(sav, agrid_flat, yindsim, ny, borrow_lim,
                                Nsim, Tsim, asim_init=asim_last)
        asim_last = asim[:, -1].copy()

        Ea = np.mean(asim[:, -1])
        L = yscale * np.mean(ysim_flat[:, -1])
        KLratio_new = Ea / L

        r_trace.append(r)
        Ks_trace.append(Ea)
        Kd_trace.append(KLratio * L)

        KLdiff = KLratio_new / KLratio - 1
        print(f"  GE iter {ge_iter:3d}: r = {r:.6f}, "
              f"K_supply = {Ea:.4f}, K_demand = {KLratio * L:.4f}, "
              f"KL diff = {KLdiff * 100:.4f}%")

        if abs(KLdiff) < tol_ge:
            print(f"\n  Converged in {ge_iter} iterations!")
            break

        KLratio = (1 - step_ge) * KLratio + step_ge * KLratio_new

    # =========================================================================
    # Equilibrium values
    # =========================================================================
    r_eq = r
    R_eq = R
    w_eq = wage
    K_eq = Ea
    L_eq = L
    Y_eq = K_eq ** alpha * L_eq ** (1 - alpha)
    egp_iters = egp_it
    ge_iters = ge_iter

    # =========================================================================
    # Compute capital supply/demand curves for the plot
    # =========================================================================
    print("\nComputing capital supply/demand curves for plotting...")
    # Range centered around equilibrium for clear crossing
    r_max_plot = min(1.0 / beta - 1.0 - 0.0002, r_eq + 0.002)
    r_min_plot = r_eq - 0.01
    r_plot_vals = np.linspace(r_min_plot, r_max_plot, 10)
    Ks_plot = []
    Kd_plot = []

    # Use equilibrium wealth as starting point for supply simulations
    wealth_eq_init = asim[:, -1].copy()

    for rp in r_plot_vals:
        KLr = ((rp + delta) / alpha) ** (1 / (alpha - 1))
        wp = (1 - alpha) * (KLr ** alpha)
        ysc = (KLr ** (-alpha)) / (ygrid.T @ ydist)[0, 0]
        Rp = 1 + rp

        # Initialize and solve EGP
        con_p = np.zeros((na, ny))
        for iy in range(ny):
            con_p[:, iy] = np.maximum(
                rp * agrid_flat + wp * ysc * ygrid[iy, 0], 1e-10)

        con_p, sav_p, _, _ = solve_egp_inner(
            agrid_flat, ygrid, ydist, na, ny, beta, Rp, wp, ysc,
            risk_aver, borrow_lim, con_p, max_iter, tol_iter)

        # Simulate starting from equilibrium wealth
        asim_p = simulate_economy(sav_p, agrid_flat, yindsim, ny, borrow_lim,
                                  Nsim, Tsim, asim_init=wealth_eq_init)

        Ea_p = np.mean(asim_p[:, -1])
        L_p = ysc * np.mean(ysim_flat[:, -1])
        Kd_p = KLr * L_p
        Ks_plot.append(Ea_p)
        Kd_plot.append(Kd_p)
        print(f"  r = {rp:.5f}: K_supply = {Ea_p:.3f}, K_demand = {Kd_p:.3f}")

    # =========================================================================
    # Wealth distribution statistics
    # =========================================================================
    wealth_eq = asim[:, -1]
    gini_w = gini_coefficient(wealth_eq)
    lorenz_pop, lorenz_w = lorenz_curve(wealth_eq)
    frac_constrained = np.mean(wealth_eq <= borrow_lim + 1e-6)
    pct_10 = np.percentile(wealth_eq, 10)
    pct_50 = np.percentile(wealth_eq, 50)
    pct_90 = np.percentile(wealth_eq, 90)
    pct_99 = np.percentile(wealth_eq, 99)

    # MPC out of a small windfall
    epsilon_a = 0.01
    n_mpc = min(Nsim, 10000)
    mpc_sim = np.zeros(n_mpc)
    for i_agent in range(n_mpc):
        iy = yindsim[i_agent, -1]
        a_curr = wealth_eq[i_agent]
        s0 = np.interp(a_curr, agrid_flat, sav[:, iy])
        s1 = np.interp(a_curr + epsilon_a, agrid_flat, sav[:, iy])
        c0 = R_eq * a_curr + w_eq * yscale * ygrid[iy, 0] - s0
        c1 = R_eq * (a_curr + epsilon_a) + w_eq * yscale * ygrid[iy, 0] - s1
        mpc_sim[i_agent] = (c1 - c0) / epsilon_a
    mean_mpc = float(np.mean(mpc_sim))

    print(f"\n{'=' * 60}")
    print("Equilibrium Results")
    print(f"{'=' * 60}")
    print(f"  Interest rate r  = {r_eq:.6f}")
    print(f"  Wage w           = {w_eq:.4f}")
    print(f"  Capital K        = {K_eq:.4f}")
    print(f"  Output Y         = {Y_eq:.4f}")
    print(f"  Wealth Gini      = {gini_w:.4f}")
    print(f"  Mean MPC         = {mean_mpc:.4f}")
    print(f"  Frac constrained = {frac_constrained:.4f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "EGP-Aiyagari Model",
        "Aiyagari (1994) general equilibrium with Endogenous Grid Points for the household problem.",
    )

    report.add_overview(
        "The Aiyagari model is the workhorse framework for studying how idiosyncratic "
        "income risk and incomplete markets shape the wealth distribution and aggregate "
        "capital accumulation in general equilibrium.\n\n"
        "Households face uninsurable IID income shocks and self-insure by accumulating "
        "a risk-free asset (capital). A representative firm rents capital and labor in "
        "competitive factor markets. The equilibrium interest rate clears the capital "
        "market: aggregate household savings must equal the firm's capital demand.\n\n"
        "We solve the household problem using the Endogenous Grid Points (EGP) method, "
        "which is dramatically faster than VFI because it avoids root-finding. This "
        "speed advantage is critical in the GE loop, where the household problem must "
        "be solved many times at different interest rates."
    )

    report.add_equations(
        r"""
**Household problem:**
$$V(a, y) = \max_{c, a'} \left\{ u(c) + \beta \, \mathbb{E}[V(a', y')] \right\}$$
$$\text{s.t.} \quad c + a' = (1+r)a + wy, \quad a' \ge 0$$

**EGP Euler equation inversion:**
$$u'(c_t) = \beta (1+r) \, \mathbb{E}[u'(c_{t+1})]$$
$$c_t = (u')^{-1}\left(\beta (1+r) \, \mathbb{E}[u'(c_{t+1})]\right)$$
$$a_t = \frac{c_t + a_{t+1} - wy}{1+r} \quad \text{(endogenous grid)}$$

**Firm problem:**
$$r = \alpha K^{\alpha-1} L^{1-\alpha} - \delta, \qquad w = (1-\alpha) K^{\alpha} L^{-\alpha}$$

**Capital market clearing:**
$$\int a \, d\mu(a, y) = K$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $\\sigma$ | {risk_aver} | CRRA risk aversion |\n"
        f"| $\\alpha$ | {alpha} | Capital share |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\mu_y$  | {mu_y} | Mean income |\n"
        f"| $\\sigma_y$ | {sd_y} | Std dev of income |\n"
        f"| Income states | {ny} | IID normal discretization |\n"
        f"| Asset grid | {na} points | $a \\in [{borrow_lim}, {amax}]$ |"
    )

    report.add_solution_method(
        "**Two-loop structure:**\n\n"
        "1. **Outer loop (K/L ratio iteration):** Starting from an initial guess for the "
        "capital-labor ratio, compute prices ($r$, $w$), solve the household problem, "
        "simulate the economy, and update the K/L ratio with dampening toward the "
        "simulated aggregate.\n\n"
        "2. **Inner loop (EGP):** For given prices, iterate on the Euler equation using "
        "the endogenous grid points method. Instead of searching for optimal savings at "
        "each grid point (as in VFI), EGP inverts the Euler equation to find the "
        "*current* assets that rationalize each *future* asset choice. This avoids "
        "root-finding entirely.\n\n"
        f"The inner EGP loop converged in **{egp_iters} iterations**. "
        f"The outer K/L iteration converged in **{ge_iters} iterations** "
        f"(tolerance = {tol_ge:.0e})."
    )

    # --- Figure 1: Consumption policy at equilibrium ---
    fig1, ax1 = plt.subplots()
    xlim_plot = min(15, amax)
    mask_plot = agrid_flat <= xlim_plot
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, ny))
    for iy in range(ny):
        label = f"y = {ygrid[iy, 0]:.2f} (p = {ydist[iy, 0]:.3f})"
        ax1.plot(agrid_flat[mask_plot], con[mask_plot, iy], color=colors[iy],
                 linewidth=2, label=label)
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("Consumption $c$")
    ax1.set_title("Consumption Policy at Equilibrium")
    ax1.legend(fontsize=8)
    report.add_figure("figures/consumption-policy.png",
                      "Consumption policy functions at equilibrium for each income state",
                      fig1)

    # --- Figure 2: Capital supply vs demand ---
    fig2, ax2 = plt.subplots()
    ax2.plot(Ks_plot, r_plot_vals, 'b-o', linewidth=2, markersize=4,
             label='Capital supply (savings)')
    ax2.plot(Kd_plot, r_plot_vals, 'r-s', linewidth=2, markersize=4,
             label='Capital demand (firm FOC)')
    ax2.axhline(r_eq, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f'$r^* = {r_eq:.4f}$')
    ax2.set_xlabel("Capital $K$")
    ax2.set_ylabel("Interest rate $r$")
    ax2.set_title("Capital Market: Supply vs Demand")
    ax2.legend()
    report.add_figure("figures/capital-supply-demand.png",
                      "Capital supply (household savings) and demand (firm FOC) as functions of r",
                      fig2)

    # --- Figure 3: Wealth distribution ---
    fig3, ax3 = plt.subplots()
    ax3.hist(wealth_eq, bins=100, density=True, color='steelblue',
             edgecolor='white', alpha=0.8)
    ax3.axvline(np.mean(wealth_eq), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {np.mean(wealth_eq):.2f}')
    ax3.axvline(np.median(wealth_eq), color='orange', linestyle='--', linewidth=1.5,
                label=f'Median = {np.median(wealth_eq):.2f}')
    ax3.set_xlabel("Wealth $a$")
    ax3.set_ylabel("Density")
    ax3.set_title("Equilibrium Wealth Distribution")
    ax3.legend()
    ax3.set_xlim(0, min(np.percentile(wealth_eq, 99.5), amax))
    report.add_figure("figures/wealth-distribution.png",
                      "Stationary wealth distribution in equilibrium",
                      fig3)

    # --- Figure 4: Lorenz curve ---
    fig4, ax4 = plt.subplots()
    ax4.plot(lorenz_pop, lorenz_w, 'b-', linewidth=2, label='Lorenz curve')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect equality')
    ax4.fill_between(lorenz_pop, lorenz_w, lorenz_pop, alpha=0.15, color='blue')
    ax4.set_xlabel("Cumulative population share")
    ax4.set_ylabel("Cumulative wealth share")
    ax4.set_title(f"Lorenz Curve (Gini = {gini_w:.3f})")
    ax4.legend()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    report.add_figure("figures/lorenz-curve.png",
                      f"Lorenz curve for wealth distribution (Gini = {gini_w:.3f})",
                      fig4)

    # --- Table: Equilibrium statistics ---
    table_data = {
        "Statistic": [
            "Interest rate r",
            "Wage w",
            "Aggregate capital K",
            "Output Y",
            "Capital-output ratio K/Y",
            "Wealth Gini",
            "Mean MPC (windfall)",
            "Fraction constrained",
            "10th percentile wealth",
            "50th percentile wealth",
            "90th percentile wealth",
            "99th percentile wealth",
        ],
        "Value": [
            f"{r_eq:.6f}",
            f"{w_eq:.4f}",
            f"{K_eq:.4f}",
            f"{Y_eq:.4f}",
            f"{K_eq / Y_eq:.4f}" if Y_eq > 0 else "N/A",
            f"{gini_w:.4f}",
            f"{mean_mpc:.4f}",
            f"{frac_constrained:.4f}",
            f"{pct_10:.4f}",
            f"{pct_50:.4f}",
            f"{pct_90:.4f}",
            f"{pct_99:.4f}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/equilibrium.csv", "Equilibrium Statistics", df)

    report.add_takeaway(
        "The Aiyagari model demonstrates how precautionary savings demand from "
        "uninsurable income risk drives the equilibrium interest rate below the "
        "rate of time preference ($r^* < 1/\\beta - 1$).\n\n"
        "**Key insights:**\n"
        "- Households over-accumulate assets as a buffer against bad income shocks. "
        "This *precautionary savings motive* pushes the capital stock above the "
        "representative-agent level and the interest rate below $1/\\beta - 1$.\n"
        "- The wealth distribution is right-skewed and exhibits a Gini coefficient "
        f"of {gini_w:.3f}. The borrowing constraint binds for {frac_constrained*100:.1f}% "
        "of households.\n"
        "- The EGP method makes the GE loop feasible: each inner solve takes only "
        f"{egp_iters} Euler-equation iterations (vs. hundreds with VFI), "
        "enabling rapid iteration over the capital-labor ratio.\n"
        "- The mean MPC out of a small windfall is "
        f"{mean_mpc:.3f}, reflecting the heterogeneity in marginal propensities to consume "
        "across the wealth distribution."
    )

    report.add_references([
        "Aiyagari, S. R. (1994). \"Uninsured Idiosyncratic Risk and Aggregate Saving.\" *Quarterly Journal of Economics*, 109(3), 659-684.",
        "Carroll, C. D. (2006). \"The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems.\" *Economics Letters*, 91(3), 312-320.",
        "Kaplan, G. (2017). Lecture notes on heterogeneous agent macroeconomics.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
