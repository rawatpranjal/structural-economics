#!/usr/bin/env python3
"""Aiyagari general equilibrium solved with Endogenous Grid Points.

The model is Aiyagari (1994): households face uninsurable idiosyncratic labor
income risk, save in a risk-free asset, and supply capital to a competitive
firm. The equilibrium interest rate clears the capital market.

The code solves the household problem with the Endogenous Grid Points (EGP)
method of Carroll (2006). EGP keeps the economics of the Euler equation front
and center: choose a grid for tomorrow's assets, infer today's consumption from
marginal utility, and then map back to the current asset grid. That makes the
household block cheap enough to solve repeatedly inside the market-clearing
loop.

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

    # Finer-grid reference at the final equilibrium prices. This is not a
    # separate GE solve; it checks the household-policy approximation that feeds
    # the displayed equilibrium.
    na_ref = 400
    agrid_ref = np.linspace(0, 1, na_ref) ** (1 / agrid_par)
    agrid_ref = borrow_lim + (amax - borrow_lim) * agrid_ref
    con_ref_init = np.zeros((na_ref, ny))
    for iy in range(ny):
        con_ref_init[:, iy] = np.maximum(
            r_eq * agrid_ref + w_eq * yscale * ygrid[iy, 0], 1e-10)

    con_ref, sav_ref, ref_it, ref_err = solve_egp_inner(
        agrid_ref, ygrid, ydist, na_ref, ny, beta, R_eq, w_eq, yscale,
        risk_aver, borrow_lim, con_ref_init, max_iter, tol_iter)

    con_ref_on_main = np.column_stack([
        np.interp(agrid_flat, agrid_ref, con_ref[:, iy])
        for iy in range(ny)
    ])
    sav_ref_on_main = np.column_stack([
        np.interp(agrid_flat, agrid_ref, sav_ref[:, iy])
        for iy in range(ny)
    ])
    ref_mask = agrid_flat <= 15.0
    consumption_ref_gap = float(np.max(np.abs(
        con[ref_mask, :] - con_ref_on_main[ref_mask, :])))
    savings_ref_gap = float(np.max(np.abs(
        sav[ref_mask, :] - sav_ref_on_main[ref_mask, :])))
    impatience_rate = 1.0 / beta - 1.0
    interest_gap = impatience_rate - r_eq

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
    print(f"  Fine-grid c gap  = {consumption_ref_gap:.2e}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Aiyagari Equilibrium with Endogenous Grid Points",
        "Capital-market clearing in an incomplete-markets economy with an EGP household block.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Aiyagari (1994) turns the buffer-stock household problem into a general "
        "equilibrium model. A continuum of households faces uninsurable labor-income "
        "risk, saves in capital, and rents that capital to a competitive firm. The "
        "interest rate is no longer a primitive. It is the price that makes aggregate "
        "household saving equal the firm's desired capital stock.\n\n"
        "Relative to the preceding [EGP household tutorial](../endogenous-grid-points/), "
        "the economic change is market clearing. Relative to the "
        "[VFI Aiyagari tutorial](../../dynamic-programming/aiyagari/), the computation "
        "changes inside the household block. Endogenous Grid Points keep the Euler "
        "equation solution fast enough that we can solve the household problem many "
        "times while searching for the equilibrium capital-labor ratio."
    )

    report.add_equations(
        r"""
Households begin the period with assets $a\geq \underline a=0$ and labor
efficiency $e_j$. Income is IID with probabilities $\pi_j$. For prices
$(r,w)$ and gross return $R=1+r$,

$$
V(a,e_j)=
\max_{a'\geq 0}
\Bigl[
u(Ra+w e_j-a')
+\beta \sum_{\ell=1}^{n_y}\pi_{\ell}V(a',e_{\ell})
\Bigr],
$$

with budget identity

$$
c(a,e_j)=Ra+w e_j-g(a,e_j),
\qquad
u(c)=\frac{c^{1-\gamma}-1}{1-\gamma}.
$$

For an interior choice, the Euler equation is

$$
u'(c(a,e_j))
=
\beta R
\sum_{\ell=1}^{n_y}
\pi_{\ell}u'\!\left(c(g(a,e_j),e_{\ell})\right).
$$

EGP inverts this equation at candidate next assets $a'_i$:

$$
c_i =
\left[
\beta R
\sum_{\ell=1}^{n_y}
\pi_{\ell}u'\!\left(c(a'_i,e_{\ell})\right)
\right]^{-1/\gamma},
\qquad
a^{endo}_{ij}=\frac{c_i+a'_i-w e_j}{R}.
$$

If the current exogenous asset grid lies below the first endogenous point, the
borrowing constraint binds and $g(a,e_j)=0$.

The firm has Cobb-Douglas technology

$$
Y=K^\alpha L^{1-\alpha},
\qquad
r(k)=\alpha k^{\alpha-1}-\delta,
\qquad
w(k)=(1-\alpha)k^\alpha,
$$

where $k=K/L$. Given the invariant household distribution $\mu_k$, capital
market clearing requires

$$
K^s(k)
=
\int g(a,e_j)\,d\mu_k(a,e_j)
=
kL.
$$
"""
    )

    report.add_model_setup(
        "The exercise uses IID income risk to keep the income process from becoming "
        "the main object. The raw income states are a five-point normal approximation. "
        "For each candidate capital-labor ratio $k$, the code rescales efficiency "
        "units so aggregate output is normalized near one; this normalization changes "
        "units, not the market-clearing logic.\n\n"
        f"| Primitive | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $\\gamma$ | {risk_aver} | CRRA risk aversion |\n"
        f"| $\\alpha$ | {alpha} | Capital share |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\mu_y$ | {mu_y} | Mean raw labor income |\n"
        f"| $\\sigma_y$ | {sd_y} | Raw income standard deviation |\n"
        f"| $n_y$ | {ny} | IID income states |\n"
        f"| $\\underline a$ | {borrow_lim} | Borrowing limit |\n"
        f"| $\\bar a$ | {amax} | Upper asset-grid bound |\n"
        f"| Main asset grid | {na} points | Exponential spacing near $\\underline a$ |\n"
        f"| Reference asset grid | {na_ref} points | Policy check at final prices |\n"
        f"| Simulation | {Nsim:,} households, {Tsim} periods | Terminal cross section for $\\mu_k$ |"
    )

    report.add_solution_method(
        "There are two fixed points. For a candidate $k=K/L$, prices determine a "
        "household saving rule. That rule induces a stationary distribution and hence "
        "aggregate capital supply. General equilibrium is the $k$ for which this "
        "supply equals the firm's implied capital demand.\n\n"
        "```text\n"
        "Input: asset grid A, income states e_j with probabilities pi_j,\n"
        "       primitives beta, gamma, alpha, delta, borrowing limit a_min\n"
        "Initialize a capital-labor ratio k_0\n"
        "For m = 0, 1, 2, ...:\n"
        "    Compute prices r_m = alpha k_m^(alpha-1) - delta and w_m = (1-alpha) k_m^alpha\n"
        "    Solve the household problem by EGP:\n"
        "        Initialize c_0(a,e_j)\n"
        "        For n = 0, 1, 2, ...:\n"
        "            For each candidate next asset a_i' in A:\n"
        "                M_i = sum_j pi_j u'(c_n(a_i',e_j))\n"
        "                c_i = (beta (1+r_m) M_i)^(-1/gamma)\n"
        "            For each current income e_j:\n"
        "                a_ij^endo = (c_i + a_i' - w_m e_j) / (1+r_m)\n"
        "                Interpolate (a_ij^endo, a_i') back to the exogenous grid A\n"
        "                Use a' = a_min below the first endogenous point\n"
        "                Recover c_{n+1}(a,e_j) from the budget constraint\n"
        "            Stop when max_{a,j} |c_{n+1}(a,e_j)-c_n(a,e_j)| < epsilon\n"
        "    Simulate households forward under the saving rule to approximate mu_m\n"
        "    Set K_m^s = mean terminal assets and L_m = mean labor efficiency\n"
        "    Update k_{m+1} = (1-lambda) k_m + lambda K_m^s/L_m\n"
        "    Stop when |K_m^s/L_m / k_m - 1| < tolerance\n"
        "Output: equilibrium prices, policy functions, and simulated wealth distribution\n"
        "```\n\n"
        "The run converged to the displayed equilibrium in "
        f"**{ge_iters}** damped market-clearing iterations. At the final prices, "
        f"the main EGP solve took **{egp_iters}** iterations. A {na_ref}-point "
        "asset-grid check at those same prices gives a maximum consumption-policy "
        f"gap of **{consumption_ref_gap:.2e}** over $a\\leq 15$."
    )

    # --- Figure 1: Consumption policy at equilibrium ---
    fig1, ax1 = plt.subplots()
    xlim_plot = min(15, amax)
    mask_plot = agrid_flat <= xlim_plot
    mask_ref_plot = agrid_ref <= xlim_plot
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, ny))
    for iy in range(ny):
        label = f"y = {ygrid[iy, 0]:.2f} (p = {ydist[iy, 0]:.3f})"
        ax1.plot(agrid_flat[mask_plot], con[mask_plot, iy], color=colors[iy],
                 linewidth=2, label=label)
        if iy in (0, ny - 1):
            ref_label = f"{na_ref}-grid ref., y = {ygrid[iy, 0]:.2f}"
            ax1.plot(agrid_ref[mask_ref_plot], con_ref[mask_ref_plot, iy],
                     color=colors[iy], linewidth=1.5, linestyle='--',
                     label=ref_label)
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("Consumption $c$")
    ax1.set_title("Equilibrium Consumption Policy")
    ax1.legend(fontsize=8)
    report.add_figure("figures/consumption-policy.png",
                      "Equilibrium consumption policy with fine-grid reference",
                      fig1,
        description="The equilibrium policy is still a buffer-stock rule. Low-income "
        "households consume less at a given asset level because a bad draw both lowers "
        "current resources and raises the value of keeping assets for self-insurance. "
        "The dashed reference curves solve the same household problem on a finer asset "
        f"grid; over $a\\leq 15$ the largest consumption gap is {consumption_ref_gap:.2e}.")

    # --- Figure 2: Capital supply vs demand ---
    fig2, ax2 = plt.subplots()
    ax2.plot(Ks_plot, r_plot_vals, 'b-o', linewidth=2, markersize=4,
             label='Capital supply (savings)')
    ax2.plot(Kd_plot, r_plot_vals, 'r-s', linewidth=2, markersize=4,
             label='Capital demand (firm FOC)')
    ax2.axhline(r_eq, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f'$r^{{*}} = {r_eq:.4f}$')
    ax2.axhline(impatience_rate, color='black', linestyle=':', linewidth=1.2,
                alpha=0.8, label=r'$1/\beta - 1$')
    ax2.set_xlabel("Capital $K$")
    ax2.set_ylabel("Interest rate $r$")
    ax2.set_title("Capital Market Clearing")
    ax2.legend()
    report.add_figure("figures/capital-supply-demand.png",
                      "Capital supply (household savings) and demand (firm FOC) as functions of r",
                      fig2,
        description="The crossing is the equilibrium price of capital. Household capital "
        "supply rises with the return, while firm demand falls with the marginal product "
        "of capital. The dotted line marks the no-risk Euler benchmark $1/\\beta-1$; "
        "the incomplete-markets equilibrium lies below it because households want a "
        "precautionary buffer.")

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
                      fig3,
        description="The terminal simulated cross section approximates the invariant "
        "wealth distribution under the equilibrium policy. The mass near low assets is "
        "the borrowing constraint and bad income draws at work; the right tail comes "
        "from households with long favorable income histories. This IID calibration "
        "keeps the tail modest relative to persistent-income Aiyagari models.")

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
                      fig4,
        description="The Lorenz curve summarizes the same distribution in inequality "
        "terms. A Gini around this level should be read as the inequality generated by "
        "borrowing limits plus IID income risk, not as a full empirical wealth model. "
        "Persistent earnings risk, lifecycle structure, and heterogeneous returns would "
        "all move this object.")

    # --- Table: Equilibrium statistics ---
    table_data = {
        "Statistic": [
            "Interest rate r",
            "No-risk Euler rate 1/beta - 1",
            "Gap: 1/beta - 1 - r",
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
            "GE iterations",
            "Final EGP iterations",
            "Reference-grid c gap, a <= 15",
            "Reference-grid savings gap, a <= 15",
        ],
        "Value": [
            f"{r_eq:.6f}",
            f"{impatience_rate:.6f}",
            f"{interest_gap:.6f}",
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
            f"{ge_iters:d}",
            f"{egp_iters:d}",
            f"{consumption_ref_gap:.2e}",
            f"{savings_ref_gap:.2e}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/equilibrium.csv", "Equilibrium and Accuracy Checks", df,
        description="The table puts the economic objects and numerical checks together. "
        "The interest-rate gap is the main Aiyagari mechanism in this calibration. "
        "The fine-grid gaps are fixed-price household-policy diagnostics, so they "
        "check the EGP approximation rather than re-solving the full equilibrium.")

    report.add_takeaway(
        "The economic lesson is the Aiyagari interest-rate result: uninsurable income "
        "risk creates aggregate precautionary saving, so the clearing return on capital "
        f"is {r_eq:.4f}, below the no-risk Euler benchmark {impatience_rate:.4f}. "
        "The same policy generates a right-skewed wealth distribution and heterogeneous "
        f"MPCs, with a mean windfall MPC of {mean_mpc:.3f} in the simulated cross "
        "section.\n\n"
        "The computational lesson is narrower but important. EGP does not change the "
        "equilibrium concept. It changes the cost of the household block by replacing "
        "a search over next assets with Euler-equation inversion and interpolation. "
        "That is why this version is the natural bridge from partial-equilibrium "
        "buffer-stock saving to repeated general-equilibrium market clearing."
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
