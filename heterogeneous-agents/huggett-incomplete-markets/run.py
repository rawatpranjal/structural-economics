#!/usr/bin/env python3
"""Huggett (1993) Incomplete Markets Model in Continuous Time.

Solves the Huggett economy using the HJB-KFE approach of Achdou et al. (2022).
Agents face idiosyncratic income risk with a 2-state Markov process and trade
a single bond subject to a borrowing constraint. The interest rate clears the
bond market in general equilibrium.

References:
    Huggett, M. (1993). "The risk-free rate in heterogeneous-agent incomplete-
        insurance economies." JEDC 17(5-6), 953-969.
    Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022).
        "Income and Wealth Distribution in Macroeconomics: A Continuous-Time
        Approach." REStud 89(1), 45-86.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


# =============================================================================
# Core solver functions
# =============================================================================

def solve_hjb(r, params):
    """Solve the HJB equation for given interest rate r via implicit method.

    Returns:
        V: (I, 2) value function on asset grid x income states
        c: (I, 2) consumption policy
        s: (I, 2) savings policy s = z + r*a - c
        A: (2I, 2I) sparse transition matrix for the KFE
        info: dict with convergence information
    """
    rho = params["rho"]
    sigma = params["sigma"]
    z = params["z"]
    la = params["la"]
    a = params["a"]
    I = params["I"]
    da = params["da"]
    Delta = params["Delta"]
    maxit = params["maxit"]
    crit = params["crit"]

    # Broadcast grids: aa is (I, 2), zz is (I, 2)
    aa = np.column_stack([a, a])
    zz = np.ones((I, 1)) * z[np.newaxis, :]

    # Income switching matrix (2I x 2I)
    Aswitch = sparse.bmat([
        [-sparse.eye(I) * la[0],  sparse.eye(I) * la[0]],
        [ sparse.eye(I) * la[1], -sparse.eye(I) * la[1]],
    ], format="csc")

    # Initial guess: consume everything (steady-state guess)
    income = zz + r * aa
    income_pos = np.maximum(income, 1e-10)
    V = income_pos ** (1 - sigma) / (1 - sigma) / rho

    dVf = np.zeros((I, 2))
    dVb = np.zeros((I, 2))

    for n in range(1, maxit + 1):
        # Forward difference
        dVf[:I-1, :] = (V[1:I, :] - V[:I-1, :]) / da
        dVf[I-1, :] = np.maximum(z + r * a[-1], 1e-10) ** (-sigma)  # state constraint

        # Backward difference
        dVb[1:I, :] = (V[1:I, :] - V[:I-1, :]) / da
        dVb[0, :] = np.maximum(z + r * a[0], 1e-10) ** (-sigma)  # state constraint at borrowing limit

        # Consumption and savings from forward difference
        cf = np.maximum(dVf, 1e-10) ** (-1.0 / sigma)
        ssf = zz + r * aa - cf

        # Consumption and savings from backward difference
        cb = np.maximum(dVb, 1e-10) ** (-1.0 / sigma)
        ssb = zz + r * aa - cb

        # Consumption at steady state (zero savings)
        c0 = zz + r * aa

        # Upwind scheme: choose forward/backward/zero based on drift sign
        If = (ssf > 0).astype(float)   # positive drift -> forward
        Ib = (ssb < 0).astype(float)   # negative drift -> backward
        I0 = 1.0 - If - Ib             # at steady state

        c = cf * If + cb * Ib + c0 * I0
        u = c ** (1 - sigma) / (1 - sigma)

        # Construct the transition matrix A (upwind scheme)
        X = -np.minimum(ssb, 0) / da  # sub-diagonal
        Y = -np.maximum(ssf, 0) / da + np.minimum(ssb, 0) / da  # main diagonal
        Z = np.maximum(ssf, 0) / da   # super-diagonal

        # Build A1 (state z1) and A2 (state z2) as tridiagonal matrices
        A1 = (sparse.diags(Y[:, 0], 0, shape=(I, I))
              + sparse.diags(X[1:I, 0], -1, shape=(I, I))
              + sparse.diags(Z[:I-1, 0], 1, shape=(I, I)))
        A2 = (sparse.diags(Y[:, 1], 0, shape=(I, I))
              + sparse.diags(X[1:I, 1], -1, shape=(I, I))
              + sparse.diags(Z[:I-1, 1], 1, shape=(I, I)))

        A = sparse.bmat([[A1, None], [None, A2]], format="csc") + Aswitch

        # Implicit update: (1/Delta + rho)*I - A) * V_new = u + V_old/Delta
        B = (1.0 / Delta + rho) * sparse.eye(2 * I, format="csc") - A

        u_stacked = np.concatenate([u[:, 0], u[:, 1]])
        V_stacked = np.concatenate([V[:, 0], V[:, 1]])

        b = u_stacked + V_stacked / Delta
        V_new_stacked = spsolve(B, b)

        V_new = np.column_stack([V_new_stacked[:I], V_new_stacked[I:2*I]])

        change = np.max(np.abs(V_new - V))
        V = V_new

        if change < crit:
            break

    # Recompute final policy at converged V
    dVf[:I-1, :] = (V[1:I, :] - V[:I-1, :]) / da
    dVf[I-1, :] = np.maximum(z + r * a[-1], 1e-10) ** (-sigma)
    dVb[1:I, :] = (V[1:I, :] - V[:I-1, :]) / da
    dVb[0, :] = np.maximum(z + r * a[0], 1e-10) ** (-sigma)

    cf = np.maximum(dVf, 1e-10) ** (-1.0 / sigma)
    ssf = zz + r * aa - cf
    cb = np.maximum(dVb, 1e-10) ** (-1.0 / sigma)
    ssb = zz + r * aa - cb
    c0 = zz + r * aa

    If = (ssf > 0).astype(float)
    Ib = (ssb < 0).astype(float)
    I0 = 1.0 - If - Ib

    c = cf * If + cb * Ib + c0 * I0
    s = zz + r * aa - c

    info = {"iterations": n, "converged": change < crit, "error": change}
    return V, c, s, A, info


def solve_kfe(A, params):
    """Solve the Kolmogorov Forward Equation for stationary distribution.

    Solves A' * g = 0 with integral(g) = 1.

    Returns:
        g: (I, 2) stationary density on (a, z) grid
    """
    I = params["I"]
    da = params["da"]

    AT = A.T.tocsc()

    # Fix one equation to pin down the level (otherwise singular)
    b = np.zeros(2 * I)
    i_fix = 0
    b[i_fix] = 0.1
    # Replace row i_fix of AT with a row that picks out g[i_fix]
    AT = AT.tolil()
    AT[i_fix, :] = 0
    AT[i_fix, i_fix] = 1.0
    AT = AT.tocsc()

    gg = spsolve(AT, b)

    # Normalize so that integral g * da = 1
    g_sum = np.sum(gg) * da
    gg = gg / g_sum

    g = np.column_stack([gg[:I], gg[I:2*I]])
    return g


def excess_demand(r, params):
    """Compute excess bond demand S(r) = integral(a * g(a,z) da dz).

    Returns:
        S: excess demand (S > 0 means agents want to hold positive bonds)
        V, c, s, g: solutions at this r
    """
    V, c, s, A, info = solve_hjb(r, params)
    g = solve_kfe(A, params)
    a = params["a"]
    da = params["da"]

    S = (g[:, 0] @ a) * da + (g[:, 1] @ a) * da
    return S, V, c, s, g, info


def find_equilibrium(params, r_min=0.001, r_max=0.04, tol=1e-5, max_iter=40):
    """Find the equilibrium interest rate by bisection on the bond market.

    Returns:
        r_eq: equilibrium interest rate
        V, c, s, g: solutions at equilibrium
        r_history, S_history: for plotting the bond market clearing diagram
    """
    r_lo, r_hi = r_min, r_max
    r_history = []
    S_history = []

    r_eq = 0.5 * (r_lo + r_hi)
    V = c = s = g = info = None

    for it in range(1, max_iter + 1):
        r_mid = 0.5 * (r_lo + r_hi)
        S, V, c, s, g, info = excess_demand(r_mid, params)

        r_history.append(r_mid)
        S_history.append(S)

        print(f"  Bisection {it:2d}: r = {r_mid:.6f}, S(r) = {S:+.6f}")

        if abs(S) < tol:
            r_eq = r_mid
            print(f"  Equilibrium found: r* = {r_eq:.6f}")
            break
        elif S > 0:
            # Desired aggregate assets are positive; lower r to reduce saving.
            r_hi = r_mid
        else:
            # Desired aggregate assets are negative; raise r to reduce borrowing.
            r_lo = r_mid
        r_eq = r_mid

    return r_eq, V, c, s, g, info, np.array(r_history), np.array(S_history)


def compute_supply_curve(params, r_grid):
    """Compute S(r) on a grid of interest rates for plotting."""
    S_vals = np.zeros(len(r_grid))
    for i, r in enumerate(r_grid):
        S, *_ = excess_demand(r, params)
        S_vals[i] = S
        print(f"  Supply curve: r = {r:.4f}, S(r) = {S:+.6f}")
    return S_vals


# =============================================================================
# Main
# =============================================================================

def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    rho = 0.05      # Discount rate
    sigma = 2.0     # CRRA coefficient
    z = np.array([0.1, 0.2])       # Income states
    la = np.array([1.2, 1.2])      # Switching rates (Poisson)
    a_min = -0.15   # Borrowing constraint
    a_max = 5.0     # Upper bound on assets
    I = 500          # Grid points

    a = np.linspace(a_min, a_max, I)
    da = (a_max - a_min) / (I - 1)

    params = {
        "rho": rho, "sigma": sigma, "z": z, "la": la,
        "a": a, "I": I, "da": da,
        "Delta": 1000, "maxit": 100, "crit": 1e-6,
    }

    # =========================================================================
    # General Equilibrium: find r* that clears the bond market
    # =========================================================================
    print("Finding equilibrium interest rate by bisection...")
    r_eq, V, c, s, g, info, r_hist, S_hist = find_equilibrium(
        params, r_min=0.001, r_max=0.045, tol=1e-5, max_iter=40,
    )

    # =========================================================================
    # Compute the full supply curve S(r) for the bond market diagram
    # =========================================================================
    print("\nComputing bond market supply curve...")
    r_grid_plot = np.linspace(-0.02, 0.049, 20)
    S_grid_plot = compute_supply_curve(params, r_grid_plot)

    # =========================================================================
    # Aggregate statistics
    # =========================================================================
    mean_wealth = (g[:, 0] @ a) * da + (g[:, 1] @ a) * da
    mean_income = (g[:, 0] @ (np.ones(I) * z[0])) * da + (g[:, 1] @ (np.ones(I) * z[1])) * da
    mean_cons = (g[:, 0] @ c[:, 0]) * da + (g[:, 1] @ c[:, 1]) * da
    frac_constrained = np.sum(g[:5, :]) * da  # fraction near borrowing limit
    prob_z1 = np.sum(g[:, 0]) * da
    prob_z2 = np.sum(g[:, 1]) * da
    market_residual = abs(mean_wealth)
    mean_wealth_display = 0.0 if market_residual < 5e-5 else mean_wealth

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Huggett Equilibrium and the Risk-Free Rate",
        "A continuous-time incomplete-markets economy where idiosyncratic income risk "
        "and borrowing limits determine the equilibrium bond return.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Huggett (1993) asks how a risk-free asset is priced when households cannot "
        "perfectly insure idiosyncratic income risk. Each household can save or borrow "
        "in a single bond, but debt is limited by $a \\geq \\underline a$. Since the "
        "bond is in zero net supply, the interest rate has to make the cross-sectional "
        "demand for assets add up to zero.\n\n"
        "The economic force is precautionary saving. Low-income spells make the "
        "borrowing limit valuable, so households try to carry buffer wealth into bad "
        "states. In equilibrium that desired asset demand is offset by a lower risk-free "
        "rate. The continuous-time HJB/KFE representation from Achdou et al. (2022) "
        "makes the two equilibrium objects explicit: the household drift in asset space "
        "and the stationary density that drift induces."
    )

    report.add_equations(
        r"""
There are two income states, $i \in \{L,H\}$, with income $z_i$ and Poisson
switching intensity $\lambda_i$ into the other state $j$. Assets are denoted by
$a$, the bond return by $r$, and the stationary density by $g_i(a)$.

Household assets evolve according to
$$\dot a = s_i(a) = z_i + r a - c_i(a), \qquad a \geq \underline a.$$

The HJB equation is
$$\rho V_i(a) =
\max_{c > 0}
\Bigl[
\frac{c^{1-\sigma}}{1-\sigma}
+ V_i'(a)\,[z_i + r a - c]
+ \lambda_i [V_j(a)-V_i(a)]
\Bigr].$$

The consumption rule comes from the first-order condition
$$c_i(a)=\left[V_i'(a)\right]^{-1/\sigma},$$
so the asset drift is $s_i(a)=z_i+ra-c_i(a)$. At the borrowing limit the state
constraint requires $s_i(\underline a)\geq 0$.

The stationary distribution solves the KFE
$$0 =
-\frac{\partial}{\partial a}\left[s_i(a)g_i(a)\right]
-\lambda_i g_i(a)+\lambda_j g_j(a),$$
with total mass normalized to one. The zero-net-supply bond market clears when
$$S(r)=\int_{\underline a}^{\bar a} a\,[g_L(a)+g_H(a)]\,da=0.$$
"""
    )

    report.add_model_setup(
        "The calibration is deliberately small: two income states, symmetric switching, "
        "and a one-dimensional asset grid. This keeps the focus on how incomplete "
        "insurance pins down the risk-free rate.\n\n"
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$   | {rho} | Rate of time preference |\n"
        f"| $\\sigma$ | {sigma} | Relative risk aversion |\n"
        f"| $z$       | [{z[0]}, {z[1]}] | Income states |\n"
        f"| $\\lambda$ | [{la[0]}, {la[1]}] | Income-state switching rates |\n"
        f"| $\\underline{{a}}$ | {a_min} | Borrowing constraint |\n"
        f"| $\\bar{{a}}$       | {a_max} | Upper bound on assets |\n"
        f"| Asset grid points | {I} | Uniform spacing |"
    )

    report.add_solution_method(
        r"""
For a candidate interest rate, the HJB is solved by an implicit finite-difference
step. The only delicate choice is the derivative of $V_i(a)$. If assets drift to
the right, the update uses the forward derivative; if assets drift to the left,
it uses the backward derivative. This upwind choice is what keeps the borrowing
constraint from being crossed numerically.

The implicit update can be written as
$$\left[(\Delta^{-1}+\rho)I-A^n\right]V^{n+1}
=u(c^n)+\Delta^{-1}V^n,$$
where $A^n$ is the Markov generator over asset and income states implied by the
current drift. After the HJB step converges, the same generator is used in the
KFE, $A^\top g=0$, with $\int g\,da=1$.

```text
Inputs: candidate r, asset grid a, income states z, switching intensities lambda
Output: policies c_i(a), s_i(a), density g_i(a), aggregate assets S(r)

1. Guess V_i^0(a).
2. Until the HJB sup norm is small:
   a. Compute forward and backward derivatives of V_i^n(a).
   b. Convert each derivative into candidate consumption using u'(c)=V_i'(a).
   c. Select the derivative whose implied asset drift points into the grid.
   d. Build the generator A^n for asset drift plus income switching.
   e. Solve the sparse implicit system for V_i^{n+1}(a).
3. Solve A(r)' g = 0 and normalize the density to integrate to one.
4. Compute S(r)=integral a[g_L(a)+g_H(a)] da.
5. Use bisection on r until S(r) is close to zero.
```

There is no closed-form equilibrium to overlay as ground truth in this
truncated-grid Huggett problem. The relevant numerical checks are therefore the
HJB fixed-point tolerance and the zero-net-supply bond-market residual.
"""
        + "\n"
        + f"The final HJB step converged in **{info['iterations']} iterations** "
        + f"(sup-norm change {info['error']:.2e}). Bisection gives "
        + f"**$r^{{*}}={r_eq:.5f}$**, with a market-clearing residual of "
        + f"**{market_residual:.2e}** in aggregate assets."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(a, V[:, 0], "b-", linewidth=2, label=f"$V_1(a)$, $z = {z[0]}$")
    ax1.plot(a, V[:, 1], "r-", linewidth=2, label=f"$V_2(a)$, $z = {z[1]}$")
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V_i(a)$")
    ax1.set_title("Value Function")
    ax1.legend()
    report.add_figure(
        "figures/value-function.png",
        "Value functions by asset holdings and income state at the equilibrium interest rate",
        fig1,
        description="The value functions put the income state and the borrowing limit in "
        "one picture. High income raises continuation value at every asset level. Near "
        "$\\underline a$, both curves become steep because one more unit of wealth relaxes "
        "the state constraint and buys insurance against staying in the low-income state.",
    )

    # --- Figure 2: Savings Policy ---
    fig2, ax2 = plt.subplots()
    ax2.plot(a, s[:, 0], "b-", linewidth=2, label=f"$s_1(a)$, $z = {z[0]}$")
    ax2.plot(a, s[:, 1], "r-", linewidth=2, label=f"$s_2(a)$, $z = {z[1]}$")
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax2.axvline(a_min, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Savings $s_i(a) = z_i + ra - c_i(a)$")
    ax2.set_title("Savings Policy Function")
    ax2.set_xlim([a_min - 0.03, 1.0])
    ax2.legend()
    report.add_figure(
        "figures/savings-policy.png",
        "Asset drift policies by income state at the equilibrium interest rate",
        fig2,
        description="The savings policies show the direction of movement in asset space. "
        "Low-income households run assets down unless they are already wealthy; high-income "
        "households rebuild buffer wealth. The zero crossings are local asset targets for a "
        "fixed income state, but income switching keeps households moving across the two "
        "drift fields.",
    )

    # --- Figure 3: Stationary Wealth Distribution ---
    fig3, ax3 = plt.subplots()
    ax3.plot(a, g[:, 0], "b-", linewidth=2, label=f"$g_1(a)$, $z = {z[0]}$")
    ax3.plot(a, g[:, 1], "r-", linewidth=2, label=f"$g_2(a)$, $z = {z[1]}$")
    ax3.axvline(a_min, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Density $g_i(a)$")
    ax3.set_title("Stationary Wealth Distribution")
    ax3.set_xlim([a_min - 0.03, 1.0])
    ax3.legend()
    report.add_figure(
        "figures/wealth-distribution.png",
        "Stationary asset densities by income state",
        fig3,
        description="The KFE turns those drift policies into a cross-sectional distribution. "
        "Density is largest near the borrowing limit because low-income households drift "
        "toward it and cannot continue borrowing once they arrive. The right tail is thin "
        "but nonzero, coming from households that have spent enough time in the high-income "
        "state to accumulate assets.",
    )

    # --- Figure 4: Bond Market Clearing ---
    fig4, ax4 = plt.subplots()
    ax4.plot(S_grid_plot, r_grid_plot, "b-", linewidth=2, label="$S(r)$")
    ax4.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax4.axhline(rho, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.plot(0, r_eq, "ro", markersize=8, zorder=5, label=f"$r^{{*}} = {r_eq:.4f}$")
    ax4.set_xlabel("Aggregate asset demand $S(r) = \\int a \\, g(a) \\, da$")
    ax4.set_ylabel("Interest rate $r$")
    ax4.set_title("Bond Market Clearing")
    ax4.text(0.3, rho + 0.002, "$r = \\rho$", fontsize=10, color="gray")
    ax4.legend(loc="lower right")
    report.add_figure(
        "figures/bond-market.png",
        "Aggregate asset demand as a function of the interest rate",
        fig4,
        description="The market-clearing plot is the equilibrium argument. Higher $r$ makes "
        "saving more attractive, so aggregate asset demand rises. Because the bond is in "
        "zero net supply, equilibrium is the point where the curve crosses $S(r)=0$. That "
        "crossing is below $\\rho$: the interest rate has to fall enough to offset "
        "precautionary asset demand.",
    )

    # --- Table: Equilibrium Values ---
    table_data = {
        "Variable": [
            "Equilibrium interest rate r*",
            "Mean wealth E[a]",
            "Mean income E[z]",
            "Mean consumption E[c]",
            "Mass near borrowing limit",
            "Prob(z = z_low)",
            "Prob(z = z_high)",
            "Bond-market residual abs(E[a])",
            "HJB iterations",
            "HJB final sup-norm change",
        ],
        "Value": [
            f"{r_eq:.5f}",
            f"{mean_wealth_display:.5f}",
            f"{mean_income:.4f}",
            f"{mean_cons:.4f}",
            f"{frac_constrained:.4f}",
            f"{prob_z1:.4f}",
            f"{prob_z2:.4f}",
            f"{market_residual:.2e}",
            f"{info['iterations']}",
            f"{info['error']:.2e}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/equilibrium.csv",
        "Equilibrium Values",
        df,
        description="The table summarizes the equilibrium, not a separate calibration target. "
        "Mean assets are numerically zero because the interest rate has been chosen to clear "
        "the zero-net-supply bond market. The mass near $\\underline a$ is the visible trace "
        "of incomplete insurance in this two-state economy.",
    )

    report.add_takeaway(
        "The Huggett pricing mechanism is the lesson. With incomplete insurance, "
        "households want to self-insure by holding the risk-free bond; with zero net "
        "supply, the equilibrium return must fall until aggregate desired assets are zero. "
        "The HJB/KFE machinery keeps the individual saving drift and the induced wealth "
        "distribution in the same equilibrium calculation. This is the continuous-time "
        "counterpart to the discrete-time incomplete-markets logic in the "
        "[Aiyagari saving tutorial](../../dynamic-programming/aiyagari/), but here the "
        "asset in fixed supply is a bond rather than aggregate capital."
    )

    report.add_references([
        "Huggett, M. (1993). \"The risk-free rate in heterogeneous-agent incomplete-insurance economies.\" *Journal of Economic Dynamics and Control* 17(5-6), 953-969.",
        "Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). \"Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach.\" *Review of Economic Studies* 89(1), 45-86.",
        "Moll, B. \"Lecture notes on continuous-time heterogeneous-agent models.\" https://benjaminmoll.com/lectures/",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
