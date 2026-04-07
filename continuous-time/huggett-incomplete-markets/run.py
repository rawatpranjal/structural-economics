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
            # Excess supply of bonds -> lower r to increase demand
            r_hi = r_mid
        else:
            # Excess demand for bonds -> raise r to increase supply
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

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Huggett (1993) Incomplete Markets Model (Continuous Time)",
        "Stationary equilibrium of a heterogeneous-agent economy with "
        "idiosyncratic income risk, borrowing constraints, and a single bond.",
    )

    report.add_overview(
        "The Huggett (1993) model is a foundational heterogeneous-agent model in which "
        "a continuum of agents face uninsurable idiosyncratic income shocks and trade a "
        "single risk-free bond subject to a borrowing constraint. In the continuous-time "
        "formulation of Achdou et al. (2022), optimal behavior is characterized by a "
        "Hamilton-Jacobi-Bellman (HJB) equation, and the stationary wealth distribution "
        "satisfies a Kolmogorov Forward Equation (KFE). The equilibrium interest rate "
        "clears the bond market: net asset demand equals zero in the aggregate."
    )

    report.add_equations(
        r"""
**HJB equation:**
$$\rho V_i(a) = \max_{c} \left\{ \frac{c^{1-\sigma}}{1-\sigma} + V_i'(a)(z_i + ra - c) \right\} + \lambda_i \left[ V_j(a) - V_i(a) \right]$$

**Optimal consumption (FOC):** $c_i(a) = \left( V_i'(a) \right)^{-1/\sigma}$

**Savings policy:** $s_i(a) = z_i + ra - c_i(a)$

**KFE (Kolmogorov Forward Equation):**
$$0 = -\frac{\partial}{\partial a}\left[ s_i(a) \, g_i(a) \right] - \lambda_i \, g_i(a) + \lambda_j \, g_j(a)$$

**Bond market clearing:**
$$\int a \left[ g_1(a) + g_2(a) \right] da = 0$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$   | {rho} | Discount rate |\n"
        f"| $\\sigma$ | {sigma} | CRRA coefficient |\n"
        f"| $z$       | [{z[0]}, {z[1]}] | Income states |\n"
        f"| $\\lambda$ | [{la[0]}, {la[1]}] | Poisson switching rates |\n"
        f"| $\\underline{{a}}$ | {a_min} | Borrowing constraint |\n"
        f"| $\\bar{{a}}$       | {a_max} | Upper bound on assets |\n"
        f"| Grid points | {I} | Uniform spacing |"
    )

    report.add_solution_method(
        "**Finite-difference implicit method:** The HJB is solved using an upwind "
        "finite-difference scheme. At each grid point, the derivative $V'(a)$ is "
        "approximated by forward or backward differences depending on the sign of the "
        "drift (savings). This ensures numerical stability and respects the direction of "
        "information flow. The implicit time-stepping scheme\n\n"
        "$$\\frac{V^{n+1} - V^n}{\\Delta} + \\rho V^{n+1} = u(c^n) + A^n V^{n+1}$$\n\n"
        "is unconditionally stable, allowing large time steps ($\\Delta = 1000$) for fast "
        "convergence.\n\n"
        "**KFE:** The stationary distribution solves $A^\\top g = 0$ with $\\int g\\, da = 1$, "
        "computed via a sparse linear system.\n\n"
        "**General equilibrium:** Bisection on $r$ until $S(r) = \\int a \\, g(a)\\, da = 0$.\n\n"
        f"HJB converged in **{info['iterations']} iterations** (error = {info['error']:.2e}). "
        f"Equilibrium found at **$r^* = {r_eq:.5f}$**."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(a, V[:, 0], "b-", linewidth=2, label=f"$V_1(a)$, $z = {z[0]}$")
    ax1.plot(a, V[:, 1], "r-", linewidth=2, label=f"$V_2(a)$, $z = {z[1]}$")
    ax1.set_xlabel("Wealth $a$")
    ax1.set_ylabel("$V_i(a)$")
    ax1.set_title("Value Function")
    ax1.legend()
    report.add_figure(
        "figures/value-function.png",
        "Value function V(a) for each income state at the equilibrium interest rate",
        fig1,
    )

    # --- Figure 2: Savings Policy ---
    fig2, ax2 = plt.subplots()
    ax2.plot(a, s[:, 0], "b-", linewidth=2, label=f"$s_1(a)$, $z = {z[0]}$")
    ax2.plot(a, s[:, 1], "r-", linewidth=2, label=f"$s_2(a)$, $z = {z[1]}$")
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax2.axvline(a_min, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Wealth $a$")
    ax2.set_ylabel("Savings $s_i(a) = z_i + ra - c_i(a)$")
    ax2.set_title("Savings Policy Function")
    ax2.set_xlim([a_min - 0.03, 1.0])
    ax2.legend()
    report.add_figure(
        "figures/savings-policy.png",
        "Savings policy s(a,z) = z + r*a - c(a,z) at equilibrium; zero crossings are steady states",
        fig2,
    )

    # --- Figure 3: Stationary Wealth Distribution ---
    fig3, ax3 = plt.subplots()
    ax3.plot(a, g[:, 0], "b-", linewidth=2, label=f"$g_1(a)$, $z = {z[0]}$")
    ax3.plot(a, g[:, 1], "r-", linewidth=2, label=f"$g_2(a)$, $z = {z[1]}$")
    ax3.axvline(a_min, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Wealth $a$")
    ax3.set_ylabel("Density $g_i(a)$")
    ax3.set_title("Stationary Wealth Distribution")
    ax3.set_xlim([a_min - 0.03, 1.0])
    ax3.legend()
    report.add_figure(
        "figures/wealth-distribution.png",
        "Stationary wealth distribution g(a) by income state; mass piles up near borrowing constraint",
        fig3,
    )

    # --- Figure 4: Bond Market Clearing ---
    fig4, ax4 = plt.subplots()
    ax4.plot(S_grid_plot, r_grid_plot, "b-", linewidth=2, label="$S(r)$")
    ax4.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax4.axhline(rho, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.axvline(a_min, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax4.plot(0, r_eq, "ro", markersize=8, zorder=5, label=f"$r^* = {r_eq:.4f}$")
    ax4.set_xlabel("Bond supply $S(r) = \\int a \\, g(a) \\, da$")
    ax4.set_ylabel("Interest rate $r$")
    ax4.set_title("Bond Market Clearing")
    ax4.text(0.3, rho + 0.002, "$r = \\rho$", fontsize=10, color="gray")
    ax4.legend(loc="lower right")
    report.add_figure(
        "figures/bond-market.png",
        "Bond market: excess demand S(r) vs interest rate; equilibrium where S(r*)=0",
        fig4,
    )

    # --- Table: Equilibrium Values ---
    table_data = {
        "Variable": [
            "Equilibrium interest rate r*",
            "Mean wealth E[a]",
            "Mean income E[z]",
            "Mean consumption E[c]",
            "Prob(z = z_low)",
            "Prob(z = z_high)",
            "HJB iterations",
        ],
        "Value": [
            f"{r_eq:.5f}",
            f"{mean_wealth:.5f}",
            f"{mean_income:.4f}",
            f"{mean_cons:.4f}",
            f"{prob_z1:.4f}",
            f"{prob_z2:.4f}",
            f"{info['iterations']}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/equilibrium.csv",
        "Equilibrium Values",
        df,
    )

    report.add_takeaway(
        "The continuous-time approach to heterogeneous-agent models converts the problem "
        "into a coupled PDE system: the HJB equation characterizes optimal individual "
        "behavior, and the KFE describes the resulting cross-sectional distribution.\n\n"
        "**Key insights:**\n"
        "- The equilibrium interest rate $r^*$ is below the discount rate $\\rho$. This is "
        "the hallmark result of Huggett (1993): precautionary savings motives push agents "
        "to accumulate bonds, driving down the interest rate.\n"
        "- The wealth distribution features a mass point near the borrowing constraint "
        "$\\underline{a}$, reflecting agents hit by adverse income shocks who are unable "
        "to smooth consumption.\n"
        "- The savings function exhibits a square-root behavior near the constraint, "
        "reflecting the binding nature of the borrowing limit.\n"
        "- The upwind finite-difference scheme is essential: it selects forward or backward "
        "differences based on the direction of asset drift, ensuring stability and "
        "correctly capturing the state constraint at $\\underline{a}$."
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
