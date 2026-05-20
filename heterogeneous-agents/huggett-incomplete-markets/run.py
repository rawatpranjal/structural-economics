#!/usr/bin/env python3
"""Huggett (1993) Incomplete Markets Model in Continuous Time.

Solves the Huggett economy using the HJB-KFE approach of Achdou et al. (2022).
Households face two-state Poisson income risk and trade a single bond subject
to a hard borrowing limit. The interest rate clears the zero-net-supply bond
market in general equilibrium. A finer reference grid is solved alongside the
working grid as a discretisation audit.

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
from lib.plotting import setup_style, save_figure, save_thumbnail


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

    # Pin one equation to fix the scale (otherwise the system is singular)
    b = np.zeros(2 * I)
    i_fix = 0
    b[i_fix] = 0.1
    AT = AT.tolil()
    AT[i_fix, :] = 0
    AT[i_fix, i_fix] = 1.0
    AT = AT.tocsc()

    gg = spsolve(AT, b)

    # Normalise so the density integrates to one
    g_sum = np.sum(gg) * da
    gg = gg / g_sum

    g = np.column_stack([gg[:I], gg[I:2*I]])
    return g


def excess_demand(r, params):
    """Aggregate bond demand S(r) = integral a g(a, z) da dz at rate r."""
    V, c, s, A, info = solve_hjb(r, params)
    g = solve_kfe(A, params)
    a = params["a"]
    da = params["da"]

    S = (g[:, 0] @ a) * da + (g[:, 1] @ a) * da
    return S, V, c, s, g, info


def find_equilibrium(params, r_min=0.001, r_max=0.04, tol=1e-5, max_iter=40, label=""):
    """Bisection on r until the bond market clears."""
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

        print(f"  {label}Bisection {it:2d}: r = {r_mid:.6f}, S(r) = {S:+.6f}")

        if abs(S) < tol:
            r_eq = r_mid
            print(f"  {label}Equilibrium: r* = {r_eq:.6f}")
            break
        elif S > 0:
            # Aggregate desired assets are positive; lower r to depress saving.
            r_hi = r_mid
        else:
            # Aggregate desired assets are negative; raise r to depress borrowing.
            r_lo = r_mid
        r_eq = r_mid

    return r_eq, V, c, s, g, info, np.array(r_history), np.array(S_history)


def compute_supply_curve(params, r_grid):
    """Compute S(r) on a grid of interest rates for plotting."""
    S_vals = np.zeros(len(r_grid))
    for i, r in enumerate(r_grid):
        S, *_ = excess_demand(r, params)
        S_vals[i] = S
        print(f"  Supply curve: r = {r:+.4f}, S(r) = {S:+.6f}")
    return S_vals


def build_params(I, a_min, a_max, base):
    """Construct a params dict for a given grid size on [a_min, a_max]."""
    a = np.linspace(a_min, a_max, I)
    da = (a_max - a_min) / (I - 1)
    out = dict(base)
    out["a"] = a
    out["I"] = I
    out["da"] = da
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    # =========================================================================
    # Calibration
    # =========================================================================
    rho = 0.05      # Rate of time preference (continuous-time discount rate)
    sigma = 2.0     # CRRA coefficient
    z = np.array([0.1, 0.2])       # Income endowments in low and high state
    la = np.array([1.2, 1.2])      # Poisson switching intensities
    a_min = -0.15   # Hard borrowing limit
    a_max = 5.0     # Upper bound on the asset grid
    I = 2000        # Working grid size
    I_ref = 6000    # Reference grid for the discretisation audit

    base = {
        "rho": rho, "sigma": sigma, "z": z, "la": la,
        "Delta": 1000, "maxit": 100, "crit": 1e-6,
    }
    params = build_params(I, a_min, a_max, base)
    params_ref = build_params(I_ref, a_min, a_max, base)

    # Stationary income probabilities (analytic, from balanced-flow lambda_L p_L = lambda_H p_H)
    p_low = la[1] / (la[0] + la[1])
    p_high = la[0] / (la[0] + la[1])
    mean_z_analytic = p_low * z[0] + p_high * z[1]

    # =========================================================================
    # Equilibrium on the working grid
    # =========================================================================
    print(f"Working grid (I={I}): bisecting on r for bond market clearing")
    r_eq, V, c, s, g, info, r_hist, S_hist = find_equilibrium(
        params, r_min=0.001, r_max=0.045, tol=1e-5, max_iter=40, label=f"[I={I}] ",
    )
    a = params["a"]
    da = params["da"]

    # Aggregate moments on the working grid
    mean_wealth = (g[:, 0] @ a) * da + (g[:, 1] @ a) * da
    market_residual = abs(mean_wealth)
    mean_cons = (g[:, 0] @ c[:, 0]) * da + (g[:, 1] @ c[:, 1]) * da
    # Mass within a fixed asset-range window of the borrowing limit
    constraint_window = 0.02
    a_window_mask = a <= a_min + constraint_window
    mass_at_constraint = float(np.sum(g[a_window_mask, :]) * da)
    prob_z_low = float(np.sum(g[:, 0]) * da)
    prob_z_high = float(np.sum(g[:, 1]) * da)
    wedge = rho - r_eq
    wedge_pct = 100 * wedge / rho
    p_balance_err = abs(prob_z_low - 0.5)

    # =========================================================================
    # Reference grid: same equilibrium computation, finer asset discretisation
    # =========================================================================
    print(f"\nReference grid (I={I_ref}): bisecting on r for the audit")
    r_eq_ref, V_ref, c_ref, s_ref, g_ref, info_ref, _, _ = find_equilibrium(
        params_ref, r_min=0.001, r_max=0.045, tol=1e-5, max_iter=40, label=f"[I={I_ref}] ",
    )
    a_ref = params_ref["a"]
    da_ref = params_ref["da"]
    mean_wealth_ref = (g_ref[:, 0] @ a_ref) * da_ref + (g_ref[:, 1] @ a_ref) * da_ref

    # Discretisation diagnostics: gap in equilibrium price and policy on the
    # active range. Compare on a common subset for the policy norms.
    r_gap = abs(r_eq - r_eq_ref)
    a_lo, a_hi = a_min, 1.0
    mask = (a >= a_lo) & (a <= a_hi)
    s_ref_on_a = np.column_stack([
        np.interp(a, a_ref, s_ref[:, 0]),
        np.interp(a, a_ref, s_ref[:, 1]),
    ])
    s_gap = float(np.max(np.abs(s[mask, :] - s_ref_on_a[mask, :])))
    V_ref_on_a = np.column_stack([
        np.interp(a, a_ref, V_ref[:, 0]),
        np.interp(a, a_ref, V_ref[:, 1]),
    ])
    V_gap = float(np.max(np.abs(V[mask, :] - V_ref_on_a[mask, :])))
    V_scale = float(np.max(np.abs(V[mask, :])))
    V_gap_rel = V_gap / V_scale

    # =========================================================================
    # Bond-market supply curve for the equilibrium plot (working grid)
    # =========================================================================
    print("\nWorking grid: tracing the bond-market supply curve")
    r_grid_plot = np.linspace(-0.02, 0.049, 20)
    S_grid_plot = compute_supply_curve(params, r_grid_plot)

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()



    # =========================================================================
    # Figures
    # =========================================================================

    # --- Figure 1: Value Function (with reference overlay) ---
    fig1, ax1 = plt.subplots()
    ax1.plot(a, V[:, 0], "b-", linewidth=2, label=f"$V_L(a)$, $z_L = {z[0]}$")
    ax1.plot(a, V[:, 1], "r-", linewidth=2, label=f"$V_H(a)$, $z_H = {z[1]}$")
    ax1.plot(a_ref, V_ref[:, 0], "b--", linewidth=1.0, alpha=0.7,
             label=f"reference ({I_ref} pts)")
    ax1.plot(a_ref, V_ref[:, 1], "r--", linewidth=1.0, alpha=0.7)
    ax1.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Assets $a$")
    ax1.set_ylabel("$V_i(a)$")
    ax1.set_title("Value Function at the Equilibrium $r^{\\ast}$")
    ax1.legend(loc="lower right")
    save_figure(fig1, "figures/value-function.png", dpi=150)

    # --- Figure 2: Savings Policy (with reference overlay) ---
    fig2, ax2 = plt.subplots()
    ax2.plot(a, s[:, 0], "b-", linewidth=2, label=f"$s_L(a)$, $z_L = {z[0]}$")
    ax2.plot(a, s[:, 1], "r-", linewidth=2, label=f"$s_H(a)$, $z_H = {z[1]}$")
    ax2.plot(a_ref, s_ref[:, 0], "b--", linewidth=1.0, alpha=0.7,
             label=f"reference ({I_ref} pts)")
    ax2.plot(a_ref, s_ref[:, 1], "r--", linewidth=1.0, alpha=0.7)
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax2.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Assets $a$")
    ax2.set_ylabel("Savings drift $s_i(a) = z_i + r^{\\ast} a - c_i(a)$")
    ax2.set_title("Savings Policy by Income State")
    ax2.set_xlim([a_min - 0.03, 1.0])
    ax2.legend(loc="upper right")
    save_figure(fig2, "figures/savings-policy.png", dpi=150)

    # --- Figure 3: Stationary Wealth Distribution (with reference overlay) ---
    fig3, ax3 = plt.subplots()
    ax3.plot(a, g[:, 0], "b-", linewidth=2, label=f"$g_L(a)$, $z_L = {z[0]}$")
    ax3.plot(a, g[:, 1], "r-", linewidth=2, label=f"$g_H(a)$, $z_H = {z[1]}$")
    ax3.plot(a_ref, g_ref[:, 0], "b--", linewidth=1.0, alpha=0.7,
             label=f"reference ({I_ref} pts)")
    ax3.plot(a_ref, g_ref[:, 1], "r--", linewidth=1.0, alpha=0.7)
    ax3.axvline(a_min, color="k", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("Assets $a$")
    ax3.set_ylabel("Density $g_i(a)$")
    ax3.set_title("Stationary Wealth Distribution by Income State")
    ax3.set_xlim([a_min - 0.03, 1.0])
    ax3.legend(loc="upper right")
    save_figure(fig3, "figures/wealth-distribution.png", dpi=150)

    # --- Figure 4: Bond-market clearing ---
    fig4, ax4 = plt.subplots()
    ax4.plot(S_grid_plot, r_grid_plot, "b-", linewidth=2, label="$S(r)$ on working grid")
    ax4.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax4.axhline(rho, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.plot(0, r_eq, "ro", markersize=8, zorder=5,
             label=f"$r^{{\\ast}} = {r_eq:.5f}$")
    ax4.plot(0, r_eq_ref, "kx", markersize=8, zorder=5,
             label=f"$r^{{\\ast}}_{{\\rm ref}} = {r_eq_ref:.4f}$")
    ax4.set_xlabel("Aggregate asset demand $S(r) = \\int a\\,(g_L + g_H)\\,da$")
    ax4.set_ylabel("Interest rate $r$")
    ax4.set_title("Bond Market Clearing")
    ax4.set_xlim([-0.12, 0.12])
    ax4.text(0.04, rho + 0.001, "$r = \\rho$ (complete markets)", fontsize=9, color="gray")
    ax4.legend(loc="lower right")
    save_figure(fig4, "figures/bond-market.png", dpi=150)

    # =========================================================================
    # Equilibrium summary table
    # =========================================================================
    table_data = {
        "Statistic": [
            "Discount rate rho",
            "Equilibrium r* (working grid)",
            "Equilibrium r* (reference grid)",
            "Precautionary wedge rho - r*",
            "Mean wealth E[a]",
            "Mean income E[z]",
            "Mean consumption E[c]",
            f"Mass within {constraint_window} of borrowing limit",
            "Prob(z = z_low)",
            "Prob(z = z_high)",
            "Bond-market residual abs(S(r*))",
            "r* gap, working vs reference",
            "Sup-norm savings gap, a in [a_min, 1]",
            "Sup-norm value gap, a in [a_min, 1]",
            "Relative value gap (% of value scale)",
            "HJB iterations (working)",
            "HJB sup-norm change (working)",
        ],
        "Value": [
            f"{rho:.4f}",
            f"{r_eq:.5f}",
            f"{r_eq_ref:.5f}",
            f"{wedge:.5f}",
            f"{market_residual:.2e}",
            f"{(g[:, 0] @ (np.ones(I) * z[0])) * da + (g[:, 1] @ (np.ones(I) * z[1])) * da:.4f}",
            f"{mean_cons:.4f}",
            f"{mass_at_constraint:.4f}",
            f"{prob_z_low:.4f}",
            f"{prob_z_high:.4f}",
            f"{market_residual:.2e}",
            f"{r_gap:.2e}",
            f"{s_gap:.2e}",
            f"{V_gap:.2e}",
            f"{100 * V_gap_rel:.3f}%",
            f"{info['iterations']}",
            f"{info['error']:.2e}",
        ],
    }
    df = pd.DataFrame(table_data)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/equilibrium.csv", index=False)

    save_thumbnail("figures/value-function.png", "figures/thumb.png")
    print("\nFigures and tables written.")


if __name__ == "__main__":
    main()
