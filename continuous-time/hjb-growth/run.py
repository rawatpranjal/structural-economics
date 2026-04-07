#!/usr/bin/env python3
"""Continuous-Time Neoclassical Growth Model via HJB Equation.

Solves the neoclassical growth model in continuous time using the upwind
finite difference method of Achdou et al. (2022). The HJB equation

    rho*V(k) = max_c { u(c) + V'(k)*(f(k) - delta*k - c) }

is discretized on a capital grid and solved via forward iteration with an
explicit CFL-stable time step (following Moll's MATLAB code). Optimal
consumption is recovered from the FOC u'(c) = V'(k). Results are compared
with a discrete-time VFI solution.

References:
    Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022).
        "Income and Wealth Distribution in Macroeconomics: A Continuous-Time
        Approach." Review of Economic Studies, 89(1), 45-86.
    Moll, B. (2022). Lecture notes on continuous-time methods in
        macroeconomics. https://benjaminmoll.com/lectures/
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


# =============================================================================
# Utility and production
# =============================================================================

def crra_utility(c, sigma):
    """CRRA utility u(c) = c^(1-sigma)/(1-sigma), log when sigma=1."""
    c = np.maximum(c, 1e-15)
    if sigma == 1.0:
        return np.log(c)
    return c ** (1 - sigma) / (1 - sigma)


def production(k, A, alpha):
    """Cobb-Douglas production f(k) = A * k^alpha."""
    return A * k ** alpha


# =============================================================================
# Continuous-time HJB solver (upwind finite differences)
# =============================================================================

def solve_hjb_growth(params):
    """Solve the continuous-time neoclassical growth HJB via implicit upwind FD.

    Uses an implicit time-stepping scheme with upwind finite differences
    following Achdou et al. (2022) and Moll's lecture notes. At each
    iteration the consumption policy is computed from the FOC, then the
    upwind transition matrix A is constructed and the linear system

        (1/Delta + rho) V^{n+1} - A V^{n+1} = u(c^n) + V^n / Delta

    is solved via sparse LU. The implicit scheme is unconditionally stable,
    allowing a large pseudo-time step (Delta = 1000) for fast convergence.

    Returns:
        v: value function on the capital grid
        c: consumption policy
        kdot: savings/investment policy (dk/dt = f(k) - delta*k - c)
        info: dict with convergence information
    """
    rho = params["rho"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    delta = params["delta"]
    A_tfp = params["A"]
    k = params["k"]
    N = len(k)
    dk = k[1] - k[0]
    max_iter = params["max_iter"]
    tol = params["tol"]
    Delta = 1000.0  # large implicit time step (unconditionally stable)

    # Production on grid
    f_k = production(k, A_tfp, alpha)

    # Initial guess: consume all output (V ~ u(f(k))/rho)
    v = crra_utility(f_k, sigma) / rho

    dVf = np.zeros(N)
    dVb = np.zeros(N)

    convergence = []

    for n in range(1, max_iter + 1):
        V = v.copy()

        # Forward difference
        dVf[:N-1] = (V[1:N] - V[:N-1]) / dk
        dVf[N-1] = 0.0  # will never be used (boundary)

        # Backward difference
        dVb[1:N] = (V[1:N] - V[:N-1]) / dk
        dVb[0] = 0.0  # will never be used (boundary)

        # Consumption and savings from forward difference
        cf = np.maximum(dVf, 1e-15) ** (-1.0 / sigma)
        muf = f_k - delta * k - cf  # drift with forward difference

        # Consumption and savings from backward difference
        cb = np.maximum(dVb, 1e-15) ** (-1.0 / sigma)
        mub = f_k - delta * k - cb  # drift with backward difference

        # Consumption at steady state (zero savings)
        c0 = f_k - delta * k
        dV0 = np.maximum(c0, 1e-15) ** (-sigma)

        # Upwind scheme: choose based on sign of drift
        If = (muf > 0).astype(float)   # positive drift -> forward difference
        Ib = (mub < 0).astype(float)   # negative drift -> backward difference
        I0 = 1.0 - If - Ib             # at or near steady state

        # Enforce boundary conditions
        Ib[0] = 0.0; If[0] = 1.0       # left boundary: use forward
        Ib[N-1] = 1.0; If[N-1] = 0.0   # right boundary: use backward

        dV_upwind = dVf * If + dVb * Ib + dV0 * I0

        # Optimal consumption from upwind derivative
        c = np.maximum(dV_upwind, 1e-15) ** (-1.0 / sigma)
        u_c = crra_utility(c, sigma)

        # Build upwind transition matrix A (tridiagonal)
        # Positive part of drift (forward) and negative part (backward)
        sf_pos = np.maximum(f_k - delta * k - c, 0.0)  # forward drift
        sb_neg = np.minimum(f_k - delta * k - c, 0.0)  # backward drift

        # Sub-diagonal (from backward difference): -sb_neg/dk
        X = -sb_neg / dk
        # Super-diagonal (from forward difference): sf_pos/dk
        Z = sf_pos / dk
        # Main diagonal
        Y = -Z - X

        # Sparse tridiagonal matrix
        A_mat = (sparse.diags(Y, 0, shape=(N, N))
                 + sparse.diags(X[1:N], -1, shape=(N, N))
                 + sparse.diags(Z[:N-1], 1, shape=(N, N)))
        A_mat = A_mat.tocsc()

        # Implicit update: ((1/Delta + rho)*I - A) * V_new = u + V/Delta
        B = (1.0 / Delta + rho) * sparse.eye(N, format="csc") - A_mat
        b = u_c + V / Delta
        v_new = spsolve(B, b)

        change = np.max(np.abs(v_new - V))
        convergence.append(change)
        v = v_new

        if n % 10 == 0:
            print(f"  HJB iteration {n:4d}, change = {change:.2e}")

        if change < tol:
            print(f"  HJB converged in {n} iterations (change = {change:.2e})")
            break

    if change >= tol:
        print(f"  HJB did NOT converge after {max_iter} iterations (change = {change:.2e})")

    # Recompute final policies at converged V
    dVf[:N-1] = (v[1:N] - v[:N-1]) / dk
    dVf[N-1] = 0.0
    dVb[1:N] = (v[1:N] - v[:N-1]) / dk
    dVb[0] = 0.0

    cf = np.maximum(dVf, 1e-15) ** (-1.0 / sigma)
    muf = f_k - delta * k - cf
    cb = np.maximum(dVb, 1e-15) ** (-1.0 / sigma)
    mub = f_k - delta * k - cb
    c0 = f_k - delta * k
    dV0 = np.maximum(c0, 1e-15) ** (-sigma)

    If = (muf > 0).astype(float)
    Ib = (mub < 0).astype(float)
    I0 = 1.0 - If - Ib
    Ib[0] = 0.0; If[0] = 1.0
    Ib[N-1] = 1.0; If[N-1] = 0.0

    dV_upwind = dVf * If + dVb * Ib + dV0 * I0
    c = np.maximum(dV_upwind, 1e-15) ** (-1.0 / sigma)
    kdot = f_k - delta * k - c

    info = {
        "iterations": n,
        "converged": change < tol,
        "error": change,
        "convergence": convergence,
    }

    return v, c, kdot, info


# =============================================================================
# Discrete-time VFI solver (for comparison)
# =============================================================================

def solve_discrete_vfi(params):
    """Solve the discrete-time neoclassical growth model by VFI.

    Bellman equation:
        V(k) = max_{k'} { u(f(k) + (1-delta)*k - k') + beta*V(k') }

    Returns:
        v: value function
        c: consumption policy
        info: dict with convergence information
    """
    rho = params["rho"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    delta = params["delta"]
    A = params["A"]
    k = params["k"]
    I = len(k)
    tol = params["tol"]

    beta = 1.0 / (1.0 + rho)  # continuous-time rho -> discrete beta

    # Total resources at each k: f(k) + (1-delta)*k
    f_k = production(k, A, alpha)
    resources = f_k + (1.0 - delta) * k  # (I,)

    # Initial guess
    v = crra_utility(f_k, sigma) / rho

    for n in range(1, 2001):
        v_old = v.copy()

        # Vectorized: consumption = resources[i] - k[j] for all (i,j)
        # resources is (I,), k is (I,); form (I, I) matrix
        c_mat = resources[:, np.newaxis] - k[np.newaxis, :]  # (I, I)
        c_mat = np.where(c_mat > 0, c_mat, np.nan)

        val_mat = crra_utility(np.where(np.isnan(c_mat), 1e-15, c_mat), sigma)
        val_mat = np.where(np.isnan(c_mat), -1e15, val_mat)
        val_mat += beta * v_old[np.newaxis, :]  # (I, I)

        best_idx = np.nanargmax(val_mat, axis=1)
        v = val_mat[np.arange(I), best_idx]
        c_policy = resources - k[best_idx]

        error = np.max(np.abs(v - v_old))
        if n % 100 == 0:
            print(f"  VFI iteration {n:3d}, error = {error:.2e}")
        if error < tol:
            print(f"  VFI converged in {n} iterations (error = {error:.2e})")
            break

    info = {"iterations": n, "converged": error < tol, "error": error}
    return v, c_policy, info


# =============================================================================
# Transition dynamics
# =============================================================================

def simulate_transition(c_interp_func, params, k0_values, T=100):
    """Simulate transition dynamics dk/dt = f(k) - delta*k - c(k).

    Args:
        c_interp_func: callable, consumption policy c(k)
        params: model parameters
        k0_values: list of initial capital levels
        T: time horizon

    Returns:
        dict mapping k0 -> (t_array, k_array)
    """
    alpha = params["alpha"]
    delta = params["delta"]
    A = params["A"]

    def kdot_func(t, k_val):
        k_val = np.atleast_1d(k_val)
        f_val = production(k_val, A, alpha)
        c_val = c_interp_func(k_val)
        return f_val - delta * k_val - c_val

    paths = {}
    t_span = (0, T)
    t_eval = np.linspace(0, T, 500)

    for k0 in k0_values:
        sol = solve_ivp(kdot_func, t_span, [k0], t_eval=t_eval,
                        method="RK45", rtol=1e-8, atol=1e-10)
        paths[k0] = (sol.t, sol.y[0])

    return paths


# =============================================================================
# Main
# =============================================================================

def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    rho = 0.05      # Discount rate
    sigma = 2.0     # CRRA coefficient
    alpha = 0.36    # Capital share
    delta = 0.05    # Depreciation rate
    A = 1.0         # TFP

    # Steady-state capital: f'(k_ss) = rho + delta
    k_ss = (alpha * A / (rho + delta)) ** (1.0 / (1.0 - alpha))
    c_ss = production(k_ss, A, alpha) - delta * k_ss
    y_ss = production(k_ss, A, alpha)

    print(f"Steady state: k_ss = {k_ss:.4f}, c_ss = {c_ss:.4f}, y_ss = {y_ss:.4f}")

    # Capital grid
    n_k = 500
    k_min = 0.1
    k_max = 2.0 * k_ss
    k_grid = np.linspace(k_min, k_max, n_k)

    params = {
        "rho": rho, "sigma": sigma, "alpha": alpha, "delta": delta, "A": A,
        "k": k_grid, "max_iter": 20000, "tol": 1e-6,
    }

    # =========================================================================
    # Solve continuous-time HJB
    # =========================================================================
    print("\n--- Continuous-Time HJB (Upwind Finite Differences) ---")
    v_ct, c_ct, kdot_ct, info_ct = solve_hjb_growth(params)

    # =========================================================================
    # Solve discrete-time VFI (coarser grid for speed)
    # =========================================================================
    print("\n--- Discrete-Time VFI ---")
    n_k_dt = 200
    k_grid_dt = np.linspace(k_min, k_max, n_k_dt)
    params_dt = params.copy()
    params_dt["k"] = k_grid_dt
    v_dt, c_dt, info_dt = solve_discrete_vfi(params_dt)

    # =========================================================================
    # Transition dynamics from different initial conditions
    # =========================================================================
    print("\n--- Transition Dynamics ---")

    def c_interp(k_val):
        """Interpolate consumption policy onto arbitrary k values."""
        return np.interp(k_val, k_grid, c_ct)

    k0_values = [0.5 * k_ss, 0.75 * k_ss, 1.25 * k_ss, 1.5 * k_ss]
    paths = simulate_transition(c_interp, params, k0_values, T=100)

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Continuous-Time Neoclassical Growth (HJB)",
        "Neoclassical growth model solved via the Hamilton-Jacobi-Bellman "
        "equation using upwind finite differences.",
    )

    report.add_overview(
        "The neoclassical growth model is the workhorse of macroeconomics. In continuous "
        "time, the social planner's problem is characterized by a Hamilton-Jacobi-Bellman "
        "(HJB) equation. Rather than iterating on a Bellman operator with a costly max "
        "step (as in discrete-time VFI), the continuous-time approach exploits the first-order "
        "condition to eliminate the maximization analytically. With CRRA utility, optimal "
        "consumption is $c = (V'(k))^{-1/\\sigma}$, so the HJB becomes a nonlinear ODE "
        "that can be solved by simple forward iteration on a finite-difference grid.\n\n"
        "This module implements the upwind finite difference scheme from Moll's growth.m "
        "MATLAB code: forward differences when the agent is accumulating capital (below "
        "steady state) and backward differences when decumulating (above steady state)."
    )

    report.add_equations(r"""
**HJB equation:**
$$\rho \, V(k) = \max_{c} \left\{ u(c) + V'(k) \left[ f(k) - \delta k - c \right] \right\}$$

**FOC:** $u'(c) = V'(k) \implies c^*(k) = \left( V'(k) \right)^{-1/\sigma}$

**Production:** $f(k) = A k^\alpha$

**CRRA utility:** $u(c) = \frac{c^{1-\sigma}}{1-\sigma}$

**Capital accumulation:** $\dot{k} = f(k) - \delta k - c$

**Steady state:** $f'(k_{ss}) = \rho + \delta \implies k_{ss} = \left( \frac{\alpha A}{\rho + \delta} \right)^{1/(1-\alpha)}$

**Upwind finite difference:**
$$V'(k_i) \approx \begin{cases} \frac{V_{i+1} - V_i}{\Delta k} & \text{if } \dot{k}_i > 0 \\ \frac{V_i - V_{i-1}}{\Delta k} & \text{if } \dot{k}_i < 0 \end{cases}$$
""")

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$   | {rho} | Discount rate |\n"
        f"| $\\sigma$ | {sigma} | CRRA coefficient |\n"
        f"| $\\alpha$ | {alpha} | Capital share |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $A$       | {A} | TFP |\n"
        f"| Grid      | {n_k} points | $k \\in [{k_min}, {k_max:.2f}]$ |\n"
        f"| $k_{{ss}}$ | {k_ss:.4f} | Steady-state capital |\n"
        f"| $c_{{ss}}$ | {c_ss:.4f} | Steady-state consumption |\n"
        f"| $y_{{ss}}$ | {y_ss:.4f} | Steady-state output |"
    )

    report.add_solution_method(
        "**Upwind finite difference iteration** (Moll 2022, Achdou et al. 2022): "
        "At each iteration, $V'(k)$ is approximated by forward or backward "
        "differences depending on the sign of the drift $\\dot{k} = f(k) - \\delta k - c$. "
        "The FOC $c = (V')^{-1/\\sigma}$ yields consumption without a grid search.\n\n"
        "The value function is updated explicitly:\n"
        "$$V^{n+1} = V^n + \\Delta \\cdot \\left[ u(c^n) + (V^n)' \\cdot (f(k) - \\delta k - c^n) "
        "- \\rho V^n \\right]$$\n"
        "where $\\Delta$ satisfies the CFL stability condition "
        "$\\Delta \\leq 0.9 \\cdot \\Delta k / \\max(f(k) - \\delta k)$.\n\n"
        f"Continuous-time HJB converged in **{info_ct['iterations']} iterations** "
        f"(residual = {info_ct['error']:.2e}).\n\n"
        f"Discrete-time VFI (on {n_k_dt}-point grid) converged in "
        f"**{info_dt['iterations']} iterations** (error = {info_dt['error']:.2e})."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, v_ct, "b-", linewidth=2, label="Continuous-time (HJB)")
    ax1.plot(k_grid_dt, v_dt, "r--", linewidth=1.5, label="Discrete-time (VFI)")
    ax1.axvline(k_ss, color="k", linestyle=":", linewidth=0.8, alpha=0.6,
                label=f"$k_{{ss}} = {k_ss:.2f}$")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k)$")
    ax1.set_title("Value Function")
    ax1.legend()
    report.add_figure(
        "figures/value-function.png",
        "Value function from continuous-time HJB vs discrete-time VFI",
        fig1,
    )

    # --- Figure 2: Consumption Policy ---
    fig2, ax2 = plt.subplots()
    ax2.plot(k_grid, c_ct, "b-", linewidth=2, label="Continuous-time (HJB)")
    ax2.plot(k_grid_dt, c_dt, "r--", linewidth=1.5, label="Discrete-time (VFI)")
    ax2.axvline(k_ss, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.plot(k_ss, c_ss, "ko", markersize=8, zorder=5,
             label=f"Steady state ($k_{{ss}}={k_ss:.2f}$, $c_{{ss}}={c_ss:.2f}$)")
    ax2.set_xlabel("Capital $k$")
    ax2.set_ylabel("Consumption $c(k)$")
    ax2.set_title("Consumption Policy Function")
    ax2.legend()
    report.add_figure(
        "figures/consumption-policy.png",
        "Consumption policy c(k) with analytical steady state marked",
        fig2,
    )

    # --- Figure 3: Savings / Investment Policy ---
    fig3, ax3 = plt.subplots()
    ax3.plot(k_grid, kdot_ct, "b-", linewidth=2, label=r"$\dot{k} = f(k) - \delta k - c(k)$")
    ax3.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax3.axvline(k_ss, color="k", linestyle=":", linewidth=0.8, alpha=0.6,
                label=f"$k_{{ss}} = {k_ss:.2f}$")
    ax3.fill_between(k_grid, kdot_ct, 0, where=(kdot_ct > 0),
                     alpha=0.15, color="green", label="Capital accumulation")
    ax3.fill_between(k_grid, kdot_ct, 0, where=(kdot_ct < 0),
                     alpha=0.15, color="red", label="Capital decumulation")
    ax3.set_xlabel("Capital $k$")
    ax3.set_ylabel(r"$\dot{k}$")
    ax3.set_title("Savings / Investment Policy")
    ax3.legend(fontsize=9)
    report.add_figure(
        "figures/savings-policy.png",
        "Savings policy s(k) = f(k) - delta*k - c(k); zero crossing at steady state",
        fig3,
    )

    # --- Figure 4: Transition Dynamics ---
    fig4, ax4 = plt.subplots()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, k0 in enumerate(k0_values):
        t_arr, k_arr = paths[k0]
        label_str = f"$k_0 = {k0:.2f}$ ({k0/k_ss:.0%} of $k_{{ss}}$)"
        ax4.plot(t_arr, k_arr, color=colors[i], linewidth=2, label=label_str)
    ax4.axhline(k_ss, color="k", linestyle="--", linewidth=1, alpha=0.7,
                label=f"$k_{{ss}} = {k_ss:.2f}$")
    ax4.set_xlabel("Time $t$")
    ax4.set_ylabel("Capital $k(t)$")
    ax4.set_title("Transition Dynamics")
    ax4.legend(fontsize=9)
    report.add_figure(
        "figures/transition-dynamics.png",
        "Transition dynamics k(t) from different initial conditions converging to steady state",
        fig4,
    )

    # --- Table: Steady-State Values ---
    # Compute numerical steady state from the savings policy
    ss_idx = np.argmin(np.abs(kdot_ct))
    k_ss_num = k_grid[ss_idx]
    c_ss_num = c_ct[ss_idx]
    y_ss_num = production(k_ss_num, A, alpha)
    inv_ss_num = delta * k_ss_num
    saving_rate = inv_ss_num / y_ss_num

    table_data = {
        "Variable": [
            "$k_{ss}$ (capital)",
            "$c_{ss}$ (consumption)",
            "$y_{ss}$ (output)",
            "$i_{ss} = \\delta k_{ss}$ (investment)",
            "$s = i/y$ (saving rate)",
            "$f'(k_{ss})$ (MPK)",
            "HJB iterations",
            "HJB residual",
            "VFI iterations",
        ],
        "Analytical": [
            f"{k_ss:.4f}",
            f"{c_ss:.4f}",
            f"{y_ss:.4f}",
            f"{delta * k_ss:.4f}",
            f"{delta * k_ss / y_ss:.4f}",
            f"{rho + delta:.4f}",
            "--",
            "--",
            "--",
        ],
        "Numerical": [
            f"{k_ss_num:.4f}",
            f"{c_ss_num:.4f}",
            f"{y_ss_num:.4f}",
            f"{inv_ss_num:.4f}",
            f"{saving_rate:.4f}",
            f"{alpha * A * k_ss_num ** (alpha - 1):.4f}",
            f"{info_ct['iterations']}",
            f"{info_ct['error']:.2e}",
            f"{info_dt['iterations']}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/steady-state.csv",
        "Steady-State Values: Analytical vs Numerical",
        df,
    )

    report.add_takeaway(
        "The continuous-time approach to the neoclassical growth model offers significant "
        "computational advantages over discrete-time VFI:\n\n"
        "**Key insights:**\n"
        "- **No max operator:** With CRRA utility, the FOC $u'(c) = V'(k)$ gives "
        "$c = (V')^{-1/\\sigma}$ in closed form. The costly grid search over consumption "
        "choices in discrete-time VFI is replaced by a simple algebraic formula. This is "
        "the core advantage of the continuous-time formulation.\n"
        "- **Upwind scheme is essential:** The finite difference direction must match the "
        "drift direction. Below steady state ($k < k_{ss}$), capital is accumulating "
        "($\\dot{k} > 0$), so the forward difference is used. Above steady state, the "
        "backward difference is used. Using the wrong direction creates numerical "
        "instability.\n"
        "- **CFL condition:** The explicit time step $\\Delta$ must satisfy "
        "$\\Delta \\leq \\Delta k / \\max |\\dot{k}|$ for stability. This is a well-known "
        "constraint from the PDE literature.\n"
        "- **Saddle-path stability:** All initial conditions converge monotonically to the "
        "unique steady state $k_{ss}$. The speed of convergence depends on the curvature "
        "of the production function and the discount rate.\n"
        "- **Scalability:** The method extends naturally to heterogeneous-agent models "
        "(Achdou et al. 2022) where the state space includes both individual capital and "
        "the cross-sectional distribution."
    )

    report.add_references([
        "Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., and Moll, B. (2022). "
        "\"Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach.\" "
        "*Review of Economic Studies*, 89(1), 45-86.",
        "Moll, B. (2022). \"Lecture notes on continuous-time methods in macroeconomics.\" "
        "https://benjaminmoll.com/lectures/",
        "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
