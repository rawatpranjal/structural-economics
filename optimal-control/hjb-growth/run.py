#!/usr/bin/env python3
"""Continuous-time neoclassical growth solved from the HJB equation.

The planner chooses consumption in continuous time while capital follows

    dk/dt = f(k) - delta*k - c.

The Hamilton-Jacobi-Bellman equation is discretized on a capital grid and
solved with the implicit upwind finite-difference scheme used in Achdou et al.
(2022) and Moll's continuous-time macro notes. The first-order condition turns
the value derivative into consumption, so the algorithm avoids a grid search
over controls.

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

def solve_hjb_growth(params, verbose=True):
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

        if verbose and n % 10 == 0:
            print(f"  HJB iteration {n:4d}, change = {change:.2e}")

        if change < tol:
            if verbose:
                print(f"  HJB converged in {n} iterations (change = {change:.2e})")
            break

    if verbose and change >= tol:
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
        "k": k_grid, "max_iter": 500, "tol": 1e-6,
    }

    # =========================================================================
    # Solve continuous-time HJB
    # =========================================================================
    print("\n--- Continuous-Time HJB (Upwind Finite Differences) ---")
    v_ct, c_ct, kdot_ct, info_ct = solve_hjb_growth(params)

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
        "Ramsey Capital Accumulation by HJB Upwinding",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A Ramsey planner inherits aggregate capital $k$. Output can be consumed today "
        "or invested for future production. Scarce capital raises investment value. "
        "Abundant capital makes current consumption cheaper.\n\n"
        "The object is the consumption policy $c(k)$ and the capital drift $\\dot{k}$. "
        "Together they describe how the economy returns to its steady state.\n\n"
        "The HJB gives the value of starting from each capital stock. Its derivative is "
        "the shadow value that pins down consumption. A finite-difference scheme is "
        "needed because the nonlinear HJB has no closed-form policy on the grid. "
        "Upwinding chooses the derivative side using the policy-implied drift."
    )

    report.add_equations(r"""
The planner solves

$$
\max_{\{c(t)\}_{t \geq 0}}
\int_0^\infty e^{-\rho t} u(c(t))\,dt
\quad\text{s.t.}\quad
\dot{k}(t)=f(k(t))-\delta k(t)-c(t),
$$

Here $f(k)=Ak^\alpha$ and $u(c)=c^{1-\sigma}/(1-\sigma)$ for
$\sigma \neq 1$. The parameter $\rho$ is the continuous-time discount rate.

The HJB equation is

$$
\rho V(k)=\max_{c>0}
\left[u(c)+V'(k)\left(f(k)-\delta k-c\right)\right].
$$

The first-order condition is

$$
u'(c^{\ast}(k))=V'(k)
\quad\Longrightarrow\quad
c^{\ast}(k)=\left(V'(k)\right)^{-1/\sigma}.
$$

Substituting this policy into the drift

$$
s(k)=\dot{k}=f(k)-\delta k-c^{\ast}(k)
$$

leaves a nonlinear equation in $V$. On the grid $k_1,\ldots,k_N$ with spacing
$\Delta k$, the upwind derivative uses the direction implied by the drift:

$$
D_i V =
\begin{cases}
(V_{i+1}-V_i)/\Delta k, & s_i>0,\\
(V_i-V_{i-1})/\Delta k, & s_i<0,\\
\left(f(k_i)-\delta k_i\right)^{-\sigma}, & s_i=0.
\end{cases}
$$

The steady state satisfies $s(k_{ss})=0$ and the Euler condition
$f'(k_{ss})=\rho+\delta$, so

$$
k_{ss}=\left(\frac{\alpha A}{\rho+\delta}\right)^{1/(1-\alpha)}.
$$
""")

    report.add_model_setup(
        "The calibration uses one aggregate capital state, Cobb-Douglas production, "
        "CRRA utility, and no shocks. The grid spans low and high capital around the "
        "Ramsey steady state.\n\n"
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\rho$   | {rho} | Discount rate |\n"
        f"| $\\sigma$ | {sigma} | CRRA coefficient |\n"
        f"| $\\alpha$ | {alpha} | Capital share |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $A$       | {A} | TFP |\n"
        f"| Baseline HJB grid | {n_k} points | $k \\in [{k_min}, {k_max:.2f}]$ |\n"
        f"| $k_{{ss}}$ | {k_ss:.4f} | Steady-state capital |\n"
        f"| $c_{{ss}}$ | {c_ss:.4f} | Steady-state consumption |\n"
        f"| $y_{{ss}}$ | {y_ss:.4f} | Steady-state output |"
    )

    report.add_solution_method(
        "Start from a guessed value function on the capital grid. The update compares "
        "forward and backward slopes, maps each slope into consumption, and chooses the "
        "slope consistent with capital drift. That choice aligns the derivative with "
        "the economic law of motion.\n\n"
        "```text\n"
        "Inputs: grid {k_i}, primitives (rho, sigma, alpha, delta, A), tolerance eps\n"
        "Initialize V^0_i = u(f(k_i)) / rho\n"
        "For n = 0, 1, ... until ||V^{n+1} - V^n||_infinity < eps:\n"
        "    1. Form forward and backward slopes D^+ V^n_i and D^- V^n_i.\n"
        "    2. Use the FOC to compute candidate consumption:\n"
        "       c^+_i = (D^+ V^n_i)^(-1/sigma), c^-_i = (D^- V^n_i)^(-1/sigma).\n"
        "    3. Compute candidate drifts s^+_i = f(k_i) - delta k_i - c^+_i\n"
        "       and s^-_i = f(k_i) - delta k_i - c^-_i.\n"
        "    4. Choose the upwind derivative D_i V^n using the sign of the drift;\n"
        "       at s_i = 0 use the steady-state marginal utility.\n"
        "    5. Set c^n_i = (D_i V^n)^(-1/sigma) and build the tridiagonal\n"
        "       generator G^n from the positive and negative drift parts.\n"
        "    6. Solve the implicit linear system\n"
        "       [(1/Delta + rho) I - G^n] V^{n+1}\n"
        "       = u(c^n) + V^n / Delta.\n"
        "Output: value V, consumption policy c(k), and drift s(k)=dot{k}\n"
        "```\n\n"
        "The implicit step solves one sparse tridiagonal linear system. The large "
        "pseudo-time step $\\Delta=1000$ is numerical, not a model period. It stabilizes "
        "the fixed-point update while keeping the continuous-time HJB as the target.\n\n"
        f"The HJB converged in **{info_ct['iterations']} iterations** with final change "
        f"{info_ct['error']:.2e}."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, v_ct, color="#1f77b4", linewidth=2.1,
             label="Upwind HJB")
    ax1.axvline(k_ss, color="k", linestyle=":", linewidth=0.8, alpha=0.6,
                label=f"$k_{{ss}} = {k_ss:.2f}$")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k)$")
    ax1.set_title("Value of Capital")
    ax1.legend()
    report.add_figure(
        "figures/value-function.png",
        "Value function from the upwind HJB",
        fig1,
        description="The value function is increasing and concave. Extra capital raises "
        "future consumption, but diminishing marginal product lowers the marginal gain.",
    )

    # --- Figure 2: Consumption Policy ---
    fig2, ax2 = plt.subplots()
    net_output = production(k_grid, A, alpha) - delta * k_grid
    ax2.plot(k_grid, c_ct, color="#1f77b4", linewidth=2.1,
             label="Upwind HJB")
    ax2.plot(k_grid, net_output, color="#6b6b6b", linestyle=":", linewidth=1.5,
             label=r"Net output $f(k)-\delta k$")
    ax2.axvline(k_ss, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.plot(k_ss, c_ss, "ko", markersize=8, zorder=5,
             label=f"Steady state ($k_{{ss}}={k_ss:.2f}$, $c_{{ss}}={c_ss:.2f}$)")
    ax2.set_xlabel("Capital $k$")
    ax2.set_ylabel("Consumption $c(k)$")
    ax2.set_title("Consumption Policy")
    ax2.legend()
    report.add_figure(
        "figures/consumption-policy.png",
        "Consumption policy and net output",
        fig2,
        description="The consumption rule comes from marginal value. Below the steady "
        "state, consumption stays below net output, so capital rises. Above it, "
        "consumption exceeds net output, so capital falls.",
    )

    # --- Figure 3: Savings / Investment Policy ---
    fig3, ax3 = plt.subplots()
    ax3.plot(k_grid, kdot_ct, color="#1f77b4", linewidth=2.1,
             label=r"Drift $\dot{k}$")
    ax3.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax3.axvline(k_ss, color="k", linestyle=":", linewidth=0.8, alpha=0.6,
                label=f"$k_{{ss}} = {k_ss:.2f}$")
    ax3.fill_between(k_grid, kdot_ct, 0, where=(kdot_ct > 0),
                     alpha=0.15, color="green", label="Capital accumulation")
    ax3.fill_between(k_grid, kdot_ct, 0, where=(kdot_ct < 0),
                     alpha=0.15, color="red", label="Capital decumulation")
    ax3.set_xlabel("Capital $k$")
    ax3.set_ylabel(r"$\dot{k}$")
    ax3.set_title("Capital Drift")
    ax3.legend(fontsize=9)
    report.add_figure(
        "figures/savings-policy.png",
        "Capital drift with accumulation below steady state and decumulation above it",
        fig3,
        description="The drift $s(k)=\\dot{k}$ drives transitions and selects the "
        "upwind derivative. Positive drift points to capital accumulation. Negative "
        "drift points to decumulation. The zero crossing is the Ramsey steady state.",
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
    ax4.set_title("Transition Paths")
    ax4.legend(fontsize=9)
    report.add_figure(
        "figures/transition-dynamics.png",
        "Transition dynamics k(t) from different initial conditions converging to steady state",
        fig4,
        description="The policy-implied law of motion sends each initial capital stock "
        "toward $k_{ss}$. Low-capital economies invest because marginal product is high. "
        "High-capital economies consume more than net output and move down.",
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
        ],
        "Baseline HJB": [
            f"{k_ss_num:.4f}",
            f"{c_ss_num:.4f}",
            f"{y_ss_num:.4f}",
            f"{inv_ss_num:.4f}",
            f"{saving_rate:.4f}",
            f"{alpha * A * k_ss_num ** (alpha - 1):.4f}",
            f"{info_ct['iterations']}",
            f"{info_ct['error']:.2e}",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/steady-state.csv",
        "Steady-State Values and HJB Diagnostics",
        df,
        description="The closed-form steady state checks the finite-difference solution. "
        "The grid locates zero drift within one step.",
    )

    report.add_takeaway(
        "The computed policy follows the Ramsey Euler logic. Investment is high when "
        "capital has high marginal product. Consumption rises once capital is abundant. "
        "The path converges to $f'(k)=\\rho+\\delta$.\n\n"
        "The HJB turns this logic into a value derivative. Upwinding uses the direction "
        "of capital movement to choose the derivative. After that choice, the update is "
        "a sparse linear solve."
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
