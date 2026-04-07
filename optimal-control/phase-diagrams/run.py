#!/usr/bin/env python3
"""Phase Diagrams for Continuous-Time Economic Models.

Visualizes the dynamics of the Ramsey optimal growth model in the (k, c) phase
plane: nullclines, vector fields, saddle paths, and steady states.

Reference: Barro and Sala-i-Martin (2004), "Economic Growth," Ch. 2.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters (Ramsey-Cass-Koopmans model)
    # =========================================================================
    alpha = 0.3      # Capital share
    delta = 0.05     # Depreciation rate
    rho = 0.04       # Discount rate (time preference)
    sigma = 2.0      # Inverse elasticity of intertemporal substitution
    A = 1.0          # TFP

    # Production function and its derivative
    f = lambda k: A * k ** alpha
    f_prime = lambda k: alpha * A * k ** (alpha - 1)

    # Steady state
    k_ss = ((alpha * A) / (rho + delta)) ** (1 / (1 - alpha))
    c_ss = f(k_ss) - delta * k_ss

    print(f"Steady state: k* = {k_ss:.4f}, c* = {c_ss:.4f}")

    # =========================================================================
    # Phase diagram system: dk/dt = f(k) - delta*k - c
    #                       dc/dt = (1/sigma)*(f'(k) - delta - rho)*c
    # =========================================================================
    def dynamics(t, y):
        k, c = y
        k = max(k, 1e-10)
        c = max(c, 1e-10)
        dk = f(k) - delta * k - c
        dc = (1 / sigma) * (f_prime(k) - delta - rho) * c
        return [dk, dc]

    # =========================================================================
    # Nullclines
    # =========================================================================
    k_range = np.linspace(0.1, k_ss * 2.5, 300)
    # dk/dt = 0 nullcline: c = f(k) - delta*k
    c_nullcline = f(k_range) - delta * k_range
    # dc/dt = 0 nullcline: vertical line at k = k_ss (where f'(k) = delta + rho)

    # =========================================================================
    # Vector field
    # =========================================================================
    k_grid = np.linspace(0.5, k_ss * 2.2, 20)
    c_grid = np.linspace(0.1, c_ss * 2.0, 20)
    K, C = np.meshgrid(k_grid, c_grid)
    DK = f(K) - delta * K - C
    DC = (1 / sigma) * (f_prime(K) - delta - rho) * C

    # Normalize arrows for visual clarity
    magnitude = np.sqrt(DK ** 2 + DC ** 2)
    DK_norm = DK / (magnitude + 1e-10)
    DC_norm = DC / (magnitude + 1e-10)

    # =========================================================================
    # Saddle path (shooting method)
    # =========================================================================
    saddle_paths = []
    # Shoot forward and backward from near the steady state
    for direction in [-1, 1]:
        for eps_k in np.linspace(-0.3, 0.3, 5):
            eps_c = -eps_k * 0.5  # Approximate saddle path slope
            k0 = k_ss + eps_k
            c0 = c_ss + eps_c
            if k0 <= 0 or c0 <= 0:
                continue
            sol = solve_ivp(
                lambda t, y: [direction * d for d in dynamics(t, y)],
                [0, 100], [k0, c0],
                max_step=0.1, events=lambda t, y: min(y[0] - 0.05, y[1] - 0.05),
                dense_output=True,
            )
            if sol.success and len(sol.t) > 5:
                k_path = sol.y[0]
                c_path = sol.y[1]
                valid = (k_path > 0) & (c_path > 0) & (k_path < k_ss * 3) & (c_path < c_ss * 3)
                if direction == -1:
                    saddle_paths.append((k_path[valid][::-1], c_path[valid][::-1]))
                else:
                    saddle_paths.append((k_path[valid], c_path[valid]))

    # Proper saddle path via linearization
    # Jacobian at steady state
    J11 = f_prime(k_ss) - delta  # = rho (at ss)
    J12 = -1
    J21 = (1 / sigma) * f_prime(k_ss) * alpha * (alpha - 1) * A * k_ss ** (alpha - 2) * c_ss / (alpha * A * k_ss ** (alpha - 1))
    # Simplified: J21 = c_ss * (alpha-1) * f'(k_ss) / (sigma * k_ss)
    J21_simple = c_ss * (alpha - 1) * f_prime(k_ss) / (sigma * k_ss)
    J22 = 0  # dc/dt has no c term beyond c*(...) and at ss, (...) = 0

    # Eigenvalues
    trace = J11 + J22
    det = J11 * J22 - J12 * J21_simple
    lambda1 = (trace - np.sqrt(trace ** 2 - 4 * det)) / 2
    lambda2 = (trace + np.sqrt(trace ** 2 - 4 * det)) / 2
    print(f"Eigenvalues: {lambda1:.4f}, {lambda2:.4f} (saddle point: one negative, one positive)")

    # Stable eigenvector direction
    stable_eigvec = np.array([1, (lambda1 - J11) / J12])
    slope = stable_eigvec[1] / stable_eigvec[0]

    # Trace saddle path using stable manifold
    n_saddle = 50
    k_saddle_left = np.linspace(0.5, k_ss, n_saddle)
    c_saddle_left = c_ss + slope * (k_saddle_left - k_ss)

    k_saddle_right = np.linspace(k_ss, k_ss * 2, n_saddle)
    c_saddle_right = c_ss + slope * (k_saddle_right - k_ss)

    # =========================================================================
    # Time path simulation (starting below steady state)
    # =========================================================================
    k0 = k_ss * 0.3
    c0 = c_ss + slope * (k0 - k_ss)
    sol_time = solve_ivp(dynamics, [0, 150], [k0, max(c0, 0.01)], max_step=0.2, dense_output=True)
    t_eval = np.linspace(0, 100, 500)
    y_eval = sol_time.sol(t_eval)

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Phase Diagrams: Ramsey Optimal Growth",
        "Continuous-time dynamics of consumption and capital with saddle-path stability.",
    )

    report.add_overview(
        "Phase diagrams are the primary tool for analyzing continuous-time dynamic economic "
        "models. The Ramsey-Cass-Koopmans model has a two-dimensional state space $(k, c)$ "
        "with a unique steady state that is a *saddle point* — only one path (the stable "
        "manifold) converges to it.\n\n"
        "This module visualizes the phase plane: nullclines where $\\dot{k} = 0$ and "
        "$\\dot{c} = 0$, the vector field showing direction of motion, and the saddle path "
        "that the economy must follow for an interior optimum."
    )

    report.add_equations(r"""
**Capital accumulation:**
$$\dot{k} = f(k) - \delta k - c$$

**Euler equation (consumption):**
$$\dot{c} = \frac{1}{\sigma} \left( f'(k) - \delta - \rho \right) c$$

**Nullclines:**
- $\dot{k} = 0$: $c = f(k) - \delta k$ (hump-shaped in $k$)
- $\dot{c} = 0$: $f'(k) = \delta + \rho$, i.e., $k = k^*$ (vertical line)

**Steady state:** $k^* = \left(\frac{\alpha A}{\rho + \delta}\right)^{1/(1-\alpha)}$, $c^* = f(k^*) - \delta k^*$

**Transversality condition** selects the saddle path as the unique optimal trajectory.
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| $\\alpha$ | {alpha} | Capital share |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\rho$ | {rho} | Discount rate |\n"
        f"| $\\sigma$ | {sigma} | CRRA coefficient |\n"
        f"| $k^*$ | {k_ss:.4f} | Steady-state capital |\n"
        f"| $c^*$ | {c_ss:.4f} | Steady-state consumption |"
    )

    report.add_solution_method(
        "**Linearization:** The Jacobian at the steady state has eigenvalues "
        f"$\\lambda_1 = {lambda1:.4f}$ (stable) and $\\lambda_2 = {lambda2:.4f}$ (unstable). "
        "This confirms the steady state is a **saddle point**.\n\n"
        "**Saddle path:** The stable manifold is traced using the eigenvector associated "
        f"with $\\lambda_1$. Near the steady state, the saddle path slope is approximately {slope:.4f}.\n\n"
        "**Integration:** Time paths computed via `scipy.integrate.solve_ivp` (RK45)."
    )

    # --- Figure 1: Full phase diagram ---
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.quiver(K, C, DK_norm, DC_norm, magnitude, cmap="coolwarm", alpha=0.4, scale=30)
    ax1.plot(k_range, c_nullcline, "b-", linewidth=2.5, label="$\\dot{k}=0$ nullcline")
    ax1.axvline(k_ss, color="r", linewidth=2.5, label="$\\dot{c}=0$ nullcline")
    ax1.plot(k_saddle_left, c_saddle_left, "k-", linewidth=3, label="Saddle path")
    ax1.plot(k_saddle_right, c_saddle_right, "k-", linewidth=3)
    ax1.plot(k_ss, c_ss, "ko", markersize=12, zorder=5)
    ax1.annotate(f"$k^*={k_ss:.2f}, c^*={c_ss:.2f}$", (k_ss, c_ss),
                 textcoords="offset points", xytext=(15, -20), fontsize=10)
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("Consumption $c$")
    ax1.set_title("Phase Diagram: Ramsey Optimal Growth Model")
    ax1.set_xlim(0, k_ss * 2.5)
    ax1.set_ylim(0, c_ss * 2.2)
    ax1.legend(loc="upper right")
    report.add_figure("figures/phase-diagram.png",
                      "Phase diagram with nullclines, vector field, and saddle path", fig1)

    # --- Figure 2: Time paths ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    valid = (y_eval[0] > 0) & (y_eval[1] > 0)
    t_valid = t_eval[valid]
    ax2a.plot(t_valid, y_eval[0][valid], "b-", linewidth=2)
    ax2a.axhline(k_ss, color="k", linestyle="--", alpha=0.5, label=f"$k^*={k_ss:.2f}$")
    ax2a.set_xlabel("Time")
    ax2a.set_ylabel("Capital $k(t)$")
    ax2a.set_title("Capital Transition to Steady State")
    ax2a.legend()

    ax2b.plot(t_valid, y_eval[1][valid], "r-", linewidth=2)
    ax2b.axhline(c_ss, color="k", linestyle="--", alpha=0.5, label=f"$c^*={c_ss:.2f}$")
    ax2b.set_xlabel("Time")
    ax2b.set_ylabel("Consumption $c(t)$")
    ax2b.set_title("Consumption Transition to Steady State")
    ax2b.legend()
    fig2.tight_layout()
    report.add_figure("figures/time-paths.png",
                      "Capital and consumption converge to steady state along the saddle path", fig2)

    # --- Figure 3: Four quadrant dynamics ---
    fig3, axes = plt.subplots(2, 2, figsize=(10, 8))
    regions = [
        ("$k < k^*, c > c_{null}$\n(↙ diverge)", k_ss * 0.5, c_ss * 1.5),
        ("$k > k^*, c > c_{null}$\n(↖ diverge)", k_ss * 1.5, c_ss * 1.5),
        ("$k < k^*, c < c_{null}$\n(↗ converge)", k_ss * 0.5, c_ss * 0.5),
        ("$k > k^*, c < c_{null}$\n(↘ diverge)", k_ss * 1.5, c_ss * 0.3),
    ]
    for ax, (label, k0_r, c0_r) in zip(axes.flat, regions):
        sol_r = solve_ivp(dynamics, [0, 30], [k0_r, c0_r], max_step=0.1, dense_output=True)
        t_r = np.linspace(0, min(30, sol_r.t[-1]), 200)
        y_r = sol_r.sol(t_r)
        ax.plot(y_r[0], y_r[1], "b-", linewidth=2)
        ax.plot(y_r[0][0], y_r[1][0], "go", markersize=8)
        ax.plot(k_ss, c_ss, "ko", markersize=8)
        ax.plot(k_range, c_nullcline, "b--", alpha=0.3)
        ax.axvline(k_ss, color="r", alpha=0.3, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, k_ss * 3)
        ax.set_ylim(0, c_ss * 2.5)
    fig3.suptitle("Trajectories from Different Starting Regions", fontsize=13)
    fig3.tight_layout()
    report.add_figure("figures/four-regions.png",
                      "Only trajectories starting on the saddle path converge to steady state", fig3)

    # --- Table ---
    df = pd.DataFrame({
        "Quantity": ["$k^*$", "$c^*$", "$y^*$", "$r^*$", "$\\lambda_1$", "$\\lambda_2$"],
        "Value": [f"{k_ss:.4f}", f"{c_ss:.4f}", f"{f(k_ss):.4f}",
                  f"{f_prime(k_ss)-delta:.4f}", f"{lambda1:.4f}", f"{lambda2:.4f}"],
        "Description": ["Steady-state capital", "Steady-state consumption",
                        "Steady-state output", "Net interest rate (= rho at ss)",
                        "Stable eigenvalue", "Unstable eigenvalue"],
    })
    report.add_table("tables/steady-state.csv", "Steady-State Values and Eigenvalues", df)

    report.add_takeaway(
        "Phase diagrams reveal the qualitative dynamics of the Ramsey model:\n\n"
        "**Key insights:**\n"
        "- The steady state is a **saddle point**: most trajectories diverge. Only the "
        "saddle path (stable manifold) converges — the transversality condition selects it.\n"
        "- Above the saddle path, agents *over-consume*, depleting capital. Below it, "
        "they *over-save*, accumulating capital without bound.\n"
        "- The $\\dot{k}=0$ nullcline is the golden rule line — maximum sustainable consumption. "
        "The Ramsey steady state lies *below* the golden rule because agents are impatient ($\\rho > 0$).\n"
        "- The **speed of convergence** depends on $|\\lambda_1|$: a half-life of "
        f"$\\ln(2)/|\\lambda_1| \\approx {np.log(2)/abs(lambda1):.1f}$ periods for capital to "
        "close half the gap to steady state."
    )

    report.add_references([
        "Ramsey, F. (1928). \"A Mathematical Theory of Saving.\" *Economic Journal*, 38(152).",
        "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.",
        "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 8.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
