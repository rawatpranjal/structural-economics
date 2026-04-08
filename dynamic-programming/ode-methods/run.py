#!/usr/bin/env python3
"""Numerical ODE Methods for Economics.

Demonstrates how continuous-time economic models reduce to systems of ODEs,
solved numerically with scipy.integrate.solve_ivp. Covers the Solow growth
model (globally stable), the Ramsey optimal growth model (saddle path via
shooting), and the Lotka-Volterra predator-prey system (limit cycles).

Reference: Acemoglu (2009), Introduction to Modern Economic Growth, Ch. 2, 7-8.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Common Parameters
    # =========================================================================
    alpha = 0.3      # Capital share in production
    delta = 0.05     # Depreciation rate
    rho = 0.04       # Discount rate (Ramsey)
    sigma = 2.0      # CRRA coefficient (Ramsey)
    s = 0.3          # Savings rate (Solow)
    n = 0.02         # Population growth rate (Solow)

    # Production function and its derivative
    def f(k):
        return k ** alpha

    def f_prime(k):
        return alpha * k ** (alpha - 1.0)

    # =========================================================================
    # Example 1: Solow Growth Model
    # =========================================================================
    print("=" * 60)
    print("Example 1: Solow Growth Model")
    print("=" * 60)

    # Steady state: s * f(k*) = (n + delta) * k*  =>  k* = (s / (n + delta))^(1/(1-alpha))
    k_star_solow = (s / (n + delta)) ** (1.0 / (1.0 - alpha))
    y_star_solow = f(k_star_solow)
    c_star_solow = (1.0 - s) * y_star_solow

    print(f"  Steady-state capital:     k* = {k_star_solow:.4f}")
    print(f"  Steady-state output:      y* = {y_star_solow:.4f}")
    print(f"  Steady-state consumption: c* = {c_star_solow:.4f}")

    # Solow ODE: dk/dt = s*f(k) - (n+delta)*k
    def solow_ode(t, k):
        return s * f(k) - (n + delta) * k

    # Solve from multiple initial conditions
    T_solow = 200
    t_span_solow = (0, T_solow)
    t_eval_solow = np.linspace(0, T_solow, 1000)
    k0_values = [0.5, 2.0, 5.0, 10.0, 15.0, 20.0]

    solow_solutions = []
    for k0 in k0_values:
        sol = solve_ivp(solow_ode, t_span_solow, [k0], t_eval=t_eval_solow,
                        method='RK45', rtol=1e-10, atol=1e-12)
        solow_solutions.append(sol)

    # Phase diagram data: dk/dt as a function of k
    k_phase = np.linspace(0.01, 25.0, 500)
    dkdt_phase = s * f(k_phase) - (n + delta) * k_phase

    # =========================================================================
    # Example 2: Ramsey Optimal Growth (Saddle Path)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Ramsey Optimal Growth Model")
    print("=" * 60)

    # Steady state: f'(k*) = delta + rho  =>  k* = (alpha / (delta + rho))^(1/(1-alpha))
    k_star_ramsey = (alpha / (delta + rho)) ** (1.0 / (1.0 - alpha))
    c_star_ramsey = f(k_star_ramsey) - delta * k_star_ramsey
    y_star_ramsey = f(k_star_ramsey)

    print(f"  Steady-state capital:     k* = {k_star_ramsey:.4f}")
    print(f"  Steady-state output:      y* = {y_star_ramsey:.4f}")
    print(f"  Steady-state consumption: c* = {c_star_ramsey:.4f}")

    # Ramsey system of ODEs
    def ramsey_ode(t, state):
        k, c = state
        k = max(k, 1e-10)
        c = max(c, 1e-10)
        dk = f(k) - delta * k - c
        dc = (1.0 / sigma) * (f_prime(k) - delta - rho) * c
        return [dk, dc]

    # --- Shooting method to find saddle path ---
    # From initial k0 < k*, shoot by trying different c0 values.
    # The correct c0 puts us on the saddle path converging to (k*, c*).
    T_ramsey = 300

    def shoot(k0, c0, T=T_ramsey):
        """Integrate the Ramsey system forward and return terminal state."""
        def ramsey_with_stop(t, state):
            k, c = state
            return [max(min(dk, 1e6), -1e6) for dk in ramsey_ode(t, state)]

        sol = solve_ivp(ramsey_with_stop, (0, T), [k0, c0],
                        method='RK45', rtol=1e-10, atol=1e-12,
                        max_step=0.5, dense_output=True)
        return sol

    def find_saddle_path_from(k0, T=T_ramsey):
        """Use bisection to find the initial c that converges to steady state."""
        # Bounds: c must be between 0 and f(k0) - delta*k0 (approximately)
        c_low = 1e-6
        c_high = f(k0)  # upper bound on feasible consumption

        for _ in range(200):
            c_mid = (c_low + c_high) / 2.0
            sol = shoot(k0, c_mid, T)
            k_path = sol.y[0]
            c_path = sol.y[1]

            # Check if path overshoots or undershoots
            # If c is too high: consumption is too large, k eventually declines
            #   and c > c*, or c runs away upward
            # If c is too low: k grows past k*, c eventually drops to 0

            # Find if path crosses k* from below/above or diverges
            terminal_k = k_path[-1]
            terminal_c = c_path[-1]

            # Check for path blowing up or going negative
            if np.any(c_path < 0) or np.any(k_path < 0):
                c_high = c_mid
                continue

            # If terminal k > k* substantially, c0 was too low (saves too much)
            # If terminal k < k* substantially, c0 was too high (consumes too much)
            if terminal_k > k_star_ramsey * 1.05:
                c_low = c_mid
            elif terminal_k < k_star_ramsey * 0.95:
                c_high = c_mid
            else:
                # Close enough -- check c convergence too
                if terminal_c > c_star_ramsey * 1.05:
                    c_high = c_mid
                elif terminal_c < c_star_ramsey * 0.95:
                    c_low = c_mid
                else:
                    break

        return c_mid, shoot(k0, c_mid, T)

    # Solve saddle paths from multiple initial capital levels
    k0_ramsey_values = [1.0, 3.0, 8.0, 12.0]
    saddle_paths = []
    for k0 in k0_ramsey_values:
        c0_opt, sol = find_saddle_path_from(k0)
        saddle_paths.append((k0, c0_opt, sol))
        print(f"  k0={k0:.1f}: saddle path c0 = {c0_opt:.6f}")

    # Nullclines for phase diagram
    k_null = np.linspace(0.01, 15.0, 500)
    # dk/dt = 0 nullcline: c = f(k) - delta*k
    c_kdot_null = f(k_null) - delta * k_null
    # dc/dt = 0 nullcline: f'(k) = delta + rho  =>  vertical line at k*
    # (Also c = 0 is a nullcline for dc/dt = 0, but trivial)

    # Vector field for phase diagram
    k_vec = np.linspace(0.5, 14.0, 20)
    c_vec = np.linspace(0.1, 2.5, 20)
    K_grid, C_grid = np.meshgrid(k_vec, c_vec)
    DK = f(K_grid) - delta * K_grid - C_grid
    DC = (1.0 / sigma) * (f_prime(K_grid) - delta - rho) * C_grid
    # Normalize arrows for visual clarity
    magnitude = np.sqrt(DK**2 + DC**2)
    magnitude = np.maximum(magnitude, 1e-10)
    DK_norm = DK / magnitude
    DC_norm = DC / magnitude

    # =========================================================================
    # Example 3: Lotka-Volterra Predator-Prey
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: Lotka-Volterra Predator-Prey")
    print("=" * 60)

    # Parameters
    lv_alpha = 1.1    # Prey growth rate
    lv_beta = 0.4     # Predation rate
    lv_delta = 0.1    # Predator reproduction rate
    lv_gamma = 0.4    # Predator death rate

    # Steady states: (0,0) and (gamma/delta, alpha/beta)
    x_star_lv = lv_gamma / lv_delta
    y_star_lv = lv_alpha / lv_beta

    print(f"  Interior steady state: x* = {x_star_lv:.4f}, y* = {y_star_lv:.4f}")

    def lotka_volterra_ode(t, state):
        x, y = state
        dx = lv_alpha * x - lv_beta * x * y
        dy = lv_delta * x * y - lv_gamma * y
        return [dx, dy]

    T_lv = 50
    t_eval_lv = np.linspace(0, T_lv, 2000)
    lv_init = [5.0, 3.0]  # Initial prey and predator populations

    sol_lv = solve_ivp(lotka_volterra_ode, (0, T_lv), lv_init,
                       t_eval=t_eval_lv, method='RK45', rtol=1e-10, atol=1e-12)

    print(f"  Prey range:     [{sol_lv.y[0].min():.2f}, {sol_lv.y[0].max():.2f}]")
    print(f"  Predator range: [{sol_lv.y[1].min():.2f}, {sol_lv.y[1].max():.2f}]")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Numerical ODE Methods for Economics",
        "Solving continuous-time economic models as systems of ordinary differential equations.",
    )

    report.add_overview(
        "Many fundamental economic models are naturally formulated in continuous time, "
        "giving rise to systems of ordinary differential equations (ODEs). This module "
        "demonstrates three canonical examples:\n\n"
        "1. **Solow Growth Model** -- A single ODE with a globally stable steady state. "
        "The phase diagram immediately reveals the convergence dynamics.\n"
        "2. **Ramsey Optimal Growth Model** -- A two-dimensional ODE system with a saddle "
        "point equilibrium. Only the *saddle path* converges to the steady state; all other "
        "trajectories diverge. The shooting method finds this path numerically.\n"
        "3. **Lotka-Volterra Predator-Prey** -- A classic nonlinear ODE system exhibiting "
        "perpetual cycles, illustrating how phase diagrams reveal qualitative dynamics."
    )

    report.add_equations(
        r"""
**Solow Growth Model:**
$$\frac{dk}{dt} = s \cdot f(k) - (n + \delta) \cdot k, \qquad f(k) = k^\alpha$$

**Ramsey Optimal Growth (Euler Equation + Capital Accumulation):**
$$\frac{dc}{dt} = \frac{1}{\sigma}\left(f'(k) - \delta - \rho\right) c$$
$$\frac{dk}{dt} = f(k) - \delta k - c$$

**Lotka-Volterra Predator-Prey:**
$$\frac{dx}{dt} = \alpha x - \beta x y, \qquad \frac{dy}{dt} = \delta x y - \gamma y$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\alpha$ | {alpha} | Capital share (Cobb-Douglas) |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $s$ | {s} | Savings rate (Solow) |\n"
        f"| $n$ | {n} | Population growth rate (Solow) |\n"
        f"| $\\rho$ | {rho} | Discount rate (Ramsey) |\n"
        f"| $\\sigma$ | {sigma} | CRRA coefficient (Ramsey) |\n"
        f"| Solver | `solve_ivp` (RK45) | Adaptive Runge-Kutta 4(5) |"
    )

    report.add_solution_method(
        "All ODE systems are solved using `scipy.integrate.solve_ivp` with the RK45 "
        "(Dormand-Prince) adaptive step-size method, using tight tolerances "
        "(`rtol=1e-10`, `atol=1e-12`).\n\n"
        "**Solow model:** Direct forward integration from various initial conditions. The "
        "single ODE has a unique, globally stable steady state.\n\n"
        "**Ramsey model:** The steady state is a *saddle point* -- most initial conditions "
        "diverge. We use a **shooting method** (bisection on initial consumption $c_0$) to "
        "find the unique saddle path that converges to $(k^*, c^*)$. For each candidate "
        "$c_0$, we integrate forward and check whether the trajectory approaches or diverges "
        "from the steady state.\n\n"
        "**Lotka-Volterra:** Direct forward integration reveals perpetual cycles (the system "
        "has a conserved quantity, so orbits are closed curves in the phase plane)."
    )

    # --- Figure 1: Solow Phase Diagram ---
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: phase diagram dk/dt vs k
    ax1a.plot(k_phase, dkdt_phase, 'b-', linewidth=2, label=r'$\dot{k} = sf(k) - (n+\delta)k$')
    ax1a.axhline(y=0, color='k', linewidth=0.8, linestyle='-')
    ax1a.axvline(x=k_star_solow, color='r', linewidth=1.5, linestyle='--', alpha=0.7,
                 label=f'$k^* = {k_star_solow:.2f}$')
    ax1a.plot(k_star_solow, 0, 'ro', markersize=10, zorder=5)
    ax1a.set_xlabel('Capital per worker $k$')
    ax1a.set_ylabel(r'$\dot{k} = dk/dt$')
    ax1a.set_title('Solow Model: Phase Diagram')
    ax1a.legend(loc='upper right')
    ax1a.set_xlim(0, 25)

    # Right: time paths from different initial conditions
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(k0_values)))
    for i, (k0, sol) in enumerate(zip(k0_values, solow_solutions)):
        ax1b.plot(sol.t, sol.y[0], color=colors[i], linewidth=1.5,
                  label=f'$k_0 = {k0:.1f}$')
    ax1b.axhline(y=k_star_solow, color='r', linewidth=1.5, linestyle='--', alpha=0.7,
                 label=f'$k^* = {k_star_solow:.2f}$')
    ax1b.set_xlabel('Time $t$')
    ax1b.set_ylabel('Capital per worker $k(t)$')
    ax1b.set_title('Solow Model: Convergence to Steady State')
    ax1b.legend(loc='right', fontsize=8)

    fig1.tight_layout()
    report.add_figure("figures/solow-phase-diagram.png",
                      "Solow growth model: phase diagram (left) and convergence dynamics (right)",
                      fig1,
        description="The phase diagram shows dk/dt as a single hump-shaped curve crossing zero at k*: "
        "to the left capital accumulates, to the right it decumulates. This immediately proves global stability -- "
        "no matter the starting point, the economy is pulled toward steady state.")

    # --- Figure 2: Ramsey Phase Diagram with Saddle Path ---
    fig2, ax2 = plt.subplots(figsize=(9, 7))

    # Vector field
    ax2.quiver(K_grid, C_grid, DK_norm, DC_norm, magnitude,
               cmap='coolwarm', alpha=0.5, scale=30, width=0.003)

    # Nullclines
    ax2.plot(k_null, c_kdot_null, 'b-', linewidth=2.5,
             label=r'$\dot{k}=0$: $c = f(k) - \delta k$')
    ax2.axvline(x=k_star_ramsey, color='g', linewidth=2.5, linestyle='--',
                label=fr"$\dot{{c}}=0$: $k = k^* = {k_star_ramsey:.2f}$")

    # Saddle paths
    saddle_colors = ['#d62728', '#ff7f0e', '#9467bd', '#e377c2']
    for i, (k0, c0_opt, sol) in enumerate(saddle_paths):
        ax2.plot(sol.y[0], sol.y[1], color=saddle_colors[i], linewidth=2,
                 label=f'Saddle path from $k_0={k0:.0f}$')
        ax2.plot(k0, c0_opt, 'o', color=saddle_colors[i], markersize=6, zorder=5)

    # Steady state
    ax2.plot(k_star_ramsey, c_star_ramsey, 'k*', markersize=15, zorder=10,
             label=f'Steady state ({k_star_ramsey:.2f}, {c_star_ramsey:.2f})')

    ax2.set_xlabel('Capital $k$')
    ax2.set_ylabel('Consumption $c$')
    ax2.set_title('Ramsey Model: Phase Diagram with Saddle Paths')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 2.5)

    report.add_figure("figures/ramsey-phase-diagram.png",
                      "Ramsey optimal growth: phase diagram with nullclines, vector field, and saddle paths from multiple initial conditions",
                      fig2,
        description="The two nullclines divide the phase plane into four regions with distinct dynamics. "
        "Only the saddle paths converge to the steady state; all other trajectories violate the transversality condition. "
        "The vector field arrows reveal the unstable directions that make the shooting method necessary.")

    # --- Figure 3: Lotka-Volterra Cycles ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: time series
    ax3a.plot(sol_lv.t, sol_lv.y[0], 'b-', linewidth=2, label='Prey $x(t)$')
    ax3a.plot(sol_lv.t, sol_lv.y[1], 'r-', linewidth=2, label='Predator $y(t)$')
    ax3a.axhline(y=x_star_lv, color='b', linewidth=1, linestyle=':', alpha=0.5)
    ax3a.axhline(y=y_star_lv, color='r', linewidth=1, linestyle=':', alpha=0.5)
    ax3a.set_xlabel('Time $t$')
    ax3a.set_ylabel('Population')
    ax3a.set_title('Lotka-Volterra: Population Dynamics')
    ax3a.legend()

    # Right: phase portrait
    ax3b.plot(sol_lv.y[0], sol_lv.y[1], 'purple', linewidth=1.5, label='Orbit')
    ax3b.plot(lv_init[0], lv_init[1], 'go', markersize=8, zorder=5, label='Initial condition')
    ax3b.plot(x_star_lv, y_star_lv, 'k*', markersize=12, zorder=5,
              label=f'Steady state ({x_star_lv:.1f}, {y_star_lv:.1f})')
    # Direction arrows along the orbit
    n_arrows = 8
    arrow_indices = np.linspace(0, len(sol_lv.t) - 2, n_arrows, dtype=int)
    for idx in arrow_indices:
        dx = sol_lv.y[0][idx + 1] - sol_lv.y[0][idx]
        dy = sol_lv.y[1][idx + 1] - sol_lv.y[1][idx]
        ax3b.annotate('', xy=(sol_lv.y[0][idx + 1], sol_lv.y[1][idx + 1]),
                      xytext=(sol_lv.y[0][idx], sol_lv.y[1][idx]),
                      arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax3b.set_xlabel('Prey $x$')
    ax3b.set_ylabel('Predator $y$')
    ax3b.set_title('Lotka-Volterra: Phase Portrait')
    ax3b.legend(loc='upper right')

    fig3.tight_layout()
    report.add_figure("figures/lotka-volterra-cycles.png",
                      "Lotka-Volterra predator-prey: time series (left) and phase portrait showing closed orbits (right)",
                      fig3,
        description="The closed orbit in the phase portrait confirms that the system has a conserved quantity -- populations cycle "
        "indefinitely without converging or diverging. This contrasts sharply with the Solow and Ramsey models, illustrating how "
        "qualitatively different ODE structures (stable node, saddle, center) produce fundamentally different economic dynamics.")

    # --- Figure 4: Ramsey Time Paths ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

    for i, (k0, c0_opt, sol) in enumerate(saddle_paths):
        ax4a.plot(sol.t, sol.y[0], color=saddle_colors[i], linewidth=2,
                  label=f'$k_0={k0:.0f}$')
    ax4a.axhline(y=k_star_ramsey, color='k', linewidth=1.5, linestyle='--', alpha=0.7,
                 label=f'$k^* = {k_star_ramsey:.2f}$')
    ax4a.set_xlabel('Time $t$')
    ax4a.set_ylabel('Capital $k(t)$')
    ax4a.set_title('Ramsey Model: Capital Time Paths')
    ax4a.legend(fontsize=8)
    ax4a.set_xlim(0, 150)

    for i, (k0, c0_opt, sol) in enumerate(saddle_paths):
        ax4b.plot(sol.t, sol.y[1], color=saddle_colors[i], linewidth=2,
                  label=f'$k_0={k0:.0f}$')
    ax4b.axhline(y=c_star_ramsey, color='k', linewidth=1.5, linestyle='--', alpha=0.7,
                 label=f'$c^* = {c_star_ramsey:.2f}$')
    ax4b.set_xlabel('Time $t$')
    ax4b.set_ylabel('Consumption $c(t)$')
    ax4b.set_title('Ramsey Model: Consumption Time Paths')
    ax4b.legend(fontsize=8)
    ax4b.set_xlim(0, 150)

    fig4.tight_layout()
    report.add_figure("figures/ramsey-time-paths.png",
                      "Ramsey model: time paths of capital k(t) and consumption c(t) along saddle paths",
                      fig4,
        description="All saddle paths converge to the same steady state regardless of initial capital. "
        "Capital-poor economies grow faster (steeper slopes) due to higher marginal returns, while consumption adjusts smoothly along the Euler equation.")

    # --- Table: Steady-State Values ---
    table_data = {
        "Model": ["Solow Growth", "Solow Growth", "Solow Growth",
                   "Ramsey Optimal Growth", "Ramsey Optimal Growth", "Ramsey Optimal Growth",
                   "Lotka-Volterra", "Lotka-Volterra"],
        "Variable": ["k*", "y*", "c*",
                      "k*", "y*", "c*",
                      "x* (prey)", "y* (predator)"],
        "Value": [f"{k_star_solow:.4f}", f"{y_star_solow:.4f}", f"{c_star_solow:.4f}",
                  f"{k_star_ramsey:.4f}", f"{y_star_ramsey:.4f}", f"{c_star_ramsey:.4f}",
                  f"{x_star_lv:.4f}", f"{y_star_lv:.4f}"],
        "Formula": [
            r"(s/(n+delta))^(1/(1-alpha))",
            r"f(k*) = k*^alpha",
            r"(1-s)*y*",
            r"(alpha/(delta+rho))^(1/(1-alpha))",
            r"f(k*) = k*^alpha",
            r"f(k*) - delta*k*",
            r"gamma/delta",
            r"alpha/beta",
        ],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/steady-states.csv",
                     "Steady-State Values for Each Model", df,
        description="Comparing steady states across models highlights a key distinction: the Solow and Ramsey models share the same production "
        "technology but reach different steady states because optimal savings (Ramsey) differs from exogenous savings (Solow).")

    report.add_takeaway(
        "These three examples illustrate the rich dynamics that arise from continuous-time "
        "economic models:\n\n"
        "**Key insights:**\n"
        "- **Solow model:** The steady state is *globally stable* -- regardless of initial "
        "capital, the economy converges to $k^*$. The phase diagram (a single curve crossing "
        "zero) makes this immediately transparent.\n"
        "- **Ramsey model:** The steady state is a *saddle point*. Only one trajectory (the "
        "saddle path) converges; all others diverge. This reflects the forward-looking nature "
        "of optimal consumption: the agent must choose exactly the right initial consumption "
        "to satisfy the transversality condition.\n"
        "- **Shooting method:** Finding saddle paths numerically requires a shooting approach "
        "-- bisecting over initial conditions until convergence is achieved. This is a "
        "fundamental technique in computational economics.\n"
        "- **Lotka-Volterra:** The closed orbits (limit cycles) demonstrate that nonlinear ODE "
        "systems can exhibit qualitatively different behavior (cycles vs. convergence). Phase "
        "diagrams reveal this structure at a glance.\n"
        "- **Phase diagrams as tools:** Nullclines partition the phase space into regions "
        "with qualitatively different dynamics. Combined with vector fields, they provide "
        "complete qualitative understanding before any numerical solution is computed."
    )

    report.add_references([
        "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 2, 7-8.",
        "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 1-2.",
        "Strogatz, S. (2015). *Nonlinear Dynamics and Chaos*. Westview Press, 2nd edition.",
        "Judd, K. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 10.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
