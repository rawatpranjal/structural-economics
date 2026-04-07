#!/usr/bin/env python3
"""Ramsey-Cass-Koopmans Growth Model (Continuous Time).

Solves the Ramsey optimal growth model using the shooting method on the
saddle-path stable manifold. The household maximizes discounted CRRA utility
subject to the neoclassical capital accumulation equation.

Reference: Barro and Sala-i-Martin (2004), Economic Growth, Ch. 2.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    alpha = 0.33      # Capital share in production
    delta = 0.05      # Depreciation rate
    rho = 0.03        # Discount rate (rate of time preference)
    sigma = 2.0       # CRRA coefficient (inverse EIS)
    A = 1.0           # TFP
    T = 150.0         # Integration horizon (long enough to approach steady state)
    n_eval = 2000     # Evaluation points

    # =========================================================================
    # Steady State
    # =========================================================================
    # At steady state: f'(k*) = delta + rho  =>  alpha*A*k*^{alpha-1} = delta + rho
    # => k* = (alpha*A / (delta + rho))^{1/(1-alpha)}
    # c* = f(k*) - delta*k* = A*k*^alpha - delta*k*

    k_star = (alpha * A / (delta + rho)) ** (1.0 / (1.0 - alpha))
    c_star = A * k_star**alpha - delta * k_star

    print(f"Steady state: k* = {k_star:.4f}, c* = {c_star:.4f}")

    # =========================================================================
    # ODE System
    # =========================================================================
    def f(k):
        """Production function."""
        return A * k**alpha

    def f_prime(k):
        """Marginal product of capital."""
        return alpha * A * k**(alpha - 1)

    def ode_system(t, y):
        """Ramsey system: [dk/dt, dc/dt]."""
        k, c = y
        k = max(k, 1e-10)
        c = max(c, 1e-10)
        dk_dt = f(k) - delta * k - c
        dc_dt = (1.0 / sigma) * (f_prime(k) - delta - rho) * c
        return [dk_dt, dc_dt]

    # =========================================================================
    # Shooting Method
    # =========================================================================
    # For a given k(0), we need to find c(0) such that the system converges
    # to (k*, c*). If c(0) is too high, capital runs out; if too low, capital
    # grows without bound. We shoot for the saddle path.

    def shoot(k0, c0_guess, T_shoot=T):
        """Integrate the ODE and return terminal k deviation from k*."""
        sol = solve_ivp(
            ode_system,
            [0, T_shoot],
            [k0, c0_guess],
            method="RK45",
            max_step=0.5,
            rtol=1e-10,
            atol=1e-12,
        )
        # Check if consumption went to zero or capital went to zero
        k_terminal = sol.y[0, -1]
        c_terminal = sol.y[1, -1]
        # We want (k_terminal, c_terminal) close to (k*, c*)
        return k_terminal - k_star

    def find_saddle_path_c0(k0, c_low=1e-6, c_high=None):
        """Find initial consumption on the saddle path via bisection."""
        if c_high is None:
            c_high = f(k0) - delta * k0 + 0.01  # slightly above steady-state consumption at k0
            c_high = max(c_high, c_low + 0.01)

        # Bisection: if c0 too high, k falls below k*; if c0 too low, k rises above k*
        # For k0 < k*, the saddle path has c0 < c* and capital accumulates to k*
        # For k0 > k*, the saddle path has c0 > c* and capital decumulates to k*

        try:
            c0_opt = brentq(
                lambda c0: shoot(k0, c0),
                c_low,
                c_high,
                xtol=1e-10,
                maxiter=200,
            )
        except ValueError:
            # If brentq fails, try a wider range with manual bisection
            n_try = 500
            c_grid = np.linspace(c_low, c_high, n_try)
            residuals = np.array([shoot(k0, c) for c in c_grid])
            # Find sign change
            sign_changes = np.where(np.diff(np.sign(residuals)))[0]
            if len(sign_changes) > 0:
                idx = sign_changes[0]
                c0_opt = brentq(
                    lambda c0: shoot(k0, c0),
                    c_grid[idx],
                    c_grid[idx + 1],
                    xtol=1e-10,
                )
            else:
                # Fallback: use c that gives smallest residual
                c0_opt = c_grid[np.argmin(np.abs(residuals))]

        return c0_opt

    # =========================================================================
    # Solve for Multiple Initial Conditions
    # =========================================================================
    k0_values = [0.25 * k_star, 0.5 * k_star, 0.75 * k_star, 1.5 * k_star, 2.0 * k_star]
    k0_labels = ["0.25 k*", "0.5 k*", "0.75 k*", "1.5 k*", "2.0 k*"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    solutions = []
    t_eval = np.linspace(0, T, n_eval)

    for i, k0 in enumerate(k0_values):
        print(f"  Shooting for k0 = {k0:.4f} ({k0_labels[i]})...")
        c0 = find_saddle_path_c0(k0)
        print(f"    c0 = {c0:.6f}")

        sol = solve_ivp(
            ode_system,
            [0, T],
            [k0, c0],
            method="RK45",
            t_eval=t_eval,
            max_step=0.5,
            rtol=1e-10,
            atol=1e-12,
        )
        solutions.append(sol)

    # =========================================================================
    # Convergence Speed Analysis
    # =========================================================================
    # Linearization around steady state gives convergence rate:
    # The Jacobian at (k*, c*) has eigenvalues with negative real part mu:
    # mu = (1/2)(rho - sqrt(rho^2 + 4*(1/sigma)*c*f''(k*)))
    # f''(k) = alpha*(alpha-1)*A*k^{alpha-2}
    f_double_prime_star = alpha * (alpha - 1) * A * k_star ** (alpha - 2)
    discriminant = rho**2 + 4 * (1.0 / sigma) * (-c_star * f_double_prime_star)
    mu_negative = 0.5 * (rho - np.sqrt(max(discriminant, 0)))
    half_life = -np.log(2) / mu_negative if mu_negative < 0 else np.inf

    print(f"\nConvergence rate: mu = {mu_negative:.4f}")
    print(f"Half-life of convergence: {half_life:.1f} periods")

    # Compute actual convergence for one trajectory
    sol_ref = solutions[0]  # k0 = 0.25 k*
    k_deviation = np.abs(sol_ref.y[0] - k_star)
    # Avoid log of zero
    valid = k_deviation > 1e-12
    if np.sum(valid) > 10:
        t_valid = t_eval[valid]
        log_dev = np.log(k_deviation[valid])
        # Fit linear trend to first portion
        n_fit = min(500, len(t_valid))
        coeffs = np.polyfit(t_valid[:n_fit], log_dev[:n_fit], 1)
        mu_empirical = coeffs[0]
    else:
        mu_empirical = mu_negative

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Ramsey-Cass-Koopmans Growth Model",
        "Optimal saving and consumption in the neoclassical growth model, solved via the shooting method.",
    )

    report.add_overview(
        "The Ramsey model is the foundational continuous-time growth model. A representative "
        "household chooses a consumption path to maximize lifetime utility, taking as given "
        "the neoclassical production technology. Unlike the Solow model, the saving rate is "
        "endogenous --- it emerges from the household's intertemporal optimization.\n\n"
        "The model features saddle-path stability: for any initial capital stock, there is a "
        "unique consumption level that places the economy on the convergent path to the steady "
        "state. The shooting method exploits this structure to solve the boundary value problem."
    )

    report.add_equations(
        r"""
$$\max_{c(t)} \int_0^\infty e^{-\rho t} \, \frac{c(t)^{1-\sigma}}{1-\sigma} \, dt$$

subject to: $\dot{k} = f(k) - \delta k - c$, with $f(k) = A k^\alpha$.

**Euler equation (Keynes-Ramsey rule):**
$$\frac{\dot{c}}{c} = \frac{1}{\sigma} \left( f'(k) - \delta - \rho \right)$$

**Steady state:**
$$f'(k^*) = \delta + \rho \implies k^* = \left(\frac{\alpha A}{\delta + \rho}\right)^{\frac{1}{1-\alpha}}$$
$$c^* = f(k^*) - \delta k^*$$

**Modified golden rule:** The steady-state capital stock equates the net marginal product of capital to the discount rate.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\alpha$  | {alpha} | Capital share |\n"
        f"| $\\delta$  | {delta} | Depreciation rate |\n"
        f"| $\\rho$    | {rho} | Discount rate |\n"
        f"| $\\sigma$  | {sigma} | CRRA coefficient |\n"
        f"| $A$       | {A} | TFP |\n"
        f"| $k^*$     | {k_star:.4f} | Steady-state capital |\n"
        f"| $c^*$     | {c_star:.4f} | Steady-state consumption |"
    )

    report.add_solution_method(
        "**Shooting Method on the Saddle Path:** The Ramsey model is a boundary value problem: "
        "$k(0) = k_0$ is given and the transversality condition requires convergence to the "
        "steady state as $t \\to \\infty$.\n\n"
        "For each initial $k_0$, we search over $c(0)$ using Brent's method (bisection) to find "
        "the unique value that places the economy on the saddle path. Too-high $c(0)$ leads to "
        "capital depletion; too-low $c(0)$ leads to unbounded capital accumulation.\n\n"
        "The ODE system is integrated using `scipy.integrate.solve_ivp` (RK45, adaptive step).\n\n"
        f"**Convergence rate (linearized):** $\\mu = {mu_negative:.4f}$, "
        f"half-life $\\approx {half_life:.1f}$ years.\n"
        f"**Empirical convergence rate:** $\\hat{{\\mu}} = {mu_empirical:.4f}$."
    )

    # --- Figure 1: Phase Diagram (k, c) ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot nullclines
    k_range = np.linspace(0.1, 2.5 * k_star, 300)
    c_nullcline_k = k_star * np.ones_like(k_range)  # dk/dt = 0 is a vertical line... no
    # dk/dt = 0: c = f(k) - delta*k
    c_kdot_zero = A * k_range**alpha - delta * k_range
    c_kdot_zero = np.maximum(c_kdot_zero, 0)

    # dc/dt = 0: f'(k) = delta + rho => k = k* (vertical line)
    ax1.plot(k_range, c_kdot_zero, "k--", linewidth=1.5, alpha=0.6, label="$\\dot{k}=0$")
    ax1.axvline(k_star, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="$\\dot{c}=0$ ($k=k^*$)")

    # Plot saddle paths
    for i, sol in enumerate(solutions):
        ax1.plot(sol.y[0], sol.y[1], color=colors[i], linewidth=2, label=k0_labels[i])
        ax1.plot(sol.y[0, 0], sol.y[1, 0], "o", color=colors[i], markersize=6)

    ax1.plot(k_star, c_star, "k*", markersize=12, zorder=5, label=f"Steady state")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("Consumption $c$")
    ax1.set_title("Phase Diagram: Transition Paths in $(k, c)$ Space")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xlim(0, 2.2 * k_star)
    ax1.set_ylim(0, 1.3 * c_star)
    report.add_figure(
        "figures/phase-diagram.png",
        "Phase diagram showing saddle paths converging to the steady state from different initial capital stocks",
        fig1,
    )

    # --- Figure 2: Time Paths k(t) and c(t) ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    for i, sol in enumerate(solutions):
        ax2a.plot(t_eval, sol.y[0], color=colors[i], linewidth=1.8, label=k0_labels[i])
        ax2b.plot(t_eval, sol.y[1], color=colors[i], linewidth=1.8, label=k0_labels[i])

    ax2a.axhline(k_star, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax2a.set_xlabel("Time $t$")
    ax2a.set_ylabel("Capital $k(t)$")
    ax2a.set_title("Capital Accumulation")
    ax2a.legend(fontsize=9)
    ax2a.set_xlim(0, 100)

    ax2b.axhline(c_star, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax2b.set_xlabel("Time $t$")
    ax2b.set_ylabel("Consumption $c(t)$")
    ax2b.set_title("Consumption Path")
    ax2b.legend(fontsize=9)
    ax2b.set_xlim(0, 100)

    fig2.tight_layout()
    report.add_figure(
        "figures/time-paths.png",
        "Time paths of capital and consumption from different initial conditions, all converging to steady state",
        fig2,
    )

    # --- Figure 3: Convergence Speed ---
    fig3, ax3 = plt.subplots()

    for i, sol in enumerate(solutions):
        dev = np.abs(sol.y[0] - k_star) / k_star
        valid = dev > 1e-10
        if np.any(valid):
            ax3.semilogy(t_eval[valid], dev[valid], color=colors[i], linewidth=1.8, label=k0_labels[i])

    # Theoretical convergence line
    t_theory = np.linspace(0, 100, 200)
    ax3.semilogy(t_theory, np.exp(mu_negative * t_theory), "k--", linewidth=1.5, alpha=0.6,
                 label=f"Linearized ($\\mu={mu_negative:.3f}$)")

    ax3.set_xlabel("Time $t$")
    ax3.set_ylabel("$|k(t) - k^*| / k^*$")
    ax3.set_title("Convergence Speed to Steady State")
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 100)
    report.add_figure(
        "figures/convergence-speed.png",
        "Log-scale convergence of capital to steady state, compared with linearized prediction",
        fig3,
    )

    # --- Table ---
    table_data = {
        "Initial k": [f"{k0:.4f}" for k0 in k0_values],
        "k / k*": [f"{k0/k_star:.2f}" for k0 in k0_values],
        "c(0) (saddle)": [f"{sol.y[1, 0]:.6f}" for sol in solutions],
        "k(50)": [f"{sol.y[0, np.searchsorted(t_eval, 50)]:.4f}" for sol in solutions],
        "c(50)": [f"{sol.y[1, np.searchsorted(t_eval, 50)]:.4f}" for sol in solutions],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/shooting-results.csv",
        "Shooting Method Results for Different Initial Capital Stocks",
        df,
    )

    report.add_takeaway(
        "The Ramsey model reveals the deep structure of optimal saving and capital accumulation "
        "in the neoclassical framework.\n\n"
        "**Key insights:**\n"
        "- The steady-state capital stock satisfies the **modified golden rule**: $f'(k^*) = \\delta + \\rho$. "
        "Unlike the golden rule ($f'(k) = \\delta$), optimizing households do not maximize "
        "steady-state consumption --- they discount the future.\n"
        "- The economy exhibits **saddle-path stability**: only one consumption level is consistent "
        "with intertemporal optimization for each capital stock. All other paths violate either "
        "feasibility or the transversality condition.\n"
        f"- **Convergence is slow**: the half-life is approximately {half_life:.0f} years. This is a "
        "well-known feature of the neoclassical model and partly explains persistent cross-country "
        "income differences.\n"
        "- Along the saddle path, capital-poor economies have *lower* consumption but *higher* saving "
        "rates, consistent with the empirical convergence literature.\n"
        "- The shooting method is the natural numerical approach for saddle-path systems: it converts "
        "a two-point BVP into a sequence of initial value problems."
    )

    report.add_references([
        "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.",
        "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 8.",
        "Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 2.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
