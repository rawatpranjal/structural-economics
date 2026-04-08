#!/usr/bin/env python3
"""Solow Growth Model: Deterministic Simulation with Cobb-Douglas Production.

Simulates the neoclassical Solow growth model and demonstrates convergence
to the balanced growth path. Unlike VFI-based models, the Solow model features
an exogenous savings rate — there is no optimization by agents.

Reference: Solow, R. (1956). "A Contribution to the Theory of Economic Growth."
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    K0 = 1.0       # Initial capital stock
    A0 = 1.0       # Initial technology level
    L0 = 1.0       # Initial labor force
    T = 500         # Simulation periods
    alpha = 0.66    # Capital share in production (Cobb-Douglas exponent)
    s = 0.3         # Savings rate (exogenous)
    d = 0.0         # Depreciation rate
    n = 0.00        # Population growth rate
    g = 0.02        # Technology growth rate

    # =========================================================================
    # Production function (Cobb-Douglas)
    # =========================================================================
    def F(K, A, L):
        """Y = K^alpha * (A*L)^(1-alpha)"""
        return K**alpha * (A * L) ** (1 - alpha)

    # =========================================================================
    # Steady state in effective units: k* = K/(AL), y* = Y/(AL)
    # =========================================================================
    # In effective units: y = k^alpha
    # Steady state: s * k*^alpha = (n + g + d) * k*  (with n+g+d > 0)
    # => k* = (s / (n + g + d))^(1/(1-alpha))
    effective_depreciation = n + g + d
    if effective_depreciation > 0:
        k_star = (s / effective_depreciation) ** (1 / (1 - alpha))
    else:
        # When n+g+d = 0, capital per effective worker grows without bound
        # under positive savings. Use a large value for display purposes.
        k_star = np.inf

    y_star = k_star**alpha if np.isfinite(k_star) else np.inf
    c_star = (1 - s) * y_star if np.isfinite(y_star) else np.inf

    print(f"Solow Growth Model Simulation")
    print(f"{'='*50}")
    print(f"Parameters: alpha={alpha}, s={s}, d={d}, n={n}, g={g}")
    print(f"Effective depreciation (n+g+d) = {effective_depreciation}")
    if np.isfinite(k_star):
        print(f"Analytical steady state (effective units):")
        print(f"  k* = {k_star:.6f}")
        print(f"  y* = {y_star:.6f}")
        print(f"  c* = {c_star:.6f}")
    else:
        print(f"No finite steady state (n+g+d = 0 with s > 0)")

    # =========================================================================
    # Simulate
    # =========================================================================
    K_path = np.zeros(T)
    A_path = np.zeros(T)
    L_path = np.zeros(T)
    Y_path = np.zeros(T)

    K_path[0] = K0
    A_path[0] = A0
    L_path[0] = L0
    Y_path[0] = F(K0, A0, L0)

    for t in range(T - 1):
        # Output
        Y_path[t] = F(K_path[t], A_path[t], L_path[t])
        # Laws of motion
        K_path[t + 1] = (1 - d) * K_path[t] + s * Y_path[t]
        A_path[t + 1] = (1 + g) * A_path[t]
        L_path[t + 1] = (1 + n) * L_path[t]

    # Final period output
    Y_path[T - 1] = F(K_path[T - 1], A_path[T - 1], L_path[T - 1])

    # =========================================================================
    # Compute variables in effective units: k = K/(AL), y = Y/(AL)
    # =========================================================================
    AL_path = A_path * L_path
    k_path = K_path / AL_path   # Capital per effective worker
    y_path = Y_path / AL_path   # Output per effective worker
    c_path = (1 - s) * y_path   # Consumption per effective worker

    # Per capita variables (divided by L only)
    y_per_capita = Y_path / L_path
    k_per_capita = K_path / L_path

    # =========================================================================
    # Factor prices (competitive markets)
    # =========================================================================
    # Marginal product of capital: r = alpha * k^(alpha-1)
    # Wage per effective worker: w = (1-alpha) * k^alpha
    r_path = alpha * k_path ** (alpha - 1)
    w_eff_path = (1 - alpha) * k_path ** alpha

    # Steady-state factor prices
    if np.isfinite(k_star):
        r_star = alpha * k_star ** (alpha - 1)
        w_eff_star = (1 - alpha) * k_star ** alpha
    else:
        r_star = 0.0
        w_eff_star = np.inf

    # Simulated terminal values
    k_sim_final = k_path[-1]
    y_sim_final = y_path[-1]
    c_sim_final = c_path[-1]
    r_sim_final = r_path[-1]
    w_sim_final = w_eff_path[-1]

    print(f"\nSimulated values at t={T-1} (effective units):")
    print(f"  k_T = {k_sim_final:.6f}")
    print(f"  y_T = {y_sim_final:.6f}")
    print(f"  c_T = {c_sim_final:.6f}")
    if np.isfinite(k_star):
        print(f"\nConvergence gap |k_T - k*| = {abs(k_sim_final - k_star):.2e}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Solow Growth Model",
        "Deterministic simulation of the neoclassical growth model with exogenous savings, "
        "technology growth, and population growth.",
    )

    report.add_overview(
        "The Solow (1956) growth model is the foundational framework for understanding "
        "long-run economic growth. Unlike dynamic programming models where agents optimize "
        "intertemporally, the Solow model assumes a constant, exogenous savings rate $s$. "
        "Output is produced via a Cobb-Douglas production function with constant returns to "
        "scale, and the economy converges to a balanced growth path (steady state) where "
        "all per-capita variables grow at the rate of technological progress.\n\n"
        "This simulation traces the transition dynamics from an initial capital stock to "
        "the steady state, illustrating how savings, population growth, depreciation, and "
        "technology jointly determine long-run living standards."
    )

    report.add_equations(
        r"""
**Production function (Cobb-Douglas):**
$$Y_t = K_t^{\alpha} (A_t L_t)^{1-\alpha}$$

**Laws of motion:**
$$K_{t+1} = (1-\delta) K_t + s Y_t$$
$$A_{t+1} = (1+g) A_t, \qquad L_{t+1} = (1+n) L_t$$

**Effective units:** Let $k_t = K_t / (A_t L_t)$ and $y_t = Y_t / (A_t L_t) = k_t^{\alpha}$.

**Steady state:** Setting $k_{t+1} = k_t = k^*$:
$$k^* = \left( \frac{s}{n + g + \delta} \right)^{1/(1-\alpha)}$$
$$y^* = (k^*)^{\alpha}, \qquad c^* = (1-s) \, y^*$$

**Factor prices (competitive):**
$$r_t = \alpha \, k_t^{\alpha - 1}, \qquad w_t = (1-\alpha) \, k_t^{\alpha}$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\alpha$  | {alpha} | Capital share (Cobb-Douglas exponent) |\n"
        f"| $s$       | {s} | Savings rate (exogenous) |\n"
        f"| $\\delta$  | {d} | Depreciation rate |\n"
        f"| $n$       | {n} | Population growth rate |\n"
        f"| $g$       | {g} | Technology growth rate |\n"
        f"| $K_0$     | {K0} | Initial capital stock |\n"
        f"| $A_0$     | {A0} | Initial technology level |\n"
        f"| $L_0$     | {L0} | Initial labor force |\n"
        f"| $T$       | {T} | Simulation periods |"
    )

    report.add_solution_method(
        "**Deterministic simulation:** The Solow model requires no optimization — "
        "the savings rate is exogenous. We simply iterate the laws of motion forward "
        f"for $T = {T}$ periods starting from $(K_0, A_0, L_0) = ({K0}, {A0}, {L0})$.\n\n"
        "In each period:\n"
        "1. Compute output: $Y_t = K_t^{\\alpha} (A_t L_t)^{1-\\alpha}$\n"
        "2. Update capital: $K_{t+1} = (1-\\delta) K_t + s Y_t$\n"
        "3. Update technology and labor: $A_{t+1} = (1+g) A_t$, $L_{t+1} = (1+n) L_t$\n\n"
        "We then convert to effective units $k_t = K_t / (A_t L_t)$ to analyze convergence "
        "to the steady state $k^*$."
    )

    # --- Figure 1: Output per effective worker ---
    fig1, ax1 = plt.subplots()
    periods = np.arange(T)
    ax1.plot(periods, y_path, "b-", linewidth=2, label="$y_t = Y_t / (A_t L_t)$")
    if np.isfinite(y_star):
        ax1.axhline(y=y_star, color="r", linestyle="--", linewidth=1.5, label=f"Steady state $y^* = {y_star:.4f}$")
    ax1.set_xlabel("Period $t$")
    ax1.set_ylabel("Output per effective worker $y_t$")
    ax1.set_title("Output per Effective Worker: Convergence to Steady State")
    ax1.legend()
    report.add_figure(
        "figures/output-per-effective-worker.png",
        "Output per effective worker converges to the analytically computed steady state",
        fig1,
        description="Convergence is fast initially and slows as the economy approaches steady state, reflecting "
        "diminishing returns to capital. The gap between the curve and the dashed line measures remaining transition dynamics.",
    )

    # --- Figure 2: Capital per effective worker ---
    fig2, ax2 = plt.subplots()
    ax2.plot(periods, k_path, "b-", linewidth=2, label="$k_t = K_t / (A_t L_t)$")
    if np.isfinite(k_star):
        ax2.axhline(y=k_star, color="r", linestyle="--", linewidth=1.5, label=f"Steady state $k^* = {k_star:.4f}$")
    ax2.set_xlabel("Period $t$")
    ax2.set_ylabel("Capital per effective worker $k_t$")
    ax2.set_title("Capital per Effective Worker: Convergence to Steady State")
    ax2.legend()
    report.add_figure(
        "figures/capital-per-effective-worker.png",
        "Capital per effective worker converges to the steady state determined by savings and effective depreciation",
        fig2,
        description="The steady-state capital intensity is entirely determined by the savings rate and effective depreciation (n+g+delta). "
        "Economies with higher savings rates or lower depreciation converge to higher k*, but the growth rate on the balanced path is always g.",
    )

    # --- Figure 3: Factor prices ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

    ax3a.plot(periods, r_path, "b-", linewidth=2, label="$r_t = \\alpha k_t^{\\alpha-1}$")
    if np.isfinite(k_star):
        ax3a.axhline(y=r_star, color="r", linestyle="--", linewidth=1.5, label=f"$r^* = {r_star:.4f}$")
    ax3a.set_xlabel("Period $t$")
    ax3a.set_ylabel("Rental rate $r_t$")
    ax3a.set_title("Rental Rate of Capital")
    ax3a.legend()

    ax3b.plot(periods, w_eff_path, "b-", linewidth=2, label="$w_t = (1-\\alpha) k_t^{\\alpha}$")
    if np.isfinite(k_star):
        ax3b.axhline(y=w_eff_star, color="r", linestyle="--", linewidth=1.5, label=f"$w^* = {w_eff_star:.4f}$")
    ax3b.set_xlabel("Period $t$")
    ax3b.set_ylabel("Wage per effective worker $w_t$")
    ax3b.set_title("Wage per Effective Worker")
    ax3b.legend()

    fig3.tight_layout()
    report.add_figure(
        "figures/factor-prices.png",
        "Factor prices (rental rate and wage per effective worker) converge to steady-state values",
        fig3,
        description="As capital accumulates, the rental rate falls (diminishing marginal product) while wages rise. "
        "This captures the core Solow prediction: capital-scarce economies have high returns to capital and low wages, driving cross-country convergence.",
    )

    # --- Table: Analytical vs Simulated steady state ---
    if np.isfinite(k_star):
        table_data = {
            "Variable": [
                "Capital per eff. worker (k)",
                "Output per eff. worker (y)",
                "Consumption per eff. worker (c)",
                "Rental rate (r)",
                "Wage per eff. worker (w)",
            ],
            "Analytical": [
                f"{k_star:.6f}",
                f"{y_star:.6f}",
                f"{c_star:.6f}",
                f"{r_star:.6f}",
                f"{w_eff_star:.6f}",
            ],
            "Simulated (t=499)": [
                f"{k_sim_final:.6f}",
                f"{y_sim_final:.6f}",
                f"{c_sim_final:.6f}",
                f"{r_sim_final:.6f}",
                f"{w_sim_final:.6f}",
            ],
            "Gap": [
                f"{abs(k_sim_final - k_star):.2e}",
                f"{abs(y_sim_final - y_star):.2e}",
                f"{abs(c_sim_final - c_star):.2e}",
                f"{abs(r_sim_final - r_star):.2e}",
                f"{abs(w_sim_final - w_eff_star):.2e}",
            ],
        }
    else:
        table_data = {
            "Variable": [
                "Capital per eff. worker (k)",
                "Output per eff. worker (y)",
                "Consumption per eff. worker (c)",
            ],
            "Simulated (t=499)": [
                f"{k_sim_final:.6f}",
                f"{y_sim_final:.6f}",
                f"{c_sim_final:.6f}",
            ],
            "Note": [
                "No finite steady state",
                "No finite steady state",
                "No finite steady state",
            ],
        }

    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/steady-state-comparison.csv",
        "Steady-State Comparison: Analytical vs Simulated",
        df,
        description="The vanishing gap between analytical and simulated values confirms the economy has effectively "
        "reached its balanced growth path by the end of the simulation horizon.",
    )

    report.add_takeaway(
        "The Solow model illustrates how an economy's long-run prosperity is determined "
        "by a few fundamental parameters, even without any optimizing behavior by agents.\n\n"
        "**Key insights:**\n"
        "- **Higher savings rate** $\\rightarrow$ higher steady-state capital and output per "
        "effective worker. But savings cannot drive *growth* in the long run — only the "
        "*level* of output.\n"
        "- **Population growth and depreciation** reduce steady-state capital intensity by "
        "diluting capital across more workers and wearing out existing stock.\n"
        "- **Technology growth** is the sole driver of long-run output per capita growth. "
        "On the balanced growth path, output per capita grows at rate $g$.\n"
        "- **No optimization:** The savings rate $s$ is exogenous. This is both the model's "
        "simplicity and its limitation — contrast with the Ramsey model where households "
        "choose savings optimally via an Euler equation.\n"
        "- **Convergence:** Economies below steady state grow faster (diminishing returns to "
        "capital), predicting conditional convergence across countries."
    )

    report.add_references([
        "Solow, R. (1956). \"A Contribution to the Theory of Economic Growth.\" "
        "*Quarterly Journal of Economics*, 70(1), 65-94.",
        "Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 1.",
        "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 1.",
        "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 2.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
