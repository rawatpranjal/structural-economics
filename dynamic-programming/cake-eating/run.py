#!/usr/bin/env python3
"""Cake-Eating Problem: Optimal Consumption of a Finite Resource.

Solves the infinite-horizon cake-eating problem using value function iteration
with JAX. This is the simplest dynamic programming problem: how to optimally
consume a non-renewable resource over time.

Reference: Stokey, Lucas, and Prescott (1989), Ch. 4.
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
from lib.grids import uniform_grid
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    beta = 0.9       # Discount factor
    sigma = 1.0      # CRRA coefficient (1.0 = log utility)
    n_grid = 500     # Grid points for cake size
    n_cons = 300     # Consumption grid points for inner maximization
    w_min = 0.01     # Minimum cake size
    w_max = 1.0      # Maximum cake size (initial endowment)
    tol = 1e-6       # Convergence tolerance

    # =========================================================================
    # Grid (uniform)
    # =========================================================================
    w_grid_np = np.linspace(w_min, w_max, n_grid)
    w_grid = jnp.array(w_grid_np)

    # =========================================================================
    # Utility function
    # =========================================================================
    u_np = np.log if sigma == 1.0 else (lambda c: c ** (1 - sigma) / (1 - sigma))
    u_vec = lambda c: np.log(np.maximum(c, 1e-15))

    # =========================================================================
    # Analytical solution (used for boundary extrapolation below grid)
    # =========================================================================
    def analytical_v(w):
        return np.log((1 - beta) * np.maximum(w, 1e-15)) / (1 - beta) + beta * np.log(beta) / (1 - beta) ** 2

    def v_interp(wprime, v_np):
        """Interpolate V with analytical boundary below grid minimum."""
        result = np.interp(wprime, w_grid_np, v_np)
        below = wprime < w_grid_np[0]
        if np.any(below):
            result[below] = analytical_v(wprime[below])
        return result

    # =========================================================================
    # Solve via VFI with continuous optimization
    # =========================================================================
    # For each state W, maximize u(c) + beta * V(W-c) over a fine consumption
    # grid, interpolating V between state grid points.
    v = u_vec(w_grid_np)  # Initial guess: eat everything today

    for iteration in range(1, 501):
        v_new = np.zeros(n_grid)
        policy_c = np.zeros(n_grid)

        for ia in range(n_grid):
            cake = w_grid_np[ia]
            c_grid = np.linspace(1e-8, cake * 0.9999, n_cons)
            wprime = cake - c_grid
            values = u_vec(c_grid) + beta * v_interp(wprime, v)
            best = np.argmax(values)
            v_new[ia] = values[best]
            policy_c[ia] = c_grid[best]

        error = np.max(np.abs(v_new - v))
        if iteration % 10 == 0:
            print(f"  VFI iteration {iteration:3d}, error = {error:.2e}")
        v = v_new

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    v_star = jnp.array(v)
    consumption_policy = jnp.array(policy_c)
    policy_cake = w_grid - consumption_policy

    info = {"iterations": iteration, "converged": error < tol, "error": error}

    # =========================================================================
    # Analytical solution (log utility case)
    # =========================================================================
    v_analytical = (
        jnp.log((1 - beta) * w_grid) / (1 - beta)
        + beta * jnp.log(beta) / (1 - beta) ** 2
    )
    policy_analytical = beta * w_grid         # W' = beta * W
    consumption_analytical = (1 - beta) * w_grid  # c = (1-beta) * W

    # =========================================================================
    # Simulate cake path
    # =========================================================================
    T_sim = 30
    cake_path = jnp.zeros(T_sim)
    cake_path = cake_path.at[0].set(w_max)
    for t in range(T_sim - 1):
        # Interpolate policy
        w_prime = jnp.interp(cake_path[t], w_grid, policy_cake)
        cake_path = cake_path.at[t + 1].set(w_prime)
    consumption_path = jnp.concatenate([
        cake_path[:-1] - cake_path[1:],
        jnp.array([0.0]),
    ])

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Cake-Eating Problem",
        "Optimal consumption of a finite, non-renewable resource over an infinite horizon.",
    )

    report.add_overview(
        "The cake-eating problem is the simplest dynamic programming model. An agent "
        "has a cake of size $W$ and must decide how much to eat each period. The cake "
        "does not grow — any portion not consumed today is saved for tomorrow. The agent "
        "discounts the future at rate $\\beta$ and has CRRA utility over consumption.\n\n"
        "This model introduces the core machinery of dynamic programming: Bellman equations, "
        "value function iteration, and policy functions."
    )

    report.add_equations(
        r"""
$$V(W) = \max_{0 \le c \le W} \left\{ u(c) + \beta \, V(W - c) \right\}$$

where $W$ is the remaining cake, $c$ is consumption, and $\beta \in (0,1)$ is the
discount factor.

**CRRA utility:** $u(c) = \frac{c^{1-\sigma}}{1-\sigma}$, with $u(c) = \ln(c)$ when $\sigma = 1$.

**Analytical solution (log utility):**
$$V(W) = \frac{\ln((1-\beta) W)}{1-\beta} + \frac{\beta \ln \beta}{(1-\beta)^2}$$
$$c^*(W) = (1-\beta) W, \qquad W' = \beta W$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $\\sigma$ | {sigma} | CRRA coefficient |\n"
        f"| Grid points | {n_grid} | Uniform spacing |\n"
        f"| $W \\in$  | [{w_min}, {w_max}] | Cake size range |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI):** Starting from an initial guess "
        "$V_0(W) = u(W)$, we iterate on the Bellman equation:\n\n"
        "$$V_{n+1}(W) = \\max_{0 \\le c \\le W} \\left\\{ u(c) + \\beta \\, V_n(W-c) \\right\\}$$\n\n"
        "until $\\|V_{n+1} - V_n\\|_\\infty < 10^{-6}$. The Bellman operator is a contraction "
        "mapping (by the Blackwell sufficient conditions), guaranteeing convergence to the "
        f"unique fixed point.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e})."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(w_grid, v_star, "b-", linewidth=2, label="Numerical (VFI)")
    ax1.plot(w_grid, v_analytical, "r--", linewidth=1.5, label="Analytical")
    ax1.set_xlabel("Cake size $W$")
    ax1.set_ylabel("$V(W)$")
    ax1.set_title("Value Function")
    ax1.legend()
    report.add_figure("figures/value-function.png", "Value function: numerical VFI vs analytical solution", fig1)

    # --- Figure 2: Policy Function ---
    fig2, ax2 = plt.subplots()
    ax2.plot(w_grid, consumption_policy, "b-", linewidth=2, label="Numerical $c^*(W)$")
    ax2.plot(w_grid, consumption_analytical, "r--", linewidth=1.5, label="Analytical $(1-\\beta)W$")
    ax2.plot(w_grid, w_grid, "k:", linewidth=0.8, alpha=0.5, label="45-degree line")
    ax2.set_xlabel("Cake size $W$")
    ax2.set_ylabel("Consumption $c$")
    ax2.set_title("Consumption Policy Function")
    ax2.legend()
    report.add_figure("figures/policy-function.png", "Consumption policy: numerical vs analytical", fig2)

    # --- Figure 3: Simulation ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    periods = jnp.arange(T_sim)
    ax3a.plot(periods, cake_path, "b-o", markersize=3, linewidth=1.5)
    ax3a.set_xlabel("Period")
    ax3a.set_ylabel("Cake remaining $W_t$")
    ax3a.set_title("Cake Depletion Over Time")

    ax3b.plot(periods, consumption_path, "r-o", markersize=3, linewidth=1.5)
    ax3b.set_xlabel("Period")
    ax3b.set_ylabel("Consumption $c_t$")
    ax3b.set_title("Consumption Over Time")
    fig3.tight_layout()
    report.add_figure("figures/simulation.png", "Simulation: cake depletion and consumption paths starting from W=1", fig3)

    # --- Table: Numerical vs Analytical (skip poorly-approximated bottom) ---
    valid_start = max(1, n_grid // 10)  # Skip bottom 10% of grid
    sample_idx = jnp.linspace(valid_start, n_grid - 1, 8, dtype=jnp.int32)
    table_data = {
        "W": [f"{float(w_grid[i]):.3f}" for i in sample_idx],
        "V(W) numerical": [f"{float(v_star[i]):.4f}" for i in sample_idx],
        "V(W) analytical": [f"{float(v_analytical[i]):.4f}" for i in sample_idx],
        "c* numerical": [f"{float(consumption_policy[i]):.4f}" for i in sample_idx],
        "c* analytical": [f"{float(consumption_analytical[i]):.4f}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/comparison.csv", "Numerical vs Analytical Solution at Selected Grid Points", df)

    report.add_takeaway(
        "The cake-eating problem reveals the fundamental trade-off in intertemporal "
        "optimization: consuming today yields immediate utility, but saving preserves "
        "options for the future.\n\n"
        "**Key insights:**\n"
        "- The optimal policy is *linear* in wealth: consume a fixed fraction $(1-\\beta)$ "
        "each period. More patient agents (higher $\\beta$) consume less today.\n"
        "- The cake shrinks geometrically: $W_t = \\beta^t W_0$. The resource is never "
        "fully exhausted in finite time but asymptotically approaches zero.\n"
        "- VFI converges reliably because the Bellman operator is a contraction mapping — "
        "this is the workhorse method for solving dynamic programs.\n"
        "- The analytical solution provides a benchmark for validating numerical methods. "
        "Any VFI implementation should be tested against this known solution first."
    )

    report.add_references([
        "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 4.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
