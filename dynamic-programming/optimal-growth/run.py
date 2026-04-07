#!/usr/bin/env python3
"""Neoclassical Optimal Growth (Ramsey-Cass-Koopmans): Deterministic Case.

Solves the infinite-horizon optimal growth model using value function iteration
with JAX. The representative agent chooses consumption to maximize discounted
utility subject to a Cobb-Douglas production technology.

Reference: Stokey, Lucas, and Prescott (1989), Ch. 2 & 4.
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
    alpha = 0.3       # Capital share in production
    A = 18.5          # Total factor productivity
    beta = 0.9        # Discount factor
    n_grid = 500      # Grid points for capital
    n_kprime = 500    # Grid points for k' in inner maximization
    tol = 1e-6        # Convergence tolerance

    # Steady state capital: kss = (alpha * beta * A)^(1/(1-alpha))
    kss = (alpha * beta * A) ** (1 / (1 - alpha))

    # Grid bounds: from small positive to well above steady state
    k_min = 0.01
    k_max = kss * 2.5

    # =========================================================================
    # Grid (uniform)
    # =========================================================================
    k_grid_np = np.linspace(k_min, k_max, n_grid)

    # =========================================================================
    # Production and utility functions
    # =========================================================================
    def f_np(k):
        return A * k ** alpha

    def u_np(c):
        return np.log(np.maximum(c, 1e-15))

    # =========================================================================
    # Analytical solution (log utility, Cobb-Douglas)
    # =========================================================================
    # V(k) = E + F*log(k) where:
    #   E = (1/(1-beta))*(log(A*(1-alpha*beta)) + beta*alpha*log(A*alpha*beta)/(1-alpha*beta))
    #   F = alpha/(1-alpha*beta)
    # Policy: k'(k) = alpha*beta*A*k^alpha  (savings = alpha*beta fraction of output)
    E_const = (1 / (1 - beta)) * (
        np.log(A * (1 - alpha * beta))
        + beta * alpha * np.log(A * alpha * beta) / (1 - alpha * beta)
    )
    F_const = alpha / (1 - alpha * beta)

    def analytical_v(k):
        return E_const + F_const * np.log(np.maximum(k, 1e-15))

    def analytical_policy(k):
        """Optimal next-period capital: k' = alpha*beta*F(k)."""
        return alpha * beta * A * np.maximum(k, 1e-15) ** alpha

    # =========================================================================
    # Interpolation with analytical boundary extrapolation
    # =========================================================================
    def v_interp(kprime, v_np):
        """Interpolate V with analytical boundary below grid minimum."""
        result = np.interp(kprime, k_grid_np, v_np)
        below = kprime < k_grid_np[0]
        if np.any(below):
            result[below] = analytical_v(kprime[below])
        above = kprime > k_grid_np[-1]
        if np.any(above):
            result[above] = analytical_v(kprime[above])
        return result

    # =========================================================================
    # Solve via VFI with continuous optimization
    # =========================================================================
    # For each state k, maximize u(F(k) - k') + beta * V(k') over a fine grid
    # of k' values, interpolating V between state grid points.
    v = u_np(f_np(k_grid_np))  # Initial guess: consume all output today

    for iteration in range(1, 1001):
        v_new = np.zeros(n_grid)
        policy_kprime = np.zeros(n_grid)

        for ik in range(n_grid):
            k = k_grid_np[ik]
            output = f_np(k)
            # k' must be in [small positive, output - small positive] for c > 0
            kp_max = output * 0.9999
            kp_grid = np.linspace(1e-8, kp_max, n_kprime)
            consumption = output - kp_grid
            values = u_np(consumption) + beta * v_interp(kp_grid, v)
            best = np.argmax(values)
            v_new[ik] = values[best]
            policy_kprime[ik] = kp_grid[best]

        error = np.max(np.abs(v_new - v))
        if iteration % 10 == 0:
            print(f"  VFI iteration {iteration:3d}, error = {error:.2e}")
        v = v_new

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    v_star = jnp.array(v)
    k_grid = jnp.array(k_grid_np)
    policy_kprime_jnp = jnp.array(policy_kprime)
    consumption_policy = A * k_grid ** alpha - policy_kprime_jnp

    info = {"iterations": iteration, "converged": error < tol, "error": error}

    # =========================================================================
    # Analytical solution on the grid
    # =========================================================================
    v_analytical = jnp.array(analytical_v(k_grid_np))
    policy_kprime_analytical = jnp.array(analytical_policy(k_grid_np))
    consumption_analytical = A * k_grid ** alpha - policy_kprime_analytical

    # =========================================================================
    # Simulate capital dynamics
    # =========================================================================
    T_sim = 50
    k0 = kss * 0.1  # Start well below steady state
    capital_path = np.zeros(T_sim)
    capital_path[0] = k0
    for t in range(T_sim - 1):
        kp = np.interp(capital_path[t], k_grid_np, policy_kprime)
        capital_path[t + 1] = kp
    output_path = A * capital_path ** alpha
    consumption_path = output_path - np.concatenate([capital_path[1:], [np.nan]])
    capital_path = jnp.array(capital_path)
    output_path = jnp.array(output_path)
    consumption_path = jnp.array(consumption_path)

    print(f"\n  Steady state capital (analytical): kss = {kss:.4f}")
    print(f"  Final capital in simulation:       k_T = {float(capital_path[-1]):.4f}")
    print(f"  Optimal savings rate:              s   = alpha*beta = {alpha*beta:.2f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Neoclassical Optimal Growth Model",
        "The Ramsey-Cass-Koopmans model: optimal consumption and capital accumulation with a Cobb-Douglas production technology.",
    )

    report.add_overview(
        "The neoclassical optimal growth model (Ramsey-Cass-Koopmans) is a foundational "
        "model in macroeconomics. A representative agent chooses consumption each period "
        "to maximize discounted lifetime utility, subject to a production technology that "
        "transforms capital into output.\n\n"
        "Unlike the cake-eating problem, capital is *productive* here: saving today yields "
        "more output tomorrow via the production function $F(k) = Ak^\\alpha$. This creates "
        "a non-trivial steady state where the economy converges regardless of its initial "
        "capital stock."
    )

    report.add_equations(
        r"""
$$V(k) = \max_{0 \le k' \le F(k)} \left\{ u(F(k) - k') + \beta \, V(k') \right\}$$

where $k$ is capital, $k'$ is next-period capital, $c = F(k) - k'$ is consumption,
$F(k) = Ak^\alpha$ is the production function, and $\beta \in (0,1)$ is the discount factor.

**Log utility:** $u(c) = \ln(c)$

**Analytical solution:**
$$V(k) = E + F \ln(k), \qquad E = \frac{\ln(A(1-\alpha\beta)) + \frac{\beta\alpha\ln(A\alpha\beta)}{1-\alpha\beta}}{1-\beta}, \quad F = \frac{\alpha}{1-\alpha\beta}$$

**Optimal policy:** $k'(k) = \alpha \beta A k^\alpha$ (save fraction $\alpha\beta$ of output)

**Steady state:** $k_{ss} = (\alpha \beta A)^{1/(1-\alpha)}$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\alpha$  | {alpha} | Capital share (Cobb-Douglas) |\n"
        f"| $A$       | {A} | Total factor productivity |\n"
        f"| $\\beta$   | {beta} | Discount factor |\n"
        f"| $k_{{ss}}$ | {kss:.4f} | Steady state capital |\n"
        f"| Grid points | {n_grid} | Uniform spacing |\n"
        f"| $k \\in$   | [{k_min}, {k_max:.2f}] | Capital range |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI):** Starting from an initial guess "
        "$V_0(k) = u(F(k))$, we iterate on the Bellman equation:\n\n"
        "$$V_{n+1}(k) = \\max_{0 \\le k' \\le F(k)} \\left\\{ u(F(k) - k') + \\beta \\, V_n(k') \\right\\}$$\n\n"
        "until $\\|V_{n+1} - V_n\\|_\\infty < 10^{-6}$. At each state, we search over a fine "
        "grid of $k'$ values and interpolate the continuation value between grid points. "
        "The analytical solution provides boundary extrapolation for $k'$ values outside the "
        f"grid range.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e})."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, v_star, "b-", linewidth=2, label="Numerical (VFI)")
    ax1.plot(k_grid, v_analytical, "r--", linewidth=1.5, label="Analytical")
    ax1.axvline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"$k_{{ss}} = {kss:.2f}$")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k)$")
    ax1.set_title("Value Function")
    ax1.legend()
    report.add_figure("figures/value-function.png", "Value function: numerical VFI vs analytical solution", fig1)

    # --- Figure 2: Policy Function ---
    fig2, ax2 = plt.subplots()
    ax2.plot(k_grid, policy_kprime_jnp, "b-", linewidth=2, label="Numerical $k'(k)$")
    ax2.plot(k_grid, policy_kprime_analytical, "r--", linewidth=1.5, label="Analytical $\\alpha\\beta F(k)$")
    ax2.plot(k_grid, k_grid, "k:", linewidth=0.8, alpha=0.5, label="45-degree line")
    ax2.axvline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"$k_{{ss}}$")
    ax2.set_xlabel("Capital $k$")
    ax2.set_ylabel("Next-period capital $k'$")
    ax2.set_title("Capital Policy Function")
    ax2.legend()
    report.add_figure("figures/policy-function.png", "Capital policy function: numerical vs analytical", fig2)

    # --- Figure 3: Simulation ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    periods = jnp.arange(T_sim)

    ax3a.plot(periods, capital_path, "b-o", markersize=3, linewidth=1.5)
    ax3a.axhline(kss, color="r", linestyle="--", linewidth=1, alpha=0.7, label=f"$k_{{ss}} = {kss:.2f}$")
    ax3a.set_xlabel("Period")
    ax3a.set_ylabel("Capital $k_t$")
    ax3a.set_title("Capital Dynamics")
    ax3a.legend()

    ax3b.plot(periods[:-1], consumption_path[:-1], "r-o", markersize=3, linewidth=1.5)
    css = A * kss ** alpha - kss  # Steady state consumption
    ax3b.axhline(css, color="b", linestyle="--", linewidth=1, alpha=0.7, label=f"$c_{{ss}} = {css:.2f}$")
    ax3b.set_xlabel("Period")
    ax3b.set_ylabel("Consumption $c_t$")
    ax3b.set_title("Consumption Over Time")
    ax3b.legend()
    fig3.tight_layout()
    report.add_figure("figures/simulation.png", f"Simulation: capital and consumption converging to steady state from k0={k0:.2f}", fig3)

    # --- Table: Numerical vs Analytical ---
    valid_start = max(1, n_grid // 10)  # Skip bottom 10% of grid
    sample_idx = np.linspace(valid_start, n_grid - 1, 8, dtype=int)
    table_data = {
        "k": [f"{float(k_grid[i]):.3f}" for i in sample_idx],
        "V(k) numerical": [f"{float(v_star[i]):.4f}" for i in sample_idx],
        "V(k) analytical": [f"{float(v_analytical[i]):.4f}" for i in sample_idx],
        "k' numerical": [f"{float(policy_kprime_jnp[i]):.4f}" for i in sample_idx],
        "k' analytical": [f"{float(policy_kprime_analytical[i]):.4f}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_table("tables/comparison.csv", "Numerical vs Analytical Solution at Selected Grid Points", df)

    report.add_takeaway(
        "The neoclassical growth model reveals how productive capital creates a "
        "non-trivial steady state, unlike the cake-eating problem where the resource "
        "monotonically declines.\n\n"
        "**Key insights:**\n"
        f"- Capital converges to the steady state $k_{{ss}} = {kss:.2f}$ regardless of "
        "initial conditions. The economy self-corrects: low capital means high marginal "
        "product, incentivizing saving.\n"
        f"- The optimal savings rate is $\\alpha\\beta = {alpha*beta:.2f}$. More patient agents "
        "(higher $\\beta$) or more capital-intensive technologies (higher $\\alpha$) lead to "
        "greater capital accumulation.\n"
        "- The policy function $k'(k) = \\alpha\\beta F(k)$ shows that the agent saves a "
        "constant fraction of *output* (not wealth), reflecting the log utility / "
        "Cobb-Douglas structure.\n"
        "- VFI converges reliably because the Bellman operator is a contraction mapping. "
        "The analytical solution provides an exact benchmark for validation."
    )

    report.add_references([
        "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 2 & 4.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.",
        "Ramsey, F. (1928). A Mathematical Theory of Saving. *Economic Journal*, 38(152), 543-559.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
