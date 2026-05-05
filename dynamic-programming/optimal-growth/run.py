#!/usr/bin/env python3
"""Neoclassical Optimal Growth (Ramsey-Cass-Koopmans): Deterministic Case.

Solves the infinite-horizon optimal growth model using value function iteration
with JAX. The representative agent chooses consumption to maximize discounted
utility subject to a Cobb-Douglas production technology.

Reference: Stokey, Lucas, and Prescott (1989), Ch. 2 & 4.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def main() -> None:
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
    # Interpolation on the capital grid
    # =========================================================================
    def v_interp(kprime, v_np):
        """Interpolate the current value-function guess at off-grid choices."""
        return np.interp(kprime, k_grid_np, v_np)

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
            # k' must keep consumption positive. The analytical optimum is
            # interior for this calibration, so the state grid covers the choice
            # region used by the maximization.
            kp_max = min(output * 0.9999, k_max)
            kp_grid = np.linspace(k_min, kp_max, n_kprime)
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

    capital_path_exact = np.zeros(T_sim)
    capital_path_exact[0] = k0
    for t in range(T_sim - 1):
        capital_path_exact[t + 1] = analytical_policy(capital_path_exact[t])
    output_path_exact = A * capital_path_exact ** alpha
    consumption_path_exact = output_path_exact - np.concatenate([capital_path_exact[1:], [np.nan]])

    capital_path = jnp.array(capital_path)
    output_path = jnp.array(output_path)
    consumption_path = jnp.array(consumption_path)
    capital_path_exact = jnp.array(capital_path_exact)
    consumption_path_exact = jnp.array(consumption_path_exact)

    print(f"\n  Steady state capital (analytical): kss = {kss:.4f}")
    print(f"  Final capital in simulation:       k_T = {float(capital_path[-1]):.4f}")
    print(f"  Optimal savings rate:              s   = alpha*beta = {alpha*beta:.2f}")

    valid_start = max(1, n_grid // 10)
    value_error = np.asarray(v_star - v_analytical)
    policy_error = np.asarray(policy_kprime_jnp - policy_kprime_analytical)
    consumption_error = np.asarray(consumption_policy - consumption_analytical)
    max_value_error = float(np.max(np.abs(value_error[valid_start:])))
    max_policy_error = float(np.max(np.abs(policy_error[valid_start:])))
    max_consumption_error = float(np.max(np.abs(consumption_error[valid_start:])))
    max_path_error = float(np.max(np.abs(np.asarray(capital_path - capital_path_exact))))

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Optimal Growth by Value Function Iteration",
        "Productive capital, Euler logic, and the transition to the Ramsey steady state.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Optimal growth adds one economic force that is absent from "
        "[cake eating](../cake-eating/): the state is productive. A planner who carries "
        "capital into tomorrow gives up current consumption, but that capital raises future "
        "output through a Cobb-Douglas technology. The policy problem is therefore not just "
        "resource depletion; it is intertemporal investment.\n\n"
        "This deterministic log-utility version is deliberately transparent. It has a closed "
        "form, so value function iteration can be judged against the true value function, "
        "policy function, and transition path. The same Bellman equation logic reappears in "
        "[RBC](../rbc/) and [Aiyagari](../aiyagari/) models once shocks and equilibrium prices "
        "are added."
    )

    report.add_equations(
        r"""
Let $k_t$ be capital at the start of period $t$. Output is

$$y_t = A k_t^\alpha, \qquad A>0,\quad \alpha \in (0,1).$$

The tutorial uses the full-depreciation resource constraint

$$c_t + k_{t+1} = A k_t^\alpha, \qquad c_t>0,\quad k_{t+1}\geq 0.$$

The planner maximizes discounted log utility,

$$\sum_{t=0}^{\infty} \beta^t \log c_t, \qquad \beta \in (0,1).$$

The Bellman equation is

$$V(k) = \max_{0 < k' < A k^\alpha}
\left[\log(Ak^\alpha-k')+\beta V(k')\right].$$

The policy function is $g(k)=k'$. For log utility and Cobb-Douglas production,
the exact solution is

$$g(k)=\alpha\beta A k^\alpha,\qquad
c^{*}(k)=(1-\alpha\beta)A k^\alpha.$$

The value function is affine in $\log k$:

$$V(k)=E+B\log k,\qquad B=\frac{\alpha}{1-\alpha\beta},$$

where

$$E=\frac{\log(A(1-\alpha\beta))
+\frac{\beta\alpha}{1-\alpha\beta}\log(A\alpha\beta)}{1-\beta}.$$

The steady state solves $k=g(k)$:

$$k_{ss}=(\alpha\beta A)^{1/(1-\alpha)}.$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\alpha$  | {alpha} | Capital share in $Ak^\\alpha$ |\n"
        f"| $A$       | {A} | Total factor productivity |\n"
        f"| $\\beta$   | {beta} | Discount factor |\n"
        f"| $k_{{ss}}$ | {kss:.4f} | Steady state capital |\n"
        f"| $c_{{ss}}$ | {A * kss ** alpha - kss:.4f} | Steady state consumption |\n"
        f"| Capital grid | {n_grid} points | Uniform grid for $k$ |\n"
        f"| Choice grid | {n_kprime} points | Candidate values for $k'$ in each Bellman update |\n"
        f"| $k \\in$   | [{k_min}, {k_max:.2f}] | Capital range |\n"
        f"| Tolerance | {tol:.0e} | Sup-norm convergence criterion |\n"
        f"| Simulation periods | {T_sim} | Transition-path horizon |"
    )

    report.add_solution_method(
        "The numerical solution approximates $V(k)$ on a grid. For each capital state, "
        "the solver searches over feasible next-period capital, interpolates the current "
        "continuation-value guess at off-grid choices, and applies the Bellman operator. "
        "The closed-form policy is not used to choose the maximizer; it is held out as a "
        "ground-truth diagnostic.\n\n"
        "```text\n"
        "Algorithm: grid VFI for deterministic optimal growth\n"
        "Input: capital grid K, primitives A, alpha, beta, utility u(c)=log c, tolerance epsilon\n"
        "Output: value function V and capital policy g(k)\n"
        "Initialize V_0(k_i) = log(A k_i^alpha) for each k_i in K\n"
        "repeat for n = 0, 1, 2, ...:\n"
        "    for each capital state k_i:\n"
        "        y_i = A k_i^alpha\n"
        "        build candidate choices k' in [k_min, min(y_i, k_max)]\n"
        "        c = y_i - k'\n"
        "        continuation = interpolate V_n at k'\n"
        "        choose k' that maximizes log(c) + beta * continuation\n"
        "        record V_{n+1}(k_i) and g(k_i)\n"
        "    error = max_i |V_{n+1}(k_i) - V_n(k_i)|\n"
        "until error < epsilon\n"
        "```\n\n"
        "The Bellman operator is a contraction under the usual bounded-state numerical "
        "approximation. Here it converged in "
        f"**{info['iterations']} iterations** with sup-norm error **{info['error']:.2e}**."
    )

    # --- Figure 1: Value Function ---
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, v_star, "b-", linewidth=2, label="Numerical (VFI)")
    ax1.plot(k_grid, v_analytical, "r--", linewidth=1.5, label="Exact")
    ax1.axvline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"$k_{{ss}} = {kss:.2f}$")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k)$")
    ax1.set_title("Value Function")
    ax1.legend()
    report.add_results(
        "The value function is increasing and concave because capital relaxes the resource "
        "constraint, but with diminishing marginal product. The exact log-linear value "
        "function gives a direct error check. Outside the bottom decile of the grid, the "
        f"largest value-function deviation is **{max_value_error:.2e}**."
    )
    report.add_figure(
        "figures/value-function.png",
        "Value function: numerical VFI vs exact log-Cobb-Douglas solution",
        fig1,
        description=(
            "The numerical and exact value functions are visually indistinguishable over "
            "most of the economically relevant state space. The vertical line marks the "
            "steady state, not a kink in preferences or technology."
        ),
    )

    # --- Figure 2: Policy Function ---
    fig2, ax2 = plt.subplots()
    ax2.plot(k_grid, policy_kprime_jnp, "b-", linewidth=2, label="Numerical $k'(k)$")
    ax2.plot(k_grid, policy_kprime_analytical, "r--", linewidth=1.5, label="Exact $\\alpha\\beta A k^\\alpha$")
    ax2.plot(k_grid, k_grid, "k:", linewidth=0.8, alpha=0.5, label="45-degree line")
    ax2.axvline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"$k_{{ss}}$")
    ax2.set_xlabel("Capital $k$")
    ax2.set_ylabel("Next-period capital $k'$")
    ax2.set_title("Capital Policy Function")
    ax2.legend()
    report.add_results(
        "The policy function is the main economic object. Below $k_{ss}$, the policy lies "
        "above the 45-degree line, so the economy accumulates capital. Above $k_{ss}$, it "
        "lies below the line, so capital is run down. The largest policy deviation from "
        f"the exact rule outside the bottom decile is **{max_policy_error:.2e}**; the "
        f"corresponding consumption-policy deviation is **{max_consumption_error:.2e}**."
    )
    report.add_figure(
        "figures/policy-function.png",
        "Capital policy function: numerical VFI vs exact savings rule",
        fig2,
        description=(
            "The crossing with the 45-degree line is the steady state. The exact policy is "
            "$g(k)=\\alpha\\beta A k^\\alpha$, so this calibration saves "
            f"**{alpha * beta:.1%}** of output each period."
        ),
    )

    # --- Figure 3: Simulation ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    periods = jnp.arange(T_sim)

    ax3a.plot(periods, capital_path, "b-o", markersize=3, linewidth=1.5, label="Numerical")
    ax3a.plot(periods, capital_path_exact, "r--", linewidth=1.5, label="Exact")
    ax3a.axhline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"$k_{{ss}} = {kss:.2f}$")
    ax3a.set_xlabel("Period")
    ax3a.set_ylabel("Capital $k_t$")
    ax3a.set_title("Capital Dynamics")
    ax3a.legend()

    ax3b.plot(periods[:-1], consumption_path[:-1], "b-o", markersize=3, linewidth=1.5, label="Numerical")
    ax3b.plot(periods[:-1], consumption_path_exact[:-1], "r--", linewidth=1.5, label="Exact")
    css = A * kss ** alpha - kss  # Steady state consumption
    ax3b.axhline(css, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"$c_{{ss}} = {css:.2f}$")
    ax3b.set_xlabel("Period")
    ax3b.set_ylabel("Consumption $c_t$")
    ax3b.set_title("Consumption Over Time")
    ax3b.legend()
    fig3.tight_layout()
    report.add_results(
        "The transition path starts from one tenth of steady-state capital. High marginal "
        "product makes investment attractive, so capital rises quickly and then approaches "
        "the fixed point more slowly. The largest numerical capital-path deviation from the "
        f"exact transition over the simulation is **{max_path_error:.2e}**."
    )
    report.add_figure(
        "figures/simulation.png",
        f"Transition path from k0={k0:.2f}",
        fig3,
        description=(
            "Capital and consumption both rise along this low-capital transition. The exact "
            "path makes clear that the visible dynamics are economic convergence, while the "
            "numerical gap is a grid-search approximation error."
        ),
    )

    # --- Table: Numerical vs Analytical ---
    sample_idx = np.linspace(valid_start, n_grid - 1, 8, dtype=int)
    table_data = {
        "k": [f"{float(k_grid[i]):.3f}" for i in sample_idx],
        "V(k) numerical": [f"{float(v_star[i]):.4f}" for i in sample_idx],
        "V(k) exact": [f"{float(v_analytical[i]):.4f}" for i in sample_idx],
        "V error": [f"{float(value_error[i]):.2e}" for i in sample_idx],
        "k' numerical": [f"{float(policy_kprime_jnp[i]):.4f}" for i in sample_idx],
        "k' exact": [f"{float(policy_kprime_analytical[i]):.4f}" for i in sample_idx],
        "k' error": [f"{float(policy_error[i]):.2e}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/comparison.csv",
        "Numerical vs exact solution at selected capital states",
        df,
        description=(
            "The table reports pointwise approximation errors. The value-function errors are "
            "small relative to the value level, and the policy errors are the relevant "
            "diagnostic because policies determine simulated allocations."
        ),
    )

    report.add_takeaway(
        "Optimal growth changes the cake-eating logic by making saving productive. The "
        "state still summarizes the future, but now carrying resources forward raises "
        "tomorrow's feasible set. With log utility and full-depreciation Cobb-Douglas "
        f"production, the exact rule saves $\\alpha\\beta={alpha * beta:.2f}$ of output and "
        f"drives capital toward $k_{{ss}}={kss:.2f}$. VFI recovers that policy closely, "
        "bridging closed-form dynamic programming to "
        "stochastic growth models where the benchmark has to be replaced by Euler errors, "
        "simulation moments, or equilibrium residuals."
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
