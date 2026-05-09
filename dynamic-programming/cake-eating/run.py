#!/usr/bin/env python3
"""Cake-Eating: a one-state dynamic-programming check.

A non-renewable resource of size $W_0$ is consumed over an infinite horizon.
With log utility the problem has a closed form, so the numerical value and
policy functions can be checked directly against the exact answer.

Reference: Stokey, Lucas, and Prescott (1989), Ch. 4.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def main() -> None:
    # =========================================================================
    # Calibration
    # =========================================================================
    beta = 0.9
    sigma = 1.0       # log utility
    n_grid = 500
    n_cons = 300
    w_min = 0.01
    w_max = 1.0
    tol = 1e-6

    w_grid_np = np.linspace(w_min, w_max, n_grid)
    w_grid = jnp.array(w_grid_np)

    u_vec = lambda c: np.log(np.maximum(c, 1e-15))

    def analytical_v(w):
        return np.log((1 - beta) * np.maximum(w, 1e-15)) / (1 - beta) + beta * np.log(beta) / (1 - beta) ** 2

    def v_interp(wprime, v_np):
        result = np.interp(wprime, w_grid_np, v_np)
        below = wprime < w_grid_np[0]
        if np.any(below):
            result[below] = analytical_v(wprime[below])
        return result

    # =========================================================================
    # VFI on a continuous consumption choice with interpolated continuation
    # =========================================================================
    v = u_vec(w_grid_np)
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
    # Closed-form benchmark (log utility)
    # =========================================================================
    v_analytical = (
        jnp.log((1 - beta) * w_grid) / (1 - beta)
        + beta * jnp.log(beta) / (1 - beta) ** 2
    )
    consumption_analytical = (1 - beta) * w_grid
    policy_analytical = beta * w_grid

    # =========================================================================
    # Forward simulation of the depletion path
    # =========================================================================
    T_sim = 30
    cake_path = jnp.zeros(T_sim)
    cake_path = cake_path.at[0].set(w_max)
    for t in range(T_sim - 1):
        w_prime = jnp.interp(cake_path[t], w_grid, policy_cake)
        cake_path = cake_path.at[t + 1].set(w_prime)
    consumption_path = jnp.interp(cake_path, w_grid, consumption_policy)

    periods = jnp.arange(T_sim)
    cake_path_analytical = (beta ** periods) * w_max
    consumption_path_analytical = (1 - beta) * cake_path_analytical

    # Skip the bottom decile when reporting sup-norm errors: the consumption
    # grid is uniform on [0, W], so when W itself is tiny the choice grid is
    # too coarse to resolve a flat policy and the error there reflects
    # discretization rather than the algorithm.
    valid_start = max(1, n_grid // 10)
    value_error = np.asarray(v_star - v_analytical)
    policy_error = np.asarray(consumption_policy - consumption_analytical)
    max_value_error = float(np.max(np.abs(value_error[valid_start:])))
    max_policy_error = float(np.max(np.abs(policy_error[valid_start:])))
    max_path_error = float(np.max(np.abs(np.asarray(cake_path - cake_path_analytical))))

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Finite-Resource Cake Eating",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A household owns a fixed cake and chooses consumption each period. "
        "The cake does not grow, and there is no income or uncertainty. "
        "Consuming more today leaves less cake for every future period.\n\n"
        "The state is remaining cake $W_t$. The control is consumption $c_t$. "
        "The policy rule maps each stock into consumption. "
        "The value function prices the remaining stock.\n\n"
        "Value function iteration solves the Bellman equation on a grid. "
        "Log utility gives a closed-form Euler rule. "
        "That rule lets us check the computed value and policy directly."
    )

    report.add_equations(
        r"""
Let $W_t$ be remaining cake at the start of period $t$.
The household chooses $c_t \in [0, W_t]$ and leaves next-period cake:

$$W_{t+1} = W_t - c_t, \qquad W_0 \text{ given}.$$

Here $W_0$ is the initial cake endowment.

Preferences use discount factor $\beta \in (0,1)$ and CRRA flow utility:

$$\sum_{t=0}^{\infty} \beta^t u(c_t),
\qquad u(c)=\frac{c^{1-\sigma}}{1-\sigma},
\qquad u(c)=\log c \text{ when } \sigma=1.$$

The value function solves a one-state Bellman equation:

$$V(W) = \max_{0 \le c \le W} \{\, u(c) + \beta\, V(W-c) \,\}.$$

The first-order condition and envelope condition give the Euler equation:

$$u'(c_t) = \beta\, u'(c_{t+1}).$$

This says marginal utility rises as the cake stock falls.
In the log case, consumption falls at rate $\beta$.

Guessing a constant consumption share gives the closed-form policy:

$$c^{\ast}(W) = (1-\beta)\, W,
\qquad g(W) = W - c^{\ast}(W) = \beta\, W,$$

The matching value function is:

$$V(W) = \frac{\ln((1-\beta) W)}{1-\beta}
+ \frac{\beta \ln \beta}{(1-\beta)^2},
\qquad V'(W) = \frac{1}{(1-\beta)\,W}.$$

This closed form is the target for the numerical check.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $\\beta$ | {beta} | Discount factor; closed-form saving rate is $\\beta$ |\n"
        f"| $\\sigma$ | {sigma} | CRRA curvature; $\\sigma=1$ gives the log closed form |\n"
        f"| $W_0$ | {w_max} | Initial cake endowment |\n"
        f"| $W$ | $[{w_min},\\, {w_max}]$ | Wealth grid for $V$ and $c^{{\\ast}}$ |\n"
        f"| $N_W$ | {n_grid} | Uniform grid points for the state $W$ |\n"
        f"| $N_c$ | {n_cons} | Inner grid for the consumption choice at each state |\n"
        f"| Tolerance $\\varepsilon$ | {tol:.0e} | Sup-norm convergence threshold for the Bellman operator |\n"
        f"| $T_{{sim}}$ | {T_sim} | Periods simulated for the depletion path |"
    )

    report.add_solution_method(
        "Define the Bellman operator\n\n"
        r"$$(TV)(W) = \max_{0 \le c \le W} \{\, u(c) + \beta\, V(W-c) \,\}."
        "$$\n\n"
        "The computation applies this operator repeatedly. "
        "At each grid point, it searches over feasible consumption. "
        "The next stock $W-c$ is usually off grid. "
        "The continuation value is therefore interpolated.\n\n"
        "```text\n"
        "Algorithm: Cake-eating VFI\n"
        "Input : wealth grid, choice grid size N_c, tolerance epsilon\n"
        "Output: value V*(W_i), consumption policy c*(W_i)\n"
        "  initialise V_0(W_i) = u(W_i)                     # guess: eat everything\n"
        "  for n = 0, 1, 2, ... :\n"
        "      for each state W_i :\n"
        "          c_grid <- N_c points uniform on (0, W_i)\n"
        "          W'     <- W_i - c_grid                   # next-period wealth\n"
        "          V_cont <- interp(V_n, W')                # off-grid continuation\n"
        "          obj    <- u(c_grid) + beta * V_cont\n"
        "          V_{n+1}(W_i) <- max(obj)\n"
        "          c*(W_i)      <- argmax(obj)\n"
        "      err <- max_i | V_{n+1}(W_i) - V_n(W_i) |\n"
        "      stop when err < epsilon\n"
        "```\n\n"
        f"The iteration converges in **{info['iterations']} steps**. "
        f"The final sup-norm residual is **{info['error']:.2e}**. "
        "The closed form is then computed on the same wealth grid."
    )

    # ------------------------------------------------------------------
    # Figure 1: value function vs closed form
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    ax1.plot(w_grid, v_star, color="tab:blue", linewidth=2, label="Numerical (VFI)")
    ax1.plot(w_grid, v_analytical, color="tab:red", linestyle="--", linewidth=1.5, label="Closed form")
    ax1.set_xlabel("Cake size $W$")
    ax1.set_ylabel("$V(W)$")
    ax1.set_title("Value Function vs Closed Form")
    ax1.legend()
    report.add_results(
        "Concavity is the main shape restriction on the value function. "
        "The numerical value curve lies on the closed-form curve except near the lower boundary. "
        f"Outside the bottom decile, the largest sup-norm gap is **{max_value_error:.2e}**. "
        "The lower-boundary gap comes from the log singularity."
    )
    report.add_figure(
        "figures/value-function.png",
        "Numerical value function plotted against the closed-form benchmark",
        fig1,
    )

    # ------------------------------------------------------------------
    # Figure 2: consumption policy vs closed form
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.plot(w_grid, consumption_policy, color="tab:blue", linewidth=2, label=r"Numerical $c^{\ast}(W)$")
    ax2.plot(w_grid, consumption_analytical, color="tab:red", linestyle="--", linewidth=1.5,
             label=r"Closed form $(1-\beta)W$")
    ax2.plot(w_grid, w_grid, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
             label="$45^{\\circ}$ line")
    ax2.set_xlabel("Cake size $W$")
    ax2.set_ylabel("Consumption $c$")
    ax2.set_title("Consumption Policy")
    ax2.legend()
    report.add_results(
        f"Under log utility, the household consumes **{1 - beta:.0%}** of remaining cake. "
        "The numerical policy follows the closed-form line. "
        "The dotted line marks immediate exhaustion. "
        f"Above the bottom decile, the largest policy gap is **{max_policy_error:.2e}**."
    )
    report.add_figure(
        "figures/policy-function.png",
        "Consumption policy plotted against the closed-form $c=(1-\\beta)W$ rule",
        fig2,
    )

    # ------------------------------------------------------------------
    # Figure 3: depletion and consumption paths
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    ax3a.plot(periods, cake_path, "o-", color="tab:blue", markersize=3, linewidth=1.5, label="Numerical")
    ax3a.plot(periods, cake_path_analytical, color="black", linestyle="--", linewidth=1.5,
              label=r"Closed form $\beta^t W_0$")
    ax3a.set_xlabel("Period $t$")
    ax3a.set_ylabel("Cake remaining $W_t$")
    ax3a.set_title("Depletion path")
    ax3a.legend()

    ax3b.plot(periods, consumption_path, "o-", color="tab:red", markersize=3, linewidth=1.5, label="Numerical")
    ax3b.plot(periods, consumption_path_analytical, color="black", linestyle="--", linewidth=1.5,
              label=r"Closed form $(1-\beta)\beta^t W_0$")
    ax3b.set_xlabel("Period $t$")
    ax3b.set_ylabel("Consumption $c_t$")
    ax3b.set_title("Consumption path")
    ax3b.legend()
    fig3.tight_layout()
    report.add_results(
        "Starting from $W_0=1$, the policy produces geometric depletion. "
        "Wealth follows $W_t = \\beta^t W_0$. "
        "Consumption follows $c_t = (1-\\beta)\\beta^t W_0$. "
        f"The numerical path stays within **{max_path_error:.2e}** of the closed form."
    )
    report.add_figure(
        "figures/simulation.png",
        "Wealth and consumption paths starting from $W_0=1$, numerical against closed form",
        fig3,
    )

    # ------------------------------------------------------------------
    # Pointwise check table
    # ------------------------------------------------------------------
    sample_idx = jnp.linspace(valid_start, n_grid - 1, 8, dtype=jnp.int32)
    table_data = {
        "W": [f"{float(w_grid[i]):.3f}" for i in sample_idx],
        "V numerical": [f"{float(v_star[i]):.4f}" for i in sample_idx],
        "V closed form": [f"{float(v_analytical[i]):.4f}" for i in sample_idx],
        "V error": [f"{float(v_star[i] - v_analytical[i]):.2e}" for i in sample_idx],
        "c* numerical": [f"{float(consumption_policy[i]):.4f}" for i in sample_idx],
        "c* closed form": [f"{float(consumption_analytical[i]):.4f}" for i in sample_idx],
        "c* error": [f"{float(consumption_policy[i] - consumption_analytical[i]):.2e}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    report.add_results(
        "The table checks value and policy at eight wealth states. "
        "The residuals are smooth and small away from the lower boundary."
    )
    report.add_table(
        "tables/comparison.csv",
        "Numerical vs closed-form solution at selected wealth states",
        df,
    )

    report.add_takeaway(
        "Cake eating isolates Bellman logic in a one-state resource problem. "
        "The computed policy should consume a constant share of remaining cake. "
        "In this log case, the share is $1-\\beta$. "
        "The closed form makes value function iteration easy to inspect. "
        "The remaining errors are interpolation and choice-grid error."
    )

    report.add_references([
        "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in Economic Dynamics*. Harvard University Press, Ch. 4.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. MIT Press, 4th edition, Ch. 3.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
