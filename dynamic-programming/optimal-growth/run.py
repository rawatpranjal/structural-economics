#!/usr/bin/env python3
"""Neoclassical Optimal Growth (Ramsey-Cass-Koopmans): deterministic case.

Solves the infinite-horizon planner problem with Cobb-Douglas production, log
utility, and full depreciation. With this configuration the value and policy
functions admit a closed form, so VFI can be audited pointwise against the
exact Ramsey solution.

Reference: Stokey, Lucas, and Prescott (1989), Ch. 2 & 4.
"""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


def main() -> None:
    # =========================================================================
    # Calibration
    # =========================================================================
    alpha = 0.3       # Capital share in production
    A = 18.5          # Total factor productivity
    beta = 0.9        # Discount factor
    n_grid = 500      # Grid points for capital
    n_kprime = 500    # Inner grid for k'
    tol = 1e-6

    # Closed-form steady state under log + Cobb-Douglas + full depreciation
    kss = (alpha * beta * A) ** (1 / (1 - alpha))
    css = A * kss ** alpha - kss

    k_min = 0.01
    k_max = kss * 2.5

    # =========================================================================
    # Grid and primitives
    # =========================================================================
    k_grid_np = np.linspace(k_min, k_max, n_grid)

    def f_np(k):
        return A * k ** alpha

    def u_np(c):
        return np.log(np.maximum(c, 1e-15))

    # Closed form (log + Cobb-Douglas + full depreciation):
    #   g(k)   = alpha*beta*A*k^alpha          (saving rate = alpha*beta)
    #   c*(k)  = (1 - alpha*beta)*A*k^alpha
    #   V(k)   = E + B log k,    B = alpha/(1 - alpha*beta)
    B_const = alpha / (1 - alpha * beta)
    E_const = (1 / (1 - beta)) * (
        np.log(A * (1 - alpha * beta))
        + beta * alpha * np.log(A * alpha * beta) / (1 - alpha * beta)
    )

    def analytical_v(k):
        return E_const + B_const * np.log(np.maximum(k, 1e-15))

    def analytical_policy(k):
        return alpha * beta * A * np.maximum(k, 1e-15) ** alpha

    def v_interp(kprime, v_np):
        return np.interp(kprime, k_grid_np, v_np)

    # =========================================================================
    # Value Function Iteration
    # =========================================================================
    # For each capital state k, search over feasible next-period capital
    # k' in [k_min, min(A k^alpha, k_max)] and pick the maximizer of
    # log(A k^alpha - k') + beta * V(k'), where V is interpolated linearly
    # off the state grid.
    v = u_np(f_np(k_grid_np))  # initial guess: consume all output today

    for iteration in range(1, 1001):
        v_new = np.zeros(n_grid)
        policy_kprime = np.zeros(n_grid)

        for ik in range(n_grid):
            k = k_grid_np[ik]
            output = f_np(k)
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
    # Closed-form benchmark on the same grid
    # =========================================================================
    v_analytical = jnp.array(analytical_v(k_grid_np))
    policy_kprime_analytical = jnp.array(analytical_policy(k_grid_np))
    consumption_analytical = A * k_grid ** alpha - policy_kprime_analytical

    # =========================================================================
    # Forward-iterate the policy from a low-capital initial condition
    # =========================================================================
    T_sim = 50
    k0 = kss * 0.1
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
    consumption_path_exact = output_path_exact - np.concatenate(
        [capital_path_exact[1:], [np.nan]]
    )

    capital_path = jnp.array(capital_path)
    output_path = jnp.array(output_path)
    consumption_path = jnp.array(consumption_path)
    capital_path_exact = jnp.array(capital_path_exact)
    consumption_path_exact = jnp.array(consumption_path_exact)

    print(f"\n  Steady-state capital (closed form): k_ss = {kss:.4f}")
    print(f"  Final capital in simulation:        k_T  = {float(capital_path[-1]):.4f}")
    print(f"  Optimal saving rate:                s    = alpha*beta = {alpha*beta:.2f}")

    valid_start = max(1, n_grid // 10)
    value_error = np.asarray(v_star - v_analytical)
    policy_error = np.asarray(policy_kprime_jnp - policy_kprime_analytical)
    consumption_error = np.asarray(consumption_policy - consumption_analytical)
    max_value_error = float(np.max(np.abs(value_error[valid_start:])))
    max_policy_error = float(np.max(np.abs(policy_error[valid_start:])))
    max_consumption_error = float(np.max(np.abs(consumption_error[valid_start:])))
    max_path_error = float(np.max(np.abs(np.asarray(capital_path - capital_path_exact))))

    # =========================================================================
    # Figures
    # =========================================================================
    setup_style()

    # Figure 1: value function vs closed form
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, v_star, color="tab:blue", linewidth=2, label="Numerical (VFI)")
    ax1.plot(k_grid, v_analytical, color="tab:red", linestyle="--", linewidth=1.5, label="Closed form")
    ax1.axvline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                label=f"$k_{{ss}} = {kss:.2f}$")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k)$")
    ax1.set_title("Value Function vs Closed Form")
    ax1.legend()
    save_figure(fig1, "figures/value-function.png", dpi=150)

    # Figure 2: capital policy vs closed form
    fig2, ax2 = plt.subplots()
    ax2.plot(k_grid, policy_kprime_jnp, color="tab:blue", linewidth=2, label="Numerical $g(k)$")
    ax2.plot(k_grid, policy_kprime_analytical, color="tab:red", linestyle="--", linewidth=1.5,
             label=r"Closed form $\alpha\beta A k^{\alpha}$")
    ax2.plot(k_grid, k_grid, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
             label="$45^{\\circ}$ line")
    ax2.axvline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                label=f"$k_{{ss}}={kss:.2f}$")
    ax2.set_xlabel("Capital $k$")
    ax2.set_ylabel("Next-period capital $k'$")
    ax2.set_title("Capital Policy")
    ax2.legend()
    save_figure(fig2, "figures/policy-function.png", dpi=150)

    # Figure 3: transition paths
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    periods = jnp.arange(T_sim)

    ax3a.plot(periods, capital_path, "o-", color="tab:blue", markersize=3, linewidth=1.5,
              label="Numerical")
    ax3a.plot(periods, capital_path_exact, color="tab:red", linestyle="--", linewidth=1.5,
              label="Closed form")
    ax3a.axhline(kss, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                 label=f"$k_{{ss}}={kss:.2f}$")
    ax3a.set_xlabel("Period $t$")
    ax3a.set_ylabel("Capital $k_t$")
    ax3a.set_title("Capital transition")
    ax3a.legend()

    ax3b.plot(periods[:-1], consumption_path[:-1], "o-", color="tab:blue", markersize=3,
              linewidth=1.5, label="Numerical")
    ax3b.plot(periods[:-1], consumption_path_exact[:-1], color="tab:red", linestyle="--",
              linewidth=1.5, label="Closed form")
    ax3b.axhline(css, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                 label=f"$c_{{ss}}={css:.2f}$")
    ax3b.set_xlabel("Period $t$")
    ax3b.set_ylabel("Consumption $c_t$")
    ax3b.set_title("Consumption transition")
    ax3b.legend()
    fig3.tight_layout()
    save_figure(fig3, "figures/simulation.png", dpi=150)

    # =========================================================================
    # Tables
    # =========================================================================
    sample_idx = np.linspace(valid_start, n_grid - 1, 8, dtype=int)
    table_data = {
        "k": [f"{float(k_grid[i]):.3f}" for i in sample_idx],
        "V numerical": [f"{float(v_star[i]):.4f}" for i in sample_idx],
        "V closed form": [f"{float(v_analytical[i]):.4f}" for i in sample_idx],
        "V error": [f"{float(value_error[i]):.2e}" for i in sample_idx],
        "k' numerical": [f"{float(policy_kprime_jnp[i]):.4f}" for i in sample_idx],
        "k' closed form": [f"{float(policy_kprime_analytical[i]):.4f}" for i in sample_idx],
        "k' error": [f"{float(policy_error[i]):.2e}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/comparison.csv", index=False)

    # Committed audit artifacts so the "max ... outside bottom decile"
    # and convergence claims can be verified without re-running.
    full_errors = pd.DataFrame(
        {
            "k": k_grid_np,
            "V error": value_error,
            "k' error": policy_error,
            "c error": consumption_error,
        }
    )
    full_errors.to_csv(
        Path(__file__).resolve().parent / "tables" / "full-errors.csv", index=False
    )
    convergence_log = pd.DataFrame(
        {
            "metric": [
                "vfi_iterations",
                "vfi_sup_norm_error",
                "max_value_error_above_bottom_decile",
                "max_policy_error_above_bottom_decile",
                "max_capital_path_error",
            ],
            "value": [
                info["iterations"],
                float(info["error"]),
                max_value_error,
                max_policy_error,
                max_path_error,
            ],
        }
    )
    convergence_log.to_csv(
        Path(__file__).resolve().parent / "tables" / "convergence-log.csv", index=False
    )

    save_thumbnail("figures/value-function.png", "figures/thumb.png")
    print(f"\nGenerated figures and tables.")


if __name__ == "__main__":
    main()
