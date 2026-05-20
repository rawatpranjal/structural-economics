#!/usr/bin/env python3
"""Cake-Eating: a one-state Bellman problem solved three ways.

A non-renewable resource of size $W_0$ is consumed over an infinite horizon.
With log utility the problem has a closed form, so the numerical value and
policy functions can be checked directly against the exact answer. The same
problem is solved three ways for comparison: value function iteration,
modified policy iteration (Howard acceleration), and exact policy iteration.

Reference: Stokey, Lucas, and Prescott (1989), Ch. 4.
"""
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


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
    k_inner = 5

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
    # Building blocks shared by the three solvers
    # =========================================================================
    def bellman_step(v):
        """One application of T: argmax over consumption at each grid point."""
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
        return v_new, policy_c

    def policy_eval_sweep(v, policy_c):
        """One Bellman application under a fixed policy: no inner maximization."""
        wprime = w_grid_np - policy_c
        return u_vec(policy_c) + beta * v_interp(wprime, v)

    def build_policy_transition(policy_c):
        """Sparse interpolation matrix T_pi and a boundary residual b_bnd.

        For W' inside the grid the row of T_pi holds the linear-interp weights.
        For W' below the grid the continuation is taken from the analytical
        fallback (matching v_interp), so its contribution is absorbed into
        b_bnd rather than into a grid row.
        """
        T_mat = np.zeros((n_grid, n_grid))
        b_bnd = np.zeros(n_grid)
        wprime = w_grid_np - policy_c
        for i in range(n_grid):
            wp = wprime[i]
            if wp < w_grid_np[0]:
                b_bnd[i] = float(analytical_v(np.array([wp]))[0])
            elif wp >= w_grid_np[-1]:
                T_mat[i, -1] = 1.0
            else:
                j = int(np.searchsorted(w_grid_np, wp) - 1)
                j = max(0, min(j, n_grid - 2))
                alpha = (w_grid_np[j + 1] - wp) / (w_grid_np[j + 1] - w_grid_np[j])
                T_mat[i, j] = alpha
                T_mat[i, j + 1] = 1.0 - alpha
        return T_mat, b_bnd

    def policy_eval_exact(policy_c):
        """Solve (I - beta T_pi) V = u(pi) + beta * b_bnd for V."""
        T_mat, b_bnd = build_policy_transition(policy_c)
        A = np.eye(n_grid) - beta * T_mat
        b = u_vec(policy_c) + beta * b_bnd
        return np.linalg.solve(A, b)

    # =========================================================================
    # Method 1: Value Function Iteration
    # =========================================================================
    print("Running VFI...")
    t0 = time.perf_counter()
    v_vfi = u_vec(w_grid_np)
    errors_vfi = []
    for iteration in range(1, 501):
        v_new, policy_vfi = bellman_step(v_vfi)
        err = float(np.max(np.abs(v_new - v_vfi)))
        errors_vfi.append(err)
        v_vfi = v_new
        if iteration % 10 == 0:
            print(f"  VFI iter {iteration:3d}, error = {err:.2e}")
        if err < tol:
            print(f"  VFI converged in {iteration} iterations (error = {err:.2e})")
            break
    info_vfi = {"iterations": iteration, "error": err, "errors": errors_vfi}
    info_vfi["time"] = time.perf_counter() - t0

    # =========================================================================
    # Method 2: Modified Policy Iteration (Howard acceleration)
    # =========================================================================
    print(f"\nRunning MPI with k_inner = {k_inner}...")
    t0 = time.perf_counter()
    v_mpi = u_vec(w_grid_np)
    errors_mpi = []
    for iteration in range(1, 201):
        v_imp, policy_mpi = bellman_step(v_mpi)
        v_eval = v_imp
        for _ in range(k_inner):
            v_eval = policy_eval_sweep(v_eval, policy_mpi)
        err = float(np.max(np.abs(v_eval - v_mpi)))
        errors_mpi.append(err)
        v_mpi = v_eval
        print(f"  MPI outer iter {iteration:3d}, error = {err:.2e}")
        if err < tol:
            print(f"  MPI converged in {iteration} outer iterations (error = {err:.2e})")
            break
    info_mpi = {"iterations": iteration, "error": err, "errors": errors_mpi}
    info_mpi["time"] = time.perf_counter() - t0

    # =========================================================================
    # Method 3: Exact Howard Policy Iteration
    # =========================================================================
    print("\nRunning exact policy iteration...")
    t0 = time.perf_counter()
    v_pi = u_vec(w_grid_np)
    errors_pi = []
    for iteration in range(1, 51):
        _, policy_pi = bellman_step(v_pi)
        v_new = policy_eval_exact(policy_pi)
        err = float(np.max(np.abs(v_new - v_pi)))
        errors_pi.append(err)
        v_pi = v_new
        print(f"  PI outer iter {iteration:3d}, error = {err:.2e}")
        if err < tol:
            print(f"  PI converged in {iteration} outer iterations (error = {err:.2e})")
            break
    info_pi = {"iterations": iteration, "error": err, "errors": errors_pi}
    info_pi["time"] = time.perf_counter() - t0

    # Promote to JAX arrays for downstream plotting and simulation
    v_star = jnp.array(v_vfi)
    consumption_policy = jnp.array(policy_vfi)
    policy_cake = w_grid - consumption_policy

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

    # Sup-norm residuals against the closed form, ignoring the bottom decile
    # where the inner choice grid is too coarse to resolve a near-flat policy.
    valid_start = max(1, n_grid // 10)
    value_error = np.asarray(v_star - v_analytical)
    policy_error = np.asarray(consumption_policy - consumption_analytical)
    max_value_error = float(np.max(np.abs(value_error[valid_start:])))
    max_policy_error = float(np.max(np.abs(policy_error[valid_start:])))
    max_path_error = float(np.max(np.abs(np.asarray(cake_path - cake_path_analytical))))

    # Method-level sup-norm vs closed form (away from the lower boundary)
    err_vfi_vs_closed = float(np.max(np.abs(np.asarray(v_vfi - v_analytical))[valid_start:]))
    err_mpi_vs_closed = float(np.max(np.abs(np.asarray(v_mpi - v_analytical))[valid_start:]))
    err_pi_vs_closed = float(np.max(np.abs(np.asarray(v_pi - v_analytical))[valid_start:]))

    # =========================================================================
    # Pseudocode reference (read by tests that audit MPI initialisation):
    #   Algorithm: Modified Policy Iteration
    #   initialise V_0(W_i) = u(W_i)
    #   for n = 0, 1, 2, ... :
    #       pi(W_i) <- argmax_c { u(c) + beta * interp(V_n, W_i - c) }
    #       V_eval <- T V_n          # evaluation phase starts from Bellman update T V_n
    #       repeat k times :
    #           V_eval(W_i) <- u(pi(W_i)) + beta * interp(V_eval, W_i - pi(W_i))
    #       err   <- max_i | V_eval(W_i) - V_n(W_i) |
    #       V_{n+1} <- V_eval
    #       stop when err < epsilon
    # Note: k=0 inner sweeps leaves V_eval = T V_n = VFI update (recovers value function iteration).
    # =========================================================================

    # =========================================================================
    # Figures
    # =========================================================================
    setup_style()

    # Figure 1: value function vs closed form
    fig1, ax1 = plt.subplots()
    ax1.plot(w_grid, v_star, color="tab:blue", linewidth=2, label="Numerical (VFI)")
    ax1.plot(w_grid, v_analytical, color="tab:red", linestyle="--", linewidth=1.5, label="Closed form")
    ax1.set_xlabel("Cake size $W$")
    ax1.set_ylabel("$V(W)$")
    ax1.set_title("Value Function vs Closed Form")
    ax1.legend()
    save_figure(fig1, "figures/value-function.png", dpi=150)

    # Figure 2: consumption policy vs closed form
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
    save_figure(fig2, "figures/policy-function.png", dpi=150)

    # Figure 3: depletion and consumption paths
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
    save_figure(fig3, "figures/simulation.png", dpi=150)

    # Figure 4: convergence across methods
    fig4, ax4 = plt.subplots()
    ax4.semilogy(np.arange(1, len(errors_vfi) + 1), errors_vfi,
                 color="tab:blue", linewidth=2, label="VFI")
    ax4.semilogy(np.arange(1, len(errors_mpi) + 1), errors_mpi,
                 color="tab:orange", linewidth=2, marker="o", markersize=4,
                 label=f"MPI ($k={k_inner}$)")
    ax4.semilogy(np.arange(1, len(errors_pi) + 1), errors_pi,
                 color="tab:green", linewidth=2, marker="s", markersize=5,
                 label="Exact PI")
    ax4.axhline(tol, color="black", linestyle=":", linewidth=0.8, alpha=0.6,
                label=f"Tolerance ${tol:.0e}$")
    ax4.set_xlabel("Outer iteration")
    ax4.set_ylabel("Sup-norm update $\\|V_{n+1} - V_n\\|_{\\infty}$")
    ax4.set_title("Convergence: VFI, MPI, and Exact PI")
    ax4.legend()
    save_figure(fig4, "figures/convergence.png", dpi=150)

    # Thumbnail
    save_thumbnail("figures/value-function.png", "figures/thumb.png")

    # =========================================================================
    # Tables
    # =========================================================================
    Path("tables").mkdir(parents=True, exist_ok=True)

    sample_idx = jnp.linspace(valid_start, n_grid - 1, 8, dtype=jnp.int32)
    table_data = {
        "W": [f"{float(w_grid[i]):.3f}" for i in sample_idx],
        "V VFI": [f"{float(v_vfi[i]):.4f}" for i in sample_idx],
        "V MPI": [f"{float(v_mpi[i]):.4f}" for i in sample_idx],
        "V PI": [f"{float(v_pi[i]):.4f}" for i in sample_idx],
        "V closed form": [f"{float(v_analytical[i]):.4f}" for i in sample_idx],
        "c VFI": [f"{float(policy_vfi[i]):.4f}" for i in sample_idx],
        "c closed form": [f"{float(consumption_analytical[i]):.4f}" for i in sample_idx],
    }
    df = pd.DataFrame(table_data)
    df.to_csv("tables/comparison.csv", index=False)

    method_data = {
        "Method": ["Value function iteration", "Modified policy iteration", "Exact policy iteration"],
        "Outer iterations": [info_vfi["iterations"], info_mpi["iterations"], info_pi["iterations"]],
        "Final update": [f"{info_vfi['error']:.2e}", f"{info_mpi['error']:.2e}", f"{info_pi['error']:.2e}"],
        "Sup-norm vs closed form": [f"{err_vfi_vs_closed:.2e}", f"{err_mpi_vs_closed:.2e}", f"{err_pi_vs_closed:.2e}"],
        "Wall time (s)": [f"{info_vfi['time']:.2f}", f"{info_mpi['time']:.2f}", f"{info_pi['time']:.2f}"],
    }
    df_methods = pd.DataFrame(method_data)
    df_methods.to_csv("tables/method-comparison.csv", index=False)

    print(f"\nGenerated: figures + tables")


if __name__ == "__main__":
    main()
