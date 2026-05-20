#!/usr/bin/env python3
"""Stochastic RBC model on a global grid with endogenous labor.

Solves a representative-household RBC model by value function iteration.
The model uses capital and total factor productivity as states.
It simulates business-cycle moments from the nonlinear policy rules.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


def hp_filter(y, lam=1600):
    """Hodrick-Prescott filter. Returns trend and cyclical components."""
    T = len(y)
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    I = np.eye(T)
    trend = np.linalg.solve(I + lam * D.T @ D, y)
    cycle = y - trend
    return trend, cycle


def solve_vfi(k_grid, l_grid, z_vals, P, beta, delta, alpha, phi,
              tol=1e-5, max_iter=2000, label=""):
    """Run VFI over the joint (l, k') choice for the two-state RBC model.

    Returns the value function, capital and labor policy index arrays, and
    a small diagnostics dict. The whole flow-utility tensor is precomputed
    once; each iteration is a vectorized argmax over (l, k').
    """
    n_k = k_grid.size
    n_z = z_vals.size
    n_l = l_grid.size

    # Production: y = z * k^alpha * l^(1-alpha), shape (n_k, n_z, n_l)
    production = (
        z_vals[None, :, None]
        * k_grid[:, None, None] ** alpha
        * l_grid[None, None, :] ** (1.0 - alpha)
    )
    resources = production + (1.0 - delta) * k_grid[:, None, None]
    consumption = resources[:, :, :, None] - k_grid[None, None, None, :]

    log_leisure = np.log(1.0 - l_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        flow_utility = np.where(
            consumption > 0,
            np.log(np.maximum(consumption, 1e-300))
            + phi * log_leisure[None, None, :, None],
            -np.inf,
        )

    # Initialize V from a steady-state-like consumption rule.
    V = np.zeros((n_k, n_z))
    l_guess = 0.33
    for iz in range(n_z):
        for ik in range(n_k):
            y_guess = z_vals[iz] * k_grid[ik] ** alpha * l_guess ** (1.0 - alpha)
            c_guess = max(y_guess - delta * k_grid[ik], 0.01)
            V[ik, iz] = (np.log(c_guess) + phi * np.log(1.0 - l_guess)) / (1.0 - beta)

    policy_k = np.zeros((n_k, n_z), dtype=int)
    policy_l = np.zeros((n_k, n_z), dtype=int)

    for iteration in range(1, max_iter + 1):
        EV = V @ P.T
        total_value = flow_utility + beta * EV.T[None, :, None, :]
        total_flat = total_value.reshape(n_k, n_z, n_l * n_k)
        best_flat = np.argmax(total_flat, axis=2)
        policy_l = best_flat // n_k
        policy_k = best_flat % n_k
        V_new = np.max(total_flat, axis=2)
        error = np.max(np.abs(V_new - V))
        V = V_new
        if iteration % 100 == 0 and label:
            print(f"  [{label}] iter {iteration:4d}, error = {error:.2e}")
        if error < tol:
            break

    info = {"iterations": iteration, "converged": error < tol, "error": float(error)}
    return V, policy_k, policy_l, info


def main():
    # =========================================================================
    # Parameters (quarterly calibration)
    # =========================================================================
    beta = 0.99
    delta = 0.0233
    alpha = 1.0 / 3.0
    phi = 1.74

    z_vals = np.array([0.95, 1.05])
    P = np.array([[0.95, 0.05],
                  [0.05, 0.95]])

    # Coarse grid used for simulation; fine grid is benchmark only.
    n_k, n_l = 50, 50
    n_k_fine, n_l_fine = 200, 100
    k_min, k_max = 9.0, 12.0
    l_min, l_max = 0.2, 0.6

    k_grid = np.linspace(k_min, k_max, n_k)
    l_grid = np.linspace(l_min, l_max, n_l)
    k_grid_fine = np.linspace(k_min, k_max, n_k_fine)
    l_grid_fine = np.linspace(l_min, l_max, n_l_fine)

    tol = 1e-5

    # =========================================================================
    # Deterministic z=1 steady state (analytical reference point)
    # =========================================================================
    rental_rate_ss = 1.0 / beta - 1.0 + delta
    k_l_ratio = (rental_rate_ss / alpha) ** (1.0 / (alpha - 1.0))
    wage_ss = (1.0 - alpha) * k_l_ratio ** alpha
    cy_per_l_ss = k_l_ratio ** alpha - delta * k_l_ratio  # (y - delta k)/l
    l_ss = wage_ss / (wage_ss + phi * cy_per_l_ss)
    k_ss = k_l_ratio * l_ss
    y_ss = k_ss ** alpha * l_ss ** (1.0 - alpha)
    c_ss = y_ss - delta * k_ss
    i_ss = delta * k_ss
    print(f"Deterministic steady state: k={k_ss:.4f}, l={l_ss:.4f}, "
          f"c={c_ss:.4f}, i={i_ss:.4f}")

    # =========================================================================
    # Solve coarse grid (used for simulation) and fine grid (benchmark)
    # =========================================================================
    print("\nSolving coarse grid (50x50)...")
    V, policy_k_idx, policy_l_idx, info = solve_vfi(
        k_grid, l_grid, z_vals, P, beta, delta, alpha, phi,
        tol=tol, label="coarse",
    )
    print(f"  converged in {info['iterations']} iterations, "
          f"sup-norm = {info['error']:.2e}")

    print("\nSolving fine-grid benchmark (200x100)...")
    V_fine, pk_idx_fine, pl_idx_fine, info_fine = solve_vfi(
        k_grid_fine, l_grid_fine, z_vals, P, beta, delta, alpha, phi,
        tol=tol, label="fine",
    )
    print(f"  converged in {info_fine['iterations']} iterations, "
          f"sup-norm = {info_fine['error']:.2e}")

    # Policies in levels
    k_policy = k_grid[policy_k_idx]
    l_policy = l_grid[policy_l_idx]
    k_policy_fine = k_grid_fine[pk_idx_fine]
    l_policy_fine = l_grid_fine[pl_idx_fine]

    # Project the fine policies onto the coarse capital grid for direct
    # pointwise comparison (the fine grid contains the coarse one only
    # approximately; linear interpolation is the right operator here).
    k_policy_bench = np.column_stack([
        np.interp(k_grid, k_grid_fine, k_policy_fine[:, iz]) for iz in range(2)
    ])
    l_policy_bench = np.column_stack([
        np.interp(k_grid, k_grid_fine, l_policy_fine[:, iz]) for iz in range(2)
    ])
    V_bench = np.column_stack([
        np.interp(k_grid, k_grid_fine, V_fine[:, iz]) for iz in range(2)
    ])
    bench_k_max_abs = np.max(np.abs(k_policy - k_policy_bench))
    bench_l_max_abs = np.max(np.abs(l_policy - l_policy_bench))
    bench_V_rel = np.max(np.abs((V - V_bench) / np.abs(V_bench)))
    print(f"\nCoarse-vs-fine policy gap: max |dk'| = {bench_k_max_abs:.4f}, "
          f"max |dl| = {bench_l_max_abs:.4f}, max |dV/V| = {bench_V_rel:.2e}")

    # =========================================================================
    # Simulate the economy
    # =========================================================================
    T_sim = 5000
    T_burn = 500
    T_total = T_sim + T_burn
    np.random.seed(42)

    z_indices = np.zeros(T_total, dtype=int)
    z_indices[0] = 1
    for t in range(1, T_total):
        if np.random.rand() < P[z_indices[t - 1], z_indices[t - 1]]:
            z_indices[t] = z_indices[t - 1]
        else:
            z_indices[t] = 1 - z_indices[t - 1]

    k_sim = np.zeros(T_total)
    l_sim = np.zeros(T_total)
    y_sim = np.zeros(T_total)
    c_sim = np.zeros(T_total)
    i_sim = np.zeros(T_total)
    k_sim[0] = k_grid[n_k // 2]

    for t in range(T_total):
        iz = z_indices[t]
        ik = np.argmin(np.abs(k_grid - k_sim[t]))
        l_sim[t] = l_policy[ik, iz]
        k_next = k_policy[ik, iz]
        y_sim[t] = z_vals[iz] * k_sim[t] ** alpha * l_sim[t] ** (1.0 - alpha)
        i_sim[t] = k_next - (1.0 - delta) * k_sim[t]
        c_sim[t] = y_sim[t] - i_sim[t]
        if t < T_total - 1:
            k_sim[t + 1] = k_next

    # Discard burn-in
    k_sim, l_sim, y_sim, c_sim, i_sim = (a[T_burn:] for a in (k_sim, l_sim, y_sim, c_sim, i_sim))
    z_sim = z_vals[z_indices[T_burn:]]

    # =========================================================================
    # HP filter and business-cycle moments
    # =========================================================================
    log_y, log_c, log_i, log_k, log_l = (np.log(x) for x in (y_sim, c_sim, i_sim, k_sim, l_sim))
    _, y_cycle = hp_filter(log_y); y_cycle *= 100
    _, c_cycle = hp_filter(log_c); c_cycle *= 100
    _, i_cycle = hp_filter(log_i); i_cycle *= 100
    _, k_cycle = hp_filter(log_k); k_cycle *= 100
    _, l_cycle = hp_filter(log_l); l_cycle *= 100

    std_y, std_c, std_i, std_k, std_l = (np.std(x) for x in (y_cycle, c_cycle, i_cycle, k_cycle, l_cycle))
    corr_cy = np.corrcoef(c_cycle, y_cycle)[0, 1]
    corr_iy = np.corrcoef(i_cycle, y_cycle)[0, 1]
    corr_ky = np.corrcoef(k_cycle, y_cycle)[0, 1]
    corr_ly = np.corrcoef(l_cycle, y_cycle)[0, 1]
    rel_c, rel_i, rel_k, rel_l = std_c / std_y, std_i / std_y, std_k / std_y, std_l / std_y
    ac_y = np.corrcoef(y_cycle[1:], y_cycle[:-1])[0, 1]
    ac_c = np.corrcoef(c_cycle[1:], c_cycle[:-1])[0, 1]
    ac_i = np.corrcoef(i_cycle[1:], i_cycle[:-1])[0, 1]
    ac_k = np.corrcoef(k_cycle[1:], k_cycle[:-1])[0, 1]
    ac_l = np.corrcoef(l_cycle[1:], l_cycle[:-1])[0, 1]

    print(f"  std(Y) = {std_y:.2f}%")
    print(f"  std(C) = {std_c:.2f}% (rel {rel_c:.2f}), corr(C,Y) = {corr_cy:.2f}")
    print(f"  std(I) = {std_i:.2f}% (rel {rel_i:.2f}), corr(I,Y) = {corr_iy:.2f}")
    print(f"  std(L) = {std_l:.2f}% (rel {rel_l:.2f}), corr(L,Y) = {corr_ly:.2f}")
    print(f"  std(K) = {std_k:.2f}% (rel {rel_k:.2f}), corr(K,Y) = {corr_ky:.2f}")

    # =========================================================================
    # Figures
    # =========================================================================
    setup_style()

    # Figure 1: Value function with fine-grid benchmark
    fig1, ax1 = plt.subplots()
    ax1.plot(k_grid, V[:, 0], "b-", linewidth=2, label=f"$z_L = {z_vals[0]:.2f}$ (low)")
    ax1.plot(k_grid, V[:, 1], "r-", linewidth=2, label=f"$z_H = {z_vals[1]:.2f}$ (high)")
    ax1.plot(k_grid_fine, V_fine[:, 0], "b:", linewidth=1.0, alpha=0.7,
             label="fine grid benchmark")
    ax1.plot(k_grid_fine, V_fine[:, 1], "r:", linewidth=1.0, alpha=0.7)
    ax1.axvline(k_ss, color="k", linestyle=":", linewidth=1.0, alpha=0.5,
                label="$k_{ss}$ at $z=1$")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("$V(k, z)$")
    ax1.set_title("Value Function")
    ax1.legend(loc="lower right", fontsize=9)
    save_figure(fig1, "figures/value-function.png", dpi=150)

    # Figure 2: Capital and labor policies (two subplots) with benchmarks
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    ax2a.plot(k_grid, k_policy[:, 0], "b-", linewidth=2, label=f"$z_L = {z_vals[0]:.2f}$")
    ax2a.plot(k_grid, k_policy[:, 1], "r-", linewidth=2, label=f"$z_H = {z_vals[1]:.2f}$")
    ax2a.plot(k_grid_fine, k_policy_fine[:, 0], "b:", linewidth=1.0, alpha=0.7,
              label="fine grid")
    ax2a.plot(k_grid_fine, k_policy_fine[:, 1], "r:", linewidth=1.0, alpha=0.7)
    ax2a.plot(k_grid, k_grid, "k--", linewidth=0.8, alpha=0.5, label="$k'=k$")
    ax2a.axvline(k_ss, color="0.4", linestyle=":", linewidth=1.0, alpha=0.6,
                 label="$k_{ss}$ at $z=1$")
    ax2a.set_xlabel("Capital today $k$")
    ax2a.set_ylabel("Capital tomorrow $k'$")
    ax2a.set_title("Capital Policy $g_k(k,z)$")
    ax2a.legend(loc="upper left", fontsize=9)

    ax2b.plot(k_grid, l_policy[:, 0], "b-", linewidth=2, label=f"$z_L = {z_vals[0]:.2f}$")
    ax2b.plot(k_grid, l_policy[:, 1], "r-", linewidth=2, label=f"$z_H = {z_vals[1]:.2f}$")
    ax2b.plot(k_grid_fine, l_policy_fine[:, 0], "b:", linewidth=1.0, alpha=0.7,
              label="fine grid")
    ax2b.plot(k_grid_fine, l_policy_fine[:, 1], "r:", linewidth=1.0, alpha=0.7)
    ax2b.axhline(l_ss, color="0.4", linestyle=":", linewidth=1.0, alpha=0.6,
                 label="$l_{ss}$ at $z=1$")
    ax2b.set_xlabel("Capital today $k$")
    ax2b.set_ylabel("Hours $l$")
    ax2b.set_title("Labor Policy $g_l(k,z)$")
    ax2b.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    save_figure(fig2, "figures/policy-functions.png", dpi=150)

    # Figure 3: Simulated path
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    T_plot = 200
    periods = np.arange(T_plot)
    axes3[0].plot(periods, y_sim[:T_plot], "b-", linewidth=1, label="Output $y_t$")
    axes3[0].plot(periods, c_sim[:T_plot], "r-", linewidth=1, alpha=0.85,
                  label="Consumption $c_t$")
    axes3[0].plot(periods, i_sim[:T_plot], "g-", linewidth=1, alpha=0.85,
                  label="Investment $i_t$")
    axes3[0].set_ylabel("Level")
    axes3[0].set_title("Simulated allocation, first 200 quarters")
    axes3[0].legend(ncol=3, loc="upper right", fontsize=9)

    axes3[1].step(periods, z_sim[:T_plot], "k-", linewidth=1, where="mid")
    axes3[1].axhline(1.0, color="0.6", linestyle=":", linewidth=0.8)
    axes3[1].set_xlabel("Quarter")
    axes3[1].set_ylabel("TFP $z_t$")
    axes3[1].set_title("Productivity")
    fig3.tight_layout()
    save_figure(fig3, "figures/simulation.png", dpi=150)

    # Figure 4: HP-filtered comovements
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 8))
    T_cyc = 200

    axes4[0, 0].plot(np.arange(T_cyc), y_cycle[:T_cyc], "b-", linewidth=1, label="Output")
    axes4[0, 0].plot(np.arange(T_cyc), c_cycle[:T_cyc], "r-", linewidth=1,
                     alpha=0.85, label="Consumption")
    axes4[0, 0].axhline(0, color="0.7", linewidth=0.6)
    axes4[0, 0].set_title(f"Output vs Consumption  (corr = {corr_cy:.2f})")
    axes4[0, 0].set_ylabel("% deviation from HP trend")
    axes4[0, 0].legend(fontsize=9)

    axes4[0, 1].plot(np.arange(T_cyc), y_cycle[:T_cyc], "b-", linewidth=1, label="Output")
    axes4[0, 1].plot(np.arange(T_cyc), i_cycle[:T_cyc], "g-", linewidth=1,
                     alpha=0.85, label="Investment")
    axes4[0, 1].axhline(0, color="0.7", linewidth=0.6)
    axes4[0, 1].set_title(f"Output vs Investment  (corr = {corr_iy:.2f})")
    axes4[0, 1].set_ylabel("% deviation from HP trend")
    axes4[0, 1].legend(fontsize=9)

    axes4[1, 0].plot(np.arange(T_cyc), y_cycle[:T_cyc], "b-", linewidth=1, label="Output")
    axes4[1, 0].plot(np.arange(T_cyc), l_cycle[:T_cyc], "m-", linewidth=1,
                     alpha=0.85, label="Hours")
    axes4[1, 0].axhline(0, color="0.7", linewidth=0.6)
    axes4[1, 0].set_title(f"Output vs Hours  (corr = {corr_ly:.2f})")
    axes4[1, 0].set_xlabel("Quarter")
    axes4[1, 0].set_ylabel("% deviation from HP trend")
    axes4[1, 0].legend(fontsize=9)

    axes4[1, 1].plot(np.arange(T_cyc), y_cycle[:T_cyc], "b-", linewidth=1, label="Output")
    axes4[1, 1].plot(np.arange(T_cyc), k_cycle[:T_cyc], "c-", linewidth=1,
                     alpha=0.85, label="Capital")
    axes4[1, 1].axhline(0, color="0.7", linewidth=0.6)
    axes4[1, 1].set_title(f"Output vs Capital  (corr = {corr_ky:.2f})")
    axes4[1, 1].set_xlabel("Quarter")
    axes4[1, 1].set_ylabel("% deviation from HP trend")
    axes4[1, 1].legend(fontsize=9)
    fig4.tight_layout()
    save_figure(fig4, "figures/comovements.png", dpi=150)

    # =========================================================================
    # Tables
    # =========================================================================
    bc_data = {
        "Variable": ["Output (Y)", "Consumption (C)", "Investment (I)", "Hours (L)", "Capital (K)"],
        "Std Dev (%)": [f"{std_y:.2f}", f"{std_c:.2f}", f"{std_i:.2f}", f"{std_l:.2f}", f"{std_k:.2f}"],
        "Relative to Y": [f"{1.00:.2f}", f"{rel_c:.2f}", f"{rel_i:.2f}", f"{rel_l:.2f}", f"{rel_k:.2f}"],
        "Corr with Y": [f"{1.00:.2f}", f"{corr_cy:.2f}", f"{corr_iy:.2f}", f"{corr_ly:.2f}", f"{corr_ky:.2f}"],
        "Autocorr(1)": [f"{ac_y:.2f}", f"{ac_c:.2f}", f"{ac_i:.2f}", f"{ac_l:.2f}", f"{ac_k:.2f}"],
    }
    df = pd.DataFrame(bc_data)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/business-cycle-stats.csv", index=False)

    # Commit the fine-grid audit diagnostics so the convergence and
    # coarse-vs-fine gap numbers quoted in Solution Method are grounded in an
    # artifact, not only in the generated README.
    fine_grid_audit = pd.DataFrame(
        {
            "metric": [
                "bench_V_rel",
                "bench_k_max_abs",
                "bench_l_max_abs",
                "coarse_iterations",
                "coarse_error",
                "fine_iterations",
                "fine_error",
            ],
            "value": [
                float(bench_V_rel),
                float(bench_k_max_abs),
                float(bench_l_max_abs),
                info["iterations"],
                info["error"],
                info_fine["iterations"],
                info_fine["error"],
            ],
        }
    )
    fine_grid_audit.to_csv(
        Path(__file__).resolve().parent / "tables" / "fine-grid-audit.csv", index=False
    )

    save_thumbnail("figures/value-function.png", "figures/thumb.png")
    print(f"\nGenerated figures and tables.")


if __name__ == "__main__":
    main()
