#!/usr/bin/env python3
"""Ramsey consumption choice and saddle paths.

The tutorial studies the continuous-time Ramsey-Cass-Koopmans planner in the
(k, c) plane. It plots nullclines, the local linearization, and a nonlinear
stable arm from backward ODE integration. The stable arm selects initial
consumption for a given capital stock.

Reference: Barro and Sala-i-Martin (2004), "Economic Growth," Ch. 2.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def main():
    alpha = 0.3
    delta = 0.05
    rho = 0.04
    sigma = 2.0
    A = 1.0

    def f(k):
        return A * k ** alpha

    def f_prime(k):
        return alpha * A * k ** (alpha - 1)

    def f_second(k):
        return alpha * (alpha - 1) * A * k ** (alpha - 2)

    def dynamics(t, y):
        k, c = y
        k_safe = max(k, 1e-10)
        c_safe = max(c, 1e-10)
        dk = f(k_safe) - delta * k_safe - c_safe
        dc = (f_prime(k_safe) - delta - rho) * c_safe / sigma
        return np.array([dk, dc])

    k_ss = ((alpha * A) / (rho + delta)) ** (1 / (1 - alpha))
    c_ss = f(k_ss) - delta * k_ss

    jacobian = np.array([
        [f_prime(k_ss) - delta, -1.0],
        [f_second(k_ss) * c_ss / sigma, 0.0],
    ])
    eigvals, eigvecs = np.linalg.eig(jacobian)
    stable_idx = int(np.argmin(eigvals.real))
    unstable_idx = 1 - stable_idx
    lambda_stable = float(eigvals[stable_idx].real)
    lambda_unstable = float(eigvals[unstable_idx].real)
    stable_vec = eigvecs[:, stable_idx].real
    if stable_vec[0] < 0:
        stable_vec = -stable_vec
    slope = float(stable_vec[1] / stable_vec[0])

    print(f"Steady state: k* = {k_ss:.4f}, c* = {c_ss:.4f}")
    print(
        "Eigenvalues: "
        f"{lambda_stable:.4f}, {lambda_unstable:.4f} "
        "(saddle point: one negative, one positive)"
    )

    k_min = 0.08
    c_min = 0.02
    k_max = k_ss * 2.45
    c_max = c_ss * 2.25

    def make_stop_events():
        def low_k(t, y):
            return y[0] - k_min

        def low_c(t, y):
            return y[1] - c_min

        def high_k(t, y):
            return k_max - y[0]

        def high_c(t, y):
            return c_max - y[1]

        for event in (low_k, low_c, high_k, high_c):
            event.terminal = True
            event.direction = -1
        return [low_k, low_c, high_k, high_c]

    def trace_stable_branch(sign):
        eps = 1e-3 * k_ss
        y0 = np.array([k_ss + sign * eps, c_ss + sign * slope * eps])
        sol = solve_ivp(
            lambda t, y: -dynamics(t, y),
            (0.0, 160.0),
            y0,
            max_step=0.2,
            rtol=1e-9,
            atol=1e-11,
            events=make_stop_events(),
        )
        valid = (
            (sol.y[0] > k_min)
            & (sol.y[0] < k_max)
            & (sol.y[1] > c_min)
            & (sol.y[1] < c_max)
        )
        return sol.y[0][valid], sol.y[1][valid]

    k_left, c_left = trace_stable_branch(-1.0)
    k_right, c_right = trace_stable_branch(1.0)
    k_stable = np.concatenate([k_left[::-1], np.array([k_ss]), k_right])
    c_stable = np.concatenate([c_left[::-1], np.array([c_ss]), c_right])
    order = np.argsort(k_stable)
    k_stable = k_stable[order]
    c_stable = c_stable[order]

    k_range = np.linspace(k_min, k_max, 400)
    c_nullcline = f(k_range) - delta * k_range
    c_linear = c_ss + slope * (k_range - k_ss)
    c_linear_plot = np.where(c_linear > 0, c_linear, np.nan)

    k_grid = np.linspace(0.35, k_ss * 2.25, 22)
    c_grid = np.linspace(0.08, c_ss * 2.05, 22)
    K, C = np.meshgrid(k_grid, c_grid)
    DK = f(K) - delta * K - C
    DC = (f_prime(K) - delta - rho) * C / sigma
    speed = np.sqrt(DK ** 2 + DC ** 2)
    DK_norm = DK / (speed + 1e-12)
    DC_norm = DC / (speed + 1e-12)

    left_indices = np.where(k_stable < k_ss)[0]
    start_idx = left_indices[np.argmin(np.abs(k_stable[left_indices] - 0.4 * k_ss))]
    k0_path = float(k_stable[start_idx])
    c0_path = float(c_stable[start_idx])

    def integrate_forward(y0, horizon=65.0):
        sol = solve_ivp(
            dynamics,
            (0.0, horizon),
            y0,
            max_step=0.1,
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
            events=make_stop_events(),
        )
        t_path = np.linspace(0.0, sol.t[-1], 300)
        y_path = sol.sol(t_path)
        valid = (
            (y_path[0] > k_min)
            & (y_path[0] < k_max)
            & (y_path[1] > c_min)
            & (y_path[1] < c_max)
        )
        return y_path[0][valid], y_path[1][valid]

    off_path_gap = 0.20 * c_ss
    selection_paths = [
        ("selected path", [k0_path, c0_path], "black"),
        ("higher $c_0$", [k0_path, c0_path + off_path_gap], "#b23a48"),
        ("lower $c_0$", [k0_path, max(c_min * 2, c0_path - off_path_gap)], "#246b8e"),
    ]
    traced_selection = [
        (label, *integrate_forward(y0), color)
        for label, y0, color in selection_paths
    ]

    setup_style()

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.quiver(
        K,
        C,
        DK_norm,
        DC_norm,
        speed,
        cmap="viridis",
        alpha=0.45,
        scale=32,
        width=0.003,
    )
    ax1.plot(k_range, c_nullcline, color="#1f77b4", linewidth=2.5, label="$\\dot{k}=0$")
    ax1.axvline(k_ss, color="#c44e52", linewidth=2.5, label="$\\dot{c}=0$")
    ax1.plot(k_range, c_linear_plot, color="0.35", linestyle="--", linewidth=2.0,
             label="local linear arm")
    ax1.plot(k_stable, c_stable, color="black", linewidth=3.0, label="nonlinear stable arm")
    ax1.plot(k_ss, c_ss, "ko", markersize=12, zorder=5)
    ax1.annotate(
        f"$(k^{{*}},c^{{*}})=({k_ss:.2f},{c_ss:.2f})$",
        (k_ss, c_ss),
        textcoords="offset points",
        xytext=(12, -24),
        fontsize=10,
    )
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("Consumption $c$")
    ax1.set_title("Ramsey Phase Plane")
    ax1.set_xlim(0, k_max)
    ax1.set_ylim(0, c_max)
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    save_figure(fig1, "figures/phase-diagram.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(9, 6.5))
    ax3.quiver(
        K,
        C,
        DK_norm,
        DC_norm,
        speed,
        cmap="viridis",
        alpha=0.35,
        scale=32,
        width=0.003,
    )
    ax3.plot(k_range, c_nullcline, color="#1f77b4", linestyle="--", linewidth=1.8,
             label="$\\dot{k}=0$")
    ax3.axvline(k_ss, color="#c44e52", linestyle="--", linewidth=1.8,
                label="$\\dot{c}=0$")
    ax3.plot(k_stable, c_stable, color="0.55", linewidth=2.2, label="stable arm")
    for label, k_path, c_path, color in traced_selection:
        ax3.plot(k_path, c_path, color=color, linewidth=2.3, label=label)
        ax3.plot(k_path[0], c_path[0], marker="o", color=color, markersize=7)
    ax3.plot(k_ss, c_ss, "ko", markersize=9)
    ax3.set_xlabel("Capital $k$")
    ax3.set_ylabel("Consumption $c$")
    ax3.set_title("Initial Consumption Selects the Ramsey Path")
    ax3.set_xlim(0, k_max)
    ax3.set_ylim(0, c_max)
    ax3.legend(loc="upper right")
    fig3.tight_layout()
    save_figure(fig3, "figures/path-selection.png", dpi=150)

    save_thumbnail("figures/phase-diagram.png", "figures/thumb.png")
    print("Done: 2 figures")


if __name__ == "__main__":
    main()
