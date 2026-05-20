#!/usr/bin/env python3
"""Ramsey-Cass-Koopmans growth solved by finite-horizon shooting.

The state is aggregate capital. Initial capital is predetermined, while
initial consumption is the jump variable selected by the transversality
condition. The script regenerates figures and table.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Add repo root to path for lib/ imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def main() -> None:
    # Parameters
    alpha = 0.33      # Capital share
    delta = 0.05      # Depreciation rate
    rho = 0.03        # Continuous-time discount rate
    sigma = 2.0       # CRRA coefficient and inverse EIS
    A = 1.0           # Total factor productivity
    T = 150.0         # Finite terminal date used to approximate t -> infinity
    n_eval = 2000

    k_star = (alpha * A / (delta + rho)) ** (1.0 / (1.0 - alpha))
    c_star = A * k_star**alpha - delta * k_star

    def f(k: float | np.ndarray) -> float | np.ndarray:
        """Cobb-Douglas output."""
        return A * np.maximum(k, 1e-12) ** alpha

    def f_prime(k: float | np.ndarray) -> float | np.ndarray:
        """Marginal product of capital."""
        return alpha * A * np.maximum(k, 1e-12) ** (alpha - 1.0)

    def f_double_prime(k: float) -> float:
        """Second derivative of production."""
        return alpha * (alpha - 1.0) * A * k ** (alpha - 2.0)

    def ode_system(t: float, y: np.ndarray) -> list[float]:
        """Ramsey system in the state-control pair (k, c)."""
        k, c = y
        k_safe = max(k, 1e-10)
        c_safe = max(c, 1e-10)
        k_dot = f(k_safe) - delta * k_safe - c_safe
        c_dot = (f_prime(k_safe) - delta - rho) * c_safe / sigma
        return [float(k_dot), float(c_dot)]

    def depletion_event(t: float, y: np.ndarray) -> float:
        """Stop trial paths once capital or consumption leaves the feasible region."""
        return min(y[0], y[1]) - 1e-8

    depletion_event.terminal = True
    depletion_event.direction = -1

    def integrate_path(k0: float, c0: float, t_eval: np.ndarray | None = None):
        """Integrate the Ramsey ODE for a candidate initial consumption."""
        return solve_ivp(
            ode_system,
            [0.0, T],
            [k0, c0],
            method="RK45",
            t_eval=t_eval,
            max_step=0.5,
            rtol=1e-10,
            atol=1e-12,
            events=depletion_event,
        )

    def terminal_capital_gap(k0: float, c0: float) -> float:
        """Terminal capital minus target steady-state capital for shooting."""
        sol = integrate_path(k0, c0)
        return float(sol.y[0, -1] - k_star)

    def bracket_saddle_consumption(k0: float) -> tuple[float, float]:
        """Find a consumption bracket with opposite terminal capital signs."""
        c_low = 1e-8
        low_gap = terminal_capital_gap(k0, c_low)
        c_high = max(f(k0) - delta * k0 + 0.25 * c_star, 1.25 * c_star)
        high_gap = terminal_capital_gap(k0, c_high)

        while low_gap * high_gap > 0 and c_high < 25.0 * c_star:
            c_high *= 1.5
            high_gap = terminal_capital_gap(k0, c_high)

        if low_gap * high_gap > 0:
            raise RuntimeError(f"Could not bracket saddle-path consumption for k0={k0:.4f}")

        return c_low, c_high

    def find_saddle_path_c0(k0: float) -> float:
        """Shoot on c(0) so that k(T) matches the steady-state target."""
        c_low, c_high = bracket_saddle_consumption(k0)
        return float(
            brentq(
                lambda c0: terminal_capital_gap(k0, c0),
                c_low,
                c_high,
                xtol=1e-11,
                rtol=1e-11,
                maxiter=200,
            )
        )

    k0_multiples = np.array([0.25, 0.50, 0.75, 1.50, 2.00])
    k0_values = k0_multiples * k_star
    k0_labels = [f"{m:.2g} $k^{{*}}$" for m in k0_multiples]
    colors = ["#b23a48", "#2f6f9f", "#4b8f29", "#7b529d", "#d47f1f"]
    t_eval = np.linspace(0.0, T, n_eval)

    print(f"Steady state: k* = {k_star:.4f}, c* = {c_star:.4f}")
    solutions = []
    initial_consumption = []
    for k0, label in zip(k0_values, k0_labels, strict=True):
        print(f"  Shooting for k0 = {k0:.4f} ({label})")
        c0 = find_saddle_path_c0(float(k0))
        initial_consumption.append(c0)
        sol = integrate_path(float(k0), c0, t_eval=t_eval)
        solutions.append(sol)
        print(f"    c0 = {c0:.6f}, terminal k gap = {sol.y[0, -1] - k_star:.2e}")

    jacobian = np.array(
        [
            [f_prime(k_star) - delta, -1.0],
            [c_star * f_double_prime(k_star) / sigma, 0.0],
        ]
    )
    eigvals = np.linalg.eigvals(jacobian)
    lambda_stable = float(np.min(eigvals))
    ref_solution = solutions[0]
    ref_dev = np.abs(ref_solution.y[0] - k_star) / k_star

    setup_style()

    # Figure 1: phase diagram and selected saddle paths.
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    k_range = np.linspace(0.1, 2.35 * k_star, 350)
    c_kdot_zero = np.maximum(f(k_range) - delta * k_range, 0.0)

    ax1.plot(k_range, c_kdot_zero, "k--", linewidth=1.5, alpha=0.65, label="$\\dot{k}=0$")
    ax1.axvline(
        k_star,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.65,
        label="$\\dot{c}=0$",
    )

    for sol, color, label in zip(solutions, colors, k0_labels, strict=True):
        ax1.plot(sol.y[0], sol.y[1], color=color, linewidth=2.0, label=label)
        ax1.plot(sol.y[0, 0], sol.y[1, 0], "o", color=color, markersize=5)

    all_c = np.concatenate([sol.y[1] for sol in solutions])
    ax1.plot(k_star, c_star, "k*", markersize=12, zorder=5, label="steady state")
    ax1.set_xlabel("Capital $k$")
    ax1.set_ylabel("Consumption $c$")
    ax1.set_title("Shooting Selects the Ramsey Stable Path")
    ax1.set_xlim(0.0, 2.25 * k_star)
    ax1.set_ylim(0.0, 1.08 * max(np.max(all_c), c_star))
    ax1.legend(fontsize=9, loc="upper left")
    save_figure(fig1, "figures/phase-diagram.png", dpi=150)

    # Figure 2: transition paths.
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    for sol, color, label in zip(solutions, colors, k0_labels, strict=True):
        ax2a.plot(t_eval, sol.y[0], color=color, linewidth=1.8, label=label)
        ax2b.plot(t_eval, sol.y[1], color=color, linewidth=1.8, label=label)

    ax2a.axhline(k_star, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2a.set_xlabel("Time $t$")
    ax2a.set_ylabel("Capital $k(t)$")
    ax2a.set_title("Capital")
    ax2a.set_xlim(0, 100)
    ax2a.legend(fontsize=9)

    ax2b.axhline(c_star, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2b.set_xlabel("Time $t$")
    ax2b.set_ylabel("Consumption $c(t)$")
    ax2b.set_title("Consumption")
    ax2b.set_xlim(0, 100)
    ax2b.legend(fontsize=9)
    fig2.tight_layout()
    save_figure(fig2, "figures/time-paths.png", dpi=150)

    # Figure 3: convergence speed.
    fig3, ax3 = plt.subplots()
    for sol, color, label in zip(solutions, colors, k0_labels, strict=True):
        dev = np.abs(sol.y[0] - k_star) / k_star
        valid = dev > 1e-10
        ax3.semilogy(t_eval[valid], dev[valid], color=color, linewidth=1.8, label=label)

    anchor_t = 50.0
    anchor_idx = int(np.searchsorted(t_eval, anchor_t))
    t_theory = np.linspace(anchor_t, 120.0, 250)
    theory_line = ref_dev[anchor_idx] * np.exp(lambda_stable * (t_theory - t_eval[anchor_idx]))
    ax3.semilogy(
        t_theory,
        theory_line,
        "k--",
        linewidth=1.5,
        alpha=0.7,
        label=f"stable eigenvalue $\\lambda_s={lambda_stable:.3f}$",
    )
    ax3.set_xlabel("Time $t$")
    ax3.set_ylabel("$|k(t)-k^{*}|/k^{*}$")
    ax3.set_title("Local Convergence to the Steady State")
    ax3.set_xlim(0, 120)
    ax3.legend(fontsize=9)
    save_figure(fig3, "figures/convergence-speed.png", dpi=150)

    terminal_residuals = [abs(sol.y[0, -1] - k_star) / k_star for sol in solutions]
    table_data = {
        "$k_0/k^{*}$": [f"{m:.2f}" for m in k0_multiples],
        "$c_0$ from shooting": [f"{c0:.6f}" for c0 in initial_consumption],
        "$c_0/[f(k_0)-\\delta k_0]$": [
            f"{c0 / (f(k0) - delta * k0):.3f}"
            for k0, c0 in zip(k0_values, initial_consumption, strict=True)
        ],
        "$k(50)/k^{*}$": [
            f"{sol.y[0, np.searchsorted(t_eval, 50.0)] / k_star:.4f}" for sol in solutions
        ],
        "$c(50)/c^{*}$": [
            f"{sol.y[1, np.searchsorted(t_eval, 50.0)] / c_star:.4f}" for sol in solutions
        ],
        "Relative terminal capital gap": [
            f"{resid:.2e}" for resid in terminal_residuals
        ],
    }
    df = pd.DataFrame(table_data)
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/shooting-results.csv", index=False)

    save_thumbnail("figures/phase-diagram.png", "figures/thumb.png")
    print(f"Done: 3 figures, 1 table")


if __name__ == "__main__":
    main()
