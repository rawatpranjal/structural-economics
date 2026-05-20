#!/usr/bin/env python3
"""Solow growth in effective-labor units.

The deterministic Solow transition is iterated as a one-dimensional map for
capital per effective worker. The closed-form steady state and a local
linearization around it serve as ground truth for the simulation.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def output_per_effective_worker(k: np.ndarray | float, alpha: float) -> np.ndarray | float:
    """Cobb-Douglas output per effective worker, f(k)=k^alpha."""
    return np.asarray(k) ** alpha


def solow_next_k(
    k: float,
    alpha: float,
    savings_rate: float,
    depreciation: float,
    population_growth: float,
    technology_growth: float,
) -> float:
    """One-period transition for capital per effective worker."""
    dilution = (1.0 + technology_growth) * (1.0 + population_growth)
    return ((1.0 - depreciation) * k + savings_rate * k**alpha) / dilution


def simulate_solow_path(
    k0: float,
    periods: int,
    alpha: float,
    savings_rate: float,
    depreciation: float,
    population_growth: float,
    technology_growth: float,
) -> pd.DataFrame:
    """Simulate the Solow transition in effective-labor units."""
    k_path = np.empty(periods)
    y_path = np.empty(periods)
    c_path = np.empty(periods)
    investment_path = np.empty(periods)

    k_path[0] = k0
    for t in range(periods):
        y_path[t] = output_per_effective_worker(k_path[t], alpha)
        c_path[t] = (1.0 - savings_rate) * y_path[t]
        investment_path[t] = savings_rate * y_path[t]
        if t < periods - 1:
            k_path[t + 1] = solow_next_k(
                k_path[t],
                alpha,
                savings_rate,
                depreciation,
                population_growth,
                technology_growth,
            )

    return pd.DataFrame(
        {
            "period": np.arange(periods),
            "k": k_path,
            "y": y_path,
            "c": c_path,
            "investment": investment_path,
        }
    )


def steady_state(
    alpha: float,
    savings_rate: float,
    depreciation: float,
    population_growth: float,
    technology_growth: float,
) -> tuple[float, float, float]:
    """Closed-form k*, break-even Delta, and gross dilution (1+g)(1+n)."""
    gross_dilution = (1.0 + technology_growth) * (1.0 + population_growth)
    effective_depreciation = gross_dilution - 1.0 + depreciation
    k_star = (savings_rate / effective_depreciation) ** (1.0 / (1.0 - alpha))
    return k_star, effective_depreciation, gross_dilution


def main() -> None:
    # Annual calibration in line with standard textbook Solow numbers.
    K0 = 1.0
    A0 = 1.0
    L0 = 1.0
    periods = 160
    alpha = 0.33
    savings_rate = 0.24
    depreciation = 0.06
    population_growth = 0.01
    technology_growth = 0.02

    k0 = K0 / (A0 * L0)
    k_star, effective_depreciation, gross_dilution = steady_state(
        alpha, savings_rate, depreciation, population_growth, technology_growth
    )
    y_star = output_per_effective_worker(k_star, alpha)
    c_star = (1.0 - savings_rate) * y_star

    path = simulate_solow_path(
        k0,
        periods,
        alpha,
        savings_rate,
        depreciation,
        population_growth,
        technology_growth,
    )
    terminal = path.iloc[-1]
    local_lambda = (
        (1.0 - depreciation) + savings_rate * alpha * k_star ** (alpha - 1.0)
    ) / gross_dilution
    half_life = np.log(0.5) / np.log(local_lambda)

    print("Solow growth transition")
    print("=" * 50)
    print(
        "Parameters: "
        f"alpha={alpha}, s={savings_rate}, delta={depreciation}, "
        f"n={population_growth}, g={technology_growth}"
    )
    print(f"Break-even investment Delta = {effective_depreciation:.6f}")
    print(f"Steady state: k*={k_star:.6f}, y*={y_star:.6f}, c*={c_star:.6f}")
    print(
        f"Local convergence factor lambda = {local_lambda:.4f}, "
        f"half-life = {half_life:.2f} periods"
    )
    print(
        f"Terminal k_{periods - 1}={terminal['k']:.6f}; "
        f"gap={abs(terminal['k'] - k_star):.2e}"
    )

    setup_style()

    # ------------------------------------------------------------------
    # Figure 1 - Solow diagram in effective-labor units
    # ------------------------------------------------------------------
    k_grid = np.linspace(0.05 * k_star, 1.45 * k_star, 400)
    investment = savings_rate * output_per_effective_worker(k_grid, alpha)
    break_even = effective_depreciation * k_grid

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_grid, investment, linewidth=2.2, label=r"Investment $s\,k^\alpha$")
    ax1.plot(k_grid, break_even, linewidth=2.2, label=r"Break-even $\Delta\,k$")
    ax1.axvline(k_star, color="black", linestyle="--", linewidth=1.3,
                label=fr"$k^{{\ast}} = {k_star:.2f}$")
    ax1.scatter([k0], [savings_rate * k0**alpha], color="tab:blue", zorder=4,
                label=fr"Start: $k_0 = {k0:.2f}$")
    ax1.set_xlabel(r"Capital per effective worker $k$")
    ax1.set_ylabel("Investment per effective worker")
    ax1.set_title("Solow diagram in effective-labor units")
    ax1.legend()
    save_figure(fig1, "figures/solow-diagram.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 2 - transition with linearization overlay (ground-truth check)
    # ------------------------------------------------------------------
    periods_array = path["period"].to_numpy()
    k_linear = k_star + (k0 - k_star) * local_lambda ** periods_array

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(periods_array, path["k"] / k_star, linewidth=2.1, label=r"$k_t / k^{\ast}$")
    ax2.plot(periods_array, path["y"] / y_star, linewidth=2.1, label=r"$y_t / y^{\ast}$")
    ax2.plot(periods_array, path["c"] / c_star, linewidth=2.1, label=r"$c_t / c^{\ast}$")
    ax2.plot(periods_array, k_linear / k_star, color="black", linestyle=":", linewidth=1.5,
             label=r"Linearization $k^{\ast} + (k_0-k^{\ast})\lambda^t$")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.set_xlabel(r"Period $t$")
    ax2.set_ylabel(r"Ratio to steady-state value")
    ax2.set_title("Transition toward the balanced-growth path")
    ax2.legend()
    save_figure(fig2, "figures/transition-effective-units.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3 - conditional convergence (multi-start) and comparative statics on s
    # ------------------------------------------------------------------
    starts = [0.2 * k_star, 1.0 * k_star, 2.0 * k_star]
    horizon_cc = 200
    cc_paths = [
        simulate_solow_path(
            k_init,
            horizon_cc,
            alpha,
            savings_rate,
            depreciation,
            population_growth,
            technology_growth,
        )
        for k_init in starts
    ]

    s_alternatives = [0.18, 0.24, 0.30]
    diagram_grid = np.linspace(0.05 * k_star, 2.5 * k_star, 400)
    ks_alt = [
        steady_state(
            alpha, s_alt, depreciation, population_growth, technology_growth
        )[0]
        for s_alt in s_alternatives
    ]

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))
    cc_palette = ["tab:blue", "tab:orange", "tab:green"]
    for cc, k_init, color in zip(cc_paths, starts, cc_palette):
        ax4a.plot(
            np.arange(horizon_cc),
            cc["k"] / k_star,
            linewidth=2.0,
            color=color,
            label=fr"$k_0 = {k_init/k_star:.1f}\,k^{{\ast}}$",
        )
    ax4a.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                 label=r"$k^{\ast}$")
    ax4a.set_xlabel(r"Period $t$")
    ax4a.set_ylabel(r"$k_t / k^{\ast}$")
    ax4a.set_title("Conditional convergence: three starts, same primitives")
    ax4a.legend()

    ax4b.plot(diagram_grid, effective_depreciation * diagram_grid, color="black",
              linewidth=2.0, label=r"$\Delta\,k$")
    cs_palette = ["tab:purple", "tab:blue", "tab:red"]
    for s_alt, k_alt, color in zip(s_alternatives, ks_alt, cs_palette):
        ax4b.plot(diagram_grid, s_alt * diagram_grid ** alpha, linewidth=1.8, color=color,
                  label=fr"$s = {s_alt:.2f}$, $k^{{\ast}} = {k_alt:.2f}$")
        ax4b.axvline(k_alt, color=color, linestyle=":", linewidth=1.0, alpha=0.7)
    ax4b.set_xlabel(r"Capital per effective worker $k$")
    ax4b.set_ylabel("Investment per effective worker")
    ax4b.set_title(r"Comparative statics: shifting $s$ shifts $k^{\ast}$")
    ax4b.legend()
    fig4.tight_layout()
    save_figure(fig4, "figures/convergence-and-comparative-statics.png", dpi=150)

    # ------------------------------------------------------------------
    # Audit table - closed form vs terminal simulated values
    # ------------------------------------------------------------------
    table = pd.DataFrame(
        {
            "Object": [
                "Capital per effective worker k",
                "Output per effective worker y",
                "Consumption per effective worker c",
            ],
            "Closed form": [
                f"{k_star:.6f}",
                f"{y_star:.6f}",
                f"{c_star:.6f}",
            ],
            f"Simulated t={periods - 1}": [
                f"{terminal['k']:.6f}",
                f"{terminal['y']:.6f}",
                f"{terminal['c']:.6f}",
            ],
            "Absolute gap": [
                f"{abs(terminal['k'] - k_star):.2e}",
                f"{abs(terminal['y'] - y_star):.2e}",
                f"{abs(terminal['c'] - c_star):.2e}",
            ],
        }
    )
    Path("tables").mkdir(parents=True, exist_ok=True)
    table.to_csv("tables/steady-state-comparison.csv", index=False)

    save_thumbnail("figures/solow-diagram.png", "figures/thumb.png")
    print(f"\nDone: 3 figures + 1 table")


if __name__ == "__main__":
    main()
