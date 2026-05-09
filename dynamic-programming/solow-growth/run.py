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
from lib.output import ModelReport
from lib.plotting import setup_style


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

    report = ModelReport(
        "Solow Growth and Conditional Convergence",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A Solow economy saves a fixed share of output each period. Capital grows "
        "through investment and shrinks through depreciation. Labor and technology "
        "growth make each unit of capital serve more effective workers.\n\n"
        "The state is $k_t = K_t/(A_t L_t)$, capital per unit of effective labor. "
        "If investment exceeds break-even investment, $k_t$ rises. If investment "
        "falls short, $k_t$ falls. Concavity gives one positive steady state.\n\n"
        "The computation iterates the law of motion from an initial $k_0$. A "
        "closed-form steady state checks the path and makes convergence visible."
    )

    report.add_equations(
        r"""
Let $K_t$ denote aggregate capital, $A_t$ labor-augmenting technology, and
$L_t$ raw labor. Output is Cobb-Douglas:

$$Y_t = K_t^\alpha (A_t L_t)^{1-\alpha}, \qquad \alpha\in(0,1).$$

Capital, technology, and labor evolve as

$$K_{t+1}=(1-\delta)K_t + sY_t, \qquad
A_{t+1}=(1+g)A_t, \qquad L_{t+1}=(1+n)L_t,$$

Here $s$ is the saving rate, $\delta$ is depreciation, $g$ is technology
growth, and $n$ is labor-force growth.

Divide by $A_tL_t$ to work in effective-labor units:

$$k_t = \frac{K_t}{A_t L_t}, \qquad
y_t = \frac{Y_t}{A_t L_t} = k_t^\alpha,$$

with consumption per effective worker $c_t = (1-s)\,y_t$.

In these units, the law of motion is one scalar equation:

$$k_{t+1} = \phi(k_t) := \frac{(1-\delta)\,k_t + s\,k_t^\alpha}{(1+g)(1+n)}.$$

Define break-even investment as

$$\Delta := (1+g)(1+n) - 1 + \delta$$

The steady state $k^{\ast}$ solves $\phi(k^{\ast})=k^{\ast}$. This is
equivalent to $s(k^{\ast})^\alpha = \Delta k^{\ast}$.

The closed-form values are

$$k^{\ast}=\left(\frac{s}{\Delta}\right)^{1/(1-\alpha)}, \qquad
y^{\ast}=(k^{\ast})^\alpha, \qquad c^{\ast}=(1-s)\,y^{\ast}.$$

"""
    )

    report.add_model_setup(
        "| Symbol | Value | Role |\n"
        "|--------|------:|------|\n"
        f"| $\\alpha$ | {alpha:.2f} | Capital share in $K^\\alpha(AL)^{{1-\\alpha}}$ |\n"
        f"| $s$ | {savings_rate:.2f} | Exogenous saving rate |\n"
        f"| $\\delta$ | {depreciation:.2f} | Capital depreciation |\n"
        f"| $n$ | {population_growth:.2f} | Labor-force growth |\n"
        f"| $g$ | {technology_growth:.2f} | Labor-augmenting productivity growth |\n"
        f"| $K_0,A_0,L_0$ | {K0:.1f}, {A0:.1f}, {L0:.1f} | Initial stocks; implies $k_0={k0:.1f}$ |\n"
        f"| Horizon $T$ | {periods} | Long enough to make the terminal gap small |\n"
        f"| $\\Delta$ | {effective_depreciation:.4f} | Break-even investment per unit of $k$ |\n"
        f"| $k^{{\\ast}}$ | {k_star:.4f} | Closed-form steady-state capital per effective worker |"
    )

    report.add_solution_method(
        "There is no Bellman equation here. Once $s$ is fixed, the model is the "
        "scalar map $\\phi$. The simulation applies $\\phi$ from $k_0$ until the "
        "path is close to $k^{\\ast}$.\n\n"
        "A local linearization gives the convergence rate near the steady state:\n\n"
        "$$k_{t+1} - k^{\\ast} \\approx \\lambda\\,(k_t - k^{\\ast}), "
        "\\qquad \\lambda \\equiv \\phi'(k^{\\ast}) "
        "= \\frac{(1-\\delta) + s\\alpha\\,(k^{\\ast})^{\\alpha-1}}"
        "{(1+g)(1+n)}.$$\n\n"
        "When $\\lambda \\in (0,1)$, deviations shrink at a geometric rate. The "
        "half-life is $H := \\ln(0.5)/\\ln(\\lambda)$.\n\n"
        "```text\n"
        "Algorithm: Solow transition in effective-labor units\n"
        "Input : primitives (alpha, s, delta, n, g), initial k0, horizon T\n"
        "Output: paths {k_t, y_t, c_t}; closed-form k_star, lambda, half-life H\n"
        "\n"
        "Delta   <- (1 + g)(1 + n) - 1 + delta              # break-even per unit k\n"
        "k_star  <- (s / Delta)^(1 / (1 - alpha))           # closed-form fixed point\n"
        "lambda  <- ((1 - delta) + s * alpha * k_star^(alpha - 1)) / ((1 + g)(1 + n))\n"
        "H       <- ln(0.5) / ln(lambda)                    # local half-life\n"
        "\n"
        "set k <- k0\n"
        "for t = 0, 1, ..., T - 1:\n"
        "    y_t       <- k^alpha\n"
        "    c_t       <- (1 - s) * y_t\n"
        "    invest_t  <- s * y_t\n"
        "    k         <- ((1 - delta) * k + s * y_t) / ((1 + g)(1 + n))\n"
        "\n"
        "audit         : |k_T - k_star|, |y_T - y_star|, |c_T - c_star|\n"
        "linearization : compare k_t to k_star + (k_0 - k_star) * lambda^t\n"
        "```\n\n"
        f"For this calibration, $\\lambda \\approx {local_lambda:.3f}$ and the "
        f"local half-life is roughly {half_life:.1f} periods."
    )

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
    report.add_results(
        f"At $k_0={k0:.2f}$ the curved schedule $s k^\\alpha$ sits above the linear "
        f"break-even line $\\Delta k$. Capital deepens from the start. The curves "
        f"cross at $k^{{\\ast}}={k_star:.3f}$, the unique positive steady state. "
        "Above $k^{\\ast}$, break-even investment exceeds saving."
    )
    report.add_figure(
        "figures/solow-diagram.png",
        "Solow diagram with investment and break-even investment in effective-labor units",
        fig1,
    )

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
    report.add_results(
        "The transition plot normalizes each series by its steady-state value. "
        "Output and consumption move together because $c_t=(1-s)y_t$. Capital "
        "moves more slowly because it inherits the past stock.\n\n"
        "The dotted line is $k^{\\ast}+(k_0-k^{\\ast})\\lambda^t$. It tracks the "
        f"path well near $k^{{\\ast}}$. By period {periods - 1}, simulated $k$ is "
        f"within {abs(terminal['k']-k_star):.2e} of $k^{{\\ast}}$."
    )
    report.add_figure(
        "figures/transition-effective-units.png",
        "Capital, output, and consumption converging to steady state, with the linear approximation overlaid",
        fig2,
    )

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
    report.add_results(
        "The left panel starts three economies from different capital stocks. "
        "They share the same primitives, so they converge to the same normalized "
        "$k^{\\ast}$. Conditional convergence means convergence to that own steady state.\n\n"
        "The right panel changes the saving rate. Higher $s$ shifts investment up "
        f"and gives $k^{{\\ast}}\\in\\{{{ks_alt[0]:.2f},\\,{ks_alt[1]:.2f},\\,{ks_alt[2]:.2f}\\}}$. "
        "It raises the level of output per worker, not the long-run growth rate."
    )
    report.add_figure(
        "figures/convergence-and-comparative-statics.png",
        "Conditional convergence from three starting points and the comparative statics of the saving rate",
        fig4,
    )

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
    report.add_results(
        "The table compares the closed form with the terminal simulation. Any gap "
        "comes from finite horizon truncation. The geometric residual is about "
        f"{abs(k0 - k_star) * local_lambda ** (periods - 1):.2e}."
    )
    report.add_table(
        "tables/steady-state-comparison.csv",
        "Closed-form steady state versus terminal simulation",
        table,
    )

    report.add_takeaway(
        "Solow disciplines what saving can and cannot do. A higher $s$ raises the "
        "level of $k^{\\ast}$ but leaves the long-run growth rate of output per "
        "worker equal to $g$.\n\n"
        "Conditional convergence follows from the same steady-state logic. "
        "Economies with the same primitives approach the same balanced-growth path. "
        "Different primitives imply different paths."
    )

    report.add_references(
        [
            'Solow, R. M. (1956). "A Contribution to the Theory of Economic Growth." '
            "*Quarterly Journal of Economics*, 70(1), 65-94.",
            'Mankiw, N. G., Romer, D., and Weil, D. N. (1992). "A Contribution to '
            'the Empirics of Economic Growth." *Quarterly Journal of Economics*, '
            "107(2), 407-437.",
            "Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 1.",
            "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 1.",
            "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 2.",
        ]
    )

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
