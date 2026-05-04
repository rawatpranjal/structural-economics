#!/usr/bin/env python3
"""Solow growth in effective-labor units.

The tutorial solves the deterministic Solow transition by iterating the exact
one-dimensional law of motion for capital per effective worker.
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
    mpk_path = np.empty(periods)
    wage_eff_path = np.empty(periods)

    k_path[0] = k0
    for t in range(periods):
        y_path[t] = output_per_effective_worker(k_path[t], alpha)
        c_path[t] = (1.0 - savings_rate) * y_path[t]
        investment_path[t] = savings_rate * y_path[t]
        mpk_path[t] = alpha * k_path[t] ** (alpha - 1.0)
        wage_eff_path[t] = (1.0 - alpha) * y_path[t]
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
            "mpk": mpk_path,
            "wage_per_effective_worker": wage_eff_path,
        }
    )


def main() -> None:
    # Baseline calibration: a simple annual Solow economy.
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
    gross_dilution = (1.0 + technology_growth) * (1.0 + population_growth)
    effective_depreciation = gross_dilution - 1.0 + depreciation
    k_star = (savings_rate / effective_depreciation) ** (1.0 / (1.0 - alpha))
    y_star = output_per_effective_worker(k_star, alpha)
    c_star = (1.0 - savings_rate) * y_star
    mpk_star = alpha * k_star ** (alpha - 1.0)
    wage_eff_star = (1.0 - alpha) * y_star

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
    local_convergence_factor = (
        (1.0 - depreciation) + savings_rate * alpha * k_star ** (alpha - 1.0)
    ) / gross_dilution
    half_life = np.log(0.5) / np.log(local_convergence_factor)

    print("Solow growth transition")
    print("=" * 50)
    print(
        "Parameters: "
        f"alpha={alpha}, s={savings_rate}, delta={depreciation}, "
        f"n={population_growth}, g={technology_growth}"
    )
    print(f"Exact effective depreciation term = {effective_depreciation:.6f}")
    print(f"Steady state: k*={k_star:.6f}, y*={y_star:.6f}, c*={c_star:.6f}")
    print(
        f"Terminal k_{periods - 1}={terminal['k']:.6f}; "
        f"gap={abs(terminal['k'] - k_star):.2e}"
    )

    setup_style()

    report = ModelReport(
        "Solow Growth and Conditional Convergence",
        "A deterministic growth economy where saving is exogenous and technology pins down balanced-growth dynamics.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The Solow model is useful precisely because it removes the household "
        "Euler equation. A fixed saving rate turns growth into an accounting "
        "problem: output produces investment, investment raises the capital "
        "stock, and population growth, technology growth, and depreciation dilute "
        "capital measured per effective worker.\n\n"
        "The economic object in this tutorial is the transition of "
        "$k_t=K_t/(A_tL_t)$. If an economy starts below its balanced-growth "
        "capital intensity, investment exceeds the amount needed to keep "
        "$k_t$ constant and the economy accumulates. If it starts above that "
        "level, effective depreciation dominates and capital intensity falls. "
        "This makes Solow a natural bridge between the finite-resource and "
        "optimal-growth examples: [cake eating](../cake-eating/) has no "
        "production, [optimal growth](../optimal-growth/) makes saving optimal, "
        "and Solow sits in between as an exogenous-saving transition map."
    )

    report.add_equations(
        r"""
Let $K_t$ be aggregate capital, $A_t$ labor-augmenting technology, and $L_t$
labor. Output is Cobb-Douglas:

$$Y_t = K_t^\alpha (A_t L_t)^{1-\alpha}, \qquad \alpha\in(0,1).$$

Capital, technology, and labor evolve according to

$$K_{t+1}=(1-\delta)K_t+sY_t,$$

$$A_{t+1}=(1+g)A_t,\qquad L_{t+1}=(1+n)L_t,$$

where $s$ is the exogenous saving rate, $\delta$ is depreciation, $g$ is
technology growth, and $n$ is population growth. In effective-labor units,

$$k_t=\frac{K_t}{A_tL_t},\qquad y_t=\frac{Y_t}{A_tL_t}=k_t^\alpha,$$

so the exact discrete-time transition is

$$k_{t+1}=
\frac{(1-\delta)k_t+s k_t^\alpha}{(1+g)(1+n)}.$$

The steady state in effective units solves

$$s(k^{*})^\alpha = \Delta k^{*},$$

with

$$\Delta=(1+g)(1+n)-1+\delta.$$

Thus

$$k^{*}=\left(\frac{s}{\Delta}\right)^{1/(1-\alpha)},\qquad
y^{*}=(k^{*})^\alpha,\qquad c^{*}=(1-s)y^{*}.$$

Competitive factor prices are the marginal products

$$MPK_t=\alpha k_t^{\alpha-1},\qquad
\frac{w_t}{A_t}=(1-\alpha)k_t^\alpha.$$

The plotted wage is $w_t/A_t$, the wage per unit of effective labor. The wage
per raw worker grows with $A_t$ along the balanced-growth path.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Role |\n"
        f"|-----------|------:|------|\n"
        f"| $\\alpha$ | {alpha:.2f} | Capital share in $K^\\alpha(AL)^{{1-\\alpha}}$ |\n"
        f"| $s$ | {savings_rate:.2f} | Exogenous fraction of output invested |\n"
        f"| $\\delta$ | {depreciation:.2f} | Physical depreciation of capital |\n"
        f"| $n$ | {population_growth:.2f} | Labor-force growth |\n"
        f"| $g$ | {technology_growth:.2f} | Labor-augmenting technology growth |\n"
        f"| $K_0,A_0,L_0$ | {K0:.1f}, {A0:.1f}, {L0:.1f} | Initial aggregate stocks, implying $k_0={k0:.1f}$ |\n"
        f"| Horizon | {periods} periods | Long enough for the transition gap to be visible |\n"
        f"| $\\Delta$ | {effective_depreciation:.4f} | Exact break-even investment term in effective units |\n"
        f"| $k^{*}$ | {k_star:.4f} | Analytical steady-state capital per effective worker |"
    )

    report.add_solution_method(
        "There is no Bellman equation here. Once $s$ is fixed, the whole model "
        "is the scalar map for $k_{t+1}$. The analytical steady state is used as "
        "ground truth; the simulation is only the transition path generated by "
        "repeatedly applying that map.\n\n"
        "```text\n"
        "Algorithm: deterministic Solow transition in effective units\n"
        "Input: primitives alpha, s, delta, n, g; initial k0; horizon T\n"
        "Output: paths for k_t, y_t, c_t, MPK_t, and w_t/A_t\n"
        "Delta = (1 + g)(1 + n) - 1 + delta\n"
        "k_star = (s / Delta)^(1 / (1 - alpha))\n"
        "set k = k0\n"
        "for t = 0, 1, ..., T-1:\n"
        "    y_t = k^alpha\n"
        "    c_t = (1 - s) y_t\n"
        "    investment_t = s y_t\n"
        "    break_even_t = Delta k\n"
        "    MPK_t = alpha k^(alpha - 1)\n"
        "    w_t / A_t = (1 - alpha) k^alpha\n"
        "    k = ((1 - delta) k + s k^alpha) / ((1 + g)(1 + n))\n"
        "compare the terminal path to k_star, y_star, and c_star\n"
        "```\n\n"
        f"With this calibration, the local convergence factor around $k^{*}$ is "
        f"**{local_convergence_factor:.3f}**, implying a half-life of about "
        f"**{half_life:.1f} periods** for small deviations from the balanced-growth path."
    )

    k_grid = np.linspace(0.05 * k_star, 1.45 * k_star, 400)
    investment = savings_rate * output_per_effective_worker(k_grid, alpha)
    break_even = effective_depreciation * k_grid

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_grid, investment, linewidth=2.2, label="$s k^\\alpha$")
    ax1.plot(k_grid, break_even, linewidth=2.2, label="$\\Delta k$")
    ax1.axvline(k_star, color="black", linestyle="--", linewidth=1.3, label="$k^{*}$")
    ax1.scatter([k0], [savings_rate * output_per_effective_worker(k0, alpha)], color="tab:blue", zorder=4)
    ax1.set_xlabel("Capital per effective worker $k$")
    ax1.set_ylabel("Investment per effective worker")
    ax1.set_title("Solow Diagram in Effective-Labor Units")
    ax1.legend()
    report.add_results(
        "The first figure is the analytical Solow diagram. The curved schedule is "
        "actual investment per effective worker, $s k^\\alpha$. The line is the "
        "investment required to offset depreciation, population growth, and "
        "technology growth, $\\Delta k$. Their intersection is not estimated from "
        f"the simulation; it is the exact $k^{*}={k_star:.3f}$ implied by the primitives."
    )
    report.add_figure(
        "figures/solow-diagram.png",
        "Solow diagram with investment and break-even investment in effective-labor units",
        fig1,
        description="Because $k_0=1.000$ lies to the left of the intersection, the economy begins with investment above break-even investment. Capital per effective worker therefore rises.",
    )

    periods_array = path["period"].to_numpy()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(periods_array, path["k"] / k_star, linewidth=2.1, label="$k_t/k^{*}$")
    ax2.plot(periods_array, path["y"] / y_star, linewidth=2.1, label="$y_t/y^{*}$")
    ax2.plot(periods_array, path["c"] / c_star, linewidth=2.1, label="$c_t/c^{*}$")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Period $t$")
    ax2.set_ylabel("Ratio to steady-state value")
    ax2.set_title("Transition Toward the Balanced-Growth Path")
    ax2.legend()
    report.add_results(
        "The transition figure normalizes capital, output, and consumption by "
        "their effective-unit steady states. Capital moves more slowly than "
        "output because production is concave: as $k_t$ rises, the marginal "
        "product of the next unit of capital falls. By the terminal period, "
        f"$|k_{{T-1}}-k^{*}|$ is **{abs(terminal['k'] - k_star):.2e}**."
    )
    report.add_figure(
        "figures/transition-effective-units.png",
        "Transition of capital, output, and consumption toward steady state",
        fig2,
        description="Consumption and output have the same normalized path because consumption is the fixed share $(1-s)$ of output. This is the mechanical implication of exogenous saving.",
    )

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax3a.plot(periods_array, path["mpk"], linewidth=2.1)
    ax3a.axhline(mpk_star, color="black", linestyle="--", linewidth=1.2)
    ax3a.set_xlabel("Period $t$")
    ax3a.set_ylabel("$MPK_t$")
    ax3a.set_title("Marginal Product of Capital")

    ax3b.plot(periods_array, path["wage_per_effective_worker"], linewidth=2.1, color="tab:green")
    ax3b.axhline(wage_eff_star, color="black", linestyle="--", linewidth=1.2)
    ax3b.set_xlabel("Period $t$")
    ax3b.set_ylabel("$w_t/A_t$")
    ax3b.set_title("Effective Wage")
    fig3.tight_layout()
    report.add_results(
        "Factor prices make the convergence mechanism observable. Starting from "
        "low capital, the marginal product of capital is high and the effective "
        "wage is low. As capital deepens, $MPK_t$ falls toward "
        f"**{mpk_star:.3f}** while $w_t/A_t$ rises toward **{wage_eff_star:.3f}**. "
        "The same diminishing-returns force underlies conditional convergence."
    )
    report.add_figure(
        "figures/factor-prices.png",
        "Factor prices along the Solow transition",
        fig3,
        description="The dashed lines are analytical steady-state values. The simulation approaches them because capital per effective worker approaches $k^{*}$.",
    )

    table = pd.DataFrame(
        {
            "Object": [
                "Capital per effective worker k",
                "Output per effective worker y",
                "Consumption per effective worker c",
                "Marginal product of capital MPK",
                "Effective wage w/A",
            ],
            "Analytical steady state": [
                f"{k_star:.6f}",
                f"{y_star:.6f}",
                f"{c_star:.6f}",
                f"{mpk_star:.6f}",
                f"{wage_eff_star:.6f}",
            ],
            f"Simulated t={periods - 1}": [
                f"{terminal['k']:.6f}",
                f"{terminal['y']:.6f}",
                f"{terminal['c']:.6f}",
                f"{terminal['mpk']:.6f}",
                f"{terminal['wage_per_effective_worker']:.6f}",
            ],
            "Absolute gap": [
                f"{abs(terminal['k'] - k_star):.2e}",
                f"{abs(terminal['y'] - y_star):.2e}",
                f"{abs(terminal['c'] - c_star):.2e}",
                f"{abs(terminal['mpk'] - mpk_star):.2e}",
                f"{abs(terminal['wage_per_effective_worker'] - wage_eff_star):.2e}",
            ],
        }
    )
    report.add_table(
        "tables/steady-state-comparison.csv",
        "Analytical steady state versus terminal simulation",
        table,
        description="The table is a check on the simulation, not a separate estimator. Since the transition map and steady state are both analytical, the remaining gap is just the finite horizon.",
    )

    report.add_takeaway(
        "Solow separates level effects from growth effects. A higher saving rate "
        "raises capital and output per effective worker, but it does not change "
        "the balanced-growth rate of output per worker. In this model, long-run "
        "per-capita growth comes from $g$; saving and depreciation determine the "
        "level around which the economy grows. That distinction is exactly what "
        "the Ramsey and RBC tutorials complicate by making saving an equilibrium "
        "choice rather than an imposed fraction of output."
    )

    report.add_references(
        [
            'Solow, R. (1956). "A Contribution to the Theory of Economic Growth." '
            "*Quarterly Journal of Economics*, 70(1), 65-94.",
            "Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 1.",
            "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 1.",
            "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 2.",
        ]
    )

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
