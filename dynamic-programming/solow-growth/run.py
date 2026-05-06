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
        "Exogenous saving, a one-line transition map, and a closed-form steady state for the level-versus-trend split.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Solow strips out the household optimization problem. With saving fixed at "
        "a constant fraction of output, capital accumulation reduces to a one-line "
        "scalar map and the entire model can be solved with a calculator. That is "
        "not a limitation; it is what makes the model useful. The level effects of "
        "saving and depreciation become legible without taking a stand on "
        "preferences or the intertemporal elasticity of substitution.\n\n"
        "The state is $k_t = K_t/(A_t L_t)$, capital per unit of effective labor. "
        "Whether $k_t$ rises or falls depends only on whether gross investment "
        "$s k_t^\\alpha$ exceeds the dilution required to keep $k_t$ constant "
        "against depreciation, population growth, and labor-augmenting technical "
        "change. Below the cutoff capital deepens; above it dilution wins. "
        "Concavity of $f(k)=k^\\alpha$ pins down a unique nonzero fixed point.\n\n"
        "Solow sits between two nearby tutorials. [Cake eating](../cake-eating/) "
        "has a Bellman equation but no production. [Optimal growth](../optimal-growth/) "
        "has both production and an Euler equation. Solow keeps the production "
        "side and drops the Euler equation by fiat, which is what makes the "
        "level-versus-trend split so easy to read off."
    )

    report.add_equations(
        r"""
Let $K_t$ denote aggregate capital, $A_t$ labor-augmenting technology, and
$L_t$ raw labor. Output is Cobb-Douglas:

$$Y_t = K_t^\alpha (A_t L_t)^{1-\alpha}, \qquad \alpha\in(0,1).$$

Capital, technology, and labor evolve as

$$K_{t+1}=(1-\delta)K_t + sY_t, \qquad
A_{t+1}=(1+g)A_t, \qquad L_{t+1}=(1+n)L_t,$$

where $s$ is the saving rate, $\delta$ is depreciation, $g$ is
labor-augmenting productivity growth, and $n$ is labor-force growth. Switching
to effective-labor units,

$$k_t = \frac{K_t}{A_t L_t}, \qquad
y_t = \frac{Y_t}{A_t L_t} = k_t^\alpha,$$

the discrete-time law of motion collapses to a single scalar equation,

$$k_{t+1} = \phi(k_t) \;:=\; \frac{(1-\delta)\,k_t + s\,k_t^\alpha}{(1+g)(1+n)}.$$

The steady state $k^{\ast}$ solves $\phi(k^{\ast})=k^{\ast}$, equivalently
$s(k^{\ast})^\alpha = \Delta k^{\ast}$, where

$$\Delta \;:=\; (1+g)(1+n) - 1 + \delta$$

is the per-unit break-even investment required to keep $k$ constant. Hence

$$k^{\ast}=\left(\frac{s}{\Delta}\right)^{1/(1-\alpha)}, \qquad
y^{\ast}=(k^{\ast})^\alpha, \qquad c^{\ast}=(1-s)\,y^{\ast}.$$

Competitive factor prices follow from marginal products:

$$MPK_t = \alpha\, k_t^{\alpha-1}, \qquad
\frac{w_t}{A_t} = (1-\alpha)\, k_t^\alpha.$$

The plotted wage is $w_t/A_t$, the wage per unit of effective labor. The wage
per raw worker is $w_t = (1-\alpha)\,A_t\,k_t^\alpha$ and grows with $A_t$
along the balanced-growth path.
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
        f"| Horizon $T$ | {periods} | Long enough to make the residual finite-horizon gap visible |\n"
        f"| $\\Delta$ | {effective_depreciation:.4f} | Break-even investment per unit of $k$ |\n"
        f"| $k^{{\\ast}}$ | {k_star:.4f} | Closed-form steady-state capital per effective worker |"
    )

    report.add_solution_method(
        "There is no Bellman equation here. Once $s$ is fixed, the model is the "
        "scalar map $\\phi$ from the previous section, and $k^{\\ast}$ has a "
        "closed form. The simulation iterates $\\phi$ from $k_0$, and the "
        "closed-form steady state plays the role that a finely solved benchmark "
        "plays in less tractable problems: ground truth that the iteration is "
        "audited against.\n\n"
        "Local convergence is read off the linearization of $\\phi$ at the "
        "steady state:\n\n"
        "$$k_{t+1} - k^{\\ast} \\;\\approx\\; \\lambda\\,(k_t - k^{\\ast}), "
        "\\qquad \\lambda \\;\\equiv\\; \\phi'(k^{\\ast}) "
        "\\;=\\; \\frac{(1-\\delta) + s\\alpha\\,(k^{\\ast})^{\\alpha-1}}"
        "{(1+g)(1+n)}.$$\n\n"
        "When $\\lambda \\in (0,1)$, deviations from the balanced-growth path "
        "decay geometrically with half-life $\\ln(0.5)/\\ln(\\lambda)$. With "
        "saving rates and depreciation rates calibrated to advanced economies, "
        "$\\lambda$ is typically close to one and the half-life runs to decades. "
        "That slow rate is the empirical fact behind Mankiw, Romer, and Weil "
        "(1992): countries do converge to their own balanced-growth paths, but "
        "slowly enough that initial conditions still show up in cross-section "
        "growth regressions a generation later.\n\n"
        "```text\n"
        "Algorithm: Solow transition in effective-labor units\n"
        "Input : primitives (alpha, s, delta, n, g), initial k0, horizon T\n"
        "Output: paths {k_t, y_t, c_t, MPK_t, w_t/A_t};\n"
        "        closed-form k_star, local rate lambda, half-life H\n"
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
        "    MPK_t     <- alpha * k^(alpha - 1)\n"
        "    w_t / A_t <- (1 - alpha) * k^alpha\n"
        "    k         <- ((1 - delta) * k + s * y_t) / ((1 + g)(1 + n))\n"
        "\n"
        "audit         : |k_T - k_star|, |y_T - y_star|, |c_T - c_star|\n"
        "linearization : compare k_t to k_star + (k_0 - k_star) * lambda^t\n"
        "```\n\n"
        f"For this calibration, $\\lambda \\approx {local_lambda:.3f}$ and the "
        f"local half-life is roughly {half_life:.1f} periods. Annual "
        "$g+n+\\delta$ near nine percent makes the transition feel slow no "
        "matter what $s$ is set to."
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
        f"break-even line $\\Delta k$, so $k_t$ deepens from the start. The two curves "
        f"cross at $k^{{\\ast}}={k_star:.3f}$, the unique nonzero fixed point. "
        "Concavity of $f(k)=k^\\alpha$ guarantees a single intersection and a stable "
        "one: above $k^{\\ast}$ the linear schedule grows faster than the concave one "
        "and dilution wins. The crossing is not estimated from the simulation; it is "
        "the closed-form $(s/\\Delta)^{1/(1-\\alpha)}$ implied by the primitives."
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
        "All three series are normalized by their balanced-growth values. Output and "
        "consumption track each other identically because $c_t=(1-s)y_t$; the saving "
        "rate is the choice the model has refused to make. Capital lags both because "
        "it inherits its own past stock, and the gap closes at the geometric rate "
        "$\\lambda$ derived above. The dotted black line is the linear-approximation "
        "prediction $k^{\\ast}+(k_0-k^{\\ast})\\lambda^t$, plotted in the same "
        "normalized units. Linearization tracks the simulation closely as $k_t$ nears "
        "$k^{\\ast}$ but is visibly off early on, where the curvature of $\\phi$ "
        "still matters. By the terminal period, the simulated $k$ sits within "
        f"{abs(terminal['k']-k_star):.2e} of $k^{{\\ast}}$ in absolute terms."
    )
    report.add_figure(
        "figures/transition-effective-units.png",
        "Capital, output, and consumption converging to steady state, with the linear approximation overlaid",
        fig2,
    )

    # ------------------------------------------------------------------
    # Figure 3 - factor prices
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax3a.plot(periods_array, path["mpk"], linewidth=2.1)
    ax3a.axhline(mpk_star, color="black", linestyle="--", linewidth=1.2,
                 label=fr"$MPK^{{\ast}}={mpk_star:.3f}$")
    ax3a.set_xlabel(r"Period $t$")
    ax3a.set_ylabel(r"$MPK_t$")
    ax3a.set_title("Marginal product of capital")
    ax3a.legend()

    ax3b.plot(periods_array, path["wage_per_effective_worker"], linewidth=2.1, color="tab:green")
    ax3b.axhline(wage_eff_star, color="black", linestyle="--", linewidth=1.2,
                 label=fr"$(w/A)^{{\ast}}={wage_eff_star:.3f}$")
    ax3b.set_xlabel(r"Period $t$")
    ax3b.set_ylabel(r"$w_t / A_t$")
    ax3b.set_title("Effective wage")
    ax3b.legend()
    fig3.tight_layout()
    report.add_results(
        "Factor prices read the same convergence story from the firm side. Early on, "
        "capital is scarce, so $MPK_t$ is high and the effective wage is depressed. "
        f"As $k_t$ deepens, both move monotonically toward the steady-state values "
        f"implied by $k^{{\\ast}}$: $MPK^{{\\ast}}={mpk_star:.3f}$ and "
        f"$(w/A)^{{\\ast}}={wage_eff_star:.3f}$. In a multi-country reading this is "
        "the textbook prediction that the return on capital should be falling and "
        "wages rising as poorer economies catch up to richer ones with the same "
        "technology."
    )
    report.add_figure(
        "figures/factor-prices.png",
        "Factor prices along the Solow transition",
        fig3,
    )

    # ------------------------------------------------------------------
    # Figure 4 - conditional convergence (multi-start) and comparative statics on s
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
        "Two ways to read the same model. The left panel runs three economies with "
        "identical primitives but different starting capital: $k_0$ at one fifth, "
        "one, and twice the steady state. All three converge to the same "
        "$k^{\\ast}=1$ in normalized units. *Conditional* matters here, because the "
        "common steady state is the one pinned down by $(s,n,g,\\delta,\\alpha)$, "
        "not a common world level. The right panel makes the same point algebraic. "
        "Three saving rates, $s\\in\\{0.18, 0.24, 0.30\\}$, slide the investment "
        "schedule up while leaving $\\Delta k$ fixed; the new intersections give "
        f"$k^{{\\ast}}\\in\\{{{ks_alt[0]:.2f},\\,{ks_alt[1]:.2f},\\,{ks_alt[2]:.2f}\\}}$. "
        "Doubling $s$ raises $k^{\\ast}$ by a factor of "
        f"$2^{{1/(1-\\alpha)}}\\approx {2 ** (1 / (1 - alpha)):.2f}$, but once at "
        "the new steady state output per worker still grows at $g$. Permanently "
        "faster growth in this model has to come from $g$, not $s$."
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
                "Marginal product of capital MPK",
                "Effective wage w/A",
            ],
            "Closed form": [
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
    report.add_results(
        "Both the transition map and the steady state are analytical, so any "
        "remaining gap in the table is finite-horizon truncation, not numerical "
        "error. The bound is the geometric residual "
        f"$|k_0-k^{{\\ast}}|\\,\\lambda^{{T-1}}$, which at "
        f"$\\lambda={local_lambda:.3f}$ and $T={periods}$ is on the order of "
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
        "worker equal to $g$. The same logic delivers conditional convergence: "
        "economies with the same primitives approach the same balanced-growth path, "
        "while economies that differ in $s$, $n$, or $\\delta$ approach different "
        "ones. [Optimal growth](../optimal-growth/) lifts the constant-saving "
        "assumption and lets an Euler equation choose $s$ endogenously; what "
        "survives is the same level-versus-trend split that Solow puts on a single "
        "line of algebra."
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
