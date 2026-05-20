#!/usr/bin/env python3
"""Ramsey-Cass-Koopmans growth solved by finite-horizon shooting.

The state is aggregate capital. Initial capital is predetermined, while
initial consumption is the jump variable selected by the transversality
condition. The script regenerates the tutorial README, figures, and table.
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
from lib.output import ModelReport
from lib.plotting import setup_style


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

    report = ModelReport(
        "Ramsey Saving by Saddle-Path Shooting",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "In Ramsey growth, an economy inherits its capital stock. A capital-poor economy saves "
        "to build productive capacity. A capital-rich economy can consume more while capital "
        "falls toward its long-run level.\n\n"
        "The object is the date-zero consumption choice $c_0$. History fixes $k_0$, but "
        "consumption can jump. The right jump places the economy on the saddle path to the "
        "Ramsey steady state.\n\n"
        "Shooting treats $c_0$ as the unknown. Each guess defines a full path through the "
        "Euler equation and resource law. A root search chooses the guess whose terminal "
        "capital is near $k^{*}$."
    )

    report.add_equations(
        rf"""
The planner chooses a feasible path $\{{c(t)\}}_{{t\geq 0}}$:

$$
\max_{{\{{c(t)\}}}} \int_0^\infty e^{{-\rho t}}
\frac{{c(t)^{{1-\sigma}}}}{{1-\sigma}}\,dt
\quad\text{{s.t.}}\quad
\dot{{k}}(t)=f(k(t))-\delta k(t)-c(t),
\qquad f(k)=Ak^\alpha .
$$

Here $\rho$ is the continuous-time discount rate. The parameter $\delta$ is
depreciation. The parameter $\sigma$ is the CRRA coefficient and inverse EIS.
The parameter $A$ is total factor productivity. The parameter $\alpha$ is the capital share.

The Euler equation is the Keynes-Ramsey rule

$$
\frac{{\dot{{c}}(t)}}{{c(t)}}=
\frac{{f'(k(t))-\delta-\rho}}{{\sigma}}.
$$

Together with the resource law, this gives the two-dimensional system solved by
the code:

$$
\dot{{k}}=Ak^\alpha-\delta k-c,
\qquad
\dot{{c}}=\frac{{\alpha A k^{{\alpha-1}}-\delta-\rho}}{{\sigma}}c .
$$

The steady state satisfies

$$
f'(k^{{*}})=\rho+\delta,
\qquad
k^{{*}}=\left(\frac{{\alpha A}}{{\rho+\delta}}\right)^{{1/(1-\alpha)}},
\qquad
c^{{*}}=f(k^{{*}})-\delta k^{{*}}.
$$

The saddle path starts from the inherited $k_0$. It also satisfies the
infinite-horizon boundary condition

$$
\lim_{{t\to\infty}} e^{{-\rho t}}u'(c(t))k(t)=0
$$

along with the two differential equations above. The finite shooting
calculation chooses $c_0$ so the path is near $(k^{{*}},c^{{*}})$ at date $T$.
Here $u(c)=c^{{1-\sigma}}/(1-\sigma)$ is the period utility function, so $u'(c)=c^{{-\sigma}}$.
"""
    )

    report.add_model_setup(
        "The calibration is deterministic and close to textbook growth examples. "
        "Initial states range from scarce capital to excess capital. The terminal date "
        "approximates the transversality condition; it is not an economic horizon.\n\n"
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| $\\alpha$ | {alpha:.2f} | Capital share in $Ak^\\alpha$ |\n"
        f"| $\\delta$ | {delta:.2f} | Depreciation rate |\n"
        f"| $\\rho$ | {rho:.2f} | Discount rate |\n"
        f"| $\\sigma$ | {sigma:.1f} | CRRA coefficient and inverse EIS |\n"
        f"| $A$ | {A:.1f} | Total factor productivity |\n"
        f"| $T$ | {T:.0f} | Terminal date for shooting |\n"
        f"| Initial capital | $0.25k^{{*}}$ to $2.00k^{{*}}$ | Predetermined state values |\n"
        f"| $k^{{*}}$ | {k_star:.4f} | Ramsey steady-state capital |\n"
        f"| $c^{{*}}$ | {c_star:.4f} | Ramsey steady-state consumption |"
    )

    report.add_solution_method(
        "Shooting solves the Ramsey boundary value problem with repeated initial value "
        "problems. For fixed $k_0$, define the terminal gap "
        "$G(c_0;k_0)=k(T;c_0)-k^{*}$. A trial $c_0$ gives a sign: positive $G$ means "
        "early consumption was too low, and negative $G$ means it was too high.\n\n"
        "The algorithm brackets $c_0$ with one low guess and one high guess. Brent's "
        "method then searches for the jump that makes the terminal gap zero. The final "
        "integration gives the selected capital and consumption paths.\n\n"
        "```text\n"
        "Algorithm: finite-horizon shooting for Ramsey growth\n"
        "Inputs: primitives (alpha, delta, rho, sigma, A), initial capital k0, terminal date T\n"
        "1. Compute (k*, c*) from f'(k*) = rho + delta and c* = f(k*) - delta k*.\n"
        "2. Pick a low c0 guess and a high c0 guess with opposite signs of k(T; c0) - k*.\n"
        "3. For a trial c0, integrate\n"
        "       dot{k} = f(k) - delta k - c,\n"
        "       dot{c}/c = [f'(k) - delta - rho] / sigma\n"
        "   from t = 0 to T, stopping early if feasibility fails.\n"
        "4. Use bisection/Brent updates on c0 until abs(k(T; c0) - k*) is small.\n"
        "5. Reintegrate the ODE with the selected c0 to obtain k(t) and c(t).\n"
        "Output: the saddle-path initial consumption c0(k0) and transition path.\n"
        "```"
    )

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
    report.add_figure(
        "figures/phase-diagram.png",
        "Ramsey phase diagram with selected saddle paths from different initial capital stocks",
        fig1,
        description=(
            "The phase diagram shows how shooting selects the path. The dashed curve is net "
            "output, where $\\dot{k}=0$. The vertical line is $k^{*}$, where $\\dot{c}=0$. "
            "Each colored path starts from a different $k_0$ and uses the chosen $c_0$. "
            "Below $k^{*}$, consumption starts low enough to build capital. Above $k^{*}$, "
            "consumption starts high enough to run capital down."
        ),
    )

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
    report.add_figure(
        "figures/time-paths.png",
        "Ramsey transition paths for capital and consumption after shooting selects c0",
        fig2,
        description=(
            "The time paths show the saving rule along the selected path. A capital-poor "
            "economy keeps consumption below output and lets capital rise. A capital-rich "
            "economy consumes more than net output and lets capital fall. Consumption moves "
            "with the Euler equation as the marginal product changes."
        ),
    )

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
    report.add_figure(
        "figures/convergence-speed.png",
        "Log convergence of capital to the Ramsey steady state with the stable eigenvalue benchmark",
        fig3,
        description=(
            "The log-scale plot shows convergence after the saddle path is selected. Far from "
            "the steady state, the paths bend as the marginal product changes. Near $k^{*}$, "
            "$|k(t)-k^{*}|$ falls close to the stable-eigenvalue rate."
        ),
    )

    terminal_residuals = [abs(sol.y[0, -1] - k_star) / k_star for sol in solutions]
    table_data = {
        "$k_0/k^{*}$": [f"{m:.2f}" for m in k0_multiples],
        "$c_0$ from shooting": [f"{c0:.6g}" for c0 in initial_consumption],
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
    report.add_table(
        "tables/shooting-results.csv",
        "Shooting Diagnostics",
        df,
        description=(
            "The table records the jump chosen by the root search. The consumption ratio is "
            "below one when the planner builds capital. It is above one when the planner "
            "runs capital down. The last column reports the relative terminal capital gap, "
            "the distance $|k(T)-k^{*}|$ expressed as a fraction of $k^{*}$."
        ),
    )

    report.add_takeaway(
        "History fixes $k_0$, but optimality selects $c_0$. A wrong jump sends the economy "
        "toward capital exhaustion or overaccumulation. Shooting finds the jump that keeps "
        "the path feasible and near the Ramsey steady state.\n\n"
        "The selected path gives the Ramsey saving logic. Build capital when it is scarce. "
        "Run capital down when it is abundant. Converge toward the modified golden-rule "
        "point $f'(k^{*})=\\rho+\\delta$."
    )

    report.add_references([
        'Ramsey, F. (1928). "A Mathematical Theory of Saving." *Economic Journal*, 38(152).',
        "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.",
        "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 8.",
        "Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 2.",
        "**See also.** The same Ramsey model is solved by upwind HJB "
        "finite differences in [`optimal-control/hjb-growth/`](../../optimal-control/hjb-growth/) and by "
        "phase-plane eigenanalysis with backward integration in "
        "[`optimal-control/phase-diagrams/`](../../optimal-control/phase-diagrams/).",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} table")


if __name__ == "__main__":
    main()
