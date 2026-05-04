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
    lambda_unstable = float(np.max(eigvals))
    half_life = np.log(2.0) / abs(lambda_stable)

    ref_solution = solutions[0]
    ref_dev = np.abs(ref_solution.y[0] - k_star) / k_star
    fit_mask = (t_eval >= 35.0) & (t_eval <= 110.0) & (ref_dev > 1e-10)
    empirical_rate = float(np.polyfit(t_eval[fit_mask], np.log(ref_dev[fit_mask]), 1)[0])

    setup_style()

    report = ModelReport(
        "Ramsey Growth by Shooting",
        "A continuous-time Ramsey planner picks the one initial consumption level that puts capital on the stable path.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The Ramsey-Cass-Koopmans model asks how much output a planner should consume "
        "today and how much should be invested for future production. Capital $k(0)$ is "
        "inherited from the past, but consumption can jump at date zero. The economic "
        "selection problem is therefore sharp: for a given $k_0$, which $c_0$ is consistent "
        "with optimal intertemporal saving?\n\n"
        "Most initial consumption choices are wrong. If $c_0$ is too high, the economy runs "
        "capital down too aggressively and violates feasibility. If it is too low, the "
        "planner overaccumulates capital relative to the present-value boundary condition. "
        "The shooting method turns that economic restriction into a one-dimensional root "
        "search over $c_0$.\n\n"
        "This tutorial is the algorithmic companion to the neighboring "
        "[Ramsey phase-diagram](../phase-diagrams/) example and the HJB formulation in "
        "[HJB growth](../hjb-growth/). The model is the same; the numerical representation "
        "is different."
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

Here $\rho$ is the continuous-time discount rate, $\delta$ is depreciation,
$\sigma$ is the CRRA coefficient and inverse intertemporal elasticity of
substitution, and $A$ is total factor productivity.

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

The finite shooting calculation approximates the infinite-horizon boundary
condition

$$
\lim_{{t\to\infty}} e^{{-\rho t}}u'(c(t))k(t)=0
$$

by choosing $c_0$ so that the path is near $(k^{{*}},c^{{*}})$ at a long terminal date.
"""
    )

    report.add_model_setup(
        "The calibration is deterministic and deliberately close to textbook growth "
        "examples. The terminal date is a numerical device used to approximate the "
        "infinite-horizon transversality condition; it is not an economic horizon.\n\n"
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
        "The method solves a boundary value problem by repeated initial value problems. "
        "Capital at date zero is fixed. Initial consumption is guessed, the Ramsey ODE is "
        "integrated forward, and the terminal capital gap determines whether the guess was "
        "too low or too high. Brent bisection then finds the root.\n\n"
        "For this model, the terminal residual is monotone in the relevant bracket. Low "
        "$c_0$ leaves too much capital at $T$; high $c_0$ exhausts capital before the "
        "terminal date or leaves too little capital. The bracket must be wide enough for "
        "initial states above $k^{*}$, where the optimal path begins with consumption above "
        "net output so that capital decumulates.\n\n"
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
        "```\n\n"
        "The local speed check comes from the Jacobian of $(\\dot{k},\\dot{c})$ at the "
        "steady state. Its eigenvalues are "
        f"$\\lambda_s={lambda_stable:.4f}$ and $\\lambda_u={lambda_unstable:.4f}$, so the "
        f"local half-life is $\\ln(2)/|\\lambda_s|={half_life:.1f}$ time units. On the "
        f"computed path from $0.25k^{{*}}$, the fitted late-transition rate is "
        f"$\\hat{{\\lambda}}={empirical_rate:.4f}$."
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
            "The phase diagram shows why shooting is an economic selection rule, not only a "
            "numerical trick. The dashed curve is net output, where $\\dot{k}=0$; the vertical "
            "line is $k^{*}$, where $\\dot{c}=0$. Each colored path starts from a different "
            "$k_0$ and uses the $c_0$ found by shooting. Below $k^{*}$, consumption starts low "
            "enough for investment to build capital. Above $k^{*}$, consumption starts above "
            "net output, so capital is deliberately run down."
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
            "The time paths make the saving logic easier to read. A capital-poor economy "
            "keeps consumption below output and lets capital rise; a capital-rich economy "
            "consumes more than current net output and moves down. Consumption is not fixed "
            "at a constant saving rate. It moves according to the Euler equation as the "
            "marginal product of capital changes along the transition."
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
            "The log-scale convergence plot separates nonlinear transition dynamics from the "
            "local stable-root approximation. Far from the steady state, the paths bend because "
            "the marginal product changes quickly. Once the economy is close to $k^{*}$, the "
            "decline in $|k(t)-k^{*}|$ is approximately exponential at the stable eigenvalue."
        ),
    )

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
        "Terminal capital gap": [f"{resid:.2e}" for resid in terminal_residuals],
    }
    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/shooting-results.csv",
        "Shooting Diagnostics",
        df,
        description=(
            "The table records the jump variable selected by the root search. The consumption "
            "ratio is below one when the planner is building capital and above one when the "
            "planner is running capital down. The last column is the finite-horizon shooting "
            "residual, kept visible so the boundary-condition approximation is auditable."
        ),
    )

    report.add_takeaway(
        "The Ramsey shooting problem is a clean example of how economics and numerics line up. "
        "History fixes $k_0$, but optimality selects $c_0$. The root search is finding the "
        "initial consumption level that keeps the path feasible and satisfies the "
        "transversality condition.\n\n"
        "The exercise also shows why saddle-path systems are easy to state but delicate to "
        "compute. A small error in $c_0$ sends the economy toward capital exhaustion or "
        "overaccumulation. Once the correct path is selected, the model delivers the usual "
        "Ramsey logic: invest when capital is scarce, decumulate when capital is abundant, and "
        "converge toward the modified golden-rule point $f'(k^{*})=\\rho+\\delta$."
    )

    report.add_references([
        'Ramsey, F. (1928). "A Mathematical Theory of Saving." *Economic Journal*, 38(152).',
        "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.",
        "Acemoglu, D. (2009). *Introduction to Modern Economic Growth*. Princeton University Press, Ch. 8.",
        "Romer, D. (2019). *Advanced Macroeconomics*. McGraw-Hill, 5th edition, Ch. 2.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} table")


if __name__ == "__main__":
    main()
