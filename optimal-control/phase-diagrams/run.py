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
from lib.plotting import setup_style
from lib.output import ModelReport


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

    report = ModelReport(
        "Ramsey Consumption Choice and Saddle Paths",
        "Trace the stable arm that selects initial Ramsey consumption.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "An economy begins with inherited capital. A Ramsey planner chooses consumption "
        "today and lets investment carry capital forward. The wrong choice either runs "
        "capital down or delays consumption too long.\n\n"
        "The object is the phase diagram in $(k,c)$. Capital is the state. Consumption "
        "is the control. Nullclines show where each variable stops moving. The stable arm "
        "is the curve of choices that converges to the saddle steady state.\n\n"
        "The computation traces that arm. The code linearizes the ODE at the steady state, "
        "uses the stable eigenvector, and integrates backward. Forward paths then show "
        "which initial consumption choices miss the boundary condition."
    )

    report.add_equations(r"""
The planner solves

$$
\max_{\{c(t)\}_{t \geq 0}}
\int_0^\infty e^{-\rho t}\frac{c(t)^{1-\sigma}}{1-\sigma}\,dt
\quad\text{s.t.}\quad
\dot{k}(t)=Ak(t)^\alpha-\delta k(t)-c(t).
$$

The Euler equation and the resource law form the two-dimensional system

$$
\dot{k}=f(k)-\delta k-c,
\qquad
\frac{\dot{c}}{c}=\frac{f'(k)-\delta-\rho}{\sigma},
\qquad
f(k)=Ak^\alpha .
$$

The capital nullcline is

$$
\dot{k}=0
\quad\Longleftrightarrow\quad
c=f(k)-\delta k .
$$

The consumption nullcline is

$$
\dot{c}=0
\quad\Longleftrightarrow\quad
f'(k)=\rho+\delta
\quad\Longleftrightarrow\quad
k=k^{*}
=\left(\frac{\alpha A}{\rho+\delta}\right)^{1/(1-\alpha)} .
$$

Steady-state consumption is $c^{*}=f(k^{*})-\delta k^{*}$. The boundary
condition selecting the planner's path is transversality:

$$
\lim_{t\to\infty} e^{-\rho t}u'(c(t))k(t)=0 .
$$
""")

    report.add_model_setup(
        "The calibration keeps the mechanics visible. Output is Cobb-Douglas. "
        "Preferences are CRRA. There are no shocks, so each arrow shows the same law "
        "of motion.\n\n"
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| $\\alpha$ | {alpha:.2f} | Capital share |\n"
        f"| $\\delta$ | {delta:.2f} | Depreciation rate |\n"
        f"| $\\rho$ | {rho:.2f} | Continuous-time discount rate |\n"
        f"| $\\sigma$ | {sigma:.1f} | CRRA coefficient and inverse EIS |\n"
        f"| $A$ | {A:.1f} | Total factor productivity |\n"
        f"| $k^{{*}}$ | {k_ss:.4f} | Ramsey steady-state capital |\n"
        f"| $c^{{*}}$ | {c_ss:.4f} | Ramsey steady-state consumption |"
    )

    report.add_solution_method(
        "The steady state anchors the stable-arm calculation. "
        "The Jacobian of $(\\dot{k},\\dot{c})$ at $(k^{*},c^{*})$ is\n\n"
        "$$\n"
        "J=\n"
        "\\begin{bmatrix}\n"
        "f'(k^{*})-\\delta & -1 \\\\\n"
        "c^{*}f''(k^{*})/\\sigma & 0\n"
        "\\end{bmatrix}.\n"
        "$$\n\n"
        f"The eigenvalues are $\\lambda_s={lambda_stable:.4f}$ and "
        f"$\\lambda_u={lambda_unstable:.4f}$. One is negative and one is positive. "
        "Nearby paths split into stable and unstable directions. "
        f"The stable eigenvector has local slope $dc/dk={slope:.4f}$. "
        "That line is only local. To draw the nonlinear arm, the code starts near "
        "the steady state and integrates backward.\n\n"
        "```text\n"
        "Algorithm: trace the Ramsey stable arm\n"
        "Inputs: primitives (alpha, delta, rho, sigma, A), bounds for plotted k and c\n"
        "1. Compute (k*, c*) from f'(k*) = rho + delta and c* = f(k*) - delta k*.\n"
        "2. Form the Jacobian J of F(k,c) = (dot{k}, dot{c}) at (k*, c*).\n"
        "3. Let lambda_s < 0 and v_s = (1, m_s) be the stable eigenpair.\n"
        "4. Start just below and just above (k*, c*) along v_s.\n"
        "5. Integrate d(k,c)/d tau = -F(k,c) away from the steady state.\n"
        "6. Stop when a branch leaves the plotted economic region.\n"
        "7. Sort the branches by k and read c(k) as the selected initial consumption rule.\n"
        "Output: nullclines, local linear arm, nonlinear stable arm, and sample forward paths.\n"
        "```\n\n"
        "Backward integration is a plotting device. In economic time, points on the arm "
        "converge to the steady state. Points above or below it miss transversality."
    )

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
    report.add_figure(
        "figures/phase-diagram.png",
        "Ramsey phase plane with nullclines, local linear arm, and nonlinear stable arm",
        fig1,
        description=(
            "The blue curve is the capital nullcline. The red line is the consumption "
            "nullcline. Below the blue curve, capital rises. Left of $k^{*}$, consumption "
            "rises because the marginal product is high. The black curve is the stable "
            "arm. For each capital stock shown, it gives the initial consumption that "
            "reaches the steady state. The dashed line is the local linear approximation."
        ),
    )

    fig3, ax3 = plt.subplots(figsize=(9, 6.5))
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
    report.add_figure(
        "figures/path-selection.png",
        "Forward trajectories from the same initial capital but different initial consumption",
        fig3,
        description=(
            "Holding initial capital fixed shows the selection problem. A higher "
            "consumption choice starts above the stable arm and runs capital down. "
            "A lower choice starts below it and builds too much capital. Arrows explain "
            "motion, but the stable arm selects the path."
        ),
    )

    report.add_takeaway(
        "The phase diagram shows direction. It does not choose initial consumption. "
        "The transversality condition selects the stable arm. Local linearization gives "
        "the slope near the steady state. Backward integration draws the nonlinear path "
        "away from it."
    )

    report.add_references([
        "Ramsey, F. (1928). \"A Mathematical Theory of Saving.\" *Economic Journal*, 38(152).",
        "Barro, R. and Sala-i-Martin, X. (2004). *Economic Growth*. MIT Press, 2nd edition, Ch. 2.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
