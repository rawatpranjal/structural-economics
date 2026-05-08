#!/usr/bin/env python3
"""Scalar root finding: bisection and Newton-Raphson on a market-clearing condition.

The capital market in a deterministic Cobb-Douglas economy clears when the
marginal product of capital equals the user cost. That is a scalar equation
in $k$ whose root has a closed form, so the two basic root finders can be
checked directly against the exact answer.

Reference: Mukoyama, T. (2021), Basic Numerical Methods, ECON 606, Georgetown.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def main() -> None:
    # =========================================================================
    # Calibration: Cobb-Douglas steady-state capital market
    # =========================================================================
    alpha = 0.36
    beta = 0.96
    delta = 0.08
    r_bar = 1.0 / beta - 1.0 + delta   # user cost of capital

    def F(k):
        return alpha * np.power(k, alpha - 1.0) - r_bar

    def Fprime(k):
        return alpha * (alpha - 1.0) * np.power(k, alpha - 2.0)

    # Closed-form root of F(k) = 0
    k_star = (alpha / r_bar) ** (1.0 / (1.0 - alpha))

    tol = 1e-10
    max_iter = 200

    # =========================================================================
    # Method 1: Bisection on a sign-change bracket
    # =========================================================================
    a0, b0 = 0.1, 30.0
    fa0, fb0 = F(a0), F(b0)
    assert fa0 * fb0 < 0.0, "Initial bracket must change sign"

    a, b = a0, b0
    fa = fa0
    bisection_rows = []
    for n in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = F(m)
        bisection_rows.append((n, a, b, m, abs(m - k_star)))
        if abs(fm) < tol or 0.5 * (b - a) < tol:
            break
        if fa * fm < 0.0:
            b = m
        else:
            a = m
            fa = fm
    bisection_history = np.array(bisection_rows, dtype=float)
    bisection_root = float(bisection_history[-1, 3])
    bisection_iter = int(bisection_history[-1, 0])
    bisection_residual = float(F(bisection_root))

    # =========================================================================
    # Method 2: Newton-Raphson with a moderate starting guess
    # =========================================================================
    x0 = 1.0
    newton_rows = [(0, x0, abs(x0 - k_star))]
    x = x0
    for n in range(1, max_iter + 1):
        fx = F(x)
        if abs(fx) < tol:
            break
        x = x - fx / Fprime(x)
        newton_rows.append((n, x, abs(x - k_star)))
    newton_history = np.array(newton_rows, dtype=float)
    newton_root = float(newton_history[-1, 1])
    newton_iter = int(newton_history[-1, 0])
    newton_residual = float(F(newton_root))

    # =========================================================================
    # Sensitivity: Newton iteration count vs starting point
    # Newton can step into k <= 0 from a far-above start, producing a NaN
    # residual. We mark those runs as diverged and report them separately.
    # =========================================================================
    starting_points = np.array([0.05, 0.2, 0.5, 1.0, 2.0, k_star, 8.0, 15.0, 25.0])
    newton_counts = []
    newton_status = []      # "converged" or "diverged"
    bisection_counts = []
    for x0_ in starting_points:
        x = float(x0_)
        status = "diverged"
        n = max_iter
        for n in range(1, max_iter + 1):
            if x <= 0.0 or not np.isfinite(x):
                status = "diverged"
                break
            with np.errstate(invalid="ignore"):
                fx = F(x)
            if not np.isfinite(fx):
                status = "diverged"
                break
            if abs(fx) < tol:
                status = "converged"
                break
            x = x - fx / Fprime(x)
        newton_counts.append(n)
        newton_status.append(status)

        # Bisection cost depends only on the bracket width, but for a fair
        # comparison we centre a width-2 bracket on the same starting point
        # and fall back to the global bracket when no sign change is present.
        a_ = max(1e-3, float(x0_) - 1.0)
        b_ = float(x0_) + 1.0
        if F(a_) * F(b_) >= 0.0:
            a_, b_ = a0, b0
        fa_ = F(a_)
        for n in range(1, max_iter + 1):
            m_ = 0.5 * (a_ + b_)
            fm_ = F(m_)
            if abs(fm_) < tol or 0.5 * (b_ - a_) < tol:
                break
            if fa_ * fm_ < 0.0:
                b_ = m_
            else:
                a_ = m_
                fa_ = fm_
        bisection_counts.append(n)

    newton_counts = np.array(newton_counts)
    bisection_counts = np.array(bisection_counts)
    n_diverged = sum(1 for s in newton_status if s == "diverged")

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Scalar Root Finding for Market Clearing",
        "Bisection and Newton-Raphson find the steady-state capital stock that clears a Cobb-Douglas capital market.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A representative-firm economy reaches a steady state when the net "
        "marginal product of capital equals the rate of time preference plus "
        "depreciation. With Cobb-Douglas production that condition is a "
        "scalar equation in $k$.\n\n"
        "The unknown is the steady-state capital stock $k^{\\ast}$. The "
        "equation has a closed form, so any solver can be checked directly. "
        "This is the cleanest setup for comparing two basic root finders.\n\n"
        "Bisection halves a sign-change bracket. Newton-Raphson extrapolates "
        "the tangent line at the current iterate. The first is globally safe "
        "and slow; the second is locally fast and depends on the starting "
        "point."
    )

    report.add_equations(
        r"""
The capital market clears when the marginal product of capital nets out
the user cost:

$$F'(k^{\ast}) = \tfrac{1}{\beta} - 1 + \delta.$$

For Cobb-Douglas technology $f(k) = k^{\alpha}$ this is the scalar equation

$$F(k) = \alpha\, k^{\alpha - 1} - \bar{r} = 0,
\qquad \bar{r} \equiv \tfrac{1}{\beta} - 1 + \delta.$$

The closed-form root is

$$k^{\ast} = \left( \frac{\alpha}{\bar{r}} \right)^{\frac{1}{1 - \alpha}}.$$

Bisection halves a sign-change bracket $[a, b]$ around the root:

$$m_n = \tfrac{a_n + b_n}{2},
\qquad \mathrm{sign}\, F(a_{n+1}) \neq \mathrm{sign}\, F(b_{n+1}),
\qquad b_{n+1} - a_{n+1} = \tfrac{1}{2}(b_n - a_n).$$

Newton-Raphson follows the tangent line at the current iterate:

$$x_{n+1} = x_n - \frac{F(x_n)}{F'(x_n)}.$$

Near a simple root, the bracket of bisection contracts at a linear rate of
$\tfrac{1}{2}$ per step, while the Newton residual squares each step.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $\\alpha$ | {alpha} | Capital share in Cobb-Douglas production |\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $\\bar{{r}}$ | {r_bar:.4f} | User cost $1/\\beta - 1 + \\delta$ |\n"
        f"| $k^{{\\ast}}$ | {k_star:.4f} | Closed-form steady-state capital |\n"
        f"| Bracket $[a_0, b_0]$ | $[{a0},\\, {b0}]$ | Initial sign-change bracket for bisection |\n"
        f"| Newton start $x_0$ | {x0} | Starting iterate for Newton-Raphson |\n"
        f"| Tolerance $\\varepsilon$ | {tol:.0e} | Stopping rule on the residual and the bracket width |"
    )

    report.add_solution_method(
        "Both methods solve the same scalar equation $F(k) = 0$. Bisection "
        "needs only a sign-change bracket; Newton-Raphson needs the "
        "derivative and a single starting point.\n\n"
        "```text\n"
        "Bisection                       | Newton-Raphson\n"
        "Input: a, b with F(a)F(b) < 0   | Input: x_0, tolerance eps\n"
        "       tolerance eps            |        F, F'\n"
        "for n = 1, 2, ... :             | for n = 0, 1, ... :\n"
        "    m <- (a + b) / 2            |     x_{n+1} <- x_n - F(x_n) / F'(x_n)\n"
        "    fm <- F(m)                  |     stop when |F(x_n)| < eps\n"
        "    if |fm| < eps: stop         |\n"
        "    if F(a) * fm < 0: b <- m    |\n"
        "    else            : a <- m    |\n"
        "    stop when (b - a) < eps     |\n"
        "```\n\n"
        f"Starting from the bracket $[{a0},\\, {b0}]$, bisection converges in "
        f"**{bisection_iter} iterations** with residual $|F(k)|$ = "
        f"**{abs(bisection_residual):.2e}**. Starting from $x_0 = {x0}$, "
        f"Newton-Raphson converges in **{newton_iter} iterations** with "
        f"residual $|F(k)|$ = **{abs(newton_residual):.2e}**."
    )

    # ------------------------------------------------------------------
    # Figure 1: F(k) and the bracket shrinking around k*
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    k_plot = np.linspace(0.5, 15.0, 400)
    ax1.plot(k_plot, F(k_plot), color="tab:blue", linewidth=2, label=r"$F(k) = \alpha k^{\alpha-1} - \bar{r}$")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.axvline(k_star, color="tab:red", linestyle="--", linewidth=1.5, label=fr"$k^{{\ast}} = {k_star:.3f}$")

    # Overlay the first few bisection brackets
    n_show = min(6, len(bisection_history))
    for i in range(n_show):
        n_, a_, b_, m_, _ = bisection_history[i]
        y = -0.05 - 0.02 * i
        ax1.plot([a_, b_], [y, y], color="tab:orange", linewidth=2, alpha=0.7,
                 label="Bisection bracket" if i == 0 else None)
        ax1.plot([m_], [y], "o", color="tab:orange", markersize=4)
    ax1.set_xlabel(r"Capital stock $k$")
    ax1.set_ylabel(r"$F(k)$")
    ax1.set_title(r"Market-clearing residual and bisection brackets")
    ax1.legend()
    report.add_results(
        "The residual $F(k)$ is monotone and crosses zero exactly once at "
        f"$k^{{\\ast}} = {k_star:.3f}$. The first six bisection brackets, drawn "
        "below the curve, halve in width each iteration while keeping the "
        "sign change."
    )
    report.add_figure(
        "figures/residual-and-bracket.png",
        "Market-clearing residual $F(k)$ with the first bisection brackets contracting to the root",
        fig1,
    )

    # ------------------------------------------------------------------
    # Figure 2: convergence (error vs iteration, log axis)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.semilogy(bisection_history[:, 0], np.maximum(bisection_history[:, 4], 1e-16),
                 "o-", color="tab:blue", markersize=3, linewidth=1.5, label="Bisection")
    ax2.semilogy(newton_history[:, 0], np.maximum(newton_history[:, 2], 1e-16),
                 "s-", color="tab:red", markersize=4, linewidth=1.5, label="Newton-Raphson")
    ax2.set_xlabel("Iteration $n$")
    ax2.set_ylabel(r"$|x_n - k^{\ast}|$")
    ax2.set_title("Convergence to the steady-state root")
    ax2.legend()
    report.add_results(
        "Both methods reach the closed-form root, but at different rates. "
        f"Bisection halves its error each step and needs **{bisection_iter} "
        "iterations** to hit the tolerance. Newton-Raphson squares its "
        f"residual once near the root and needs only **{newton_iter} "
        "iterations** from the same problem. The Newton curve drops off a "
        "cliff once the iterate enters the basin of quadratic convergence."
    )
    report.add_figure(
        "figures/convergence.png",
        "Distance from the closed-form root vs iteration, log axis",
        fig2,
    )

    # ------------------------------------------------------------------
    # Figure 3: sensitivity to starting point
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots()
    width = 0.35
    idx = np.arange(len(starting_points))
    ax3.bar(idx - width / 2, bisection_counts, width, color="tab:blue", label="Bisection")
    newton_colors = ["tab:red" if s == "converged" else "lightgray" for s in newton_status]
    newton_hatches = ["" if s == "converged" else "//" for s in newton_status]
    bars = ax3.bar(idx + width / 2, newton_counts, width, color=newton_colors,
                   edgecolor="tab:red", label="Newton-Raphson")
    for bar, hatch in zip(bars, newton_hatches):
        bar.set_hatch(hatch)
    for i, status in enumerate(newton_status):
        if status == "diverged":
            ax3.text(idx[i] + width / 2, newton_counts[i] + 2, "DNC",
                     ha="center", va="bottom", color="tab:red", fontsize=8)
    ax3.set_xticks(idx)
    ax3.set_xticklabels([f"{x:.2f}" for x in starting_points], rotation=45)
    ax3.set_xlabel(r"Starting capital stock $x_0$")
    ax3.set_ylabel("Iterations to convergence")
    ax3.set_title(r"Iteration count vs starting point (tolerance $10^{-10}$)")
    ax3.legend()
    fig3.tight_layout()
    report.add_results(
        "Bisection iteration counts are flat across starting points: each "
        "step halves the bracket regardless of where it is centred. Newton "
        "iteration counts climb when the start is far below $k^{\\ast}$, "
        "where the tangent step is small relative to the gap, and collapse "
        f"near the root, where quadratic convergence kicks in. From "
        f"**{n_diverged} of {len(starting_points)}** starting points above "
        "$k^{\\ast}$, the Newton step overshoots into $k < 0$ where the "
        "Cobb-Douglas residual is undefined and the iteration diverges "
        "(hatched bars marked DNC)."
    )
    report.add_figure(
        "figures/sensitivity.png",
        "Iterations to converge as a function of starting point for both methods",
        fig3,
    )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    table_data = {
        "Method": ["Bisection", "Newton-Raphson"],
        "Start": [f"[{a0}, {b0}]", f"x_0 = {x0}"],
        "Iterations": [bisection_iter, newton_iter],
        "Final residual": [f"{abs(bisection_residual):.2e}", f"{abs(newton_residual):.2e}"],
        "Error in k": [f"{abs(bisection_root - k_star):.2e}", f"{abs(newton_root - k_star):.2e}"],
        "Convergence rate": ["linear (1/2)", "quadratic"],
    }
    df = pd.DataFrame(table_data)
    report.add_results(
        "The table summarises the two solves on the same calibration. Both "
        "land within machine tolerance of the closed-form root."
    )
    report.add_table(
        "tables/comparison.csv",
        "Bisection vs Newton-Raphson on the steady-state capital market",
        df,
    )

    report.add_takeaway(
        "Bisection is the safe default when only a sign-change bracket is "
        "available: it halves the error every step and never leaves the "
        "bracket. Newton-Raphson is much faster once the iterate is near a "
        "simple root, because the tangent extrapolation squares the residual "
        "each step. The trade-off shows up at large $x_0$, where the Newton "
        "step here overshoots into $k < 0$ and the Cobb-Douglas residual "
        "becomes undefined. Aiyagari- and Huggett-style interest-rate solves "
        "later in the catalog use bisection on $r$ for exactly this reason."
    )

    report.add_references([
        "Mukoyama, T. (2021). *Basic Numerical Methods*. ECON 606 lecture slides, Georgetown University.",
        "Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 9.",
        "Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 5.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
