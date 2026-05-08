#!/usr/bin/env python3
"""Scalar optimization: golden section search and Newton on the FOC.

The per-state inner Bellman maximization in cake-eating, $\\max_{c \\in [0, W]}
u(c) + \\beta V(W - c)$, is a one-dimensional optimization on a bounded
interval. With log utility and the closed-form value function the inner
optimum has a known analytic answer, so the two basic 1D optimizers can be
checked directly against the exact $c^{\\ast} = (1 - \\beta) W$.

Reference: Mukoyama, T. (2021), Basic Numerical Methods, ECON 606, Georgetown.
"""
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


def main() -> None:
    # =========================================================================
    # Calibration: cake-eating inner step at a single wealth level
    # =========================================================================
    beta = 0.9
    W = 1.0
    K = beta / (1.0 - beta)        # FOC ratio of marginal-utility weights

    # Closed-form continuation value V(W) under log utility
    def V(w):
        return np.log((1.0 - beta) * w) / (1.0 - beta) + beta * np.log(beta) / (1.0 - beta) ** 2

    # Inner Bellman objective and its first two derivatives
    def g(c):
        return np.log(c) + beta * V(W - c)

    def gp(c):
        return 1.0 / c - K / (W - c)

    def gpp(c):
        return -1.0 / c ** 2 - K / (W - c) ** 2

    c_star = (1.0 - beta) * W      # closed-form inner argmax

    tol = 1e-10
    max_iter = 200

    # =========================================================================
    # Method 1: Golden section search on a unimodal bracket
    # =========================================================================
    phi = (math.sqrt(5.0) - 1.0) / 2.0    # 0.618...

    a0, b0 = 1e-3, W - 1e-3
    a, b = a0, b0
    cL = b - phi * (b - a)
    cR = a + phi * (b - a)
    fL, fR = g(cL), g(cR)
    golden_rows = [(0, a, b, 0.5 * (a + b), abs(0.5 * (a + b) - c_star))]
    for n in range(1, max_iter + 1):
        if fL > fR:
            b = cR
            cR = cL
            fR = fL
            cL = b - phi * (b - a)
            fL = g(cL)
        else:
            a = cL
            cL = cR
            fL = fR
            cR = a + phi * (b - a)
            fR = g(cR)
        midpoint = 0.5 * (a + b)
        golden_rows.append((n, a, b, midpoint, abs(midpoint - c_star)))
        if (b - a) < tol:
            break
    golden_history = np.array(golden_rows, dtype=float)
    golden_root = float(golden_history[-1, 3])
    golden_iter = int(golden_history[-1, 0])
    golden_residual = float(abs(gp(golden_root)))

    # =========================================================================
    # Method 2: Newton on the FOC, starting from a feasible guess below c*
    # =========================================================================
    x0 = 0.05
    newton_rows = [(0, x0, abs(x0 - c_star))]
    x = x0
    for n in range(1, max_iter + 1):
        gpx = gp(x)
        if abs(gpx) < tol:
            break
        x_new = x - gpx / gpp(x)
        if not (1e-12 < x_new < W - 1e-12) or not np.isfinite(x_new):
            # Step would leave the feasible interior; record and stop.
            newton_rows.append((n, x_new, float("nan")))
            break
        x = x_new
        newton_rows.append((n, x, abs(x - c_star)))
    newton_history = np.array(newton_rows, dtype=float)
    newton_root = float(newton_history[-1, 1])
    newton_iter = int(newton_history[-1, 0])
    newton_residual = float(abs(gp(newton_root)))

    # =========================================================================
    # Sensitivity: Newton iteration count vs starting point
    # Newton step c - g'(c)/g''(c) can land outside (0, W) from a bad start;
    # we record divergence so the figure can mark the failure cases.
    # =========================================================================
    starting_points = np.array([0.02, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.50, 0.80])
    newton_counts = []
    newton_status = []
    golden_counts = []
    for x0_ in starting_points:
        x = float(x0_)
        status = "diverged"
        n = max_iter
        for n in range(1, max_iter + 1):
            if not (1e-12 < x < W - 1e-12) or not np.isfinite(x):
                status = "diverged"
                break
            with np.errstate(invalid="ignore", divide="ignore"):
                gpx = gp(x)
                gppx = gpp(x)
            if not np.isfinite(gpx) or not np.isfinite(gppx):
                status = "diverged"
                break
            if abs(gpx) < tol:
                status = "converged"
                break
            x = x - gpx / gppx
        newton_counts.append(n)
        newton_status.append(status)

        # Golden section: contract [eps, W-eps] regardless of x0_ (the bracket
        # does not depend on a starting point); same iteration count for all.
        a_, b_ = a0, b0
        cL_ = b_ - phi * (b_ - a_)
        cR_ = a_ + phi * (b_ - a_)
        fL_, fR_ = g(cL_), g(cR_)
        for n in range(1, max_iter + 1):
            if fL_ > fR_:
                b_ = cR_
                cR_ = cL_
                fR_ = fL_
                cL_ = b_ - phi * (b_ - a_)
                fL_ = g(cL_)
            else:
                a_ = cL_
                cL_ = cR_
                fL_ = fR_
                cR_ = a_ + phi * (b_ - a_)
                fR_ = g(cR_)
            if (b_ - a_) < tol:
                break
        golden_counts.append(n)

    newton_counts = np.array(newton_counts)
    golden_counts = np.array(golden_counts)
    n_diverged = sum(1 for s in newton_status if s == "diverged")

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "One-Dimensional Optimization for Bellman Inner Steps",
        "Golden section search and Newton-on-FOC find the per-state cake-eating maximum, with a log-utility closed-form check.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The inner step in cake-eating value function iteration is a "
        "one-dimensional maximization on a bounded interval.\n\n"
        "$$\\max_{c \\in [0, W]} u(c) + \\beta\\, V(W - c).$$\n\n"
        "Under log utility the closed-form inner optimum is "
        "$c^{\\ast} = (1 - \\beta) W$.\n\n"
        "Golden section search contracts a bracket $[a, b]$ around the maximum.\n\n"
        "Newton on the first-order condition extrapolates a quadratic "
        "surrogate at the current iterate.\n\n"
        "Golden section needs only unimodality and is globally safe.\n\n"
        "Newton is locally fast and depends on the starting point."
    )

    report.add_equations(
        r"""
Let $u(c) = \log c$ and let $V$ be the closed-form cake-eating value
function

$$V(W) = \frac{\log((1-\beta) W)}{1-\beta} + \frac{\beta \log \beta}{(1-\beta)^2}.$$

The inner Bellman objective is

$$g(c) = \log c + \beta\, V(W - c),$$

with derivatives

$$g'(c) = \frac{1}{c} - \frac{\beta}{1 - \beta}\, \frac{1}{W - c},
\qquad
g''(c) = -\frac{1}{c^2} - \frac{\beta}{1 - \beta}\, \frac{1}{(W - c)^2} < 0.$$

The objective is strictly concave on $(0, W)$, so the maximum is unique.
Setting $g'(c) = 0$ gives the closed-form policy

$$c^{\ast} = (1 - \beta) W.$$

Golden section search shrinks a bracket $[a, b]$ around the maximum using
$\phi = (\sqrt{5} - 1)/2 \approx 0.618$:

$$c_n = b_n - \phi (b_n - a_n),
\qquad d_n = a_n + \phi (b_n - a_n).$$

The next bracket keeps whichever endpoint is closer to the larger of
$g(c_n)$ and $g(d_n)$, contracting the width by a factor $\phi$ each step.

Newton on the FOC follows the tangent of $g'$ at the current iterate:

$$x_{n+1} = x_n - \frac{g'(x_n)}{g''(x_n)}.$$

Equivalently, Newton maximizes a parabolic surrogate
$q_n(c) = g(x_n) + g'(x_n)(c - x_n) + \tfrac{1}{2} g''(x_n)(c - x_n)^2$.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $\\beta$ | {beta} | Discount factor; closed-form inner share is $1 - \\beta$ |\n"
        f"| $W$ | {W} | Wealth at the inner state being optimized |\n"
        f"| $c^{{\\ast}}$ | {c_star:.4f} | Closed-form inner argmax $(1 - \\beta) W$ |\n"
        f"| Bracket $[a_0, b_0]$ | $[{a0:.0e},\\, {b0:.4f}]$ | Initial unimodal bracket for golden section |\n"
        f"| Newton start $x_0$ | {x0} | Starting iterate for Newton-on-FOC |\n"
        f"| Tolerance $\\varepsilon$ | {tol:.0e} | Stopping rule on bracket width and on $g'(x_n)$ |"
    )

    report.add_solution_method(
        "Both methods solve the same maximization. Golden section needs "
        "only that $g$ is unimodal on the bracket. Newton on the FOC needs "
        "$g'$ and $g''$.\n\n"
        "**Golden section search.** Contract a bracket using the golden "
        "ratio so one interior point is reused each step.\n\n"
        "```text\n"
        "Algorithm: Golden section search\n"
        "Input : a, b with g unimodal on [a, b]; tolerance eps\n"
        "Output: c_n\n"
        "  phi <- (sqrt(5) - 1) / 2\n"
        "  c   <- b - phi (b - a)\n"
        "  d   <- a + phi (b - a)\n"
        "  for n = 1, 2, ... :\n"
        "      if g(c) > g(d): b <- d\n"
        "      else          : a <- c\n"
        "      recompute c, d\n"
        "      stop when (b - a) < eps\n"
        "```\n\n"
        "**Newton on the FOC.** Step along the tangent of $g'$ at the "
        "current iterate. Equivalently, jump to the argmax of a parabolic "
        "surrogate of $g$.\n\n"
        "```text\n"
        "Algorithm: Newton on FOC\n"
        "Input : x_0; tolerance eps; g', g''\n"
        "Output: x_n\n"
        "  for n = 0, 1, ... :\n"
        "      x_{n+1} <- x_n - g'(x_n) / g''(x_n)\n"
        "      stop when |g'(x_n)| < eps\n"
        "```\n\n"
        f"Starting from the bracket $[{a0:.0e},\\, {b0:.4f}]$, golden "
        f"section converges in **{golden_iter} iterations** with residual "
        f"$|g'(c)| =$ **{golden_residual:.2e}**. From $x_0 = {x0}$ Newton "
        f"converges in **{newton_iter} iterations** with residual "
        f"**{newton_residual:.2e}**."
    )

    # ------------------------------------------------------------------
    # Figure 1: g(c) with shrinking golden-section brackets
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    c_plot = np.linspace(0.005, W - 0.005, 400)
    ax1.plot(c_plot, g(c_plot), color="tab:blue", linewidth=2, label=r"$g(c) = \log c + \beta V(W - c)$")
    ax1.axvline(c_star, color="tab:red", linestyle="--", linewidth=1.5, label=fr"$c^{{\ast}} = {c_star:.3f}$")

    n_show = min(6, len(golden_history))
    g_min = float(np.min(g(c_plot)))
    g_span = float(np.max(g(c_plot))) - g_min
    base = g_min - 0.05 * g_span
    for i in range(n_show):
        n_, a_, b_, m_, _ = golden_history[i]
        y = base - 0.05 * i * g_span
        ax1.plot([a_, b_], [y, y], color="tab:orange", linewidth=2, alpha=0.7,
                 label="Golden bracket" if i == 0 else None)
        ax1.plot([m_], [y], "o", color="tab:orange", markersize=4)
    ax1.set_xlabel("Inner control $c$")
    ax1.set_ylabel(r"$g(c)$")
    ax1.set_title("Inner Bellman objective and golden-section brackets")
    ax1.legend(loc="lower right")
    report.add_results(
        f"The inner objective $g(c)$ peaks at $c^{{\\ast}} = {c_star:.3f}$.\n\n"
        "The first six golden-section brackets sit below the curve. Each "
        "step contracts the bracket by a factor $\\phi \\approx 0.618$ "
        "while keeping the maximum inside."
    )
    report.add_figure(
        "figures/golden-section-trace.png",
        "Cake-eating inner objective $g(c)$ with the first golden-section brackets contracting around $c^{\\ast}$",
        fig1,
    )

    # ------------------------------------------------------------------
    # Figure 2: Newton parabolic surrogates at successive iterates
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    c_zoom = np.linspace(0.02, 0.5, 400)
    ax2.plot(c_zoom, g(c_zoom), color="tab:blue", linewidth=2, label=r"$g(c)$")
    colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown"]
    for i, (niter, xi, _err) in enumerate(newton_history[:4]):
        gi, gpi, gppi = g(xi), gp(xi), gpp(xi)
        q = gi + gpi * (c_zoom - xi) + 0.5 * gppi * (c_zoom - xi) ** 2
        ni = int(niter)
        ax2.plot(c_zoom, q, color=colors[i % len(colors)], linestyle=":", linewidth=1.2,
                 label=fr"$q_{{{ni}}}(c)$ at $x_{{{ni}}} = {xi:.3f}$")
        ax2.plot([xi], [gi], "o", color=colors[i % len(colors)], markersize=5)
    ax2.axvline(c_star, color="tab:red", linestyle="--", linewidth=1.0, label=fr"$c^{{\ast}} = {c_star:.3f}$")
    ax2.set_xlabel("Inner control $c$")
    ax2.set_ylabel(r"$g(c)$ and Newton surrogates")
    ax2.set_title("Newton parabolic surrogates at successive iterates")
    ax2.legend(loc="lower right", fontsize=8)
    ymin = float(g(c_zoom).min())
    ymax = float(g(c_star)) + 0.05 * (float(g(c_star)) - ymin)
    ax2.set_ylim(ymin, ymax)
    report.add_results(
        "Each Newton iterate defines a parabolic surrogate $q_n$ that "
        "matches $g$ in value, slope, and curvature.\n\n"
        f"Newton jumps to the argmax of $q_n$. From $x_0 = {x0}$ the "
        "iterates close in on $c^{\\ast}$ within a handful of steps."
    )
    report.add_figure(
        "figures/newton-step.png",
        "Newton parabolic surrogates at the first iterates of the inner Bellman maximization",
        fig2,
    )

    # ------------------------------------------------------------------
    # Figure 3: convergence and Newton sensitivity (two panels)
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    ax3a.semilogy(golden_history[:, 0], np.maximum(golden_history[:, 4], 1e-16),
                  "o-", color="tab:blue", markersize=3, linewidth=1.5, label="Golden section")
    ax3a.semilogy(newton_history[:, 0], np.maximum(newton_history[:, 2], 1e-16),
                  "s-", color="tab:red", markersize=4, linewidth=1.5, label="Newton on FOC")
    ax3a.set_xlabel("Iteration $n$")
    ax3a.set_ylabel(r"$|x_n - c^{\ast}|$")
    ax3a.set_title("Convergence to the inner argmax")
    ax3a.legend()

    width = 0.35
    idx = np.arange(len(starting_points))
    ax3b.bar(idx - width / 2, golden_counts, width, color="tab:blue", label="Golden section")
    newton_colors = ["tab:red" if s == "converged" else "lightgray" for s in newton_status]
    newton_hatches = ["" if s == "converged" else "//" for s in newton_status]
    bars = ax3b.bar(idx + width / 2, newton_counts, width, color=newton_colors,
                    edgecolor="tab:red", label="Newton on FOC")
    for bar, hatch in zip(bars, newton_hatches):
        bar.set_hatch(hatch)
    for i, status in enumerate(newton_status):
        if status == "diverged":
            ax3b.text(idx[i] + width / 2, newton_counts[i] + 2, "DNC",
                      ha="center", va="bottom", color="tab:red", fontsize=8)
    ax3b.set_xticks(idx)
    ax3b.set_xticklabels([f"{x:.2f}" for x in starting_points], rotation=45)
    ax3b.set_xlabel(r"Starting iterate $x_0$")
    ax3b.set_ylabel("Iterations to convergence")
    ax3b.set_title(r"Iteration count vs starting point (tolerance $10^{-10}$)")
    ax3b.legend()
    fig3.tight_layout()
    report.add_results(
        f"Golden section needs **{golden_iter} bracket halvings** and "
        f"Newton needs only **{newton_iter} steps** from the same "
        "calibration.\n\n"
        "The right panel shows the trade-off. Golden-section counts are "
        "flat across starting points: the bracket is the same. Newton "
        f"counts depend on $x_0$, and **{n_diverged} of "
        f"{len(starting_points)}** starts overshoot outside $(0, W)$ and "
        "diverge (hatched bars marked DNC)."
    )
    report.add_figure(
        "figures/convergence.png",
        "Distance from the closed-form inner argmax (left) and Newton sensitivity to starting point (right)",
        fig3,
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    table_data = {
        "Method": ["Golden section", "Newton on FOC"],
        "Start": [f"[{a0:.0e}, {b0:.3f}]", f"x_0 = {x0}"],
        "Iterations": [golden_iter, newton_iter],
        "Final residual": [f"{golden_residual:.2e}", f"{newton_residual:.2e}"],
        "Error in c": [f"{abs(golden_root - c_star):.2e}", f"{abs(newton_root - c_star):.2e}"],
        "Convergence rate": ["linear (phi)", "quadratic"],
    }
    df = pd.DataFrame(table_data)
    report.add_results(
        "The table summarises both solves on the same calibration. Both land "
        "within the chosen tolerance of the closed-form inner argmax."
    )
    report.add_table(
        "tables/comparison.csv",
        "Golden section vs Newton-on-FOC on the cake-eating inner step",
        df,
    )

    report.add_takeaway(
        "Golden section is the safe default for a one-state Bellman inner "
        "step: it only needs unimodality and a bracket, and contracts at "
        "a fixed factor regardless of where the optimum sits.\n\n"
        "Newton on the FOC is much faster when $g'$ and $g''$ are "
        "available and $x_0$ is inside the basin of attraction.\n\n"
        "A far-off start makes the parabolic extrapolation overshoot "
        "outside the feasible interval.\n\n"
        "Cake-eating and consumption-savings VFI use golden section for "
        "this reason. Smooth problems with cheap derivatives can graduate "
        "to Newton."
    )

    report.add_references([
        "Mukoyama, T. (2021). *Basic Numerical Methods*. ECON 606 lecture slides, Georgetown University.",
        "Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 10.",
        "Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 4.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
