#!/usr/bin/env python3
"""Scalar root finding: bisection, secant, Brent, and Newton-Raphson on Z(r) = 0.

A stylized one-bond market clears at the rate that sets aggregate excess
demand to zero. With Cobb-Douglas firm-side demand and target supply set at
the deterministic Aiyagari rate, the clearing condition is a scalar
equation in $r$ whose root has a closed form. The four basic root finders
can then be compared directly.

Reference: Mukoyama, T. (2021), Basic Numerical Methods, ECON 606, Georgetown.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


# =============================================================================
# Method implementations (each records its iterate trace)
# =============================================================================

def bisection(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("Bracket must change sign")
    history = [(0, 0.5 * (a + b), 0.5 * (a + b), 0.5 * (b - a))]
    for n in range(1, max_iter + 1):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        history.append((n, m, fm, 0.5 * (b - a)))
        if abs(fm) < tol or 0.5 * (b - a) < tol:
            break
    return m, history


def secant(f, x0, x1, tol, max_iter):
    f0, f1 = f(x0), f(x1)
    history = [(0, x0, f0, abs(x0 - x1)), (1, x1, f1, abs(x1 - x0))]
    for n in range(2, max_iter + 1):
        if abs(f1 - f0) < 1e-15:
            break
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        f_new = f(x_new)
        history.append((n, x_new, f_new, abs(x_new - x1)))
        x0, f0 = x1, f1
        x1, f1 = x_new, f_new
        if abs(f_new) < tol:
            break
    return x1, history


def brent_with_history(f, a, b, tol, max_iter):
    """Brent-Dekker with inverse quadratic interpolation; records iterates."""
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("Bracket must change sign")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c, fc = a, fa
    d = a
    mflag = True
    history = [(0, b, fb, 0.0)]
    for n in range(1, max_iter + 1):
        if abs(fa - fc) > 1e-14 and abs(fb - fc) > 1e-14:
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            s = b - fb * (b - a) / (fb - fa)
        lo = min((3.0 * a + b) / 4.0, b)
        hi = max((3.0 * a + b) / 4.0, b)
        cond1 = not (lo < s < hi)
        cond2 = mflag and abs(s - b) >= 0.5 * abs(b - c)
        cond3 = (not mflag) and abs(s - b) >= 0.5 * abs(c - d)
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol
        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d = c
        c, fc = b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        history.append((n, b, fb, abs(b - a)))
        if abs(fb) < tol or abs(b - a) < tol:
            break
    return b, history


def newton(f, fprime, x0, tol, max_iter, lo=None, hi=None):
    history = [(0, x0, f(x0), 0.0)]
    x = x0
    status = "converged"
    for n in range(1, max_iter + 1):
        fx = f(x)
        if abs(fx) < tol:
            break
        fpx = fprime(x)
        if abs(fpx) < 1e-15:
            status = "diverged"
            break
        x_new = x - fx / fpx
        if lo is not None and hi is not None:
            if not (lo < x_new < hi) or not np.isfinite(x_new):
                history.append((n, x_new, float("nan"), abs(x_new - x)))
                status = "diverged"
                break
        history.append((n, x_new, f(x_new), abs(x_new - x)))
        x = x_new
    return x, history, status


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # =========================================================================
    # Calibration (deterministic Aiyagari capital market)
    # =========================================================================
    alpha = 0.36
    beta = 0.96
    delta = 0.08
    r_star = 1.0 / beta - 1.0
    K_target = (alpha / (r_star + delta)) ** (1.0 / (1.0 - alpha))

    def Z(r):
        return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha)) - K_target

    def Zprime(r):
        return -(1.0 / (1.0 - alpha)) * (
            (alpha / (r + delta)) ** (1.0 / (1.0 - alpha)) / (r + delta)
        )

    tol = 1e-10
    max_iter = 200

    a0, b0 = 0.0 + 1e-6, 0.10
    bis_root, bis_hist = bisection(Z, a0, b0, tol, max_iter)
    sec_root, sec_hist = secant(Z, a0, b0, tol, max_iter)
    bre_root, bre_hist = brent_with_history(Z, a0, b0, tol, max_iter)
    newton_x0 = 0.02
    new_root, new_hist, _ = newton(
        Z, Zprime, newton_x0, tol, max_iter, lo=1e-9, hi=0.5
    )

    scipy_root = brentq(Z, a0, b0, xtol=tol)

    methods = [
        ("Bisection", bis_root, bis_hist, "linear (1/2)", "sign-change bracket"),
        ("Secant", sec_root, sec_hist, "superlinear (~1.618)", "two starting points"),
        ("Brent", bre_root, bre_hist, "superlinear", "sign-change bracket"),
        ("Newton-Raphson", new_root, new_hist, "quadratic", "x_0 and Z'"),
    ]
    histories = {
        name: np.array([(h[0], h[1], h[2]) for h in hist], dtype=float)
        for name, _, hist, _, _ in methods
    }

    # =========================================================================
    # Sensitivity to starting point / bracket
    # =========================================================================
    starts = np.array([0.001, 0.005, 0.01, 0.02, 0.03, r_star, 0.06, 0.08, 0.12])
    bis_counts, sec_counts, bre_counts, new_counts = [], [], [], []
    new_status_list = []
    for x0_ in starts:
        a_ = max(1e-6, float(x0_) - 0.02)
        b_ = float(x0_) + 0.02
        if Z(a_) * Z(b_) >= 0.0:
            a_, b_ = a0, b0
        _, h_b = bisection(Z, a_, b_, tol, max_iter)
        bis_counts.append(int(h_b[-1][0]))
        _, h_br = brent_with_history(Z, a_, b_, tol, max_iter)
        bre_counts.append(int(h_br[-1][0]))
        s0, s1 = max(1e-6, float(x0_) - 0.005), float(x0_) + 0.005
        try:
            _, h_s = secant(Z, s0, s1, tol, max_iter)
            sec_counts.append(int(h_s[-1][0]))
        except (ZeroDivisionError, OverflowError):
            sec_counts.append(max_iter)
        _, h_n, st = newton(Z, Zprime, float(x0_), tol, max_iter, lo=1e-9, hi=0.5)
        new_counts.append(int(h_n[-1][0]))
        new_status_list.append(st)
    bis_counts = np.array(bis_counts)
    sec_counts = np.array(sec_counts)
    bre_counts = np.array(bre_counts)
    new_counts = np.array(new_counts)
    n_diverged = sum(1 for s in new_status_list if s == "diverged")

    # =========================================================================
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Scalar Root Finding for Equilibrium Rates",
        "Bisection, secant, Brent, and Newton-Raphson recover the rate that clears a stylized bond market with a closed-form root.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A representative-firm economy with Cobb-Douglas production has a "
        "closed-form clearing rate $r^{\\ast} = 1/\\beta - 1$. "
        "The market-clearing condition is one scalar equation in $r$.\n\n"
        "Bisection halves a sign-change bracket. "
        "Secant fits a chord through the last two iterates. "
        "Brent combines bisection's bracket with inverse quadratic "
        "interpolation when the fast step stays inside. "
        "Newton-Raphson uses the analytic derivative.\n\n"
        "These are the four solvers behind $\\mathrm{scipy.optimize.brentq}$ "
        "and the equilibrium clearings in Aiyagari and Huggett."
    )

    report.add_equations(
        r"""
Aggregate capital demand from the firm-side first-order condition is

$$K_d(r) = \left( \frac{\alpha}{r + \delta} \right)^{\frac{1}{1 - \alpha}}.$$

Target supply is $K^{\ast} \equiv K_d(r^{\ast})$ at $r^{\ast} = 1/\beta - 1$.
Excess demand is then

$$Z(r) = K_d(r) - K^{\ast}, \qquad Z(r^{\ast}) = 0.$$

The derivative used by Newton is

$$Z'(r) = -\frac{1}{1 - \alpha} \frac{K_d(r)}{r + \delta} < 0.$$

Bisection halves a sign-change bracket:

$$m_n = \frac{a_n + b_n}{2}, \qquad b_{n+1} - a_{n+1} = \frac{1}{2}(b_n - a_n).$$

Secant fits a chord through the last two iterates:

$$x_{n+1} = x_n - Z(x_n) \frac{x_n - x_{n-1}}{Z(x_n) - Z(x_{n-1})}.$$

Newton-Raphson follows the tangent:

$$x_{n+1} = x_n - \frac{Z(x_n)}{Z'(x_n)}.$$

Brent's method tries inverse quadratic interpolation through the last
three ordinates, falls back to secant when ordinates coincide, and
falls back to bisection when the proposed step would leave the bracket
or fails to halve the previous step. The bracket invariant is
maintained at every iteration.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $\\alpha$ | {alpha} | Capital share in Cobb-Douglas production |\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $\\delta$ | {delta} | Depreciation rate |\n"
        f"| $r^{{\\ast}}$ | {r_star:.6f} | Closed-form clearing rate $1/\\beta - 1$ |\n"
        f"| $K^{{\\ast}}$ | {K_target:.4f} | Target aggregate capital at $r^{{\\ast}}$ |\n"
        f"| Bracket $[a_0, b_0]$ | $[{a0:.0e},\\, {b0}]$ | Sign-change bracket for bisection and Brent |\n"
        f"| Secant seeds | $[{a0:.0e},\\, {b0}]$ | Two starting points for secant |\n"
        f"| Newton start $x_0$ | {newton_x0} | Starting iterate for Newton-Raphson |\n"
        f"| Tolerance $\\varepsilon$ | {tol:.0e} | Stopping rule on residual and bracket width |"
    )

    report.add_solution_method(
        "All four methods solve the same scalar equation $Z(r) = 0$. They "
        "differ in what they need (bracket, two seeds, derivative) and how "
        "fast they converge.\n\n"
        "**Bisection.** Halve a sign-change bracket until its width is below "
        "tolerance.\n\n"
        "```text\n"
        "Algorithm: Bisection\n"
        "Input : a, b with Z(a) Z(b) < 0; tolerance eps\n"
        "Output: r_n\n"
        "  fa <- Z(a)\n"
        "  for n = 1, 2, ... :\n"
        "      m  <- (a + b) / 2\n"
        "      fm <- Z(m)\n"
        "      if fa * fm < 0: b <- m\n"
        "      else          : a <- m; fa <- fm\n"
        "      stop when |fm| < eps or (b - a) / 2 < eps\n"
        "```\n\n"
        "**Secant.** Step along the chord through the last two iterates.\n\n"
        "```text\n"
        "Algorithm: Secant\n"
        "Input : x_0, x_1; tolerance eps\n"
        "Output: x_n\n"
        "  f0 <- Z(x_0); f1 <- Z(x_1)\n"
        "  for n = 2, 3, ... :\n"
        "      x_n  <- x_1 - f1 (x_1 - x_0) / (f1 - f0)\n"
        "      fn   <- Z(x_n)\n"
        "      stop when |fn| < eps\n"
        "      shift: x_0 <- x_1, f0 <- f1; x_1 <- x_n, f1 <- fn\n"
        "```\n\n"
        "**Brent.** Try inverse quadratic interpolation through the last "
        "three iterates. Fall back to secant when ordinates coincide, and "
        "to bisection when the proposed step would leave the bracket.\n\n"
        "```text\n"
        "Algorithm: Brent-Dekker\n"
        "Input : a, b with Z(a) Z(b) < 0; tolerance eps\n"
        "Output: r_n\n"
        "  for n = 1, 2, ... :\n"
        "      try inverse quadratic interpolation -> candidate s\n"
        "      if s leaves [a, b] or fails half-step rule:\n"
        "          s <- (a + b) / 2     # bisect\n"
        "      fs <- Z(s)\n"
        "      update bracket so it still contains the root\n"
        "      stop when |fs| < eps or (b - a) < eps\n"
        "```\n\n"
        "**Newton-Raphson.** Step along the tangent at the current iterate.\n\n"
        "```text\n"
        "Algorithm: Newton-Raphson\n"
        "Input : x_0; tolerance eps; Z, Z'\n"
        "Output: x_n\n"
        "  for n = 0, 1, ... :\n"
        "      x_{n+1} <- x_n - Z(x_n) / Z'(x_n)\n"
        "      stop when |Z(x_n)| < eps\n"
        "```\n\n"
        f"On the same calibration, bisection takes **{int(bis_hist[-1][0])} "
        f"iterations**, secant takes **{int(sec_hist[-1][0])}**, Brent takes "
        f"**{int(bre_hist[-1][0])}**, and Newton from $x_0 = {newton_x0}$ "
        f"takes **{int(new_hist[-1][0])}**. The hand-coded Brent root "
        f"matches $\\mathrm{{scipy.optimize.brentq}}$ to "
        f"**{abs(bre_root - scipy_root):.2e}**."
    )

    # =========================================================================
    # Figures: trajectories (2x2 grid) + convergence-and-sensitivity (stacked)
    # =========================================================================
    method_colors = {
        "Bisection": "tab:orange",
        "Secant": "tab:green",
        "Brent": "tab:purple",
        "Newton-Raphson": "tab:brown",
    }
    method_markers = {
        "Bisection": "o", "Secant": "s", "Brent": "^", "Newton-Raphson": "D",
    }
    n_show = 4

    # ---- Figure 1: trajectories (2x2 grid) ----
    fig_traj, axes_traj = plt.subplots(2, 2, figsize=(10, 8))
    r_plot = np.linspace(a0, b0, 400)
    z_plot = Z(r_plot)
    grid_positions = {
        "Bisection": (0, 0),
        "Secant": (0, 1),
        "Brent": (1, 0),
        "Newton-Raphson": (1, 1),
    }
    for name, (row, col) in grid_positions.items():
        ax = axes_traj[row, col]
        ax.plot(r_plot, z_plot, color="tab:blue", linewidth=1.5)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.axvline(r_star, color="tab:red", linestyle="--", linewidth=1.0,
                   label=fr"$r^{{\ast}} = {r_star:.4f}$")
        h = histories[name][:n_show + 1]
        ax.plot(h[:, 1], h[:, 2], method_markers[name],
                color=method_colors[name], markersize=7, alpha=0.9, label=name)
        ax.set_title(name)
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$Z(r)$")
        ax.legend(loc="upper right", fontsize=9)
    fig_traj.tight_layout()

    report.add_results(
        "Each trajectory subplot plots $Z(r)$ with the first four iterates "
        "of one method on top.\n\n"
        "Bisection moves to the midpoint, then halves the bracket each step.\n\n"
        "Secant draws chords through the last two iterates and accelerates "
        "near the root.\n\n"
        "Brent looks like secant but cuts to a bisection step whenever the "
        "fast extrapolation would leave the bracket.\n\n"
        "Newton uses the tangent slope, so its iterates can leap further "
        "than the bracketed methods can."
    )
    report.add_figure(
        "figures/trajectories.png",
        "First iterates of each method overlaid on $Z(r)$",
        fig_traj,
    )

    # ---- Figure 2: convergence (top) + sensitivity (bottom) ----
    fig_diag, (ax_conv, ax_sens) = plt.subplots(2, 1, figsize=(11, 9))

    for name in method_colors:
        h = histories[name]
        err = np.maximum(np.abs(h[:, 1] - r_star), 1e-16)
        ax_conv.semilogy(h[:, 0], err, "-" + method_markers[name],
                         color=method_colors[name], markersize=4, linewidth=1.5,
                         label=name)
    ax_conv.set_xlabel("Iteration $n$")
    ax_conv.set_ylabel(r"$|x_n - r^{\ast}|$")
    ax_conv.set_title("Convergence to the closed-form clearing rate")
    ax_conv.legend()

    width = 0.20
    idx = np.arange(len(starts))
    ax_sens.bar(idx - 1.5 * width, bis_counts, width, color="tab:orange", label="Bisection")
    ax_sens.bar(idx - 0.5 * width, sec_counts, width, color="tab:green", label="Secant")
    ax_sens.bar(idx + 0.5 * width, bre_counts, width, color="tab:purple", label="Brent")
    new_colors = ["tab:brown" if s == "converged" else "lightgray" for s in new_status_list]
    new_hatches = ["" if s == "converged" else "//" for s in new_status_list]
    bars_n = ax_sens.bar(idx + 1.5 * width, new_counts, width, color=new_colors,
                         edgecolor="tab:brown", label="Newton")
    for bar, hatch in zip(bars_n, new_hatches):
        bar.set_hatch(hatch)
    for i, st in enumerate(new_status_list):
        if st == "diverged":
            ax_sens.text(idx[i] + 1.5 * width, new_counts[i] + 1, "DNC",
                         ha="center", va="bottom", color="tab:brown", fontsize=8)
    ax_sens.set_xticks(idx)
    ax_sens.set_xticklabels([f"{x:.3f}" for x in starts], rotation=45)
    ax_sens.set_xlabel(r"Starting point or bracket centre $x_0$")
    ax_sens.set_ylabel("Iterations to convergence")
    ax_sens.set_title(r"Iteration count vs starting point (tolerance $10^{-10}$)")
    ax_sens.legend(fontsize=9)
    fig_diag.tight_layout()

    report.add_results(
        "On a log axis the convergence rates are easy to read. Bisection "
        "halves its error each step. Secant accelerates once the iterates "
        f"settle near the root. Newton drops off a cliff after the first "
        "quadratic step. Brent matches the late-stage speed of secant or "
        "inverse quadratic interpolation.\n\n"
        "The sensitivity panel changes the starting point or bracket "
        "centre. Bisection and Brent stay flat: bracket halving is "
        "independent of where the bracket sits. Secant and Newton counts "
        f"depend on the start. **{n_diverged} of {len(starts)}** Newton "
        "starts step outside the feasible range and diverge (hatched bars "
        "marked DNC)."
    )
    report.add_figure(
        "figures/convergence-and-sensitivity.png",
        "Log-axis convergence (top) and iteration-count sensitivity to the starting point (bottom)",
        fig_diag,
    )

    # =========================================================================
    # Comparison table
    # =========================================================================
    table_data = {
        "Method": [m[0] for m in methods],
        "Inputs": [m[4] for m in methods],
        "Iterations": [int(histories[m[0]][-1, 0]) for m in methods],
        "Final residual": [f"{abs(histories[m[0]][-1, 2]):.2e}" for m in methods],
        "Error in r": [f"{abs(histories[m[0]][-1, 1] - r_star):.2e}" for m in methods],
        "Convergence rate": [m[3] for m in methods],
    }
    df = pd.DataFrame(table_data)
    report.add_results(
        "All four methods reach the closed-form root within tolerance. "
        "Brent and Newton finish in roughly an order of magnitude fewer "
        "iterations than bisection."
    )
    report.add_table(
        "tables/comparison.csv",
        "Bisection, secant, Brent, and Newton-Raphson on the stylized bond market",
        df,
    )

    report.add_takeaway(
        "Brent's method is the right default for production equilibrium "
        "solves. It inherits bisection's bracket invariant and adds "
        "superlinear speed via inverse quadratic interpolation when the "
        "bracket is preserved.\n\n"
        "Bisection is the safe fallback when no derivative is available. "
        "Secant is a no-derivative alternative to Newton with similar "
        "fragility from far-off seeds. "
        "Newton is fastest near a simple root but needs a derivative and a "
        "starting point inside the basin of attraction.\n\n"
        "$\\mathrm{scipy.optimize.brentq}$ is the production default for "
        "exactly these reasons."
    )

    report.add_references([
        "Mukoyama, T. (2021). *Basic Numerical Methods*. ECON 606 lecture slides, Georgetown University.",
        "Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall, Ch. 4.",
        "Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 9.",
        "Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 5.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")
    print(f"Brent (hand) - scipy.brentq: {abs(bre_root - scipy_root):.2e}")


if __name__ == "__main__":
    main()
