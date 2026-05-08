#!/usr/bin/env python3
"""Scalar root finding: bisection, secant, Brent, and Newton-Raphson on Z(r) = 0.

A stylized one-bond market clears at the rate that sets aggregate excess
demand to zero. With Cobb-Douglas firm-side capital demand and a target
aggregate stock at the deterministic Aiyagari rate, the market-clearing
condition is a scalar equation in $r$ whose root has a closed form. The
four basic root finders can be compared on convergence speed, robustness,
and sensitivity to the starting bracket or guess.

Reference: Mukoyama, T. (2021), Basic Numerical Methods, ECON 606, Georgetown.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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
    """Brent's method with inverse quadratic interpolation; records iterates.

    Follows the standard Brent-Dekker algorithm: try inverse quadratic
    interpolation when three distinct ordinates are available, else secant,
    else bisection. Falls back to bisection when the candidate would step
    outside the active bracket or fails to make at least half-step progress
    relative to the previous trial.
    """
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
    r_star = 1.0 / beta - 1.0           # closed-form clearing rate
    K_target = (alpha / (r_star + delta)) ** (1.0 / (1.0 - alpha))

    # Excess capital demand: Cobb-Douglas firm side minus target supply.
    def Z(r):
        return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha)) - K_target

    def Zprime(r):
        return -(1.0 / (1.0 - alpha)) * (
            (alpha / (r + delta)) ** (1.0 / (1.0 - alpha)) / (r + delta)
        )

    tol = 1e-10
    max_iter = 200

    a0, b0 = 0.0 + 1e-6, 0.10
    # Bisection / Brent run on the same bracket; secant starts at the bracket
    # endpoints; Newton starts at the lower endpoint.
    bis_root, bis_hist = bisection(Z, a0, b0, tol, max_iter)
    sec_root, sec_hist = secant(Z, a0, b0, tol, max_iter)
    bre_root, bre_hist = brent_with_history(Z, a0, b0, tol, max_iter)
    newton_x0 = 0.02
    new_root, new_hist, new_status = newton(Z, Zprime, newton_x0, tol, max_iter,
                                            lo=1e-9, hi=0.5)

    # Cross-check Brent against scipy
    scipy_root = brentq(Z, a0, b0, xtol=tol)

    methods = [
        ("Bisection", bis_root, bis_hist, "linear (1/2)", "sign-change bracket"),
        ("Secant", sec_root, sec_hist, "superlinear (~1.618)", "two starting points"),
        ("Brent", bre_root, bre_hist, "superlinear", "sign-change bracket"),
        ("Newton-Raphson", new_root, new_hist, "quadratic", "x_0 and Z'"),
    ]

    # Convert histories to numpy arrays of (iter, x, f, _)
    histories = {name: np.array([(h[0], h[1], h[2]) for h in hist], dtype=float)
                 for name, _, hist, _, _ in methods}

    # =========================================================================
    # Sensitivity to starting point / bracket
    # =========================================================================
    starts = np.array([0.001, 0.005, 0.01, 0.02, 0.03, r_star, 0.06, 0.08, 0.12])
    bis_counts, sec_counts, bre_counts, new_counts = [], [], [], []
    new_status_list = []
    for x0_ in starts:
        # Bisection / Brent: width-2*x0_ centred on x0_, clipped to (eps, 0.2);
        # if no sign change, fall back to the global bracket.
        a_ = max(1e-6, float(x0_) - 0.02)
        b_ = float(x0_) + 0.02
        if Z(a_) * Z(b_) >= 0.0:
            a_, b_ = a0, b0
        _, h_b = bisection(Z, a_, b_, tol, max_iter)
        bis_counts.append(int(h_b[-1][0]))
        _, h_br = brent_with_history(Z, a_, b_, tol, max_iter)
        bre_counts.append(int(h_br[-1][0]))
        # Secant: two seeds bracketing x0_
        s0, s1 = max(1e-6, float(x0_) - 0.005), float(x0_) + 0.005
        try:
            _, h_s = secant(Z, s0, s1, tol, max_iter)
            sec_counts.append(int(h_s[-1][0]))
        except (ZeroDivisionError, OverflowError):
            sec_counts.append(max_iter)
        # Newton from the same x0_
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
        "A representative-firm economy with Cobb-Douglas production and a "
        "target aggregate capital stock has a closed-form clearing rate "
        "$r^{\\ast} = 1/\\beta - 1$. The market-clearing condition is a "
        "scalar equation in $r$, and the four basic root finders all solve "
        "the same problem.\n\n"
        "Bisection halves a sign-change bracket. Secant fits a line through "
        "the last two iterates. Brent's method combines bisection's bracket "
        "invariant with inverse quadratic interpolation when accepting the "
        "fast step preserves the bracket. Newton-Raphson uses the analytic "
        "derivative for quadratic local convergence.\n\n"
        "These are the four solvers a first-year PhD student needs when "
        "Aiyagari- or Huggett-style equilibrium code stops converging. "
        "$\\mathrm{scipy.optimize.brentq}$ is the production default for "
        "exactly this reason."
    )

    report.add_equations(
        r"""
Aggregate capital demand from the firm-side first-order condition is

$$K_d(r) = \left( \frac{\alpha}{r + \delta} \right)^{\frac{1}{1 - \alpha}}.$$

Setting target aggregate supply at the deterministic clearing rate
$r^{\ast} = 1/\beta - 1$ gives $K^{\ast} \equiv K_d(r^{\ast})$. Excess
demand is

$$Z(r) = K_d(r) - K^{\ast},
\qquad Z(r^{\ast}) = 0,
\qquad r^{\ast} = \tfrac{1}{\beta} - 1.$$

The derivative used by Newton is

$$Z'(r) = -\frac{1}{1 - \alpha}\, \frac{K_d(r)}{r + \delta} < 0.$$

The four iterations are

$$\text{Bisection: } m_n = \tfrac{a_n + b_n}{2}, \quad
\text{keep the half with the sign change}.$$

$$\text{Secant: } x_{n+1} = x_n - Z(x_n)\, \frac{x_n - x_{n-1}}{Z(x_n) - Z(x_{n-1})}.$$

$$\text{Newton-Raphson: } x_{n+1} = x_n - \frac{Z(x_n)}{Z'(x_n)}.$$

Brent's method is a hybrid: it tries inverse quadratic interpolation
through the last three ordinates, falls back to secant when ordinates
coincide, and falls back to bisection when the proposed step would leave
the bracket or fails to halve the previous step length. The bracket
invariant is maintained at every iteration.
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
        f"| Secant seeds | $[{a0:.0e},\\, {b0}]$ | Two starting points for the secant iteration |\n"
        f"| Newton start $x_0$ | {newton_x0} | Starting iterate for Newton-Raphson |\n"
        f"| Tolerance $\\varepsilon$ | {tol:.0e} | Stopping rule on residual and bracket width |"
    )

    report.add_solution_method(
        "All four methods solve the same scalar equation $Z(r) = 0$. They "
        "differ in what they need (bracket, two seeds, derivative) and how "
        "fast they converge.\n\n"
        "```text\n"
        "Bisection           | Secant                      | Brent                              | Newton-Raphson\n"
        "Inputs: a, b, eps   | Inputs: x_0, x_1, eps       | Inputs: a, b, eps                  | Inputs: x_0, eps, Z, Z'\n"
        "fa <- Z(a)          | f0 <- Z(x_0); f1 <- Z(x_1)  | (use IQI when 3 distinct values,   | for n = 0, 1, ... :\n"
        "fb <- Z(b)          | for n = 2, 3, ... :         |  else secant; fall back to bisect  |     x_{n+1} <- x_n - Z(x_n)/Z'(x_n)\n"
        "for n = 1, 2, ... : |     dx <- (x_1 - x_0)       |  if step leaves bracket or fails   |     stop when |Z(x_n)| < eps\n"
        "    m <- (a+b)/2    |     dn <- f1 - f0           |  half-step progress test)          |\n"
        "    fm <- Z(m)      |     x_2 <- x_1 - f1 dx/dn   | maintain sign-change bracket each  |\n"
        "    if fa fm < 0:   |     stop when |f(x_2)| < eps|  iteration                         |\n"
        "        b, fb<-m,fm |     shift x_0 <- x_1        |                                    |\n"
        "    else:           |             x_1 <- x_2      |                                    |\n"
        "        a, fa<-m,fm |                             |                                    |\n"
        "    stop on tol     |                             |                                    |\n"
        "```\n\n"
        f"Starting from the bracket $[{a0:.0e},\\, {b0}]$, bisection takes "
        f"**{int(bis_hist[-1][0])} iterations**, secant takes "
        f"**{int(sec_hist[-1][0])}**, and Brent takes "
        f"**{int(bre_hist[-1][0])}**. Newton from $x_0 = {newton_x0}$ takes "
        f"**{int(new_hist[-1][0])} iterations**. The hand-coded Brent root "
        f"matches $\\mathrm{{scipy.optimize.brentq}}$ to "
        f"**{abs(bre_root - scipy_root):.2e}**."
    )

    # ------------------------------------------------------------------
    # Figure 1: Z(r) with method iterates overlaid
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    r_plot = np.linspace(a0, b0, 400)
    ax1.plot(r_plot, Z(r_plot), color="tab:blue", linewidth=2, label=r"$Z(r) = K_d(r) - K^{\ast}$")
    ax1.axhline(0.0, color="black", linewidth=0.6)
    ax1.axvline(r_star, color="tab:red", linestyle="--", linewidth=1.5,
                label=fr"$r^{{\ast}} = {r_star:.4f}$")

    method_colors = {"Bisection": "tab:orange", "Secant": "tab:green",
                     "Brent": "tab:purple", "Newton-Raphson": "tab:brown"}
    method_markers = {"Bisection": "o", "Secant": "s", "Brent": "^", "Newton-Raphson": "D"}
    n_show = 4
    for name in ["Bisection", "Secant", "Brent", "Newton-Raphson"]:
        h = histories[name][:n_show + 1]
        xs = h[:, 1]
        ys = h[:, 2]
        ax1.plot(xs, ys, method_markers[name], color=method_colors[name],
                 markersize=6, alpha=0.85, label=name)
    ax1.set_xlabel(r"Interest rate $r$")
    ax1.set_ylabel(r"$Z(r)$")
    ax1.set_title(r"Excess capital demand and first iterates of each method")
    ax1.legend(loc="upper right", fontsize=9)
    report.add_results(
        "Excess demand $Z(r)$ is monotone decreasing and crosses zero at "
        f"$r^{{\\ast}} = {r_star:.4f}$. The first {n_show} iterates of each "
        "method, plotted at $(x_n, Z(x_n))$, sit on the curve and march "
        "toward the root from different directions: bisection from the "
        "midpoint, secant along the chord, Brent mostly via the chord but "
        "occasionally bisecting, and Newton along the tangent."
    )
    report.add_figure(
        "figures/excess-demand.png",
        "Excess demand $Z(r)$ with the first iterates of bisection, secant, Brent, and Newton overlaid",
        fig1,
    )

    # ------------------------------------------------------------------
    # Figure 2: convergence on log axis for all four methods
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    for name in ["Bisection", "Secant", "Brent", "Newton-Raphson"]:
        h = histories[name]
        err = np.maximum(np.abs(h[:, 1] - r_star), 1e-16)
        ax2.semilogy(h[:, 0], err, "-" + method_markers[name],
                     color=method_colors[name], markersize=4, linewidth=1.5,
                     label=name)
    ax2.set_xlabel("Iteration $n$")
    ax2.set_ylabel(r"$|x_n - r^{\ast}|$")
    ax2.set_title("Convergence to the closed-form clearing rate")
    ax2.legend()
    report.add_results(
        "On a log axis the convergence rates are easy to read off. "
        "Bisection halves its error each step (a straight line with slope "
        "$\\log_{10}(1/2)$). Secant accelerates once the iterates settle near "
        "the root. Newton drops off a cliff after the first quadratic step. "
        "Brent inherits the bracket safety of bisection and the late-stage "
        "speed of secant or inverse quadratic interpolation, hitting the "
        "tolerance in only a handful of iterations."
    )
    report.add_figure(
        "figures/convergence.png",
        "Distance from the closed-form clearing rate vs iteration, log axis, all four methods",
        fig2,
    )

    # ------------------------------------------------------------------
    # Figure 3: sensitivity to starting point / bracket
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    width = 0.20
    idx = np.arange(len(starts))
    ax3.bar(idx - 1.5 * width, bis_counts, width, color="tab:orange", label="Bisection")
    ax3.bar(idx - 0.5 * width, sec_counts, width, color="tab:green", label="Secant")
    ax3.bar(idx + 0.5 * width, bre_counts, width, color="tab:purple", label="Brent")
    new_colors = ["tab:brown" if s == "converged" else "lightgray" for s in new_status_list]
    new_hatches = ["" if s == "converged" else "//" for s in new_status_list]
    bars_n = ax3.bar(idx + 1.5 * width, new_counts, width, color=new_colors,
                     edgecolor="tab:brown", label="Newton")
    for bar, hatch in zip(bars_n, new_hatches):
        bar.set_hatch(hatch)
    for i, st in enumerate(new_status_list):
        if st == "diverged":
            ax3.text(idx[i] + 1.5 * width, new_counts[i] + 1, "DNC",
                     ha="center", va="bottom", color="tab:brown", fontsize=8)
    ax3.set_xticks(idx)
    ax3.set_xticklabels([f"{x:.3f}" for x in starts], rotation=45)
    ax3.set_xlabel(r"Starting point or bracket centre $x_0$")
    ax3.set_ylabel("Iterations to convergence")
    ax3.set_title(r"Iteration count vs starting point (tolerance $10^{-10}$)")
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    report.add_results(
        "Bracketed methods (bisection, Brent) are insensitive to where the "
        "bracket is centred: the bisection counts are flat and Brent's "
        "stays in single digits. Secant and Newton iteration counts depend "
        f"on the start; **{n_diverged} of {len(starts)}** Newton starts "
        "step outside the feasible range and diverge (hatched bars marked "
        "DNC). Brent is the right production default precisely because it "
        "matches secant's late-stage speed without losing bisection's "
        "bracket invariant."
    )
    report.add_figure(
        "figures/sensitivity.png",
        "Iterations to converge as a function of starting point or bracket centre, all four methods",
        fig3,
    )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
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
        "The table summarises the four solves on the same calibration. All "
        "four reach the closed-form root within tolerance; Brent and Newton "
        "do so in roughly an order of magnitude fewer iterations than "
        "bisection."
    )
    report.add_table(
        "tables/comparison.csv",
        "Bisection, secant, Brent, and Newton-Raphson on the stylized bond market",
        df,
    )

    report.add_takeaway(
        "Brent's method is the right default for production equilibrium "
        "solves: it inherits bisection's bracket invariant and adds "
        "superlinear speed via inverse quadratic interpolation when the "
        "bracket is preserved. Bisection is the safe fallback when the "
        "derivative is unavailable. Secant is a no-derivative alternative "
        "to Newton with similar fragility from far-off seeds. Newton is "
        "fastest near a simple root but needs a derivative and a starting "
        "point inside the basin of attraction. This is the trade-off behind "
        "$\\mathrm{scipy.optimize.brentq}$ in Aiyagari- and Huggett-style "
        "equilibrium clearings later in the catalog."
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
