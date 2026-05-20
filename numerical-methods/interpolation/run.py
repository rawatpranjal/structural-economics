#!/usr/bin/env python3
"""Off-grid function approximation: piecewise linear, cubic spline, and PCHIP.

VFI and EGP solvers store value or policy functions at discrete grid points
and need to evaluate them off the grid every iteration. Three classical
choices differ on smooth and kinked targets: piecewise linear preserves
shape but loses smoothness; natural cubic spline gives the best convergence
on smooth functions but rings near kinks; PCHIP keeps monotonicity at the
cost of $C^1$ smoothness.

Target 1: closed-form cake-eating value function $V(W)$ - smooth, monotone.
Target 2: a stylized consumption policy with a borrowing-constraint kink.

Reference: Mukoyama, T. (2021), Basic Numerical Methods, ECON 606, Georgetown.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport
from lib.interpolate import linear_interp


# =============================================================================
# Target 1: smooth cake-eating value function (closed form)
# =============================================================================

def make_cake_value(beta=0.9):
    def V(W):
        return (
            np.log((1.0 - beta) * np.maximum(W, 1e-12)) / (1.0 - beta)
            + beta * np.log(beta) / (1.0 - beta) ** 2
        )
    return V


# =============================================================================
# Target 2: kinked consumption policy at a borrowing constraint
# =============================================================================

def make_consumption_policy(r=0.04, y=0.5, a_kink=0.5, mpc=0.1):
    """c(a) is continuous; the slope drops from (1+r) to (1+r)*mpc at a_kink."""
    slope_below = 1.0 + r
    slope_above = (1.0 + r) * mpc
    c_kink = slope_below * a_kink + y

    def c(a):
        a = np.asarray(a)
        out = np.empty_like(a, dtype=float)
        below = a <= a_kink
        out[below] = slope_below * a[below] + y
        out[~below] = c_kink + slope_above * (a[~below] - a_kink)
        return out

    return c, a_kink, mpc


# =============================================================================
# Interpolation wrappers (linear via lib.interpolate; spline + PCHIP via scipy)
# =============================================================================

def interp_linear(x_nodes, y_nodes, x_query):
    return linear_interp(x_nodes, y_nodes, x_query)


def interp_cubic(x_nodes, y_nodes, x_query):
    spline = CubicSpline(x_nodes, y_nodes, bc_type="natural")
    return spline(x_query)


def interp_pchip(x_nodes, y_nodes, x_query):
    pchip = PchipInterpolator(x_nodes, y_nodes)
    return pchip(x_query)


METHODS = [
    ("Piecewise linear", interp_linear, "tab:blue", "-"),
    ("Cubic spline (natural)", interp_cubic, "tab:red", "--"),
    ("PCHIP (shape-preserving)", interp_pchip, "tab:green", "-."),
]


def errors_at_nodes(target, method_fn, n_nodes, x_min, x_max, n_query=2000):
    x_nodes = np.linspace(x_min, x_max, n_nodes)
    y_nodes = np.asarray(target(x_nodes))
    x_query = np.linspace(x_min, x_max, n_query)
    y_true = np.asarray(target(x_query))
    y_hat = np.asarray(method_fn(x_nodes, y_nodes, x_query))
    err = y_hat - y_true
    return {
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "x_query": x_query,
        "y_true": y_true,
        "y_hat": y_hat,
        "err": err,
        "sup_err": float(np.max(np.abs(err))),
        "l2_err": float(np.sqrt(np.mean(err ** 2))),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    beta = 0.9
    V = make_cake_value(beta=beta)
    c_policy, a_kink, mpc = make_consumption_policy()

    smooth_domain = (0.05, 1.0)
    kinked_domain = (0.05, 5.0)
    n_show = 10

    fits = {}
    for target_name, target, domain in [
        ("smooth", V, smooth_domain),
        ("kinked", c_policy, kinked_domain),
    ]:
        fits[target_name] = {}
        for method_name, method_fn, *_ in METHODS:
            fits[target_name][method_name] = errors_at_nodes(
                target, method_fn, n_show, domain[0], domain[1]
            )

    node_counts = np.array([5, 10, 20, 40, 80])
    convergence = {name: np.zeros(len(node_counts)) for name, *_ in METHODS}
    for i, n in enumerate(node_counts):
        for method_name, method_fn, *_ in METHODS:
            res = errors_at_nodes(V, method_fn, int(n), *smooth_domain)
            convergence[method_name][i] = res["sup_err"]

    setup_style()

    report = ModelReport(
        "Off-Grid Function Approximation by Interpolation",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Value function iteration stores $V$ at a finite grid and reads "
        "it off-grid every step. "
        "Three classical interpolators are the workhorses: piecewise "
        "linear, natural cubic spline, and PCHIP (piecewise cubic Hermite "
        "interpolating polynomial).\n\n"
        "This tutorial fits each one to two targets. "
        "The first target is the closed-form cake-eating value function "
        "$V(W)$, which is smooth and monotone. "
        "The second is a stylized consumption policy with a borrowing-constraint kink. "
        "The level is continuous but the slope drops sharply at $a_{\\text{kink}}$.\n\n"
        "Cubic splines shine on the smooth target. "
        "They ring on the kinked one. "
        "Linear interpolation and PCHIP do not."
    )

    report.add_equations(
        r"""The general problem is to recover an unknown function $f : [x_0, x_N] \to \mathbb{R}$ from values $\{(x_i, y_i)\}_{i=0}^{N}$ at a finite set of nodes.
An interpolant $\hat f$ matches the data ($\hat f(x_i) = y_i$ for every $i$) and provides a rule for evaluating $\hat f(x)$ at any $x$ between nodes.

### The test instances

Two targets stress different smoothness regimes.
The first target is the closed-form log-utility cake-eating value function on a smooth interior.

$$V(W) = \frac{\log((1-\beta) W)}{1-\beta} + \frac{\beta \log \beta}{(1-\beta)^2}.$$

The second target is a stylised consumption policy with a borrowing constraint at $a_{\text{kink}}$.

$$c(a) =
\begin{cases}
(1 + r)\, a + y, & a \leq a_{\text{kink}} \\
c(a_{\text{kink}}) + (1 + r)\, \mathrm{MPC}\, (a - a_{\text{kink}}), & a > a_{\text{kink}}.
\end{cases}$$

Below the kink the agent is constrained and consumes everything.
Above the kink they save with marginal propensity to consume $\mathrm{MPC} < 1$.
The function is continuous in level.
The slope drops from $(1 + r)$ to $(1 + r)\,\mathrm{MPC}$ at $a_{\text{kink}}$.

The next three subsections describe one method at a time.

### Method 1: Piecewise linear

Piecewise linear interpolation connects adjacent nodes with straight segments.
For a query $x$ in $[x_i, x_{i+1}]$ the interpolant is the convex combination of the bracketing values.

$$\hat{f}(x) = \frac{x_{i+1} - x}{x_{i+1} - x_i}\, f(x_i) + \frac{x - x_i}{x_{i+1} - x_i}\, f(x_{i+1}).$$

The interpolant is $C^0$ but generally not differentiable at the nodes.

### Method 2: Natural cubic spline

The natural cubic spline fits a piecewise cubic with $\hat{f}, \hat{f}', \hat{f}''$ continuous everywhere and $\hat{f}''(x_0) = \hat{f}''(x_N) = 0$.
The coefficients solve a tridiagonal linear system for the second derivatives at interior nodes.
The result is $C^2$ and is the smoothest interpolant in the integrated-squared-second-derivative sense.

### Method 3: PCHIP

PCHIP fits a piecewise cubic Hermite polynomial whose endpoint slopes are chosen by a monotonicity-preserving rule (Fritsch-Carlson 1980).
The result is $C^1$ and never overshoots a monotone target.
The trade against the cubic spline is between curvature and shape preservation.
Cubic splines bend smoothly but can ring near a kink.
PCHIP holds the shape but drops one order of smoothness.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $\\beta$ | {beta} | Discount factor in the cake-eating target |\n"
        f"| Smooth domain $[W_{{\\min}}, W_{{\\max}}]$ | $[{smooth_domain[0]},\\, {smooth_domain[1]}]$ | Wealth range for the smooth target |\n"
        f"| Kinked domain $[a_{{\\min}}, a_{{\\max}}]$ | $[{kinked_domain[0]},\\, {kinked_domain[1]}]$ | Asset range for the kinked target |\n"
        f"| $a_{{\\text{{kink}}}}$ | {a_kink} | Borrowing-constraint kink in the policy |\n"
        f"| $r$ | 0.04 | Interest rate in the consumption policy |\n"
        f"| $y$ | 0.5 | Endowment (income) in the consumption policy |\n"
        f"| $\\mathrm{{MPC}}$ | {mpc} | Marginal propensity to consume above the kink |\n"
        f"| Display node count $N$ | {n_show} | Nodes per fit in the target-vs-fit figure |\n"
        f"| Convergence sweep | {list(node_counts)} | Node counts for the smooth-target sup-norm sweep |"
    )

    report.add_solution_method(
        "Each method takes nodes $(x_i, y_i)$ and returns a function on "
        "$[x_0, x_N]$.\n\n"
        "**Piecewise linear.** Connect adjacent nodes with straight "
        "segments. The convex-combination formula evaluates the segment "
        "containing the query point.\n\n"
        "```text\n"
        "Algorithm: Piecewise linear\n"
        "Input : nodes (x_i, y_i); query x in [x_i, x_{i+1}]\n"
        "Output: y_hat\n"
        "  h_i   <- x_{i+1} - x_i\n"
        "  w     <- (x - x_i) / h_i\n"
        "  y_hat <- (1 - w) y_i + w y_{i+1}\n"
        "```\n\n"
        "**Natural cubic spline.** Fit a piecewise cubic with $C^2$ "
        "continuity and zero second derivatives at the endpoints.\n\n"
        "```text\n"
        "Algorithm: Natural cubic spline\n"
        "Input : nodes (x_i, y_i)\n"
        "Output: spline S(x)\n"
        "  build tridiagonal system in y''_1, ..., y''_{N-1}\n"
        "  with natural BC y''_0 = y''_N = 0\n"
        "  solve once for the second-derivative values\n"
        "  on [x_i, x_{i+1}], evaluate the cubic from\n"
        "    y_i, y_{i+1}, y''_i, y''_{i+1}\n"
        "```\n\n"
        "**PCHIP (shape-preserving).** Fit a piecewise cubic Hermite "
        "polynomial whose endpoint slopes are chosen by the Fritsch-"
        "Carlson rule so the result preserves monotonicity.\n\n"
        "```text\n"
        "Algorithm: PCHIP\n"
        "Input : nodes (x_i, y_i)\n"
        "Output: H(x)\n"
        "  m_i <- (y_{i+1} - y_i) / (x_{i+1} - x_i)   # secant slopes\n"
        "  pick endpoint slopes d_i by Fritsch-Carlson rule\n"
        "    so that monotonicity of {y_i} is preserved\n"
        "  on [x_i, x_{i+1}], evaluate Hermite cubic from\n"
        "    y_i, y_{i+1}, d_i, d_{i+1}\n"
        "```\n\n"
        "The linear branch reuses `lib.interpolate.linear_interp`. The "
        "cubic and PCHIP branches use `scipy.interpolate.CubicSpline` "
        "(`bc_type='natural'`) and `scipy.interpolate.PchipInterpolator`."
    )

    fig1, (axS, axK) = plt.subplots(1, 2, figsize=(12, 5))
    sm = fits["smooth"]["Piecewise linear"]
    axS.plot(sm["x_query"], sm["y_true"], color="black", linewidth=1.5, label="True $V(W)$")
    for method_name, _, color, ls in METHODS:
        f = fits["smooth"][method_name]
        axS.plot(f["x_query"], f["y_hat"], color=color, linestyle=ls, linewidth=1.5,
                 label=method_name)
    axS.plot(sm["x_nodes"], sm["y_nodes"], "o", color="black", markersize=5, label="Nodes")
    axS.set_xlabel(r"$W$")
    axS.set_ylabel(r"$V(W)$")
    axS.set_title(f"Smooth target ({n_show} nodes)")
    axS.legend(fontsize=8, loc="lower right")

    kn = fits["kinked"]["Piecewise linear"]
    axK.plot(kn["x_query"], kn["y_true"], color="black", linewidth=1.5, label=r"True $c(a)$")
    for method_name, _, color, ls in METHODS:
        f = fits["kinked"][method_name]
        axK.plot(f["x_query"], f["y_hat"], color=color, linestyle=ls, linewidth=1.5,
                 label=method_name)
    axK.plot(kn["x_nodes"], kn["y_nodes"], "o", color="black", markersize=5, label="Nodes")
    axK.axvline(a_kink, color="gray", linestyle=":", linewidth=1.0,
                label=fr"$a_{{\text{{kink}}}} = {a_kink}$")
    axK.set_xlabel(r"$a$")
    axK.set_ylabel(r"$c(a)$")
    axK.set_title(f"Kinked target ({n_show} nodes)")
    axK.legend(fontsize=8, loc="lower right")
    fig1.tight_layout()
    report.add_results(
        f"At {n_show} nodes the three methods agree closely on the smooth "
        "value function.\n\n"
        "On the kinked policy the cubic spline rings near "
        "$a_{\\text{kink}}$: $C^2$ smoothness forces it to oscillate "
        "around the slope discontinuity.\n\n"
        "Piecewise linear and PCHIP track the kink without overshoot, at "
        "the cost of a corner where the slope changes."
    )
    report.add_figure(
        "figures/target-vs-fit.png",
        "Three approximations against the smooth (left) and kinked (right) targets at the same node count",
        fig1,
    )

    fig2, (axES, axEK) = plt.subplots(1, 2, figsize=(12, 5))
    for method_name, _, color, ls in METHODS:
        f = fits["smooth"][method_name]
        axES.plot(f["x_query"], f["err"], color=color, linestyle=ls, linewidth=1.5,
                  label=method_name)
    axES.axhline(0.0, color="black", linewidth=0.5)
    axES.set_xlabel(r"$W$")
    axES.set_ylabel(r"$\hat{V}(W) - V(W)$")
    axES.set_title("Smooth target: pointwise error")
    axES.legend(fontsize=8)

    for method_name, _, color, ls in METHODS:
        f = fits["kinked"][method_name]
        axEK.plot(f["x_query"], f["err"], color=color, linestyle=ls, linewidth=1.5,
                  label=method_name)
    axEK.axhline(0.0, color="black", linewidth=0.5)
    axEK.axvline(a_kink, color="gray", linestyle=":", linewidth=1.0)
    axEK.set_xlabel(r"$a$")
    axEK.set_ylabel(r"$\hat{c}(a) - c(a)$")
    axEK.set_title("Kinked target: pointwise error")
    axEK.legend(fontsize=8)
    fig2.tight_layout()
    sup_lin = fits["kinked"]["Piecewise linear"]["sup_err"]
    sup_cub = fits["kinked"]["Cubic spline (natural)"]["sup_err"]
    sup_pch = fits["kinked"]["PCHIP (shape-preserving)"]["sup_err"]
    report.add_results(
        "On the smooth target all three errors concentrate near $W = 0$, "
        "where curvature is largest. PCHIP is uniformly smallest, ahead of "
        "the cubic spline at this node count.\n\n"
        "On the kinked target the cubic-spline error oscillates above "
        f"and below zero around $a_{{\\text{{kink}}}}$ (sup-error "
        f"**{sup_cub:.2e}**).\n\n"
        f"PCHIP eliminates the ringing at the same node count (sup-error "
        f"**{sup_pch:.2e}**).\n\n"
        f"Piecewise linear under-shoots in the same interval (sup-error "
        f"**{sup_lin:.2e}**) but stays monotone."
    )
    report.add_figure(
        "figures/error-curves.png",
        "Pointwise error of each method on the smooth and kinked targets at $N=10$ nodes",
        fig2,
    )

    fig3, ax3 = plt.subplots()
    method_styles = {name: (c, ls) for name, _, c, ls in METHODS}
    method_markers = {"Piecewise linear": "o", "Cubic spline (natural)": "s",
                      "PCHIP (shape-preserving)": "^"}
    for method_name in convergence:
        c, ls = method_styles[method_name]
        ax3.loglog(node_counts, convergence[method_name],
                   linestyle=ls, marker=method_markers[method_name], color=c,
                   linewidth=1.5, markersize=5, label=method_name)
    ax3.set_xlabel("Nodes $N$")
    ax3.set_ylabel("Sup-norm error on smooth target")
    ax3.set_title("Convergence vs node count, smooth target")
    ax3.legend()
    log_n = np.log(node_counts)
    slopes = {
        name: float(np.polyfit(log_n, np.log(convergence[name]), 1)[0])
        for name in convergence
    }
    slope_lin = slopes["Piecewise linear"]
    slope_cub = slopes["Cubic spline (natural)"]
    slope_pch = slopes["PCHIP (shape-preserving)"]
    report.add_results(
        "The log-log sup-norm slopes on the smooth target are "
        f"**{slope_lin:.1f}** for piecewise linear, **{slope_cub:.1f}** for "
        f"the cubic spline, and **{slope_pch:.1f}** for PCHIP.\n\n"
        "All three fall short of their textbook asymptotic rates. "
        "The cake-eating value function $V(W)$ has a logarithmic "
        "singularity as $W \\to 0$, so curvature blows up near the left "
        "edge of the domain. That near-singular region keeps every method "
        "below its smooth-function rate at these node counts; the cubic "
        "spline does not reach the $-4$ slope a fully smooth target would "
        "give.\n\n"
        "On a kinked target the smoothness advantage disappears entirely, "
        "and PCHIP becomes the right default because it preserves shape."
    )
    report.add_figure(
        "figures/convergence-vs-nodes.png",
        "Sup-norm error vs node count on the smooth cake-eating target, log-log axes",
        fig3,
    )

    rows = []
    for method_name, *_ in METHODS:
        rows.append({
            "Method": method_name,
            "Smooth sup-error": f"{fits['smooth'][method_name]['sup_err']:.2e}",
            "Smooth L2 error": f"{fits['smooth'][method_name]['l2_err']:.2e}",
            "Kinked sup-error": f"{fits['kinked'][method_name]['sup_err']:.2e}",
            "Kinked L2 error": f"{fits['kinked'][method_name]['l2_err']:.2e}",
        })
    df = pd.DataFrame(rows)
    smooth_winner = min(
        fits["smooth"], key=lambda m: fits["smooth"][m]["sup_err"]
    )
    kinked_winner = min(
        fits["kinked"], key=lambda m: fits["kinked"][m]["sup_err"]
    )
    report.add_results(
        f"At a fixed budget of {n_show} nodes the table summarises sup-norm "
        f"and L2 errors for each method on both targets. {smooth_winner} is "
        f"the lowest-error choice on the smooth target; {kinked_winner} is "
        f"the lowest-error choice on the kinked one."
    )
    report.add_table(
        "tables/comparison.csv",
        f"Sup-norm and L2 errors at $N = {n_show}$ nodes for each method on the smooth and kinked targets",
        df,
    )

    report.add_takeaway(
        "Piecewise linear is the safe default for value functions with "
        "borrowing constraints. It preserves shape, never overshoots, and "
        "requires no setup. "
        "Natural cubic spline is accurate on smooth functions but rings "
        "near kinks and can violate monotonicity. "
        "PCHIP gives the steepest log-log convergence slope on the smooth "
        "target here, ahead of the cubic spline, and is the right default "
        "for monotone-but-non-smooth policies. It beats linear on accuracy "
        "and cubic on shape preservation at the same node count.\n\n"
        "`lib.interpolate.linear_interp` is what the existing tutorials "
        "use today. Promoting cubic and PCHIP wrappers to "
        "`lib/interpolate.py` is worth doing once a second tutorial needs "
        "them."
    )

    report.add_references([
        "Mukoyama, T. (2021). *Basic Numerical Methods*. ECON 606 lecture slides, Georgetown University.",
        "Fritsch, F. N. and Carlson, R. E. (1980). *Monotone Piecewise Cubic Interpolation*. SIAM Journal on Numerical Analysis 17(2), 238-246.",
        "Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 3.",
        "Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 6.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
