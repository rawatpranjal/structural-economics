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
from lib.plotting import setup_style, save_figure, save_thumbnail
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
    save_figure(fig1, "figures/target-vs-fit.png", dpi=150)

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
    save_figure(fig2, "figures/error-curves.png", dpi=150)

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
    save_figure(fig3, "figures/convergence-vs-nodes.png", dpi=150)

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
    Path("tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("tables/comparison.csv", index=False)

    save_thumbnail("figures/target-vs-fit.png", "figures/thumb.png")
    print(f"Generated: figures/ (3 figures + thumb) + tables/ (1 table)")


if __name__ == "__main__":
    main()
