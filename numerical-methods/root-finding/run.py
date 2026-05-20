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
from lib.plotting import setup_style, save_figure, save_thumbnail


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
    # Figures and tables
    # =========================================================================
    setup_style()

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
    save_figure(fig_traj, "figures/trajectories.png", dpi=150)

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
    save_figure(fig_diag, "figures/convergence-and-sensitivity.png", dpi=150)

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

    # Persist the Brent-vs-scipy residual so audits can ground the README
    # value without re-running the tutorial.
    scipy_match_df = pd.DataFrame(
        {"brent_minus_scipy": [f"{abs(bre_root - scipy_root):.2e}"]}
    )
    (Path(__file__).resolve().parent / "tables").mkdir(exist_ok=True)
    scipy_match_df.to_csv(
        Path(__file__).resolve().parent / "tables" / "scipy_match.csv",
        index=False,
    )
    df.to_csv(
        Path(__file__).resolve().parent / "tables" / "comparison.csv",
        index=False,
    )

    save_thumbnail("figures/trajectories.png", "figures/thumb.png")
    print(f"Generated: figures/ (2 figures + thumb) + tables/ (2 tables)")
    print(f"Brent (hand) - scipy.brentq: {abs(bre_root - scipy_root):.2e}")


if __name__ == "__main__":
    main()
