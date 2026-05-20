#!/usr/bin/env python3
"""Scalar optimization for monopoly pricing under constant-elasticity demand.

Four methods are compared against the closed-form Lerner markup:
deterministic grid search, stochastic random search, derivative-free
golden-section search, and derivative-based Newton on the first-order
condition. A start in the convex region of the profit function gives
the Newton failure mode that motivates the bracket safeguard.

References:
- Tirole (1988) The Theory of Industrial Organization, Ch. 1.
- Press, Teukolsky, Vetterling, and Flannery (2007) Numerical Recipes, Ch. 10.
- Judd (1998) Numerical Methods in Economics, Ch. 4.
- Nocedal and Wright (2006) Numerical Optimization, Ch. 3.
"""
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def main() -> None:
    # =========================================================================
    # Calibration
    # =========================================================================
    A = 1.0
    epsilon = 2.5
    c = 1.0

    p_star = epsilon * c / (epsilon - 1.0)
    lerner = 1.0 / epsilon
    p_inflect = (epsilon + 1.0) * c / (epsilon - 1.0)

    def D(p):
        return A * np.power(p, -epsilon)

    def profit(p):
        return (p - c) * D(p)

    def profit_prime(p):
        return A * np.power(p, -(epsilon + 1.0)) * ((1.0 - epsilon) * p + epsilon * c)

    def profit_double_prime(p):
        return -A * epsilon * np.power(p, -(epsilon + 2.0)) * ((1.0 - epsilon) * p + (epsilon + 1.0) * c)

    p_low = 1.05
    p_high = 5.00

    tol = 1e-10
    max_iter = 200

    # =========================================================================
    # Method 1: deterministic grid search
    # =========================================================================
    sample_sizes = [11, 21, 51, 101, 501, 1001, 5001]
    grid_errors_by_n = []
    for n in sample_sizes:
        grid_n = np.linspace(p_low, p_high, n)
        i_best = int(np.argmax(profit(grid_n)))
        grid_errors_by_n.append(abs(grid_n[i_best] - p_star))

    n_grid_main = 1001
    main_grid = np.linspace(p_low, p_high, n_grid_main)
    grid_root = float(main_grid[int(np.argmax(profit(main_grid)))])

    # =========================================================================
    # Method 2: stochastic random search
    # =========================================================================
    seed_main = 42
    n_replications = 50

    random_errors_by_n = []
    for n in sample_sizes:
        errs = []
        for s in range(n_replications):
            rng_s = np.random.default_rng(s)
            draws = rng_s.uniform(p_low, p_high, n)
            i_best = int(np.argmax(profit(draws)))
            errs.append(abs(draws[i_best] - p_star))
        random_errors_by_n.append(float(np.mean(errs)))

    rng_main = np.random.default_rng(seed_main)
    draws_main = rng_main.uniform(p_low, p_high, n_grid_main)
    random_root = float(draws_main[int(np.argmax(profit(draws_main)))])

    rng_show = np.random.default_rng(123)
    draws_show = rng_show.uniform(p_low, p_high, 20)

    # =========================================================================
    # Method 3: golden-section search
    # =========================================================================
    phi = (math.sqrt(5.0) - 1.0) / 2.0

    def golden_section(a, b):
        pL = b - phi * (b - a)
        pR = a + phi * (b - a)
        fL, fR = profit(pL), profit(pR)
        rows = [(0, a, b, 0.5 * (a + b), abs(0.5 * (a + b) - p_star))]
        for n in range(1, max_iter + 1):
            if fL > fR:
                b = pR
                pR = pL
                fR = fL
                pL = b - phi * (b - a)
                fL = profit(pL)
            else:
                a = pL
                pL = pR
                fL = fR
                pR = a + phi * (b - a)
                fR = profit(pR)
            mid = 0.5 * (a + b)
            rows.append((n, a, b, mid, abs(mid - p_star)))
            if (b - a) < tol:
                break
        return np.array(rows, dtype=float)

    golden_history = golden_section(p_low, p_high)
    golden_root = float(golden_history[-1, 3])
    golden_iter = int(golden_history[-1, 0])
    golden_residual = float(abs(profit_prime(golden_root)))

    # =========================================================================
    # Method 4: Newton on the FOC, vanilla and safeguarded
    # =========================================================================
    def newton_run(x0, safeguard=False, bracket=(p_low, p_high)):
        a, b = bracket
        x = float(x0)
        rows = [(0, x, abs(x - p_star))]
        status = "max_iter"
        for n in range(1, max_iter + 1):
            fp = profit_prime(x)
            if abs(fp) < tol:
                status = "converged"
                break
            fpp = profit_double_prime(x)
            if not np.isfinite(fp) or not np.isfinite(fpp):
                status = "diverged"
                break
            if safeguard and fpp >= 0.0:
                x_new = 0.5 * (b + x) if fp > 0.0 else 0.5 * (a + x)
            elif fpp == 0.0:
                status = "diverged"
                break
            else:
                x_new = x - fp / fpp
                if safeguard and not (a < x_new < b):
                    margin = 0.001 * (b - a)
                    x_new = float(np.clip(x_new, a + margin, b - margin))
            if not np.isfinite(x_new):
                rows.append((n, x_new, float("nan")))
                status = "diverged"
                break
            if not safeguard and not (a < x_new < b):
                rows.append((n, x_new, float("nan")))
                status = "diverged"
                break
            x = x_new
            rows.append((n, x, abs(x - p_star)))
        return np.array(rows, dtype=float), status

    x0_good = 1.20
    newton_good_history, newton_good_status = newton_run(x0_good, safeguard=False)
    newton_good_root = float(newton_good_history[-1, 1])
    newton_good_iter = int(newton_good_history[-1, 0])
    newton_good_residual = float(abs(profit_prime(newton_good_root)))

    x0_bad = 3.00
    newton_bad_history, newton_bad_status = newton_run(x0_bad, safeguard=False)

    newton_safe_history, newton_safe_status = newton_run(x0_bad, safeguard=True, bracket=(p_low, p_high))
    newton_safe_root = float(newton_safe_history[-1, 1])
    newton_safe_iter = int(newton_safe_history[-1, 0])
    newton_safe_residual = float(abs(profit_prime(newton_safe_root)))

    # =========================================================================
    # Sensitivity sweeps
    # =========================================================================
    starting_points = np.array([1.05, 1.20, 1.40, 1.60, 1.80, 2.00, 2.50, 3.50, 4.50])
    newton_counts = []
    newton_status_list = []
    for x0_ in starting_points:
        hist, status = newton_run(float(x0_), safeguard=False)
        newton_counts.append(int(hist[-1, 0]))
        newton_status_list.append(status)
    n_diverged = sum(1 for s in newton_status_list if s == "diverged")

    eps_values = [1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
    eps_rows = []
    for eps in eps_values:
        p_s = eps * c / (eps - 1.0)
        markup = 1.0 / eps
        prof = (p_s - c) * A * p_s ** (-eps)

        def prof_eps(pp, _eps=eps):
            return (pp - c) * A * pp ** (-_eps)

        a_, b_ = p_low, p_high
        pL_ = b_ - phi * (b_ - a_)
        pR_ = a_ + phi * (b_ - a_)
        fL_, fR_ = prof_eps(pL_), prof_eps(pR_)
        for _ in range(max_iter):
            if fL_ > fR_:
                b_ = pR_
                pR_ = pL_
                fR_ = fL_
                pL_ = b_ - phi * (b_ - a_)
                fL_ = prof_eps(pL_)
            else:
                a_ = pL_
                pL_ = pR_
                fL_ = fR_
                pR_ = a_ + phi * (b_ - a_)
                fR_ = prof_eps(pR_)
            if (b_ - a_) < tol:
                break
        p_golden_eps = 0.5 * (a_ + b_)
        eps_rows.append({
            "epsilon": float(eps),
            "p_star": float(p_s),
            "Lerner markup": float(markup),
            "profit at p_star": float(prof),
            "golden-section error": float(abs(p_golden_eps - p_s)),
        })

    # =========================================================================
    # Figures
    # =========================================================================
    setup_style()

    # ------------------------------------------------------------------
    # Figure 1: profit curve
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    p_plot = np.linspace(p_low, p_high, 400)
    ax1.plot(p_plot, profit(p_plot), color="tab:blue", linewidth=2,
             label=r"$\pi(p) = (p - c)\, A p^{-\epsilon}$")
    ax1.axvline(p_star, color="tab:red", linestyle="--", linewidth=1.5,
                label=fr"$p^{{\ast}} = \epsilon c / (\epsilon - 1) = {p_star:.3f}$")
    ax1.axvline(c, color="tab:gray", linestyle=":", linewidth=1.0,
                label=fr"$c = {c}$")
    ax1.axvline(p_inflect, color="tab:purple", linestyle=":", linewidth=1.0,
                label=fr"$p_{{\mathrm{{inflect}}}} = {p_inflect:.3f}$")
    pi_at_star = float(profit(p_star))
    ax1.annotate(
        fr"Lerner markup $1/\epsilon = {lerner:.3f}$",
        xy=(p_star, pi_at_star),
        xytext=(p_star + 0.7, pi_at_star * 0.65),
        fontsize=10, color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", linewidth=1.0),
    )
    ax1.set_xlabel("Price $p$")
    ax1.set_ylabel(r"Profit $\pi(p)$")
    ax1.set_title("Constant-elasticity profit and the closed-form optimum")
    ax1.legend(loc="upper right", fontsize=9)
    save_figure(fig1, "figures/profit-curve.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 2: method paths overlaid on the profit curve
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.plot(p_plot, profit(p_plot), color="tab:blue", linewidth=2, label=r"$\pi(p)$")
    ax2.axvline(p_star, color="tab:red", linestyle="--", linewidth=1.5,
                label=fr"$p^{{\ast}} = {p_star:.3f}$")

    n_show_grid = 11
    p_grid_show = np.linspace(p_low, p_high, n_show_grid)
    ax2.plot(p_grid_show, profit(p_grid_show), "o", color="tab:gray", markersize=4,
             alpha=0.7, label=fr"Grid search ($N = {n_show_grid}$)")

    ax2.plot(draws_show, profit(draws_show), "v", color="tab:cyan",
             markersize=5, alpha=0.7,
             label=fr"Random search ($N = {len(draws_show)}$, seed $123$)")

    pi_min = float(np.min(profit(p_plot)))
    pi_max = float(np.max(profit(p_plot)))
    pi_span = pi_max - pi_min
    base = pi_min - 0.10 * pi_span
    n_show_bracket = min(6, len(golden_history))
    for i in range(n_show_bracket):
        _, a_, b_, m_, _ = golden_history[i]
        y = base - 0.04 * i * pi_span
        ax2.plot([a_, b_], [y, y], color="tab:orange", linewidth=2.0, alpha=0.7,
                 label="Golden bracket" if i == 0 else None)
        ax2.plot([m_], [y], "o", color="tab:orange", markersize=4)

    for i, (_niter, xi, _err) in enumerate(newton_good_history[:5]):
        ax2.plot([xi], [profit(xi)], "s", color="tab:green", markersize=6,
                 label=f"Newton, $x_0 = {x0_good}$" if i == 0 else None)

    ax2.set_xlabel("Price $p$")
    ax2.set_ylabel(r"$\pi(p)$")
    ax2.set_title("Iterates of all four methods on the profit curve")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_ylim(base - 0.04 * pi_span * n_show_bracket, pi_max + 0.10 * pi_span)
    save_figure(fig2, "figures/method-paths.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3: Newton failure mode and the bracket safeguard
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    p_wide = np.linspace(p_low, 8.0, 400)
    ax3a.plot(p_wide, profit(p_wide), color="tab:blue", linewidth=2, label=r"$\pi(p)$")
    ax3a.axvline(p_star, color="tab:red", linestyle="--", linewidth=1.2,
                 label=fr"$p^{{\ast}} = {p_star:.3f}$")
    ax3a.axvline(p_inflect, color="tab:purple", linestyle=":", linewidth=1.0,
                 label=fr"$p_{{\mathrm{{inflect}}}} = {p_inflect:.3f}$")
    ax3a.axvspan(p_low, p_high, color="tab:gray", alpha=0.07,
                 label=fr"Search bracket $[{p_low:.2f},\, {p_high:.2f}]$")
    n_show_bad = len(newton_bad_history)
    for i in range(n_show_bad):
        xi = float(newton_bad_history[i, 1])
        if not np.isfinite(xi):
            continue
        if xi <= p_wide[-1] and xi > 0:
            ax3a.plot([xi], [profit(xi)], "x", color="tab:red", markersize=9,
                      markeredgewidth=2.0,
                      label="Newton, no safeguard" if i == 0 else None)
            if i + 1 < n_show_bad:
                xj = float(newton_bad_history[i + 1, 1])
                if np.isfinite(xj) and xj <= p_wide[-1] and xj > 0:
                    ax3a.annotate(
                        "", xy=(xj, profit(xj)), xytext=(xi, profit(xi)),
                        arrowprops=dict(arrowstyle="->", color="tab:red", linewidth=1.0, alpha=0.7),
                    )
    final_x_bad = float(newton_bad_history[-1, 1])
    if np.isfinite(final_x_bad) and final_x_bad > p_high:
        ax3a.text(min(final_x_bad, p_wide[-1] - 0.1),
                  profit(min(final_x_bad, p_wide[-1] - 0.1)) + 0.02 * pi_span,
                  fr"$x_1 = {final_x_bad:.2f}$ (out of bracket)",
                  fontsize=9, color="tab:red", ha="right")
    ax3a.set_xlabel("Price $p$")
    ax3a.set_ylabel(r"$\pi(p)$")
    ax3a.set_title(fr"Vanilla Newton from $x_0 = {x0_bad}$ in the convex region")
    ax3a.legend(loc="upper right", fontsize=9)

    p_zoom = np.linspace(p_low, p_high, 400)
    ax3b.plot(p_zoom, profit(p_zoom), color="tab:blue", linewidth=2, label=r"$\pi(p)$")
    ax3b.axvline(p_star, color="tab:red", linestyle="--", linewidth=1.2,
                 label=fr"$p^{{\ast}} = {p_star:.3f}$")
    ax3b.axvline(p_inflect, color="tab:purple", linestyle=":", linewidth=1.0,
                 label=fr"$p_{{\mathrm{{inflect}}}}$")
    n_show_safe = min(8, len(newton_safe_history))
    for i in range(n_show_safe):
        xi = float(newton_safe_history[i, 1])
        ax3b.plot([xi], [profit(xi)], "s", color="tab:green", markersize=6,
                  label="Newton + safeguard" if i == 0 else None)
        if i + 1 < n_show_safe:
            xj = float(newton_safe_history[i + 1, 1])
            ax3b.annotate(
                "", xy=(xj, profit(xj)), xytext=(xi, profit(xi)),
                arrowprops=dict(arrowstyle="->", color="tab:green", linewidth=1.0, alpha=0.7),
            )
    ax3b.set_xlabel("Price $p$")
    ax3b.set_ylabel(r"$\pi(p)$")
    ax3b.set_title(fr"Safeguarded Newton from the same $x_0 = {x0_bad}$")
    ax3b.legend(loc="lower right", fontsize=9)
    fig3.tight_layout()
    save_figure(fig3, "figures/newton-failure.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 4: convergence diagnostics
    # ------------------------------------------------------------------
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

    ax4a.semilogy(golden_history[:, 0], np.maximum(golden_history[:, 4], 1e-16),
                  "o-", color="tab:orange", markersize=3, linewidth=1.5,
                  label="Golden section")
    ax4a.semilogy(newton_good_history[:, 0], np.maximum(newton_good_history[:, 2], 1e-16),
                  "s-", color="tab:green", markersize=4, linewidth=1.5,
                  label=fr"Newton, $x_0 = {x0_good}$")
    ax4a.semilogy(newton_safe_history[:, 0], np.maximum(newton_safe_history[:, 2], 1e-16),
                  "d-", color="tab:purple", markersize=4, linewidth=1.5,
                  label=fr"Newton + safeguard, $x_0 = {x0_bad}$")
    ax4a.set_xlabel("Iteration $n$")
    ax4a.set_ylabel(r"$|x_n - p^{\ast}|$")
    ax4a.set_title("Convergence to the closed-form optimum")
    ax4a.legend(loc="upper right", fontsize=9)

    n_arr = np.array(sample_sizes, dtype=float)
    ax4b.loglog(n_arr, np.maximum(grid_errors_by_n, 1e-16),
                "o-", color="tab:gray", markersize=5, linewidth=1.5,
                label="Grid search")
    ax4b.loglog(n_arr, np.maximum(random_errors_by_n, 1e-16),
                "v-", color="tab:cyan", markersize=5, linewidth=1.5,
                label=fr"Random search (mean over {n_replications} seeds)")
    ref_n = (p_high - p_low) / n_arr
    ax4b.loglog(n_arr, ref_n, "--", color="tab:red", linewidth=1.0,
                label=r"Reference $\propto 1/N$")
    ref_sqrt = (p_high - p_low) / np.sqrt(n_arr)
    ax4b.loglog(n_arr, ref_sqrt, ":", color="tab:purple", linewidth=1.0,
                label=r"Reference $\propto 1/\sqrt{N}$")
    ax4b.set_xlabel("Sample budget $N$")
    ax4b.set_ylabel(r"$|p_{\mathrm{hat}} - p^{\ast}|$")
    ax4b.set_title("Sampling error vs sample budget")
    ax4b.legend(loc="upper right", fontsize=8)
    fig4.tight_layout()
    save_figure(fig4, "figures/convergence.png", dpi=150)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    bad_p_str = (f"{float(newton_bad_history[-1, 1]):.4f}"
                 if np.isfinite(newton_bad_history[-1, 1]) else "n/a")
    bad_err_str = (f"{abs(float(newton_bad_history[-1, 1]) - p_star):.2e}"
                   if np.isfinite(newton_bad_history[-1, 1]) else "n/a")

    method_table = pd.DataFrame({
        "Method": [
            "Grid search",
            "Random search",
            "Golden section",
            "Newton (good start)",
            "Newton (bad start)",
            "Newton with safeguard (bad start)",
        ],
        "Setting": [
            f"{n_grid_main} grid nodes",
            f"{n_grid_main} random draws, seed {seed_main}",
            f"Bracket from {p_low:.2f} to {p_high:.2f}",
            f"Starting price {x0_good:.2f}",
            f"Starting price {x0_bad:.2f}",
            f"Starting price {x0_bad:.2f}",
        ],
        "Estimated optimum": [
            f"{grid_root:.4f}",
            f"{random_root:.4f}",
            f"{golden_root:.4f}",
            f"{newton_good_root:.4f}",
            bad_p_str,
            f"{newton_safe_root:.4f}",
        ],
        "Absolute error": [
            f"{abs(grid_root - p_star):.2e}",
            f"{abs(random_root - p_star):.2e}",
            f"{abs(golden_root - p_star):.2e}",
            f"{abs(newton_good_root - p_star):.2e}",
            bad_err_str,
            f"{abs(newton_safe_root - p_star):.2e}",
        ],
        "Iterations": [
            n_grid_main,
            n_grid_main,
            golden_iter,
            newton_good_iter,
            int(newton_bad_history[-1, 0]),
            newton_safe_iter,
        ],
        "Status": [
            "converged",
            "converged",
            "converged",
            newton_good_status,
            newton_bad_status,
            newton_safe_status,
        ],
    })
    Path("tables").mkdir(parents=True, exist_ok=True)
    method_table.to_csv("tables/method_comparison.csv", index=False)

    eps_print = pd.DataFrame({
        "Elasticity": [f"{r['epsilon']:.2f}" for r in eps_rows],
        "Closed-form price": [f"{r['p_star']:.4f}" for r in eps_rows],
        "Lerner markup": [f"{r['Lerner markup']:.4f}" for r in eps_rows],
        "Profit at the optimum": [f"{r['profit at p_star']:.4f}" for r in eps_rows],
        "Golden-section error": [f"{r['golden-section error']:.2e}" for r in eps_rows],
    })
    eps_print.to_csv("tables/elasticity_sensitivity.csv", index=False)

    sweep_table = pd.DataFrame({
        "Starting price": [f"{x:.2f}" for x in starting_points],
        "Iterations": newton_counts,
        "Status": newton_status_list,
        "Above inflection point": ["yes" if x > p_inflect else "no" for x in starting_points],
    })
    sweep_table.to_csv("tables/newton_sensitivity.csv", index=False)

    save_thumbnail("figures/profit-curve.png", "figures/thumb.png")
    print(f"Done: 4 figures, 3 tables")


if __name__ == "__main__":
    main()
