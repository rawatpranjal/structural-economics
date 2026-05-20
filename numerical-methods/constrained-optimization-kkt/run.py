#!/usr/bin/env python3
"""Constrained optimization for budget allocation across diminishing-returns projects.

A planner allocates a fixed budget across three projects with diminishing
marginal returns and non-negativity bounds. The unconstrained-on-bounds
solver returns a negative allocation, which is the failure baseline that
motivates the KKT machinery. Four methods are compared: a closed-form
Lagrangian on the budget alone, projected gradient onto the simplex, an
interior-point log barrier, and SLSQP via scipy. KKT residuals on
stationarity, primal feasibility, dual feasibility, and complementary
slackness are the diagnostics, not the value of the objective alone.

References:
- Boyd and Vandenberghe (2004) Convex Optimization, Ch. 5, 11.
- Nocedal and Wright (2006) Numerical Optimization, Ch. 12, 17, 19.
- Bertsekas (1999) Nonlinear Programming, Ch. 2-3.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from scipy.optimize import brentq, minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def main() -> None:
    # =========================================================================
    # Calibration: three projects, diagonal Hessian, binding budget
    # =========================================================================
    a = np.array([4.0, 3.0, 0.5])
    B = np.eye(3)
    I_total = 3.0
    n_proj = 3

    x_star = np.array([2.0, 1.0, 0.0])
    lambda_star = 2.0
    mu_star = np.array([0.0, 0.0, 1.5])
    u_star = float(a @ x_star - 0.5 * x_star @ B @ x_star)

    def utility(x):
        return float(a @ x - 0.5 * x @ B @ x)

    def utility_grad(x):
        return a - B @ x

    # =========================================================================
    # Projection onto the budget simplex {x >= 0, sum(x) = I_total}
    # =========================================================================
    def project_simplex(v, total):
        n = len(v)
        u_sorted = np.sort(v)[::-1]
        cssv = np.cumsum(u_sorted) - total
        idx = np.arange(1, n + 1)
        cond = u_sorted - cssv / idx > 0
        rho = int(np.where(cond)[0][-1])
        theta = cssv[rho] / (rho + 1)
        return np.maximum(v - theta, 0.0)

    # =========================================================================
    # Multiplier recovery from a candidate x
    # =========================================================================
    eps_active = 1e-6

    def recover_multipliers(x):
        active = x > eps_active
        grad_neg = a - B @ x
        if active.any():
            lam = float(np.mean(grad_neg[active]))
        else:
            lam = float(np.max(a))
        mu = np.zeros_like(x)
        for j in range(len(x)):
            if not active[j]:
                mu[j] = max(0.0, lam - float(grad_neg[j]))
        return lam, mu

    def kkt_residuals(x, lam=None, mu=None):
        if lam is None or mu is None:
            lam, mu = recover_multipliers(x)
        stat = float(np.linalg.norm((a - B @ x) - lam * np.ones_like(x) + mu))
        primal_eq = abs(float(x.sum()) - I_total)
        primal_neg = float(np.sum(np.maximum(-x, 0.0)))
        primal = primal_eq + primal_neg
        dual = max(0.0, -lam) + float(np.sum(np.maximum(-mu, 0.0)))
        compl = float(np.sum(mu * np.abs(x))) + abs(lam * (I_total - float(x.sum())))
        return stat, primal, dual, compl

    # =========================================================================
    # Method 1: Lagrangian on the budget alone (fails on non-negativity)
    # =========================================================================
    lam_baseline = (a.sum() - I_total) / n_proj
    x_baseline = a - lam_baseline
    bl_stat, bl_primal, bl_dual, bl_compl = kkt_residuals(x_baseline)
    bl_lam, bl_mu = recover_multipliers(x_baseline)
    bl_utility = utility(x_baseline)

    # =========================================================================
    # Method 2: projected gradient onto the budget simplex
    # =========================================================================
    step = 0.25
    pg_x = np.array([0.5, 0.5, 2.0])  # heavy on project 3, far from x*
    pg_history = [pg_x.copy()]
    pg_residuals = [float(np.linalg.norm(pg_x - x_star))]
    pg_kkt_trace = [kkt_residuals(pg_x)]
    pg_max_iter = 500
    pg_tol = 1e-12
    for _ in range(1, pg_max_iter + 1):
        grad = utility_grad(pg_x)
        pg_x = project_simplex(pg_x + step * grad, I_total)
        pg_history.append(pg_x.copy())
        pg_residuals.append(float(np.linalg.norm(pg_x - x_star)))
        pg_kkt_trace.append(kkt_residuals(pg_x))
        if pg_residuals[-1] < pg_tol:
            break
    pg_iter = len(pg_history) - 1
    pg_x_final = pg_history[-1]
    pg_lam, pg_mu = recover_multipliers(pg_x_final)
    pg_stat, pg_primal, pg_dual, pg_compl = kkt_residuals(pg_x_final)
    pg_utility = utility(pg_x_final)

    # =========================================================================
    # Method 3: interior-point log barrier
    # =========================================================================
    def x_of_lambda(lam, t):
        d = a - lam
        return 0.5 * (d + np.sqrt(d ** 2 + 4 * t))

    def find_lambda(t):
        return brentq(lambda lam: x_of_lambda(lam, t).sum() - I_total, -100.0, 100.0, xtol=1e-13)

    barriers = [10.0, 1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
    barrier_history = []
    barrier_residuals = []
    barrier_kkt_trace = []
    for t in barriers:
        lam_t = find_lambda(t)
        x_t = x_of_lambda(lam_t, t)
        barrier_history.append((t, x_t, lam_t))
        barrier_residuals.append(float(np.linalg.norm(x_t - x_star)))
        # Exact barrier-derived multipliers: mu_j = t / x_j on the central path.
        mu_t = t / x_t
        barrier_kkt_trace.append(kkt_residuals(x_t, lam_t, mu_t))
    barrier_iter = len(barriers)
    barrier_x_final = barrier_history[-1][1]
    barrier_lam_final = barrier_history[-1][2]
    barrier_lam, barrier_mu = recover_multipliers(barrier_x_final)
    barrier_stat, barrier_primal, barrier_dual, barrier_compl = kkt_residuals(barrier_x_final)
    barrier_utility = utility(barrier_x_final)

    # =========================================================================
    # Method 4: SLSQP via scipy.optimize.minimize
    # =========================================================================
    def neg_utility(x):
        return -utility(x)

    def neg_utility_grad(x):
        return -utility_grad(x)

    slsqp_result = minimize(
        neg_utility,
        x0=np.array([1.0, 1.0, 1.0]),
        jac=neg_utility_grad,
        method='SLSQP',
        bounds=[(0.0, None)] * n_proj,
        constraints=[{
            'type': 'eq',
            'fun': lambda x: float(x.sum() - I_total),
            'jac': lambda x: np.ones(n_proj),
        }],
        options={'ftol': 1e-12, 'maxiter': 200, 'disp': False},
    )
    x_slsqp = np.asarray(slsqp_result.x, dtype=float)
    slsqp_iter = int(slsqp_result.nit)
    slsqp_lam, slsqp_mu = recover_multipliers(x_slsqp)
    slsqp_stat, slsqp_primal, slsqp_dual, slsqp_compl = kkt_residuals(x_slsqp, slsqp_lam, slsqp_mu)
    slsqp_utility = utility(x_slsqp)

    # =========================================================================
    # Figures and tables
    # =========================================================================
    setup_style()

    # ------------------------------------------------------------------
    # Figure 1: projected-gradient path on the budget simplex
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    triangle = Polygon(
        [(0.0, 0.0), (I_total, 0.0), (0.0, I_total)],
        closed=True, facecolor="tab:blue", edgecolor="tab:blue", alpha=0.08, linewidth=1.5,
    )
    ax1.add_patch(triangle)
    ax1.plot([0.0, I_total, 0.0, 0.0], [0.0, 0.0, I_total, 0.0],
             color="tab:blue", linewidth=1.5, alpha=0.6)
    pg_arr = np.array(pg_history)
    ax1.plot(pg_arr[:, 0], pg_arr[:, 1], "-", color="tab:orange", linewidth=1.0, alpha=0.7)
    n_show_pg = min(15, len(pg_arr))
    ax1.plot(pg_arr[:n_show_pg, 0], pg_arr[:n_show_pg, 1],
             "o", color="tab:orange", markersize=5, label="Projected gradient iterate")
    ax1.plot(pg_arr[0, 0], pg_arr[0, 1], "o", color="tab:gray", markersize=8,
             markeredgecolor="black", label=fr"Start $x_0 = (0.5,\, 0.5,\, 2.0)$")
    ax1.plot(x_star[0], x_star[1], "*", color="tab:red", markersize=18,
             label=fr"$x^{{\ast}} = (2,\, 1,\, 0)$")
    ax1.text(I_total + 0.05, 0.05, "$x_3 = 0$\n(project 3 inactive)", fontsize=9, color="tab:purple")
    ax1.set_xlabel("Project 1 allocation $x_1$")
    ax1.set_ylabel("Project 2 allocation $x_2$")
    ax1.set_title(fr"Projected gradient on the budget simplex (project 3 implicit, $x_3 = I - x_1 - x_2$)")
    ax1.set_xlim(-0.2, I_total + 0.5)
    ax1.set_ylim(-0.2, I_total + 0.5)
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", fontsize=9)
    save_figure(fig1, "figures/simplex-paths.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 2: barrier path approaching the boundary
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    triangle2 = Polygon(
        [(0.0, 0.0), (I_total, 0.0), (0.0, I_total)],
        closed=True, facecolor="tab:blue", edgecolor="tab:blue", alpha=0.08, linewidth=1.5,
    )
    ax2.add_patch(triangle2)
    ax2.plot([0.0, I_total, 0.0, 0.0], [0.0, 0.0, I_total, 0.0],
             color="tab:blue", linewidth=1.5, alpha=0.6)
    barrier_arr = np.array([h[1] for h in barrier_history])
    ax2.plot(barrier_arr[:, 0], barrier_arr[:, 1], "-", color="tab:purple",
             linewidth=1.5, alpha=0.7, label="Central path")
    for i, (t, x_t, _) in enumerate(barrier_history):
        ax2.plot(x_t[0], x_t[1], "d", color="tab:purple", markersize=7)
    # Label only the first and last points to avoid clutter.
    t0, x0_arr, _ = barrier_history[0]
    tend, xend, _ = barrier_history[-1]
    ax2.annotate(fr"$t = {t0:.0e}$ (most interior)",
                 xy=(x0_arr[0], x0_arr[1]), xytext=(x0_arr[0] - 0.55, x0_arr[1] + 0.5),
                 fontsize=9, color="tab:purple",
                 arrowprops=dict(arrowstyle="->", color="tab:purple", linewidth=0.8, alpha=0.7))
    ax2.annotate(fr"$t = {tend:.0e}$ (near $x^{{\ast}}$)",
                 xy=(xend[0], xend[1]), xytext=(xend[0] + 0.20, xend[1] - 0.5),
                 fontsize=9, color="tab:purple",
                 arrowprops=dict(arrowstyle="->", color="tab:purple", linewidth=0.8, alpha=0.7))
    ax2.plot(x_star[0], x_star[1], "*", color="tab:red", markersize=18,
             label=fr"$x^{{\ast}} = (2,\, 1,\, 0)$")
    ax2.set_xlabel("Project 1 allocation $x_1$")
    ax2.set_ylabel("Project 2 allocation $x_2$")
    ax2.set_title("Interior-point central path as the barrier shrinks")
    ax2.set_xlim(-0.2, I_total + 0.5)
    ax2.set_ylim(-0.2, I_total + 0.5)
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=9)
    save_figure(fig2, "figures/barrier-path.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3: KKT residuals over iterations
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

    pg_kkt = np.array(pg_kkt_trace)
    pg_iters_axis = np.arange(len(pg_kkt))
    ax3a.semilogy(pg_iters_axis, np.maximum(pg_kkt[:, 0], 1e-16), "-", color="tab:orange",
                  linewidth=1.5, label="Stationarity")
    ax3a.semilogy(pg_iters_axis, np.maximum(pg_kkt[:, 1], 1e-16), "-", color="tab:green",
                  linewidth=1.5, label="Primal feasibility")
    ax3a.semilogy(pg_iters_axis, np.maximum(pg_kkt[:, 3], 1e-16), "-", color="tab:purple",
                  linewidth=1.5, label="Complementarity")
    ax3a.set_xlabel("Iteration $k$")
    ax3a.set_ylabel("KKT residual")
    ax3a.set_title("Projected gradient: KKT residuals across iterations")
    ax3a.legend(loc="upper right", fontsize=9)

    barrier_kkt = np.array(barrier_kkt_trace)
    barrier_idx = np.arange(len(barrier_kkt))
    ax3b.semilogy(barrier_idx, np.maximum(barrier_kkt[:, 0], 1e-16), "d-", color="tab:orange",
                  markersize=5, linewidth=1.5, label="Stationarity")
    ax3b.semilogy(barrier_idx, np.maximum(barrier_kkt[:, 1], 1e-16), "d-", color="tab:green",
                  markersize=5, linewidth=1.5, label="Primal feasibility")
    ax3b.semilogy(barrier_idx, np.maximum(barrier_kkt[:, 3], 1e-16), "d-", color="tab:purple",
                  markersize=5, linewidth=1.5, label="Complementarity")
    ax3b.set_xticks(barrier_idx)
    ax3b.set_xticklabels([f"{t:.0e}" for t in barriers], rotation=45, fontsize=8)
    ax3b.set_xlabel("Barrier parameter $t$")
    ax3b.set_ylabel("KKT residual")
    ax3b.set_title("Interior point: KKT residuals along the central path")
    ax3b.legend(loc="upper right", fontsize=9)
    fig3.tight_layout()
    save_figure(fig3, "figures/kkt-residuals.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 4: shadow prices at the SLSQP optimum
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    labels = [r"Budget $\lambda$",
              r"Project 1 bound $\mu_1$",
              r"Project 2 bound $\mu_2$",
              r"Project 3 bound $\mu_3$"]
    closed_form = [lambda_star, mu_star[0], mu_star[1], mu_star[2]]
    slsqp_vals = [slsqp_lam, slsqp_mu[0], slsqp_mu[1], slsqp_mu[2]]
    pos = np.arange(len(labels))
    width = 0.35
    ax4.bar(pos - width / 2, closed_form, width, color="tab:gray", alpha=0.7,
            label="Closed form")
    ax4.bar(pos + width / 2, slsqp_vals, width, color="tab:red", alpha=0.85,
            label="SLSQP recovery")
    ax4.set_xticks(pos)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_ylabel("Multiplier value")
    ax4.set_title("Shadow prices: closed form vs SLSQP recovery")
    ax4.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax4.legend(loc="upper right", fontsize=9)
    for i, (cf, sl) in enumerate(zip(closed_form, slsqp_vals)):
        if abs(cf) > 1e-8 or abs(sl) > 1e-8:
            ax4.text(i, max(cf, sl) + 0.05, fr"$\lambda^{{\ast}} = {cf:.2f}$" if i == 0 else fr"${cf:.2f}$",
                     ha="center", fontsize=8)
    save_figure(fig4, "figures/shadow-prices.png", dpi=150)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    solution_table = pd.DataFrame({
        "Method": [
            "Baseline failure: Lagrangian, budget only",
            "Method 1: Projected gradient",
            "Method 2: Interior-point log barrier",
            "Method 3: SLSQP",
            "Closed form",
        ],
        "Project 1": [
            f"{x_baseline[0]:.4f}",
            f"{pg_x_final[0]:.4f}",
            f"{barrier_x_final[0]:.4f}",
            f"{x_slsqp[0]:.4f}",
            f"{x_star[0]:.4f}",
        ],
        "Project 2": [
            f"{x_baseline[1]:.4f}",
            f"{pg_x_final[1]:.4f}",
            f"{barrier_x_final[1]:.4f}",
            f"{x_slsqp[1]:.4f}",
            f"{x_star[1]:.4f}",
        ],
        "Project 3": [
            f"{x_baseline[2]:.4f}",
            f"{pg_x_final[2]:.4f}",
            f"{barrier_x_final[2]:.4f}",
            f"{x_slsqp[2]:.4f}",
            f"{x_star[2]:.4f}",
        ],
        "Total spend": [
            f"{x_baseline.sum():.4f}",
            f"{pg_x_final.sum():.4f}",
            f"{barrier_x_final.sum():.4f}",
            f"{x_slsqp.sum():.4f}",
            f"{x_star.sum():.4f}",
        ],
        "Utility": [
            f"{bl_utility:.4f}",
            f"{pg_utility:.4f}",
            f"{barrier_utility:.4f}",
            f"{slsqp_utility:.4f}",
            f"{u_star:.4f}",
        ],
        "Iterations": [
            "1 (closed form)",
            f"{pg_iter}",
            f"{barrier_iter} barrier values",
            f"{slsqp_iter}",
            "n/a",
        ],
        "Feasible?": [
            "no, x_3 < 0",
            "yes",
            "yes",
            "yes",
            "yes",
        ],
    })
    Path("tables").mkdir(parents=True, exist_ok=True)
    solution_table.to_csv("tables/solution_comparison.csv", index=False)

    kkt_table = pd.DataFrame({
        "Method": [
            "Baseline failure: Lagrangian, budget only",
            "Method 1: Projected gradient",
            "Method 2: Interior-point log barrier",
            "Method 3: SLSQP",
        ],
        "Stationarity error": [
            f"{bl_stat:.2e}",
            f"{pg_stat:.2e}",
            f"{barrier_stat:.2e}",
            f"{slsqp_stat:.2e}",
        ],
        "Feasibility error": [
            f"{bl_primal:.2e}",
            f"{pg_primal:.2e}",
            f"{barrier_primal:.2e}",
            f"{slsqp_primal:.2e}",
        ],
        "Dual feasibility error": [
            f"{bl_dual:.2e}",
            f"{pg_dual:.2e}",
            f"{barrier_dual:.2e}",
            f"{slsqp_dual:.2e}",
        ],
        "Complementarity error": [
            f"{bl_compl:.2e}",
            f"{pg_compl:.2e}",
            f"{barrier_compl:.2e}",
            f"{slsqp_compl:.2e}",
        ],
        "Active constraints recovered": [
            "budget only (mis-recovered)",
            "budget; project 3 bound",
            "budget; project 3 bound",
            "budget; project 3 bound",
        ],
    })
    kkt_table.to_csv("tables/kkt_check.csv", index=False)

    shadow_table = pd.DataFrame({
        "Constraint": [
            "Budget $\\sum_j x_j \\leq I$",
            "Project 1 bound $x_1 \\geq 0$",
            "Project 2 bound $x_2 \\geq 0$",
            "Project 3 bound $x_3 \\geq 0$",
        ],
        "Multiplier": [
            f"{lambda_star:.2f}",
            f"{mu_star[0]:.2f}",
            f"{mu_star[1]:.2f}",
            f"{mu_star[2]:.2f}",
        ],
        "Status": [
            "binding",
            "slack",
            "slack",
            "binding",
        ],
        "Economic interpretation": [
            "Utility gain from one extra unit of budget",
            "Project 1 receives interior allocation; bound has no value",
            "Project 2 receives interior allocation; bound has no value",
            "Utility loss avoided by holding project 3 at zero",
        ],
    })
    shadow_table.to_csv("tables/shadow_prices.csv", index=False)

    save_thumbnail("figures/simplex-paths.png", "figures/thumb.png")
    print(f"Generated: figures/ (4 figures + thumb) + tables/ (3 tables)")


if __name__ == "__main__":
    main()
