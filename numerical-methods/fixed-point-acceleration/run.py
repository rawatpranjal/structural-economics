#!/usr/bin/env python3
"""Fixed-point iteration and acceleration for solving x = T(x).

Three fixed-point methods are compared on the same test instance:
vanilla Picard iteration, damped Picard, and Anderson acceleration
with memory five. The test instance is plain-logit share inversion,
which has a closed-form benchmark for checking the iterates. A small
Cournot best-response example shows the same machinery on a static
game where the fixed point is a Nash equilibrium.

References:
- Anderson, D. G. (1965) Iterative Procedures for Nonlinear Integral Equations.
- Walker and Ni (2011) Anderson Acceleration for Fixed-Point Iterations.
- Berry, S. (1994) Estimating Discrete-Choice Models of Product Differentiation. (test instance)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def main() -> None:
    # =========================================================================
    # Calibration: four inside products plus an outside option
    # =========================================================================
    delta_star = np.array([1.0, 0.5, -0.3, -1.2])
    n_prod = len(delta_star)

    def predicted_shares(delta):
        e = np.exp(delta)
        denom = 1.0 + e.sum()
        return e / denom, 1.0 / denom

    s_obs, s0_obs = predicted_shares(delta_star)

    tol = 1e-12
    max_iter = 200

    # =========================================================================
    # Fixed-point map: Berry contraction
    # =========================================================================
    def T(delta):
        s, _ = predicted_shares(delta)
        return delta + np.log(s_obs) - np.log(s)

    # =========================================================================
    # Method 1: Picard iteration
    # =========================================================================
    def picard(delta0, max_it=max_iter, tol_=tol):
        delta = np.asarray(delta0, dtype=float).copy()
        history = [delta.copy()]
        residuals = []
        errors = [float(np.linalg.norm(delta - delta_star, np.inf))]
        for _ in range(max_it):
            delta_new = T(delta)
            residuals.append(float(np.linalg.norm(delta_new - delta, np.inf)))
            delta = delta_new
            history.append(delta.copy())
            errors.append(float(np.linalg.norm(delta - delta_star, np.inf)))
            if residuals[-1] < tol_:
                break
        return np.array(history), np.array(residuals), np.array(errors)

    delta0 = np.zeros(n_prod)

    pi_history, pi_residuals, pi_errors = picard(delta0)
    pi_iter = len(pi_residuals)

    # =========================================================================
    # Method 2: damped Picard
    # =========================================================================
    def damped_picard(delta0, alpha=0.5, max_it=max_iter, tol_=tol):
        delta = np.asarray(delta0, dtype=float).copy()
        history = [delta.copy()]
        residuals = []
        errors = [float(np.linalg.norm(delta - delta_star, np.inf))]
        for _ in range(max_it):
            delta_new = (1 - alpha) * delta + alpha * T(delta)
            residuals.append(float(np.linalg.norm(delta_new - delta, np.inf)))
            delta = delta_new
            history.append(delta.copy())
            errors.append(float(np.linalg.norm(delta - delta_star, np.inf)))
            if residuals[-1] < tol_:
                break
        return np.array(history), np.array(residuals), np.array(errors)

    damping = 0.5
    _, dp_residuals, dp_errors = damped_picard(delta0, alpha=damping)
    dp_iter = len(dp_residuals)

    # =========================================================================
    # Method 3: Anderson acceleration with memory m
    # =========================================================================
    def anderson(delta0, m_max=5, max_it=max_iter, tol_=tol, safeguard=True):
        x = np.asarray(delta0, dtype=float).copy()
        g = T(x)
        x_hist = [x.copy()]
        g_hist = [g.copy()]
        residuals = []
        errors = [float(np.linalg.norm(x - delta_star, np.inf))]
        prev_residual = float(np.linalg.norm(g - x, np.inf))
        for _ in range(max_it):
            m_k = min(m_max, len(x_hist) - 1)
            f_k = g_hist[-1] - x_hist[-1]
            if m_k == 0:
                x_new = g_hist[-1]
            else:
                F = np.column_stack([
                    (g_hist[-i] - x_hist[-i]) - (g_hist[-i - 1] - x_hist[-i - 1])
                    for i in range(1, m_k + 1)
                ])
                G = np.column_stack([
                    g_hist[-i] - g_hist[-i - 1]
                    for i in range(1, m_k + 1)
                ])
                gamma, *_ = np.linalg.lstsq(F, f_k, rcond=None)
                x_new = g_hist[-1] - G @ gamma
                if safeguard:
                    g_new_check = T(x_new)
                    new_residual = float(np.linalg.norm(g_new_check - x_new, np.inf))
                    if new_residual > 2.0 * prev_residual:
                        # Anderson step bad; revert to one damped Picard step.
                        x_new = (1 - 0.5) * x_hist[-1] + 0.5 * g_hist[-1]
            x_hist.append(x_new.copy())
            g_hist.append(T(x_new))
            residuals.append(float(np.linalg.norm(g_hist[-1] - x_hist[-1], np.inf)))
            errors.append(float(np.linalg.norm(x_new - delta_star, np.inf)))
            prev_residual = residuals[-1]
            if residuals[-1] < tol_:
                break
        return np.array(x_hist), np.array(residuals), np.array(errors)

    _, an_residuals, an_errors = anderson(delta0, m_max=5)
    an_iter = len(an_residuals)

    # =========================================================================
    # Stress test: shrink the outside share so deltas are larger
    # =========================================================================
    stress_outsides = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    stress_rows = []
    for s0_target in stress_outsides:
        delta_true = np.array([
            np.log(0.40 / s0_target),
            np.log(0.25 / s0_target),
            np.log(0.20 / s0_target),
            np.log(s0_target * 0.5 / s0_target),  # always 0.5
        ])
        # Renormalize so shares actually sum to 1
        # We want: s_j + s_0 = 1 with s_0 = s0_target. Free parameter is delta_3.
        # Solve: e_3 = (1 - s_0_target) / s_0_target - sum_{j<3} e_j
        e = np.exp(delta_true[:3])
        e3 = (1.0 - s0_target) / s0_target - e.sum()
        if e3 <= 0:
            continue
        delta_true_full = np.concatenate([delta_true[:3], [np.log(e3)]])

        s_obs_local, _ = predicted_shares(delta_true_full)
        # Local fixed-point map and methods
        def T_local(delta, s_obs_=s_obs_local):
            s, _ = predicted_shares(delta)
            return delta + np.log(s_obs_) - np.log(s)

        # Picard
        d = np.zeros(n_prod)
        n_pi = 0
        for k in range(max_iter):
            d_new = T_local(d)
            if np.linalg.norm(d_new - d, np.inf) < tol:
                d = d_new
                n_pi = k + 1
                break
            d = d_new
        else:
            n_pi = max_iter
        pi_residual_local = float(np.linalg.norm(T_local(d) - d, np.inf))
        # Anderson
        d_an = np.zeros(n_prod)
        x_hist_l = [d_an.copy()]
        g_hist_l = [T_local(d_an).copy()]
        n_an = 0
        prev_r = float(np.linalg.norm(g_hist_l[-1] - x_hist_l[-1], np.inf))
        m_max_l = 5
        for k in range(max_iter):
            m_k = min(m_max_l, len(x_hist_l) - 1)
            f_k = g_hist_l[-1] - x_hist_l[-1]
            if m_k == 0:
                x_new = g_hist_l[-1]
            else:
                F = np.column_stack([
                    (g_hist_l[-i] - x_hist_l[-i]) - (g_hist_l[-i - 1] - x_hist_l[-i - 1])
                    for i in range(1, m_k + 1)
                ])
                G = np.column_stack([g_hist_l[-i] - g_hist_l[-i - 1] for i in range(1, m_k + 1)])
                gamma, *_ = np.linalg.lstsq(F, f_k, rcond=None)
                x_new = g_hist_l[-1] - G @ gamma
                # Safeguard
                g_check = T_local(x_new)
                new_r = float(np.linalg.norm(g_check - x_new, np.inf))
                if new_r > 2.0 * prev_r:
                    x_new = 0.5 * x_hist_l[-1] + 0.5 * g_hist_l[-1]
            x_hist_l.append(x_new.copy())
            g_hist_l.append(T_local(x_new))
            r = float(np.linalg.norm(g_hist_l[-1] - x_hist_l[-1], np.inf))
            prev_r = r
            if r < tol:
                n_an = k + 1
                break
        else:
            n_an = max_iter
        an_residual_local = float(np.linalg.norm(g_hist_l[-1] - x_hist_l[-1], np.inf))

        stress_rows.append({
            "s_outside": s0_target,
            "picard_iter": n_pi,
            "picard_residual": pi_residual_local,
            "anderson_iter": n_an,
            "anderson_residual": an_residual_local,
        })

    # =========================================================================
    # Cournot best-response mini extension
    # =========================================================================
    a_demand = 10.0
    c_marginal = 1.0
    q_star = (a_demand - c_marginal) / 3.0

    def br(q_other):
        return max(0.0, (a_demand - c_marginal - q_other) / 2.0)

    def cournot_T(q):
        return np.array([br(q[1]), br(q[0])])

    def cournot_picard(q0, alpha=1.0, max_it=max_iter, tol_=tol):
        q = np.asarray(q0, dtype=float).copy()
        residuals = []
        history = [q.copy()]
        for _ in range(max_it):
            q_new = (1 - alpha) * q + alpha * cournot_T(q)
            residuals.append(float(np.linalg.norm(q_new - q, np.inf)))
            q = q_new
            history.append(q.copy())
            if residuals[-1] < tol_:
                break
        return np.array(history), np.array(residuals)

    q0 = np.array([0.0, 0.0])
    cournot_pi_hist, cournot_pi_res = cournot_picard(q0, alpha=1.0)
    cournot_dp_hist, cournot_dp_res = cournot_picard(q0, alpha=0.5)

    # =========================================================================
    # Figures and tables
    # =========================================================================
    setup_style()

    # ------------------------------------------------------------------
    # Figure 1: shares observed vs predicted at start, mid, end
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    products = [f"Product {j+1}" for j in range(n_prod)]
    width = 0.18
    pos = np.arange(n_prod)
    pred_initial, _ = predicted_shares(delta0)
    mid_idx = max(1, pi_iter // 4)
    pred_mid, _ = predicted_shares(pi_history[mid_idx])
    pred_final, _ = predicted_shares(pi_history[-1])
    ax1.bar(pos - 1.5 * width, s_obs, width, color="tab:blue", label="Observed")
    ax1.bar(pos - 0.5 * width, pred_initial, width, color="tab:gray", alpha=0.7,
            label=fr"Initial $\delta^0 = 0$")
    ax1.bar(pos + 0.5 * width, pred_mid, width, color="tab:orange", alpha=0.85,
            label=fr"Picard iterate $t = {mid_idx}$")
    ax1.bar(pos + 1.5 * width, pred_final, width, color="tab:green", alpha=0.85,
            label=fr"Picard iterate $t = {pi_iter}$ (final)")
    ax1.set_xticks(pos)
    ax1.set_xticklabels(products)
    ax1.set_ylabel("Inside-product share")
    ax1.set_title("Observed and predicted shares as Picard iterates approach the fixed point")
    ax1.legend(loc="upper right", fontsize=9)
    save_figure(fig1, "figures/share-fit.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 2: residual and error convergence on log scale
    # ------------------------------------------------------------------
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    ax2a.semilogy(np.arange(1, len(pi_residuals) + 1), np.maximum(pi_residuals, 1e-16),
                  "o-", color="tab:orange", markersize=3, linewidth=1.5, label="Picard")
    ax2a.semilogy(np.arange(1, len(dp_residuals) + 1), np.maximum(dp_residuals, 1e-16),
                  "s-", color="tab:purple", markersize=3, linewidth=1.5, label=fr"Damped Picard ($\alpha = {damping}$)")
    ax2a.semilogy(np.arange(1, len(an_residuals) + 1), np.maximum(an_residuals, 1e-16),
                  "d-", color="tab:green", markersize=4, linewidth=1.5, label="Anderson ($m = 5$)")
    ax2a.set_xlabel("Iteration $t$")
    ax2a.set_ylabel(r"$\| T(\delta^t) - \delta^t \|_\infty$")
    ax2a.set_title("Fixed-point residual across iterations")
    ax2a.legend(loc="upper right", fontsize=9)

    ax2b.semilogy(np.arange(len(pi_errors)), np.maximum(pi_errors, 1e-16),
                  "o-", color="tab:orange", markersize=3, linewidth=1.5, label="Picard")
    ax2b.semilogy(np.arange(len(dp_errors)), np.maximum(dp_errors, 1e-16),
                  "s-", color="tab:purple", markersize=3, linewidth=1.5, label=fr"Damped Picard ($\alpha = {damping}$)")
    ax2b.semilogy(np.arange(len(an_errors)), np.maximum(an_errors, 1e-16),
                  "d-", color="tab:green", markersize=4, linewidth=1.5, label="Anderson ($m = 5$)")
    ax2b.set_xlabel("Iteration $t$")
    ax2b.set_ylabel(r"$\| \delta^t - \delta^{\ast} \|_\infty$")
    ax2b.set_title("Distance from the closed-form benchmark")
    ax2b.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    save_figure(fig2, "figures/convergence.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3: stress test for shrinking outside share
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    s0_arr = np.array([r["s_outside"] for r in stress_rows])
    pi_arr = np.array([r["picard_iter"] for r in stress_rows])
    an_arr = np.array([r["anderson_iter"] for r in stress_rows])
    ax3.plot(s0_arr, pi_arr, "o-", color="tab:orange", linewidth=1.5, markersize=6, label="Picard")
    ax3.plot(s0_arr, an_arr, "d-", color="tab:green", linewidth=1.5, markersize=6, label="Anderson")
    ax3.set_xscale("log")
    ax3.set_xlabel("Outside share $s_0^{\\mathrm{obs}}$")
    ax3.set_ylabel("Iterations to tolerance")
    ax3.set_title("Iteration count vs outside share")
    ax3.invert_xaxis()
    ax3.legend(loc="upper left", fontsize=9)
    save_figure(fig3, "figures/stress-test.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 4: Cournot best-response paths
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    q_grid = np.linspace(0, 6, 200)
    ax4.plot(q_grid, (a_demand - c_marginal - q_grid) / 2.0, color="tab:blue",
             linewidth=1.5, label=r"Best response $q_1 = (a - c - q_2) / 2$")
    ax4.plot((a_demand - c_marginal - q_grid) / 2.0, q_grid, color="tab:red",
             linewidth=1.5, label=r"Best response $q_2 = (a - c - q_1) / 2$")
    ax4.plot(cournot_pi_hist[:8, 0], cournot_pi_hist[:8, 1], "o-", color="tab:orange",
             markersize=5, linewidth=1.0, alpha=0.8, label="Picard, oscillating")
    ax4.plot(cournot_dp_hist[:8, 0], cournot_dp_hist[:8, 1], "s-", color="tab:green",
             markersize=5, linewidth=1.0, alpha=0.8, label="Damped Picard, monotone")
    ax4.plot(q_star, q_star, "*", color="tab:red", markersize=18,
             label=fr"$q^{{\ast}} = ({q_star:.2f},\, {q_star:.2f})$")
    ax4.set_xlabel(r"$q_1$")
    ax4.set_ylabel(r"$q_2$")
    ax4.set_title("Cournot best-response iteration to the symmetric Nash quantity")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.set_xlim(0, 6)
    ax4.set_ylim(0, 6)
    ax4.set_aspect("equal")
    save_figure(fig4, "figures/cournot-best-response.png", dpi=150)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def termination_status(residual, n_iter, max_it=max_iter, tol_=tol):
        """Report convergence honestly from the termination condition.

        A method "converged" only if its final residual met the
        sup-norm tolerance. If the loop instead exhausted max_iter,
        report that explicitly so the table does not claim a method
        reached tolerance when it did not.
        """
        if residual < tol_:
            return "converged"
        return f"stopped at max_iter = {max_it}"

    pi_status = termination_status(pi_residuals[-1], pi_iter)
    dp_status = termination_status(dp_residuals[-1], dp_iter)
    an_status = termination_status(an_residuals[-1], an_iter)

    method_table = pd.DataFrame({
        "Method": ["Picard", "Damped Picard", "Anderson (m = 5)"],
        "Setting": [
            "no damping",
            f"damping alpha = {damping}",
            "memory 5 with residual safeguard",
        ],
        "Iterations": [pi_iter, dp_iter, an_iter],
        "Final residual": [
            f"{pi_residuals[-1]:.2e}",
            f"{dp_residuals[-1]:.2e}",
            f"{an_residuals[-1]:.2e}",
        ],
        "Distance to closed form": [
            f"{pi_errors[-1]:.2e}",
            f"{dp_errors[-1]:.2e}",
            f"{an_errors[-1]:.2e}",
        ],
        "Status": [pi_status, dp_status, an_status],
    })
    Path("tables").mkdir(parents=True, exist_ok=True)
    method_table.to_csv("tables/method_comparison.csv", index=False)

    stress_print = pd.DataFrame({
        "Outside share": [f"{r['s_outside']:.2f}" for r in stress_rows],
        "Picard iterations": [r["picard_iter"] for r in stress_rows],
        "Picard residual": [f"{r['picard_residual']:.2e}" for r in stress_rows],
        "Anderson iterations": [r["anderson_iter"] for r in stress_rows],
        "Anderson residual": [f"{r['anderson_residual']:.2e}" for r in stress_rows],
    })
    stress_print.to_csv("tables/stress_test.csv", index=False)

    # Cournot table
    cournot_pi_iter = len(cournot_pi_res)
    cournot_dp_iter = len(cournot_dp_res)
    cournot_table = pd.DataFrame({
        "Method": ["Vanilla Picard", "Damped Picard"],
        "Quantity firm 1": [
            f"{cournot_pi_hist[-1, 0]:.4f}",
            f"{cournot_dp_hist[-1, 0]:.4f}",
        ],
        "Quantity firm 2": [
            f"{cournot_pi_hist[-1, 1]:.4f}",
            f"{cournot_dp_hist[-1, 1]:.4f}",
        ],
        "Iterations": [cournot_pi_iter, cournot_dp_iter],
        "Final residual": [
            f"{cournot_pi_res[-1]:.2e}",
            f"{cournot_dp_res[-1]:.2e}",
        ],
    })
    cournot_table.to_csv("tables/cournot_summary.csv", index=False)

    save_thumbnail("figures/share-fit.png", "figures/thumb.png")
    print(f"Generated: figures/ (4 figures + thumb) + tables/ (3 tables)")


if __name__ == "__main__":
    main()
