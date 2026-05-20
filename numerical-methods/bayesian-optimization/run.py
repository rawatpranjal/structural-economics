#!/usr/bin/env python3
"""Bayesian optimization with a Gaussian-process surrogate on a nonconcave pricing objective.

A monopolist sells to two consumer segments. The mixture profit function has
two local maxima. Earlier we showed that multi-start L-BFGS-B, random search,
and simulated annealing all recover the global peak but spend hundreds to
thousands of function evaluations doing so. Bayesian optimization instead
fits a probabilistic surrogate to the evaluations it already has, then
chooses each next evaluation to maximize an acquisition function that
balances exploration and exploitation. Thirty evaluations are enough.

The surrogate is a Gaussian process with a squared-exponential kernel.
The acquisition rule is Expected Improvement. The comparison is run on the
same objective and bracket as `numerical-methods/global-search-multistart/`
so the per-evaluation budget is directly comparable.

References:
- Mockus, J., Tiesis, V., and Zilinskas, A. (1978). The application of Bayesian methods for seeking the extremum.
- Jones, D. R., Schonlau, M., and Welch, W. J. (1998). Efficient Global Optimization of Expensive Black-Box Functions.
- Snoek, J., Larochelle, H., and Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms.
- Frazier, P. I. (2018). A Tutorial on Bayesian Optimization.
- Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian Processes for Machine Learning, Ch. 2 and 5.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm
from scipy.optimize import minimize, dual_annealing

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


# =============================================================================
# Gaussian process surrogate with squared-exponential kernel
# =============================================================================
def rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float, sigma_f: float) -> np.ndarray:
    """Squared-exponential kernel for 1-D inputs."""
    x1 = np.atleast_1d(x1).astype(float)
    x2 = np.atleast_1d(x2).astype(float)
    diff = x1[:, None] - x2[None, :]
    return (sigma_f ** 2) * np.exp(-0.5 * (diff / length_scale) ** 2)


class GaussianProcess:
    """Gaussian-process regressor with a squared-exponential kernel.

    The mean function is the training-target mean. The kernel signal variance
    and length scale are user-supplied; the noise variance is small to
    represent the (essentially deterministic) profit evaluations.
    """

    def __init__(self, length_scale: float = 1.0, sigma_f: float = 2.0, sigma_n: float = 1e-3):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.X = None
        self.y = None
        self.y_mean = 0.0
        self._cho = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.X = np.atleast_1d(X).astype(float)
        self.y = np.atleast_1d(y).astype(float)
        self.y_mean = float(self.y.mean())
        y_centered = self.y - self.y_mean
        K = rbf_kernel(self.X, self.X, self.length_scale, self.sigma_f)
        K += (self.sigma_n ** 2) * np.eye(len(self.X))
        self._cho = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._cho, y_centered)
        return self

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_star = np.atleast_1d(X_star).astype(float)
        K_s = rbf_kernel(self.X, X_star, self.length_scale, self.sigma_f)
        mu = self.y_mean + K_s.T @ self._alpha
        v = cho_solve(self._cho, K_s)
        var = (self.sigma_f ** 2) - np.sum(K_s * v, axis=0)
        var = np.clip(var, 1e-10, None)
        return mu, np.sqrt(var)

    def log_marginal_likelihood(self) -> float:
        L = self._cho[0]
        y_centered = self.y - self.y_mean
        n = len(self.y)
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        return float(-0.5 * y_centered @ self._alpha - 0.5 * log_det_K - 0.5 * n * np.log(2.0 * np.pi))


def fit_length_scale(X: np.ndarray, y: np.ndarray, sigma_f: float, sigma_n: float,
                     length_grid: np.ndarray) -> float:
    """Pick the length scale that maximizes the GP log marginal likelihood."""
    best_ll, best_ell = -np.inf, float(length_grid[0])
    for ell in length_grid:
        gp = GaussianProcess(length_scale=float(ell), sigma_f=sigma_f, sigma_n=sigma_n).fit(X, y)
        ll = gp.log_marginal_likelihood()
        if ll > best_ll:
            best_ll, best_ell = ll, float(ell)
    return best_ell


# =============================================================================
# Acquisition functions
# =============================================================================
def expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float, xi: float = 0.0) -> np.ndarray:
    """Expected Improvement for maximization.

    EI(x) = (mu - f_best - xi) * Phi(z) + sigma * phi(z),
    z = (mu - f_best - xi) / sigma when sigma > 0, and zero otherwise.
    """
    sigma = np.maximum(sigma, 1e-9)
    improvement = mu - f_best - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    # -------------------------------------------------------------------------
    # Calibration: same two-segment monopoly as global-search-multistart
    # -------------------------------------------------------------------------
    A_L, b_L = 10.0, 5.0
    A_H, b_H = 8.0, 1.0
    c = 0.5
    lam = 0.6

    p_low_peak = 10.9 / 6.8
    profit_low_peak = (p_low_peak - c) * (lam * (A_L - b_L * p_low_peak) + (1 - lam) * (A_H - b_H * p_low_peak))
    p_high_peak = 4.25
    profit_high_peak = (p_high_peak - c) * (1 - lam) * (A_H - b_H * p_high_peak)
    p_kink = A_L / b_L

    p_lo, p_hi = c + 1e-3, 8.0
    p_global, profit_global = (p_high_peak, profit_high_peak) if profit_high_peak >= profit_low_peak else (p_low_peak, profit_low_peak)

    def D_L(p):
        return np.maximum(0.0, A_L - b_L * p)

    def D_H(p):
        return np.maximum(0.0, A_H - b_H * p)

    def profit(p):
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = (p_arr - c) * (lam * D_L(p_arr) + (1 - lam) * D_H(p_arr))
        return float(out.item()) if out.size == 1 else out

    def neg_profit_scalar(p):
        if np.ndim(p) > 0:
            return -float(profit(float(p[0])))
        return -float(profit(float(p)))

    # -------------------------------------------------------------------------
    # Bayesian-optimization configuration
    # -------------------------------------------------------------------------
    n_initial = 5
    n_iter_bo = 25
    n_total = n_initial + n_iter_bo
    seed_bo = 0
    sigma_f = 2.0
    sigma_n = 1e-3
    length_grid = np.linspace(0.3, 2.5, 12)
    acq_grid = np.linspace(p_lo, p_hi, 1001)
    xi_ei = 0.0

    # -------------------------------------------------------------------------
    # BO with Expected Improvement (main run, snapshots saved for plotting)
    # -------------------------------------------------------------------------
    snapshot_iters = [n_initial, 10, 20, n_total]

    def take_snapshot(X_list, y_list):
        X_arr = np.array(X_list)
        y_arr = np.array(y_list)
        f_best = float(np.max(y_arr))
        ell = fit_length_scale(X_arr, y_arr, sigma_f, sigma_n, length_grid)
        gp = GaussianProcess(length_scale=ell, sigma_f=sigma_f, sigma_n=sigma_n).fit(X_arr, y_arr)
        mu, sd = gp.predict(acq_grid)
        ei = expected_improvement(mu, sd, f_best, xi=xi_ei)
        return {
            "X": X_arr.copy(),
            "y": y_arr.copy(),
            "mu": mu.copy(),
            "sd": sd.copy(),
            "acq": ei.copy(),
            "ell": ell,
            "next_x": float(acq_grid[int(np.argmax(ei))]),
        }

    rng = np.random.default_rng(seed_bo)
    X_ei = list(rng.uniform(p_lo, p_hi, n_initial))
    y_ei = [float(profit(x)) for x in X_ei]
    bo_log = []
    snapshots = {}

    # Log the initial design.
    for i in range(n_initial):
        bo_log.append({
            "iteration": i + 1,
            "phase": "initial",
            "x": X_ei[i],
            "f": y_ei[i],
            "best_so_far": float(np.max(y_ei[: i + 1])),
        })
    if n_initial in snapshot_iters:
        snapshots[n_initial] = take_snapshot(X_ei, y_ei)

    # EI-guided iterations.
    for t in range(n_initial + 1, n_total + 1):
        X_arr = np.array(X_ei)
        y_arr = np.array(y_ei)
        f_best = float(np.max(y_arr))
        ell = fit_length_scale(X_arr, y_arr, sigma_f, sigma_n, length_grid)
        gp = GaussianProcess(length_scale=ell, sigma_f=sigma_f, sigma_n=sigma_n).fit(X_arr, y_arr)
        mu, sd = gp.predict(acq_grid)
        ei = expected_improvement(mu, sd, f_best, xi=xi_ei)
        x_next = float(acq_grid[int(np.argmax(ei))])
        y_next = float(profit(x_next))
        X_ei.append(x_next)
        y_ei.append(y_next)
        bo_log.append({
            "iteration": t,
            "phase": "EI-guided",
            "x": x_next,
            "f": y_next,
            "best_so_far": float(np.max(y_ei)),
        })
        if t in snapshot_iters:
            snapshots[t] = take_snapshot(X_ei, y_ei)

    bo_log_df = pd.DataFrame(bo_log)
    best_bo_ei = float(np.max(y_ei))
    p_bo_ei = float(X_ei[int(np.argmax(y_ei))])
    eval_to_global_ei = next((row["iteration"] for row in bo_log if row["best_so_far"] >= profit_global - 1e-3), None)

    # -------------------------------------------------------------------------
    # Comparison baselines on the same objective
    # -------------------------------------------------------------------------
    def best_so_far(values: list[float]) -> np.ndarray:
        return np.maximum.accumulate(np.asarray(values, dtype=float))

    # Random search
    n_random = 500
    rng_rs = np.random.default_rng(seed_bo + 1)
    rs_draws = rng_rs.uniform(p_lo, p_hi, n_random)
    rs_values = profit(rs_draws)
    rs_curve = best_so_far(list(rs_values))
    eval_to_global_rs = int(np.argmax(rs_curve >= profit_global - 1e-3)) + 1 if np.any(rs_curve >= profit_global - 1e-3) else None
    p_rs = float(rs_draws[int(np.argmax(rs_values))])
    profit_rs = float(np.max(rs_values))

    # Multi-start L-BFGS-B with calls instrumented
    multi_calls: list[float] = []

    def profit_logged(p):
        x = float(np.atleast_1d(p)[0])
        v = float(profit(x))
        multi_calls.append(v)
        return -v

    rng_ms = np.random.default_rng(seed_bo + 2)
    n_starts = 50
    starts = rng_ms.uniform(p_lo, p_hi, n_starts)
    best_multi = -np.inf
    p_multi = float("nan")
    for p0 in starts:
        res = minimize(profit_logged, x0=np.array([p0]), method="L-BFGS-B", bounds=[(p_lo, p_hi)])
        pf = float(res.x[0])
        v = float(profit(pf))
        if v > best_multi:
            best_multi = v
            p_multi = pf
    multi_curve = best_so_far(multi_calls)
    eval_to_global_ms = int(np.argmax(multi_curve >= profit_global - 1e-3)) + 1 if np.any(multi_curve >= profit_global - 1e-3) else None

    # Simulated annealing
    sa_calls: list[float] = []

    def profit_logged_sa(p):
        v = float(profit(float(p[0])))
        sa_calls.append(v)
        return -v

    res_sa = dual_annealing(profit_logged_sa, bounds=[(p_lo, p_hi)], seed=seed_bo + 3, maxiter=500)
    sa_curve = best_so_far(sa_calls)
    p_sa = float(res_sa.x[0])
    profit_sa = float(profit(p_sa))
    eval_to_global_sa = int(np.argmax(sa_curve >= profit_global - 1e-3)) + 1 if np.any(sa_curve >= profit_global - 1e-3) else None

    bo_curve = best_so_far(y_ei)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    setup_style()

    # Figure 1: profit surface with the two peaks
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    p_grid = np.linspace(p_lo, p_hi, 600)
    ax1.plot(p_grid, profit(p_grid), color="tab:blue", linewidth=2, label=r"$\pi(p)$")
    ax1.axvline(p_kink, color="tab:gray", linestyle=":", linewidth=1.0,
                label=fr"Low-segment exit $p_L^{{\max}} = {p_kink:.2f}$")
    ax1.plot(p_low_peak, profit_low_peak, "o", color="tab:orange", markersize=10,
             label=fr"Low peak $p_L^{{\ast}} = {p_low_peak:.3f}$, $\pi = {profit_low_peak:.3f}$")
    ax1.plot(p_high_peak, profit_high_peak, "*", color="tab:red", markersize=18,
             label=fr"High peak $p_H^{{\ast}} = {p_high_peak:.3f}$, $\pi = {profit_high_peak:.3f}$ (global)")
    ax1.set_xlabel("Price $p$")
    ax1.set_ylabel(r"Profit $\pi(p)$")
    ax1.set_title("Two-segment monopoly profit and its two local peaks")
    ax1.legend(loc="upper right", fontsize=9)
    save_figure(fig1, "figures/profit-surface.png", dpi=150)

    # Figure 2: BO iteration snapshots (GP posterior + EI)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panel_order = sorted(snapshots.keys())
    f_true_grid = profit(acq_grid)
    for ax, k in zip(axes2.flat, panel_order):
        snap = snapshots[k]
        ax.plot(acq_grid, f_true_grid, color="tab:gray", linestyle="--", linewidth=1.2,
                label=r"True $\pi(p)$")
        ax.plot(acq_grid, snap["mu"], color="tab:blue", linewidth=1.8, label="GP posterior mean")
        ax.fill_between(acq_grid, snap["mu"] - 2 * snap["sd"], snap["mu"] + 2 * snap["sd"],
                        color="tab:blue", alpha=0.15, label="GP posterior $\\pm 2\\sigma$")
        ax.plot(snap["X"], snap["y"], "o", color="tab:orange", markersize=7, label="Evaluations")
        if snap["next_x"] is not None and k < n_total:
            ax.axvline(snap["next_x"], color="tab:green", linestyle="-", linewidth=1.0, alpha=0.7,
                       label=f"Next pick: $p = {snap['next_x']:.2f}$")
        ax.plot(p_high_peak, profit_high_peak, "*", color="tab:red", markersize=14,
                label="Global peak")
        ax.set_title(fr"After {k} evaluations ($\ell = {snap['ell']:.2f}$)")
        ax.set_ylabel(r"$\pi(p)$")
        ax.set_ylim(-1.5, 7.5)
        if k == panel_order[0]:
            ax.legend(loc="lower right", fontsize=8)
    for ax in axes2[-1, :]:
        ax.set_xlabel("Price $p$")
    fig2.suptitle("Gaussian-process posterior and next acquisition pick across BO iterations", y=1.00)
    fig2.tight_layout()
    save_figure(fig2, "figures/bo-iterations.png", dpi=150)

    # Figure 3: convergence comparison vs other global methods
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    ax3.plot(np.arange(1, len(bo_curve) + 1), bo_curve, "o-", color="tab:purple",
             linewidth=2, markersize=4, label=f"Bayesian optimization (EI), best at eval {eval_to_global_ei}")
    ax3.plot(np.arange(1, len(multi_curve) + 1), multi_curve, color="tab:blue",
             linewidth=1.5, alpha=0.8,
             label=f"Multi-start L-BFGS-B, global at eval {eval_to_global_ms}")
    ax3.plot(np.arange(1, len(rs_curve) + 1), rs_curve, color="tab:green",
             linewidth=1.5, alpha=0.8,
             label=f"Random search, global at eval {eval_to_global_rs}")
    ax3.plot(np.arange(1, len(sa_curve) + 1), sa_curve, color="tab:orange",
             linewidth=1.2, alpha=0.85,
             label=f"Simulated annealing, global at eval {eval_to_global_sa}")
    ax3.axhline(profit_global, color="tab:red", linestyle="--", linewidth=1.5,
                label=fr"Global $\pi^{{\ast}} = {profit_global:.3f}$")
    ax3.axhline(profit_low_peak, color="tab:gray", linestyle=":", linewidth=1.0,
                label=fr"Local-only $\pi = {profit_low_peak:.3f}$")
    ax3.set_xscale("log")
    ax3.set_xlabel("Number of objective evaluations (log scale)")
    ax3.set_ylabel("Best profit found so far")
    ax3.set_title("Best-so-far profit by evaluation count, BO versus three baselines")
    ax3.legend(loc="lower right", fontsize=8)
    fig3.tight_layout()
    save_figure(fig3, "figures/convergence-comparison.png", dpi=150)

    # -------------------------------------------------------------------------
    # Tables
    # -------------------------------------------------------------------------
    sa_ratio = len(sa_curve) / n_total
    rs_ratio = n_random / n_total
    ms_ratio = len(multi_curve) / n_total
    method_table = pd.DataFrame({
        "Method": [
            "Bayesian optimization (EI)",
            "Multi-start L-BFGS-B",
            "Random search",
            "Simulated annealing",
        ],
        "Setting": [
            f"{n_initial} initial + {n_iter_bo} EI steps, seed {seed_bo}",
            f"{n_starts} starts, seed {seed_bo + 2}",
            f"{n_random} draws, seed {seed_bo + 1}",
            f"max iterations 500, seed {seed_bo + 3}",
        ],
        "Estimated optimum": [
            f"{p_bo_ei:.4f}",
            f"{p_multi:.4f}",
            f"{p_rs:.4f}",
            f"{p_sa:.4f}",
        ],
        "Profit": [
            f"{best_bo_ei:.4f}",
            f"{best_multi:.4f}",
            f"{profit_rs:.4f}",
            f"{profit_sa:.4f}",
        ],
        "Function evaluations": [
            f"{n_total}",
            f"{len(multi_curve)}",
            f"{n_random}",
            f"{len(sa_curve)}",
        ],
        "Evaluations to global": [
            f"{eval_to_global_ei}" if eval_to_global_ei else "not reached",
            f"{eval_to_global_ms}" if eval_to_global_ms else "not reached",
            f"{eval_to_global_rs}" if eval_to_global_rs else "not reached",
            f"{eval_to_global_sa}" if eval_to_global_sa else "not reached",
        ],
    })
    Path("tables/method_comparison.csv").parent.mkdir(parents=True, exist_ok=True)
    method_table.to_csv("tables/method_comparison.csv", index=False)

    bo_print = bo_log_df.copy()
    bo_print["x"] = bo_print["x"].map(lambda v: f"{v:.4f}")
    bo_print["f"] = bo_print["f"].map(lambda v: f"{v:.4f}")
    bo_print["best_so_far"] = bo_print["best_so_far"].map(lambda v: f"{v:.4f}")
    bo_print = bo_print.rename(columns={
        "iteration": "Iteration",
        "phase": "Phase",
        "x": "Price evaluated",
        "f": "Profit observed",
        "best_so_far": "Best profit so far",
    })
    Path("tables/bo_iteration_log.csv").parent.mkdir(parents=True, exist_ok=True)
    bo_print.to_csv("tables/bo_iteration_log.csv", index=False)

    save_thumbnail("figures/profit-surface.png", "figures/thumb.png")
    print(f"Figures and tables written. BO budget: {n_total}, SA: {len(sa_curve)}, RS: {n_random}, MS: {len(multi_curve)}")
    print(f"SA/BO ratio: {sa_ratio:.1f}x, RS/BO: {rs_ratio:.1f}x, MS/BO: {ms_ratio:.1f}x")


if __name__ == "__main__":
    main()
