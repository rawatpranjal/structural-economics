#!/usr/bin/env python3
"""Global search and multi-start diagnostics on a nonconcave pricing objective.

A monopolist faces two consumer segments with different demand schedules.
The mixture profit function has two local maxima: a low-price peak that
serves both segments and a higher-price peak that serves only the
high-valuation segment. A single-start local optimizer converges to one
or the other depending on its initial price. Five methods are compared:
single-start L-BFGS-B, multi-start L-BFGS-B with fifty starts, plain
random search, Nelder-Mead, and simulated annealing. The lesson is that
optimizer convergence is not a verdict on global optimality; reporting
discipline is.

References:
- Nocedal and Wright (2006) Numerical Optimization, Ch. 6, 9.
- Press, Teukolsky, Vetterling, and Flannery (2007) Numerical Recipes, Ch. 10.
- Tirole (1988) The Theory of Industrial Organization, Ch. 3 on segmented markets.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, dual_annealing

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def main() -> None:
    # =========================================================================
    # Calibration: two segments, mixture profit with two peaks
    # =========================================================================
    A_L, b_L = 10.0, 5.0    # low-valuation: D_L(p) = max(0, A_L - b_L * p)
    A_H, b_H = 8.0, 1.0     # high-valuation: D_H(p) = max(0, A_H - b_H * p)
    c = 0.5                  # marginal cost
    lam = 0.6                # share of low-valuation consumers

    p_low_peak = 10.9 / 6.8           # ~1.6029, local maximum in low-price regime
    profit_low_peak = (p_low_peak - c) * (lam * (A_L - b_L * p_low_peak) + (1 - lam) * (A_H - b_H * p_low_peak))

    p_high_peak = 4.25                # local maximum in high-price regime
    profit_high_peak = (p_high_peak - c) * (1 - lam) * (A_H - b_H * p_high_peak)

    p_kink = A_L / b_L                # 2.0; low-segment exits at this price

    p_lo, p_hi = c + 1e-3, 8.0
    n_starts_main = 50
    seed_main = 42

    def D_L(p):
        return np.maximum(0.0, A_L - b_L * p)

    def D_H(p):
        return np.maximum(0.0, A_H - b_H * p)

    def profit(p):
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = (p_arr - c) * (lam * D_L(p_arr) + (1 - lam) * D_H(p_arr))
        return float(out.item()) if out.size == 1 else out

    def neg_profit(p):
        if np.ndim(p) > 0:
            return -profit(p[0])
        return -profit(p)

    # Best peak among the two analytical candidates.
    if profit_high_peak >= profit_low_peak:
        p_global, profit_global = p_high_peak, profit_high_peak
        p_local, profit_local = p_low_peak, profit_low_peak
    else:
        p_global, profit_global = p_low_peak, profit_low_peak
        p_local, profit_local = p_high_peak, profit_high_peak

    # =========================================================================
    # Method 1: single-start L-BFGS-B
    # =========================================================================
    # p0 = 1.7 sits inside the low-price basin of attraction. The basin
    # boundary for L-BFGS-B is empirically near p = 1.5, not the economic
    # kink at p = 2.0: from starts below 1.5 the positive gradient overshoots
    # the low peak and the quasi-Newton step descends into the global basin.
    p0_single = 1.7
    t0 = time.perf_counter()
    res_single = minimize(
        neg_profit, x0=np.array([p0_single]),
        method='L-BFGS-B', bounds=[(p_lo, p_hi)],
    )
    runtime_single = (time.perf_counter() - t0) * 1000.0
    p_single = float(res_single.x[0])
    profit_single = float(profit(p_single))
    nfev_single = int(res_single.nfev)

    # =========================================================================
    # Method 2: multi-start L-BFGS-B
    # =========================================================================
    rng_starts = np.random.default_rng(seed_main)
    starts = rng_starts.uniform(p_lo, p_hi, n_starts_main)
    t0 = time.perf_counter()
    multistart_records = []
    for k, p0 in enumerate(starts):
        res = minimize(neg_profit, x0=np.array([p0]),
                       method='L-BFGS-B', bounds=[(p_lo, p_hi)])
        p_final = float(res.x[0])
        multistart_records.append({
            "start_id": k,
            "p_start": float(p0),
            "p_final": p_final,
            "profit_final": float(profit(p_final)),
            "nfev": int(res.nfev),
        })
    runtime_multi = (time.perf_counter() - t0) * 1000.0
    multi_df = pd.DataFrame(multistart_records)

    def basin_label(p_final):
        return "low-price" if p_final < (p_low_peak + p_high_peak) / 2 else "high-price"

    multi_df["basin"] = multi_df["p_final"].map(basin_label)
    best_idx = int(multi_df["profit_final"].idxmax())
    p_multi = float(multi_df.loc[best_idx, "p_final"])
    profit_multi = float(multi_df.loc[best_idx, "profit_final"])
    nfev_multi = int(multi_df["nfev"].sum())

    # =========================================================================
    # Method 3: random search
    # =========================================================================
    n_random = 500
    rng_random = np.random.default_rng(seed_main + 1)
    t0 = time.perf_counter()
    draws = rng_random.uniform(p_lo, p_hi, n_random)
    profits_draws = profit(draws)
    i_best = int(np.argmax(profits_draws))
    runtime_random = (time.perf_counter() - t0) * 1000.0
    p_random = float(draws[i_best])
    profit_random = float(profits_draws[i_best])

    # =========================================================================
    # Method 4: Nelder-Mead from the same single start
    # =========================================================================
    t0 = time.perf_counter()
    res_nm = minimize(neg_profit, x0=np.array([p0_single]),
                      method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-10})
    runtime_nm = (time.perf_counter() - t0) * 1000.0
    p_nm = float(res_nm.x[0])
    profit_nm = float(profit(p_nm))
    nfev_nm = int(res_nm.nfev)

    # =========================================================================
    # Method 5: simulated annealing via dual_annealing
    # =========================================================================
    t0 = time.perf_counter()
    res_sa = dual_annealing(
        lambda p: neg_profit(p),
        bounds=[(p_lo, p_hi)],
        seed=seed_main + 2,
        maxiter=500,
    )
    runtime_sa = (time.perf_counter() - t0) * 1000.0
    p_sa = float(res_sa.x[0])
    profit_sa = float(profit(p_sa))
    nfev_sa = int(res_sa.nfev)

    # =========================================================================
    # Best-of-N curve for multi-start
    # =========================================================================
    n_replications = 30
    n_starts_grid = [1, 2, 3, 5, 8, 12, 20, 30, 50]
    best_curve = []
    for N in n_starts_grid:
        best_per_rep = []
        for s in range(n_replications):
            rng_s = np.random.default_rng(s)
            sample_starts = rng_s.uniform(p_lo, p_hi, N)
            best_local = -np.inf
            for p0 in sample_starts:
                res = minimize(neg_profit, x0=np.array([p0]),
                               method='L-BFGS-B', bounds=[(p_lo, p_hi)])
                pf = float(res.x[0])
                pi_val = float(profit(pf))
                if pi_val > best_local:
                    best_local = pi_val
            best_per_rep.append(best_local)
        best_curve.append({
            "n_starts": N,
            "mean_best": float(np.mean(best_per_rep)),
            "p10_best": float(np.percentile(best_per_rep, 10)),
            "p90_best": float(np.percentile(best_per_rep, 90)),
            "share_global": float(np.mean(np.array(best_per_rep) > profit_global - 1e-6)),
        })

    # =========================================================================
    # Figures and tables
    # =========================================================================
    setup_style()

    # ------------------------------------------------------------------
    # Figure 1: profit surface with both peaks
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Figure 2: optimizer paths from many starts (basin map)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    starts_dense = np.linspace(p_lo, p_hi, 200)
    finals_dense = []
    for p0 in starts_dense:
        res = minimize(neg_profit, x0=np.array([p0]),
                       method='L-BFGS-B', bounds=[(p_lo, p_hi)])
        finals_dense.append(float(res.x[0]))
    finals_dense = np.array(finals_dense)
    basin_color = np.where(finals_dense < (p_low_peak + p_high_peak) / 2, "tab:orange", "tab:red")
    for color, label in (("tab:orange", "Low-price basin"), ("tab:red", "High-price basin")):
        mask = basin_color == color
        ax2.scatter(starts_dense[mask], finals_dense[mask], c=color, s=18, alpha=0.7,
                    label=label)
    ax2.axhline(p_low_peak, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axhline(p_high_peak, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.axvline(p_kink, color="tab:gray", linestyle=":", linewidth=1.0,
                label=fr"$p_L^{{\max}} = {p_kink:.2f}$")
    ax2.plot([p_lo, p_hi], [p_lo, p_hi], color="tab:gray", linewidth=0.5, alpha=0.3)
    ax2.set_xlabel("Starting price $p_0$")
    ax2.set_ylabel("L-BFGS-B converged price $p_{\\mathrm{final}}$")
    ax2.set_title("Basin of attraction map for L-BFGS-B")
    ax2.legend(loc="center right", fontsize=9)
    pct_high = float(np.mean(basin_color == "tab:red") * 100.0)
    pct_low = 100.0 - pct_high
    low_mask = basin_color == "tab:orange"
    boundary_lo = float(starts_dense[low_mask].min()) if low_mask.any() else float("nan")
    boundary_hi = float(starts_dense[low_mask].max()) if low_mask.any() else float("nan")
    save_figure(fig2, "figures/basin-map.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 3: best-of-N curve for multi-start
    # ------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    n_arr = np.array([r["n_starts"] for r in best_curve])
    mean_best = np.array([r["mean_best"] for r in best_curve])
    p10 = np.array([r["p10_best"] for r in best_curve])
    p90 = np.array([r["p90_best"] for r in best_curve])
    share_global = np.array([r["share_global"] for r in best_curve])

    ax3a.fill_between(n_arr, p10, p90, alpha=0.25, color="tab:blue", label="10th to 90th percentile")
    ax3a.plot(n_arr, mean_best, "o-", color="tab:blue", linewidth=1.5, markersize=6,
              label=fr"Mean best profit across {n_replications} seeds")
    ax3a.axhline(profit_global, color="tab:red", linestyle="--", linewidth=1.5,
                 label=fr"Global $\pi^{{\ast}} = {profit_global:.3f}$")
    ax3a.axhline(profit_local, color="tab:orange", linestyle="--", linewidth=1.0,
                 label=fr"Local-only $\pi = {profit_local:.3f}$")
    ax3a.set_xscale("log")
    ax3a.set_xlabel("Number of starts $N$")
    ax3a.set_ylabel("Best profit found")
    ax3a.set_title("Best-of-$N$ profit improvement with more multi-starts")
    ax3a.legend(loc="lower right", fontsize=9)

    ax3b.plot(n_arr, share_global, "s-", color="tab:green", linewidth=1.5, markersize=6,
              label="Empirical share")
    ax3b.set_xscale("log")
    ax3b.set_xlabel("Number of starts $N$")
    ax3b.set_ylabel("Probability of finding the global peak")
    ax3b.set_title("Probability of recovering the global peak")
    ax3b.set_ylim(-0.05, 1.05)
    ax3b.axhline(1.0, color="tab:gray", linestyle=":", linewidth=0.8)
    ax3b.legend(loc="lower right", fontsize=9)
    fig3.tight_layout()
    save_figure(fig3, "figures/best-objective-by-starts.png", dpi=150)

    # ------------------------------------------------------------------
    # Figure 4: optimizer paths for the four methods on the profit curve
    # ------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(p_grid, profit(p_grid), color="tab:blue", linewidth=2, label=r"$\pi(p)$")
    ax4.plot(p_high_peak, profit_high_peak, "*", color="tab:red", markersize=18,
             label=fr"Global $p_H^{{\ast}} = {p_high_peak:.3f}$")
    ax4.plot(p0_single, profit(p0_single), "o", color="tab:gray", markersize=8,
             label=fr"Shared start $p_0 = {p0_single}$")
    ax4.plot(p_single, profit_single, "s", color="tab:orange", markersize=10,
             label=fr"L-BFGS-B end ($\pi = {profit_single:.3f}$)")
    ax4.plot(p_nm, profit_nm, "v", color="tab:green", markersize=10,
             label=fr"Nelder-Mead end ($\pi = {profit_nm:.3f}$)")
    ax4.plot(p_multi, profit_multi, "P", color="tab:purple", markersize=12,
             label=fr"Multi-start best ($\pi = {profit_multi:.3f}$)")
    ax4.plot(p_sa, profit_sa, "X", color="tab:red", markersize=12,
             label=fr"Simulated annealing ($\pi = {profit_sa:.3f}$)")
    ax4.set_xlabel("Price $p$")
    ax4.set_ylabel(r"Profit $\pi(p)$")
    ax4.set_title("Final answer of each method on the same profit surface")
    ax4.legend(loc="lower right", fontsize=9)
    save_figure(fig4, "figures/optimizer-paths.png", dpi=150)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    method_table = pd.DataFrame({
        "Method": [
            "Single-start L-BFGS-B",
            "Multi-start L-BFGS-B",
            "Random search",
            "Nelder-Mead",
            "Simulated annealing",
        ],
        "Setting": [
            f"Starting price {p0_single}",
            f"{n_starts_main} starts, seed {seed_main}",
            f"{n_random} draws, seed {seed_main + 1}",
            f"Starting price {p0_single}",
            f"max iterations 500, seed {seed_main + 2}",
        ],
        "Estimated optimum": [
            f"{p_single:.4f}",
            f"{p_multi:.4f}",
            f"{p_random:.4f}",
            f"{p_nm:.4f}",
            f"{p_sa:.4f}",
        ],
        "Profit": [
            f"{profit_single:.4f}",
            f"{profit_multi:.4f}",
            f"{profit_random:.4f}",
            f"{profit_nm:.4f}",
            f"{profit_sa:.4f}",
        ],
        "Function evaluations": [
            f"{nfev_single}",
            f"{nfev_multi}",
            f"{n_random}",
            f"{nfev_nm}",
            f"{nfev_sa}",
        ],
        "Found global?": [
            "no" if profit_single < profit_global - 1e-3 else "yes",
            "yes" if profit_multi >= profit_global - 1e-3 else "no",
            "yes" if profit_random >= profit_global - 1e-3 else "no",
            "no" if profit_nm < profit_global - 1e-3 else "yes",
            "yes" if profit_sa >= profit_global - 1e-3 else "no",
        ],
    })
    Path("tables").mkdir(parents=True, exist_ok=True)
    method_table.to_csv("tables/method_comparison.csv", index=False)

    multi_print = multi_df.copy()
    multi_print["p_start"] = multi_print["p_start"].map(lambda x: f"{x:.4f}")
    multi_print["p_final"] = multi_print["p_final"].map(lambda x: f"{x:.4f}")
    multi_print["profit_final"] = multi_print["profit_final"].map(lambda x: f"{x:.4f}")
    multi_print = multi_print.rename(columns={
        "start_id": "Start id",
        "p_start": "Starting price",
        "p_final": "Converged price",
        "profit_final": "Converged profit",
        "nfev": "Function evaluations",
        "basin": "Basin",
    })
    multi_print.to_csv("tables/multistart_results.csv", index=False)

    basin_summary = multi_df.groupby("basin").agg(
        count=("start_id", "size"),
        best_profit=("profit_final", "max"),
        mean_profit=("profit_final", "mean"),
        representative_p=("p_final", "mean"),
    ).reset_index()
    basin_print = basin_summary.copy()
    basin_print["best_profit"] = basin_print["best_profit"].map(lambda x: f"{x:.4f}")
    basin_print["mean_profit"] = basin_print["mean_profit"].map(lambda x: f"{x:.4f}")
    basin_print["representative_p"] = basin_print["representative_p"].map(lambda x: f"{x:.4f}")
    basin_print = basin_print.rename(columns={
        "basin": "Basin",
        "count": "Start count",
        "best_profit": "Best profit",
        "mean_profit": "Mean profit",
        "representative_p": "Representative price",
    })
    basin_print.to_csv("tables/basin_summary.csv", index=False)

    save_thumbnail("figures/profit-surface.png", "figures/thumb.png")
    print(f"Generated: figures/ (4 figures + thumb) + tables/ (3 tables)")


if __name__ == "__main__":
    main()
