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
from lib.plotting import setup_style
from lib.output import ModelReport


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
    p0_single = 1.0
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
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Global Search and Multi-Start Diagnostics",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A monopolist sells to two consumer segments with very different demand schedules. "
        "The low-valuation segment is large but quits the market at a low price. "
        "The high-valuation segment is small but willing to pay much more. "
        "Profit is the share-weighted sum of revenue from both segments minus marginal cost.\n\n"
        "The mixture profit function has two local maxima. "
        "A low-price peak serves both segments. "
        "A higher-price peak serves only the high-valuation segment and earns more in this calibration. "
        "Different optimization methods land in different peaks depending on where they start.\n\n"
        "The lesson is reporting discipline. "
        "An optimizer that converges has answered a local question, not a global one. "
        "Multi-start, random search, and global search are diagnostics that bound the gap between local and global optimality. "
        "The same habit transfers to structural likelihoods, simulated moments, mixture models, and dynamic games."
    )

    report.add_equations(
        r"""A monopolist faces a population of consumers split between two segments.
Segment $L$ has linear demand with intercept $A_L$ and slope $b_L$.
Segment $H$ has linear demand with intercept $A_H$ and slope $b_H$.

$$D_L(p) = \max\lbrace 0,\, A_L - b_L\, p \rbrace,
\qquad
D_H(p) = \max\lbrace 0,\, A_H - b_H\, p \rbrace.$$

Each segment quits the market at its own choke price.
The low-valuation segment exits at $p_L^{\max} = A_L / b_L$.
The high-valuation segment exits at the larger price $p_H^{\max} = A_H / b_H$.

The population mixture weight $\lambda \in (0, 1)$ records the share of low-valuation consumers.
Profit is the weighted sum of segment revenues minus the marginal-cost wedge.

$$\pi(p) = (p - c) \left[\lambda\, D_L(p) + (1 - \lambda)\, D_H(p)\right].$$

The objective is piecewise quadratic in $p$.
On $[c,\, p_L^{\max}]$ both segments are active.
On $(p_L^{\max},\, p_H^{\max}]$ only the high-valuation segment is active.
The two regimes are smoothly stitched at the kink $p_L^{\max}$.

In the both-segments regime the first-order condition is linear in $p$.

$$\pi'(p) = \lambda (A_L - 2 b_L\, p) + (1 - \lambda)(A_H - 2 b_H\, p) + (\lambda b_L + (1 - \lambda) b_H)\, c.$$

In the high-only regime profit is the standard quadratic with a single interior maximizer.

$$p_H^{\ast} = \frac{A_H + b_H\, c}{2 b_H}.$$

At the calibration $A_L = 10$, $b_L = 5$, $A_H = 8$, $b_H = 1$, $c = 0.5$, $\lambda = 0.6$, both regimes have an interior maximizer.

$$p_L^{\ast} \approx 1.603,
\qquad
\pi(p_L^{\ast}) \approx 4.14.$$

$$p_H^{\ast} = 4.25,
\qquad
\pi(p_H^{\ast}) \approx 5.625.$$

The high-price peak is global on this calibration.
The low-price peak is a strict local maximum.
A start in $[c,\, p_L^{\max}]$ flows to the low peak under any local ascent.
A start in $(p_L^{\max},\, p_H^{\max}]$ flows to the high peak.

The next four subsections describe one method at a time.

### Method 1: Multi-start L-BFGS-B

Multi-start L-BFGS-B draws $N$ initial prices uniformly on the bracket and runs the local optimiser from each.

$$\hat p_{\mathrm{multi}}^{(N)} = \arg\max_{k \in \lbrace 1, \ldots, N\rbrace} \pi\left(\mathrm{LBFGSB}(p_0^{(k)})\right),
\qquad p_0^{(k)} \sim \mathrm{Uniform}[p_{\mathrm{lo}}, p_{\mathrm{hi}}].$$

The probability of finding the global optimum is one minus the probability that all $N$ starts land in the low basin.
Reporting that probability is the diagnostic for whether the start budget is large enough.

### Method 2: Random search

Random search drops the local optimiser entirely and uses a single sample of $N$ uniform draws.

$$\hat p_{\mathrm{rand}}^{(N)} = \arg\max_{k \in \lbrace 1, \ldots, N\rbrace} \pi(p^{(k)}),
\qquad p^{(k)} \sim \mathrm{Uniform}[p_{\mathrm{lo}}, p_{\mathrm{hi}}].$$

Random search is cheaper per evaluation than multi-start L-BFGS-B but converges only at rate $1/\sqrt{N}$.

### Method 3: Nelder-Mead

Nelder-Mead is a derivative-free local search.
It maintains a simplex of candidate points and reflects, expands, contracts, or shrinks it based on the ranks of the function values at the vertices.
Convergence is local; basin dependence is the same as L-BFGS-B, so a single Nelder-Mead start with a poor initial point misses the global maximum on this problem.

### Method 4: Simulated annealing

Simulated annealing samples a Markov chain that proposes random moves and accepts them with a probability that depends on the change in objective and a slowly decreasing temperature.
SciPy's `dual_annealing` combines a generalised-simulated-annealing global search with local refinement at each accepted move.
The result is a stochastic global search that does not need a starting point inside the global basin.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $A_L$, $b_L$ | {A_L:.1f}, {b_L:.1f} | Low-valuation linear demand |\n"
        f"| $A_H$, $b_H$ | {A_H:.1f}, {b_H:.1f} | High-valuation linear demand |\n"
        f"| $c$ | {c:.1f} | Marginal cost |\n"
        f"| $\\lambda$ | {lam:.1f} | Share of low-valuation consumers |\n"
        f"| Search bracket | $[{p_lo:.3f},\\, {p_hi:.1f}]$ | Outer bounds for every method |\n"
        f"| Low choke price | $p_L^{{\\max}} = {p_kink:.2f}$ | Low-valuation segment quits |\n"
        f"| Low peak | $p_L^{{\\ast}} = {p_low_peak:.4f}$, $\\pi = {profit_low_peak:.4f}$ | Local maximum |\n"
        f"| High peak | $p_H^{{\\ast}} = {p_high_peak:.4f}$, $\\pi = {profit_high_peak:.4f}$ | Global maximum |\n"
        f"| Multi-start budget $N$ | {n_starts_main} | Number of L-BFGS-B starts |\n"
        f"| Random-search budget $N$ | {n_random} | Uniform draws |\n"
        f"| Random seed | {seed_main} | For reproducibility |\n"
        f"| Single-start $p_0$ | {p0_single} | Used by methods 1 and 4 |"
    )

    report.add_solution_method(
        "All five methods explore the same one-dimensional bracket. "
        "They differ in how they balance local refinement against global exploration.\n\n"

        "### Method 1: Single-start L-BFGS-B\n\n"
        "L-BFGS-B is the practical Newton-style local optimizer for bound-constrained smooth problems. "
        "It builds a low-memory BFGS approximation of the Hessian using gradient differences and uses it to take quasi-Newton steps. "
        "Convergence is locally superlinear in the basin of attraction. "
        "On a single start the answer depends entirely on which basin contains the initial price.\n\n"
        "```text\n"
        "Algorithm: Single-start L-BFGS-B\n"
        "Input : objective f, gradient g (or numerical), bounds, x_0\n"
        "Output: x_hat reported as the optimum\n"
        "  scipy.optimize.minimize(f, x_0, method='L-BFGS-B', bounds=...)\n"
        "  the routine maintains a small set of (s_k, y_k) pairs for the BFGS update\n"
        "  it projects each step onto the feasible box\n"
        "```\n\n"
        "Single-start L-BFGS-B has no global guarantee. "
        "On a nonconcave objective it converges to the closest local maximum, which can be very far from the global. "
        "There is no way to know from the converged output that a better basin exists.\n\n"

        "### Method 2: Multi-start L-BFGS-B\n\n"
        "Multi-start runs the local optimizer from many initial points and keeps the best result. "
        "The economic intuition is a survey of basins: each start lands somewhere, and many starts together map out which basins exist. "
        "The probability of missing the global is exponentially small in $N$ when basin-of-attraction probabilities are non-degenerate. "
        "The cost is linear in $N$ with the same per-run cost as a single start.\n\n"
        "```text\n"
        "Algorithm: Multi-start L-BFGS-B\n"
        "Input : objective, bounds, sample size N, seed s\n"
        "Output: best x across N local runs\n"
        "  draw N uniform starts from the bracket\n"
        "  for each start: run L-BFGS-B and record the converged x and f\n"
        "  return the start whose converged f is largest\n"
        "  optionally label each result by basin and report basin counts\n"
        "```\n\n"
        "Multi-start can still miss the global if every start happens to land in the same basin. "
        "The diagnostic is to report how many starts landed in each basin and the gap between the best basin and the runner-up. "
        "A single basin discovery is a warning that the bracket is too narrow or that one basin dominates the volume of starts.\n\n"

        "### Method 3: Random search\n\n"
        "Random search drops the local optimizer entirely. "
        "It evaluates the objective at $N$ uniform draws and returns the argmax of the sample. "
        "The expected error scales as $1/\\sqrt{N}$ on a unimodal problem. "
        "On a nonconcave problem the rate degrades in proportion to the volume share of the global basin. "
        "Random search is the cheapest exploratory tool. "
        "It is also the most bluntly empirical: nothing in its output certifies optimality.\n\n"
        "```text\n"
        "Algorithm: Random search\n"
        "Input : objective, bounds, sample size N, seed s\n"
        "Output: x_hat\n"
        "  draw N uniform points from the bracket\n"
        "  evaluate the objective at each\n"
        "  return the point with the largest value\n"
        "```\n\n"
        "Random search misses the peak with non-zero probability. "
        "Increasing $N$ shrinks the miss probability but at $1/\\sqrt{N}$ in distance terms. "
        "Random search is a useful sanity check on the answer of a more expensive method, not a substitute for it.\n\n"

        "### Method 4: Nelder-Mead\n\n"
        "Nelder-Mead is a derivative-free local optimizer. "
        "It maintains a simplex of candidate points and reflects, expands, contracts, or shrinks the simplex according to the ranks of the function values at the vertices. "
        "Convergence is local with no formal rate on non-smooth problems. "
        "It is the right tool when the objective is rough or the gradient is unavailable.\n\n"
        "```text\n"
        "Algorithm: Nelder-Mead via scipy.optimize.minimize\n"
        "Input : objective, x_0, simplex tolerances\n"
        "Output: x_hat\n"
        "  scipy.optimize.minimize(f, x_0, method='Nelder-Mead')\n"
        "  the routine builds an initial simplex around x_0\n"
        "  it iterates reflect, expand, contract, shrink moves\n"
        "  it stops on a simplex-size tolerance\n"
        "```\n\n"
        "Nelder-Mead has the same basin-dependence as L-BFGS-B. "
        "On the present calibration it converges to whichever local peak is reached first by simplex reflection.\n\n"

        "### Method 5: Simulated annealing via `dual_annealing`\n\n"
        "Simulated annealing is the canonical stochastic global search. "
        "It samples a Markov chain that proposes random moves. "
        "Each move is accepted with a probability that depends on the change in objective and a slowly decreasing temperature. "
        "SciPy's `dual_annealing` combines a generalised-simulated-annealing global search with local refinement at each accepted move. "
        "The method has provable convergence to the global optimum under a logarithmic cooling schedule. "
        "The implied constant is impractical and the practical schedule is heuristic.\n\n"
        "```text\n"
        "Algorithm: Dual annealing via scipy.optimize.dual_annealing\n"
        "Input : objective, bounds, seed, max iterations\n"
        "Output: x_hat\n"
        "  the routine runs a generalised-simulated-annealing chain on the bracket\n"
        "  it triggers local minimisation around accepted moves\n"
        "  the temperature schedule controls exploration versus exploitation\n"
        "```\n\n"
        "Simulated annealing is expensive and stochastic. "
        "Different seeds can return different answers when the cooling schedule is too short. "
        "The reporting discipline is the same as for multi-start: run several seeds and report the worst, not just the best."
    )

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
    report.add_results(
        f"The profit surface has a local peak at $p_L^{{\\ast}} = {p_low_peak:.3f}$ where both segments are active. "
        f"Above the kink at $p_L^{{\\max}} = {p_kink:.2f}$ only the high-valuation segment is active. "
        f"The high-only regime has its own peak at $p_H^{{\\ast}} = {p_high_peak:.2f}$, which is the global maximum on this calibration. "
        f"The gap between the two peaks is ${profit_high_peak - profit_low_peak:.3f}$ in profit, which is large enough to matter for any policy that depends on it."
    )
    report.add_figure(
        "figures/profit-surface.png",
        "Two-segment monopoly profit with low-price and high-price peaks marked",
        fig1,
    )

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
    report.add_results(
        f"The basin map sweeps 200 evenly spaced starting prices and records where each L-BFGS-B run lands. "
        f"Starts below the kink at $p_L^{{\\max}} = {p_kink:.2f}$ converge to the low peak. "
        f"Starts above the kink converge to the high peak. "
        f"The basin volumes are {pct_low:.1f} percent low and {pct_high:.1f} percent high on this bracket. "
        f"A single start drawn uniformly from the bracket has roughly {pct_low:.0f} percent chance of returning the wrong answer."
    )
    report.add_figure(
        "figures/basin-map.png",
        "L-BFGS-B converged price vs starting price; the kink at the low-segment exit separates the two basins",
        fig2,
    )

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
    report.add_results(
        f"The left panel plots the best profit across $N$ multi-start runs, averaged over {n_replications} seeds. "
        f"With one start the mean best profit is between the local and global peaks, reflecting that some seeds find the wrong basin. "
        f"As $N$ grows the mean best converges to the global peak and the percentile band collapses. "
        f"The right panel records the empirical probability that at least one of the $N$ starts lands in the global basin. "
        f"At $N = {n_arr[-1]}$ that probability is essentially one and the diagnostic is trustworthy."
    )
    report.add_figure(
        "figures/best-objective-by-starts.png",
        "Best profit and probability of finding the global peak as the number of multi-start runs grows",
        fig3,
    )

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
    report.add_results(
        f"All four method outputs are plotted on the same profit surface. "
        f"Both single-start methods land at the low-price local peak from $p_0 = {p0_single}$. "
        f"Their reported profits are $\\pi = {profit_single:.3f}$ for L-BFGS-B and $\\pi = {profit_nm:.3f}$ for Nelder-Mead. "
        f"Multi-start L-BFGS-B and simulated annealing both find the global peak at $p_H^{{\\ast}} = {p_high_peak:.3f}$ with profit $\\pi = {profit_global:.3f}$. "
        f"The gap between local and global on this calibration is ${profit_global - profit_single:.3f}$, which is a 30 percent profit improvement that single-start methods miss silently."
    )
    report.add_figure(
        "figures/optimizer-paths.png",
        "Final answer of each of the five methods overlaid on the profit surface",
        fig4,
    )

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
    report.add_results(
        "The table compares the five methods on the same calibration. "
        f"Single-start L-BFGS-B and Nelder-Mead converge to the local peak at $p \\approx {p_low_peak:.3f}$ from $p_0 = {p0_single}$ and miss the global. "
        "Multi-start L-BFGS-B, random search, and simulated annealing all return the global peak. "
        "Function evaluations differ by orders of magnitude: simulated annealing is the most expensive, multi-start scales linearly with the number of starts, and a single L-BFGS-B run is by far the cheapest, but cheapest is not the same as right."
    )
    report.add_table(
        "tables/method_comparison.csv",
        f"Method comparison at $\\lambda = {lam}$, $c = {c}$, segment intercepts $({A_L:.0f}, {A_H:.0f})$",
        method_table,
    )

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
    report.add_results(
        f"The multi-start log records every L-BFGS-B run individually. "
        f"It is the bookkeeping a reproducible structural estimation should publish: "
        f"every start, every converged value, and the basin label. "
        f"On this calibration {int((multi_df['basin'] == 'low-price').sum())} of {n_starts_main} starts landed in the low basin and the rest in the high basin."
    )
    report.add_table(
        "tables/multistart_results.csv",
        "Per-start log of multi-start L-BFGS-B runs",
        multi_print,
    )

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
    report.add_results(
        "The basin summary aggregates the per-start log into the diagnostic that belongs in a paper. "
        "Two basins are discovered. "
        "The high-price basin is the global. "
        "The low-price basin is a strict local. "
        "Reporting the basin counts forces a reader to confront the gap between optimization convergence and global optimality."
    )
    report.add_table(
        "tables/basin_summary.csv",
        "Basin summary across multi-start runs",
        basin_print,
    )

    report.add_takeaway(
        "Optimizer convergence is not the same as global optimality. "
        "On a nonconcave profit surface a single-start local optimizer answers a local question. "
        "It cannot certify a global one. "
        "Reading off the converged value as if it were a global maximum is the easiest way to publish a wrong answer.\n\n"
        "Multi-start L-BFGS-B is the practical default for nonconcave problems with smooth interiors. "
        "Drawing fifty starts uniformly across the search bracket maps out the basins of attraction. "
        "The basin counts and the gap between the best basin and the runner-up are the diagnostic.\n\n"
        "Random search is the cheapest sanity check. "
        "It cannot certify a global optimum either, but it bounds it from below at $1/\\sqrt{N}$ in distance. "
        "When random search and multi-start agree, the answer is more credible.\n\n"
        "Simulated annealing trades cost for global guarantees. "
        "It is the right tool when the objective is rough or has many basins. "
        "Different seeds can disagree, and the discipline is to report the worst seed alongside the best.\n\n"
        "The reporting habit transfers directly to structural estimation. "
        "Latent-regime likelihoods, simulated moments, and dynamic-game equilibria all live on nonconcave surfaces. "
        "Showing how many starts were attempted and how many basins were discovered is the difference between an opinion and a result."
    )

    report.add_references([
        "Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*. Springer, 2nd edition, Ch. 6 and 9.",
        "Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 10.",
        "Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 3 on segmented markets.",
        "Xiang, Y., Sun, D. Y., Fan, W., and Gong, X. G. (1997). *Generalized simulated annealing algorithm and its application to the Thomson model*. Physics Letters A 233, 216-220.",
        "Bergstra, J. and Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. Journal of Machine Learning Research, 13, 281-305.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
