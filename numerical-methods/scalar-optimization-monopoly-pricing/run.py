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
from lib.plotting import setup_style
from lib.output import ModelReport


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
    # Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Scalar Optimization for Monopoly Pricing",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A monopolist with constant marginal cost faces a constant-elasticity demand. "
        "Pricing is one-dimensional. "
        "The profit-maximizing price has a closed form. "
        "That closed form, together with the Lerner markup, pins down what every numerical method should agree on.\n\n"
        "Four paradigms are compared on the same profit curve:\n"
        "- deterministic sampling on a uniform mesh,\n"
        "- stochastic sampling by uniform draws,\n"
        "- derivative-free contraction of a unimodal bracket,\n"
        "- derivative-based local extrapolation.\n\n"
        "Each paradigm is the canonical entry point to a wider family used elsewhere in the catalog.\n\n"
        "The lesson is that solving the first-order condition is not automatically safer than maximizing the objective. "
        "A starting price in the convex region of profit produces a Newton step that points away from the maximum. "
        "Adding a bracket safeguard recovers convergence at negligible cost."
    )

    report.add_equations(
        r"""The general problem is to maximise a scalar profit function $\pi : [p_{\mathrm{lo}}, p_{\mathrm{hi}}] \to \mathbb{R}$ on a bounded interval.
The methods below differ in what they evaluate (the function, its derivative, both) and in how they update the candidate optimum.

### The test instance

The test instance is monopoly pricing under constant-elasticity demand.
Three constants pin down the demand curve.
$A$ is a scale parameter that absorbs market size.
$\epsilon$ is the demand elasticity, and $\epsilon > 1$ is required for the optimum to exist.
$c$ is the constant marginal cost.

$$D(p) = A\, p^{-\epsilon}.$$

Profit is the price-cost margin times the quantity sold.

$$\pi(p) = (p - c)\, D(p) = A\, (p - c)\, p^{-\epsilon}.$$

The first-order condition $\pi'(p) = 0$ has a closed-form root that pins down what every method should return.

$$\pi'(p) = A\, p^{-(\epsilon + 1)} \left[(1 - \epsilon)\, p + \epsilon\, c\right],
\qquad
p^{\ast} = \frac{\epsilon}{\epsilon - 1}\, c.$$

Rearranging the optimum gives the Lerner price-cost margin.

$$\frac{p^{\ast} - c}{p^{\ast}} = \frac{1}{\epsilon}.$$

At the baseline calibration $\epsilon = 2.5$ and $c = 1$ the closed form gives $p^{\ast} = 5/3 \approx 1.667$ and Lerner markup $1/2.5 = 0.4$.

The second derivative is needed by Newton and to identify the inflection point of $\pi$.

$$\pi''(p) = -A\, \epsilon\, p^{-(\epsilon + 2)} \left[(1 - \epsilon)\, p + (\epsilon + 1)\, c\right],
\qquad
p_{\mathrm{inflect}} = \frac{\epsilon + 1}{\epsilon - 1}\, c.$$

Profit is concave on $(0, p_{\mathrm{inflect}})$ and convex on $(p_{\mathrm{inflect}}, \infty)$.
The optimum sits strictly inside the concave region, with $p^{\ast} < p_{\mathrm{inflect}}$.

The next four subsections describe one method at a time.

### Method 1: Grid search

Grid search covers the bracket with a uniform mesh of $N$ nodes and returns the argmax over the mesh.

$$\hat p_{\mathrm{grid}} = \arg\max_{i \in \lbrace 1, \ldots, N\rbrace} \pi(p_i),
\qquad p_i = p_{\mathrm{lo}} + \frac{(i - 1)\,(p_{\mathrm{hi}} - p_{\mathrm{lo}})}{N - 1}.$$

The distance from the nearest mesh point to $p^{\ast}$ is at most half the spacing, so the error scales as $1/N$.

### Method 2: Random search

Random search draws $N$ prices uniformly on the bracket and returns the argmax of the sampled profits.

$$\hat p_{\mathrm{rand}} = \arg\max_{i \in \lbrace 1, \ldots, N\rbrace} \pi(p_i),
\qquad p_i \stackrel{\mathrm{iid}}{\sim} \mathrm{Uniform}[p_{\mathrm{lo}}, p_{\mathrm{hi}}].$$

The expected error scales as $1/N$ in one dimension, the same order as the grid but with stochastic noise on each draw.

### Method 3: Golden-section search

Golden-section search contracts a unimodal bracket $[a_n, b_n]$ using the golden ratio.
Two interior probes split the bracket so that one is reused after each shrink.

$$\phi = \frac{\sqrt{5} - 1}{2} \approx 0.618,
\qquad
p_n = b_n - \phi\, (b_n - a_n),
\qquad
q_n = a_n + \phi\, (b_n - a_n).$$

Here $p_n$ and $q_n$ are the left and right probe prices at iteration $n$, distinct from the price control $p$.
The bracket shrinks by a constant factor $\phi$ each step, giving linear convergence.

### Method 4: Newton on the FOC

Newton follows the tangent of $\pi'$ at the current iterate.

$$x_{n+1} = x_n - \frac{\pi'(x_n)}{\pi''(x_n)}.$$

Newton is equivalent to maximising a parabolic surrogate that matches $\pi$ in value, slope, and curvature at $x_n$.
The surrogate is concave only when $\pi''(x_n)$ is negative, which holds only when $x_n$ lies below $p_{\mathrm{inflect}}$.
A start in the convex region drives the iterates away from $p^{\ast}$.
"""
    )

    report.add_model_setup(
        f"| Symbol | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $A$ | {A} | Demand scale |\n"
        f"| $\\epsilon$ | {epsilon} | Demand elasticity, $\\epsilon > 1$ |\n"
        f"| $c$ | {c} | Constant marginal cost |\n"
        f"| $p^{{\\ast}}$ | {p_star:.4f} | Closed-form optimum $\\epsilon c / (\\epsilon - 1)$ |\n"
        f"| $1/\\epsilon$ | {lerner:.4f} | Lerner markup |\n"
        f"| $p_{{\\mathrm{{inflect}}}}$ | {p_inflect:.4f} | Inflection point of $\\pi$ |\n"
        f"| Bracket $[p_{{\\mathrm{{lo}}}}, p_{{\\mathrm{{hi}}}}]$ | $[{p_low:.2f},\\, {p_high:.2f}]$ | Search interval for grid, random, golden section |\n"
        f"| Sample budget $N$ at headline run | {n_grid_main} | Used for grid and random search comparison row |\n"
        f"| Random seed | {seed_main} | Seed for the headline random-search run |\n"
        f"| Replications across $N$ | {n_replications} | Used to average random-search error in the convergence figure |\n"
        f"| Newton good start $x_0$ | {x0_good} | Start below $p^{{\\ast}}$ in the concave region |\n"
        f"| Newton bad start $x_0$ | {x0_bad} | Start above $p_{{\\mathrm{{inflect}}}}$ in the convex region |\n"
        f"| Tolerance $\\eta$ | {tol:.0e} | Stopping rule on bracket width and on $\\pi'$ |"
    )

    report.add_solution_method(
        "All four methods solve the same maximization on the bounded interval. "
        "They differ in what they evaluate and in how they update.\n\n"

        "### Method 1: Grid search\n\n"
        "Grid search covers the bracket with a uniform mesh of $N$ nodes. "
        "It returns the argmax over the mesh. "
        "The distance from the closest mesh point to $p^{\\ast}$ is at most half the spacing. "
        "The error therefore scales as $1/N$. "
        "Doubling $N$ halves the error.\n\n"
        "```text\n"
        "Algorithm: Grid search\n"
        "Input : bracket [p_lo, p_hi]; grid size N\n"
        "Output: p_hat\n"
        "  build N equally spaced prices p_1 < ... < p_N on [p_lo, p_hi]\n"
        "  i_best <- argmax over i of pi(p_i)\n"
        "  p_hat  <- p_{i_best}\n"
        "```\n\n"
        "Grid search has no failure mode in one dimension. "
        "Its limitation is dimensional. "
        "Reaching $10^{-10}$ accuracy needs about $N \\sim 4 \\times 10^{10}$ nodes. "
        "The cost grows as $N^d$ in $d$ dimensions.\n\n"

        "### Method 2: Random search\n\n"
        "Random search replaces the deterministic mesh with $N$ uniform draws on the bracket. "
        "The expected distance from the closest draw to $p^{\\ast}$ scales as $1/\\sqrt{N}$. "
        "That is slower than grid search in one dimension. "
        "The strength of random search is dimensional. "
        "The $1/\\sqrt{N}$ rate is independent of how many price dimensions one adds.\n\n"
        "```text\n"
        "Algorithm: Random search\n"
        "Input : bracket [p_lo, p_hi]; sample budget N; seed s\n"
        "Output: p_hat\n"
        "  rng <- random number generator seeded by s\n"
        "  draw N prices p_1, ..., p_N independently from Uniform[p_lo, p_hi]\n"
        "  i_best <- argmax over i of pi(p_i)\n"
        "  p_hat  <- p_{i_best}\n"
        "```\n\n"
        "Random search misses with non-zero probability. "
        "A single run can leave a wide gap to $p^{\\ast}$. "
        "The standard discipline is to repeat across seeds and report the worst run. "
        "Averaging across seeds at each $N$ recovers the smooth $1/\\sqrt{N}$ rate.\n\n"

        "### Method 3: Golden-section search\n\n"
        "Golden-section search contracts a unimodal bracket without using derivatives. "
        "Two interior probes split the bracket so that one of them is reused after each shrink. "
        "That halves the function-evaluation budget compared to bisection. "
        "The bracket width shrinks by the golden ratio every iteration. "
        "Linear convergence in the bracket width follows directly. "
        "Reaching $10^{-10}$ accuracy needs about 50 iterations regardless of where $p^{\\ast}$ sits inside the initial bracket.\n\n"
        "```text\n"
        "Algorithm: Golden-section search\n"
        "Input : a, b with pi unimodal on [a, b]; tolerance eta\n"
        "Output: p_n\n"
        "  phi <- (sqrt(5) - 1) / 2\n"
        "  p   <- b - phi (b - a)\n"
        "  q   <- a + phi (b - a)\n"
        "  for n = 1, 2, ... :\n"
        "      if pi(p) > pi(q): b <- q\n"
        "      else            : a <- p\n"
        "      recompute p, q\n"
        "      stop when (b - a) < eta\n"
        "```\n\n"
        "Golden section assumes unimodality. "
        "A profit curve with two peaks would mislead the bracket-shrink rule. "
        "Constant-elasticity profit is single-peaked. "
        "The unimodality assumption holds here.\n\n"

        "### Method 4: Newton on the FOC\n\n"
        "Newton fits a parabolic surrogate to $\\pi$ at the current iterate. "
        "The iterate jumps to the argmax of the surrogate. "
        "The fit matches the value, slope, and curvature of $\\pi$ at $x_n$. "
        "Quadratic convergence follows from a Taylor expansion around $p^{\\ast}$ whenever the start lies in the basin of attraction. "
        "Each iteration roughly doubles the number of correct digits. "
        "A handful of steps suffices for ten-decimal accuracy.\n\n"
        "```text\n"
        "Algorithm: Newton on FOC, with optional bracket safeguard\n"
        "Input : x_0; tolerance eta; pi', pi''; bracket [p_lo, p_hi]\n"
        "Output: x_n\n"
        "  for n = 0, 1, ... :\n"
        "      if safeguard and pi''(x_n) >= 0:\n"
        "          if pi'(x_n) > 0: x_{n+1} <- (p_hi + x_n) / 2\n"
        "          else           : x_{n+1} <- (p_lo + x_n) / 2\n"
        "      else:\n"
        "          x_{n+1} <- x_n - pi'(x_n) / pi''(x_n)\n"
        "          if safeguard and x_{n+1} not in (p_lo, p_hi):\n"
        "              x_{n+1} <- clip(x_{n+1}, p_lo, p_hi)\n"
        "      stop when |pi'(x_n)| < eta\n"
        "```\n\n"
        "Newton fails when the start sits in the convex region above the inflection point. "
        "The second derivative flips sign there. "
        "The parabolic surrogate becomes a minimum. "
        "The step points away from $p^{\\ast}$. "
        "The bracket safeguard handles this in two ways. "
        "When the curvature is non-negative it takes a bisection step in the direction of profit ascent. "
        "When the Newton step exits the bracket it clips the iterate to the interior. "
        "The two rules together turn the failure mode into a delayed convergence rather than divergence."
    )

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
    report.add_results(
        f"At baseline the closed-form price is $p^{{\\ast}} = {p_star:.3f}$ and the Lerner markup is $1/\\epsilon = {lerner:.3f}$. "
        f"Profit is concave below $p_{{\\mathrm{{inflect}}}} = {p_inflect:.3f}$ and convex above it. "
        "The maximum sits in the concave region. "
        "An iterate that lands above the inflection point misreads the local curvature."
    )
    report.add_figure(
        "figures/profit-curve.png",
        "Constant-elasticity monopoly profit with closed-form optimum and inflection point",
        fig1,
    )

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

    report.add_results(
        f"Grid search on $N = {n_show_grid}$ points pins $p^{{\\ast}}$ to the nearest node. "
        f"Random search on $N = {len(draws_show)}$ uniform draws scatters across the bracket. "
        f"Golden section contracts the bracket $[{p_low:.2f},\\, {p_high:.2f}]$ at the fixed factor $\\phi$ each step. "
        f"Newton from $x_0 = {x0_good}$ enters the basin of attraction directly and reaches the tolerance in a handful of steps."
    )
    report.add_figure(
        "figures/method-paths.png",
        "Iterates of grid, random, golden-section, and Newton overlaid on the profit curve",
        fig2,
    )

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

    report.add_results(
        f"At $x_0 = {x0_bad}$ the FOC residual is $\\pi'(x_0) = {profit_prime(x0_bad):.3e}$. "
        f"The curvature is $\\pi''(x_0) = {profit_double_prime(x0_bad):.3e}$, positive because the start lies in the convex region. "
        f"The vanilla Newton step is positive and lands at $x_1 = {final_x_bad:.3f}$. "
        f"That iterate exits the search bracket immediately, and the run is flagged **{newton_bad_status}** after one iteration.\n\n"
        f"The bracket safeguard recovers convergence from the same $x_0 = {x0_bad}$. "
        f"It first takes a bisection step in the direction of profit ascent because $\\pi''(x_0) \\geq 0$. "
        f"Once back in the concave region the standard quadratic Newton convergence kicks in. "
        f"Safeguarded Newton converges in **{newton_safe_iter} iterations** with residual **{newton_safe_residual:.2e}**."
    )
    report.add_figure(
        "figures/newton-failure.png",
        "Newton on the FOC fails from a start in the convex region; the bisection-uphill safeguard recovers convergence",
        fig3,
    )

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

    report.add_results(
        "Golden section contracts at a constant factor every step. "
        f"Newton from $x_0 = {x0_good}$ shows the quadratic regime once inside the basin. "
        f"The safeguarded run from $x_0 = {x0_bad}$ spends its first iteration on the bisection-uphill step. "
        "It then enters the same quadratic regime once back in the concave region.\n\n"
        "Grid-search error scales as $1/N$ in the right panel. "
        "Random-search error scales as $1/\\sqrt{N}$ on average across seeds. "
        "Grid is faster than random in one dimension. "
        "The gap closes and reverses as the dimension grows.\n\n"
        "Reaching $10^{-3}$ accuracy needs roughly $N \\sim 4000$ grid evaluations. "
        "Golden section achieves the same accuracy in about a dozen iterations."
    )
    report.add_figure(
        "figures/convergence.png",
        "Distance from the closed-form optimum vs iteration (left) and sampling error vs sample budget for grid and random search (right)",
        fig4,
    )

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
    report.add_results(
        f"The table collects the six headline runs at the baseline calibration $(\\epsilon, c) = ({epsilon}, {c})$. "
        "Iterations are sample evaluations for grid and random, bracket halvings for golden section, and Newton steps for the last three rows. "
        "The Newton failure from the bad start and its recovery under the safeguard sit in the same view as the closed-form benchmark."
    )
    report.add_table(
        "tables/method_comparison.csv",
        f"Method comparison on the baseline calibration ($\\epsilon = {epsilon}$, $c = {c}$)",
        method_table,
    )

    eps_print = pd.DataFrame({
        "Elasticity": [f"{r['epsilon']:.2f}" for r in eps_rows],
        "Closed-form price": [f"{r['p_star']:.4f}" for r in eps_rows],
        "Lerner markup": [f"{r['Lerner markup']:.4f}" for r in eps_rows],
        "Profit at the optimum": [f"{r['profit at p_star']:.4f}" for r in eps_rows],
        "Golden-section error": [f"{r['golden-section error']:.2e}" for r in eps_rows],
    })
    report.add_results(
        "Across $\\epsilon$, the Lerner identity $1/\\epsilon$ pins down the price-cost margin. "
        "The closed-form $p^{\\ast}$ moves smoothly as the demand becomes more or less elastic. "
        "Golden section recovers the closed form to tolerance in every row."
    )
    report.add_table(
        "tables/elasticity_sensitivity.csv",
        "Closed-form benchmarks across demand elasticities and golden-section recovery",
        eps_print,
    )

    sweep_table = pd.DataFrame({
        "Starting price": [f"{x:.2f}" for x in starting_points],
        "Iterations": newton_counts,
        "Status": newton_status_list,
        "Above inflection point": ["yes" if x > p_inflect else "no" for x in starting_points],
    })
    report.add_results(
        f"The vanilla-Newton sweep across nine starting points makes the basin of attraction visible. "
        f"Starts below $p_{{\\mathrm{{inflect}}}} = {p_inflect:.3f}$ converge in a handful of steps. "
        f"Starts above the inflection point land in the convex region. "
        f"The first Newton step from those starts exits the search bracket. "
        f"**{n_diverged} of {len(starting_points)}** starts diverge, all of them above $p_{{\\mathrm{{inflect}}}}$."
    )
    report.add_table(
        "tables/newton_sensitivity.csv",
        "Vanilla-Newton iteration count and status across starting points",
        sweep_table,
    )

    report.add_takeaway(
        "Grid search bounds the answer with a discretization error that scales as $1/N$. "
        "It is the cheapest method to reason about and the slowest to high accuracy.\n\n"
        "Random search trades the deterministic mesh for stochastic error that scales as $1/\\sqrt{N}$. "
        "It is slower than grid in one dimension. "
        "Its rate is dimension-free. "
        "That property is why it dominates in higher dimensions.\n\n"
        "Golden section is the practical default in one dimension when the objective is unimodal. "
        "It contracts at a fixed factor regardless of where $p^{\\ast}$ sits inside the bracket.\n\n"
        "Newton on the FOC is the fastest method when the start is in the concave region. "
        "A start above the inflection point flips the sign of $\\pi''$ and the vanilla step moves away from the maximum. "
        "The bracket safeguard reverts to a bisection-uphill step in the convex region and clips Newton steps that would exit the search interval. "
        "Adding the safeguard recovers convergence at negligible cost.\n\n"
        "The closed-form Lerner markup $1/\\epsilon$ is the benchmark every method should agree on."
    )

    report.add_references([
        "Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 1.",
        "Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). *Numerical Recipes*. Cambridge University Press, 3rd edition, Ch. 10.",
        "Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press, Ch. 4.",
        "Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*. Springer, 2nd edition, Ch. 3.",
        "Bergstra, J. and Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization*. Journal of Machine Learning Research, 13, 281-305.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
