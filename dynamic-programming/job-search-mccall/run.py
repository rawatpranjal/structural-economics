#!/usr/bin/env python3
"""McCall sequential job search.

Solves the finite-grid wage-search problem by VFI and benchmarks the
reservation wage against the scalar fixed point under the continuous lognormal
offer distribution. Regenerates README.md, figures, and the parameter table.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import lognorm, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def discretize_lognormal(
    mu: float,
    sigma: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate a lognormal distribution with equiprobable mean bins.

    Each bin gets probability 1/n and is represented by its conditional mean,
    so the discrete law preserves both E[W] and the right-tail moments that
    drive the reservation wage in this calibration.
    """
    edges = np.linspace(0.0, 1.0, n + 1)
    z_edges = norm.ppf(edges)
    mean = np.exp(mu + 0.5 * sigma**2)

    probs = np.diff(edges)
    interval_means = mean * (
        norm.cdf(z_edges[1:] - sigma) - norm.cdf(z_edges[:-1] - sigma)
    )
    wages = interval_means / probs
    return wages, probs


def expected_max_lognormal(r: float, mu: float, sigma: float) -> float:
    """E[max(W, r)] when log W ~ N(mu, sigma^2)."""
    mean = np.exp(mu + 0.5 * sigma**2)
    if r <= 0:
        return float(mean)
    cdf_at_r = lognorm.cdf(r, s=sigma, scale=np.exp(mu))
    tail_mean = mean * norm.sf((np.log(r) - mu - sigma**2) / sigma)
    return float(r * cdf_at_r + tail_mean)


def solve_continuous_reservation_wage(
    beta: float,
    b: float,
    mu: float,
    sigma: float,
    tol: float = 1e-12,
) -> float:
    """Scalar reservation-wage fixed point under the continuous lognormal law."""
    mean = np.exp(mu + 0.5 * sigma**2)

    def residual(r: float) -> float:
        continuation = (1.0 - beta) * b
        continuation += beta * expected_max_lognormal(r, mu, sigma)
        return r - continuation

    lower = 0.0
    upper = max(1.0, b, mean)
    while residual(upper) <= 0.0:
        upper *= 2.0

    return float(brentq(residual, lower, upper, xtol=tol, rtol=tol))


def continuous_acceptance_probability(w_star: float, mu: float, sigma: float) -> float:
    return float(1.0 - lognorm.cdf(w_star, s=sigma, scale=np.exp(mu)))


def solve_mccall(
    beta: float,
    b: float,
    wages: np.ndarray,
    probs: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 1_000,
) -> tuple[np.ndarray, float, dict[str, float | int | bool]]:
    """Finite-grid McCall VFI.

    Because the value of rejecting does not depend on the current offer, each
    sweep needs one expectation against the offer distribution rather than a
    full integral at every grid point.
    """
    accept_values = wages / (1.0 - beta)
    value = accept_values.copy()
    error = np.inf

    for iteration in range(1, max_iter + 1):
        continuation_value = b + beta * np.dot(probs, value)
        new_value = np.maximum(accept_values, continuation_value)
        error = float(np.max(np.abs(new_value - value)))
        value = new_value
        if error < tol:
            break

    continuation_value = b + beta * np.dot(probs, value)
    reservation_wage = (1.0 - beta) * continuation_value
    info = {
        "iterations": iteration,
        "converged": error < tol,
        "error": error,
        "continuation_value": float(continuation_value),
    }
    return value, float(reservation_wage), info


def main() -> None:
    beta = 0.95
    b = 1.0
    mu = 0.0
    sigma = 1.0
    n_w = 50
    tol = 1e-8

    wages, probs = discretize_lognormal(mu, sigma, n_w)
    mean_wage = float(np.dot(probs, wages))
    median_wage = float(np.exp(mu))

    print(
        f"Wage grid: [{wages.min():.3f}, {wages.max():.3f}], "
        f"E[W] = {mean_wage:.3f}, median = {median_wage:.3f}"
    )
    print("Solving baseline finite-grid McCall model...")
    value, w_star, info = solve_mccall(beta, b, wages, probs, tol=tol)
    w_star_cont = solve_continuous_reservation_wage(beta, b, mu, sigma)
    accept_frac_grid = float(np.sum(probs[wages >= w_star]))
    accept_frac_cont = continuous_acceptance_probability(w_star_cont, mu, sigma)
    grid_gap = w_star - w_star_cont
    expected_duration_cont = 1.0 / max(accept_frac_cont, 1e-12)

    print(
        f"  grid w* = {w_star:.4f}; continuous benchmark = {w_star_cont:.4f}; "
        f"gap = {grid_gap:.4f}"
    )

    accept_values = wages / (1.0 - beta)
    continuation_value = b + beta * np.dot(probs, value)

    print("Building comparative statics...")
    beta_vals = np.linspace(0.80, 0.99, 40)
    wstar_beta = np.zeros_like(beta_vals)
    wstar_beta_cont = np.zeros_like(beta_vals)
    for i, beta_i in enumerate(beta_vals):
        _, wstar_beta[i], _ = solve_mccall(beta_i, b, wages, probs, tol=tol)
        wstar_beta_cont[i] = solve_continuous_reservation_wage(beta_i, b, mu, sigma)

    b_vals = np.linspace(0.0, 3.0, 40)
    wstar_b = np.zeros_like(b_vals)
    wstar_b_cont = np.zeros_like(b_vals)
    for i, b_i in enumerate(b_vals):
        _, wstar_b[i], _ = solve_mccall(beta, b_i, wages, probs, tol=tol)
        wstar_b_cont[i] = solve_continuous_reservation_wage(beta, b_i, mu, sigma)

    print("Building parameter table...")
    table_rows = []
    for beta_t in [0.90, 0.95, 0.99]:
        for b_t in [0.5, 1.0, 2.0]:
            _, w_star_t, info_t = solve_mccall(beta_t, b_t, wages, probs, tol=tol)
            w_star_t_cont = solve_continuous_reservation_wage(beta_t, b_t, mu, sigma)
            accept_t_cont = continuous_acceptance_probability(w_star_t_cont, mu, sigma)
            table_rows.append(
                {
                    "beta": f"{beta_t:.2f}",
                    "b": f"{b_t:.1f}",
                    "w* grid": f"{w_star_t:.4f}",
                    "w* cont.": f"{w_star_t_cont:.4f}",
                    "grid gap": f"{w_star_t - w_star_t_cont:+.4f}",
                    "Accept % (cont.)": f"{100.0 * accept_t_cont:.1f}",
                    "E[duration]": f"{1.0 / max(accept_t_cont, 1e-12):.1f}",
                    "VFI iter.": info_t["iterations"],
                }
            )
    df_table = pd.DataFrame(table_rows)

    setup_style()

    report = ModelReport(
        "McCall Job Search and the Reservation Wage",
        "Sequential wage-offer search as an optimal-stopping problem with a scalar continuation value.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "An unemployed worker draws one wage offer per period from a known "
        "distribution. Accepting an offer locks in that wage forever; rejecting "
        "pays the unemployment benefit $b$ and rolls the dice again next period. "
        "The question is when to stop searching.\n\n"
        "Two features make this the cleanest optimal-stopping problem in the "
        "catalog. First, the only state is the current offer $w$, and once $w$ "
        "is rejected it is forgotten, so the value of rejection does not depend "
        "on $w$ at all. The Bellman equation reduces to a comparison between a "
        "linear function of $w$ (acceptance) and a scalar (rejection), which "
        "forces the policy to be a cutoff $w^{\\ast}$. Second, that cutoff is "
        "characterized by a one-dimensional fixed point that can be solved "
        "without iterating on the full value function, giving a closed enough "
        "benchmark to audit any discretization.\n\n"
        "Economically, $w^{\\ast}$ is the price of search: the wage at which the "
        "marginal continuation gain from waiting equals the foregone earnings "
        "from rejecting today. Patience, the benefit level, and the right tail "
        "of the offer distribution all push it up; impatience and bad benefits "
        "pull it down.\n\n"
        "This is the worker-side primitive behind frictional unemployment. "
        "The same threshold logic, with vacancies and a matching function "
        "added, drives [Diamond-Mortensen-Pissarides search and matching]"
        "(../diamond-mortensen-pissarides/). On the recursive-methods side, "
        "[cake eating](../cake-eating/) shares the scalar Bellman structure "
        "without choice under uncertainty, while [income risk and buffer-stock "
        "saving](../consumption-savings/) keeps the continuation expectation "
        "but adds a continuous endogenous state. The wage discretization here "
        "uses the same conditional-mean trick that [shock discretization]"
        "(../shock-discretization/) applies to AR(1) processes."
    )

    report.add_equations(
        r"""
Let $W$ be a wage offer with distribution $F$, and let $w$ denote the current
realization. The worker discounts at $\beta\in(0,1)$. Accepting locks in a
permanent income stream worth

$$A(w)=\frac{w}{1-\beta}.$$

Rejecting yields the unemployment benefit $b$ today plus the continuation
value of being unemployed tomorrow. Because today's offer is discarded on
rejection, that continuation does not depend on $w$:

$$C=b+\beta\,\mathbb{E}_{F}[V(W')].$$

The Bellman equation is

$$V(w)=\max\left(\frac{w}{1-\beta}, C\right).$$

Since $A(w)$ is strictly increasing in $w$ and $C$ is constant, the optimal
policy is the threshold $w^{\ast}$ defined by indifference,
$A(w^{\ast})=C$, i.e.

$$\frac{w^{\ast}}{1-\beta}=b+\beta\,\mathbb{E}_{F}[V(W')].$$

Plugging $V(W')=\max(W'/(1-\beta),C)$ back in and using
$C=w^{\ast}/(1-\beta)$ gives a scalar fixed point in $w^{\ast}$ alone:

$$w^{\ast}=(1-\beta)\,b+\beta\,\mathbb{E}_{F}[\max(W',w^{\ast})].$$

Three margins read off this equation directly. A higher $b$ raises the floor
on the right-hand side. A higher $\beta$ scales up the continuation term and
makes the worker more selective. And a thicker right tail of $F$ lifts
$\mathbb{E}_{F}[\max(W',w^{\ast})]$ above $w^{\ast}$ even when most of the
mass sits below it, which is why the cutoff can settle far above the mean
offer in fat-tailed calibrations.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| Discount factor $\\beta$ | {beta:.2f} | Weight on the next draw |\n"
        f"| Flow benefit $b$ | {b:.1f} | Per-period payoff while unemployed |\n"
        f"| Wage law | $\\log W\\sim N({mu:.1f},{sigma:.1f}^2)$ | Lognormal offer distribution |\n"
        f"| Median offer | {median_wage:.4f} | $e^{{\\mu}}$ for the lognormal |\n"
        f"| Mean offer $\\mathbb{{E}}[W]$ | {mean_wage:.4f} | Reference level, not a bound on $w^{{\\ast}}$ |\n"
        f"| Wage grid | {n_w} equiprobable bins | Each bin represented by its conditional mean |\n"
        f"| Continuous benchmark | exact lognormal moments | Ground-truth cutoff via scalar fixed point |\n"
        f"| VFI tolerance | {tol:.0e} | Sup-norm stopping rule |"
    )

    report.add_solution_method(
        "**Why VFI is essentially scalar here.** The Bellman operator "
        "$T$ acting on a candidate $V$ is\n\n"
        "$$(TV)(w)=\\max\\left(\\frac{w}{1-\\beta}, b+\\beta\\,\\mathbb{E}_{F}[V(W')]\\right).$$\n\n"
        "It is a $\\beta$-contraction in the sup norm, so iterates converge to "
        "the unique fixed point. The novelty is that the continuation term is a "
        "single number $C=b+\\beta\\,\\mathbb{E}_{F}[V]$, recomputed once per "
        "sweep. Each iteration is therefore one inner product and one elementwise "
        "max, no interpolation and no per-state expectation.\n\n"
        "**Discretization.** The continuous lognormal is replaced by an "
        f"$n_w={n_w}$-bin discrete law with equal probabilities $1/n_w$ and "
        "support points equal to the conditional mean of each bin. This "
        "preserves $\\mathbb{E}[W]$ exactly and keeps the tail moments the "
        "reservation wage actually depends on. Quantile midpoints would compress "
        "the right tail and pull $w^{\\ast}$ downward.\n\n"
        "```text\n"
        "Algorithm  Finite-grid McCall VFI\n"
        "Inputs   wages w_1,...,w_n; probabilities p_1,...,p_n;\n"
        "           discount beta in (0,1); benefit b; tolerance epsilon\n"
        "Outputs  value V_i and reservation wage w*\n"
        "\n"
        "Initialise V_i <- w_i / (1 - beta)             # accept-everything guess\n"
        "repeat n = 0, 1, 2, ...:\n"
        "    C  <- b + beta * sum_i p_i V_i             # one expectation per sweep\n"
        "    V_i_new <- max{ w_i / (1 - beta), C }      # elementwise threshold update\n"
        "    err <- max_i | V_i_new - V_i |\n"
        "    V_i <- V_i_new\n"
        "stop when err < epsilon\n"
        "w* <- (1 - beta) * (b + beta * sum_i p_i V_i)  # invert C = w* / (1 - beta)\n"
        "```\n\n"
        "**Continuous benchmark.** With the lognormal offer law the scalar "
        "fixed-point equation\n\n"
        "$$r = (1-\\beta)\\,b+\\beta\\,m(r),\\qquad m(r)=\\mathbb{E}_{F}[\\max(W,r)],$$\n\n"
        "has a closed-form $m(r)$ in terms of the standard-normal CDF. Bracketing "
        "and Brent's method give $r$ to machine precision and provide ground "
        "truth against the grid solution.\n\n"
        "```text\n"
        "Algorithm  Continuous reservation-wage benchmark\n"
        "Inputs   beta in (0,1); benefit b; lognormal parameters mu, sigma;\n"
        "           tolerance epsilon\n"
        "Output   reservation wage r\n"
        "\n"
        "Define m(r) = r * F(r) + e^{mu + sigma^2/2} * (1 - Phi((log r - mu - sigma^2)/sigma))\n"
        "Define residual(r) = r - (1 - beta)*b - beta * m(r)\n"
        "Find a bracket [lo, hi] with residual(lo) < 0 < residual(hi)\n"
        "Solve residual(r) = 0 by Brent's method to tolerance epsilon\n"
        "```\n\n"
        f"At the baseline calibration the finite-grid VFI converges in "
        f"**{info['iterations']} iterations** to sup-norm error "
        f"**{info['error']:.2e}**, giving $w^{{\\ast}}_{{\\text{{grid}}}}={w_star:.4f}$. "
        f"The continuous benchmark returns $w^{{\\ast}}_{{\\text{{cont}}}}={w_star_cont:.4f}$, "
        f"so the discretization error is **{abs(grid_gap):.1e}** in absolute "
        "terms. The two curves overlay almost everywhere in the comparative "
        "statics below; the table at the end shows where the gap is large enough "
        "to read off."
    )

    fig1, ax1 = plt.subplots()
    ax1.plot(wages, accept_values, linewidth=2, label=r"Accept: $w/(1-\beta)$")
    ax1.axhline(
        continuation_value,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label=rf"Reject: $C={continuation_value:.2f}$",
    )
    ax1.axvline(
        w_star,
        color="black",
        linestyle=":",
        linewidth=1.8,
        label=rf"Grid $w^{{\ast}}={w_star:.2f}$",
    )
    ax1.axvline(
        w_star_cont,
        color="0.45",
        linestyle="-.",
        linewidth=1.4,
        label=rf"Continuous $w^{{\ast}}={w_star_cont:.2f}$",
    )
    ymin = min(accept_values.min(), continuation_value) - 2.0
    ymax = max(accept_values.max(), continuation_value) + 2.0
    ax1.set_ylim(ymin, ymax)
    ax1.axvspan(w_star, wages.max(), color="tab:green", alpha=0.08)
    ax1.set_xlabel("Wage offer $w$")
    ax1.set_ylabel("Lifetime value")
    ax1.set_title("Threshold logic")
    ax1.legend(loc="lower right")
    report.add_figure(
        "figures/accept-vs-reject.png",
        "Accept and reject values with finite-grid and continuous reservation wages.",
        fig1,
        description=(
            "The reservation rule reads directly off the value functions. The "
            "rising line is the permanent-income value of accepting the current "
            "offer, $w/(1-\\beta)$. The flat dashed line is the rejection value "
            "$C$, which by construction does not depend on $w$. They cross at "
            "the cutoff. The shaded region marks acceptable offers, and the two "
            f"vertical lines show how close the {n_w}-bin grid gets to the "
            f"continuous-distribution benchmark. In this calibration the worker "
            f"accepts about **{100.0 * accept_frac_grid:.1f}%** of grid offers "
            f"and **{100.0 * accept_frac_cont:.1f}%** of continuous offers, so "
            f"expected unemployment duration is roughly **{expected_duration_cont:.0f} "
            "periods** before an acceptable draw arrives."
        ),
    )

    grid_x = np.linspace(0.01, max(wages.max(), 1.5 * w_star_cont), 600)
    pdf_vals = lognorm.pdf(grid_x, s=sigma, scale=np.exp(mu))
    fig2, ax2 = plt.subplots()
    ax2.plot(grid_x, pdf_vals, color="tab:blue", linewidth=2, label="Lognormal density $f(w)$")
    accept_mask = grid_x >= w_star_cont
    ax2.fill_between(
        grid_x[accept_mask],
        pdf_vals[accept_mask],
        color="tab:green",
        alpha=0.25,
        label=rf"Accept region ($w\geq w^{{\ast}}$)",
    )
    ax2.axvline(
        w_star_cont,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=rf"$w^{{\ast}}={w_star_cont:.2f}$",
    )
    ax2.axvline(
        mean_wage,
        color="0.45",
        linestyle=":",
        linewidth=1.2,
        label=rf"$\mathbb{{E}}[W]={mean_wage:.2f}$",
    )
    ax2.axvline(
        median_wage,
        color="0.7",
        linestyle=":",
        linewidth=1.2,
        label=rf"median $={median_wage:.2f}$",
    )
    ax2.set_xlabel("Wage offer $w$")
    ax2.set_ylabel("Density")
    ax2.set_title("Where the cutoff sits in the offer distribution")
    ax2.legend(loc="upper right")
    report.add_figure(
        "figures/cutoff-on-density.png",
        "Lognormal offer density with the reservation wage and acceptance region.",
        fig2,
        description=(
            "Where the cutoff sits relative to the offer distribution makes the "
            "fat-tail intuition concrete. The mean and median of $W$ are below "
            "$w^{\\ast}$, yet rejecting almost-mean offers is optimal because "
            "the right tail of $f(w)$ — visible above $w^{\\ast}$ — is thick "
            "enough that one good draw eventually compensates for many rejected "
            "ones. Acceptance is a tail event by design."
        ),
    )

    fig3, ax3 = plt.subplots()
    ax3.plot(beta_vals, wstar_beta, linewidth=2, label=f"{n_w}-bin grid")
    ax3.plot(
        beta_vals,
        wstar_beta_cont,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Continuous benchmark",
    )
    ax3.axhline(mean_wage, color="0.55", linestyle=":", linewidth=1.2, label=r"$E[W]$")
    ax3.axhline(b, color="tab:red", linestyle=":", linewidth=1.2, label="$b$")
    ax3.set_xlabel(r"Discount factor $\beta$")
    ax3.set_ylabel(r"Reservation wage $w^{\ast}$")
    ax3.set_title("Patience and the option value of waiting")
    ax3.legend()
    report.add_figure(
        "figures/wstar-vs-beta.png",
        "Reservation wage by discount factor with continuous benchmark.",
        fig3,
        description=(
            "Patience compounds the option value of waiting. As $\\beta\\to 1$ "
            "the worker treats the next draw as nearly equivalent to today's, "
            "so the cutoff explodes upward and pushes deep into the right tail. "
            "The horizontal benefit line and the mean offer are reference points "
            "only: neither bounds $w^{\\ast}$ from above when the right tail is "
            "thick enough. The grid solution tracks the closed-form benchmark "
            "everywhere on this range; the small gap at high $\\beta$ is where "
            "discretization bites the most because the cutoff probes parts of "
            "the tail that the 50-bin grid only crudely resolves."
        ),
    )

    fig4, ax4 = plt.subplots()
    ax4.plot(b_vals, wstar_b, color="tab:red", linewidth=2, label=f"{n_w}-bin grid")
    ax4.plot(
        b_vals,
        wstar_b_cont,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Continuous benchmark",
    )
    ax4.plot(b_vals, b_vals, color="0.45", linestyle=":", linewidth=1.2, label="$45^{\\circ}$ line")
    ax4.axhline(mean_wage, color="0.55", linestyle=":", linewidth=1.2, label=r"$E[W]$")
    ax4.set_xlabel("Unemployment benefit $b$")
    ax4.set_ylabel(r"Reservation wage $w^{\ast}$")
    ax4.set_title("Benefits and the outside option")
    ax4.legend()
    report.add_figure(
        "figures/wstar-vs-benefits.png",
        "Reservation wage by unemployment benefit with continuous benchmark.",
        fig4,
        description=(
            "Generosity of benefits raises the cutoff, but never one for one. "
            "If search had no upside, $w^{\\ast}$ would track the $45^{\\circ}$ "
            "line: the worker would accept any offer above the outside option. "
            "With a non-degenerate offer distribution the slope is strictly less "
            "than one because each extra dollar of $b$ also raises the value of "
            "drawing again, partially offsetting the change in the floor. The "
            "vertical gap to the $45^{\\circ}$ line is the option value of "
            "search at that benefit level."
        ),
    )

    report.add_table(
        "tables/reservation-wages.csv",
        "Reservation wages, acceptance rates, and expected unemployment duration",
        df_table,
        description=(
            "Reading the table separates the two margins. Increasing $b$ shifts "
            "the floor on the rejection value upward; increasing $\\beta$ scales "
            "up the option value of the next draw. Both push $w^{\\ast}$ up, "
            "drag acceptance rates down, and lengthen expected unemployment "
            "duration $1/\\Pr[W\\geq w^{\\ast}]$. The discretization error is "
            "modest at moderate $\\beta$ and grows visibly at $\\beta=0.99$, "
            "where the cutoff probes parts of the right tail that the 50-bin "
            "grid resolves poorly. Refining the grid, or using the scalar "
            "fixed-point solver directly, closes the gap at negligible cost."
        ),
    )

    report.add_takeaway(
        "Sequential search collapses unemployment duration into a single "
        "endogenous price, the reservation wage. Two margins move it: a higher "
        "outside option $b$ and more patience $\\beta$, both of which raise "
        "$w^{\\ast}$ and lengthen expected duration. The cutoff sits well above "
        "the mean offer in this calibration because the lognormal right tail is "
        "thick enough that rejecting a near-mean draw buys real exposure to high "
        "wages. Two computational points generalize beyond this model. First, "
        "when the rejection value is offer-independent the Bellman operator is "
        "essentially scalar, so VFI here is a useful warm-up rather than a "
        "performance bottleneck. Second, the same threshold logic is the "
        "partial-equilibrium core of the [Diamond-Mortensen-Pissarides]"
        "(../diamond-mortensen-pissarides/) matching framework, where free "
        "entry and a matching function pin down vacancies and wages on top of "
        "exactly this worker problem."
    )

    report.add_references(
        [
            "McCall, J.J. (1970). \"Economics of Information and Job Search.\" "
            "*Quarterly Journal of Economics*, 84(1), 113-126.",
            "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. "
            "MIT Press, 4th edition, Ch. 6.",
            "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in "
            "Economic Dynamics*. Harvard University Press.",
            "Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press, 2nd edition.",
        ]
    )

    report.write("README.md")
    print(
        f"Generated README.md, {len(report._figures)} figures, "
        f"and {len(report._tables)} table."
    )


if __name__ == "__main__":
    main()
