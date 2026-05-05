#!/usr/bin/env python3
"""McCall job search tutorial.

The script solves a wage-offer search problem on a finite grid and uses the
continuous lognormal offer distribution as a benchmark for the reservation wage.
It regenerates README.md, figures, tables, and the catalog thumbnail.
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
    """Approximate a lognormal distribution with equiprobable mean bins."""
    edges = np.linspace(0.0, 1.0, n + 1)
    z_edges = norm.ppf(edges)
    mean = np.exp(mu + 0.5 * sigma**2)

    # Conditional means in probability bins preserve the full mean, including
    # both tails, better than raw quantile midpoints.
    probs = np.diff(edges)
    interval_means = mean * (
        norm.cdf(z_edges[1:] - sigma) - norm.cdf(z_edges[:-1] - sigma)
    )
    wages = interval_means / probs
    return wages, probs


def expected_max_lognormal(r: float, mu: float, sigma: float) -> float:
    """Compute E[max(W, r)] for W lognormally distributed."""
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
    """Solve the scalar reservation-wage equation under the continuous law."""
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
    """Probability that a continuous lognormal offer exceeds the cutoff."""
    return float(1.0 - lognorm.cdf(w_star, s=sigma, scale=np.exp(mu)))


def solve_mccall(
    beta: float,
    b: float,
    wages: np.ndarray,
    probs: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 1_000,
) -> tuple[np.ndarray, float, dict[str, float | int | bool]]:
    """Solve the finite-grid McCall model by value function iteration."""
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

    print(
        f"Wage grid: [{wages.min():.3f}, {wages.max():.3f}], "
        f"E[W] = {mean_wage:.3f}"
    )
    print("Solving baseline finite-grid McCall model...")
    value, w_star, info = solve_mccall(beta, b, wages, probs, tol=tol)
    w_star_cont = solve_continuous_reservation_wage(beta, b, mu, sigma)
    accept_frac_grid = float(np.sum(probs[wages >= w_star]))
    accept_frac_cont = continuous_acceptance_probability(w_star_cont, mu, sigma)
    grid_gap = w_star - w_star_cont

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
                    "VFI iter.": info_t["iterations"],
                }
            )
    df_table = pd.DataFrame(table_rows)

    setup_style()

    report = ModelReport(
        "McCall Job Search and the Reservation Wage",
        "Sequential wage-offer search, option value, and threshold acceptance.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The McCall model is a small partial-equilibrium model of unemployment "
        "with a sharp economic object: the value of waiting for a better wage "
        "offer. An unemployed worker observes one offer at a time. Accepting "
        "locks in that wage forever; rejecting pays the unemployment benefit "
        "$b$ today and preserves the right to draw again tomorrow.\n\n"
        "The optimal policy is a reservation rule. The worker accepts offers at "
        "or above a cutoff $w^{*}$ and rejects offers below it. The cutoff is "
        "not a taste parameter; it is an endogenous price of search, pinned "
        "down by the wage-offer distribution, the benefit level, and the "
        "discount factor. This is the worker-side building block behind richer "
        "search models such as [search and matching unemployment](../diamond-mortensen-pissarides/). "
        "It also echoes the logic in [income-risk saving](../consumption-savings/): "
        "a current choice is valuable because it changes exposure to future states."
    )

    report.add_equations(
        r"""
Let $W$ denote a wage offer drawn from distribution $F$, and let $w$ be the
current realization. The worker discounts next period by $\beta \in (0,1)$.
Accepting $w$ gives the permanent value

$$A(w)=\frac{w}{1-\beta}.$$

Rejecting gives the common continuation value

$$C=b+\beta \mathbb{E}_{F}[V(W')],$$

so the Bellman equation is

$$V(w)=\max\bigl[ \frac{w}{1-\beta},\; b+\beta \mathbb{E}_{F}[V(W')] \bigr].$$

The reservation wage is the offer that makes the worker indifferent:

$$\frac{w^{*}}{1-\beta}=C.$$

Using this indifference condition inside the Bellman equation gives the scalar
fixed point

$$w^{*}=(1-\beta)b+\beta \mathbb{E}_{F}[\max\{W',w^{*}\}].$$

This last equation is useful for interpretation. A higher $b$ raises the value
of rejection directly. A higher $\beta$ raises the option value of future draws.
A thicker right tail also raises $w^{*}$ because rejecting a mediocre offer buys
exposure to rare high wages.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|---|---:|---|\n"
        f"| Discount factor $\\beta$ | {beta:.2f} | Weight on future search opportunities |\n"
        f"| Flow benefit $b$ | {b:.1f} | Payoff while unemployed |\n"
        f"| Wage law | $\\log W \\sim N({mu:.1f},{sigma:.1f}^2)$ | Offer distribution |\n"
        f"| Mean offer $\\mathbb{{E}}[W]$ | {mean_wage:.4f} | Reference level, not an upper bound on $w^{{*}}$ |\n"
        f"| Main wage grid | {n_w} bins | Equiprobable bins represented by conditional means |\n"
        f"| Continuous benchmark | lognormal tail moments | Held-out reservation-wage check |\n"
        f"| VFI tolerance | {tol:.0e} | Sup-norm stopping rule |"
    )

    report.add_solution_method(
        "On a finite offer grid, the Bellman iteration is especially simple "
        "because the value of rejecting does not depend on the current offer. "
        "Each iteration computes one expected continuation value and then "
        "compares it with the lifetime value of accepting each grid wage.\n\n"
        "```text\n"
        "Algorithm: finite-grid McCall VFI\n"
        "Input: wages w_i, probabilities p_i, beta, benefit b, tolerance epsilon\n"
        "Output: value function V_i and reservation wage w*\n"
        "Initialize V_i = w_i / (1 - beta)\n"
        "repeat for n = 0, 1, 2, ...:\n"
        "    C_n = b + beta * sum_i p_i V_i\n"
        "    V_i_new = max{w_i / (1 - beta), C_n} for every wage i\n"
        "    error = max_i |V_i_new - V_i|\n"
        "    set V_i = V_i_new\n"
        "until error < epsilon\n"
        "set w* = (1 - beta) * (b + beta * sum_i p_i V_i)\n"
        "```\n\n"
        "The continuous benchmark uses the scalar reservation-wage equation "
        "rather than the value function. For a candidate cutoff $r$, compute "
        "$m(r)=\\mathbb{E}[\\max\\{W,r\\}]$ under the lognormal distribution and "
        "find the root of $r-(1-\\beta)b-\\beta m(r)=0$ by bracketing.\n\n"
        "```text\n"
        "Algorithm: continuous reservation-wage benchmark\n"
        "Input: beta, b, lognormal parameters mu and sigma, tolerance epsilon\n"
        "Output: continuous-distribution cutoff r\n"
        "Define m(r) = E_F[max{W, r}] using lognormal tail moments\n"
        "Find a bracket [low, high] with residual(low) < 0 < residual(high)\n"
        "Solve residual(r) = r - (1 - beta)*b - beta*m(r) = 0\n"
        "```\n\n"
        f"The finite-grid VFI converged in **{info['iterations']} iterations** "
        f"with sup-norm error **{info['error']:.2e}**. The baseline cutoff is "
        f"$w^{{*}}_{{grid}}={w_star:.4f}$, compared with the continuous "
        f"lognormal benchmark $w^{{*}}_{{cont}}={w_star_cont:.4f}$."
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
        label=rf"Grid $w^{{*}}={w_star:.2f}$",
    )
    ax1.axvline(
        w_star_cont,
        color="0.45",
        linestyle="-.",
        linewidth=1.4,
        label=rf"Continuous $w^{{*}}={w_star_cont:.2f}$",
    )
    ymin = min(accept_values.min(), continuation_value) - 2.0
    ymax = max(accept_values.max(), continuation_value) + 2.0
    ax1.set_ylim(ymin, ymax)
    ax1.axvspan(w_star, wages.max(), color="tab:green", alpha=0.08)
    ax1.set_xlabel("Wage offer $w$")
    ax1.set_ylabel("Lifetime value")
    ax1.set_title("Threshold Logic")
    ax1.legend(loc="lower right")
    report.add_figure(
        "figures/accept-vs-reject.png",
        "Accept and reject values with finite-grid and continuous reservation wages.",
        fig1,
        description=(
            "The first figure shows the reservation rule in value units. "
            "Accepting is linear in the current offer, while rejecting is flat "
            "because it depends only on the distribution of future offers. In "
            f"the baseline calibration the finite grid accepts about "
            f"**{100.0 * accept_frac_grid:.1f}%** of grid offers. The continuous "
            f"benchmark accepts **{100.0 * accept_frac_cont:.1f}%** of offers."
        ),
    )

    fig2, ax2 = plt.subplots()
    ax2.plot(beta_vals, wstar_beta, linewidth=2, label="50-bin grid")
    ax2.plot(
        beta_vals,
        wstar_beta_cont,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Continuous benchmark",
    )
    ax2.axhline(mean_wage, color="0.55", linestyle=":", linewidth=1.2, label=r"$E[W]$")
    ax2.axhline(b, color="tab:red", linestyle=":", linewidth=1.2, label="$b$")
    ax2.set_xlabel("Discount factor $\\beta$")
    ax2.set_ylabel("Reservation wage $w^{*}$")
    ax2.set_title("Patience and Search Option Value")
    ax2.legend()
    report.add_figure(
        "figures/wstar-vs-beta.png",
        "Reservation wage by discount factor with continuous benchmark.",
        fig2,
        description=(
            "Patience makes the worker more selective because the future draw "
            "arrives almost as valuable as the current payoff. The cutoff can "
            "rise above the mean offer in a right-skewed distribution; the mean "
            "is a reference point, not a bound on optimal selectivity."
        ),
    )

    fig3, ax3 = plt.subplots()
    ax3.plot(b_vals, wstar_b, color="tab:red", linewidth=2, label="50-bin grid")
    ax3.plot(
        b_vals,
        wstar_b_cont,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Continuous benchmark",
    )
    ax3.plot(b_vals, b_vals, color="0.45", linestyle=":", linewidth=1.2, label="45-degree line")
    ax3.axhline(mean_wage, color="0.55", linestyle=":", linewidth=1.2, label=r"$E[W]$")
    ax3.set_xlabel("Unemployment benefit $b$")
    ax3.set_ylabel("Reservation wage $w^{*}$")
    ax3.set_title("Benefits and the Outside Option")
    ax3.legend()
    report.add_figure(
        "figures/wstar-vs-benefits.png",
        "Reservation wage by unemployment benefit with continuous benchmark.",
        fig3,
        description=(
            "Benefits move the outside option one for one only in the limiting "
            "case where search has no upside. With a non-degenerate wage offer "
            "distribution, a higher benefit also preserves the option to wait, "
            "so the cutoff remains above the 45-degree line over this range."
        ),
    )

    report.add_table(
        "tables/reservation-wages.csv",
        "Reservation wages and continuous-benchmark acceptance rates",
        df_table,
        description=(
            "The parameter grid separates two margins. Increasing $b$ improves "
            "the payoff from unemployment today. Increasing $\\beta$ raises the "
            "present value of future draws. Both margins reduce the probability "
            "that a fresh offer is accepted."
        ),
    )

    report.add_takeaway(
        "The McCall model recasts unemployment duration as a reservation-price "
        "problem. The worker rejects low offers not because the current benefit "
        "is large by itself, but because rejecting preserves a claim on future "
        "wage draws. In this calibration the right tail is important enough "
        "that the cutoff can sit above the mean offer, a point the finite-grid "
        "and continuous benchmarks make explicit. The same acceptance-threshold "
        "logic is the partial-equilibrium core of larger search models with "
        "matching frictions, endogenous vacancies, and equilibrium wage determination."
    )

    report.add_references(
        [
            "McCall, J.J. (1970). \"Economics of Information and Job Search.\" "
            "*Quarterly Journal of Economics*, 84(1), 113-126.",
            "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. "
            "MIT Press, 4th edition, Ch. 6.",
            "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in "
            "Economic Dynamics*. Harvard University Press.",
            "Pissarides, C.A. (2000). *Equilibrium Unemployment Theory*. MIT Press.",
        ]
    )

    report.write("README.md")
    print(
        f"Generated README.md, {len(report._figures)} figures, "
        f"and {len(report._tables)} table."
    )


if __name__ == "__main__":
    main()
