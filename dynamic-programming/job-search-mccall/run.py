#!/usr/bin/env python3
"""McCall sequential job search.

Solves the finite-grid wage-search problem by VFI and benchmarks the
reservation wage against the scalar fixed point under the continuous lognormal
offer distribution.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import lognorm, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


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
    save_figure(fig1, "figures/accept-vs-reject.png", dpi=150)

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
    save_figure(fig2, "figures/cutoff-on-density.png", dpi=150)

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
    save_figure(fig3, "figures/wstar-vs-beta.png", dpi=150)

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
    save_figure(fig4, "figures/wstar-vs-benefits.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    df_table.to_csv("tables/reservation-wages.csv", index=False)

    # Commit baseline VFI diagnostics so the sup-norm error quoted in the
    # Solution Method section is grounded in an artifact, not only the README.
    baseline_stats = pd.DataFrame(
        {
            "metric": [
                "vfi_iterations",
                "vfi_sup_norm_error",
                "w_star_grid",
                "w_star_cont",
                "abs_grid_error",
                "accept_frac_cont",
                "expected_duration_cont",
            ],
            "value": [
                info["iterations"],
                info["error"],
                w_star,
                w_star_cont,
                abs(grid_gap),
                accept_frac_cont,
                expected_duration_cont,
            ],
        }
    )
    baseline_stats.to_csv(
        Path(__file__).resolve().parent / "tables" / "baseline-stats.csv", index=False
    )

    save_thumbnail("figures/accept-vs-reject.png", "figures/thumb.png")
    print(
        f"Generated figures and tables."
    )


if __name__ == "__main__":
    main()
