#!/usr/bin/env python3
"""McCall Job Search Model: Optimal Sequential Search by an Unemployed Worker.

Solves the McCall (1970) job search model using value function iteration
with JAX. An unemployed worker draws wage offers from a known distribution
and must decide each period whether to accept (work forever at that wage)
or reject (collect unemployment benefit, draw again next period).

Reference: McCall, J.J. (1970). "Economics of Information and Job Search."
           Quarterly Journal of Economics, 84(1), 113-126.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def discretize_lognormal(mu: float, sigma: float, n: int):
    """Discretize a lognormal distribution into n points with probabilities.

    Returns (wages, probabilities) where wages are evenly spaced quantile
    midpoints and probabilities sum to 1.
    """
    # Use quantile-based discretization for better coverage
    dist = lognorm(s=sigma, scale=np.exp(mu))
    # Create n evenly spaced quantile bins
    quantile_edges = np.linspace(0.01, 0.99, n + 1)
    wages = np.array([
        dist.ppf(0.5 * (quantile_edges[i] + quantile_edges[i + 1]))
        for i in range(n)
    ])
    # Probabilities proportional to the mass in each bin
    probs = np.diff(dist.cdf(
        np.array([dist.ppf(q) for q in quantile_edges])
    ))
    probs = probs / probs.sum()  # normalize
    return wages, probs


def solve_mccall(beta: float, b: float, wages: np.ndarray, probs: np.ndarray,
                 tol: float = 1e-8, max_iter: int = 1000):
    """Solve the McCall model via value function iteration.

    Bellman equation:
        V(w) = max{ w/(1-beta),  b + beta * E[V(w')] }

    The continuation value c_val = b + beta * E[V(w')] is the same for all w,
    so the reservation wage w* satisfies: w*/(1-beta) = c_val.

    Args:
        beta: Discount factor.
        b: Unemployment benefit (per-period flow).
        wages: Discretized wage grid (n_w,).
        probs: Probability of each wage (n_w,), sums to 1.
        tol: Convergence tolerance.
        max_iter: Maximum VFI iterations.

    Returns:
        V: Value function on wage grid.
        reservation_wage: The reservation wage w*.
        info: Dict with convergence details.
    """
    n_w = len(wages)
    # Value of accepting each wage forever
    accept_values = wages / (1.0 - beta)

    # Initial guess: V(w) = accept value
    V = accept_values.copy()

    for iteration in range(1, max_iter + 1):
        # Continuation value (same for all w)
        continuation = b + beta * np.dot(probs, V)

        # Bellman update
        V_new = np.maximum(accept_values, continuation)

        error = np.max(np.abs(V_new - V))
        if iteration % 50 == 0:
            print(f"  VFI iteration {iteration:3d}, error = {error:.2e}")
        V = V_new

        if error < tol:
            print(f"  VFI converged in {iteration} iterations (error = {error:.2e})")
            break

    # Reservation wage: w*/(1-beta) = continuation value
    continuation = b + beta * np.dot(probs, V)
    reservation_wage = (1.0 - beta) * continuation

    info = {
        "iterations": iteration,
        "converged": error < tol,
        "error": float(error),
        "continuation_value": float(continuation),
    }
    return V, reservation_wage, info


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    beta = 0.95       # Discount factor
    b = 1.0           # Unemployment benefit (per-period)
    mu = 0.0          # Log-mean of wage distribution
    sigma = 1.0       # Log-std of wage distribution
    n_w = 50          # Number of discrete wage points
    tol = 1e-8        # Convergence tolerance

    # =========================================================================
    # Discretize wage distribution
    # =========================================================================
    wages, probs = discretize_lognormal(mu, sigma, n_w)
    print(f"Wage grid: [{wages.min():.3f}, {wages.max():.3f}], "
          f"E[w] = {np.dot(probs, wages):.3f}")

    # =========================================================================
    # Solve the baseline model
    # =========================================================================
    print("\nSolving baseline McCall model...")
    V, w_star, info = solve_mccall(beta, b, wages, probs, tol=tol)

    print(f"  Reservation wage w* = {w_star:.4f}")
    print(f"  E[w] = {np.dot(probs, wages):.4f}")
    print(f"  Fraction of offers accepted = "
          f"{np.sum(probs[wages >= w_star]):.4f}")

    # Accept/reject values for plotting
    accept_values = wages / (1.0 - beta)
    continuation = b + beta * np.dot(probs, V)

    # =========================================================================
    # Comparative statics: reservation wage vs beta
    # =========================================================================
    print("\nComparative statics: w*(beta)...")
    beta_vals = np.linspace(0.80, 0.99, 40)
    wstar_beta = np.zeros_like(beta_vals)
    for i, beta_i in enumerate(beta_vals):
        _, wstar_beta[i], _ = solve_mccall(beta_i, b, wages, probs, tol=tol)
    print(f"  w* range: [{wstar_beta.min():.3f}, {wstar_beta.max():.3f}]")

    # =========================================================================
    # Comparative statics: reservation wage vs b
    # =========================================================================
    print("\nComparative statics: w*(b)...")
    b_vals = np.linspace(0.0, 3.0, 40)
    wstar_b = np.zeros_like(b_vals)
    for i, b_i in enumerate(b_vals):
        _, wstar_b[i], _ = solve_mccall(beta, b_i, wages, probs, tol=tol)
    print(f"  w* range: [{wstar_b.min():.3f}, {wstar_b.max():.3f}]")

    # =========================================================================
    # Table: reservation wages for parameter grid
    # =========================================================================
    print("\nBuilding parameter table...")
    beta_table = [0.90, 0.95, 0.99]
    b_table = [0.5, 1.0, 2.0]
    table_rows = []
    for beta_t in beta_table:
        for b_t in b_table:
            _, w_star_t, info_t = solve_mccall(beta_t, b_t, wages, probs, tol=tol)
            accept_frac = float(np.sum(probs[wages >= w_star_t]))
            table_rows.append({
                "beta": f"{beta_t:.2f}",
                "b": f"{b_t:.1f}",
                "w*": f"{w_star_t:.4f}",
                "Accept %": f"{100 * accept_frac:.1f}",
                "Iterations": info_t["iterations"],
            })
    df_table = pd.DataFrame(table_rows)

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "McCall Job Search Model",
        "Optimal sequential search by an unemployed worker drawing wage offers "
        "from a known distribution.",
    )

    report.add_overview(
        "The McCall (1970) job search model is the foundational model of frictional "
        "unemployment. An unemployed worker sequentially draws wage offers from a known "
        "distribution. Each period, the worker must make an irreversible decision: accept "
        "the current offer (and work at that wage forever) or reject it (receive unemployment "
        "benefit $b$ and draw a new offer next period).\n\n"
        "The key question: how selective should the worker be? The answer is a simple "
        "**reservation wage** policy: accept any offer above a threshold $w^*$ and reject "
        "anything below it. This threshold balances the cost of continued search (foregone "
        "wages) against the option value of potentially finding a better match."
    )

    report.add_equations(
        r"""
$$V(w) = \max\left\{ \frac{w}{1-\beta},\; b + \beta \, \mathbb{E}[V(w')] \right\}$$

where $w/(1-\beta)$ is the lifetime value of accepting wage $w$ forever, and
$b + \beta \, \mathbb{E}[V(w')]$ is the value of rejecting (collecting benefit $b$
today, then drawing a new offer $w'$ tomorrow).

**Reservation wage** $w^*$ solves:
$$\frac{w^*}{1-\beta} = b + \beta \, \mathbb{E}[V(w')]$$

The optimal policy is a threshold rule: accept if $w \ge w^*$, reject otherwise.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$  | {beta} | Discount factor |\n"
        f"| $b$       | {b} | Unemployment benefit (per period) |\n"
        f"| $\\mu$    | {mu} | Log-mean of wage distribution |\n"
        f"| $\\sigma$ | {sigma} | Log-std of wage distribution |\n"
        f"| Grid      | {n_w} points | Quantile-based discretization |"
    )

    report.add_solution_method(
        "**Value Function Iteration (VFI):** Starting from an initial guess "
        "$V_0(w) = w/(1-\\beta)$ (accept everything), we iterate on the Bellman equation:\n\n"
        "$$V_{n+1}(w) = \\max\\left\\{ \\frac{w}{1-\\beta},\\; "
        "b + \\beta \\sum_{w'} p(w') V_n(w') \\right\\}$$\n\n"
        "The continuation value $b + \\beta \\, \\mathbb{E}[V(w')]$ is the same for all wages, "
        "which makes the McCall model particularly clean: each iteration requires only a single "
        "dot product to compute the expected value, then an elementwise max.\n\n"
        f"Converged in **{info['iterations']} iterations** (error = {info['error']:.2e}).\n\n"
        f"**Baseline reservation wage:** $w^* = {w_star:.4f}$"
    )

    # --- Figure 1: Accept vs Reject Values ---
    fig1, ax1 = plt.subplots()
    ax1.plot(wages, accept_values, "b-", linewidth=2,
             label=r"Accept: $w/(1-\beta)$")
    ax1.axhline(continuation, color="r", linestyle="--", linewidth=2,
                label=f"Reject: $b + \\beta E[V(w')]$ = {continuation:.2f}")
    ax1.axvline(w_star, color="k", linestyle=":", linewidth=1.5, alpha=0.7,
                label=f"$w^* = {w_star:.3f}$")
    ax1.fill_between(wages, 0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 100,
                     where=wages >= w_star, alpha=0.08, color="green",
                     label="Accept region")
    # Reset ylim after fill_between
    ymin = min(accept_values.min(), continuation) - 2
    ymax = max(accept_values.max(), continuation) + 2
    ax1.set_ylim(ymin, ymax)
    ax1.fill_between(wages, ymin, ymax,
                     where=wages >= w_star, alpha=0.08, color="green")
    ax1.set_xlabel("Wage offer $w$")
    ax1.set_ylabel("Lifetime value")
    ax1.set_title("Accept vs Reject: Identifying the Reservation Wage")
    ax1.legend(loc="lower right")
    report.add_figure("figures/accept-vs-reject.png",
                      "Value of accepting vs rejecting each wage offer. "
                      "The reservation wage is where the two curves intersect.",
                      fig1)

    # --- Figure 2: Reservation wage vs beta ---
    fig2, ax2 = plt.subplots()
    ax2.plot(beta_vals, wstar_beta, "b-", linewidth=2)
    ax2.axhline(np.dot(probs, wages), color="gray", linestyle=":", alpha=0.5,
                label=f"$E[w] = {np.dot(probs, wages):.2f}$")
    ax2.axhline(b, color="r", linestyle=":", alpha=0.5,
                label=f"$b = {b:.1f}$")
    ax2.set_xlabel("Discount factor $\\beta$")
    ax2.set_ylabel("Reservation wage $w^*$")
    ax2.set_title("Reservation Wage vs Patience")
    ax2.legend()
    report.add_figure("figures/wstar-vs-beta.png",
                      "More patient workers (higher beta) are more selective, "
                      "demanding higher wages before accepting.",
                      fig2)

    # --- Figure 3: Reservation wage vs b ---
    fig3, ax3 = plt.subplots()
    ax3.plot(b_vals, wstar_b, "r-", linewidth=2)
    ax3.plot(b_vals, b_vals, "k:", linewidth=1, alpha=0.5, label="45-degree line")
    ax3.axhline(np.dot(probs, wages), color="gray", linestyle=":", alpha=0.5,
                label=f"$E[w] = {np.dot(probs, wages):.2f}$")
    ax3.set_xlabel("Unemployment benefit $b$")
    ax3.set_ylabel("Reservation wage $w^*$")
    ax3.set_title("Reservation Wage vs Unemployment Benefits")
    ax3.legend()
    report.add_figure("figures/wstar-vs-benefits.png",
                      "Higher unemployment benefits raise the reservation wage: "
                      "workers can afford to be more selective when the safety net is generous.",
                      fig3)

    # --- Table ---
    report.add_table("tables/reservation-wages.csv",
                     "Reservation wage and acceptance probability for different "
                     "parameter combinations",
                     df_table)

    report.add_takeaway(
        "The McCall model provides the cleanest illustration of the **search-theoretic "
        "approach** to unemployment. Unemployment is not involuntary idleness but an "
        "investment in finding a better match.\n\n"
        "**Key insights:**\n"
        "- The optimal policy is a simple **threshold rule**: accept any wage above $w^*$, "
        "reject below. No complicated history-dependence is needed.\n"
        "- **Higher unemployment benefits $\\Rightarrow$ higher reservation wage $\\Rightarrow$ "
        "longer unemployment spells** but better eventual matches. This is the core policy "
        "trade-off in unemployment insurance design.\n"
        "- **More patient workers (higher $\\beta$) are more selective.** A patient worker "
        "values the option of future draws more, so they hold out for better offers.\n"
        "- The reservation wage $w^*$ is *always above* the unemployment benefit $b$ "
        "(since the option value of search is positive) but *below* the expected wage "
        "$E[w]$ (since accepting a good offer now avoids the risk of worse draws later).\n"
        "- Despite its simplicity, this model is the building block for richer search models "
        "with on-the-job search, matching functions, and equilibrium wage determination "
        "(Mortensen-Pissarides)."
    )

    report.add_references([
        "McCall, J.J. (1970). \"Economics of Information and Job Search.\" "
        "*Quarterly Journal of Economics*, 84(1), 113-126.",
        "Ljungqvist, L. and Sargent, T. (2018). *Recursive Macroeconomic Theory*. "
        "MIT Press, 4th edition, Ch. 6.",
        "Stokey, N., Lucas, R., and Prescott, E. (1989). *Recursive Methods in "
        "Economic Dynamics*. Harvard University Press.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + "
          f"{len(report._tables)} tables")


if __name__ == "__main__":
    main()
