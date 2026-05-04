#!/usr/bin/env python3
"""Projection methods with Chebyshev polynomials.

The tutorial solves a small dynamic problem by approximating the policy function
directly. It uses a model with a known closed-form solution so approximation
error and Euler equation residuals can be inspected rather than guessed.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


BETA = 0.95
ALPHA = 0.36
PRODUCTIVITY = 1.0
N_BASIS_MAIN = 8


def production(capital: np.ndarray | float) -> np.ndarray | float:
    """Cobb-Douglas output with full depreciation."""
    return PRODUCTIVITY * np.asarray(capital) ** ALPHA


def true_policy(capital: np.ndarray | float) -> np.ndarray | float:
    """Closed-form next-period capital for log utility and full depreciation."""
    return ALPHA * BETA * production(capital)


def true_consumption(capital: np.ndarray | float) -> np.ndarray | float:
    """Closed-form consumption policy."""
    return (1.0 - ALPHA * BETA) * production(capital)


def steady_state_capital() -> float:
    """Return the deterministic steady state under the true policy."""
    return float((ALPHA * BETA * PRODUCTIVITY) ** (1.0 / (1.0 - ALPHA)))


def scale_to_unit(capital: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map capital from [lower, upper] to [-1, 1]."""
    return 2.0 * (capital - lower) / (upper - lower) - 1.0


def unit_to_capital(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map Chebyshev coordinates from [-1, 1] to capital."""
    return lower + 0.5 * (x + 1.0) * (upper - lower)


def cheb_matrix(capital: np.ndarray, n_basis: int, lower: float, upper: float) -> np.ndarray:
    """Chebyshev Vandermonde matrix evaluated on the capital grid."""
    x = scale_to_unit(np.asarray(capital), lower, upper)
    return np.polynomial.chebyshev.chebvander(x, n_basis - 1)


def policy_from_coeffs(
    coeffs: np.ndarray,
    capital: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Evaluate the positive policy approximation k' = exp(T(k) theta)."""
    log_policy = cheb_matrix(capital, len(coeffs), lower, upper) @ coeffs
    return np.exp(log_policy)


def euler_residuals(
    coeffs: np.ndarray,
    capital: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Log Euler residuals at collocation points."""
    next_capital = policy_from_coeffs(coeffs, capital, lower, upper)
    consumption = production(capital) - next_capital
    future_next_capital = policy_from_coeffs(coeffs, next_capital, lower, upper)
    future_consumption = production(next_capital) - future_next_capital
    marginal_product = ALPHA * PRODUCTIVITY * next_capital ** (ALPHA - 1.0)

    invalid = (
        (next_capital <= lower)
        | (next_capital >= upper)
        | (consumption <= 0.0)
        | (future_consumption <= 0.0)
    )
    ratio = BETA * marginal_product * consumption / future_consumption
    residual = np.log(np.maximum(ratio, 1e-300))
    residual = np.where(invalid, 1e3, residual)
    return residual


def solve_projection(
    n_basis: int,
    collocation_capital: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """Solve collocation equations for Chebyshev policy coefficients."""
    initial_policy = 0.75 * true_policy(collocation_capital) + 0.25 * collocation_capital
    initial = np.linalg.lstsq(
        cheb_matrix(collocation_capital, n_basis, lower, upper),
        np.log(initial_policy),
        rcond=None,
    )[0]
    result = least_squares(
        euler_residuals,
        initial,
        args=(collocation_capital, lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=5_000,
    )
    return result.x


def chebyshev_nodes(n_nodes: int, lower: float, upper: float) -> np.ndarray:
    """Return Chebyshev roots mapped to the capital interval."""
    j = np.arange(1, n_nodes + 1)
    roots = np.cos((2.0 * j - 1.0) * np.pi / (2.0 * n_nodes))
    return np.sort(unit_to_capital(roots, lower, upper))


def approximation_table(
    basis_counts: list[int],
    eval_grid: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Solve several projection orders and summarize accuracy."""
    rows = []
    coeffs_by_basis = {}
    for n_basis in basis_counts:
        nodes = chebyshev_nodes(n_basis, lower, upper)
        coeffs = solve_projection(n_basis, nodes, lower, upper)
        coeffs_by_basis[n_basis] = coeffs
        projected = policy_from_coeffs(coeffs, eval_grid, lower, upper)
        policy_error = np.abs(projected - true_policy(eval_grid))
        euler_error = np.abs(np.exp(euler_residuals(coeffs, eval_grid, lower, upper)) - 1.0)
        rows.append(
            {
                "Basis terms": n_basis,
                "Max policy error": np.max(policy_error),
                "Median policy error": np.median(policy_error),
                "Max Euler error": np.max(euler_error),
                "Median Euler error": np.median(euler_error),
            }
        )
    return pd.DataFrame(rows), coeffs_by_basis


def simulate_capital_path(
    policy_coeffs: np.ndarray,
    initial_capital: float,
    lower: float,
    upper: float,
    periods: int = 35,
) -> np.ndarray:
    """Simulate capital using the projected policy."""
    path = np.empty(periods, dtype=float)
    path[0] = initial_capital
    for t in range(1, periods):
        path[t] = policy_from_coeffs(policy_coeffs, np.array([path[t - 1]]), lower, upper)[0]
    return path


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format projection accuracy table."""
    out = df.copy()
    for col in out.columns:
        if col == "Basis terms":
            out[col] = out[col].map(lambda x: f"{int(x)}")
        else:
            out[col] = out[col].map(lambda x: f"{float(x):.2e}")
    return out


def main() -> None:
    setup_style()
    k_ss = steady_state_capital()
    lower = 0.25 * k_ss
    upper = 1.75 * k_ss
    eval_grid = np.linspace(lower, upper, 320)
    basis_counts = [2, 3, 5, 8]
    table, coeffs_by_basis = approximation_table(basis_counts, eval_grid, lower, upper)
    main_coeffs = coeffs_by_basis[N_BASIS_MAIN]
    main_policy = policy_from_coeffs(main_coeffs, eval_grid, lower, upper)
    main_euler_error = np.abs(np.exp(euler_residuals(main_coeffs, eval_grid, lower, upper)) - 1.0)

    print("Projection methods tutorial")
    print(f"  steady state capital={k_ss:.4f}")
    print(f"  basis terms={N_BASIS_MAIN}")
    print(f"  max Euler error={np.max(main_euler_error):.2e}")

    report = ModelReport(
        "Projection Methods with Chebyshev Polynomials",
        "Function approximation and Euler-equation residuals in a dynamic decision problem.",
    )

    report.add_overview(
        "Projection methods solve models by approximating an unknown function with a small "
        "number of basis functions. Instead of storing a value or policy at every grid point, "
        "we choose coefficients that make the model's residual equations close to zero.\n\n"
        "This tutorial uses Chebyshev polynomials to approximate the capital policy in a simple "
        "growth problem. The example is intentionally transparent: the true policy is known, so "
        "the reader can see how collocation, approximation order, and Euler residuals fit together."
    )

    report.add_equations(
        r"""
The planner chooses next-period capital:

$$
V(k) = \max_{k'} \bigl[\log(c) + \beta V(k')\bigr],
\qquad
c = A k^\alpha - k'.
$$

The Euler equation is:

$$
\frac{1}{c_t}
= \beta \frac{\alpha A k_{t+1}^{\alpha-1}}{c_{t+1}}.
$$

Projection approximates the policy with Chebyshev basis functions:

$$
\log g(k;\theta)
= \sum_{j=0}^{n-1} \theta_j T_j(x(k)),
\qquad
x(k) \in [-1,1].
$$
"""
    )

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        f"| Discount factor $\\beta$ | {BETA:.2f} |\n"
        f"| Capital share $\\alpha$ | {ALPHA:.2f} |\n"
        f"| Productivity $A$ | {PRODUCTIVITY:.1f} |\n"
        f"| Steady-state capital | {k_ss:.4f} |\n"
        f"| Approximation interval | [{lower:.4f}, {upper:.4f}] |\n"
        f"| Main basis terms | {N_BASIS_MAIN} |"
    )

    report.add_solution_method(
        "The code maps capital into the Chebyshev domain [-1,1], represents log next-period "
        "capital as a Chebyshev polynomial, and chooses coefficients by collocation. The residual "
        "at each collocation node is the log Euler equation error.\n\n"
        "The known closed-form policy is used only as a benchmark for the tutorial. In a model "
        "without a closed-form solution, the same workflow would use residuals and simulation "
        "diagnostics to judge accuracy."
    )

    x_plot = np.linspace(-1.0, 1.0, 300)
    basis = np.polynomial.chebyshev.chebvander(x_plot, N_BASIS_MAIN - 1)
    fig1, ax1 = plt.subplots(figsize=(8.0, 5.2))
    for j in range(N_BASIS_MAIN):
        ax1.plot(x_plot, basis[:, j], label=f"T{j}")
    ax1.set_xlabel("Chebyshev coordinate x")
    ax1.set_ylabel("Basis value")
    ax1.set_title("First Eight Chebyshev Basis Functions")
    ax1.legend(ncol=4, fontsize=8)
    report.add_figure(
        "figures/chebyshev-basis.png",
        "Chebyshev basis functions on [-1,1]",
        fig1,
        description=(
            "Chebyshev polynomials oscillate in a controlled way over the approximation interval. "
            "They are popular because they approximate smooth functions accurately with few terms."
        ),
    )

    fig2, ax2 = plt.subplots()
    ax2.plot(eval_grid, true_policy(eval_grid), color="black", linestyle="--", label="closed form")
    for n_basis in [3, 5, 8]:
        ax2.plot(
            eval_grid,
            policy_from_coeffs(coeffs_by_basis[n_basis], eval_grid, lower, upper),
            label=f"{n_basis} terms",
        )
    ax2.scatter(
        chebyshev_nodes(N_BASIS_MAIN, lower, upper),
        true_policy(chebyshev_nodes(N_BASIS_MAIN, lower, upper)),
        color="crimson",
        s=28,
        label="8 collocation nodes",
        zorder=5,
    )
    ax2.set_xlabel("Capital today")
    ax2.set_ylabel("Capital tomorrow")
    ax2.set_title("Projected Capital Policy")
    ax2.legend()
    report.add_figure(
        "figures/policy-functions.png",
        "Projected policy functions against the closed-form policy",
        fig2,
        description=(
            "More basis terms allow the projected policy to follow the curvature of the true "
            "decision rule over the full interval."
        ),
    )

    fig3, ax3 = plt.subplots()
    for n_basis in basis_counts:
        errors = np.abs(
            np.exp(euler_residuals(coeffs_by_basis[n_basis], eval_grid, lower, upper)) - 1.0
        )
        ax3.plot(eval_grid, np.log10(np.maximum(errors, 1e-14)), label=f"{n_basis} terms")
    ax3.set_xlabel("Capital today")
    ax3.set_ylabel("log10 absolute Euler error")
    ax3.set_title("Euler Equation Errors")
    ax3.legend()
    report.add_figure(
        "figures/euler-errors.png",
        "Euler equation errors by approximation order",
        fig3,
        description=(
            "Euler errors are the main diagnostic. They ask whether the approximated policy "
            "satisfies the model's optimality condition away from the collocation nodes."
        ),
    )

    fig4, ax4 = plt.subplots()
    for initial in [0.4 * k_ss, 1.6 * k_ss]:
        projected_path = simulate_capital_path(main_coeffs, initial, lower, upper)
        true_path = np.empty_like(projected_path)
        true_path[0] = initial
        for t in range(1, len(true_path)):
            true_path[t] = true_policy(true_path[t - 1])
        ax4.plot(true_path, color="black", linestyle="--", alpha=0.85)
        ax4.plot(projected_path, label=f"projected from k0={initial:.3f}")
    ax4.axhline(k_ss, color="black", linewidth=0.8, alpha=0.5)
    ax4.set_xlabel("Period")
    ax4.set_ylabel("Capital")
    ax4.set_title("Simulation Under the Projected Policy")
    ax4.legend()
    report.add_figure(
        "figures/simulated-paths.png",
        "Capital paths generated by the projected policy",
        fig4,
        description=(
            "A good approximation should not only satisfy equations at nodes. It should also "
            "produce sensible dynamics when iterated forward."
        ),
    )

    report.add_table(
        "tables/projection-accuracy.csv",
        "Projection accuracy by basis size",
        format_table(table),
        description="Errors are computed on a dense grid, not only at collocation nodes.",
    )

    report.add_results(
        f"With {N_BASIS_MAIN} Chebyshev terms, the maximum Euler error on the dense grid is "
        f"{np.max(main_euler_error):.2e}. The policy is stored with only eight coefficients, "
        "but it can be evaluated smoothly at any capital value inside the interval."
    )

    report.add_takeaway(
        "Projection replaces a large table of function values with a small set of coefficients. "
        "That can be powerful when the unknown object is smooth. The tradeoff is that accuracy "
        "has to be checked globally: a low residual at collocation nodes is not enough unless "
        "Euler errors and simulated behavior also look good."
    )

    report.add_references(
        [
            "Chang, M. ECON 609 lecture slides: Projection.",
            "[Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press.](https://mitpress.mit.edu/9780262100717/numerical-methods-in-economics/)",
            "[Miranda, M. J. and Fackler, P. L. (2002). *Applied Computational Economics and Finance*. MIT Press.](https://mitpress.mit.edu/9780262633093/applied-computational-economics-and-finance/)",
        ]
    )

    report.write("README.md")


if __name__ == "__main__":
    main()
