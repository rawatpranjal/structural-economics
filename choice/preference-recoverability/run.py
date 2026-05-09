#!/usr/bin/env python3
"""Preference bounds from finite revealed-preference data.

Constructs one concave monotone utility function that rationalizes
GARP-consistent budget choices and compares its implied contour with
the Cobb-Douglas data-generating benchmark.

Reference: Varian (1982), "The Nonparametric Approach to Demand Analysis."
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


TOL = 1e-10


def generate_cobb_douglas_data(
    alpha: float,
    n_obs: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate two-good choices from a Cobb-Douglas utility maximizer."""
    rng = np.random.default_rng(seed)
    prices = rng.uniform(0.5, 2.0, (n_obs, 2))
    income = rng.uniform(5.0, 15.0, n_obs)
    quantities = np.column_stack([
        alpha * income / prices[:, 0],
        (1.0 - alpha) * income / prices[:, 1],
    ])
    return prices, quantities, income


def cobb_douglas_utility(quantities: np.ndarray, alpha: float) -> np.ndarray:
    """Evaluate the Cobb-Douglas utility used to generate the synthetic data."""
    return quantities[:, 0] ** alpha * quantities[:, 1] ** (1.0 - alpha)


def direct_revealed_preference(
    prices: np.ndarray,
    quantities: np.ndarray,
) -> np.ndarray:
    """Return R[i,j]=1 when bundle i directly reveals preference over j."""
    n_obs = len(prices)
    relation = np.zeros((n_obs, n_obs), dtype=bool)
    expenditure = np.einsum("ij,ij->i", prices, quantities)
    for i in range(n_obs):
        for j in range(n_obs):
            if expenditure[i] + TOL >= prices[i] @ quantities[j]:
                relation[i, j] = True
    return relation


def transitive_closure(relation: np.ndarray) -> np.ndarray:
    """Compute the transitive closure of a finite revealed-preference relation."""
    closure = relation.copy()
    n_obs = relation.shape[0]
    for k in range(n_obs):
        for i in range(n_obs):
            for j in range(n_obs):
                if closure[i, k] and closure[k, j]:
                    closure[i, j] = True
    return closure


def garp_violations(
    prices: np.ndarray,
    quantities: np.ndarray,
    closure: np.ndarray,
) -> list[tuple[int, int]]:
    """List pairs that violate GARP."""
    violations = []
    expenditure = np.einsum("ij,ij->i", prices, quantities)
    n_obs = len(prices)
    for i in range(n_obs):
        for j in range(n_obs):
            if i == j:
                continue
            if closure[i, j] and expenditure[j] > prices[j] @ quantities[i] + TOL:
                violations.append((i, j))
    return violations


def afriat_numbers(
    prices: np.ndarray,
    quantities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute Afriat numbers under the expenditure normalization lambda_t=1/m_t.

    The normalization fixes the units of the utility index. For this generated
    Cobb-Douglas sample the resulting difference constraints are feasible.
    """
    n_obs = len(prices)
    expenditure = np.einsum("ij,ij->i", prices, quantities)
    lambdas = 1.0 / expenditure

    relation = direct_revealed_preference(prices, quantities)
    closure = transitive_closure(relation)

    a_ub = []
    b_ub = []
    for i in range(n_obs):
        for j in range(n_obs):
            row = np.zeros(n_obs)
            row[i] = 1.0
            row[j] -= 1.0
            a_ub.append(row)
            b_ub.append(lambdas[j] * prices[j] @ (quantities[i] - quantities[j]))

    # Utility is ordinal. Setting the sample average to one fixes the affine
    # level of the recovered index without using the Cobb-Douglas ground truth.
    a_eq = np.ones((1, n_obs))
    b_eq = np.array([float(n_obs)])

    result = linprog(
        c=np.zeros(n_obs),
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, None) for _ in range(n_obs)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Afriat construction failed: {result.message}")

    utilities = result.x
    residual = np.empty((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            residual[i, j] = (
                utilities[i]
                - utilities[j]
                - lambdas[j] * prices[j] @ (quantities[i] - quantities[j])
            )
    max_violation = float(np.max(residual))
    return utilities, lambdas, relation, closure, max_violation


def afriat_utility(
    points: np.ndarray,
    prices: np.ndarray,
    quantities: np.ndarray,
    utilities: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Evaluate the recovered Afriat utility at arbitrary two-good bundles."""
    values = []
    for point in points:
        supports = utilities + lambdas * np.einsum("ij,ij->i", prices, point - quantities)
        values.append(float(np.min(supports)))
    return np.asarray(values)


def afriat_frontier(
    grid_x1: np.ndarray,
    target_obs: int,
    prices: np.ndarray,
    quantities: np.ndarray,
    utilities: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Lower boundary of the recovered upper-contour set through target_obs."""
    target_utility = utilities[target_obs]
    frontier = np.empty_like(grid_x1)
    for idx, x1_value in enumerate(grid_x1):
        x2_bounds = (
            quantities[:, 1]
            + (
                target_utility
                - utilities
                - lambdas * prices[:, 0] * (x1_value - quantities[:, 0])
            )
            / (lambdas * prices[:, 1])
        )
        frontier[idx] = np.max(x2_bounds)
    return frontier


def main() -> None:
    # =========================================================================
    # Generate a rational two-good sample
    # =========================================================================
    alpha_true = 0.60
    n_obs = 18
    prices, quantities, income = generate_cobb_douglas_data(alpha_true, n_obs)
    expenditure = np.einsum("ij,ij->i", prices, quantities)
    true_utility = cobb_douglas_utility(quantities, alpha_true)
    true_utility_scaled = true_utility / true_utility.mean()

    # =========================================================================
    # Recover one Afriat utility index
    # =========================================================================
    utilities, lambdas, relation, closure, max_afriat_residual = afriat_numbers(
        prices,
        quantities,
    )
    violations = garp_violations(prices, quantities, closure)

    print(f"GARP violations: {len(violations)}")
    print(f"Max Afriat inequality residual: {max_afriat_residual:.2e}")

    fitted_utility = afriat_utility(quantities, prices, quantities, utilities, lambdas)
    fit_error = fitted_utility - utilities
    utility_corr = float(np.corrcoef(utilities, true_utility_scaled)[0, 1])

    # =========================================================================
    # Recover an indifference frontier through a target observed bundle
    # =========================================================================
    target_obs = int(np.argsort(true_utility)[n_obs // 2])
    grid_x1 = np.linspace(0.4, quantities[:, 0].max() * 1.35, 220)
    recovered_x2 = afriat_frontier(
        grid_x1,
        target_obs,
        prices,
        quantities,
        utilities,
        lambdas,
    )

    target_true_utility = true_utility[target_obs]
    true_x2 = (target_true_utility / grid_x1 ** alpha_true) ** (1.0 / (1.0 - alpha_true))
    plot_y_max = quantities[:, 1].max() * 2.2
    valid_recovered = (recovered_x2 > 0.0) & (recovered_x2 < plot_y_max)
    valid_true = (true_x2 > 0.0) & (true_x2 < plot_y_max)
    overlap = valid_recovered & valid_true
    median_contour_ratio = float(np.median(recovered_x2[overlap] / true_x2[overlap]))
    max_contour_gap = float(np.max(np.abs(recovered_x2[overlap] - true_x2[overlap])))

    # =========================================================================
    # Generate report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Recovering Preference Bounds from Budget Choices",
        "Afriat numbers turn finite budget data into a rationalizing utility contour.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Suppose we observe a consumer choosing bundles from several two-good budgets. "
        "Prices rotate the budget line, and income shifts it. The analyst sees prices "
        "and chosen bundles, not the utility function.\n\n"
        "The object is the preference ordering implied by these finite choices. "
        "A GARP-consistent sample can be rationalized by many utility functions. "
        "We recover one upper-contour boundary through a chosen bundle.\n\n"
        "The computation uses Afriat inequalities. A linear program finds utility "
        "scores and supporting slopes. The lower envelope of those supports draws "
        "the contour."
    )

    report.add_equations(
        r"""
There are two goods and $T$ budget-choice observations. Observation $t$ has
prices $p_t=(p_{1t},p_{2t})\in\mathbb{R}_{++}^{2}$ and chosen bundle
$x_t=(x_{1t},x_{2t})\in\mathbb{R}_{+}^{2}$. Expenditure is $m_t=p_t\cdot x_t$.

Afriat recovery asks for ordinal utility scores $u_t$ and positive supporting
slopes $\lambda_t$ such that

$$
u_i-u_j \leq \lambda_j p_j\cdot(x_i-x_j)
\qquad \text{for all } i,j=1,\ldots,T .
$$

When these inequalities are feasible, one rationalizing utility index is

$$
\widehat U(y)=\min_{j=1,\ldots,T}
\left[u_j+\lambda_j p_j\cdot(y-x_j)\right].
$$

This utility is the lower envelope of affine supporting functions. It is
concave, monotone when prices and $\lambda_j$ are positive, and satisfies
$\widehat U(x_t)=u_t$ at the observed choices.

For a target observation $k$, the recovered upper-contour set is

$$
\widehat U(y)\geq u_k .
$$

Writing $y=(y_1,y_2)$, its lower boundary can be computed pointwise:

$$
y_2(y_1)=
\max_{j=1,\ldots,T}
\left[
x_{2j}
+
\frac{
u_k-u_j-\lambda_j p_{1j}(y_1-x_{1j})
}{
\lambda_j p_{2j}
}
\right].
$$

The data-generating benchmark, used only for comparison, is

$$
U^0(x)=x_1^{\alpha}x_2^{1-\alpha},\qquad \alpha=0.60 .
$$
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role in the exercise |\n"
        f"|---|---:|---|\n"
        f"| Observations $T$ | {n_obs} | Price-bundle pairs observed by the analyst |\n"
        f"| Goods | 2 | Makes the recovered contour visible |\n"
        f"| True $\\alpha$ | {alpha_true:.2f} | Cobb-Douglas benchmark, hidden from recovery |\n"
        f"| Income range | [{income.min():.2f}, {income.max():.2f}] | Moves budget lines outward or inward |\n"
        f"| Price range | [{prices.min():.2f}, {prices.max():.2f}] | Rotates the observed budgets |\n"
        f"| GARP violations | {len(violations)} | Screen before utility recovery |\n"
        f"| Max Afriat residual | {max_afriat_residual:.2e} | Feasibility error from the inequalities |\n"
        f"| Target observation | {target_obs + 1} | Bundle whose contour is drawn |"
    )

    report.add_solution_method(
        "Afriat recovery uses a linear program because the unknown utility scores enter "
        "only through pairwise inequalities. The GARP screen protects the economic "
        "interpretation: if a strict revealed-preference cycle exists, no monotone "
        "concave utility can rationalize all choices. Once the screen passes, the "
        "linear program chooses one ordinal utility score for each observed bundle. "
        "The lower envelope then extends those scores to nearby bundles.\n\n"
        "```text\n"
        "Algorithm: Afriat contour recovery\n"
        "Input: budgets (p_t, x_t) for t=1,...,T and target observation k\n"
        "Output: recovered utility index U_hat and contour through x_k\n"
        "\n"
        "1. Mark i R j when bundle x_j was affordable under budget i.\n"
        "2. Close R transitively and check for a strict revealed-preference reversal.\n"
        "3. Set lambda_t = 1 / (p_t . x_t) to normalize supporting slopes.\n"
        "4. Solve for ordinal scores u_t subject to\n"
        "       u_i - u_j <= lambda_j p_j . (x_i - x_j) for every pair (i,j),\n"
        "       average_t u_t = 1, and u_t >= 0.\n"
        "5. Define U_hat(y) = min_j [u_j + lambda_j p_j . (y - x_j)].\n"
        "6. For a grid of y_1 values, compute the smallest y_2 that satisfies\n"
        "       U_hat((y_1,y_2)) >= u_k.\n"
        "```"
    )

    # --- Figure 1: observed budgets and choices ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, n_obs))
    for t in range(n_obs):
        x1_line = np.array([0.0, income[t] / prices[t, 0]])
        x2_line = np.array([income[t] / prices[t, 1], 0.0])
        ax1.plot(x1_line, x2_line, color=colors[t], alpha=0.28, linewidth=1.0)
        ax1.scatter(quantities[t, 0], quantities[t, 1], color=colors[t], s=38, zorder=4)
    ax1.scatter(
        quantities[target_obs, 0],
        quantities[target_obs, 1],
        color="#b3202a",
        edgecolor="white",
        linewidth=0.8,
        marker="*",
        s=180,
        zorder=6,
        label=f"Target observation {target_obs + 1}",
    )
    ax1.set_xlabel("Good 1 quantity")
    ax1.set_ylabel("Good 2 quantity")
    ax1.set_title("Observed Budgets and Choices")
    ax1.set_xlim(0, max(income / prices[:, 0]) * 1.08)
    ax1.set_ylim(0, max(income / prices[:, 1]) * 1.08)
    ax1.legend(frameon=False, loc="upper right")

    report.add_results(
        "Each line is an observed budget set. Each dot is the chosen bundle. "
        "Prices rotate budget lines, and income shifts them. The starred bundle is "
        "the target contour. The recovery ignores the Cobb-Douglas source."
    )
    report.add_figure(
        "figures/budget-lines.png",
        "Observed budget lines and chosen bundles with the target observation starred",
        fig1,
    )

    # --- Figure 2: recovered contour vs true benchmark ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(
        grid_x1[valid_recovered],
        recovered_x2[valid_recovered],
        color="steelblue",
        linewidth=2.2,
        label="Afriat rationalizer",
    )
    ax2.plot(
        grid_x1[valid_true],
        true_x2[valid_true],
        color="#b3202a",
        linestyle="--",
        linewidth=1.8,
        label="True Cobb-Douglas",
    )
    ax2.scatter(
        quantities[:, 0],
        quantities[:, 1],
        color="gray",
        alpha=0.35,
        s=24,
        label="Observed choices",
    )
    ax2.scatter(
        quantities[target_obs, 0],
        quantities[target_obs, 1],
        color="#b3202a",
        edgecolor="white",
        linewidth=0.8,
        marker="*",
        s=180,
        zorder=6,
    )
    ax2.set_xlabel("Good 1 quantity")
    ax2.set_ylabel("Good 2 quantity")
    ax2.set_title("Recovered Upper-Contour Boundary")
    ax2.set_ylim(0, plot_y_max)
    ax2.legend(frameon=False)

    report.add_results(
        "The blue curve is not a Cobb-Douglas estimate. It is one concave contour "
        "through the target choice. It also rationalizes every observed bundle. "
        "The dashed curve is the true simulation contour. It is shown only as a "
        f"benchmark. On the plotted overlap, the median recovered-to-true $x_2$ "
        f"ratio is **{median_contour_ratio:.2f}**. The largest absolute contour "
        f"gap is **{max_contour_gap:.2f}** units of good 2."
    )
    report.add_figure(
        "figures/indifference-curve.png",
        "Recovered Afriat contour and the held-out Cobb-Douglas contour",
        fig2,
    )

    # --- Figure 3: Afriat numbers and true utility diagnostic ---
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    ax3a.scatter(true_utility_scaled, utilities, color="steelblue", s=52)
    lower = min(true_utility_scaled.min(), utilities.min()) * 0.95
    upper = max(true_utility_scaled.max(), utilities.max()) * 1.05
    ax3a.plot([lower, upper], [lower, upper], color="gray", linestyle=":", linewidth=1.0)
    ax3a.set_xlim(lower, upper)
    ax3a.set_ylim(lower, upper)
    ax3a.set_xlabel("True $U^0(x_t)$, normalized")
    ax3a.set_ylabel("Recovered $u_t$")
    ax3a.set_title("Ordinal Utility Levels")

    ax3b.scatter(expenditure, lambdas, color="#b65f00", s=52)
    expenditure_grid = np.linspace(expenditure.min(), expenditure.max(), 100)
    ax3b.plot(expenditure_grid, 1.0 / expenditure_grid, color="gray", linestyle=":", linewidth=1.0)
    ax3b.set_xlabel("Expenditure $m_t$")
    ax3b.set_ylabel("$\\lambda_t$")
    ax3b.set_title("Expenditure Normalization")
    fig3.tight_layout()

    report.add_results(
        "Afriat numbers are a finite-data certificate. The scores $u_t$ rank observed "
        "bundles while respecting budget comparisons. The $\\lambda_t$ values give "
        "supporting slopes in utility units. The simulated sample lets us compare the "
        f"recovered ordering with true utility. The correlation is **{utility_corr:.3f}**."
    )
    report.add_figure(
        "figures/afriat-numbers.png",
        "Afriat utility levels and marginal utility normalizations",
        fig3,
    )

    # --- Table ---
    df = pd.DataFrame({
        "Observation": np.arange(1, n_obs + 1),
        "Expenditure": [f"{value:.2f}" for value in expenditure],
        "u_t": [f"{value:.4f}" for value in utilities],
        "lambda_t": [f"{value:.4f}" for value in lambdas],
        "True U normalized": [f"{value:.4f}" for value in true_utility_scaled],
        "Fit error": [f"{value:.2e}" for value in fit_error],
    })
    report.add_table(
        "tables/afriat-numbers.csv",
        "Afriat numbers and fit diagnostics",
        df,
        description=(
            "The last column checks that recovered utility equals $u_t$ at each "
            "observed bundle. The true-utility column is a simulation diagnostic, "
            "not an input."
        ),
    )

    report.add_takeaway(
        "Finite choices can do more than test rationalizability. They still do not "
        "identify one full utility function. Afriat numbers recover one utility "
        "index and one local contour. Those objects respect every observed budget "
        "comparison. Budgets pin down preferences only where prices create support."
    )

    report.add_references([
        "Afriat, S. N. (1967). The Construction of Utility Functions from Expenditure Data. "
        "*International Economic Review*, 8(1), 67-77.",
        "Varian, H. R. (1982). The Nonparametric Approach to Demand Analysis. "
        "*Econometrica*, 50(4), 945-973.",
        "Varian, H. R. (2006). Revealed Preference. In M. Szenberg et al. (Eds.), "
        "*Samuelsonian Economics and the Twenty-First Century*. Oxford University Press.",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
