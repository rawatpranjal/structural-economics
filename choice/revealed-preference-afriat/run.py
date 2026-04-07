#!/usr/bin/env python3
"""Revealed Preference and Afriat Inequalities: Testing Consumer Rationality.

Checks whether observed consumption data can be rationalized by a well-behaved
utility function using the Generalized Axiom of Revealed Preference (GARP)
and Afriat's theorem. Implements Warshall's algorithm for transitive closure
of the revealed preference relation.

Reference: Afriat (1967), Varian (1982) "The Nonparametric Approach to Demand Analysis"
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


# =============================================================================
# Core algorithms
# =============================================================================

def direct_revealed_preference(prices, quantities):
    """Build the direct revealed preference relation matrix.

    R[i,j] = 1 if bundle i is directly revealed preferred to bundle j,
    i.e., p_i . x_i >= p_i . x_j  (bundle j was affordable when i was chosen).
    """
    T = len(prices)
    R = np.zeros((T, T), dtype=int)
    for i in range(T):
        expenditure_i = np.dot(prices[i], quantities[i])
        for j in range(T):
            if i != j:
                cost_j_at_pi = np.dot(prices[i], quantities[j])
                if expenditure_i >= cost_j_at_pi:
                    R[i, j] = 1
    return R


def warshall_transitive_closure(R):
    """Warshall's algorithm: compute the transitive closure of relation R.

    Returns the indirect revealed preference relation. If R*[i,j] = 1,
    then i is (directly or indirectly) revealed preferred to j.
    """
    T = R.shape[0]
    R_star = R.copy()
    for k in range(T):
        for i in range(T):
            for j in range(T):
                if R_star[i, k] and R_star[k, j]:
                    R_star[i, j] = 1
    return R_star


def check_garp(prices, quantities):
    """Check the Generalized Axiom of Revealed Preference (GARP).

    GARP is violated if there exist i, j such that:
      - i is (directly or indirectly) revealed preferred to j  (R*[i,j] = 1)
      - AND j is directly revealed strictly preferred to i
        (p_j . x_j > p_j . x_i, so i was strictly inside j's budget)

    Equivalently (standard form): GARP fails if i R* j and p_j . x_j > p_j . x_i.

    Returns (satisfies_garp, violations_list).
    """
    T = len(prices)
    R = direct_revealed_preference(prices, quantities)
    R_star = warshall_transitive_closure(R)

    violations = []
    for i in range(T):
        for j in range(T):
            if i != j and R_star[i, j]:
                # Check if j is directly revealed strictly preferred to i
                exp_j = np.dot(prices[j], quantities[j])
                cost_i_at_pj = np.dot(prices[j], quantities[i])
                if exp_j > cost_i_at_pj:
                    violations.append((i, j))

    return len(violations) == 0, violations, R, R_star


def generate_consistent_data(T, n_goods, rng):
    """Generate consumption data consistent with GARP.

    Strategy: generate data from a Cobb-Douglas utility maximizer.
    u(x) = prod(x_k^alpha_k), budget: p.x = income.
    Optimal demand: x_k = (alpha_k / p_k) * income.
    """
    # Random utility weights (Cobb-Douglas exponents summing to 1)
    alpha = rng.dirichlet(np.ones(n_goods))

    prices = np.zeros((T, n_goods))
    quantities = np.zeros((T, n_goods))

    for t in range(T):
        p = rng.uniform(0.5, 3.0, size=n_goods)
        income = rng.uniform(5.0, 15.0)
        # Cobb-Douglas demand
        x = (alpha / p) * income
        prices[t] = p
        quantities[t] = x

    return prices, quantities, alpha


def generate_inconsistent_data(T, n_goods, rng):
    """Generate consumption data that violates GARP.

    Strategy: start with consistent data and perturb a pair of observations
    to create a cycle in the revealed preference relation.
    """
    for attempt in range(200):
        prices, quantities, alpha = generate_consistent_data(T, n_goods, rng)

        # Swap two quantity bundles to break consistency
        i, j = rng.choice(T, size=2, replace=False)
        quantities_bad = quantities.copy()
        quantities_bad[i] = quantities[j]
        quantities_bad[j] = quantities[i]

        satisfies, violations, _, _ = check_garp(prices, quantities_bad)
        if not satisfies:
            return prices, quantities_bad, violations

    # Fallback: construct a known violation manually
    prices = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ] + [rng.uniform(0.5, 3.0, size=n_goods).tolist() for _ in range(T - 3)])
    quantities = np.array([
        [4.0, 1.0, 2.0],
        [1.0, 4.0, 2.0],
        [2.0, 2.0, 3.0],
    ] + [(rng.dirichlet(np.ones(n_goods)) * rng.uniform(5, 15)).tolist()
         for _ in range(T - 3)])
    _, violations, _, _ = check_garp(prices, quantities)
    return prices, quantities, violations


def power_of_garp_test(n_trials, T_values, n_goods, rng):
    """Compute the fraction of random datasets that violate GARP.

    For purely random (price, quantity) pairs, GARP violations become
    more likely as T increases -- this measures the 'power' of the test.
    """
    violation_rates = []
    for T in T_values:
        n_violations = 0
        for _ in range(n_trials):
            prices = rng.uniform(0.5, 3.0, size=(T, n_goods))
            quantities = rng.uniform(0.5, 5.0, size=(T, n_goods))
            satisfies, _, _, _ = check_garp(prices, quantities)
            if not satisfies:
                n_violations += 1
        violation_rates.append(n_violations / n_trials)
    return violation_rates


# =============================================================================
# Visualization helpers
# =============================================================================

def plot_budget_lines_and_bundles(prices, quantities, title, consistent=True):
    """Plot budget lines and chosen bundles (2D projection onto goods 0 and 1)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    T = len(prices)
    colors = plt.cm.tab10(np.linspace(0, 1, T))

    x_max = 0
    y_max = 0

    for t in range(T):
        p0, p1 = prices[t, 0], prices[t, 1]
        income = np.dot(prices[t], quantities[t])

        # Budget line in 2D: p0*x0 + p1*x1 = income (holding other goods fixed)
        # Project: fix other goods at observed levels, plot the residual budget
        residual_income = income - np.dot(prices[t, 2:], quantities[t, 2:])
        if residual_income <= 0:
            continue

        x0_intercept = residual_income / p0
        x1_intercept = residual_income / p1

        ax.plot([0, x0_intercept], [x1_intercept, 0],
                color=colors[t], linewidth=1.5, alpha=0.6,
                label=f"$t={t+1}$" if t < 8 else None)

        x_max = max(x_max, x0_intercept)
        y_max = max(y_max, x1_intercept)

    # Plot chosen bundles
    marker = "o" if consistent else "X"
    for t in range(T):
        kwargs = dict(color=colors[t], s=100, zorder=5, marker=marker)
        if consistent:
            kwargs.update(edgecolors="black", linewidths=0.5)
        ax.scatter(quantities[t, 0], quantities[t, 1], **kwargs)

    ax.set_xlabel("Good 1 quantity")
    ax.set_ylabel("Good 2 quantity")
    ax.set_title(title)
    ax.set_xlim(0, x_max * 1.1)
    ax.set_ylim(0, y_max * 1.1)
    if T <= 10:
        ax.legend(fontsize=8, loc="upper right")
    return fig


def plot_revealed_preference_graph(R, R_star, title):
    """Plot the revealed preference relation as a directed graph."""
    T = R.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (matrix, subtitle) in enumerate([
        (R, "Direct Revealed Preference"),
        (R_star, "Transitive Closure (Warshall)")
    ]):
        ax = axes[idx]

        # Arrange nodes in a circle
        angles = np.linspace(0, 2 * np.pi, T, endpoint=False)
        radius = 2.0
        node_x = radius * np.cos(angles)
        node_y = radius * np.sin(angles)

        # Draw edges
        for i in range(T):
            for j in range(T):
                if i != j and matrix[i, j]:
                    # Check for mutual relation (cycle indicator)
                    mutual = matrix[j, i] == 1
                    color = "red" if mutual else "steelblue"
                    lw = 2.0 if mutual else 1.0

                    dx = node_x[j] - node_x[i]
                    dy = node_y[j] - node_y[i]
                    dist = np.sqrt(dx**2 + dy**2)

                    # Shorten arrow to not overlap node circles
                    shrink = 0.25 / dist if dist > 0 else 0
                    ax.annotate(
                        "", xy=(node_x[j] - dx * shrink, node_y[j] - dy * shrink),
                        xytext=(node_x[i] + dx * shrink, node_y[i] + dy * shrink),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=lw, connectionstyle="arc3,rad=0.15"),
                    )

        # Draw nodes
        for i in range(T):
            circle = plt.Circle((node_x[i], node_y[i]), 0.22,
                                color="white", ec="black", linewidth=1.5, zorder=5)
            ax.add_patch(circle)
            ax.text(node_x[i], node_y[i], str(i + 1),
                    ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)

        # Legend
        blue_patch = mpatches.Patch(color="steelblue", label="One-way preference")
        red_patch = mpatches.Patch(color="red", label="Mutual (potential cycle)")
        ax.legend(handles=[blue_patch, red_patch], fontsize=8, loc="lower right")

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(subtitle)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_garp_power(T_values, violation_rates):
    """Plot the power of the GARP test: violation rate vs number of observations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_values, violation_rates, "bo-", linewidth=2, markersize=6)
    ax.fill_between(T_values, 0, violation_rates, alpha=0.15, color="blue")
    ax.set_xlabel("Number of observations $T$")
    ax.set_ylabel("Fraction of random datasets violating GARP")
    ax.set_title("Power of the GARP Test (Random Data)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    rng = np.random.default_rng(42)
    T = 10        # Number of observations
    n_goods = 3   # Number of goods

    # =========================================================================
    # Example 1: Consistent data (satisfies GARP)
    # =========================================================================
    print("=" * 60)
    print("Example 1: Consistent data (Cobb-Douglas utility maximizer)")
    print("=" * 60)

    p_con, q_con, alpha = generate_consistent_data(T, n_goods, rng)
    satisfies_con, violations_con, R_con, Rstar_con = check_garp(p_con, q_con)

    print(f"  Observations: {T}, Goods: {n_goods}")
    print(f"  Cobb-Douglas weights: [{', '.join(f'{a:.3f}' for a in alpha)}]")
    print(f"  GARP satisfied: {satisfies_con}")
    print(f"  Violations: {len(violations_con)}")
    print()

    # =========================================================================
    # Example 2: Inconsistent data (violates GARP)
    # =========================================================================
    print("=" * 60)
    print("Example 2: Inconsistent data (GARP violations)")
    print("=" * 60)

    p_inc, q_inc, violations_inc_raw = generate_inconsistent_data(T, n_goods, rng)
    satisfies_inc, violations_inc, R_inc, Rstar_inc = check_garp(p_inc, q_inc)

    print(f"  Observations: {T}, Goods: {n_goods}")
    print(f"  GARP satisfied: {satisfies_inc}")
    print(f"  Number of violations: {len(violations_inc)}")
    if violations_inc:
        i0, j0 = violations_inc[0]
        print(f"  First violation: observation {i0+1} R* {j0+1}, "
              f"but p_{j0+1}.x_{j0+1} = {np.dot(p_inc[j0], q_inc[j0]):.3f} > "
              f"p_{j0+1}.x_{i0+1} = {np.dot(p_inc[j0], q_inc[i0]):.3f}")
    print()

    # =========================================================================
    # Power of the GARP test
    # =========================================================================
    print("=" * 60)
    print("Power of GARP test (random data)")
    print("=" * 60)

    n_trials = 500
    T_values = [2, 3, 5, 8, 10, 15, 20, 30, 50]
    violation_rates = power_of_garp_test(n_trials, T_values, n_goods, rng)

    for Tv, vr in zip(T_values, violation_rates):
        print(f"  T = {Tv:3d}: {vr:.1%} violation rate")
    print()

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Revealed Preference and Afriat Inequalities",
        "Testing whether observed consumption data is consistent with utility maximization.",
    )

    report.add_overview(
        "Revealed preference theory asks a fundamental empirical question: can observed "
        "consumer choices be rationalized by *any* well-behaved utility function? Afriat's "
        "theorem (1967) provides a complete answer — observed data $(p_t, x_t)_{t=1}^T$ "
        "is consistent with utility maximization if and only if it satisfies the Generalized "
        "Axiom of Revealed Preference (GARP).\n\n"
        "This is the empirical foundation of consumer theory. Unlike parametric demand "
        "estimation, the revealed preference approach is entirely nonparametric — it does "
        "not assume a functional form for utility. If the data passes GARP, there EXISTS "
        "a nonsatiated, continuous, concave, monotone utility function that rationalizes it."
    )

    report.add_equations(
        r"""
**Direct Revealed Preference:** Observation $i$ is *directly revealed preferred* to $j$ if:
$$p_i \cdot x_i \geq p_i \cdot x_j$$
i.e., bundle $x_j$ was affordable when $x_i$ was chosen.

**GARP:** For all $i, j$: if $x_i \; R^* \; x_j$ (revealed preferred, possibly indirectly), then:
$$p_j \cdot x_j \leq p_j \cdot x_i$$
i.e., $x_i$ must not lie strictly inside $j$'s budget set.

**Afriat Inequalities:** Data is rationalizable $\iff$ there exist scalars $u_i, \lambda_i > 0$ such that:
$$u_i - u_j \leq \lambda_j \, p_j \cdot (x_i - x_j) \quad \forall \, i, j$$

**Afriat's Theorem (1967):** The following are equivalent:
1. Data satisfies GARP
2. Afriat inequalities have a solution
3. Data can be rationalized by a nonsatiated, continuous, concave utility function
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $T$ | {T} | Number of observations |\n"
        f"| Goods | {n_goods} | Number of goods per bundle |\n"
        f"| Example 1 | Cobb-Douglas | Data generated from utility maximizer |\n"
        f"| Example 2 | Perturbed | Swapped bundles to induce GARP violation |\n"
        f"| Power test | {n_trials} trials | Random data, $T \\in \\{{{', '.join(str(t) for t in T_values)}\\}}$ |"
    )

    report.add_solution_method(
        "**Step 1 — Direct Revealed Preference:** For each pair $(i, j)$, check if "
        "$p_i \\cdot x_i \\geq p_i \\cdot x_j$. This builds the direct preference matrix $R$.\n\n"
        "**Step 2 — Transitive Closure (Warshall's Algorithm):** Compute $R^*$, the "
        "transitive closure of $R$. If $i \\; R \\; k$ and $k \\; R \\; j$, then $i \\; R^* \\; j$. "
        "This runs in $O(T^3)$ time.\n\n"
        "**Step 3 — GARP Check:** For all pairs where $i \\; R^* \\; j$, verify that "
        "$p_j \\cdot x_j \\leq p_j \\cdot x_i$. Any violation means the data cannot be "
        "rationalized by utility maximization.\n\n"
        f"**Example 1** (consistent): GARP satisfied = **{satisfies_con}**, "
        f"violations = {len(violations_con)}.\n\n"
        f"**Example 2** (inconsistent): GARP satisfied = **{satisfies_inc}**, "
        f"violations = **{len(violations_inc)}**."
    )

    # --- Figure 1: Budget lines and bundles ---
    fig1 = plot_budget_lines_and_bundles(
        p_con, q_con,
        "Example 1: Consistent Data (GARP Satisfied)",
        consistent=True,
    )
    report.add_figure(
        "figures/budget-lines-consistent.png",
        "Budget lines and chosen bundles (2D projection) for consistent data generated "
        "from a Cobb-Douglas utility maximizer. All choices lie on their respective budget lines.",
        fig1,
    )

    fig1b = plot_budget_lines_and_bundles(
        p_inc, q_inc,
        "Example 2: Inconsistent Data (GARP Violated)",
        consistent=False,
    )
    report.add_figure(
        "figures/budget-lines-inconsistent.png",
        "Budget lines and chosen bundles for inconsistent data. Swapped bundles create "
        "revealed preference cycles that cannot be rationalized by any utility function.",
        fig1b,
    )

    # --- Figure 2: Revealed preference graph ---
    fig2 = plot_revealed_preference_graph(
        R_con, Rstar_con,
        "Example 1: Consistent Data — Revealed Preference Graph",
    )
    report.add_figure(
        "figures/rp-graph-consistent.png",
        "Directed graph of the revealed preference relation for consistent data. "
        "Red edges indicate mutual relations. No GARP-violating cycles exist.",
        fig2,
    )

    fig2b = plot_revealed_preference_graph(
        R_inc, Rstar_inc,
        "Example 2: Inconsistent Data — Revealed Preference Graph",
    )
    report.add_figure(
        "figures/rp-graph-inconsistent.png",
        "Directed graph for inconsistent data. Mutual red edges in the transitive closure "
        "reveal GARP-violating cycles — these observations cannot be rationalized.",
        fig2b,
    )

    # --- Figure 3: Power of the GARP test ---
    fig3 = plot_garp_power(T_values, violation_rates)
    report.add_figure(
        "figures/garp-power.png",
        "Power of the GARP test: fraction of random datasets that violate GARP as a "
        "function of the number of observations T. With more data, random choices are "
        "increasingly likely to produce violations — GARP has real empirical bite.",
        fig3,
    )

    # --- Table: Revealed preference matrix (consistent example) ---
    labels = [f"Obs {i+1}" for i in range(T)]
    rp_data = {"": labels}
    for j in range(T):
        col = []
        for i in range(T):
            if i == j:
                col.append("--")
            elif Rstar_con[i, j]:
                col.append("R*" if not R_con[i, j] else "R")
            else:
                col.append("")
        rp_data[f"Obs {j+1}"] = col

    df_rp = pd.DataFrame(rp_data)
    report.add_table(
        "tables/revealed-preference-matrix.csv",
        "Pairwise Revealed Preference Relation (Example 1: Consistent). "
        "R = directly revealed preferred, R* = indirectly (via transitive closure)",
        df_rp,
    )

    report.add_takeaway(
        "Afriat's theorem provides the deepest link between observable behavior and "
        "economic theory. It says that the neoclassical model of consumer choice — "
        "utility maximization subject to a budget constraint — is *testable* with "
        "finite data, and the test is constructive.\n\n"
        "**Key insights:**\n"
        "- **Nonparametric power:** GARP does not assume Cobb-Douglas, CES, or any "
        "specific functional form. If the data passes, *some* well-behaved utility "
        "function rationalizes it. If it fails, *no* such function exists.\n"
        "- **Empirical bite increases with data:** As the number of observations $T$ "
        "grows, random data is increasingly likely to violate GARP. This means GARP "
        "is not vacuous — it makes sharp, falsifiable predictions.\n"
        "- **Constructive theorem:** When GARP holds, Afriat's proof constructs an "
        "explicit utility function (piecewise linear) via the Afriat numbers $u_i, \\lambda_i$.\n"
        "- **Foundation for welfare analysis:** If choices are rationalizable, we can "
        "perform welfare comparisons, compute equivalent/compensating variation, and "
        "do counterfactual policy analysis — all without specifying a parametric model."
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
