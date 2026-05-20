#!/usr/bin/env python3
"""Afriat's revealed-preference test for finite consumer-choice data.

Builds the revealed-preference relation, computes transitive closure, and checks
the Generalized Axiom of Revealed Preference.

References: Afriat (1967), Varian (1982) "The Nonparametric Approach to Demand Analysis"
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


TOL = 1e-10


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
                if expenditure_i + TOL >= cost_j_at_pi:
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
                if exp_j > cost_i_at_pj + TOL:
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


def plot_revealed_preference_graph(R, R_star, title, violations=None):
    """Plot the revealed preference relation as a directed graph."""
    T = R.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    violation_pairs = set(violations or [])

    for idx, (matrix, subtitle) in enumerate([
        (R, "Direct Revealed Preference"),
        (R_star, "Transitive Closure")
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
                    is_violation_pair = (i, j) in violation_pairs or (j, i) in violation_pairs
                    color = "#b3202a" if is_violation_pair else "steelblue"
                    lw = 2.0 if is_violation_pair else 1.0

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

        blue_patch = mpatches.Patch(color="steelblue", label="Revealed preference")
        handles = [blue_patch]
        if violation_pairs:
            red_patch = mpatches.Patch(color="#b3202a", label="GARP violation pair")
            handles.append(red_patch)
        ax.legend(handles=handles, fontsize=8, loc="lower right")

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_title(subtitle)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
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
    # Figures
    # =========================================================================
    setup_style()

    # --- Figure 1: Budget lines and bundles ---
    fig1 = plot_budget_lines_and_bundles(
        p_con, q_con,
        "Cobb-Douglas Sample: GARP Satisfied",
        consistent=True,
    )
    save_figure(fig1, "figures/budget-lines-consistent.png", dpi=150)

    fig1b = plot_budget_lines_and_bundles(
        p_inc, q_inc,
        "Corrupted Sample: GARP Violated",
        consistent=False,
    )
    save_figure(fig1b, "figures/budget-lines-inconsistent.png", dpi=150)

    # --- Figure 2: Revealed preference graph ---
    fig2 = plot_revealed_preference_graph(
        R_con, Rstar_con,
        "Cobb-Douglas Sample: Revealed Preference Graph",
        violations_con,
    )
    save_figure(fig2, "figures/rp-graph-consistent.png", dpi=150)

    fig2b = plot_revealed_preference_graph(
        R_inc, Rstar_inc,
        "Corrupted Sample: Revealed Preference Graph",
        violations_inc,
    )
    save_figure(fig2b, "figures/rp-graph-inconsistent.png", dpi=150)

    save_thumbnail("figures/budget-lines-consistent.png", "figures/thumb.png")
    print(f"Done: 4 figures")


if __name__ == "__main__":
    main()
