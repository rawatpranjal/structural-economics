#!/usr/bin/env python3
"""Afriat's revealed-preference test for finite consumer-choice data.

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
from lib.plotting import setup_style
from lib.output import ModelReport


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


def plot_garp_power(T_values, random_violation_rates, rational_violation_rates):
    """Plot the power of the GARP test: violation rate vs number of observations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_values, random_violation_rates, "o-", linewidth=2, markersize=6,
            color="steelblue", label="Random prices and quantities")
    ax.plot(T_values, rational_violation_rates, "--", linewidth=2,
            color="#b3202a", label="Cobb-Douglas benchmark")
    ax.fill_between(T_values, 0, random_violation_rates, alpha=0.12, color="steelblue")
    ax.set_xlabel("Number of observations $T$")
    ax.set_ylabel("Fraction of datasets violating GARP")
    ax.set_title("GARP Rejects Random Choice, Not Rational Choice")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
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
    rational_violation_rates = [0.0 for _ in T_values]

    for Tv, vr in zip(T_values, violation_rates):
        print(f"  T = {Tv:3d}: {vr:.1%} violation rate")
    print()

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Afriat's Revealed-Preference Test",
        "Testing finite choice data for consistency with utility maximization.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Suppose we observe a finite panel of choices: at prices $p_t$, the "
        "consumer chooses bundle $x_t$. Without estimating a Cobb-Douglas, CES, or "
        "logit demand system, what does utility maximization require of those "
        "choices?\n\n"
        "The restriction is about revealed tradeoffs. If bundle $x_j$ was affordable "
        "when $x_i$ was chosen, then the data reveal $x_i$ to be at least as good as "
        "$x_j$. Chains of those comparisons cannot come back and make an earlier "
        "choice strictly cheaper at a later budget. That no-cycle condition is GARP. "
        "Afriat's theorem says this finite condition is exactly equivalent to the "
        "existence of a locally nonsatiated, monotone, concave utility function that "
        "rationalizes the observations.\n\n"
        "The tutorial uses three checks: a known rational Cobb-Douglas sample, a "
        "small corrupted sample that breaks rationalizability, and a Bronars-style "
        "power exercise showing that random choices usually fail once $T$ is large."
    )

    report.add_equations(
        r"""
Let the data be $\mathcal{D}=\{(p_t,x_t)\}_{t=1}^T$, where $p_t\in\mathbb{R}_{++}^L$ and $x_t\in\mathbb{R}_{+}^L$. Expenditure at observation $t$ is $m_t=p_t\cdot x_t$.

Observation $i$ is directly revealed weakly preferred to observation $j$, written $iRj$, when
$$
p_i\cdot x_i \geq p_i\cdot x_j .
$$
The bundle $x_j$ was affordable when $x_i$ was chosen.

Let $R^{*}$ denote the transitive closure of $R$. GARP requires that no pair $(i,j)$ satisfies both
$$
iR^{*}j
\quad\text{and}\quad
p_j\cdot x_j > p_j\cdot x_i .
$$
The first statement says $x_i$ is revealed at least as good as $x_j$ through a chain of budgets. The second says that, at budget $j$, $x_i$ was strictly cheaper than the bundle actually chosen.

Afriat's inequalities give the constructive equivalent condition. The data are rationalizable if and only if there exist numbers $u_t$ and $\lambda_t>0$ such that
$$
u_i-u_j \leq \lambda_j p_j\cdot(x_i-x_j)
\quad \forall i,j .
$$
When those inequalities are feasible, a rationalizing utility can be written as
$$
\widehat U(x)=\min_j\{u_j+\lambda_j p_j\cdot(x-x_j)\}.
$$
This run checks GARP directly; the neighboring [preference-recoverability](../preference-recoverability/) tutorial uses the Afriat numbers to draw preference bounds.
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role in the exercise |\n"
        f"|---|---|---|\n"
        f"| Observations $T$ | {T} | Budget-choice pairs in the two worked examples |\n"
        f"| Goods $L$ | {n_goods} | Three-good bundles, with figures projected onto goods 1 and 2 |\n"
        f"| Cobb-Douglas weights | {', '.join(f'{a:.3f}' for a in alpha)} | Ground-truth rational benchmark |\n"
        f"| Corrupted sample | {len(violations_inc)} violations | Two chosen bundles are swapped until GARP fails |\n"
        f"| Power exercise | {n_trials} trials | Random independent prices and quantities for each $T$ |\n"
        f"| Rational benchmark | 0 violations | Utility-maximizing Cobb-Douglas choices should always pass GARP |"
    )

    report.add_solution_method(
        "The computation is a graph problem on observations. Nodes are observed "
        "bundles; directed edges record revealed weak preference. Warshall's "
        "transitive closure is enough here because the sample is small and the "
        "object of interest is reachability, not a parametric demand curve.\n\n"
        "```text\n"
        "Input: prices p_t and chosen bundles x_t for t=1,...,T\n"
        "Output: pass/fail GARP decision and violating observation pairs\n\n"
        "1. For each pair (i,j), set R[i,j] = 1 if p_i . x_i >= p_i . x_j.\n"
        "2. Initialize R_star = R.\n"
        "3. For each intermediate node k:\n"
        "       for each origin i and destination j:\n"
        "           set R_star[i,j] = R_star[i,j] or (R_star[i,k] and R_star[k,j]).\n"
        "4. For each reachable pair (i,j), flag a violation if p_j . x_j > p_j . x_i.\n"
        "5. The data pass GARP exactly when the violation set is empty.\n"
        "```\n\n"
        "This is $O(T^3)$, which is trivial for the samples used here and clear enough "
        "to expose the economics. In larger revealed-preference datasets one would "
        "usually keep the same objects but implement the graph operations with sparse "
        "matrices or specialized reachability routines.\n\n"
        f"The Cobb-Douglas sample passes with {len(violations_con)} violations. "
        f"The corrupted sample fails with {len(violations_inc)} violating pairs."
    )

    report.add_results(
        "The first pair of figures plots the residual budget line for goods 1 and 2, "
        "holding the other good at its observed quantity. The projection is not the "
        "full three-good budget set, but it makes the revealed-preference comparison "
        "visible: rational data can look irregular across budgets without creating a "
        "strict cycle."
    )

    # --- Figure 1: Budget lines and bundles ---
    fig1 = plot_budget_lines_and_bundles(
        p_con, q_con,
        "Cobb-Douglas Sample: GARP Satisfied",
        consistent=True,
    )
    report.add_figure(
        "figures/budget-lines-consistent.png",
        "Budget lines and chosen bundles for the GARP-satisfying sample.",
        fig1,
        description="In the rational benchmark, every observation comes from the same "
        "Cobb-Douglas preference vector. The chosen bundles need not line up on a "
        "smooth two-dimensional curve, because prices and income vary, but the budget "
        "comparisons do not contradict one another.",
    )

    fig1b = plot_budget_lines_and_bundles(
        p_inc, q_inc,
        "Corrupted Sample: GARP Violated",
        consistent=False,
    )
    report.add_figure(
        "figures/budget-lines-inconsistent.png",
        "Budget lines and chosen bundles for the GARP-violating sample.",
        fig1b,
        description="After two bundles are swapped, the same price variation now creates "
        "a strict revealed-preference cycle. The failure is not a bad functional-form "
        "fit; it is a logical inconsistency under the maintained utility-maximization "
        "model.",
    )

    report.add_results(
        "The graph view is often the cleanest way to read the test. An arrow from "
        "$i$ to $j$ means the data reveal $x_i$ to be weakly preferred to $x_j$. "
        "The right panel adds indirect comparisons. Red arrows mark pairs involved "
        "in the strict GARP contradiction."
    )

    # --- Figure 2: Revealed preference graph ---
    fig2 = plot_revealed_preference_graph(
        R_con, Rstar_con,
        "Cobb-Douglas Sample: Revealed Preference Graph",
        violations_con,
    )
    report.add_figure(
        "figures/rp-graph-consistent.png",
        "Revealed-preference graph for the GARP-satisfying sample.",
        fig2,
        description="The rational sample has many revealed-preference links, especially "
        "after transitive closure, but none of those links returns to a strictly "
        "cheaper rejected bundle. That is the finite-data content of GARP.",
    )

    fig2b = plot_revealed_preference_graph(
        R_inc, Rstar_inc,
        "Corrupted Sample: Revealed Preference Graph",
        violations_inc,
    )
    report.add_figure(
        "figures/rp-graph-inconsistent.png",
        "Revealed-preference graph for the GARP-violating sample.",
        fig2b,
        description="In the corrupted sample, transitive revealed preference points one "
        "way while a later budget strictly reveals the reverse comparison. Those red "
        "pairs are enough to reject rationalizability for the whole dataset.",
    )

    report.add_results(
        "The power exercise asks whether this test has bite. Independent random "
        "prices and quantities are not an economic model; they are a useful null for "
        "seeing how quickly arbitrary behavior violates revealed preference. The "
        "Cobb-Douglas line is the known rational benchmark."
    )

    # --- Figure 3: Power of the GARP test ---
    fig3 = plot_garp_power(T_values, violation_rates, rational_violation_rates)
    report.add_figure(
        "figures/garp-power.png",
        "GARP violation rates for random choice data and a rational Cobb-Douglas benchmark.",
        fig3,
        description="Random behavior begins to fail with only a few observations and is "
        "almost always rejected by $T=50$. The zero line is the ground-truth rational "
        "benchmark: utility-maximizing Cobb-Douglas choices satisfy GARP by construction.",
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
        "Pairwise revealed-preference relation in the Cobb-Douglas sample",
        df_rp,
        description="The matrix records the same object algebraically. `R` is a direct "
        "budget comparison; `R*` is an indirect comparison added by transitive closure. "
        "Because this sample is rationalizable, these chains never produce a strict "
        "GARP contradiction.",
    )

    report.add_takeaway(
        "Afriat's theorem turns a familiar consumer-theory restriction into a finite "
        "sample test. Passing GARP does not identify a unique utility function, and it "
        "does not say preferences are Cobb-Douglas or smooth in any parametric sense. "
        "It says something sharper and more primitive: the observed choices can be "
        "ordered by some monotone concave utility function. Failing GARP is equally "
        "sharp, because no utility function in that class can rationalize the full "
        "dataset.\n\n"
        "That makes this tutorial the entry point for the revealed-preference sequence. "
        "Use [preference recoverability](../preference-recoverability/) when the data "
        "pass and the question is what utility or welfare bounds are implied. Use "
        "[Houtman-Maks](../houtman-maks-rational-subsets/) and the "
        "[money pump index](../money-pump-index/) when the data fail and the question "
        "is which observations drive the failure or how severe the cycle is."
    )

    report.add_references([
        "Afriat, S. N. (1967). The Construction of Utility Functions from Expenditure Data. "
        "*International Economic Review*, 8(1), 67-77.",
        "Bronars, S. G. (1987). The Power of Nonparametric Tests of Preference Maximization. "
        "*Econometrica*, 55(3), 693-698.",
        "Varian, H. R. (1982). The Nonparametric Approach to Demand Analysis. "
        "*Econometrica*, 50(4), 945-973.",
        "Varian, H. R. (2006). Revealed Preference. In M. Szenberg et al. (Eds.), "
        "*Samuelsonian Economics and the Twenty-First Century*. Oxford University Press.",
    ])

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
