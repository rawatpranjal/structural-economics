#!/usr/bin/env python3
"""Houtman-Maks rational subsets for revealed-preference outlier diagnosis."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


TOL = 1e-10


def generate_receipt_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a mostly rational dataset with one corrupted observation."""
    rng = np.random.default_rng(1)
    observations = 12
    goods = 3
    alpha = np.array([0.45, 0.35, 0.20])
    prices = rng.uniform(0.60, 2.20, size=(observations, goods))
    incomes = rng.uniform(8.0, 16.0, size=observations)
    rational_quantities = alpha * incomes[:, None] / prices

    corrupted_quantities = rational_quantities.copy()
    swapped_rows = np.array([2, 3], dtype=int)
    corrupted_quantities[swapped_rows] = corrupted_quantities[swapped_rows[::-1]]
    return prices, corrupted_quantities, rational_quantities, swapped_rows


def preference_matrices(
    prices: np.ndarray, quantities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weak and strict direct revealed-preference matrices."""
    costs = prices @ quantities.T
    own = np.diag(costs)
    weak = own[:, None] >= costs - TOL
    strict = own[:, None] > costs + TOL
    return weak, strict, costs


def transitive_closure(relation: np.ndarray) -> np.ndarray:
    """Compute Boolean transitive closure."""
    reach = relation.copy()
    for k in range(reach.shape[0]):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def garp_violations(
    prices: np.ndarray, quantities: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Return GARP-violating pairs and graph objects."""
    weak, strict, _ = preference_matrices(prices, quantities)
    reach = transitive_closure(weak)
    violations = [
        (i, j)
        for i in range(len(prices))
        for j in range(len(prices))
        if i != j and reach[i, j] and strict[j, i]
    ]
    return violations, weak, strict, reach


def satisfies_garp(prices: np.ndarray, quantities: np.ndarray) -> bool:
    """Return whether the selected observations satisfy GARP."""
    violations, _, _, _ = garp_violations(prices, quantities)
    return len(violations) == 0


def tarjan_scc(relation: np.ndarray) -> list[list[int]]:
    """Strongly connected components of a directed Boolean graph."""
    n = relation.shape[0]
    index = 0
    stack: list[int] = []
    on_stack = [False] * n
    indices = [-1] * n
    lowlink = [0] * n
    components: list[list[int]] = []

    def visit(v: int) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in np.flatnonzero(relation[v]):
            w = int(w)
            if w == v:
                continue
            if indices[w] == -1:
                visit(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for node in range(n):
        if indices[node] == -1:
            visit(node)
    return components


def exact_houtman_maks(prices: np.ndarray, quantities: np.ndarray) -> list[int]:
    """Find the largest GARP-consistent subset by exhaustive search."""
    observations = len(prices)
    for size in range(observations, 0, -1):
        for subset in itertools.combinations(range(observations), size):
            idx = list(subset)
            if satisfies_garp(prices[idx], quantities[idx]):
                return idx
    return []


def greedy_houtman_maks(
    prices: np.ndarray, quantities: np.ndarray,
) -> tuple[list[int], list[dict[str, int]]]:
    """SCC-aware greedy removal heuristic for large samples."""
    remaining = list(range(len(prices)))
    removal_log: list[dict[str, int]] = []

    while True:
        idx = np.array(remaining)
        violations, weak, strict, _ = garp_violations(prices[idx], quantities[idx])
        if not violations:
            return remaining, removal_log

        components = tarjan_scc(weak)
        bad_nodes: set[int] = set()
        for component in components:
            has_strict_arc = any(
                strict[i, j]
                for i in component
                for j in component
                if i != j
            )
            if len(component) > 1 and has_strict_arc:
                bad_nodes.update(component)

        if not bad_nodes:
            bad_nodes = set(range(len(remaining)))

        participation = np.zeros(len(remaining), dtype=int)
        for i, j in violations:
            participation[i] += 1
            participation[j] += 1

        def score(local_node: int) -> tuple[int, int, int]:
            strict_degree = int(strict[local_node].sum() + strict[:, local_node].sum())
            return (int(participation[local_node]), strict_degree, -remaining[local_node])

        to_remove = max(bad_nodes, key=score)
        removal_log.append(
            {
                "observation": remaining[to_remove],
                "violation_count": int(participation[to_remove]),
                "strict_degree": int(strict[to_remove].sum() + strict[:, to_remove].sum()),
            }
        )
        remaining.pop(to_remove)


def plot_conflict_graph(
    weak: np.ndarray,
    strict: np.ndarray,
    exact_keep: list[int],
    greedy_keep: list[int],
    synthetic_swapped_rows: np.ndarray,
    violation_counts: np.ndarray,
) -> plt.Figure:
    """Plot the violating SCC and the exact Houtman-Maks removal."""
    n = weak.shape[0]
    keep = set(exact_keep)
    greedy_removed = set(range(n)) - set(greedy_keep)
    swapped = set(int(x) for x in synthetic_swapped_rows)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.c_[np.cos(angles), np.sin(angles)]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for i in range(n):
        for j in range(n):
            if i == j or not weak[i, j]:
                continue
            color = "#b3202a" if strict[i, j] and (violation_counts[i] or violation_counts[j]) else "#c2c7cf"
            width = 1.9 if color == "#b3202a" else 0.7
            alpha = 0.78 if color == "#b3202a" else 0.35
            ax.annotate(
                "",
                xy=xy[j],
                xytext=xy[i],
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=width,
                    alpha=alpha,
                    shrinkA=17,
                    shrinkB=17,
                    connectionstyle="arc3,rad=0.12",
                ),
            )

    for node, (x_pos, y_pos) in enumerate(xy):
        removed = node not in keep
        color = "#b3202a" if removed else "#f7f3e8"
        edge = "#c58b11" if node in swapped else "#222222"
        linewidth = 2.8 if node in swapped else 1.2
        ax.scatter(x_pos, y_pos, s=720, color=color, edgecolor=edge, linewidth=linewidth, zorder=3)
        if node in swapped:
            ax.scatter(
                x_pos,
                y_pos,
                s=930,
                facecolors="none",
                edgecolors="#c58b11",
                linewidths=1.7,
                zorder=4,
            )
        if node in greedy_removed:
            ax.scatter(x_pos, y_pos, s=230, marker="x", color="#111111", linewidths=2.8, zorder=5)
        text_color = "white" if removed else "#222222"
        ax.text(x_pos, y_pos + 0.02, str(node + 1), ha="center", va="center", color=text_color, weight="bold")
        if violation_counts[node] > 0:
            ax.text(
                x_pos,
                y_pos - 0.22,
                f"{int(violation_counts[node])}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color if removed else "#b3202a",
            )

    ax.set_title("One Observation Explains the Revealed-Preference Conflict")
    ax.text(
        0,
        -1.31,
        "Small red numbers count conflict participation.\n"
        "Gold rings mark swapped rows; black x is the greedy deletion; red fill is the exact deletion.",
        ha="center",
        fontsize=9.5,
    )
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.42, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def plot_violation_heatmaps(
    prices: np.ndarray,
    quantities: np.ndarray,
    exact_keep: list[int],
) -> plt.Figure:
    """Show GARP violations before and after removing the HM outlier."""
    violations_full, _, _, reach_full = garp_violations(prices, quantities)
    full_matrix = np.zeros_like(reach_full, dtype=float)
    for i, j in violations_full:
        full_matrix[i, j] = 1.0

    idx = np.array(exact_keep)
    violations_keep, _, _, reach_keep = garp_violations(prices[idx], quantities[idx])
    keep_matrix = np.zeros_like(reach_keep, dtype=float)
    for i, j in violations_keep:
        keep_matrix[i, j] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    panels = [
        (full_matrix, "Full dataset", len(violations_full), list(range(len(prices)))),
        (keep_matrix, "Retained core", len(violations_keep), exact_keep),
    ]
    for ax, (matrix, title, count, labels) in zip(axes, panels):
        ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"{title}: {count} violations")
        ax.set_xlabel("Observation j")
        ax.set_ylabel("Observation i")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([str(x + 1) for x in labels], fontsize=8)
        ax.set_yticklabels([str(x + 1) for x in labels], fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0:
                    ax.text(j, i, "x", ha="center", va="center", color="white", weight="bold")
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()

    prices, quantities, rational_quantities, synthetic_swapped_rows = generate_receipt_data()
    exact_keep = exact_houtman_maks(prices, quantities)
    greedy_keep, removal_log = greedy_houtman_maks(prices, quantities)
    violations, weak, strict, _ = garp_violations(prices, quantities)
    violation_counts = np.zeros(len(prices), dtype=int)
    for i, j in violations:
        violation_counts[i] += 1
        violation_counts[j] += 1

    exact_removed = sorted(set(range(len(prices))) - set(exact_keep))
    greedy_removed = [entry["observation"] for entry in removal_log]

    diagnostic_rows = []
    synthetic_swapped = set(int(x) for x in synthetic_swapped_rows)
    for obs in range(len(prices)):
        diagnostic_rows.append(
            {
                "Observation": obs + 1,
                "Violation participation": int(violation_counts[obs]),
                "Synthetic swap row": "yes" if obs in synthetic_swapped else "no",
                "Exact HM action": "remove" if obs in exact_removed else "keep",
                "Greedy action": "remove" if obs in greedy_removed else "keep",
                "Observed good 1": f"{quantities[obs, 0]:.2f}",
                "Pre-swap good 1": f"{rational_quantities[obs, 0]:.2f}",
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)

    report = ModelReport(
        "Houtman-Maks Rational Subsets",
        "Finding the largest rationalizable core after a revealed-preference rejection.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A GARP rejection is not automatically a rejection of economic choice theory. In "
        "household scanner data, lab choices, or administrative purchase records, the failure "
        "could come from a different decision problem, a transcription error, or one receipt "
        "that should not be pooled with the rest. The Houtman-Maks index asks how much of the "
        "sample can still be read as utility-maximizing behavior under one stable preference "
        "ordering.\n\n"
        "This tutorial uses a small synthetic demand panel where the uncorrupted choices come "
        "from Cobb-Douglas budget shares. Two chosen bundles are then swapped across receipts. "
        "The full dataset fails GARP, but the largest rationalizable core keeps "
        f"{len(exact_keep)} of {len(prices)} observations. The example is useful because the "
        "simulation gives an oracle label for the swapped rows, so the exact Houtman-Maks "
        "answer and a greedy large-sample diagnostic can be compared to a known source of "
        "contamination."
    )

    report.add_equations(
        r"""
There are $T$ observations. Observation $t$ contains prices $p_t \in \mathbb{R}_{+}^{J}$ and the chosen bundle $x_t \in \mathbb{R}_{+}^{J}$. Expenditure is $m_t=p_t \cdot x_t$.

Choice $t$ directly weakly reveals $x_t$ preferred to $x_s$ when $x_s$ was affordable at prices $p_t$:

$$x_t R^D x_s \quad \Longleftrightarrow \quad p_t \cdot x_t \geq p_t \cdot x_s.$$

The direct relation is strict when the inequality is strict. Let $R$ be the transitive closure of $R^D$. GARP holds on a subset $S$ if there is no pair $t,s \in S$ such that

$$x_t R x_s \quad \text{and} \quad p_s \cdot x_s > p_s \cdot x_t.$$

For any subset $S$, let $\operatorname{GARP}(S)=1$ when these restrictions hold after keeping only observations in $S$. The Houtman-Maks index is

$$HM = \max_{S \subseteq \{1,\ldots,T\}} |S| \quad \text{s.t.} \quad \operatorname{GARP}(S)=1.$$

The minimum number of observations needed to restore GARP is

$$T - HM.$$
"""
    )

    report.add_model_setup(
        "| Object | Value | Interpretation |\n"
        "|---|---:|---|\n"
        f"| Observations $T$ | {len(prices)} | Shopping trips with prices and chosen bundles |\n"
        "| Goods $J$ | 3 | Small multi-good demand environment |\n"
        "| Data-generating preferences | Cobb-Douglas shares $(0.45,0.35,0.20)$ | Rational benchmark before corruption |\n"
        "| Synthetic corruption | rows 3 and 4 swapped | Oracle label available only because this is simulated data |\n"
        f"| Full-sample GARP violations | {len(violations)} | Contradictions after taking transitive closure |\n"
        f"| Exact Houtman-Maks index | {len(exact_keep)} | Largest rationalizable subset size |\n"
        f"| Greedy deletion | observation {greedy_removed[0] + 1} | Same deletion selected by the heuristic |"
    )

    solution_method = r"""
The exact calculation treats Houtman-Maks as a finite combinatorial problem. For $T=12$, exhaustive subset search is small enough to be the benchmark.

```text
Algorithm: exact Houtman-Maks core
Inputs: observations {(p_t, x_t)}_{t=1}^T
Output: largest subset S* satisfying GARP

for k = T, T-1, ..., 1:
    for each subset S with |S| = k:
        build R^D on S using p_t dot x_t >= p_t dot x_s
        compute the transitive closure R of R^D
        if no strict budget cycle remains:
            return S* = S and HM = k
```

That search is transparent but not scalable. The second calculation keeps the same revealed-preference object and uses the graph structure to choose deletions. A violating strongly connected component is a set of observations tied together by revealed-preference paths, with at least one strict comparison closing the cycle.

```text
Algorithm: SCC greedy Houtman-Maks diagnosis
Inputs: observations {(p_t, x_t)}_{t=1}^T
Output: a GARP-consistent retained set S

initialize S = {1, ..., T}
while GARP(S) fails:
    compute weak arcs, strict arcs, and violating pairs on S
    find strongly connected components of the weak graph
    restrict attention to components containing a strict internal arc
    remove the observation with the most violation participation
return S
```

In this run, the greedy rule removes observation """
    solution_method += (
        f"{greedy_removed[0] + 1}, the same receipt removed by exact search. The exact result is "
        "the benchmark; the greedy result is a scalable diagnostic for larger panels."
    )
    report.add_solution_method(solution_method)

    report.add_table(
        "tables/houtman-maks-diagnostics.csv",
        "Which Receipts Carry the Rejection",
        diagnostics,
        description=(
            "The table separates three objects that are easy to conflate. The synthetic swap "
            "column is the oracle label from the simulation. The exact Houtman-Maks action is "
            "the maximum rationalizable subset. The greedy action is the approximation one "
            "would use when exact subset enumeration is too expensive."
        ),
    )

    fig1 = plot_conflict_graph(weak, strict, exact_keep, greedy_keep, synthetic_swapped_rows, violation_counts)
    report.add_figure(
        "figures/conflict-graph.png",
        "Preference conflict graph comparing exact, greedy, and synthetic corruption markers.",
        fig1,
        description=(
            "The revealed-preference graph makes the comparison concrete. Red fill marks the "
            "exact Houtman-Maks deletion, the black x marks the greedy deletion, and gold rings "
            "mark the two receipts whose bundles were swapped in the simulation. The method "
            "does not need to remove both swapped rows: dropping one side of the conflict is "
            "enough to recover a GARP-consistent core."
        ),
    )

    fig2 = plot_violation_heatmaps(prices, quantities, exact_keep)
    report.add_figure(
        "figures/violations-before-after.png",
        "GARP violations before and after removing the Houtman-Maks outlier.",
        fig2,
        description=(
            "The heat maps are the certificate. The full sample has strict budget-cycle "
            "contradictions. After removing the exact Houtman-Maks deletion, the retained "
            "core has none."
        ),
    )

    report.add_takeaway(
        "Houtman-Maks turns a binary revealed-preference rejection into a question about the "
        "size of the rationalizable core. Here the rejection is concentrated: one deletion "
        "restores GARP for 11 of 12 observations. The synthetic oracle also shows the main "
        "interpretive limit. The index is not a causal label for every corrupted row; it is a "
        "finite-data robustness measure for how much of the sample can still support utility "
        "maximization."
    )

    report.add_references(
        [
            "Houtman, M., & Maks, J. A. H. (1985). Determining all maximal data subsets consistent with revealed preference. Kwantitatieve Methoden, 19, 89-104.",
            "Heufer, J., & Hjertstrand, P. (2015). Consistent subsets: Computationally feasible methods to compute the Houtman-Maks-index. Economics Letters, 128, 87-89.",
            "Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.",
        ]
    )

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
