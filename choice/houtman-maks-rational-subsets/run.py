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
from lib.plotting import save_figure, save_thumbnail, setup_style


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

    prices, quantities, _rational_quantities, synthetic_swapped_rows = generate_receipt_data()
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
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)

    Path("tables").mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv("tables/houtman-maks-diagnostics.csv", index=False)

    fig1 = plot_conflict_graph(weak, strict, exact_keep, greedy_keep, synthetic_swapped_rows, violation_counts)
    save_figure(fig1, "figures/conflict-graph.png", dpi=150)

    fig2 = plot_violation_heatmaps(prices, quantities, exact_keep)
    save_figure(fig2, "figures/violations-before-after.png", dpi=150)

    save_thumbnail("figures/conflict-graph.png", "figures/thumb.png")
    print(f"Done: 2 figures, 1 table, thumb reproduced.")


if __name__ == "__main__":
    main()
