#!/usr/bin/env python3
"""Revealed-preference cycles and the Money Pump Index.

A finite dataset can fail GARP by a small margin or through a large cycle of
budget slack. The tutorial builds the weighted revealed-preference graph and
uses Karp's maximum mean-cycle dynamic program to price the worst cycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


TOL = 1e-10


def cost_matrix(prices: np.ndarray, quantities: np.ndarray) -> np.ndarray:
    """Return C[i, j] = p_i dot x_j."""
    return prices @ quantities.T


def direct_revealed_savings(
    prices: np.ndarray, quantities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build direct revealed-preference edges and relative savings weights."""
    costs = cost_matrix(prices, quantities)
    own = np.diag(costs)
    savings = (own[:, None] - costs) / own[:, None]
    edges = (savings > TOL) & ~np.eye(len(prices), dtype=bool)
    return edges, savings, costs


def transitive_closure(relation: np.ndarray) -> np.ndarray:
    """Compute Boolean transitive closure with Warshall's algorithm."""
    reach = relation.copy()
    for k in range(reach.shape[0]):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def garp_violations(prices: np.ndarray, quantities: np.ndarray) -> list[tuple[int, int]]:
    """Return GARP-violating pairs in the standard revealed-preference graph."""
    costs = cost_matrix(prices, quantities)
    own = np.diag(costs)
    weak = own[:, None] >= costs - TOL
    strict = own[:, None] > costs + TOL
    reach = transitive_closure(weak)
    return [
        (i, j)
        for i in range(len(prices))
        for j in range(len(prices))
        if i != j and reach[i, j] and strict[j, i]
    ]


def simple_cycles(edges: np.ndarray) -> list[list[int]]:
    """Enumerate simple directed cycles once, for labeling small examples."""
    n = edges.shape[0]
    cycles: list[list[int]] = []

    def dfs(start: int, current: int, path: list[int], seen: set[int]) -> None:
        for nxt in np.flatnonzero(edges[current]):
            if nxt == start and len(path) > 1:
                cycles.append(path.copy())
            elif nxt not in seen and nxt >= start:
                seen.add(nxt)
                path.append(nxt)
                dfs(start, int(nxt), path, seen)
                path.pop()
                seen.remove(nxt)

    for start in range(n):
        dfs(start, start, [start], {start})
    return cycles


def cycle_mean(cycle: list[int], weights: np.ndarray) -> float:
    """Average edge weight along a directed cycle."""
    total = 0.0
    for pos, i in enumerate(cycle):
        j = cycle[(pos + 1) % len(cycle)]
        total += weights[i, j]
    return total / len(cycle)


def best_label_cycle(edges: np.ndarray, weights: np.ndarray) -> tuple[list[int], float]:
    """Find the highest-mean cycle by enumeration for a small display graph."""
    cycles = simple_cycles(edges)
    if not cycles:
        return [], 0.0
    best = max(cycles, key=lambda cyc: cycle_mean(cyc, weights))
    return best, cycle_mean(best, weights)


def karp_maximum_mean_cycle(edges: np.ndarray, weights: np.ndarray) -> float:
    """Karp dynamic program for the maximum mean-weight directed cycle."""
    if not simple_cycles(edges):
        return 0.0

    n = edges.shape[0]
    arc_list = [
        (i, j, float(weights[i, j]))
        for i in range(n)
        for j in range(n)
        if edges[i, j]
    ]
    neg_inf = -1.0e100
    dp = np.full((n + 1, n), neg_inf)
    dp[0, :] = 0.0

    for k in range(1, n + 1):
        for i, j, weight in arc_list:
            candidate = dp[k - 1, i] + weight
            if candidate > dp[k, j]:
                dp[k, j] = candidate

    best = 0.0
    for v in range(n):
        if dp[n, v] <= neg_inf / 2:
            continue
        ratios = [
            (dp[n, v] - dp[k, v]) / (n - k)
            for k in range(n)
            if dp[k, v] > neg_inf / 2
        ]
        if ratios:
            best = max(best, min(ratios))
    return max(0.0, best)


def make_cycle_case(label: str, savings: tuple[float, float, float]) -> dict[str, object]:
    """Create a three-bundle dataset with a controlled revealed-preference cycle."""
    quantities = np.eye(3)
    s12, s23, s31 = savings
    prices = np.array(
        [
            [1.00, 1.00 - s12, 1.20],
            [1.20, 1.00, 1.00 - s23],
            [1.00 - s31, 1.20, 1.00],
        ]
    )
    edges, weights, costs = direct_revealed_savings(prices, quantities)
    mpi = karp_maximum_mean_cycle(edges, weights)
    cycle, cycle_mpi = best_label_cycle(edges, weights)
    return {
        "label": label,
        "designed_slack": savings,
        "prices": prices,
        "quantities": quantities,
        "edges": edges,
        "weights": weights,
        "costs": costs,
        "mpi": mpi,
        "cycle": cycle,
        "cycle_mpi": cycle_mpi,
        "garp_violations": garp_violations(prices, quantities),
    }


def make_rational_case() -> dict[str, object]:
    """Create a baseline with no exploitable cycle."""
    quantities = np.eye(3)
    prices = np.array(
        [
            [1.00, 1.08, 1.20],
            [1.18, 1.00, 1.10],
            [1.12, 1.15, 1.00],
        ]
    )
    edges, weights, costs = direct_revealed_savings(prices, quantities)
    mpi = karp_maximum_mean_cycle(edges, weights)
    cycle, cycle_mpi = best_label_cycle(edges, weights)
    return {
        "label": "No cycle",
        "designed_slack": (),
        "prices": prices,
        "quantities": quantities,
        "edges": edges,
        "weights": weights,
        "costs": costs,
        "mpi": mpi,
        "cycle": cycle,
        "cycle_mpi": cycle_mpi,
        "garp_violations": garp_violations(prices, quantities),
    }


def plot_money_pump_graph(case: dict[str, object]) -> plt.Figure:
    """Plot the economic cycle that creates the money pump."""
    edges = case["edges"]
    weights = case["weights"]
    cycle = set(case["cycle"])
    labels = ["A", "B", "C"]
    xy = np.array([[0.0, 1.0], [-0.95, -0.55], [0.95, -0.55]])

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for i in range(3):
        for j in range(3):
            if not edges[i, j]:
                continue
            color = "#b3202a" if i in cycle and j in cycle else "#586f8f"
            width = 2.7 if i in cycle and j in cycle else 1.6
            rad = 0.22
            ax.annotate(
                "",
                xy=xy[j],
                xytext=xy[i],
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=width,
                    shrinkA=23,
                    shrinkB=23,
                    connectionstyle=f"arc3,rad={rad}",
                ),
            )
            midpoint = (xy[i] + xy[j]) / 2
            offset = np.array([-(xy[j] - xy[i])[1], (xy[j] - xy[i])[0]])
            offset = 0.09 * offset / max(np.linalg.norm(offset), 1e-8)
            ax.text(
                midpoint[0] + offset[0],
                midpoint[1] + offset[1],
                f"{weights[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.8),
            )

    for i, (x_pos, y_pos) in enumerate(xy):
        ax.scatter(x_pos, y_pos, s=1250, color="#f7f3e8", edgecolor="#222222", zorder=3)
        ax.text(x_pos, y_pos + 0.03, labels[i], ha="center", va="center", fontsize=16, weight="bold")
        ax.text(x_pos, y_pos - 0.20, f"$x_{i + 1}$", ha="center", va="center", fontsize=11)

    cycle_label = " -> ".join(labels[i] for i in case["cycle"] + [case["cycle"][0]])
    ax.set_title(f"Exploitable Preference Cycle: {cycle_label}")
    ax.text(
        0.0,
        -1.28,
        f"Money Pump Index = {case['mpi']:.1%} average budget slack per trade",
        ha="center",
        fontsize=11,
    )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def plot_severity_comparison(cases: list[dict[str, object]]) -> plt.Figure:
    """Compare pass/fail GARP with MPI severity."""
    labels = [str(case["label"]) for case in cases]
    mpi = np.array([float(case["mpi"]) for case in cases])
    garp_fail = np.array([len(case["garp_violations"]) > 0 for case in cases], dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].bar(labels, garp_fail, color=["#6c8f70" if x == 0 else "#b3202a" for x in garp_fail])
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("GARP rejects")
    axes[0].set_title("Pass/fail test")
    axes[0].set_yticks([0, 1], ["No", "Yes"])
    axes[0].tick_params(axis="x", rotation=25)

    colors = ["#6c8f70", "#d2a24c", "#c56f3f", "#b3202a"]
    axes[1].bar(labels, mpi, color=colors)
    axes[1].set_ylabel("Money Pump Index")
    axes[1].set_title("Expenditure exposed by the cycle")
    axes[1].set_ylim(0, max(mpi) * 1.25)
    axes[1].tick_params(axis="x", rotation=25)
    for x_pos, value in enumerate(mpi):
        axes[1].text(x_pos, value + 0.004, f"{value:.1%}", ha="center", fontsize=9)

    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()

    cases = [
        make_rational_case(),
        make_cycle_case("Small cycle", (0.03, 0.04, 0.02)),
        make_cycle_case("Medium cycle", (0.10, 0.13, 0.08)),
        make_cycle_case("Severe cycle", (0.18, 0.24, 0.08)),
    ]
    severe = cases[-1]

    rows = []
    for case in cases:
        cycle = case["cycle"]
        designed_slack = case["designed_slack"]
        rows.append(
            {
                "Dataset": case["label"],
                "GARP rejects": "yes" if case["garp_violations"] else "no",
                "Best cycle": "none" if not cycle else " -> ".join(str(i + 1) for i in cycle + [cycle[0]]),
                "Designed slack": (
                    "none"
                    if not designed_slack
                    else ", ".join(f"{slack:.0%}" for slack in designed_slack)
                ),
                "Karp MPI": f"{float(case['mpi']):.3f}",
            }
        )
    summary = pd.DataFrame(rows)

    report = ModelReport(
        "Revealed-Preference Cycles and the Money Pump Index",
        "Measuring the expenditure exposed by inconsistent choices.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A consumer buys bundle A at one price vector, bundle B at another, and bundle C "
        "at a third. These choices can violate GARP because each chosen bundle makes "
        "another affordable bundle look strictly better.\n\n"
        "The Money Pump Index prices the violation. It measures the average budget slack "
        "along the worst revealed-preference cycle.\n\n"
        "The task becomes a finite graph problem. Nodes are observations, edges carry "
        "budget slack, and Karp's dynamic program finds the maximum mean cycle."
    )

    report.add_equations(
        r"""
There are $T$ observations. Observation $i$ records a price vector
$p_i \in \mathbb{R}^G_+$ and chosen bundle $x_i \in \mathbb{R}^G_+$.
Let

$$E_{ij}=p_i \cdot x_j$$

be the cost of bundle $j$ at observation $i$ prices. Choosing $x_i$ when
$x_j$ was affordable, $E_{ii} \ge E_{ij}$, directly reveals
$x_i \succeq^D x_j$. For strict comparisons, define the relative budget slack
on a direct revealed-preference edge as

$$w_{ij} = \frac{E_{ii} - E_{ij}}{E_{ii}}.$$

The graph keeps edges with $w_{ij}>0$. For a directed cycle
$C=(i_1,\ldots,i_m,i_1)$, average slack is

$$\bar w(C)=\frac{1}{m}\sum_{\ell=1}^{m} w_{i_\ell,i_{\ell+1}}.$$

The Money Pump Index is the largest average slack over all directed cycles in
the revealed-preference graph:

$$\mathrm{MPI} = \max_C \bar w(C).$$
"""
    )

    report.add_model_setup(
        "| Object | Value | Interpretation |\n"
        "|---|---:|---|\n"
        "| Observations | 3 | One price vector and one chosen bundle in each row |\n"
        "| Bundles | 3 | A, B, and C are the only candidate bundles |\n"
        "| Own expenditure | 1.00 | Each chosen bundle costs one at its own prices |\n"
        "| Severe-cycle slack | 18%, 24%, 8% | Slack on A over B, B over C, and C over A |\n"
        f"| Severe MPI | {float(severe['mpi']):.3f} | Average extractable slack per trade |"
    )

    report.add_solution_method(
        "After the budget comparisons, the data are a directed graph. Each observation "
        "is a node. An edge $i \\to j$ exists when bundle $j$ was strictly cheaper at "
        "prices $i$. The edge weight is the saved budget share. Karp's dynamic program "
        "computes the maximum mean weight cycle.\n\n"
        "```text\n"
        "Inputs: prices p_i, bundles x_i, tolerance eps\n"
        "1. Form E_ij = p_i . x_j for all observations i,j.\n"
        "2. Add arc i -> j when (E_ii - E_ij) / E_ii > eps.\n"
        "3. Attach weight w_ij = (E_ii - E_ij) / E_ii to each arc.\n"
        "4. Let D_k(v) be the largest total weight of a k-arc path ending at v.\n"
        "5. Update D_k(v) = max_{u -> v} D_{k-1}(u) + w_uv for k = 1,...,T.\n"
        "6. Return max_v min_{0 <= k < T} [D_T(v) - D_k(v)] / (T - k).\n"
        "Output: MPI, the maximum average budget slack in a cycle.\n"
        "```"
    )

    report.add_table(
        "tables/mpi-summary.csv",
        "GARP Rejection and Money Pump Severity",
        summary,
        description=(
            "The table separates a logical rejection from expenditure at stake. All "
            "three inconsistent datasets reject GARP. Their MPI values range from "
            "0.030 to 0.167."
        ),
    )

    fig1 = plot_money_pump_graph(severe)
    report.add_figure(
        "figures/money-pump-cycle.png",
        "The severe revealed-preference cycle and its edge-level budget slack.",
        fig1,
        description=(
            "Each arrow points from an observed choice to a cheaper bundle at the same "
            "prices. The red cycle exposes 16.7 percent average budget slack."
        ),
    )

    fig2 = plot_severity_comparison(cases)
    report.add_figure(
        "figures/mpi-severity-comparison.png",
        "GARP is binary, while the Money Pump Index ranks the severity of failures.",
        fig2,
        description=(
            "The left panel records pass/fail GARP. The right panel keeps the "
            "expenditure scale across small, medium, and severe cycles."
        ),
    )

    report.add_takeaway(
        "Binary GARP tests say whether choices pass the axioms. The Money Pump Index "
        "says how much expenditure the worst cycle exposes. That scale separates small "
        "inconsistencies from large money-pump opportunities."
    )

    report.add_references(
        [
            "Echenique, F., Lee, S., & Shum, M. (2011). The money pump as a measure of revealed preference violations. Journal of Political Economy, 119(6), 1201-1223.",
            "Karp, R. M. (1978). A characterization of the minimum cycle mean in a digraph. Discrete Mathematics, 23(3), 309-311.",
            "Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.",
        ]
    )

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
