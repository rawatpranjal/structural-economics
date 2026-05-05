#!/usr/bin/env python3
"""Money Pump Index for revealed-preference violations.

The example treats a GARP violation as an economically exploitable cycle. The
Money Pump Index (MPI) asks how much budget slack is available, on average, in
the worst revealed-preference cycle.
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
    """Compare binary GARP failure with MPI severity."""
    labels = [str(case["label"]) for case in cases]
    mpi = np.array([float(case["mpi"]) for case in cases])
    exact_cycle_mean = np.array([float(case["cycle_mpi"]) for case in cases])
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
    axes[1].scatter(
        labels,
        exact_cycle_mean,
        color="#222222",
        marker="D",
        s=38,
        label="enumerated cycle mean",
        zorder=3,
    )
    axes[1].set_ylabel("Money Pump Index")
    axes[1].set_title("Economic severity")
    axes[1].set_ylim(0, max(mpi) * 1.25)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
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
                "Enumerated mean": f"{float(case['cycle_mpi']):.3f}",
                "Karp MPI": f"{float(case['mpi']):.3f}",
                "Violating pairs": len(case["garp_violations"]),
            }
        )
    summary = pd.DataFrame(rows)

    report = ModelReport(
        "Money Pump Index for Revealed Preference",
        "Putting economic size on a revealed-preference violation.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A GARP rejection is a statement about internal consistency, not about economic "
        "magnitude. Two datasets can both violate revealed preference, but one violation "
        "may be a nearly indifferent accounting error while another creates a large "
        "cyclic arbitrage. The money-pump interpretation asks how much budget slack an "
        "outside trader could extract by repeatedly moving the consumer around the "
        "revealed-preference cycle.\n\n"
        "The example keeps the economic object deliberately small: three observed choices, "
        "three bundles, and own expenditure normalized to one. In the severe case the "
        "consumer chooses A when B was 18 percent cheaper, chooses B when C was 24 percent "
        "cheaper, and chooses C when A was 8 percent cheaper. GARP says this cannot come "
        "from one stable utility ordering. The Money Pump Index says the exploitable cycle "
        f"is worth {float(severe['mpi']):.1%} of expenditure per trade."
    )

    report.add_equations(
        r"""
There are $T$ observations. Observation $i$ records a price vector $p_i \in \mathbb{R}^G_+$
and chosen bundle $x_i \in \mathbb{R}^G_+$. Let

$$E_{ij}=p_i \cdot x_j$$

be the cost of bundle $j$ at prices $i$. Choosing $x_i$ directly reveals
$x_i \succeq^D x_j$ when $E_{ii} \ge E_{ij}$. For strict comparisons, define
the relative budget slack on a direct revealed-preference edge as

$$w_{ij} = \frac{E_{ii} - E_{ij}}{E_{ii}}.$$

In this tutorial the graph keeps edges with $w_{ij}>0$. For a directed cycle
$C=(i_1,\ldots,i_m,i_1)$, the average slack is

$$\bar w(C)=\frac{1}{m}\sum_{\ell=1}^{m} w_{i_\ell,i_{\ell+1}}.$$

The Money Pump Index is the largest average slack over all directed cycles in
the revealed-preference graph:

$$\operatorname{MPI} = \max_C \bar w(C).$$
"""
    )

    report.add_model_setup(
        "| Object | Value | Interpretation |\n"
        "|---|---:|---|\n"
        "| Observations | 3 | One price vector and one chosen bundle per observation |\n"
        "| Bundles | 3 | A, B, and C are the only candidate bundles |\n"
        "| Own expenditure | 1.00 | Each chosen bundle is normalized to cost one |\n"
        "| Severe-cycle slack | 18%, 24%, 8% | Slack on A over B, B over C, and C over A |\n"
        f"| Severe MPI | {float(severe['mpi']):.3f} | Average extractable slack per trade |\n"
        "| Nearby tutorials | Afriat, Houtman-Maks | Afriat asks whether choices are rationalizable; Houtman-Maks asks which rows to drop |"
    )

    report.add_solution_method(
        "The first step is still the revealed-preference test: compare every chosen bundle "
        "with every other bundle at the same prices. The extra step is to put weights on "
        "strict revealed-preference arcs and solve a maximum mean-cycle problem. Karp's "
        "dynamic program is exact for this finite graph. The script also enumerates cycles "
        "only because the graph has three nodes, which gives a transparent benchmark for "
        "the plotted examples.\n\n"
        "```text\n"
        "Inputs: prices p_i, bundles x_i, tolerance eps\n"
        "1. Form E_ij = p_i . x_j for all observations i,j.\n"
        "2. Add arc i -> j when (E_ii - E_ij) / E_ii > eps.\n"
        "3. Attach weight w_ij = (E_ii - E_ij) / E_ii to each arc.\n"
        "4. Let D_k(v) be the largest total weight of any k-arc path ending at v.\n"
        "5. Iterate D_k(v) = max_{u -> v} D_{k-1}(u) + w_uv for k = 1,...,T.\n"
        "6. Return max_v min_{0 <= k < T} [D_T(v) - D_k(v)] / (T-k).\n"
        "Output: MPI, the maximum average budget slack in a cycle.\n"
        "```\n\n"
        f"For the severe example, the enumerated cycle mean and Karp's MPI both equal "
        f"$\\operatorname{{MPI}}={float(severe['mpi']):.3f}$."
    )

    report.add_table(
        "tables/mpi-summary.csv",
        "GARP Rejection and Money Pump Severity",
        summary,
        description=(
            "The first column of interest is the GARP rejection: it is the same yes/no "
            "answer for all three inconsistent datasets. The severity columns show why "
            "that is not enough. In this small graph, direct enumeration gives the same "
            "cycle mean as Karp's dynamic program."
        ),
    )

    fig1 = plot_money_pump_graph(severe)
    report.add_figure(
        "figures/money-pump-cycle.png",
        "The severe revealed-preference cycle and its edge-level budget slack.",
        fig1,
        description=(
            "Each arrow points from a chosen bundle to another bundle that was strictly "
            "cheaper at the same prices. The red cycle is not just a graph-theoretic "
            "oddity: it is the sequence of trades that creates the money pump."
        ),
    )

    fig2 = plot_severity_comparison(cases)
    report.add_figure(
        "figures/mpi-severity-comparison.png",
        "GARP is binary, while the Money Pump Index ranks the severity of failures.",
        fig2,
        description=(
            "The left panel deliberately throws away magnitude. The right panel keeps it: "
            "small, medium, and severe cycles all fail the same rationalizability test, "
            "but the average slack available to a trader is very different. The black "
            "diamonds are the exact cycle means obtained by enumeration in this three-node "
            "example."
        ),
    )

    report.add_takeaway(
        "The Money Pump Index is useful because it keeps the economic content of a "
        "revealed-preference failure in view. GARP asks whether the finite dataset is "
        "rationalizable. MPI asks how much expenditure is exposed by the worst cycle. "
        "That makes it a complement to the "
        "[Afriat test](../revealed-preference-afriat/) and to "
        "[Houtman-Maks deletion diagnostics](../houtman-maks-rational-subsets/): "
        "after rejection, one can ask whether the problem is small, large, or "
        "concentrated in a few observations."
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
