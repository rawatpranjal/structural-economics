#!/usr/bin/env python3
"""Revealed price preference and GAPP."""
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


def transitive_closure(relation: np.ndarray) -> np.ndarray:
    """Compute Boolean transitive closure."""
    reach = relation.copy()
    for k in range(reach.shape[0]):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def bundle_garp(prices: np.ndarray, quantities: np.ndarray) -> tuple[bool, list[tuple[int, int]]]:
    """Standard GARP check over chosen bundles."""
    costs = prices @ quantities.T
    own = np.diag(costs)
    weak = own[:, None] >= costs - TOL
    strict = own[:, None] > costs + TOL
    reach = transitive_closure(weak)
    violations = [
        (i, j)
        for i in range(len(prices))
        for j in range(len(prices))
        if i != j and reach[i, j] and strict[j, i]
    ]
    return len(violations) == 0, violations


def price_preference(
    prices: np.ndarray, quantities: np.ndarray,
) -> tuple[bool, list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """Check GAPP over price vectors instead of chosen bundles."""
    costs = prices @ quantities.T
    own_bundle_cost = np.diag(costs)
    weak = costs <= own_bundle_cost[None, :] + TOL
    strict = costs < own_bundle_cost[None, :] - TOL
    reach = transitive_closure(weak)
    violations = [
        (s, t)
        for s in range(len(prices))
        for t in range(len(prices))
        if s != t and reach[s, t] and strict[t, s]
    ]
    return len(violations) == 0, violations, weak, strict, reach


def synthetic_case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic three-observation example."""
    rng = np.random.default_rng(seed)
    prices = rng.uniform(0.5, 3.0, size=(3, 3))
    quantities = rng.uniform(0.2, 6.0, size=(3, 3))
    return prices, quantities


def case_catalog() -> list[dict[str, object]]:
    """Examples that separate bundle rationality from price rationality."""
    raw = [
        ("A", "Bundle-rational, price-inconsistent", 0),
        ("B", "Both restrictions pass", 2),
        ("C", "Bundle-inconsistent, price-rational", 5),
        ("D", "Both restrictions fail", 16),
    ]
    cases = []
    for case_id, label, seed in raw:
        prices, quantities = synthetic_case(seed)
        garp_ok, garp_violations = bundle_garp(prices, quantities)
        gapp_ok, gapp_violations, weak, strict, reach = price_preference(prices, quantities)
        cases.append(
            {
                "case_id": case_id,
                "label": label,
                "seed": seed,
                "prices": prices,
                "quantities": quantities,
                "garp_ok": garp_ok,
                "gapp_ok": gapp_ok,
                "garp_violations": garp_violations,
                "gapp_violations": gapp_violations,
                "price_weak": weak,
                "price_strict": strict,
                "price_reach": reach,
            }
        )
    return cases


def plot_cost_ratio_heatmap(case: dict[str, object]) -> plt.Figure:
    """Plot p_s dot x_t relative to p_t dot x_t for price preference."""
    prices = case["prices"]
    quantities = case["quantities"]
    costs = prices @ quantities.T
    own = np.diag(costs)
    ratio = costs / own[None, :]
    violations = set(case["gapp_violations"])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    image = ax.imshow(ratio, cmap="RdYlGn_r", vmin=0.65, vmax=1.35)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("$p^s \\cdot x^t / p^t \\cdot x^t$")

    for s in range(ratio.shape[0]):
        for t in range(ratio.shape[1]):
            label = f"{ratio[s, t]:.2f}"
            text_color = "white" if ratio[s, t] < 0.82 or ratio[s, t] > 1.22 else "#222222"
            ax.text(t, s, label, ha="center", va="center", color=text_color, weight="bold")
            if (s, t) in violations:
                ax.scatter(t, s, s=520, facecolors="none", edgecolors="#1f1f1f", linewidths=2.0)

    ax.set_xticks(range(3), [f"Bundle {t + 1}" for t in range(3)])
    ax.set_yticks(range(3), [f"Price {s + 1}" for s in range(3)])
    ax.set_title("Case A: Cost the Same Bundle Under Competing Price Vectors")
    ax.set_xlabel("Observed chosen bundle")
    ax.set_ylabel("Candidate price vector")
    fig.tight_layout()
    return fig


def plot_price_preference_graph(case: dict[str, object]) -> plt.Figure:
    """Plot the direct price-preference graph."""
    weak = case["price_weak"]
    strict = case["price_strict"]
    violation_pairs = set(case["gapp_violations"])
    xy = np.array([[0.0, 1.0], [-0.95, -0.55], [0.95, -0.55]])

    fig, ax = plt.subplots(figsize=(7.0, 5.7))
    for s in range(3):
        for t in range(3):
            if s == t or not weak[s, t]:
                continue
            is_violation = (s, t) in violation_pairs or (t, s) in violation_pairs
            color = "#b3202a" if is_violation else "#586f8f"
            width = 2.5 if strict[s, t] else 1.3
            ax.annotate(
                "",
                xy=xy[t],
                xytext=xy[s],
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=width,
                    shrinkA=23,
                    shrinkB=23,
                    connectionstyle="arc3,rad=0.20",
                ),
            )

    for node, (x_pos, y_pos) in enumerate(xy):
        ax.scatter(x_pos, y_pos, s=1150, color="#f7f3e8", edgecolor="#222222", zorder=3)
        ax.text(x_pos, y_pos + 0.03, f"$p^{node + 1}$", ha="center", va="center", fontsize=15, weight="bold")
        ax.text(x_pos, y_pos - 0.20, f"price {node + 1}", ha="center", va="center", fontsize=9)

    ax.set_title("Case A: A Cycle in Revealed Price Preference")
    ax.text(
        0,
        -1.25,
        "Arrows compare price vectors while holding the observed bundle fixed.",
        ha="center",
        fontsize=10,
    )
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.38, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def plot_garp_gapp_cases(cases: list[dict[str, object]]) -> plt.Figure:
    """Show that GARP and GAPP are distinct empirical restrictions."""
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    positions = {
        (True, True): (1, 1),
        (True, False): (1, 0),
        (False, True): (0, 1),
        (False, False): (0, 0),
    }
    colors = {
        (True, True): "#6c8f70",
        (True, False): "#d2a24c",
        (False, True): "#586f8f",
        (False, False): "#b3202a",
    }
    for case in cases:
        key = (bool(case["garp_ok"]), bool(case["gapp_ok"]))
        x_pos, y_pos = positions[key]
        ax.scatter(x_pos, y_pos, s=950, color=colors[key], edgecolor="#222222", zorder=3)
        ax.text(x_pos, y_pos, str(case["case_id"]), ha="center", va="center", color="white", weight="bold")

    ax.set_xticks([0, 1], ["GARP fails", "GARP passes"])
    ax.set_yticks([0, 1], ["GAPP fails", "GAPP passes"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Bundle rationalizability")
    ax.set_ylabel("Price rationalizability")
    ax.set_title("GARP and GAPP Test Different Objects")
    ax.grid(True, alpha=0.25)
    ax.text(0.5, -0.38, "Letters match the example labels in the diagnostic table.", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    setup_style()

    cases = case_catalog()
    focal = cases[0]

    summary = pd.DataFrame(
        [
            {
                "Case": case["case_id"],
                "Economic comparison": case["label"],
                "GARP": "pass" if case["garp_ok"] else "fail",
                "GAPP": "pass" if case["gapp_ok"] else "fail",
                "Bundle violations": len(case["garp_violations"]),
                "Price violations": len(case["gapp_violations"]),
            }
            for case in cases
        ]
    )

    report = ModelReport(
        "Revealed Price Preference",
        "When the object being compared is a price regime, not a chosen bundle.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Many revealed-preference exercises ask whether observed bundles could have come "
        "from one stable utility ordering. Sometimes the empirical object is different. A "
        "researcher may want to compare tax schedules, tariffs, insurance menus, or other "
        "price regimes and ask whether the data rank those regimes consistently.\n\n"
        "Revealed price preference keeps the same price-quantity observations "
        "$(p^t,x^t)$, but it reverses the object of comparison. GARP asks whether the "
        "chosen bundles can be ordered. GAPP asks whether the observed price vectors can "
        "be ordered by the bundles they make cheap. A price vector $p^s$ is better than "
        "$p^t$ for observation $t$ if it would have made the bundle actually chosen under "
        "$p^t$ weakly cheaper. The tutorial uses small deterministic panels to show that "
        "bundle rationalizability and price-regime rationalizability are distinct "
        "restrictions, not two descriptions of the same test."
    )

    report.add_equations(
        r"""
The data are $\mathcal D=\{(p^t,x^t)\}_{t=1}^{T}$, where
$p^t\in\mathbb R_{++}^{L}$ is the observed price vector and
$x^t\in\mathbb R_{+}^{L}$ is the chosen bundle. Own expenditure is
$m_t=p^t\cdot x^t$.

For price-regime comparisons, define the cross-cost matrix
$$
C_{st}=p^s\cdot x^t .
$$
Price vector $s$ is directly revealed weakly preferred to price vector $t$ when
it would have made the bundle chosen at $t$ no more expensive than it actually
was:
$$
sR_p^D t
\quad\Longleftrightarrow\quad
C_{st}\le C_{tt}=m_t .
$$

The strict relation is
$$
sP_p^D t
\quad\Longleftrightarrow\quad
C_{st}<C_{tt}.
$$
Let $R_p$ be the transitive closure of $R_p^D$. GAPP holds when there is no
pair $(s,t)$ such that
$$
sR_p t
\quad\text{and}\quad
tP_p^D s .
$$
The first statement says the data indirectly rank price vector $s$ at least as
good as price vector $t$. The second says the data strictly rank $t$ above $s$
in the direct reverse comparison. Together they form the price-regime analogue
of a revealed-preference cycle.
"""
    )

    report.add_model_setup(
        "| Object | Value | Interpretation |\n"
        "|---|---:|---|\n"
        "| Observations $T$ | 3 | Each case has three price-quantity observations |\n"
        "| Goods $L$ | 3 | Bundles are finite consumption vectors |\n"
        "| Deterministic cases | 4 | The examples cover every GARP/GAPP pass-fail cell |\n"
        "| Focal example | Case A | Bundle GARP passes while price GAPP fails |\n"
        f"| Focal GAPP violations | {len(focal['gapp_violations'])} | Strict reverse comparisons closing a price cycle |"
    )

    report.add_solution_method(
        "The computation is the same kind of finite graph exercise used in the "
        "[Afriat revealed-preference test](../revealed-preference-afriat/), but the "
        "nodes are price vectors rather than bundles. The only costly step is the "
        "transitive closure, which is $O(T^3)$ and exact for these finite panels.\n\n"
        "```text\n"
        "Algorithm: GAPP test for revealed price preference\n"
        "Input: price vectors p^t and chosen bundles x^t for t=1,...,T\n"
        "Output: pass/fail GAPP decision and violating price-vector pairs\n\n"
        "1. Form C_st = p^s . x^t for every pair of observations (s,t).\n"
        "2. Set R_p^D[s,t] = 1 if C_st <= C_tt.\n"
        "3. Set P_p^D[s,t] = 1 if C_st < C_tt.\n"
        "4. Compute the transitive closure R_p of R_p^D.\n"
        "5. For every pair (s,t), flag a violation if R_p[s,t] = 1 and P_p^D[t,s] = 1.\n"
        "6. The data pass GAPP exactly when the violation set is empty.\n"
        "```\n\n"
        "The script also runs ordinary bundle GARP on the same observations. That side-by-side "
        "comparison is deliberate: if the empirical question is about price regimes, a clean "
        "bundle-rationalizability test can miss the restriction that matters."
    )

    report.add_table(
        "tables/garp-gapp-examples.csv",
        "Bundle GARP and Price GAPP Diagnostics",
        summary,
        description=(
            "The four synthetic panels occupy all four pass-fail cells. Case A is the focal "
            "example because it looks rational when the bundles are tested, but it fails once "
            "the price vectors themselves are treated as the objects being ranked."
        ),
    )

    fig1 = plot_cost_ratio_heatmap(focal)
    report.add_figure(
        "figures/price-cost-ratios.png",
        "Cost ratios used to reveal preferences over price vectors.",
        fig1,
        description=(
            "The heat map shows the cross-cost ratio $C_{st}/C_{tt}$. Rows are candidate "
            "price vectors and columns are observed bundles. Entries below one mean that the "
            "row price vector would have made the column's bundle cheaper than the price "
            "vector under which that bundle was actually chosen."
        ),
    )

    fig2 = plot_price_preference_graph(focal)
    report.add_figure(
        "figures/price-preference-graph.png",
        "A cycle in the price-preference graph rejects GAPP.",
        fig2,
        description=(
            "The graph translates those cost comparisons into revealed preferences over "
            "price vectors. The arrows do not compare bundles. They say which price vector "
            "is revealed to be at least as attractive after holding a chosen bundle fixed."
        ),
    )

    fig3 = plot_garp_gapp_cases(cases)
    report.add_figure(
        "figures/garp-vs-gapp-cases.png",
        "GARP and GAPP classify the same datasets differently.",
        fig3,
        description=(
            "Across the four deterministic panels, GARP and GAPP separate cleanly. The same "
            "price-quantity data can support utility maximization over bundles while rejecting "
            "a consistent ordering of price regimes, or vice versa."
        ),
    )

    report.add_takeaway(
        "Revealed price preference is useful when the economic question is about price "
        "regimes rather than about the chosen bundles themselves. GARP asks whether a stable "
        "utility ordering can rationalize choices over bundles. GAPP asks whether observed "
        "price vectors can be ranked consistently by the bundles they make affordable. Since "
        "the two tests can disagree on the same finite data, the right diagnostic depends on "
        "which object the empirical exercise is trying to compare. After a standard "
        "[Afriat test](../revealed-preference-afriat/), this tutorial is the natural dual "
        "check for applications where prices, tariffs, or schedules are the object of welfare "
        "comparison."
    )

    report.add_references(
        [
            "Deb, R., Kitamura, Y., Quah, J. K. H., & Stoye, J. (2023). Revealed price preference: Theory and empirical analysis. Review of Economic Studies, 90(2), 707-743.",
            "Varian, H. R. (1982). The nonparametric approach to demand analysis. Econometrica, 50(4), 945-973.",
            "Chambers, C. P., & Echenique, F. (2016). Revealed Preference Theory. Cambridge University Press.",
        ]
    )

    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
