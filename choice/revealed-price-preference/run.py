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
    ax.set_title("Price Preference Comes from Costing the Same Bundle at Other Prices")
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

    ax.set_title("GAPP Fails When Price Preferences Cycle")
    ax.text(
        0,
        -1.25,
        "Arrows mean one price vector is revealed cheaper for an observed bundle.",
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
        "Testing rationalizability of price vectors rather than chosen bundles.",
    )

    report.add_overview(
        "Standard revealed preference asks whether chosen bundles can be rationalized by "
        "utility maximization. Revealed price preference asks a dual question: do the "
        "price vectors themselves behave as if there is a consistent preference over "
        "prices? This is useful when the empirical object is a price regime, tax schedule, "
        "or tariff vector rather than only a consumed bundle.\n\n"
        "The key distinction is simple. GARP compares bundles at the prices faced when a "
        "bundle was chosen. GAPP fixes an observed chosen bundle and asks whether another "
        "price vector would have made that same bundle weakly cheaper. Cycles in these "
        "price comparisons reject rationalizability of prices, even in examples where "
        "standard bundle GARP may pass."
    )

    report.add_equations(
        r"""
Let $p^s$ be price vector $s$ and $x^t$ be the bundle chosen under price vector $t$.
Price $s$ is directly revealed preferred to price $t$ when it makes bundle $x^t$ weakly cheaper:

$$R_p[s,t] = 1\{p^s \cdot x^t \le p^t \cdot x^t\}.$$

The strict relation is

$$P_p[s,t] = 1\{p^s \cdot x^t < p^t \cdot x^t\}.$$

GAPP fails when price $s$ is indirectly revealed preferred to price $t$, but $t$ is strictly directly revealed preferred to $s$:

$$R_p^{*}[s,t] = 1 \quad \text{and} \quad P_p[t,s] = 1.$$
"""
    )

    report.add_model_setup(
        "| Object | Value | Interpretation |\n"
        "|---|---:|---|\n"
        "| Price vectors | 3 | Candidate price regimes |\n"
        "| Goods | 3 | Bundles are costed under each price vector |\n"
        "| Focal example | Case A | GARP passes but GAPP fails |\n"
        f"| Focal price violations | {len(focal['gapp_violations'])} | Cyclic price-preference contradictions |"
    )

    report.add_solution_method(
        "The script first runs the usual GARP test on bundles. It then transposes the "
        "economic comparison: for each observed bundle, compare how costly that same bundle "
        "would have been under every observed price vector. Those comparisons form a price "
        "preference graph. Warshall closure gives indirect price preference, and GAPP checks "
        "whether any indirect price ranking is contradicted by a strict reverse comparison.\n\n"
        "The four deterministic examples show that bundle rationalizability and price "
        "rationalizability are distinct restrictions rather than two names for the same test."
    )

    report.add_table(
        "tables/garp-gapp-examples.csv",
        "Bundle GARP and Price GAPP Diagnostics",
        summary,
        description=(
            "The examples occupy all four pass/fail cells. This is the main conceptual point: "
            "testing choices over bundles and testing choices over prices are different "
            "revealed-preference exercises."
        ),
    )

    fig1 = plot_cost_ratio_heatmap(focal)
    report.add_figure(
        "figures/price-cost-ratios.png",
        "Cost ratios used to reveal preferences over price vectors.",
        fig1,
        description=(
            "Rows are candidate price vectors and columns are observed bundles. Entries below "
            "one mean the row price vector makes that column's bundle cheaper than its observed "
            "price vector did."
        ),
    )

    fig2 = plot_price_preference_graph(focal)
    report.add_figure(
        "figures/price-preference-graph.png",
        "A cycle in the price-preference graph rejects GAPP.",
        fig2,
        description=(
            "The arrows are not preferences over bundles. They are revealed comparisons among "
            "price vectors obtained by holding observed bundles fixed."
        ),
    )

    fig3 = plot_garp_gapp_cases(cases)
    report.add_figure(
        "figures/garp-vs-gapp-cases.png",
        "GARP and GAPP classify the same datasets differently.",
        fig3,
        description=(
            "Because GARP and GAPP ask different economic questions, a dataset can pass one "
            "test and fail the other."
        ),
    )

    report.add_takeaway(
        "Revealed price preference is not a cosmetic relabeling of GARP. GARP tests whether "
        "chosen bundles can be rationalized by utility maximization. GAPP tests whether price "
        "vectors are themselves consistently ranked by revealed cost comparisons. The same "
        "price-quantity data can pass one test and fail the other, so the right diagnostic "
        "depends on whether the empirical question is about bundles or price regimes."
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
