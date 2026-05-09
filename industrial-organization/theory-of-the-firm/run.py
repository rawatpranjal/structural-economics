#!/usr/bin/env python3
"""Firm boundaries, incomplete contracts, hold-up, and integration."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


REGIME_ORDER = ["Spot contract", "Long-term contract", "Vertical integration"]


def investment_outcomes(specificity: np.ndarray, theta: float = 4.0) -> pd.DataFrame:
    """Compute investment and surplus under alternative governance regimes."""
    regimes = {
        "Spot contract": {
            "incentive": 0.72 - 0.55 * specificity,
            "governance_cost": 0.02 + 0.04 * specificity,
        },
        "Long-term contract": {
            "incentive": 0.72 - 0.25 * specificity,
            "governance_cost": 0.38 + 0.03 * specificity,
        },
        "Vertical integration": {
            "incentive": 0.74 - 0.03 * specificity,
            "governance_cost": 1.05 - 0.35 * specificity,
        },
    }

    rows = []
    first_best_surplus = 0.5 * theta**2
    for name, values in regimes.items():
        incentive = np.clip(values["incentive"], 0.05, 1.0)
        investment = theta * incentive
        surplus = theta * investment - 0.5 * investment**2 - values["governance_cost"]
        for s, inc, inv, val in zip(specificity, incentive, investment, surplus):
            rows.append({
                "Specificity": s,
                "Regime": name,
                "Incentive share": inc,
                "Investment": inv,
                "Surplus": val,
                "Efficiency ratio": val / first_best_surplus,
            })
    return pd.DataFrame(rows)


def best_regime(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("Specificity")["Surplus"].idxmax()
    return df.loc[idx].sort_values("Specificity")


def summarize_regions(best: pd.DataFrame) -> list[tuple[float, float, str]]:
    """Compress the chosen regime into contiguous specificity intervals."""
    rows = best[["Specificity", "Regime"]].to_records(index=False)
    regions: list[tuple[float, float, str]] = []
    start = float(rows[0]["Specificity"])
    prev_s = start
    prev_regime = str(rows[0]["Regime"])

    for s, regime in rows[1:]:
        s = float(s)
        regime = str(regime)
        if regime != prev_regime:
            regions.append((start, prev_s, prev_regime))
            start = s
            prev_regime = regime
        prev_s = s
    regions.append((start, prev_s, prev_regime))
    return regions


def main() -> None:
    setup_style()
    theta = 4.0
    specificity = np.linspace(0.0, 1.0, 201)
    df = investment_outcomes(specificity)
    best = best_regime(df)
    fine_specificity = np.linspace(0.0, 1.0, 10001)
    fine_best = best_regime(investment_outcomes(fine_specificity, theta=theta))
    regions = summarize_regions(fine_best)
    efficient_investment = theta
    first_best_surplus = 0.5 * theta**2

    print("Theory of the firm tutorial")
    print(best.groupby("Regime").size().to_string())

    report = ModelReport(
        "Firm Boundaries, Hold-Up, and Vertical Integration",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A supplier may invest in tooling useful only for one buyer. The asset raises "
        "joint surplus. After the asset is sunk, bargaining may leave the supplier "
        "with too little of the return. The supplier then invests for private surplus.\n\n"
        "The object is a governance choice. Asset specificity $s\\in[0,1]$ measures "
        "how hard the asset is to redeploy. The choices are spot exchange, a long-term "
        "contract, and vertical integration. Each form changes investment incentives "
        "and governance cost.\n\n"
        "For each $s$, the code computes investment and surplus under each form. A grid "
        "search then selects the surplus-maximizing boundary choice. The output is a "
        "set of specificity thresholds."
    )

    report.add_equations(r"""
Let $s$ denote asset specificity.
Let $g\in\mathcal G$ index spot exchange, a long-term contract, and vertical
integration.
Relationship-specific investment $x$ creates gross value
$$V(x) = \theta x - \frac{1}{2}x^2$$

First-best investment solves $V'(x)=0$, so
$$x^{*} = \theta$$

Regime $g$ lets the investor capture share $b_g(s)$ of marginal value.
The private first-order condition is
$$b_g(s)\theta - x = 0,$$
which gives
$$x_g(s) = b_g(s)\theta$$

Total surplus subtracts governance cost $F_g(s)$:
$$W_g(s) = \theta x_g(s) - \frac{1}{2}x_g(s)^2 - F_g(s)$$

The incentive schedules are
$$b_{\text{spot}}(s)=0.72-0.55s,\quad
b_{\text{contract}}(s)=0.72-0.25s,\quad
b_{\text{integration}}(s)=0.74-0.03s.$$

Governance costs are
$$F_{\text{spot}}(s)=0.02+0.04s,\quad
F_{\text{contract}}(s)=0.38+0.03s,\quad
F_{\text{integration}}(s)=1.05-0.35s.$$

The selected governance form is
$$g^{*}(s)=\arg\max_{g\in\mathcal G} W_g(s).$$

The first-best surplus benchmark is
$$W^{*}=\frac{1}{2}\theta^2$$
""")

    report.add_model_setup(
        "The calibration is illustrative. Higher specificity weakens market incentives. "
        "Contracts protect some returns at a cost. Integration gives control rights but "
        "carries hierarchy cost.\n\n"
        "| Object | Interpretation |\n"
        "|--------|----------------|\n"
        "| $s\\in[0,1]$ | Asset specificity, with higher $s$ meaning weaker redeployability outside the relationship |\n"
        "| $\\theta=4$ | Marginal productivity scale, so the first-best investment is $x^{*}=4$ |\n"
        "| $b_g(s)$ | Share of the marginal investment return captured by the investor under governance $g$ |\n"
        "| $F_g(s)$ | Drafting, monitoring, bureaucracy, and adaptation cost under governance $g$ |\n"
        "| Spot contract | Low fixed governance cost, but incentives fall sharply as specificity rises |\n"
        "| Long-term contract | More protection against hold-up, with moderate contracting cost |\n"
        "| Vertical integration | Stronger residual control rights, with higher internal governance cost |"
    )

    report.add_solution_method(
        "Given $b_g(s)$, private investment has the closed form $x_g(s)=b_g(s)\\theta$. "
        "The only numerical step is a grid comparison over $s$. At each point, the "
        "code evaluates $W_g(s)$ for the three governance forms. It keeps the form with "
        "the largest surplus.\n\n"
        "```text\n"
        "Inputs: specificity grid S, regimes G, productivity theta,\n"
        "        incentive schedules b_g(s), governance costs F_g(s)\n"
        "\n"
        "First-best benchmark:\n"
        "    x_star = theta\n"
        "    W_star = 0.5 * theta^2\n"
        "\n"
        "For each s in S:\n"
        "    For each governance regime g in G:\n"
        "        x_g(s) = b_g(s) * theta\n"
        "        W_g(s) = theta * x_g(s) - 0.5 * x_g(s)^2 - F_g(s)\n"
        "    Choose g_star(s) = argmax_g W_g(s)\n"
        "\n"
        "Outputs: investment schedules, surplus schedules, and governance regions\n"
        "```\n\n"
        f"In this calibration, spot exchange wins for $s\\lesssim {regions[0][1]:.2f}$. "
        f"Long-term contracts win for ${regions[1][0]:.2f}\\lesssim s\\lesssim "
        f"{regions[1][1]:.2f}$. "
        f"Vertical integration wins for $s\\gtrsim {regions[2][0]:.2f}$. "
        "These thresholds come from surplus comparisons."
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for regime in REGIME_ORDER:
        gdf = df[df["Regime"] == regime]
        ax1.plot(gdf["Specificity"], gdf["Investment"], label=regime)
    ax1.axhline(efficient_investment, color="black", linestyle="--", label="Efficient investment")
    ax1.set_xlabel("Asset specificity $s$")
    ax1.set_ylabel("Investment $x_g(s)$")
    ax1.set_title("Investment Incentives and the First Best")
    ax1.legend()
    report.add_figure(
        "figures/investment-incentives.png",
        "Relationship-specific investment by governance regime",
        fig1,
        description="The dashed line is the first-best investment. Spot exchange falls "
        "quickly as specificity rises. Integration stays close to first best. Surplus "
        "decides whether higher investment is worth hierarchy cost.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for regime in REGIME_ORDER:
        gdf = df[df["Regime"] == regime]
        ax2.plot(gdf["Specificity"], gdf["Surplus"], label=regime)
    ax2.axhline(first_best_surplus, color="black", linestyle="--", label="First-best gross surplus")
    ax2.set_xlabel("Asset specificity $s$")
    ax2.set_ylabel("Surplus $W_g(s)$")
    ax2.set_title("Surplus Net of Governance Costs")
    ax2.legend()
    report.add_figure(
        "figures/surplus-by-regime.png",
        "Surplus by governance regime",
        fig2,
        description="Surplus changes with incentives and governance cost. Spot contracts "
        "win when redeployment is easy. Vertical integration wins when hold-up losses "
        "exceed hierarchy cost.",
    )

    regime_codes = {"Spot contract": 0, "Long-term contract": 1, "Vertical integration": 2}
    fig3, ax3 = plt.subplots(figsize=(8, 2.4))
    ax3.scatter(
        best["Specificity"],
        np.zeros(len(best)),
        c=best["Regime"].map(regime_codes),
        cmap="viridis",
        s=55,
    )
    ax3.set_yticks([])
    ax3.set_xlabel("Asset specificity $s$")
    ax3.set_title("Surplus-Maximizing Governance Form")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=plt.cm.viridis(v / 2), label=k)
        for k, v in regime_codes.items()
    ]
    ax3.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.25))
    report.add_figure(
        "figures/governance-regions.png",
        "Governance regions over asset specificity",
        fig3,
        description="These regions plot the surplus winner for each specificity value. "
        "Long-term contracts occupy the middle. They protect investment at lower cost "
        "than integration.",
    )

    selected = df[df["Specificity"].isin([0.0, 0.3, 0.5, 1.0])].copy()
    best_lookup = best.set_index("Specificity")["Regime"]
    selected["Chosen regime"] = [
        "yes" if best_lookup.loc[s] == regime else ""
        for s, regime in zip(selected["Specificity"], selected["Regime"])
    ]
    selected["Regime order"] = selected["Regime"].map({name: i for i, name in enumerate(REGIME_ORDER)})
    selected = selected.sort_values(["Specificity", "Regime order"]).drop(columns="Regime order")
    selected["Specificity"] = selected["Specificity"].map(lambda x: f"{x:.1f}")
    selected["Incentive share"] = selected["Incentive share"].map(lambda x: f"{x:.2f}")
    selected["Investment"] = selected["Investment"].map(lambda x: f"{x:.2f}")
    selected["Surplus"] = selected["Surplus"].map(lambda x: f"{x:.2f}")
    selected["Efficiency ratio"] = selected["Efficiency ratio"].map(lambda x: f"{100*x:.1f}%")
    report.add_table(
        "tables/governance-comparison.csv",
        "Governance comparison at selected levels of asset specificity",
        selected,
        description="The table shows the accounting at four specificity values. Low "
        "specificity favors market exchange. Middle values favor a contract. High "
        "specificity favors integration.",
    )

    report.add_takeaway(
        "Firm boundaries follow the hold-up tradeoff. Market exchange works when assets "
        "are easy to redeploy. Integration pays when stronger control rights offset "
        "hierarchy cost. Long-term contracts fill the middle range."
    )

    report.add_references([
        "Williamson, O. (1975). *Markets and Hierarchies*. Free Press.",
        "Grossman, S., and Hart, O. (1986). The Costs and Benefits of Ownership. *Journal of Political Economy*, 94(4), 691-719.",
        "Hart, O., and Moore, J. (1990). Property Rights and the Nature of the Firm. *Journal of Political Economy*, 98(6), 1119-1158.",
        "Lecture 6 Slides 2023: Theory of the Firm and incomplete contracts.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
