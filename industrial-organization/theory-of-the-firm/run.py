#!/usr/bin/env python3
"""Theory of the Firm: incomplete contracts, hold-up, and integration."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


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
    efficient_surplus = 0.5 * theta**2
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
                "Efficiency ratio": val / efficient_surplus,
            })
    return pd.DataFrame(rows)


def best_regime(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("Specificity")["Surplus"].idxmax()
    return df.loc[idx].sort_values("Specificity")


def main() -> None:
    setup_style()
    specificity = np.linspace(0.0, 1.0, 101)
    df = investment_outcomes(specificity)
    best = best_regime(df)
    efficient_investment = 4.0

    print("Theory of the firm tutorial")
    print(best.groupby("Regime").size().to_string())

    report = ModelReport(
        "Theory of the Firm: Incomplete Contracts and Hold-Up",
        "Asset specificity, relationship-specific investment, and the vertical integration boundary.",
    )

    report.add_overview(
        "Incomplete contracts matter when parties must invest before all future "
        "contingencies can be written down and enforced. If an investment is specific "
        "to one trading relationship, the investing party expects to be held up in "
        "ex-post bargaining and invests too little. Vertical integration reallocates "
        "residual control rights. It can strengthen investment incentives, but it also "
        "brings governance and adaptation costs.\n\n"
        "This tutorial turns that logic into a small numerical model. The state is asset "
        "specificity: low values describe transactions that can move through spot markets, "
        "high values describe assets whose value is tied to one buyer-seller pair."
    )

    report.add_equations(r"""
Investment $x$ raises relationship value:
$$V(x) = \theta x - \frac{1}{2}x^2$$

The efficient investment satisfies:
$$x^* = \theta$$

Under regime $g$, the investor captures only share $b_g(s)$ of marginal returns:
$$x_g(s) = b_g(s)\theta$$

Total surplus subtracts governance cost $F_g(s)$:
$$W_g(s) = \theta x_g(s) - \frac{1}{2}x_g(s)^2 - F_g(s)$$
""")

    report.add_model_setup(
        "| Object | Interpretation |\n"
        "|--------|----------------|\n"
        "| Specificity $s$ | Share of investment value locked into one relationship |\n"
        "| Spot contract | Low governance cost, weak protection against hold-up |\n"
        "| Long-term contract | Better protection, moderate drafting and monitoring cost |\n"
        "| Vertical integration | Stronger control rights, higher bureaucracy cost |\n"
        "| $\\theta=4$ | Efficient investment benchmark $x^*=4$ |"
    )

    report.add_solution_method(
        "For each value of specificity, the script computes the private investment "
        "implied by each governance regime and then evaluates total surplus. The "
        "preferred regime is the one with the highest surplus, not the one with the "
        "largest investment mechanically."
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for regime, gdf in df.groupby("Regime"):
        ax1.plot(gdf["Specificity"], gdf["Investment"], label=regime)
    ax1.axhline(efficient_investment, color="black", linestyle="--", label="Efficient investment")
    ax1.set_xlabel("Asset specificity")
    ax1.set_ylabel("Relationship-specific investment")
    ax1.set_title("Hold-Up Reduces Investment as Specificity Rises")
    ax1.legend()
    report.add_figure(
        "figures/investment-incentives.png",
        "Relationship-specific investment by governance regime",
        fig1,
        description="Spot contracts provide weak protection when assets are highly specific. "
        "Integration keeps investment closer to the efficient benchmark at high specificity, "
        "but it is not free.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for regime, gdf in df.groupby("Regime"):
        ax2.plot(gdf["Specificity"], gdf["Surplus"], label=regime)
    ax2.set_xlabel("Asset specificity")
    ax2.set_ylabel("Total surplus")
    ax2.set_title("Governance Tradeoff")
    ax2.legend()
    report.add_figure(
        "figures/surplus-by-regime.png",
        "Surplus by governance regime",
        fig2,
        description="The best governance form changes with specificity because incentive "
        "benefits and governance costs move in opposite directions.",
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
    ax3.set_xlabel("Asset specificity")
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
        description="The model reproduces the Williamson/GHM comparative static: integration "
        "is most attractive where relationship-specific investment is hardest to protect "
        "with ordinary contracts.",
    )

    selected = df[df["Specificity"].isin([0.0, 0.5, 1.0])].copy()
    selected["Specificity"] = selected["Specificity"].map(lambda x: f"{x:.1f}")
    selected["Incentive share"] = selected["Incentive share"].map(lambda x: f"{x:.2f}")
    selected["Investment"] = selected["Investment"].map(lambda x: f"{x:.2f}")
    selected["Surplus"] = selected["Surplus"].map(lambda x: f"{x:.2f}")
    selected["Efficiency ratio"] = selected["Efficiency ratio"].map(lambda x: f"{100*x:.1f}%")
    report.add_table(
        "tables/governance-comparison.csv",
        "Governance comparison at low, medium, and high asset specificity",
        selected,
    )

    report.add_takeaway(
        "The firm boundary is not a generic preference for hierarchy. Integration is "
        "useful when incomplete contracts and asset specificity make hold-up severe. "
        "When assets are easy to redeploy, ordinary market contracts or longer-term "
        "contracts can dominate because they avoid the internal governance cost."
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
