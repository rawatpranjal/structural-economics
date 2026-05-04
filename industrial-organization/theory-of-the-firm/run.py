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
    region_text = (
        f"{regions[0][2]} for $s\\lesssim {regions[0][1]:.2f}$; "
        f"{regions[1][2]} for ${regions[1][0]:.2f}\\lesssim s\\lesssim {regions[1][1]:.2f}$; "
        f"{regions[2][2]} for $s\\gtrsim {regions[2][0]:.2f}$"
    )

    print("Theory of the firm tutorial")
    print(best.groupby("Regime").size().to_string())

    report = ModelReport(
        "Theory of the Firm: Incomplete Contracts and Hold-Up",
        "Asset specificity, relationship-specific investment, and the vertical integration boundary.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A firm boundary question is usually a contracting question before it is an "
        "organizational chart question. Suppose a supplier can make an investment that "
        "is valuable mainly inside one buyer-seller relationship. If the investment is "
        "hard to describe in a court-enforceable contract, the supplier expects some of "
        "the return to be bargained away after the sunk cost has been paid. The hold-up "
        "problem is then not that trade fails mechanically; it is that the investment "
        "made before trade is too small.\n\n"
        "The tutorial uses one state variable, asset specificity $s\\in[0,1]$, to make "
        "that tradeoff explicit. Spot exchange is cheap but offers weak protection when "
        "$s$ is high. A long-term contract protects more of the investment return, at a "
        "drafting and monitoring cost. Vertical integration gives stronger residual "
        "control rights, but hierarchy itself is costly. The exercise asks where each "
        "governance form maximizes total surplus.\n\n"
        "This is the firm-boundary counterpart to the downstream-pricing examples in "
        "[vertical relationships](../vertical-relationships/) and "
        "[Bertrand pricing with logit demand](../bertrand-logit-demand/): here the "
        "object is not a price equilibrium, but the allocation of control rights before "
        "relationship-specific investment is sunk."
    )

    report.add_equations(r"""
Let $s$ denote asset specificity and let $g\in\mathcal G$ index governance
regimes: spot exchange, a long-term contract, and vertical integration.
Relationship-specific investment $x$ creates gross value
$$V(x) = \theta x - \frac{1}{2}x^2$$

so the first-best investment, before contracting frictions, is
$$x^{*} = \theta$$

Under regime $g$, the investor internalizes only share $b_g(s)$ of the marginal
return. The private first-order condition is
$$b_g(s)\theta - x = 0,$$
which gives the regime-specific investment rule
$$x_g(s) = b_g(s)\theta$$

Total surplus nets out the governance cost $F_g(s)$:
$$W_g(s) = \theta x_g(s) - \frac{1}{2}x_g(s)^2 - F_g(s)$$

The selected governance form is
$$g^{*}(s)=\arg\max_{g\in\mathcal G} W_g(s).$$

The first-best surplus line used in the figures is
$$W^{*}=\frac{1}{2}\theta^2,$$
which is a benchmark, not an attainable governance regime once hold-up and
governance costs are present.
""")

    report.add_model_setup(
        "The calibration is deliberately stylized. It is not estimating a boundary "
        "of the firm from data; it is making the Williamson/Grossman-Hart-Moore "
        "comparative static visible with transparent primitives.\n\n"
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
        "The computation is a regime comparison on a grid for $s$. There is no dynamic "
        "programming or equilibrium fixed point here; the useful discipline is to keep "
        "private investment incentives separate from total surplus. Integration can "
        "raise $x_g(s)$ and still fail to maximize $W_g(s)$ if its governance cost is "
        "too high.\n\n"
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
        "In this calibration the approximate surplus-maximizing regions are: "
        f"{region_text}. The switch points are not parameters of the model; they are "
        "the outcome of comparing the incentive gains from stronger control rights "
        "with the resource costs of writing contracts or running hierarchy."
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
        description="The dashed line is the first-best investment $x^{*}=\\theta$. "
        "Spot exchange loses investment incentives as specificity rises because the "
        "investor expects more ex-post bargaining. Integration keeps investment close "
        "to the benchmark, but the next figure shows why that alone does not settle "
        "the firm-boundary question.",
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
        description="The surplus ranking changes because each governance form moves two "
        "objects at once: investment incentives and governance cost. Spot contracts are "
        "best when assets are easy to redeploy. Vertical integration becomes attractive "
        "only after the hold-up cost of market exchange dominates the internal cost of "
        "hierarchy.",
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
        description="The governance regions summarize the same comparison without the "
        "surplus levels. The middle interval is useful: a long-term contract can dominate "
        "both spot exchange and integration when it protects enough investment without "
        "bringing the full internal governance cost.",
    )

    selected = df[df["Specificity"].isin([0.0, 0.5, 1.0])].copy()
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
        "Governance comparison at low, medium, and high asset specificity",
        selected,
        description="The table keeps the accounting transparent at three values of $s$. "
        "At low specificity, cheap market exchange wins despite underinvestment. At "
        "medium and high specificity in this calibration, integration's incentive effect "
        "is large enough to offset its governance cost.",
    )

    report.add_takeaway(
        "The firm boundary is not a generic preference for hierarchy. Integration is "
        "valuable when noncontractible, relationship-specific investment is important "
        "enough that stronger control rights pay for themselves. When assets are easy "
        "to redeploy, market exchange can dominate because it avoids the internal "
        "costs of hierarchy. Between those cases, a long-term contract can be the "
        "surplus-maximizing compromise."
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
