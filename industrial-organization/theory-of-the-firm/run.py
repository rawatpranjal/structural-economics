#!/usr/bin/env python3
"""Firm boundaries, incomplete contracts, hold-up, and integration."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


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

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for regime in REGIME_ORDER:
        gdf = df[df["Regime"] == regime]
        ax1.plot(gdf["Specificity"], gdf["Investment"], label=regime)
    ax1.axhline(efficient_investment, color="black", linestyle="--", label="Efficient investment")
    ax1.set_xlabel("Asset specificity $s$")
    ax1.set_ylabel("Investment $x_g(s)$")
    ax1.set_title("Investment Incentives and the First Best")
    ax1.legend()
    save_figure(fig1, "figures/investment-incentives.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for regime in REGIME_ORDER:
        gdf = df[df["Regime"] == regime]
        ax2.plot(gdf["Specificity"], gdf["Surplus"], label=regime)
    ax2.axhline(first_best_surplus, color="black", linestyle="--", label="First-best gross surplus")
    ax2.set_xlabel("Asset specificity $s$")
    ax2.set_ylabel("Surplus $W_g(s)$")
    ax2.set_title("Surplus Net of Governance Costs")
    ax2.legend()
    save_figure(fig2, "figures/surplus-by-regime.png", dpi=150)

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
    save_figure(fig3, "figures/governance-regions.png", dpi=150)

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
    Path("tables/governance-comparison.csv").parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv("tables/governance-comparison.csv", index=False)

    save_thumbnail("figures/investment-incentives.png", "figures/thumb.png")
    print(f"Figures and tables written. Regions: {regions}")


if __name__ == "__main__":
    main()
