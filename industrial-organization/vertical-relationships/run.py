#!/usr/bin/env python3
"""Vertical relationships: double marginalization and vertical restraints."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def demand(price: float, a: float, b: float) -> float:
    return max(a - b * price, 0.0)


def outcome(price: float, wholesale: float, cm: float, cr: float, a: float, b: float) -> dict[str, float]:
    q = demand(price, a, b)
    manufacturer_profit = (wholesale - cm) * q
    retailer_profit = (price - wholesale - cr) * q
    consumer_surplus = 0.5 * q * (a / b - price)
    return {
        "Retail price": price,
        "Wholesale price": wholesale,
        "Quantity": q,
        "Manufacturer profit": manufacturer_profit,
        "Retailer profit": retailer_profit,
        "Channel profit": manufacturer_profit + retailer_profit,
        "Consumer surplus": consumer_surplus,
        "Total surplus": manufacturer_profit + retailer_profit + consumer_surplus,
    }


def solve_contracts(a: float, b: float, cm: float, cr: float) -> pd.DataFrame:
    total_cost = cm + cr
    monopoly_price = (a + b * total_cost) / (2 * b)

    wholesale_dm = (a / b - cr + cm) / 2
    retail_dm = (a + b * (wholesale_dm + cr)) / (2 * b)

    two_part_w = cm
    two_part_price = monopoly_price
    two_part = outcome(two_part_price, two_part_w, cm, cr, a, b)
    two_part_fee = two_part["Retailer profit"]
    two_part["Manufacturer profit"] += two_part_fee
    two_part["Retailer profit"] -= two_part_fee
    two_part["Channel profit"] = two_part["Manufacturer profit"] + two_part["Retailer profit"]

    contracts = {
        "Integrated channel": outcome(monopoly_price, cm, cm, cr, a, b),
        "Linear wholesale": outcome(retail_dm, wholesale_dm, cm, cr, a, b),
        "Two-part tariff": two_part,
        "RPM at monopoly price": outcome(monopoly_price, wholesale_dm, cm, cr, a, b),
    }

    rows = []
    for name, values in contracts.items():
        row = {"Contract": name}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)


def wholesale_pass_through(a: float, b: float, cr: float) -> pd.DataFrame:
    ws = np.linspace(0.5, 6.0, 80)
    rows = []
    for w in ws:
        p = (a + b * (w + cr)) / (2 * b)
        rows.append({"Wholesale price": w, "Retail price": p, "Quantity": demand(p, a, b)})
    return pd.DataFrame(rows)


def main() -> None:
    setup_style()
    a, b, cm, cr = 20.0, 2.0, 2.0, 1.0
    df = solve_contracts(a, b, cm, cr)
    pass_df = wholesale_pass_through(a, b, cr)

    print("Vertical relationships tutorial")
    print(df[["Contract", "Retail price", "Quantity", "Channel profit"]].to_string(index=False))

    report = ModelReport(
        "Vertical Relationships and Double Marginalization",
        "Wholesale pricing, retail pricing, two-part tariffs, and simple vertical restraints.",
    )

    report.add_overview(
        "A vertically separated manufacturer and retailer can both have market power. "
        "With a linear wholesale price, the retailer treats the wholesale price as a "
        "marginal cost and adds its own margin on top. The result is double "
        "marginalization: price is too high and quantity is too low relative to the "
        "joint-profit benchmark.\n\n"
        "Vertical contracts can repair this distortion. A two-part tariff sets the "
        "per-unit wholesale price close to marginal cost and uses a fixed fee to "
        "transfer surplus. Resale price maintenance can also target the retail price, "
        "though its welfare effect depends on the environment."
    )

    report.add_equations(r"""
Retail demand is linear:
$$q(p) = a - bp$$

Given wholesale price $w$, the retailer chooses:
$$p_R(w) = \frac{a + b(w+c_R)}{2b}$$

The manufacturer anticipates that response and chooses:
$$w^{*} = \frac{a/b - c_R + c_M}{2}$$

The integrated channel sets:
$$p^I = \frac{a + b(c_M+c_R)}{2b}$$
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| $a$ | {a:.1f} | Demand intercept |\n"
        f"| $b$ | {b:.1f} | Demand slope |\n"
        f"| $c_M$ | {cm:.1f} | Manufacturer marginal cost |\n"
        f"| $c_R$ | {cr:.1f} | Retail service cost |\n"
        "| Contracts | 4 | Integration, linear wholesale, two-part tariff, RPM |"
    )

    report.add_solution_method(
        "The model is solved by backward induction. The retailer first solves its "
        "static pricing problem for any wholesale price. The manufacturer then chooses "
        "the wholesale price anticipating the retailer response. Counterfactual "
        "contracts change either the marginal wholesale price, the fixed transfer, or "
        "the retail price target."
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    ax1.bar(x - 0.18, df["Retail price"], 0.36, label="Retail price", color="#4C78A8")
    ax1.bar(x + 0.18, df["Quantity"], 0.36, label="Quantity", color="#F58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Contract"], rotation=20, ha="right")
    ax1.set_title("Double Marginalization Raises Price and Lowers Quantity")
    ax1.legend()
    report.add_figure(
        "figures/price-quantity.png",
        "Price and quantity by vertical contract",
        fig1,
        description="The linear wholesale contract produces the highest retail price. "
        "The two-part tariff restores the integrated-channel price by removing the "
        "manufacturer's margin from the retailer's marginal cost.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x, df["Consumer surplus"], label="Consumer surplus", color="#54A24B")
    ax2.bar(x, df["Channel profit"], bottom=df["Consumer surplus"], label="Channel profit", color="#E45756")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Contract"], rotation=20, ha="right")
    ax2.set_ylabel("Dollars")
    ax2.set_title("Surplus Decomposition")
    ax2.legend()
    report.add_figure(
        "figures/surplus-decomposition.png",
        "Consumer surplus and channel profit by contract",
        fig2,
        description="Double marginalization destroys surplus. A two-part tariff improves "
        "channel profit and consumer surplus in this simple environment because it lowers "
        "the marginal wholesale price.",
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(pass_df["Wholesale price"], pass_df["Retail price"], label="Retail price")
    ax3.plot(pass_df["Wholesale price"], pass_df["Quantity"], label="Quantity")
    ax3.set_xlabel("Wholesale price")
    ax3.set_title("Retail Pass-Through of Wholesale Prices")
    ax3.legend()
    report.add_figure(
        "figures/wholesale-pass-through.png",
        "Retail pass-through as wholesale price changes",
        fig3,
        description="A higher wholesale price shifts the retailer's perceived marginal cost. "
        "Only part of the increase is passed through to price, but quantity still falls.",
    )

    table = df.copy()
    for col in table.columns:
        if col != "Contract":
            table[col] = table[col].map(lambda v: f"{v:.2f}")
    report.add_table("tables/vertical-contracts.csv", "Contract outcomes", table)

    report.add_takeaway(
        "The central IO lesson is marginal incentives. A fixed fee only transfers surplus, "
        "while a high wholesale price changes the retailer's marginal pricing decision. "
        "That is why nonlinear pricing can eliminate double marginalization even when the "
        "manufacturer still extracts profit."
    )

    report.add_references([
        "Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press.",
        "Asker, J. (2016). Diagnosing Foreclosure Due to Exclusive Dealing. *Journal of Industrial Economics*, 64(3), 375-410.",
        "Lecture 7 Slides 2023: Vertical relationships, double marginalization, and restraints.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
