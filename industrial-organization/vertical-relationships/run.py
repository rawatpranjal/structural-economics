#!/usr/bin/env python3
"""Double marginalization in a manufacturer-retailer channel."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def demand(price: float, a: float, b: float) -> float:
    return max(a - b * price, 0.0)


def outcome(
    price: float,
    wholesale: float,
    cm: float,
    cr: float,
    a: float,
    b: float,
    fixed_fee: float = 0.0,
) -> dict[str, float]:
    q = demand(price, a, b)
    manufacturer_profit = (wholesale - cm) * q + fixed_fee
    retailer_profit = (price - wholesale - cr) * q - fixed_fee
    consumer_surplus = 0.5 * q * (a / b - price)
    return {
        "Retail price": price,
        "Wholesale price": wholesale,
        "Fixed fee": fixed_fee,
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
    two_part_before_fee = outcome(monopoly_price, two_part_w, cm, cr, a, b)
    two_part_fee = two_part_before_fee["Retailer profit"]
    two_part = outcome(monopoly_price, two_part_w, cm, cr, a, b, fixed_fee=two_part_fee)

    contracts = {
        "Integrated benchmark": outcome(monopoly_price, cm, cm, cr, a, b),
        "Linear wholesale": outcome(retail_dm, wholesale_dm, cm, cr, a, b),
        "Two-part tariff": two_part,
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
    benchmark = df[df["Contract"] == "Integrated benchmark"].iloc[0]
    linear = df[df["Contract"] == "Linear wholesale"].iloc[0]
    contract_labels = [label.replace(" ", "\n") for label in df["Contract"]]

    print("Vertical relationships tutorial")
    print(df[["Contract", "Retail price", "Quantity", "Channel profit"]].to_string(index=False))

    x = np.arange(len(df))
    fig1, axes1 = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True)
    axes1[0].bar(x, df["Retail price"], color="#4C78A8")
    axes1[0].axhline(benchmark["Retail price"], color="black", linestyle="--", label="Integrated benchmark")
    axes1[0].set_ylabel("Retail price")
    axes1[0].set_title("Price")
    axes1[0].legend()
    axes1[1].bar(x, df["Quantity"], color="#F58518")
    axes1[1].axhline(benchmark["Quantity"], color="black", linestyle="--", label="Integrated benchmark")
    axes1[1].set_ylabel("Quantity")
    axes1[1].set_title("Quantity")
    axes1[1].legend()
    for ax in axes1:
        ax.set_xticks(x)
        ax.set_xticklabels(contract_labels, rotation=0)
    fig1.suptitle("Double Marginalization Against the Integrated Benchmark")
    fig1.tight_layout()
    save_figure(fig1, "figures/price-quantity.png", dpi=150)

    fig3, axes3 = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True)
    axes3[0].plot(pass_df["Wholesale price"], pass_df["Retail price"], label="$p_R(w)$")
    axes3[0].axhline(benchmark["Retail price"], color="black", linestyle="--", label="$p^I$")
    axes3[0].axvline(cm, color="#54A24B", linestyle=":", label="$w=c_M$")
    axes3[0].axvline(linear["Wholesale price"], color="#E45756", linestyle=":", label="$w^{DM}$")
    axes3[0].set_ylabel("Retail price")
    axes3[0].set_xlabel("Wholesale price")
    axes3[0].set_title("Retail best response")
    axes3[0].legend()
    axes3[1].plot(pass_df["Wholesale price"], pass_df["Quantity"], color="#F58518", label="$q(p_R(w))$")
    axes3[1].axhline(benchmark["Quantity"], color="black", linestyle="--", label="$q^I$")
    axes3[1].axvline(cm, color="#54A24B", linestyle=":", label="$w=c_M$")
    axes3[1].axvline(linear["Wholesale price"], color="#E45756", linestyle=":", label="$w^{DM}$")
    axes3[1].set_ylabel("Quantity")
    axes3[1].set_xlabel("Wholesale price")
    axes3[1].set_title("Quantity sold")
    axes3[1].legend()
    fig3.suptitle("Wholesale Prices Work Through Retail Marginal Cost")
    fig3.tight_layout()
    save_figure(fig3, "figures/wholesale-pass-through.png", dpi=150)

    table = df[
        [
            "Contract",
            "Retail price",
            "Wholesale price",
            "Fixed fee",
            "Quantity",
            "Channel profit",
            "Consumer surplus",
            "Total surplus",
            "Manufacturer profit",
            "Retailer profit",
        ]
    ].copy()
    for col in table.columns:
        if col != "Contract":
            table[col] = table[col].map(lambda v: f"{v:.2f}")
    Path("tables/vertical-contracts.csv").parent.mkdir(parents=True, exist_ok=True)
    table.to_csv("tables/vertical-contracts.csv", index=False)

    save_thumbnail("figures/price-quantity.png", "figures/thumb.png")
    print("Figures and tables written.")


if __name__ == "__main__":
    main()
