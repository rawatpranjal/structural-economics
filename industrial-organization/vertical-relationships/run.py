#!/usr/bin/env python3
"""Double marginalization in a manufacturer-retailer channel."""
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

    report = ModelReport(
        "Double Marginalization in Vertical Supply Chains",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A manufacturer sells through an independent retailer. The manufacturer sets "
        "wholesale terms. The retailer sets the shelf price. Consumers buy from the "
        "retailer.\n\n"
        "The object is double marginalization. A linear wholesale price makes the "
        "retailer treat the upstream markup as marginal cost. The retailer then adds "
        "its own markup, so the channel charges too much and sells too little.\n\n"
        "The computation solves the integrated channel and the separated game. "
        "Backward induction gives the wholesale price, retail price, quantity, and "
        "profits under each contract."
    )

    report.add_equations(r"""
Demand follows
$$q(p)=a-bp,\qquad p\leq \bar p\equiv a/b,$$
where $p$ is the retail price. The choke price is $\bar p$. Costs are $c_M$
upstream and $c_R$ downstream.

The integrated channel solves
$$\Pi^I(p)=(p-c_M-c_R)q(p),$$
so the joint-profit price is
$$p^I=\frac{\bar p+c_M+c_R}{2}.$$

Under a linear wholesale price $w$, the retailer solves
$$\max_p\ (p-w-c_R)q(p).$$
Its best response is
$$p_R(w)=\frac{\bar p+w+c_R}{2}.$$

The manufacturer chooses $w$ while anticipating that response:
$$\max_w\ (w-c_M)q(p_R(w)),$$
which gives
$$w^{DM}=\frac{\bar p-c_R+c_M}{2}.$$

Because $w^{DM}>c_M$, the retailer acts as if marginal cost is too high.

A two-part tariff sets
$$w^{TPT}=c_M$$
and uses the fixed fee
$$F=(p^I-c_M-c_R)q(p^I)$$
to transfer operating profit upstream.

The fee changes the profit split without changing the retailer's margin.
""")

    report.add_model_setup(
        "The calibration is small enough to solve analytically. Each number uses the "
        "same unit. The integrated channel is a benchmark, not an ownership "
        "assumption.\n\n"
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| $a$ | {a:.1f} | Demand intercept |\n"
        f"| $b$ | {b:.1f} | Demand slope |\n"
        f"| $\\bar p=a/b$ | {a / b:.1f} | Choke price |\n"
        f"| $c_M$ | {cm:.1f} | Manufacturer marginal cost |\n"
        f"| $c_R$ | {cr:.1f} | Retail service cost |\n"
        "| Contracts | 3 | Integrated benchmark, linear wholesale, two-part tariff |"
    )

    report.add_solution_method(
        "The solution follows the order of moves. Each step uses the same demand "
        "curve, so price and quantity are comparable across contracts.\n\n"
        "```text\n"
        "Inputs: demand q(p)=a-bp, costs c_M and c_R\n"
        "\n"
        "1. Integrated channel\n"
        "    p_I = (a/b + c_M + c_R) / 2\n"
        "    q_I = q(p_I)\n"
        "\n"
        "2. Linear wholesale game\n"
        "    Retailer best response: p_R(w) = (a/b + w + c_R) / 2\n"
        "    Manufacturer FOC for max (w-c_M) q(p_R(w)) has the closed-form\n"
        "        solution w_DM = (a/b - c_R + c_M) / 2, evaluated directly\n"
        "    Evaluate p_R(w_DM), q(p_R(w_DM)), profits, and surplus\n"
        "\n"
        "3. Two-part tariff\n"
        "    Two-part tariff: set w=c_M, p=p_I, and fixed fee F=(p_I-c_M-c_R)q_I\n"
        "\n"
        "Outputs: contract outcomes and pass-through curve p_R(w)\n"
        "```\n\n"
        "The comparison treats the fixed fee as a transfer. It changes the profit "
        "split, not the retailer's marginal cost."
    )

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
    report.add_figure(
        "figures/price-quantity.png",
        "Price and quantity by vertical contract",
        fig1,
        description="The integrated channel charges "
        f"${benchmark['Retail price']:.2f}$ and sells {benchmark['Quantity']:.1f} units. "
        "Linear wholesale pricing raises the retail price to "
        f"${linear['Retail price']:.2f}$ and cuts quantity to {linear['Quantity']:.1f}. "
        "The two-part tariff returns price and quantity to the integrated line.",
    )

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
    report.add_figure(
        "figures/wholesale-pass-through.png",
        "Retail pass-through as wholesale price changes",
        fig3,
        description="The wholesale-price sweep varies only $w$. The retailer's best "
        "response price rises with $w$. At $w^{DM}$, quantity falls below the "
        "integrated benchmark.",
    )

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
    report.add_table(
        "tables/vertical-contracts.csv",
        "Contract outcomes",
        table,
        description="The table reports the same comparison in numbers. Channel profit "
        "and consumer surplus fall only when quantity falls. The fixed fee moves "
        "operating profit upstream.",
    )

    report.add_takeaway(
        "Double marginalization comes from the retailer's perceived marginal cost. "
        "A high wholesale price raises that cost and lowers quantity. A two-part "
        "tariff sets $w=c_M$, so the retailer chooses the integrated price. The "
        "fixed fee then allocates profit."
    )

    report.add_references([
        "Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
