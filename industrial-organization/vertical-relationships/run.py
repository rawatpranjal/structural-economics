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
        "Resale price maintenance": outcome(monopoly_price, wholesale_dm, cm, cr, a, b),
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
        "Vertical Relationships and Double Marginalization",
        "How marginal wholesale prices distort a separated retail channel.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "The economic issue is not that a manufacturer and a retailer disagree about "
        "the monopoly price. They would both like the channel to sell the "
        "joint-profit-maximizing quantity. The problem is the instrument. With a "
        "linear wholesale price, the upstream margin becomes part of the retailer's "
        "marginal cost, so the downstream firm adds another margin on top of it. The "
        "channel then under-sells relative to an integrated firm.\n\n"
        "Demand is simple so the contract logic stays visible. A two-part tariff uses a "
        "low per-unit wholesale price to restore the downstream pricing incentive, then "
        "uses a fixed fee to move profit upstream. Resale price maintenance instead "
        "controls the retail price directly. These mechanisms are the pricing counterpart "
        "to the firm-boundary problem in [theory of the firm](../theory-of-the-firm/) and "
        "a simpler precursor to the assortment contracts in "
        "[vertical contracts](../vertical-contracts/)."
    )

    report.add_equations(r"""
Let demand be summarized by the linear quantity rule
$$q(p)=a-bp,\qquad p\leq \bar p\equiv a/b,$$
where $p$ is the retail price, $a$ is market size, $b$ is the demand slope, and
$\bar p$ is the choke price. The upstream marginal cost is $c_M$ and the
retailer's own marginal selling cost is $c_R$.

The integrated channel chooses $p$ to maximize
$$\Pi^I(p)=(p-c_M-c_R)q(p),$$
so the joint-profit price is
$$p^I=\frac{\bar p+c_M+c_R}{2}.$$

Under a linear wholesale contract, the manufacturer sets a per-unit wholesale
price $w$ and the retailer solves
$$\max_p\ (p-w-c_R)q(p).$$
The retailer's best response is
$$p_R(w)=\frac{\bar p+w+c_R}{2}.$$

The manufacturer anticipates that response and solves
$$\max_w\ (w-c_M)q(p_R(w)),$$
which gives
$$w^{DM}=\frac{\bar p-c_R+c_M}{2}.$$
Because $w^{DM}>c_M$, the retailer prices as if marginal cost is higher than
the true channel cost.

A two-part tariff replaces the high marginal wholesale price with
$$w^{TPT}=c_M$$
and lets the fixed fee
$$F=(p^I-c_M-c_R)q(p^I)$$
transfer the retailer's operating profit upstream. Resale price maintenance
sets the retail price at $p^I$ directly; in this calibration the wholesale price
can then be chosen so the downstream participation constraint binds.
""")

    report.add_model_setup(
        "The calibration is small enough to solve analytically. All dollar amounts "
        "below are in the same arbitrary unit, and the integrated channel is a "
        "benchmark rather than an observed ownership structure.\n\n"
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| $a$ | {a:.1f} | Demand intercept |\n"
        f"| $b$ | {b:.1f} | Demand slope |\n"
        f"| $\\bar p=a/b$ | {a / b:.1f} | Choke price |\n"
        f"| $c_M$ | {cm:.1f} | Manufacturer marginal cost |\n"
        f"| $c_R$ | {cr:.1f} | Retail service cost |\n"
        "| Contracts | 4 | Integrated benchmark, linear wholesale, two-part tariff, resale price maintenance |"
    )

    report.add_solution_method(
        "The solution is backward induction with closed-form first-order conditions. "
        "The computation is mainly accounting: evaluate each contract using the same "
        "demand curve, then compare price, quantity, channel profit, and consumer "
        "surplus to the integrated benchmark.\n\n"
        "```text\n"
        "Inputs: demand q(p)=a-bp, costs c_M and c_R, contract set K\n"
        "\n"
        "Integrated benchmark:\n"
        "    p_I = (a/b + c_M + c_R) / 2\n"
        "    q_I = q(p_I)\n"
        "\n"
        "Linear wholesale contract:\n"
        "    For any wholesale price w:\n"
        "        retailer best response p_R(w) = (a/b + w + c_R) / 2\n"
        "    Manufacturer chooses w_DM to maximize (w-c_M) q(p_R(w))\n"
        "    Evaluate p_R(w_DM), q(p_R(w_DM)), and surplus\n"
        "\n"
        "Counterfactual contracts:\n"
        "    Two-part tariff: set w=c_M, p=p_I, and fixed fee F=(p_I-c_M-c_R)q_I\n"
        "    Resale price maintenance: set p=p_I and choose w so retailer profit is zero\n"
        "\n"
        "Outputs: contract outcomes and pass-through curve p_R(w)\n"
        "```\n\n"
        "The integrated solution is the analytic ground truth for the channel's "
        "joint-profit problem. The figures use that benchmark to mark what is lost "
        "under the separated linear contract and what the alternative contracts "
        "restore."
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
        description="The separated linear contract is the outlier. The wholesale price "
        f"${linear['Wholesale price']:.2f}$ makes the retailer behave as if its marginal "
        "cost is well above the true channel cost, so price rises and quantity falls. "
        "The two-part tariff and the resale-price-maintenance counterfactual both put "
        "the retail price back on the integrated-channel line.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x, df["Consumer surplus"], label="Consumer surplus", color="#54A24B")
    ax2.bar(x, df["Channel profit"], bottom=df["Consumer surplus"], label="Channel profit", color="#E45756")
    ax2.axhline(benchmark["Total surplus"], color="black", linestyle="--", label="Integrated total surplus")
    ax2.set_xticks(x)
    ax2.set_xticklabels(contract_labels, rotation=0)
    ax2.set_ylabel("Dollars")
    ax2.set_title("Surplus Lost When the Channel Under-Sells")
    ax2.legend()
    report.add_figure(
        "figures/surplus-decomposition.png",
        "Consumer surplus and channel profit by contract",
        fig2,
        description="The transfer between upstream and downstream firms is not the welfare "
        "loss. The loss comes from the smaller quantity sold under the linear wholesale "
        "contract. Once the marginal wholesale price is neutralized, consumer surplus "
        "and channel profit both return to the integrated benchmark in this simple "
        "single-product environment.",
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
        description="The pass-through exercise varies only the per-unit wholesale price. "
        "Moving from $w=c_M$ to $w^{DM}$ traces the double-marginalization mechanism: "
        "the retailer's best response price rises mechanically with perceived marginal "
        "cost, and the quantity gap opens relative to the integrated benchmark.",
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
        description="The table separates efficiency from incidence. Channel profit and "
        "consumer surplus describe the real allocation. Manufacturer and retailer "
        "profits show how each contract allocates that profit; for the integrated "
        "benchmark the internal split is just a transfer convention.",
    )

    report.add_takeaway(
        "Distinguish marginal incentives from transfers. A high wholesale price changes "
        "the downstream pricing first-order condition, so it creates a real quantity "
        "distortion. A fixed fee moves profit without changing the retailer's marginal "
        "cost. Nonlinear pricing can therefore remove double marginalization while still "
        "letting the upstream firm extract the channel's profit."
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
