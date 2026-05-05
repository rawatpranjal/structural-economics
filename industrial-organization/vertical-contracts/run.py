#!/usr/bin/env python3
"""Vertical contracts and capacity-constrained vending assortments."""
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


def product_catalog() -> pd.DataFrame:
    """Small product universe for the vending-machine assortment problem."""
    return pd.DataFrame({
        "Product": [
            "Choco Bar", "Caramel Cup", "Peanut Bites", "Mint Chew", "Nougat",
            "Fruit Chews", "Granola", "Trail Mix", "Protein Bite", "Cookie",
            "Pretzels", "Gum",
        ],
        "Manufacturer": [
            "Mars", "Mars", "Mars", "Mars", "Mars",
            "Rival", "Rival", "Rival", "Rival", "Rival", "Rival", "Rival",
        ],
        "Demand intercept": [13.2, 12.5, 11.8, 9.5, 8.8, 12.2, 10.5, 9.8, 9.4, 8.7, 7.8, 7.0],
        "Marginal cost": [0.58, 0.55, 0.52, 0.48, 0.50, 0.54, 0.62, 0.65, 0.70, 0.50, 0.42, 0.35],
    })


def retail_outcome(intercept: float, wholesale: float, slope: float = 4.0) -> tuple[float, float]:
    """Interior monopoly price under linear demand, with a small margin floor."""
    price = max((intercept + slope * wholesale) / (2 * slope), wholesale + 0.05)
    quantity = max(intercept - slope * price, 0.0)
    return price, quantity


def evaluate_subset(products: pd.DataFrame, subset: tuple[int, ...], contract: str, threshold: int = 4) -> dict[str, float]:
    """Evaluate a retailer-selected assortment under one vertical contract."""
    mars_count = int((products.loc[list(subset), "Manufacturer"] == "Mars").sum())
    rows = []
    for idx in subset:
        row = products.loc[idx]
        is_mars = row["Manufacturer"] == "Mars"
        wholesale = row["Marginal cost"] + 0.42
        fixed_fee = 0.0
        if contract == "All-unit discount" and is_mars and mars_count >= threshold:
            wholesale -= 0.18
        if contract == "Slotting fees":
            fixed_fee = 1.10 if is_mars else 0.35
        price, quantity = retail_outcome(row["Demand intercept"], wholesale)
        retailer_var = (price - wholesale) * quantity
        upstream_var = (wholesale - row["Marginal cost"]) * quantity - fixed_fee
        rows.append((idx, price, quantity, retailer_var, upstream_var, fixed_fee))
    arr = pd.DataFrame(rows, columns=[
        "Index", "Price", "Quantity", "Retailer variable profit",
        "Upstream profit", "Fixed fee",
    ])
    return {
        "Retailer objective": float(arr["Retailer variable profit"].sum() + arr["Fixed fee"].sum()),
        "Retailer variable profit": float(arr["Retailer variable profit"].sum()),
        "Upstream profit": float(arr["Upstream profit"].sum()),
        "Fixed fees": float(arr["Fixed fee"].sum()),
        "Average price": float(arr["Price"].mean()),
        "Total quantity": float(arr["Quantity"].sum()),
        "Mars slots": mars_count,
    }


def choose_assortment(products: pd.DataFrame, capacity: int, contract: str, threshold: int = 4) -> tuple[tuple[int, ...], dict[str, float]]:
    """Find the exact finite-set optimum by checking every feasible assortment."""
    best_subset: tuple[int, ...] | None = None
    best_values: dict[str, float] | None = None
    for subset in itertools.combinations(products.index, capacity):
        values = evaluate_subset(products, subset, contract, threshold)
        if best_values is None or values["Retailer objective"] > best_values["Retailer objective"]:
            best_subset = subset
            best_values = values
    assert best_subset is not None and best_values is not None
    return best_subset, best_values


def main() -> None:
    setup_style()
    products = product_catalog()
    capacity = 7
    contracts = ["Wholesale only", "All-unit discount", "Slotting fees"]

    outcomes = []
    selected = {}
    for contract in contracts:
        subset, values = choose_assortment(products, capacity, contract)
        selected[contract] = subset
        row = {"Contract": contract}
        row.update(values)
        outcomes.append(row)
    outcome_df = pd.DataFrame(outcomes)

    threshold_rows = []
    for threshold in range(1, capacity + 1):
        subset, values = choose_assortment(products, capacity, "All-unit discount", threshold=threshold)
        threshold_rows.append({
            "Threshold": threshold,
            "Mars slots": values["Mars slots"],
            "Retailer objective": values["Retailer objective"],
            "Upstream profit": values["Upstream profit"],
            "Selected products": ", ".join(products.loc[list(subset), "Product"]),
        })
    threshold_df = pd.DataFrame(threshold_rows)

    print("Vertical contracts tutorial")
    print(outcome_df[["Contract", "Mars slots", "Retailer objective", "Upstream profit"]].to_string(index=False))

    report = ModelReport(
        "Vertical Contracts and Vending Assortments",
        "All-unit discounts, slotting fees, and assortment choice in a capacity-constrained vending channel.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Vertical-contract evidence often starts with availability, not only with prices. "
        "A vending operator has a small number of slots, the upstream firms want their "
        "products inside the machine, and contract terms decide how attractive each slot "
        "is to the downstream firm.\n\n"
        "Demand is small. Each product has linear retail demand and a wholesale cost. The "
        "economic action is the retailer's assortment choice: seven products must be "
        "selected from a larger catalog. The benchmark in "
        "[Vertical Relationships and Double Marginalization](../vertical-relationships/) "
        "shows how wholesale margins distort retail prices; here the same vertical logic "
        "is pushed into product availability through rebates and slotting fees."
    )

    report.add_equations(r"""
Let $\mathcal J$ be the product catalog and let $K$ be the number of vending
slots. Product $j$ has demand intercept $a_j$, marginal cost $c_j$, and
manufacturer label $m(j)$. Retail demand is separable:
$$
q_j(p_j)=\max\{a_j-bp_j,0\}.
$$

Given wholesale price $w_j$, the retailer sets the product price by
$$
p_j^{*}(w_j)
=\arg\max_{p_j\geq w_j} (p_j-w_j)q_j(p_j).
$$
For an interior product, this gives
$$
p_j^{*}(w_j)=\frac{a_j+bw_j}{2b}.
$$

Contract $C$ maps an assortment $A$ into a wholesale price
$w_j^C(A)$ and a fixed transfer $F_j^C(A)$ paid by the upstream side to the
retailer. The retailer chooses
$$
A_C^{*}
=\arg\max_{A\subset\mathcal J:\ |A|=K}
\sum_{j\in A}\left[(p_j^{*}-w_j^C(A))q_j(p_j^{*})+F_j^C(A)\right].
$$

The upstream-profit diagnostic is
$$
\Pi_C^U(A)=
\sum_{j\in A}\left[(w_j^C(A)-c_j)q_j(p_j^{*})-F_j^C(A)\right].
$$

In the all-unit discount case, the retailer pays a lower wholesale price on
Mars products only if the selected assortment contains at least $\tau$ Mars
products:
$$
w_j^C(A)=c_j+\mu-d\,\mathbf 1\{m(j)=\text{Mars}\}\mathbf 1\{M(A)\geq\tau\},
\quad
M(A)=\sum_{j\in A}\mathbf 1\{m(j)=\text{Mars}\}.
$$
Slotting fees instead leave wholesale margins unchanged and work through
$F_j^C(A)$.
""")

    report.add_model_setup(
        "The primitives are chosen so the finite assortment problem can be solved exactly. "
        "The retailer optimizes its own payoff; upstream profit is reported only to show "
        "which side bears the cost of the contract.\n\n"
        "| Object | Value |\n"
        "|--------|-------|\n"
        f"| Products | {len(products)} candy and snack alternatives |\n"
        f"| Machine capacity | {capacity} slots |\n"
        "| Demand | Product-specific intercepts with common slope $b=4$ |\n"
        "| Wholesale-only contract | Per-unit margin $\mu=0.42$, no fixed transfers |\n"
        "| All-unit discount | Mars margin falls by $d=0.18$ once the shelf target is met |\n"
        "| Slotting-fee contract | Fixed payments of 1.10 for Mars products and 0.35 for rival products |"
    )

    report.add_solution_method(r"""
The computational object is the retailer's exact finite-choice optimum. There
are only $\binom{12}{7}=792$ feasible assortments, so the script evaluates every
candidate rather than using a local search heuristic. The plotted choices are
the finite-catalog optimum, not a simulation approximation.

```text
Inputs: product catalog J, capacity K, contract C, rebate threshold tau
Output: optimal assortment A*_C and payoff diagnostics

for each feasible assortment A with |A| = K:
    count Mars products M(A)
    for each product j in A:
        set wholesale price w_j^C(A)
        set fixed transfer F_j^C(A)
        solve the retail pricing first-order condition for p_j*(w_j)
        compute q_j, retailer payoff, and upstream payoff
    add product-level payoffs to get Pi^D_C(A) and Pi^U_C(A)

choose A*_C in argmax_A Pi^D_C(A)
repeat over rebate thresholds tau to see where the all-unit discount binds
```

This is closer to an empirical counterfactual than to an abstract combinatorics
exercise: the contract changes the retailer's objective, and the retailer responds
by reallocating scarce shelf space.
""")

    selection_matrix = pd.DataFrame(0, index=products["Product"], columns=contracts)
    for contract, subset in selected.items():
        selection_matrix.loc[products.loc[list(subset), "Product"], contract] = 1

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im = ax1.imshow(selection_matrix.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(np.arange(len(contracts)))
    ax1.set_xticklabels(contracts, rotation=20, ha="right")
    ax1.set_yticks(np.arange(len(products)))
    product_labels = products["Product"] + " (" + products["Manufacturer"].str[0] + ")"
    ax1.set_yticklabels(product_labels)
    ax1.set_title("Retailer-Selected Assortment")
    for i in range(selection_matrix.shape[0]):
        for j in range(selection_matrix.shape[1]):
            label = "in" if selection_matrix.iloc[i, j] else ""
            ax1.text(j, i, label, ha="center", va="center", color="white")
    fig1.colorbar(im, ax=ax1, ticks=[0, 1], label="Selected")
    report.add_figure(
        "figures/assortment-selection.png",
        "Assortment selected under each vertical contract",
        fig1,
        description="Reading the assortment problem directly. Wholesale-only pricing leaves "
        "the retailer with four Mars products. The rebate and slotting-fee contracts both "
        "move one more scarce slot toward Mars, even though retail prices are still chosen "
        "product by product.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(outcome_df))
    ax2.bar(x - 0.2, outcome_df["Retailer objective"], 0.4, label="Retailer objective", color="#4C78A8")
    ax2.bar(x + 0.2, outcome_df["Upstream profit"], 0.4, label="Upstream profit", color="#F58518")
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcome_df["Contract"], rotation=20, ha="right")
    ax2.set_ylabel("Profit")
    ax2.set_title("Profit Incidence by Contract")
    ax2.legend()
    report.add_figure(
        "figures/profit-incidence.png",
        "Retailer and upstream payoffs by contract",
        fig2,
        description="Separating the retailer's objective from total upstream profit. Slotting "
        "fees raise the retailer's payoff mechanically because they are transfers into the "
        "downstream objective. Upstream profit falls in this calibration; that is the cost "
        "of buying placement.",
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(threshold_df["Threshold"], threshold_df["Mars slots"], marker="o", label="Mars slots")
    wholesale_mars_slots = float(outcome_df.loc[outcome_df["Contract"] == "Wholesale only", "Mars slots"].iloc[0])
    wholesale_upstream_profit = float(outcome_df.loc[outcome_df["Contract"] == "Wholesale only", "Upstream profit"].iloc[0])
    ax3.axhline(
        wholesale_mars_slots, color="#4C78A8", linestyle="--", alpha=0.45,
        label="Wholesale-only Mars slots",
    )
    ax3.set_xlabel("All-unit discount threshold")
    ax3.set_ylabel("Mars slots selected")
    ax3.set_ylim(0, capacity + 0.5)
    ax3_t = ax3.twinx()
    ax3_t.plot(threshold_df["Threshold"], threshold_df["Upstream profit"], color="#E45756", marker="s", label="Upstream profit")
    ax3_t.axhline(
        wholesale_upstream_profit, color="#E45756", linestyle=":", alpha=0.45,
        label="Wholesale-only upstream profit",
    )
    ax3_t.set_ylabel("Upstream profit")
    ax3.set_title("Assortment Response to Rebate Thresholds")
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_t.get_legend_handles_labels()
    ax3.legend(
        lines + lines2, labels + labels2,
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
    )
    report.add_figure(
        "figures/rebate-thresholds.png",
        "Mars slots and upstream profit as the all-unit discount threshold changes",
        fig3,
        description="Rebate design is not monotone in the target. A low target gives away "
        "margin on products the retailer would have stocked anyway. A high target may not "
        "move the assortment. The useful region is where the threshold changes the exact "
        "assortment optimum relative to the wholesale-only benchmark.",
    )

    table = outcome_df.copy()
    for col in table.columns:
        if col != "Contract":
            table[col] = table[col].map(lambda v: f"{v:.2f}")
    report.add_table(
        "tables/contract-outcomes.csv",
        "Contract outcomes",
        table,
        description="The table collects the same objects behind the figures. Average retail "
        "prices move little compared with the change in availability, which is the point of "
        "using an assortment model for these contracts.",
    )

    report.add_takeaway(
        "A vertical contract can matter even when the observed retail-price effect is small. "
        "All-unit discounts and slotting fees change the downstream firm's objective over "
        "scarce product slots, so the relevant empirical object is often the joint distribution "
        "of prices, availability, and transfers rather than prices alone."
    )

    report.add_references([
        "Conlon, C., and Mortimer, J. (2021). JPE article on vertical contracts in vending-machine markets.",
        "Hristakeva, S. (2022). JPE article on vertical contracts and product selection in retail markets.",
        "Lecture 9 Slides 2023: Vertical contracts, vending assortments, and slotting fees.",
    ])
    report.write("README.md")
    print(f"Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
