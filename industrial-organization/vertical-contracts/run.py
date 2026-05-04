#!/usr/bin/env python3
"""Vertical contracts: vending assortments, rebates, and slotting fees."""
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
    price = max((intercept + slope * wholesale) / (2 * slope), wholesale + 0.05)
    quantity = max(intercept - slope * price, 0.0)
    return price, quantity


def evaluate_subset(products: pd.DataFrame, subset: tuple[int, ...], contract: str, threshold: int = 4) -> dict[str, float]:
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
        manufacturer_var = (wholesale - row["Marginal cost"]) * quantity - fixed_fee
        rows.append((idx, price, quantity, retailer_var, manufacturer_var, fixed_fee))
    arr = pd.DataFrame(rows, columns=[
        "Index", "Price", "Quantity", "Retailer variable profit",
        "Manufacturer profit", "Fixed fee",
    ])
    return {
        "Retailer objective": float(arr["Retailer variable profit"].sum() + arr["Fixed fee"].sum()),
        "Retailer variable profit": float(arr["Retailer variable profit"].sum()),
        "Manufacturer profit": float(arr["Manufacturer profit"].sum()),
        "Fixed fees": float(arr["Fixed fee"].sum()),
        "Average price": float(arr["Price"].mean()),
        "Total quantity": float(arr["Quantity"].sum()),
        "Mars slots": mars_count,
    }


def choose_assortment(products: pd.DataFrame, capacity: int, contract: str, threshold: int = 4) -> tuple[tuple[int, ...], dict[str, float]]:
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
            "Manufacturer profit": values["Manufacturer profit"],
            "Selected products": ", ".join(products.loc[list(subset), "Product"]),
        })
    threshold_df = pd.DataFrame(threshold_rows)

    print("Vertical contracts tutorial")
    print(outcome_df[["Contract", "Mars slots", "Retailer objective", "Manufacturer profit"]].to_string(index=False))

    report = ModelReport(
        "Vertical Contracts and Vending Assortments",
        "All-unit discounts, slotting fees, and assortment choice in a capacity-constrained vending channel.",
    )

    report.add_overview(
        "Vending and retail contracts often combine per-unit wholesale prices with fixed "
        "payments, rebates, or slotting fees. Those terms matter because the downstream "
        "firm chooses which products receive scarce shelf or machine space. A contract "
        "can change assortment even when retail prices show little variation.\n\n"
        "The tutorial builds a small vending-machine problem with seven confection slots. "
        "The retailer chooses the assortment that maximizes its own payoff. The manufacturer "
        "can use an all-unit discount or fixed payments to shift which products are stocked."
    )

    report.add_equations(r"""
For product $j$, demand is:
$$q_j(p_j) = a_j - bp_j$$

Given wholesale price $w_j$, the retailer chooses:
$$p_j^{*}(w_j) = \frac{a_j + bw_j}{2b}$$

The retailer chooses an assortment $A$ with capacity $K$:
$$A^{*} = \arg\max_{A: |A|=K} \sum_{j\in A}(p_j-w_j)q_j + F_j(A)$$

All-unit discounts make $w_j$ depend on whether the manufacturer's shelf target is met.
Slotting fees enter through fixed payments $F_j(A)$.
""")

    report.add_model_setup(
        "| Object | Value |\n"
        "|--------|-------|\n"
        f"| Products | {len(products)} candy and snack alternatives |\n"
        f"| Machine capacity | {capacity} slots |\n"
        "| Wholesale only | Linear wholesale prices, no fixed payments |\n"
        "| All-unit discount | Mars wholesale price falls if enough Mars products are stocked |\n"
        "| Slotting fees | Fixed payments reward placement, separate from per-unit margins |"
    )

    report.add_solution_method(
        "The assortment problem is solved by enumerating all feasible seven-product "
        "subsets. For each subset and contract, the script computes retail prices, "
        "quantities, downstream variable profit, manufacturer profit, and fixed transfers. "
        "Enumeration keeps the economics transparent and is fast because the capacity "
        "problem is small."
    )

    selection_matrix = pd.DataFrame(0, index=products["Product"], columns=contracts)
    for contract, subset in selected.items():
        selection_matrix.loc[products.loc[list(subset), "Product"], contract] = 1

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im = ax1.imshow(selection_matrix.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(np.arange(len(contracts)))
    ax1.set_xticklabels(contracts, rotation=20, ha="right")
    ax1.set_yticks(np.arange(len(products)))
    ax1.set_yticklabels(products["Product"])
    ax1.set_title("Selected Vending Assortments")
    for i in range(selection_matrix.shape[0]):
        for j in range(selection_matrix.shape[1]):
            ax1.text(j, i, "in" if selection_matrix.iloc[i, j] else "", ha="center", va="center")
    fig1.colorbar(im, ax=ax1, ticks=[0, 1], label="Selected")
    report.add_figure(
        "figures/assortment-selection.png",
        "Assortment selected under each vertical contract",
        fig1,
        description="The all-unit discount and slotting-fee contracts shift scarce machine "
        "slots toward the manufacturer's products. The contract changes product availability, "
        "not just transfers between firms.",
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(outcome_df))
    ax2.bar(x - 0.2, outcome_df["Retailer objective"], 0.4, label="Retailer objective", color="#4C78A8")
    ax2.bar(x + 0.2, outcome_df["Manufacturer profit"], 0.4, label="Manufacturer profit", color="#F58518")
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcome_df["Contract"], rotation=20, ha="right")
    ax2.set_ylabel("Profit")
    ax2.set_title("Profit Incidence of Contract Terms")
    ax2.legend()
    report.add_figure(
        "figures/profit-incidence.png",
        "Retailer and manufacturer payoffs by contract",
        fig2,
        description="Fixed payments can make the retailer willing to stock products that are "
        "not selected under wholesale-only incentives. The manufacturer pays for placement "
        "when the incremental product-level margin is worth it.",
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(threshold_df["Threshold"], threshold_df["Mars slots"], marker="o", label="Mars slots")
    ax3.set_xlabel("All-unit discount threshold")
    ax3.set_ylabel("Mars slots selected")
    ax3.set_ylim(0, capacity + 0.5)
    ax3_t = ax3.twinx()
    ax3_t.plot(threshold_df["Threshold"], threshold_df["Manufacturer profit"], color="#E45756", marker="s", label="Manufacturer profit")
    ax3_t.set_ylabel("Manufacturer profit")
    ax3.set_title("Assortment Response to Rebate Thresholds")
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_t.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="best")
    report.add_figure(
        "figures/rebate-thresholds.png",
        "Mars slots and manufacturer profit as the all-unit discount threshold changes",
        fig3,
        description="A target that is too low gives away margin without changing behavior. "
        "A target that is too high may be unreachable. The interesting region is where the "
        "threshold changes the retailer's assortment choice.",
    )

    table = outcome_df.copy()
    for col in table.columns:
        if col != "Contract":
            table[col] = table[col].map(lambda v: f"{v:.2f}")
    report.add_table("tables/contract-outcomes.csv", "Contract outcomes", table)

    report.add_takeaway(
        "Vertical contracts are identification problems as much as pricing problems. "
        "A fixed payment does not show up as product-level price variation, yet it can "
        "explain why a product is present in the assortment. That is why vending and "
        "slotting-fee papers model assortment and transfers together."
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
