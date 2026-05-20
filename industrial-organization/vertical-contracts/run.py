#!/usr/bin/env python3
"""Vertical contracts and capacity-constrained vending assortments."""
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


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

    print("Vertical contracts tutorial")
    print(outcome_df[["Contract", "Mars slots", "Retailer objective", "Upstream profit"]].to_string(index=False))

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
    save_figure(fig1, "figures/assortment-selection.png", dpi=150)

    table = outcome_df.copy()
    for col in table.columns:
        if col != "Contract":
            table[col] = table[col].map(lambda v: f"{v:.2f}")
    Path("tables/contract-outcomes.csv").parent.mkdir(parents=True, exist_ok=True)
    table.to_csv("tables/contract-outcomes.csv", index=False)

    save_thumbnail("figures/assortment-selection.png", "figures/thumb.png")
    print("Figures and tables written.")


if __name__ == "__main__":
    main()
