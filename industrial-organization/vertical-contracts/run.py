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

    print("Vertical contracts tutorial")
    print(outcome_df[["Contract", "Mars slots", "Retailer objective", "Upstream profit"]].to_string(index=False))

    report = ModelReport(
        "Vending Assortments Under Vertical Contracts",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Vending space is scarce. A retailer has seven slots and twelve snack products. "
        "A manufacturer can use contract terms to make its products more attractive to stock.\n\n"
        "The object is the retailer's assortment. Each selected product has linear demand, "
        "a wholesale price, and a retail price chosen after stocking. The contracts are "
        "wholesale pricing, an all-unit discount, and slotting fees.\n\n"
        "Exact enumeration checks every feasible seven-product subset. One extra Mars item "
        "can trigger a rebate on all Mars items. The script compares the retailer's best "
        "assortment under each contract."
    )

    report.add_equations(r"""
Let $\mathcal J$ be the product catalog and let $K$ be the number of vending
slots. Product $j$ has demand intercept $a_j$, marginal cost $c_j$, and
manufacturer label $m(j)$. Retail demand is separable:

$$
q_j(p_j)=\max\{a_j-bp_j,0\}.
$$

Here $b$ is the common demand slope. Given wholesale price $w_j$, the retailer sets the product price by

$$
p_j^{*}(w_j)
=\arg\max_{p_j\geq w_j} (p_j-w_j)q_j(p_j).
$$

For an interior product, the unconstrained optimum is $(a_j+bw_j)/(2b)$. The
retailer never prices below a small markup over wholesale, so the price used
is

$$
p_j^{*}(w_j)=\max\lbrace (a_j+bw_j)/(2b),\ w_j+\epsilon\rbrace,
\quad \epsilon=0.05.
$$

The margin floor $\epsilon=0.05$ is a numerical guard; it never binds under
the calibration in this tutorial.

Contract $C$ maps assortment $A$ into wholesale prices and fixed transfers.
The upstream side pays $F_j^C(A)$ to the retailer. The retailer chooses

$$
A_C^{*}
=\arg\max_{A\subset\mathcal J:\ |A|=K}
\sum_{j\in A}\left[(p_j^{*}-w_j^C(A))q_j(p_j^{*})+F_j^C(A)\right].
$$

Write $\Pi^D_C(A)$ for the sum inside the argmax, the retailer's total payoff under contract $C$.

The upstream payoff reported in the results is

$$
\Pi_C^U(A)=
\sum_{j\in A}\left[(w_j^C(A)-c_j)q_j(p_j^{*})-F_j^C(A)\right].
$$

In the all-unit discount case, Mars products can receive a lower wholesale
price. The discount applies only if the assortment contains at least $\tau$
Mars products, with $\tau=4$ in this tutorial:

$$
w_j^C(A)=c_j+\mu-d\,\mathbf 1\{m(j)=\text{Mars}\}\mathbf 1\{M(A)\geq\tau\},
\quad
M(A)=\sum_{j\in A}\mathbf 1\{m(j)=\text{Mars}\}.
$$

Slotting fees instead leave wholesale margins unchanged and work through
$F_j^C(A)$.
""")

    report.add_model_setup(
        "One machine can hold seven of twelve products. Mars controls five products, "
        "and rivals control seven. Demand intercepts and costs differ by product. "
        "The retailer chooses the assortment and then sets selected product prices. "
        "Upstream profit is reported because transfers move surplus across the channel.\n\n"
        "| Object | Value |\n"
        "|--------|-------|\n"
        f"| Products | {len(products)} candy and snack alternatives |\n"
        f"| Machine capacity | {capacity} slots |\n"
        "| Demand | Product-specific intercepts with common slope $b=4$ |\n"
        "| Wholesale-only contract | Per-unit margin $\mu=0.42$, no fixed transfers |\n"
        "| All-unit discount | Mars margin falls by $d=0.18$ once the shelf holds at least $\\tau=4$ Mars products |\n"
        "| Slotting-fee contract | Fixed payments of 1.10 for Mars products and 0.35 for rival products |"
    )

    report.add_solution_method(r"""
The assortment problem is a finite subset search. With twelve products and
seven slots, there are 792 feasible assortments. The script evaluates each
subset exactly and keeps the one with the highest retailer objective.

```text
Inputs: product catalog J, capacity K, contract C
Output: optimal assortment A*_C and payoffs

for each feasible assortment A with |A| = K:
    count Mars products M(A)
    for each product j in A:
        set wholesale price w_j^C(A)
        set fixed transfer F_j^C(A)
        solve the retail pricing first-order condition for p_j*(w_j)
        compute q_j, retailer payoff, and upstream payoff
    add product-level payoffs to get Pi^D_C(A) and Pi^U_C(A)

choose A*_C in argmax_A Pi^D_C(A)
```

Enumeration keeps the shelf-space margin visible. One Mars slot can activate
lower wholesale prices on other Mars products. No simulation approximation is
needed.
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
        description="The heat map shows selected products. Wholesale pricing leaves the "
        "retailer with four Mars items. The rebate and slotting fee each add one Mars "
        "slot. Average retail prices change little, so availability carries the response.",
    )

    table = outcome_df.copy()
    for col in table.columns:
        if col != "Contract":
            table[col] = table[col].map(lambda v: f"{v:.2f}")
    report.add_table(
        "tables/contract-outcomes.csv",
        "Contract outcomes",
        table,
        description="The table gives the payoff objects behind the assortment choice. "
        "The retailer objective includes fixed fees. Upstream profit subtracts those "
        "transfers.",
    )

    report.add_takeaway(
        "Vertical contracts can change availability without large retail price movement. "
        "In this example, rebates and slotting fees move scarce slots toward Mars. "
        "Empirical work on vertical contracts needs product availability and transfers."
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
