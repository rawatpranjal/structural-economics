#!/usr/bin/env python3
"""Nash-in-Nash bargaining in a hospital-insurer network."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


def main() -> None:
    # Two hospitals bargain with two insurers over per-enrollee transfers.
    # Premiums are fixed so the exercise isolates the network-bargaining object.
    tau = 0.5
    n_hospitals = 2
    n_insurers = 2
    market_size = 1000
    logit_scale = 5.0

    hospital_quality = np.array([20.0, 18.0])
    second_hospital_value = 3.0
    hospital_cost = np.array([1.0, 1.2])
    insurer_cost = np.array([1.0, 1.0])
    premiums = np.array([8.0, 8.5])

    full_networks = [[0, 1], [0, 1]]

    def network_quality(network: list[int]) -> float:
        """Consumer value from the hospital set in one insurer network."""
        if not network:
            return 0.0
        best_hospital = max(hospital_quality[h] for h in network)
        return best_hospital + second_hospital_value * (len(network) - 1)

    def demand(networks: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Logit demand for insurers given their hospital networks."""
        values = np.array(
            [network_quality(network) - premiums[d] for d, network in enumerate(networks)]
        )
        exp_values = np.exp(values / logit_scale)
        shares = exp_values / (1.0 + exp_values.sum())
        quantities = market_size * shares
        return quantities, shares, values

    def remove_hospital(networks: list[list[int]], hospital: int, insurer: int) -> list[list[int]]:
        """Drop one hospital from one insurer while leaving all other deals fixed."""
        counterfactual = [list(network) for network in networks]
        counterfactual[insurer].remove(hospital)
        return counterfactual

    def remove_system(networks: list[list[int]], insurer: int) -> list[list[int]]:
        """Drop the merged hospital system from one insurer."""
        counterfactual = [list(network) for network in networks]
        counterfactual[insurer] = []
        return counterfactual

    full_quantities, full_shares, full_values = demand(full_networks)
    insurer_margins = premiums - insurer_cost

    def bilateral_bargaining(
        tau_value: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve each bilateral Nash bargain, taking other links as fixed."""
        transfers = np.zeros((n_hospitals, n_insurers))
        gross_value = np.zeros((n_hospitals, n_insurers))
        surplus = np.zeros((n_hospitals, n_insurers))
        disagreement_quantities = np.zeros((n_hospitals, n_insurers))

        for h in range(n_hospitals):
            for d in range(n_insurers):
                counterfactual = remove_hospital(full_networks, h, d)
                q_drop, _, _ = demand(counterfactual)
                disagreement_quantities[h, d] = q_drop[d]

                gross_value[h, d] = insurer_margins[d] * (
                    full_quantities[d] - q_drop[d]
                )
                surplus[h, d] = gross_value[h, d] - hospital_cost[h] * full_quantities[d]
                transfers[h, d] = (
                    hospital_cost[h] + tau_value * surplus[h, d] / full_quantities[d]
                )

        return transfers, gross_value, surplus, disagreement_quantities

    def merged_system_bargaining(tau_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bargain over a single payment to the merged hospital system."""
        system_cost = hospital_cost.sum()
        system_transfers = np.zeros(n_insurers)
        system_gross_value = np.zeros(n_insurers)
        system_disagreement_quantities = np.zeros(n_insurers)

        for d in range(n_insurers):
            counterfactual = remove_system(full_networks, d)
            q_drop, _, _ = demand(counterfactual)
            system_disagreement_quantities[d] = q_drop[d]
            system_gross_value[d] = insurer_margins[d] * (
                full_quantities[d] - q_drop[d]
            )
            system_surplus = (
                system_gross_value[d] - system_cost * full_quantities[d]
            )
            system_transfers[d] = (
                system_cost + tau_value * system_surplus / full_quantities[d]
            )

        return system_transfers, system_gross_value, system_disagreement_quantities

    transfers, gross_value, surplus, disagreement_quantities = bilateral_bargaining(tau)
    system_transfers, system_gross_value, system_disagreement_quantities = (
        merged_system_bargaining(tau)
    )
    separate_total_transfers = transfers.sum(axis=0)

    hospital_profits = ((transfers - hospital_cost[:, None]) * full_quantities).sum(axis=1)
    insurer_profits = insurer_margins * full_quantities - (
        transfers * full_quantities
    ).sum(axis=0)

    tau_grid = np.linspace(0.0, 1.0, 51)
    hospital_profit_by_tau = np.zeros((len(tau_grid), n_hospitals))
    insurer_profit_by_tau = np.zeros((len(tau_grid), n_insurers))

    for i, tau_value in enumerate(tau_grid):
        prices_tau, _, _, _ = bilateral_bargaining(tau_value)
        hospital_profit_by_tau[i] = (
            (prices_tau - hospital_cost[:, None]) * full_quantities
        ).sum(axis=1)
        insurer_profit_by_tau[i] = insurer_margins * full_quantities - (
            prices_tau * full_quantities
        ).sum(axis=0)

    pair_labels = [f"H{h + 1}-I{d + 1}" for h in range(n_hospitals) for d in range(n_insurers)]
    pair_transfers = transfers.flatten()
    pair_costs = np.repeat(hospital_cost, n_insurers)
    pair_disagreement_loss = (
        full_quantities[None, :] - disagreement_quantities
    ).flatten()

    bilateral_rows = []
    for h in range(n_hospitals):
        for d in range(n_insurers):
            bilateral_rows.append(
                {
                    "Pair": f"Hospital {h + 1} - Insurer {d + 1}",
                    "Full demand": f"{full_quantities[d]:.1f}",
                    "Disagreement demand": f"{disagreement_quantities[h, d]:.1f}",
                    "Demand loss": f"{full_quantities[d] - disagreement_quantities[h, d]:.1f}",
                    "Gross value / enrollee": f"{gross_value[h, d] / full_quantities[d]:.3f}",
                    "Surplus / enrollee": f"{surplus[h, d] / full_quantities[d]:.3f}",
                    "Hospital cost": f"{hospital_cost[h]:.3f}",
                    "Transfer": f"{transfers[h, d]:.3f}",
                }
            )

    merger_rows = []
    for d in range(n_insurers):
        merger_rows.append(
            {
                "Insurer": f"Insurer {d + 1}",
                "Full demand": f"{full_quantities[d]:.1f}",
                "Demand without system": f"{system_disagreement_quantities[d]:.1f}",
                "Separate hospital transfers": f"{separate_total_transfers[d]:.3f}",
                "Merged system transfer": f"{system_transfers[d]:.3f}",
                "Change (%)": (
                    f"{100.0 * (system_transfers[d] / separate_total_transfers[d] - 1.0):.1f}"
                ),
            }
        )

    bilateral_df = pd.DataFrame(bilateral_rows)
    merger_df = pd.DataFrame(merger_rows)

    setup_style()

    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(pair_labels))
    axes[0].bar(x, pair_transfers, color="#4477AA", label="Transfer")
    axes[0].scatter(x, pair_costs, color="#222222", zorder=3, label="Hospital cost")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pair_labels)
    axes[0].set_ylabel("Per-enrollee transfer")
    axes[0].set_title("Bilateral Transfers")
    axes[0].legend()

    axes[1].bar(x, pair_disagreement_loss, color="#66A61E")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pair_labels)
    axes[1].set_ylabel("Enrollment loss")
    axes[1].set_title("Demand Lost in Disagreement")
    fig1.tight_layout()

    save_figure(fig1, "figures/negotiated-prices.png", dpi=150)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4.8))
    for h in range(n_hospitals):
        ax2a.plot(
            tau_grid,
            hospital_profit_by_tau[:, h],
            linewidth=2,
            label=f"Hospital {h + 1}",
        )
    ax2a.axvline(tau, color="#444444", linestyle=":", linewidth=1)
    ax2a.set_xlabel("Hospital bargaining weight $\\tau$")
    ax2a.set_ylabel("Profit")
    ax2a.set_title("Hospital Profit")
    ax2a.legend()

    for d in range(n_insurers):
        ax2b.plot(
            tau_grid,
            insurer_profit_by_tau[:, d],
            linewidth=2,
            label=f"Insurer {d + 1}",
        )
    ax2b.axvline(tau, color="#444444", linestyle=":", linewidth=1)
    ax2b.set_xlabel("Hospital bargaining weight $\\tau$")
    ax2b.set_ylabel("Profit")
    ax2b.set_title("Insurer Profit")
    ax2b.legend()
    fig2.tight_layout()

    save_figure(fig2, "figures/profits-vs-bargaining.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(8, 4.8))
    d_x = np.arange(n_insurers)
    width = 0.34
    ax3.bar(
        d_x - width / 2,
        separate_total_transfers,
        width,
        label="Separate hospitals",
        color="#4477AA",
    )
    ax3.bar(
        d_x + width / 2,
        system_transfers,
        width,
        label="Merged system",
        color="#CC6677",
    )
    ax3.set_xticks(d_x)
    ax3.set_xticklabels([f"Insurer {d + 1}" for d in range(n_insurers)])
    ax3.set_ylabel("Total hospital payment per enrollee")
    ax3.set_title("Ownership Changes the Disagreement Point")
    ax3.legend()
    fig3.tight_layout()

    save_figure(fig3, "figures/merger-prices.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    bilateral_df.to_csv("tables/nash-in-nash-results.csv", index=False)
    merger_df.to_csv("tables/merged-system-results.csv", index=False)

    save_thumbnail("figures/negotiated-prices.png", "figures/thumb.png")

    print(f"Full-network insurer shares: {full_shares.round(3).tolist()}")
    print(f"Full-network utilities: {full_values.round(2).tolist()}")
    print(f"Baseline hospital profits: {hospital_profits.round(2).tolist()}")
    print(f"Baseline insurer profits: {insurer_profits.round(2).tolist()}")


if __name__ == "__main__":
    main()
