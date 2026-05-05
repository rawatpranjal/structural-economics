#!/usr/bin/env python3
"""Nash-in-Nash bargaining in a hospital-insurer network."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


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

    report = ModelReport(
        "Nash-in-Nash Hospital-Insurer Bargaining",
        "Outside options, network value, and bilateral transfers in a vertical market.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A hospital-insurer contract is valuable because it changes the insurer's network. "
        "If an insurer loses a hospital, some enrollees may switch to another insurer or to "
        "the outside option. Nash-in-Nash bargaining turns that lost demand into a transfer: "
        "the hospital is paid for the incremental value it brings to the insurer, net of the "
        "hospital's cost of serving those enrollees.\n\n"
        "The point of the example is not that every hospital merger raises payments. It is "
        "to show the object that empirical IO papers compute. A bilateral disagreement "
        "removes one link and leaves all other links in place. A system-level merger changes "
        "that disagreement point: the insurer may lose the whole system, not just one "
        "hospital. That distinction is why Nash-in-Nash is a vertical-market model rather "
        "than a generic Nash equilibrium calculation.\n\n"
        "Premiums are held fixed to keep the bargaining object clean. The "
        "[vertical relationships](../vertical-relationships/) tutorial studies double "
        "marginalization in a simpler channel model, while "
        "[merger simulation](../merger-simulation/) turns changed ownership into final-price "
        "counterfactuals."
    )

    report.add_equations(
        r"""
Let $d \in \{1,\ldots,D\}$ index insurers and $h \in \{1,\ldots,H\}$ index
hospitals. Insurer $d$ has hospital network $G_d$. In the full agreement
network $G$, each insurer carries both hospitals.

Demand is a logit over insurers and an outside option:

$$
q_d(G) =
M \frac{\exp(v_d(G_d) / \sigma_\varepsilon)}
{1 + \sum_{\ell=1}^{D} \exp(v_\ell(G_\ell) / \sigma_\varepsilon)} .
$$

The deterministic utility of an insurer is

$$
v_d(G_d) = Q(G_d) - P_d,
$$

where $P_d$ is the premium. The network value is

$$
Q(\emptyset)=0,\qquad
Q(G_d)=\max_{h \in G_d} a_h + \eta(|G_d|-1)
\quad\text{when }G_d\neq\emptyset .
$$

Here $a_h$ is hospital quality and $\eta$ is the incremental value of a second
in-network hospital. Let $m_d=P_d-c_d^D$ be the insurer margin before hospital
transfers. If the link $(h,d)$ fails, the disagreement network is $G^{-hd}$.
The gross incremental value of hospital $h$ to insurer $d$ is

$$
\Delta_{hd}=m_d\left[q_d(G)-q_d(G^{-hd})\right].
$$

The bilateral surplus net of hospital cost $c_h^H$ is

$$
S_{hd}=\Delta_{hd}-c_h^H q_d(G).
$$

The Nash bargain over the per-enrollee hospital transfer $w_{hd}$ solves

$$
\max_{w_{hd}}
\left[(w_{hd}-c_h^H)q_d(G)\right]^\tau
\times
\left[\Delta_{hd}-w_{hd}q_d(G)\right]^{1-\tau},
$$

so the transfer is

$$
w_{hd}=c_h^H + \tau \frac{S_{hd}}{q_d(G)}
      =(1-\tau)c_h^H+\tau\frac{\Delta_{hd}}{q_d(G)} .
$$

For a merged hospital system $H$, the relevant disagreement removes all system
hospitals from insurer $d$. With $C_H=\sum_h c_h^H$,

$$
W_{Hd}=C_H+\tau\frac{
m_d[q_d(G)-q_d(G^{-Hd})]-C_Hq_d(G)
}{q_d(G)}
$$

is the system-level per-enrollee transfer.
"""
    )

    report.add_model_setup(
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        f"| Hospitals | {n_hospitals} | Upstream negotiators |\n"
        f"| Insurers | {n_insurers} | Downstream plans selling to consumers |\n"
        f"| Market size $M$ | {market_size} | Potential enrollees |\n"
        f"| Bargaining weight $\\tau$ | {tau:.2f} | Hospital share of bilateral surplus |\n"
        f"| Hospital qualities $a_h$ | {', '.join(f'{x:.1f}' for x in hospital_quality)} | Network utility shifters |\n"
        f"| Hospital costs $c_h^H$ | {', '.join(f'{x:.1f}' for x in hospital_cost)} | Cost per enrolled member |\n"
        f"| Insurer premiums $P_d$ | {', '.join(f'{x:.1f}' for x in premiums)} | Fixed downstream prices |\n"
        f"| Insurer costs $c_d^D$ | {', '.join(f'{x:.1f}' for x in insurer_cost)} | Non-hospital marginal costs |\n"
        f"| Second-hospital value $\\eta$ | {second_hospital_value:.1f} | Extra network value beyond the best hospital |\n"
        f"| Logit scale $\\sigma_\\varepsilon$ | {logit_scale:.1f} | Controls substitution across insurers |"
    )

    report.add_solution_method(
        "The computation is finite because every disagreement network can be enumerated. "
        "The important modeling choice is what remains fixed in each disagreement: in a "
        "bilateral Nash-in-Nash bargain, all other hospital-insurer links stay in force. "
        "For a merged system, the disagreement removes the whole system from the insurer's "
        "network.\n\n"
        "```text\n"
        "Algorithm: Nash-in-Nash transfers in a hospital-insurer network\n"
        "Input: full networks G, premiums P, costs c^D and c^H, demand q(.), weight tau\n"
        "Output: bilateral transfers w_hd and merged-system transfers W_Hd\n"
        "Compute full-agreement demand q_d(G) for every insurer d\n"
        "for each hospital h and insurer d:\n"
        "    form G^{-hd} by removing hospital h only from insurer d\n"
        "    compute disagreement demand q_d(G^{-hd})\n"
        "    Delta_hd = (P_d - c_d^D) * [q_d(G) - q_d(G^{-hd})]\n"
        "    S_hd = Delta_hd - c_h^H * q_d(G)\n"
        "    w_hd = c_h^H + tau * S_hd / q_d(G)\n"
        "for each insurer d under hospital-system ownership:\n"
        "    form G^{-Hd} by removing the whole hospital system from insurer d\n"
        "    compute the system surplus using the same demand object\n"
        "    W_Hd = system cost + tau * system surplus / q_d(G)\n"
        "```\n\n"
        f"At the baseline $\\tau={tau:.2f}$, Hospital 1 receives higher transfers because "
        "it is the higher-quality hospital. Insurer 2 pays somewhat more because its higher "
        "premium gives it a larger per-enrollee margin to lose when its network deteriorates."
    )

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

    report.add_results(
        "The bilateral transfers are not arbitrary markups over hospital cost. They come from "
        "the enrollment loss created by a specific broken link. Dropping Hospital 1 hurts "
        "more than dropping Hospital 2 because the first hospital has higher network value; "
        "Insurer 2's higher premium also raises the dollar value of lost enrollment."
    )
    report.add_figure(
        "figures/negotiated-prices.png",
        "Bilateral hospital-insurer transfers and disagreement demand losses",
        fig1,
        description=(
            "The left panel shows per-enrollee transfers with hospital costs marked as "
            "black points. The right panel shows the enrollment loss in each pair's "
            "disagreement network."
        ),
    )

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

    report.add_results(
        "Changing $\\tau$ holds demand and disagreement networks fixed, so it only changes "
        "the division of surplus. Hospital profits rise mechanically with the bargaining "
        "weight, while insurer profits fall because a larger share of the same incremental "
        "network value is paid upstream."
    )
    report.add_figure(
        "figures/profits-vs-bargaining.png",
        "Surplus division as the hospital bargaining weight changes",
        fig2,
        description=(
            "The vertical guide marks the baseline calibration. The exercise is a surplus "
            "split, not a new demand equilibrium at each value of tau."
        ),
    )

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

    report.add_results(
        "The merger comparison adds the two separate hospital transfers and compares that "
        "sum with a single system payment. In this calibration, either hospital alone keeps "
        "an insurer's network viable. Losing the merged system is much worse because the "
        "insurer has no in-network hospital, so the system-level disagreement payoff is "
        "lower and the negotiated payment is higher."
    )
    report.add_figure(
        "figures/merger-prices.png",
        "Separate hospital payments versus merged-system payment",
        fig3,
        description=(
            "The comparison is at the insurer level because a merged system bargains over "
            "one total transfer, not two independent hospital prices."
        ),
    )

    report.add_table(
        "tables/nash-in-nash-results.csv",
        "Bilateral Bargaining Diagnostics",
        bilateral_df,
        description=(
            "The table reports the quantities used in the Nash bargain. Gross value is the "
            "downstream margin times the enrollment loss; surplus subtracts hospital cost."
        ),
    )

    report.add_table(
        "tables/merged-system-results.csv",
        "Ownership Counterfactual",
        merger_df,
        description=(
            "The merged-system rows use a different disagreement event from the bilateral "
            "rows: the insurer loses both hospitals at once."
        ),
    )

    report.add_takeaway(
        "Nash-in-Nash bargaining is useful because it maps a vertical contract into an "
        "observable counterfactual: what does the insurer lose if this agreement fails while "
        "the rest of the contracting network remains intact? The answer is not just market "
        "share or hospital cost. It depends on substitution across insurers, the hospital's "
        "incremental network value, and the ownership structure that defines the relevant "
        "outside option."
    )

    report.add_references(
        [
            'Horn, H. and Wolinsky, A. (1988). "Bilateral Monopolies and Incentives for Merger." *RAND Journal of Economics*, 19(3).',
            'Crawford, G. and Yurukoglu, A. (2012). "The Welfare Effects of Bundling in Multichannel Television Markets." *American Economic Review*, 102(2).',
            'Ho, K. and Lee, R. (2017). "Insurer Competition in Health Care Markets." *Econometrica*, 85(2).',
        ]
    )

    report.write("README.md")
    print(
        f"\nGenerated: README.md + {len(report._figures)} figures + "
        f"{len(report._tables)} tables"
    )
    print(f"Full-network insurer shares: {full_shares.round(3).tolist()}")
    print(f"Full-network utilities: {full_values.round(2).tolist()}")
    print(f"Baseline hospital profits: {hospital_profits.round(2).tolist()}")
    print(f"Baseline insurer profits: {insurer_profits.round(2).tolist()}")


if __name__ == "__main__":
    main()
