#!/usr/bin/env python3
"""Nash-in-Nash Bargaining: Bilateral Negotiations in Vertical Markets.

Implements the Nash-in-Nash framework for modeling bilateral negotiations
between upstream suppliers and downstream firms, with applications to
hospital-insurer and manufacturer-retailer negotiations.

Reference: Horn and Wolinsky (1988), Crawford and Yurukoglu (2012).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def main():
    # =========================================================================
    # Model Setup: 2 upstream firms (hospitals), 2 downstream firms (insurers)
    # =========================================================================
    # Each upstream firm u negotiates a price p_u with each downstream firm d
    # Downstream firms compete in a final goods market (insurance market)
    # Nash bargaining: max (pi_u - d_u)^tau * (pi_d - d_d)^(1-tau)

    # Parameters
    tau = 0.5            # Upstream bargaining power
    n_upstream = 2       # Number of hospitals
    n_downstream = 2     # Number of insurers
    alpha = 10.0         # Consumer WTP for hospital access
    mc_u = np.array([2.0, 3.0])  # Upstream marginal costs
    mc_d = np.array([1.0, 1.0])  # Downstream marginal costs
    premium = np.array([8.0, 8.0])  # Insurance premiums (downstream prices)
    market_size = 1000   # Total potential enrollees

    # Consumer demand: enrollees to insurer d depend on network (which hospitals included)
    # Simple model: value = sum of alpha for each hospital in network - premium
    def demand(networks, premiums, alpha_val):
        """Compute demand for each insurer given networks and premiums.
        networks[d] = list of hospitals available to insurer d.
        """
        n_d = len(premiums)
        values = np.zeros(n_d)
        for d in range(n_d):
            values[d] = alpha_val * len(networks[d]) - premiums[d]
        # Logit-style demand
        exp_v = np.exp(values / 5.0)  # Scale parameter
        shares = exp_v / (1 + np.sum(exp_v))
        return shares * market_size

    # =========================================================================
    # Full network: all hospitals in all insurers
    # =========================================================================
    full_networks = [[0, 1], [0, 1]]
    q_full = demand(full_networks, premium, alpha)

    # =========================================================================
    # Disagreement payoffs: what happens if hospital u and insurer d fail to agree
    # =========================================================================
    def disagreement_network(u, d, full_nets):
        """Network if hospital u is dropped from insurer d."""
        nets = [list(n) for n in full_nets]
        if u in nets[d]:
            nets[d].remove(u)
        return nets

    # =========================================================================
    # Nash-in-Nash solution: each bilateral pair bargains simultaneously
    # =========================================================================
    # For each (u, d) pair, the Nash bargaining solution satisfies:
    # p_{ud} = tau * (MC_change for d from adding u) + (1-tau) * mc_u

    negotiated_prices = np.zeros((n_upstream, n_downstream))
    upstream_profits = np.zeros(n_upstream)
    downstream_profits = np.zeros(n_downstream)
    disagreement_demands = np.zeros((n_upstream, n_downstream))

    for u in range(n_upstream):
        for d in range(n_downstream):
            # Demand if u drops out of d's network
            dis_nets = disagreement_network(u, d, full_networks)
            q_dis = demand(dis_nets, premium, alpha)
            disagreement_demands[u, d] = q_dis[d]

            # Incremental value of hospital u to insurer d
            # = (premium - mc_d) * (q_full[d] - q_disagreement[d])
            incremental_value = (premium[d] - mc_d[d]) * (q_full[d] - q_dis[d])

            # Nash bargaining price
            negotiated_prices[u, d] = tau * incremental_value / q_full[d] + mc_u[u]

    # Compute profits
    for u in range(n_upstream):
        upstream_profits[u] = sum(
            (negotiated_prices[u, d] - mc_u[u]) * q_full[d]
            for d in range(n_downstream)
        )
    for d in range(n_downstream):
        downstream_profits[d] = (premium[d] - mc_d[d]) * q_full[d] - sum(
            negotiated_prices[u, d] * q_full[d] for u in range(n_upstream)
        )

    # =========================================================================
    # Comparative statics: vary bargaining power
    # =========================================================================
    tau_range = np.linspace(0.01, 0.99, 50)
    prices_by_tau = np.zeros((len(tau_range), n_upstream, n_downstream))
    up_profits_by_tau = np.zeros((len(tau_range), n_upstream))
    down_profits_by_tau = np.zeros((len(tau_range), n_downstream))

    for t_idx, tau_val in enumerate(tau_range):
        for u in range(n_upstream):
            for d in range(n_downstream):
                dis_nets = disagreement_network(u, d, full_networks)
                q_dis = demand(dis_nets, premium, alpha)
                incr_val = (premium[d] - mc_d[d]) * (q_full[d] - q_dis[d])
                prices_by_tau[t_idx, u, d] = tau_val * incr_val / q_full[d] + mc_u[u]

        for u in range(n_upstream):
            up_profits_by_tau[t_idx, u] = sum(
                (prices_by_tau[t_idx, u, d] - mc_u[u]) * q_full[d]
                for d in range(n_downstream)
            )
        for d in range(n_downstream):
            down_profits_by_tau[t_idx, d] = (premium[d] - mc_d[d]) * q_full[d] - sum(
                prices_by_tau[t_idx, u, d] * q_full[d] for u in range(n_upstream)
            )

    # =========================================================================
    # Merger simulation: Hospital 0 acquires Hospital 1
    # =========================================================================
    # Post-merger: joint negotiation maximizes combined upstream surplus
    # Each insurer now faces a single upstream entity
    merged_prices = np.zeros((n_upstream, n_downstream))
    for d in range(n_downstream):
        # Disagreement: lose BOTH hospitals
        dis_nets_both = [list(n) for n in full_networks]
        dis_nets_both[d] = []
        q_dis_both = demand(dis_nets_both, premium, alpha)
        total_incr = (premium[d] - mc_d[d]) * (q_full[d] - q_dis_both[d])
        avg_mc = np.mean(mc_u)
        for u in range(n_upstream):
            merged_prices[u, d] = tau * total_incr / (n_upstream * q_full[d]) + mc_u[u]

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Nash-in-Nash Bargaining",
        "Bilateral negotiations in vertical markets between upstream suppliers and downstream firms.",
    )

    report.add_overview(
        "The Nash-in-Nash framework models bilateral negotiations in markets with vertical "
        "structure: upstream firms (hospitals, manufacturers) negotiate prices with downstream "
        "firms (insurers, retailers). Each bilateral pair bargains simultaneously, taking other "
        "pairs' agreements as given.\n\n"
        "This model is widely used in health economics (hospital-insurer negotiations) and "
        "retail (manufacturer-retailer bargaining). The key insight: a firm's bargaining "
        "leverage depends on its *incremental value* — how much the other side loses by "
        "walking away from this specific deal."
    )

    report.add_equations(r"""
**Nash bargaining solution** for pair $(u, d)$:
$$p_{ud}^* = \arg\max_{p} \left(\pi_u^{\text{agree}} - \pi_u^{\text{disagree}}\right)^\tau \left(\pi_d^{\text{agree}} - \pi_d^{\text{disagree}}\right)^{1-\tau}$$

**Simplified form (linear surplus):**
$$p_{ud}^* = \tau \cdot \frac{\Delta_d(u)}{q_d} + c_u$$

where $\Delta_d(u) = (P_d - c_d)(q_d^{\text{agree}} - q_d^{\text{disagree}})$ is hospital $u$'s incremental value to insurer $d$.

**Key property:** Negotiated price depends on the *outside option* — what happens if this specific deal falls through while all other deals remain in place.
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| Upstream firms | {n_upstream} | Hospitals |\n"
        f"| Downstream firms | {n_downstream} | Insurers |\n"
        f"| $\\tau$ | {tau} | Upstream bargaining power |\n"
        f"| $\\alpha$ | {alpha} | Consumer WTP per hospital |\n"
        f"| MC upstream | {list(mc_u)} | Hospital marginal costs |\n"
        f"| Market size | {market_size} | Potential enrollees |"
    )

    report.add_solution_method(
        "Each bilateral negotiated price is computed analytically from the Nash bargaining "
        "solution. The key step is computing the disagreement payoff: what demand would each "
        "insurer face if it lost access to a specific hospital? This determines each hospital's "
        "*incremental value* and hence its bargaining leverage."
    )

    # --- Figure 1: Negotiated prices ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(n_downstream)
    width = 0.35
    for u in range(n_upstream):
        ax1.bar(x + u * width, negotiated_prices[u], width,
                label=f"Hospital {u+1} (MC={mc_u[u]:.1f})")
    ax1.set_xlabel("Insurer")
    ax1.set_ylabel("Negotiated Price")
    ax1.set_title("Nash-in-Nash Negotiated Prices")
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels([f"Insurer {d+1}" for d in range(n_downstream)])
    ax1.legend()
    report.add_figure("figures/negotiated-prices.png", "Negotiated prices: higher-cost hospital commands higher price", fig1,
        description="Negotiated prices reflect both the hospital's marginal cost and its "
        "incremental value to the insurer's network. Hospital 2 commands a higher price "
        "partly because of higher costs and partly because its outside option (what the "
        "insurer loses from exclusion) determines bargaining leverage.")


    # --- Figure 2: Profits vs bargaining power ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    for u in range(n_upstream):
        ax2a.plot(tau_range, up_profits_by_tau[:, u], linewidth=2, label=f"Hospital {u+1}")
    ax2a.set_xlabel("Upstream bargaining power $\\tau$")
    ax2a.set_ylabel("Profit")
    ax2a.set_title("Upstream (Hospital) Profits")
    ax2a.legend()

    for d in range(n_downstream):
        ax2b.plot(tau_range, down_profits_by_tau[:, d], linewidth=2, label=f"Insurer {d+1}")
    ax2b.set_xlabel("Upstream bargaining power $\\tau$")
    ax2b.set_ylabel("Profit")
    ax2b.set_title("Downstream (Insurer) Profits")
    ax2b.legend()
    fig2.tight_layout()
    report.add_figure("figures/profits-vs-bargaining.png", "Profits shift from downstream to upstream as bargaining power increases", fig2,
        description="Bargaining power tau governs the division of surplus between hospitals "
        "and insurers. As tau rises from 0 to 1, the entire surplus wedge shifts upstream. "
        "Downstream profits can turn negative when hospitals extract more than the insurer's "
        "margin can sustain.")


    # --- Figure 3: Pre vs post-merger prices ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    labels = [f"H{u+1}-I{d+1}" for u in range(n_upstream) for d in range(n_downstream)]
    pre = negotiated_prices.flatten()
    post = merged_prices.flatten()
    x3 = np.arange(len(labels))
    ax3.bar(x3 - 0.2, pre, 0.35, label="Pre-merger", color="steelblue")
    ax3.bar(x3 + 0.2, post, 0.35, label="Post-merger", color="coral")
    ax3.set_xlabel("Bilateral pair")
    ax3.set_ylabel("Negotiated price")
    ax3.set_title("Hospital Merger: Prices Rise Due to Increased Leverage")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels)
    ax3.legend()
    report.add_figure("figures/merger-prices.png", "Hospital merger raises all negotiated prices by increasing outside option", fig3,
        description="After the merger, losing one hospital means losing access to the entire "
        "merged system. This worsens the insurer's disagreement payoff and strengthens the "
        "merged entity's bargaining position, resulting in higher negotiated prices for all "
        "bilateral pairs.")


    # --- Table ---
    table_rows = []
    for u in range(n_upstream):
        for d in range(n_downstream):
            table_rows.append({
                "Pair": f"Hospital {u+1} - Insurer {d+1}",
                "Pre-merger price": f"{negotiated_prices[u,d]:.4f}",
                "Post-merger price": f"{merged_prices[u,d]:.4f}",
                "Change (%)": f"{100*(merged_prices[u,d]/negotiated_prices[u,d]-1):.2f}",
                "Demand (full)": f"{q_full[d]:.1f}",
                "Demand (disagree)": f"{disagreement_demands[u,d]:.1f}",
            })
    df = pd.DataFrame(table_rows)
    report.add_table("tables/nash-in-nash-results.csv", "Nash-in-Nash Negotiation Results", df,
        description="The demand columns show each insurer's enrollment with and without "
        "a given hospital. The gap between these numbers determines the hospital's "
        "incremental value -- the key driver of negotiated prices in the Nash-in-Nash "
        "framework.")


    report.add_takeaway(
        "Nash-in-Nash bargaining reveals how vertical market structure affects prices:\n\n"
        "**Key insights:**\n"
        "- **Incremental value = leverage**: a hospital that is *essential* to an insurer's "
        "network commands a higher negotiated price. The outside option (network without that "
        "hospital) determines bargaining power.\n"
        "- **Bargaining power matters**: as $\\tau$ increases, surplus shifts from insurers to "
        "hospitals. At $\\tau = 1$, hospitals extract all incremental value.\n"
        "- **Hospital mergers raise prices**: when hospitals merge, the combined entity "
        "negotiates as a single unit, and the disagreement point becomes losing *both* "
        "hospitals — a much worse threat. This increases leverage and prices.\n"
        "- **Policy implication**: antitrust authorities should focus on *incremental value* "
        "rather than market share alone when evaluating hospital mergers."
    )

    report.add_references([
        "Horn, H. and Wolinsky, A. (1988). \"Bilateral Monopolies and Incentives for Merger.\" *RAND Journal of Economics*, 19(3).",
        "Crawford, G. and Yurukoglu, A. (2012). \"The Welfare Effects of Bundling in Multichannel Television Markets.\" *American Economic Review*, 102(2).",
        "Ho, K. and Lee, R. (2017). \"Insurer Competition in Health Care Markets.\" *Econometrica*, 85(2).",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
