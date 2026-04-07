#!/usr/bin/env python3
"""Bertrand-Nash Pricing with Logit Demand and Merger Simulation.

Implements differentiated product oligopoly pricing with logit demand,
calibrates structural parameters from market data, and simulates the
price effects of horizontal mergers.

Reference: Werden and Froeb (1994), "The Effects of Mergers in
Differentiated Products Industries."
"""
import sys
from pathlib import Path

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


# =========================================================================
# Model Functions
# =========================================================================

def ownership_matrix(p2f: np.ndarray) -> np.ndarray:
    """Ownership matrix: Omega[i,j] = 1 if product i and j belong to same firm."""
    J = len(p2f)
    omega = np.zeros((J, J))
    for i in range(J):
        for j in range(J):
            if p2f[i] == p2f[j]:
                omega[i, j] = 1
    return omega


def shares_logit(p: np.ndarray, alpha: float, xi: np.ndarray) -> np.ndarray:
    """Logit market shares: s_j = exp(alpha*p_j + xi_j) / (1 + sum exp(...))."""
    v = np.exp(alpha * p + xi)
    return v / (1 + np.sum(v))


def dqdp_logit(p: np.ndarray, alpha: float, xi: np.ndarray) -> np.ndarray:
    """Demand derivatives for logit: ds_j/dp_k."""
    s = shares_logit(p, alpha, xi)
    cross = -np.outer(s, s)
    np.fill_diagonal(cross, s * (1 - s))
    return alpha * cross


def foc_logit(p: np.ndarray, mc: np.ndarray, alpha: float,
              xi: np.ndarray, p2f: np.ndarray) -> np.ndarray:
    """First-order conditions for Bertrand-Nash equilibrium with logit demand."""
    omega = ownership_matrix(p2f)
    dqdp = dqdp_logit(p, alpha, xi)
    s = shares_logit(p, alpha, xi)
    return -p + mc - np.linalg.solve(omega * dqdp.T, s)


def calibrate(margin: float, shares: np.ndarray, prices: np.ndarray,
              p2f: np.ndarray) -> dict:
    """Calibrate structural parameters from observed data."""
    omega = ownership_matrix(p2f)
    J = len(p2f)
    c1 = prices[0] * (1 - margin)

    # Price coefficient from margin of first product
    alpha = -1 / (1 - shares[0]) / (prices[0] - c1)

    # Demand derivatives
    cross = -np.outer(shares, shares)
    np.fill_diagonal(cross, shares * (1 - shares))
    dqdp = alpha * cross

    # Marginal costs (invert FOC)
    mc = prices + np.linalg.solve(omega * dqdp.T, shares)

    # Mean valuations
    xi = np.log(shares / (1 - np.sum(shares))) - alpha * prices

    # Diversion ratios
    div = np.multiply(shares, 1 / (1 - shares).reshape(-1, 1))
    np.fill_diagonal(div, -1)

    # GUPPI (Gross Upward Pricing Pressure Index)
    guppi = np.zeros(J)
    for j in range(J):
        for k in range(J):
            if j != k and p2f[j] == p2f[k]:
                guppi[j] += div[k, j] * (prices[k] - mc[k])

    return {
        "alpha": alpha, "xi": xi, "mc": mc, "dqdp": dqdp,
        "diversion": div, "guppi": guppi,
    }


def main():
    # =========================================================================
    # Market Data
    # =========================================================================
    shares = np.array([0.15, 0.15, 0.30, 0.30])  # Market shares (outside good: 0.10)
    prices = np.array([1.0, 1.0, 1.0, 1.0])       # Pre-merger prices
    p2f = np.array([1, 2, 3, 4])                   # Product-to-firm mapping
    margin = 0.50                                    # Price-cost margin for calibration

    n_products = len(shares)
    product_names = [f"Product {j+1}" for j in range(n_products)]

    # =========================================================================
    # Calibrate structural parameters
    # =========================================================================
    cal = calibrate(margin, shares, prices, p2f)
    alpha, xi, mc = cal["alpha"], cal["xi"], cal["mc"]

    print(f"Price coefficient alpha = {alpha:.4f}")
    print(f"Marginal costs: {mc}")
    print(f"Valuations xi: {xi}")

    # Verify: FOC at pre-merger prices should be zero
    foc_check = foc_logit(prices, mc, alpha, xi, p2f)
    print(f"FOC check (should be ~0): {foc_check}")

    # =========================================================================
    # Merger Simulation 1: Firm 1 acquires Firm 2
    # =========================================================================
    p2f_merger1 = np.array([1, 1, 3, 4])

    # Upward pricing pressure (FOC at pre-merger prices with post-merger ownership)
    upp_merger1 = foc_logit(prices, mc, alpha, xi, p2f_merger1)

    # Solve for post-merger equilibrium prices
    p_merger1 = scipy.optimize.fsolve(
        foc_logit, x0=prices * 1.1, args=(mc, alpha, xi, p2f_merger1)
    )
    s_merger1 = shares_logit(p_merger1, alpha, xi)

    print(f"\nMerger 1 (Firm 1+2): prices = {p_merger1}")
    print(f"  shares = {s_merger1}")

    # =========================================================================
    # Merger Simulation 2: Merger with 10% cost reduction
    # =========================================================================
    mc_reduced = mc * np.array([0.9, 0.9, 1.0, 1.0])
    p_merger2 = scipy.optimize.fsolve(
        foc_logit, x0=prices * 1.1, args=(mc_reduced, alpha, xi, p2f_merger1)
    )
    s_merger2 = shares_logit(p_merger2, alpha, xi)

    # =========================================================================
    # Merger Simulation 3: Full collusion (monopoly)
    # =========================================================================
    p2f_collusion = np.array([1, 1, 1, 1])
    p_collusion = scipy.optimize.fsolve(
        foc_logit, x0=prices * 1.5, args=(mc, alpha, xi, p2f_collusion)
    )
    s_collusion = shares_logit(p_collusion, alpha, xi)

    # =========================================================================
    # Comparative statics: price effects across different mergers
    # =========================================================================
    merger_scenarios = {
        "Pre-merger": (prices, shares, p2f),
        "Merger 1+2": (p_merger1, s_merger1, p2f_merger1),
        "Merger 1+2 (cost savings)": (p_merger2, s_merger2, p2f_merger1),
        "Full collusion": (p_collusion, s_collusion, p2f_collusion),
    }

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Bertrand-Nash Pricing with Logit Demand",
        "Differentiated product oligopoly pricing, calibration, and merger simulation.",
    )

    report.add_overview(
        "This model implements the standard toolkit for antitrust merger analysis in "
        "differentiated product markets. Firms compete in prices (Bertrand-Nash), consumers "
        "choose products according to a logit discrete choice model, and the analyst calibrates "
        "the structural parameters (price sensitivity, product quality, marginal costs) from "
        "observed market data.\n\n"
        "The key application: given a proposed merger, predict the equilibrium price increase "
        "by re-solving the pricing game under the new ownership structure."
    )

    report.add_equations(r"""
**Logit demand:** Consumer $i$ chooses product $j$ with probability:
$$s_j = \frac{\exp(\alpha p_j + \xi_j)}{1 + \sum_{k=1}^{J} \exp(\alpha p_k + \xi_k)}$$

where $\alpha < 0$ is the price coefficient and $\xi_j$ is product $j$'s quality.

**Bertrand-Nash FOC:** Each multi-product firm $f$ sets prices to satisfy:
$$s_j + \sum_{k \in \mathcal{F}_f} (p_k - c_k) \frac{\partial s_k}{\partial p_j} = 0 \quad \forall j \in \mathcal{F}_f$$

In matrix form: $\mathbf{s} + (\Omega \circ \Delta') (\mathbf{p} - \mathbf{c}) = 0$

where $\Omega$ is the ownership matrix and $\Delta = \partial \mathbf{s} / \partial \mathbf{p}'$ is the demand Jacobian.

**Diversion ratio:** $D_{j \to k} = \frac{s_k}{1 - s_j}$ — fraction of product $j$'s lost sales captured by product $k$.

**GUPPI:** $\text{GUPPI}_j = \sum_{k \in \mathcal{F}_f, k \neq j} D_{k \to j} (p_k - c_k)$ — upward pricing pressure from merger.
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| Products | {n_products} | 4 single-product firms + outside good |\n"
        f"| Shares | {list(shares)} | Market shares (outside good: {1-sum(shares):.2f}) |\n"
        f"| Prices | {list(prices)} | Pre-merger prices |\n"
        f"| Margin | {margin} | Price-cost margin (firm 1) |\n"
        f"| $\\alpha$ | {alpha:.4f} | Calibrated price coefficient |"
    )

    report.add_solution_method(
        "**Step 1: Calibrate** structural parameters ($\\alpha, \\xi, c$) by inverting the "
        "Bertrand-Nash FOC from observed prices, shares, and margins.\n\n"
        "**Step 2: Verify** that the calibrated model replicates observed equilibrium "
        f"(FOC residuals: {np.max(np.abs(foc_check)):.2e}).\n\n"
        "**Step 3: Simulate** mergers by changing the ownership matrix $\\Omega$ and "
        "solving the new pricing game via `scipy.optimize.fsolve`."
    )

    # --- Figure 1: Price comparison across scenarios ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(n_products)
    width = 0.2
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
    for i, (name, (p_s, s_s, _)) in enumerate(merger_scenarios.items()):
        ax1.bar(x + i * width, p_s, width, label=name, color=colors[i])
    ax1.set_xlabel("Product")
    ax1.set_ylabel("Price")
    ax1.set_title("Equilibrium Prices Across Merger Scenarios")
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels(product_names)
    ax1.legend()
    report.add_figure("figures/price-comparison.png", "Equilibrium prices across merger scenarios", fig1)

    # --- Figure 2: Market shares comparison ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for i, (name, (p_s, s_s, _)) in enumerate(merger_scenarios.items()):
        ax2.bar(x + i * width, s_s, width, label=name, color=colors[i])
    ax2.set_xlabel("Product")
    ax2.set_ylabel("Market Share")
    ax2.set_title("Market Shares Across Merger Scenarios")
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels(product_names)
    ax2.legend()
    report.add_figure("figures/share-comparison.png", "Market shares shift as prices rise post-merger", fig2)

    # --- Figure 3: Diversion ratios heatmap ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    im = ax3.imshow(cal["diversion"], cmap="RdBu_r", vmin=-1, vmax=1)
    ax3.set_xticks(range(n_products))
    ax3.set_yticks(range(n_products))
    ax3.set_xticklabels(product_names, fontsize=9)
    ax3.set_yticklabels(product_names, fontsize=9)
    ax3.set_title("Diversion Ratios $D_{j \\to k}$")
    ax3.set_xlabel("To product $k$")
    ax3.set_ylabel("From product $j$")
    for i in range(n_products):
        for j in range(n_products):
            ax3.text(j, i, f"{cal['diversion'][i,j]:.2f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax3)
    report.add_figure("figures/diversion-ratios.png", "Diversion ratios: where do lost sales go?", fig3)

    # --- Table: Merger results ---
    table_data = {
        "Scenario": [],
        "Avg Price": [],
        "Price Change (%)": [],
        "Consumer Surplus": [],
        "HHI": [],
    }
    s0_total = np.sum(shares)
    for name, (p_s, s_s, p2f_s) in merger_scenarios.items():
        hhi = sum(
            (100 * np.sum(s_s[p2f_s == f]) / np.sum(s_s)) ** 2
            for f in np.unique(p2f_s)
        )
        cs = np.sum(s_s * (xi + alpha * p_s)) / (-alpha)  # Approximate CS
        table_data["Scenario"].append(name)
        table_data["Avg Price"].append(f"{np.mean(p_s):.4f}")
        table_data["Price Change (%)"].append(f"{100*(np.mean(p_s)/np.mean(prices)-1):.2f}")
        table_data["Consumer Surplus"].append(f"{cs:.4f}")
        table_data["HHI"].append(f"{hhi:.0f}")

    df = pd.DataFrame(table_data)
    report.add_table("tables/merger-results.csv", "Merger Simulation Results", df)

    report.add_takeaway(
        "Merger simulation reveals the tension between market power and efficiency:\n\n"
        "**Key insights:**\n"
        "- **Unilateral effects**: When Firm 1 merges with Firm 2, both products' prices rise "
        "because the merged firm internalizes the diversion between them. Competing products' "
        "prices also rise (strategic complements).\n"
        "- **Diversion ratios** determine merger severity: if products 1 and 2 are close "
        "substitutes (high diversion), the merger causes larger price increases.\n"
        "- **Cost efficiencies** can offset market power: a 10% marginal cost reduction "
        "partially or fully reverses the price increase from the merger.\n"
        "- **Full collusion** (monopoly) produces the highest prices — this is the upper "
        "bound on how harmful market concentration can be.\n"
        "- The logit model's **IIA property** (Independence of Irrelevant Alternatives) means "
        "diversion ratios are proportional to market shares — a strong assumption. The BLP "
        "random coefficients model relaxes this."
    )

    report.add_references([
        "Berry, S. (1994). \"Estimating Discrete-Choice Models of Product Differentiation.\" *RAND Journal of Economics*, 25(2).",
        "Werden, G. and Froeb, L. (1994). \"The Effects of Mergers in Differentiated Products Industries.\" *Journal of Law, Economics, & Organization*, 10(2).",
        "Nevo, A. (2000). \"Mergers with Differentiated Products: The Case of the Ready-to-Eat Cereal Industry.\" *RAND Journal of Economics*, 31(3).",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
