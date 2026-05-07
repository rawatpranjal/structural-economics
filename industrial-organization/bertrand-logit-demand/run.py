#!/usr/bin/env python3
"""Differentiated-products merger pricing with logit demand.

Calibrates a small Bertrand-Nash pricing model from observed prices, shares,
and one margin, then resolves the pricing first-order conditions after a
horizontal merger and a cost-efficiency counterfactual.

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
from lib.plotting import setup_style
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

    # GUPPI-style dollar pressure from products owned by the same firm.
    guppi = np.zeros(J)
    for j in range(J):
        for k in range(J):
            if j != k and p2f[j] == p2f[k]:
                guppi[j] += div[j, k] * (prices[k] - mc[k])

    return {
        "alpha": alpha, "xi": xi, "mc": mc, "dqdp": dqdp,
        "diversion": div, "guppi": guppi,
    }


def outside_share(shares: np.ndarray) -> float:
    """Return the no-purchase share implied by inside shares."""
    return float(1.0 - np.sum(shares))


def logit_consumer_surplus(p: np.ndarray, alpha: float, xi: np.ndarray) -> float:
    """Representative logit consumer surplus, up to an income constant."""
    inclusive_value = np.log1p(np.sum(np.exp(alpha * p + xi)))
    return float(inclusive_value / (-alpha))


def inside_hhi(shares: np.ndarray, p2f: np.ndarray) -> float:
    """HHI computed on inside-good firm shares, scaled by 10,000."""
    inside_total = np.sum(shares)
    return float(sum(
        (100 * np.sum(shares[p2f == f]) / inside_total) ** 2
        for f in np.unique(p2f)
    ))


def fmt_vector(values: np.ndarray, digits: int = 2) -> str:
    """Format a short numeric vector for a Markdown table."""
    return "[" + ", ".join(f"{float(v):.{digits}f}" for v in values) + "]"


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
    # Comparative statics: price effects across different ownership structures
    # =========================================================================
    merger_scenarios = {
        "Pre-merger": {
            "prices": prices, "shares": shares, "ownership": p2f, "costs": mc,
        },
        "Merger 1+2": {
            "prices": p_merger1, "shares": s_merger1,
            "ownership": p2f_merger1, "costs": mc,
        },
        "Merger 1+2, lower costs": {
            "prices": p_merger2, "shares": s_merger2,
            "ownership": p2f_merger1, "costs": mc_reduced,
        },
        "Common ownership": {
            "prices": p_collusion, "shares": s_collusion,
            "ownership": p2f_collusion, "costs": mc,
        },
    }

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Differentiated-Products Merger Pricing with Logit Demand",
        "Ownership, diversion, and Bertrand-Nash price effects in a four-product market.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Take a four-product category with four single-product firms. Products 1 and 2 "
        "are candidates for common ownership. If Product 1 raises its price before the "
        "merger, some consumers leave Product 1 for Product 2, but Firm 1 does not earn "
        "those sales. After the merger, that same diversion becomes revenue inside the "
        "merged firm. Pricing incentives change even if demand and marginal costs are "
        "unchanged.\n\n"
        "To quantify the price effect, the tutorial builds the demand and supply objects "
        "used in merger simulation. It calibrates a simple logit model to pre-merger "
        "prices, shares, and one observed margin, recovers marginal costs from the "
        "Bertrand first-order conditions, then solves those conditions again under new "
        "ownership and under a cost-efficiency case. The core computation is a nonlinear "
        "root-finding problem: find prices that make every product's pricing condition "
        "hold for a given ownership matrix. [BLP random coefficients](../blp-random-coefficients/) "
        "relaxes the logit substitution pattern; [merger simulation across demand "
        "systems](../merger-simulation/) compares the consequences of that modeling choice."
    )

    report.add_equations(r"""
There are $J$ inside products and an outside good. Product $j$ has price $p_j$,
marginal cost $c_j$, and mean non-price utility $\xi_j$. With
$\alpha<0$, mean utility is
$$\delta_j(p)=\xi_j+\alpha p_j.$$

Logit demand gives inside share
$$
s_j(p)=
\frac{\exp(\delta_j(p))}
{1+\sum_{\ell=1}^J \exp(\delta_\ell(p))},
\qquad
s_0(p)=
\frac{1}
{1+\sum_{\ell=1}^J \exp(\delta_\ell(p))}.
$$

The demand derivative used by the pricing equation is
$$
\frac{\partial s_k}{\partial p_j}
=\alpha s_k(\mathbf 1\{j=k\}-s_j).
$$

Let $\Omega_{jk}=1$ when products $j$ and $k$ are controlled by the same firm
and zero otherwise. Bertrand-Nash pricing satisfies, for each product $j$,
$$
0=s_j(p)+\sum_{k=1}^J
\Omega_{jk}(p_k-c_k)\frac{\partial s_k(p)}{\partial p_j}.
$$
If $\Delta_{jk}=\partial s_j/\partial p_k$, the markup equation is
$$
p-c=-(\Omega\circ \Delta')^{-1}s.
$$

The diversion ratio from product $j$ to product $k$ is
$$
D_{j\to k}=\frac{s_k}{1-s_j},\qquad j\neq k.
$$
Under simple logit this depends only on product $k$'s share and the outside
option. That is the IIA restriction: substitution is not allowed to depend on
which products are objectively closer substitutes.

For welfare comparisons, the logit consumer-surplus index is
$$
CS(p)=\frac{1}{-\alpha}
\log\left(1+\sum_{j=1}^J \exp(\xi_j+\alpha p_j)\right),
$$
up to the usual income constant.
""")

    report.add_model_setup(
        "The calibration is deliberately small. Products 1 and 2 are candidate "
        "merger partners in one category; products 3 and 4 remain rival products. "
        "The numbers are not estimated from a dataset. They make the mapping from "
        "shares and prices to markups visible.\n\n"
        "| Object | Value | Role |\n"
        "|--------|-------|------|\n"
        f"| Inside products | {n_products} | Four single-product firms before the merger |\n"
        f"| Inside shares | {fmt_vector(shares)} | Observed product shares |\n"
        f"| Outside share | {outside_share(shares):.2f} | No-purchase option |\n"
        f"| Prices | {fmt_vector(prices)} | Pre-merger prices |\n"
        f"| Product 1 margin | {margin:.2f} | Pins down the logit price coefficient |\n"
        f"| $\\alpha$ | {alpha:.4f} | Calibrated price sensitivity |\n"
        f"| Marginal costs | {fmt_vector(mc)} | Recovered from the pre-merger FOCs |\n"
        "| Counterfactuals | merger 1+2, merger 1+2 with lower costs, common ownership | Ownership and cost experiments |"
    )

    report.add_solution_method(
        "The algorithm first makes the observed pre-merger prices an equilibrium of "
        "the calibrated model. It then asks how the same demand system prices the "
        "category when the ownership matrix changes. Because each price changes "
        "shares for every product, the post-merger calculation is a coupled nonlinear "
        "system rather than four separate markup calculations.\n\n"
        "```text\n"
        "Inputs: pre-merger prices p, shares s, firm labels f(j),\n"
        "        one observed margin, and counterfactual ownership labels\n"
        "Outputs: calibrated demand/costs and equilibrium outcomes by scenario\n\n"
        "1. Compute the outside share: s0 = 1 - sum_j s_j.\n"
        "2. Infer alpha from Product 1's observed margin and single-product FOC.\n"
        "3. Back out mean utilities: xi_j = log(s_j / s0) - alpha p_j.\n"
        "4. Form Delta(p), the logit demand Jacobian at observed prices.\n"
        "5. Invert the pre-merger markup equation to recover marginal costs c.\n"
        "6. Replace Omega and c when ownership or efficiencies change.\n"
        "7. Use root finding on the Bertrand FOCs F(p; Omega, c) = 0.\n"
        "8. Report prices, shares, outside share, consumer surplus, HHI, and residuals.\n"
        "```\n\n"
        f"The pre-merger FOC residual after calibration is {np.max(np.abs(foc_check)):.2e}. "
        "The post-merger solutions below come from the same pricing equations, not from "
        "a reduced-form pass-through rule."
    )

    # --- Figure 1: Price comparison across scenarios ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(n_products)
    scenario_names = list(merger_scenarios.keys())
    offsets = (np.arange(len(scenario_names)) - (len(scenario_names) - 1) / 2) * 0.18
    width = 0.18
    colors = ["steelblue", "coral", "seagreen", "mediumpurple"]
    for i, name in enumerate(scenario_names):
        p_s = merger_scenarios[name]["prices"]
        ax1.bar(x + offsets[i], p_s, width, label=name, color=colors[i])
    ax1.axhline(np.mean(prices), color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax1.set_xlabel("Product")
    ax1.set_ylabel("Price")
    ax1.set_title("Equilibrium Prices Under Alternative Ownership")
    ax1.set_xticks(x)
    ax1.set_xticklabels(product_names)
    ax1.set_ylim(0, max(np.max(v["prices"]) for v in merger_scenarios.values()) * 1.12)
    ax1.legend(frameon=False, ncol=2)
    report.add_figure(
        "figures/price-comparison.png",
        "Equilibrium prices under alternative ownership",
        fig1,
        description=(
            "After Products 1 and 2 merge, the common owner raises both prices because a "
            "lost sale to the partner product is no longer fully lost. Products 3 and 4 "
            "also rise because prices are strategic complements in this demand system. "
            "The 10 percent cost reduction mutes the increase but does not erase it."
        ),
    )


    # --- Figure 2: Market shares comparison ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    share_names = product_names + ["Outside"]
    x_share = np.arange(n_products + 1)
    for i, name in enumerate(scenario_names):
        s_s = merger_scenarios[name]["shares"]
        all_shares = np.append(s_s, outside_share(s_s))
        ax2.bar(x_share + offsets[i], all_shares, width, label=name, color=colors[i])
    ax2.set_xlabel("Alternative")
    ax2.set_ylabel("Market Share")
    ax2.set_title("Market Shares and the Outside Option")
    ax2.set_xticks(x_share)
    ax2.set_xticklabels(share_names)
    ax2.legend(frameon=False, ncol=2)
    report.add_figure(
        "figures/share-comparison.png",
        "Market shares and the outside option",
        fig2,
        description=(
            "Volume leaves the merged products after their price increases. Some sales move "
            "to rival inside products, and some leave the inside market through the outside "
            "good. The outside option limits how much price pressure any owner can internalize."
        ),
    )


    # --- Figure 3: Diversion ratios heatmap ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    diversion_plot = cal["diversion"].copy()
    np.fill_diagonal(diversion_plot, np.nan)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f0f0f0")
    im = ax3.imshow(diversion_plot, cmap=cmap, vmin=0, vmax=np.nanmax(diversion_plot))
    ax3.set_xticks(range(n_products))
    ax3.set_yticks(range(n_products))
    ax3.set_xticklabels(product_names, fontsize=9)
    ax3.set_yticklabels(product_names, fontsize=9)
    ax3.set_title("Diversion Ratios")
    ax3.set_xlabel("Product gaining the sale")
    ax3.set_ylabel("Product losing the sale")
    for i in range(n_products):
        for j in range(n_products):
            label = "" if i == j else f"{cal['diversion'][i, j]:.2f}"
            ax3.text(j, i, label, ha="center", va="center", fontsize=10)
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("Diversion share")
    report.add_figure(
        "figures/diversion-ratios.png",
        "Diversion ratios between products",
        fig3,
        description=(
            "Each row is the product losing a marginal sale; each column is the product that "
            "receives it. Under logit, larger-share products absorb more diverted sales from "
            "every other product. This makes the first merger calculation transparent, but it "
            "also shows why richer demand systems matter when product closeness drives the case."
        ),
    )


    # --- Table: Merger results ---
    table_data = {
        "Scenario": [],
        "Avg Price": [],
        "Price Change (%)": [],
        "Inside Share": [],
        "Outside Share": [],
        "CS Change": [],
        "HHI": [],
        "FOC Residual": [],
    }
    baseline_cs = logit_consumer_surplus(prices, alpha, xi)
    for name, outcome in merger_scenarios.items():
        p_s = outcome["prices"]
        s_s = outcome["shares"]
        p2f_s = outcome["ownership"]
        mc_s = outcome["costs"]
        hhi = inside_hhi(s_s, p2f_s)
        cs = logit_consumer_surplus(p_s, alpha, xi)
        residual = np.max(np.abs(foc_logit(p_s, mc_s, alpha, xi, p2f_s)))
        table_data["Scenario"].append(name)
        table_data["Avg Price"].append(f"{np.mean(p_s):.4f}")
        table_data["Price Change (%)"].append(f"{100*(np.mean(p_s)/np.mean(prices)-1):.2f}")
        table_data["Inside Share"].append(f"{np.sum(s_s):.4f}")
        table_data["Outside Share"].append(f"{outside_share(s_s):.4f}")
        table_data["CS Change"].append(f"{cs - baseline_cs:+.4f}")
        table_data["HHI"].append(f"{hhi:.0f}")
        table_data["FOC Residual"].append(f"{residual:.1e}")

    df = pd.DataFrame(table_data)
    report.add_table(
        "tables/merger-results.csv",
        "Merger simulation outcomes",
        df,
        description=(
            "HHI is computed on inside-good firm shares, so it captures the ownership change "
            "rather than the outside option. Consumer surplus is reported as a change from "
            "the pre-merger calibration. The FOC residuals check that the reported prices "
            "solve the post-merger pricing equations."
        ),
    )


    report.add_takeaway(
        "Merger simulation here is an ownership-matrix exercise inside a pricing first-order "
        "condition. The same calibrated demand and costs support separate firms, common "
        "ownership, cost efficiencies, and monopoly. Diversion gives the merged firm its "
        "upward pricing incentive, while cost reductions push in the other direction. Simple "
        "logit makes diversion transparent, but its IIA pattern also limits what this "
        "benchmark can say about product closeness."
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
