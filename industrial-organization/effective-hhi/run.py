#!/usr/bin/env python3
"""Herfindahl-Hirschman Index (HHI) and Market Concentration.

Computes the standard HHI for various market structures, demonstrates how
mergers change concentration (delta-HHI), and compares segmented versus
differentiated product markets. Includes DOJ/FTC merger guideline thresholds.

Reference: U.S. Department of Justice & FTC, Horizontal Merger Guidelines (2010).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


# =============================================================================
# Core HHI functions
# =============================================================================

def compute_hhi(shares):
    """Compute HHI from market shares (fractions summing to 1).

    HHI = sum(s_i^2) * 10000, ranging from ~0 (perfect competition) to
    10000 (monopoly).
    """
    shares = np.asarray(shares, dtype=float)
    return float(np.sum(shares ** 2) * 10000)


def delta_hhi(s1, s2):
    """Change in HHI from merging two firms with shares s1 and s2.

    delta-HHI = 2 * s1 * s2 * 10000.
    """
    return 2 * s1 * s2 * 10000


def shares_from_quantities(q, product_to_firm):
    """Aggregate product-level quantities to firm-level market shares."""
    q = np.asarray(q, dtype=float)
    product_to_firm = np.asarray(product_to_firm, dtype=int)
    firm_ids = np.unique(product_to_firm)
    firm_q = np.array([q[product_to_firm == f].sum() for f in firm_ids])
    return firm_q / firm_q.sum()


def hhi_from_quantities(q, product_to_firm):
    """Compute HHI from product quantities and ownership mapping."""
    return compute_hhi(shares_from_quantities(q, product_to_firm))


def classify_hhi(hhi):
    """Classify HHI per DOJ/FTC Horizontal Merger Guidelines (2010)."""
    if hhi < 1500:
        return "Unconcentrated"
    elif hhi < 2500:
        return "Moderately Concentrated"
    else:
        return "Highly Concentrated"


def equal_shares(n):
    """Market shares for n equal-sized firms."""
    return np.ones(n) / n


def lorenz_curve(shares):
    """Compute Lorenz curve from market shares.

    Returns (cumulative fraction of firms, cumulative fraction of output).
    """
    shares = np.sort(np.asarray(shares, dtype=float))
    cum_shares = np.cumsum(shares) / shares.sum()
    cum_firms = np.arange(1, len(shares) + 1) / len(shares)
    # Prepend origin
    return np.concatenate([[0], cum_firms]), np.concatenate([[0], cum_shares])


# =============================================================================
# Differentiated products: ownership matrix and Nash equilibrium
# =============================================================================

def ownership_matrix(product_to_firm):
    """Build ownership matrix Omega from product-to-firm mapping."""
    p2f = np.asarray(product_to_firm, dtype=int)
    J = len(p2f)
    omega = np.zeros((J, J))
    for i in range(J):
        for j in range(J):
            if p2f[i] == p2f[j]:
                omega[i, j] = 1.0
    return omega


def demand_derivatives_matrix(N, alpha, beta):
    """Demand derivative matrix for linear differentiated demand.

    Own-price effect alpha on diagonal, cross-price effect beta off-diagonal.
    """
    dqdp = beta * np.ones((N, N))
    np.fill_diagonal(dqdp, alpha)
    return dqdp


def solve_nash_prices(c, alpha, beta, product_to_firm, a):
    """Solve for Bertrand-Nash equilibrium prices in differentiated products.

    Linear demand: q = a + dqdp @ p
    FOC: p - c + inv(Omega * dqdp') @ q = 0
    """
    from scipy.optimize import fsolve

    N = len(c)
    omega = ownership_matrix(product_to_firm)
    dqdp = demand_derivatives_matrix(N, alpha, beta)

    def foc(p):
        q = a + dqdp @ p
        return -p + c - np.linalg.inv(omega * dqdp.T) @ q

    p_eq = fsolve(foc, x0=c + 0.1, full_output=False)
    q_eq = a + dqdp @ p_eq
    return p_eq, q_eq


# =============================================================================
# Main
# =============================================================================

def main():
    # =====================================================================
    # 1. Example markets
    # =====================================================================
    markets = {
        "Perfect competition (100 firms)": equal_shares(100),
        "10 equal firms": equal_shares(10),
        "5 equal firms": equal_shares(5),
        "Asymmetric (40-30-20-10)": np.array([0.40, 0.30, 0.20, 0.10]),
        "Duopoly (50-50)": np.array([0.50, 0.50]),
        "Dominant firm (70-10-10-10)": np.array([0.70, 0.10, 0.10, 0.10]),
        "Near-monopoly (90-5-5)": np.array([0.90, 0.05, 0.05]),
        "Monopoly": np.array([1.0]),
    }

    market_table = []
    for name, shares in markets.items():
        hhi = compute_hhi(shares)
        market_table.append({
            "Market Structure": name,
            "N Firms": len(shares),
            "Top Share (%)": f"{shares.max() * 100:.0f}",
            "HHI": f"{hhi:.0f}",
            "Classification": classify_hhi(hhi),
        })
    df_markets = pd.DataFrame(market_table)

    # =====================================================================
    # 2. HHI as function of number of equal-sized firms
    # =====================================================================
    n_firms_range = np.arange(1, 51)
    hhi_equal = np.array([compute_hhi(equal_shares(n)) for n in n_firms_range])

    # =====================================================================
    # 3. Merger analysis: delta-HHI for various starting structures
    # =====================================================================
    # Consider merging the two largest firms in each structure
    merger_cases = {
        "10 equal firms\n(merge 2 of 10)": equal_shares(10),
        "5 equal firms\n(merge 2 of 5)": equal_shares(5),
        "Asymmetric\n40-30-20-10": np.array([0.40, 0.30, 0.20, 0.10]),
        "Duopoly\n50-50": np.array([0.50, 0.50]),
        "Dominant\n70-10-10-10": np.array([0.70, 0.10, 0.10, 0.10]),
    }

    merger_results = []
    for label, shares in merger_cases.items():
        sorted_s = np.sort(shares)[::-1]
        s1, s2 = sorted_s[0], sorted_s[1]
        hhi_before = compute_hhi(shares)
        d_hhi = delta_hhi(s1, s2)
        hhi_after = hhi_before + d_hhi
        merger_results.append({
            "label": label,
            "hhi_before": hhi_before,
            "delta_hhi": d_hhi,
            "hhi_after": hhi_after,
        })

    # =====================================================================
    # 4. Segmented vs differentiated product markets
    # =====================================================================
    # Setup: 4 products, initial quantities, calibrate costs
    alpha = -1.0      # own-price sensitivity
    beta_diff = 0.1   # cross-price sensitivity (differentiated)
    beta_seg = 0.0    # cross-price sensitivity (segmented = 0)
    a_init = np.array([1.0, 1.0, 0.9, 0.9])
    c_init = np.array([0.7, 0.7, 0.8, 0.8])

    # Baseline: all separate firms
    p2f_baseline = np.array([0, 1, 2, 3])

    # Merger: firms 0 and 1 merge
    p2f_merged = np.array([0, 0, 1, 2])

    # Solve equilibria
    p_seg_base, q_seg_base = solve_nash_prices(
        c_init, alpha, beta_seg, p2f_baseline, a_init
    )
    p_seg_merge, q_seg_merge = solve_nash_prices(
        c_init, alpha, beta_seg, p2f_merged, a_init
    )
    p_diff_base, q_diff_base = solve_nash_prices(
        c_init, alpha, beta_diff, p2f_baseline, a_init
    )
    p_diff_merge, q_diff_merge = solve_nash_prices(
        c_init, alpha, beta_diff, p2f_merged, a_init
    )

    seg_hhi_before = hhi_from_quantities(q_seg_base, p2f_baseline)
    seg_hhi_after = hhi_from_quantities(q_seg_merge, p2f_merged)
    diff_hhi_before = hhi_from_quantities(q_diff_base, p2f_baseline)
    diff_hhi_after = hhi_from_quantities(q_diff_merge, p2f_merged)

    # =====================================================================
    # 5. Lorenz curves for selected markets
    # =====================================================================
    lorenz_markets = {
        "10 equal firms": equal_shares(10),
        "Asymmetric (40-30-20-10)": np.array([0.40, 0.30, 0.20, 0.10]),
        "Dominant firm (70-10-10-10)": np.array([0.70, 0.10, 0.10, 0.10]),
    }

    # =====================================================================
    # Generate Report
    # =====================================================================
    setup_style()

    report = ModelReport(
        "Effective HHI: Market Concentration and Merger Analysis",
        "The Herfindahl-Hirschman Index as the standard antitrust screening tool "
        "for market concentration and merger review.",
    )

    report.add_overview(
        "The Herfindahl-Hirschman Index (HHI) is the most widely used measure of market "
        "concentration in antitrust economics. It equals the sum of squared market shares "
        "(times 10,000) and ranges from near zero (atomistic competition) to 10,000 "
        "(monopoly). The U.S. DOJ and FTC use HHI thresholds to screen horizontal mergers.\n\n"
        "This model computes HHI for a variety of market structures, demonstrates the "
        "delta-HHI formula for mergers, and compares how concentration changes in segmented "
        "versus differentiated product markets."
    )

    report.add_equations(
        r"""
**Herfindahl-Hirschman Index:**

$$\text{HHI} = \sum_{i=1}^{N} s_i^2 \times 10{,}000$$

where $s_i$ is firm $i$'s market share (as a fraction).

**Delta-HHI from a merger of firms $i$ and $j$:**

$$\Delta\text{HHI} = 2 \, s_i \, s_j \times 10{,}000$$

This follows because the merged firm's share is $s_i + s_j$, so
$(s_i + s_j)^2 - s_i^2 - s_j^2 = 2 s_i s_j$.

**DOJ/FTC Merger Guidelines thresholds:**
- HHI < 1,500: **Unconcentrated** — mergers unlikely to raise concerns
- 1,500 $\le$ HHI < 2,500: **Moderately concentrated** — mergers raising HHI by more than 100 warrant scrutiny
- HHI $\ge$ 2,500: **Highly concentrated** — mergers raising HHI by more than 200 presumed to enhance market power
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $N$ | 1 to 100 | Number of firms |\n"
        f"| $s_i$ | Various | Market shares (fractions summing to 1) |\n"
        f"| $\\alpha$ | {alpha} | Own-price demand sensitivity |\n"
        f"| $\\beta_{{\\text{{seg}}}}$ | {beta_seg} | Cross-price sensitivity (segmented) |\n"
        f"| $\\beta_{{\\text{{diff}}}}$ | {beta_diff} | Cross-price sensitivity (differentiated) |"
    )

    report.add_solution_method(
        "**HHI computation** is direct summation of squared shares. For merger analysis, "
        "we use the closed-form delta-HHI = $2 s_i s_j \\times 10{,}000$.\n\n"
        "**Differentiated products equilibrium** uses Bertrand-Nash pricing. Each firm "
        "maximizes profit taking rivals' prices as given, with linear demand "
        "$q = a + (\\partial q / \\partial p) \\cdot p$. The FOC is:\n\n"
        "$$p - c + (\\Omega \\circ (\\partial q / \\partial p)^\\top)^{-1} q = 0$$\n\n"
        "where $\\Omega$ is the ownership matrix. We solve this system of equations via "
        "`scipy.optimize.fsolve` and compare HHI before and after mergers change $\\Omega$."
    )

    # --- Figure 1: HHI vs number of equal-sized firms ---
    fig1, ax1 = plt.subplots()
    ax1.plot(n_firms_range, hhi_equal, "b-", linewidth=2)
    # Shade threshold regions
    ax1.axhspan(0, 1500, alpha=0.10, color="green", label="Unconcentrated (< 1500)")
    ax1.axhspan(1500, 2500, alpha=0.10, color="orange", label="Moderate (1500-2500)")
    ax1.axhspan(2500, 10500, alpha=0.10, color="red", label="Highly Concentrated (> 2500)")
    ax1.set_xlabel("Number of Equal-Sized Firms ($N$)")
    ax1.set_ylabel("HHI")
    ax1.set_title("HHI as a Function of Number of Equal-Sized Firms")
    ax1.set_xlim(1, 50)
    ax1.set_ylim(0, 10500)
    ax1.legend(loc="upper right", fontsize=9)
    # Annotate key points
    for n_mark in [2, 4, 7, 10]:
        hhi_mark = compute_hhi(equal_shares(n_mark))
        ax1.annotate(
            f"N={n_mark}\nHHI={hhi_mark:.0f}",
            xy=(n_mark, hhi_mark),
            xytext=(n_mark + 3, hhi_mark + 500),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )
    report.add_figure(
        "figures/hhi-vs-nfirms.png",
        "HHI declines as 10000/N for equal-sized firms, with DOJ/FTC threshold regions shaded",
        fig1,
        description="HHI falls as 1/N for symmetric firms, so concentration drops rapidly with "
        "the first few entrants but the marginal effect of additional firms diminishes. A market "
        "needs at least 7 equal-sized firms to fall below the DOJ's 'moderately concentrated' "
        "threshold of 1,500.",
    )

    # --- Figure 2: Merger bar chart (before/after HHI) ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(merger_results))
    width = 0.35
    bars_before = ax2.bar(
        x - width / 2,
        [m["hhi_before"] for m in merger_results],
        width,
        label="HHI Before Merger",
        color="#4878CF",
        edgecolor="white",
    )
    bars_after = ax2.bar(
        x + width / 2,
        [m["hhi_after"] for m in merger_results],
        width,
        label="HHI After Merger",
        color="#D65F5F",
        edgecolor="white",
    )
    # Add delta-HHI labels
    for i, m in enumerate(merger_results):
        ax2.text(
            i + width / 2,
            m["hhi_after"] + 80,
            f"$\\Delta$={m['delta_hhi']:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.axhline(y=1500, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(y=2500, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(len(merger_results) - 0.5, 1560, "Unconcentrated threshold (1500)", fontsize=8, color="green")
    ax2.text(len(merger_results) - 0.5, 2560, "Highly concentrated threshold (2500)", fontsize=8, color="orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m["label"] for m in merger_results], fontsize=9)
    ax2.set_ylabel("HHI")
    ax2.set_title("HHI Before and After Merger of Two Largest Firms")
    ax2.legend()
    fig2.tight_layout()
    report.add_figure(
        "figures/merger-delta-hhi.png",
        "HHI before and after merger of the two largest firms across market structures",
        fig2,
        description="The delta-HHI labels above each bar show the mechanical increase in "
        "concentration from the merger. Mergers between large firms (high-share pairs) "
        "generate disproportionately larger jumps because delta-HHI equals 2*s1*s2*10,000.",
    )

    # --- Figure 3: Lorenz curves ---
    fig3, ax3 = plt.subplots()
    colors = ["#4878CF", "#D65F5F", "#6ACC65"]
    for (name, shares), color in zip(lorenz_markets.items(), colors):
        cum_firms, cum_output = lorenz_curve(shares)
        ax3.plot(cum_firms, cum_output, "-o", color=color, linewidth=2,
                 markersize=5, label=f"{name} (HHI={compute_hhi(shares):.0f})")
    ax3.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect equality")
    ax3.set_xlabel("Cumulative Fraction of Firms")
    ax3.set_ylabel("Cumulative Fraction of Market Output")
    ax3.set_title("Lorenz Curves of Market Concentration")
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    report.add_figure(
        "figures/lorenz-curves.png",
        "Lorenz curves: more bowed curves indicate greater concentration and higher HHI",
        fig3,
        description="The Lorenz curve complements HHI by visualizing the full distribution of "
        "market shares. Equal-sized firms produce a straight line; a dominant firm pushes the "
        "curve away from the diagonal. The area between the curve and the equality line is the "
        "Gini coefficient of market concentration.",
    )

    # --- Table 1: Example markets ---
    report.add_table(
        "tables/market-hhi.csv",
        "HHI for Example Market Structures",
        df_markets,
        description="The classification column maps HHI to DOJ/FTC merger guideline categories. "
        "Note how asymmetry amplifies concentration: a dominant firm with 70% share produces "
        "higher HHI than a symmetric 5-firm market, even though both have similar firm counts.",
    )

    # --- Results: segmented vs differentiated comparison ---
    report.add_results(
        "**Segmented vs. Differentiated Product Markets (merger of firms 1 and 2):**\n\n"
        f"| Market Type | HHI Before | HHI After | $\\Delta$HHI |\n"
        f"|-------------|-----------|----------|------------|\n"
        f"| Segmented ($\\beta=0$) | {seg_hhi_before:.0f} | {seg_hhi_after:.0f} "
        f"| {seg_hhi_after - seg_hhi_before:.0f} |\n"
        f"| Differentiated ($\\beta={beta_diff}$) | {diff_hhi_before:.0f} | {diff_hhi_after:.0f} "
        f"| {diff_hhi_after - diff_hhi_before:.0f} |\n\n"
        "In segmented markets ($\\beta = 0$, no cross-price effects), a merger changes "
        "ownership but cannot raise prices because products are independent. HHI still "
        "changes mechanically through quantity reallocation. In differentiated markets "
        "($\\beta > 0$), merged firms internalize cross-price externalities, raising prices "
        "on substitutes and amplifying concentration."
    )

    report.add_takeaway(
        "The HHI is the workhorse screening tool for antitrust enforcement. Its appeal "
        "lies in simplicity: it requires only market shares and has a clean algebraic "
        "relationship to merger-induced concentration changes.\n\n"
        "**Key insights:**\n"
        "- For $N$ equal-sized firms, HHI $= 10{,}000/N$. Moving from 10 to 5 firms "
        "doubles HHI from 1,000 to 2,000.\n"
        "- Delta-HHI from a merger equals $2 s_i s_j \\times 10{,}000$ — mergers between "
        "larger firms generate disproportionately bigger jumps in concentration.\n"
        "- HHI is a *necessary* but not *sufficient* indicator of market power. Two markets "
        "can have the same HHI but very different competitive conditions depending on "
        "product differentiation, entry barriers, and demand elasticities.\n"
        "- In differentiated product markets, the ownership matrix $\\Omega$ governs which "
        "cross-price effects are internalized. A merger changes $\\Omega$ and thereby "
        "changes equilibrium prices — even holding costs and demand parameters fixed.\n"
        "- The DOJ/FTC thresholds (1,500 and 2,500) are screens, not bright lines. "
        "Context-specific analysis — including efficiencies, entry, and buyer power — "
        "determines the ultimate competitive assessment."
    )

    report.add_references([
        "U.S. Department of Justice & Federal Trade Commission (2010). "
        "*Horizontal Merger Guidelines*.",
        "Werden, G. (1991). \"A Robust Test for Consumer Welfare Enhancing Mergers Among "
        "Sellers of Differentiated Products.\" *Journal of Industrial Economics*, 39(4).",
        "Farrell, J. and Shapiro, C. (1990). \"Horizontal Mergers: An Equilibrium Analysis.\" "
        "*American Economic Review*, 80(1), 107-126.",
        "Tirole, J. (1988). *The Theory of Industrial Organization*. MIT Press, Ch. 5.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
