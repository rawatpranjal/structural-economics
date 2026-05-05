#!/usr/bin/env python3
"""Herfindahl-Hirschman Index (HHI) and merger screening.

Computes concentration measures, merger-induced delta-HHI, and a small
Bertrand example that separates ownership aggregation from price effects.

Reference: U.S. Department of Justice & FTC, Merger Guidelines (2023).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
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


def effective_firms(shares):
    """Equivalent number of equal-sized firms implied by the HHI."""
    hhi = compute_hhi(shares)
    return 10000 / hhi


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
    """Classify HHI using the 2023 DOJ/FTC structural thresholds."""
    if hhi < 1000:
        return "Unconcentrated"
    elif hhi <= 1800:
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
            "Effective N": f"{effective_firms(shares):.2f}",
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
    seg_price_change = (p_seg_merge[:2].mean() / p_seg_base[:2].mean() - 1) * 100
    diff_price_change = (p_diff_merge[:2].mean() / p_diff_base[:2].mean() - 1) * 100
    seg_output_change = (q_seg_merge.sum() / q_seg_base.sum() - 1) * 100
    diff_output_change = (q_diff_merge.sum() / q_diff_base.sum() - 1) * 100

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
        "HHI, Effective Firms, and Merger Screens",
        "Market concentration is a useful antitrust screen, but it is not a model of "
        "competitive effects.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "HHI answers a narrow but important question: how concentrated is control of sales "
        "in a relevant market? It is cheap to compute, transparent to explain, and therefore "
        "useful as an early merger screen. The same simplicity is also its limitation. HHI "
        "knows ownership shares, not diversion ratios, entry, efficiencies, or demand "
        "curvature.\n\n"
        "This tutorial keeps that distinction explicit. The first part works through the "
        "index arithmetic, including the effective number of equal-sized firms implied by a "
        "given HHI. The second part puts the same ownership change inside a four-product "
        "Bertrand model. When products are segmented, HHI rises but prices do not move. When "
        "products substitute, common ownership changes the pricing FOC. For the fuller "
        "counterfactual exercise, compare the neighboring "
        "[merger simulation](../merger-simulation/) and "
        "[logit supply-side](../logit-supply-side/) tutorials."
    )

    report.add_equations(
        r"""
Let firms be indexed by $f=1,\ldots,F$, with market shares $s_f$ measured as
fractions that sum to one. The Herfindahl-Hirschman Index is

$$
\text{HHI}=10{,}000\sum_{f=1}^{F}s_f^2.
$$

The associated effective number of equal-sized firms is

$$
N_{\text{eff}}=\frac{1}{\sum_f s_f^2}=\frac{10{,}000}{\text{HHI}}.
$$

Thus a market with HHI 2,000 has the same concentration as five symmetric
firms, even if the actual firm count is different.

If firms $a$ and $b$ merge while all quantities are held fixed, the arithmetic
change is

$$
\Delta\text{HHI}
=10{,}000[(s_a+s_b)^2-s_a^2-s_b^2]
=20{,}000 s_a s_b.
$$

For product-level data, product $j$ belongs to firm $f(j)$ and sells quantity
$q_j$. Firm shares aggregate product quantities:

$$
s_f=\frac{\sum_{j:f(j)=f}q_j}{\sum_{\ell}q_{\ell}}.
$$

The small structural comparison uses linear differentiated-products demand,

$$
q(p)=a+Dp,\qquad D_{jj}=\alpha<0,\quad D_{jk}=\beta\geq 0\ (j\neq k).
$$

Let $\Omega_{jk}=1$ if products $j$ and $k$ are commonly owned. Bertrand-Nash
prices satisfy

$$
q(p)+(\Omega\circ D^\top)(p-c)=0.
$$

The 2023 DOJ/FTC Merger Guidelines treat HHI above 1,800 as highly
concentrated and an HHI increase above 100 points as significant for the
structural presumption. The tutorial also reports the familiar category scale:
below 1,000 is unconcentrated, 1,000 to 1,800 is moderately concentrated, and
above 1,800 is highly concentrated.
"""
    )

    report.add_model_setup(
        "The index calculations use share vectors chosen to isolate firm count from "
        "asymmetry. The Bertrand comparison uses four products. Initially each product is "
        "owned by a separate firm; the merger puts products 1 and 2 under common ownership.\n\n"
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| $F$ | 1 to 100 | Firm counts for the symmetric HHI benchmark |\n"
        f"| $s_f$ | Several share vectors | Firm shares used in the concentration table |\n"
        f"| Products | 4 | Two high-demand products and two lower-demand products |\n"
        f"| $\\alpha$ | {alpha:.1f} | Own-price slope in linear demand |\n"
        f"| $\\beta_{{\\text{{seg}}}}$ | {beta_seg:.1f} | No cross-price substitution |\n"
        f"| $\\beta_{{\\text{{diff}}}}$ | {beta_diff:.1f} | Positive cross-price substitution |\n"
        f"| Merger | products 1 and 2 | Same ownership change in both demand environments |"
    )

    report.add_solution_method(
        "The concentration part is exact arithmetic. The only equilibrium computation is "
        "the four-product pricing problem, where the ownership matrix changes the markup "
        "equation.\n\n"
        "```text\n"
        "Inputs: firm shares s, product quantities q, costs c, demand slopes D, ownership f(j)\n"
        "Outputs: HHI, effective firm count, delta-HHI, equilibrium prices\n\n"
        "1. For each market, compute HHI = 10000 * sum_f s_f^2.\n"
        "2. Report N_eff = 10000 / HHI to put asymmetric markets on a symmetric scale.\n"
        "3. For a candidate merger (a,b), compute delta-HHI = 20000 * s_a * s_b.\n"
        "4. In product data, aggregate q_j to firm shares using the ownership map f(j).\n"
        "5. Build Omega_jk = 1[f(j) = f(k)].\n"
        "6. Solve q(p) + (Omega .* D') (p - c) = 0 for Bertrand-Nash prices.\n"
        "7. Recompute firm shares and HHI under the post-merger ownership map.\n"
        "```\n\n"
        "Step 6 is solved by root finding. In this linear example the root is a numerical "
        "way to solve a small system of first-order conditions, not the economic point of "
        "the tutorial. The economic point is that HHI is an ownership screen, while the "
        "pricing effect appears only through substitution and the Bertrand FOC."
    )

    report.add_results(
        "The screen and the pricing model give different objects. In the segmented case, "
        "the merged products are independent, so prices and total quantity are unchanged. "
        "HHI still jumps because the two product shares are now counted under one owner. "
        "With positive cross-price substitution, common ownership also changes the pricing "
        "problem, so the merged products become more expensive.\n\n"
        f"| Demand environment | HHI before | HHI after | $\\Delta$HHI | "
        f"Merged-price change | Total-output change |\n"
        f"|---|---:|---:|---:|---:|---:|\n"
        f"| Segmented ($\\beta={beta_seg:.1f}$) | {seg_hhi_before:.0f} | "
        f"{seg_hhi_after:.0f} | {seg_hhi_after - seg_hhi_before:.0f} | "
        f"{seg_price_change:.2f}% | {seg_output_change:.2f}% |\n"
        f"| Differentiated ($\\beta={beta_diff:.1f}$) | {diff_hhi_before:.0f} | "
        f"{diff_hhi_after:.0f} | {diff_hhi_after - diff_hhi_before:.0f} | "
        f"{diff_price_change:.2f}% | {diff_output_change:.2f}% |"
    )

    # --- Figure 1: HHI vs number of equal-sized firms ---
    fig1, ax1 = plt.subplots()
    ax1.plot(n_firms_range, hhi_equal, "b-", linewidth=2)
    # Shade threshold regions
    ax1.axhspan(0, 1000, alpha=0.10, color="green", label="Unconcentrated (< 1000)")
    ax1.axhspan(1000, 1800, alpha=0.10, color="orange", label="Moderate (1000-1800)")
    ax1.axhspan(1800, 10500, alpha=0.10, color="red", label="Highly Concentrated (> 1800)")
    ax1.set_xlabel("Number of Equal-Sized Firms ($N$)")
    ax1.set_ylabel("HHI")
    ax1.set_title("HHI on the Equal-Sized-Firm Scale")
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
        "HHI equals 10000/N for equal-sized firms, with DOJ/FTC threshold regions shaded",
        fig1,
        description="For symmetric firms, HHI is exactly $10{,}000/N$. Moving from monopoly "
        "to five equal firms does most of the work: HHI falls from 10,000 to 2,000. The "
        "highly concentrated threshold of 1,800 corresponds to about 5.6 equal-sized firms, "
        "while the unconcentrated threshold of 1,000 corresponds to ten equal firms.",
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
    ax2.axhline(y=1000, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax2.axhline(y=1800, color="orange", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(len(merger_results) - 0.5, 1060, "Unconcentrated threshold (1000)", fontsize=8, color="green")
    ax2.text(len(merger_results) - 0.5, 1860, "Highly concentrated threshold (1800)", fontsize=8, color="orange")
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
        description="The merger bars are pure index arithmetic. The same formula, "
        "$20{,}000 s_a s_b$, makes a 40-30 merger much larger than a merger of two small firms. "
        "That is why HHI is informative as a first screen, even before estimating demand.",
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
        description="The Lorenz curves show what the scalar index compresses. Equal firms stay "
        "on the diagonal. A dominant firm bends the curve away from the diagonal because many "
        "firms account for little output while one firm accounts for most of it. The table "
        "below turns the same share vectors into HHI and effective firm counts.",
    )

    # --- Table 1: Example markets ---
    report.add_table(
        "tables/market-hhi.csv",
        "HHI for Example Market Structures",
        df_markets,
        description="The effective firm count makes asymmetry visible. A 70-10-10-10 market "
        "has four firms, but its HHI of 5,200 is equivalent to fewer than two equal-sized "
        "firms. Firm count alone would miss that concentration.",
    )

    report.add_takeaway(
        "HHI is valuable because it is transparent: it converts shares into a concentration "
        "number and gives a closed-form delta for mergers. But the segmented-product example "
        "is the warning label. Ownership aggregation can raise HHI even when the maintained "
        "demand model implies no price effect. Once products substitute, the same ownership "
        "change works through the Bertrand FOC and prices move. In applied work, HHI should "
        "start the antitrust conversation, not end it."
    )

    report.add_references([
        "U.S. Department of Justice & Federal Trade Commission (2023). "
        "*Merger Guidelines*.",
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
