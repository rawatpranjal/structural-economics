#!/usr/bin/env python3
"""Cereal demand with nested logit substitution.

Estimates a nested logit demand system for a synthetic cereal market. Products
belong to sugary and healthy nests, so substitution can be stronger inside a
group than across the whole product space.

References: Berry (1994), McFadden (1978).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style
from lib.output import ModelReport


# =============================================================================
# True parameters (data-generating process)
# =============================================================================
TRUE_ALPHA = 1.5       # Price sensitivity
TRUE_BETA_SUGAR = 0.3  # Taste for sugar
TRUE_BETA_CONST = 1.0  # Base utility constant
TRUE_SIGMA = 0.7       # Nesting parameter (0 = logit, 1 = perfect within-nest)


# =============================================================================
# Synthetic data generation
# =============================================================================

def generate_product_data(n_markets: int = 30) -> pd.DataFrame:
    """Generate a panel of cereal products across markets with nest assignments."""
    np.random.seed(42)

    products = {
        "product_id": [1, 2, 3, 4],
        "product_name": ["Choco-Bombs", "Store-Frosted", "Fiber-Bran", "Granola-Crunch"],
        "sugar": [10.0, 8.0, 1.0, 2.0],
        "xi": [0.5, -0.1, 0.3, 0.1],
        "firm_id": [1, 2, 3, 4],
        "nest_id": [1, 1, 2, 2],
        "nest_name": ["Sugary", "Sugary", "Healthy", "Healthy"],
        "marginal_cost_base": [1.5, 1.0, 2.5, 2.0],
    }

    rows = []
    for t in range(n_markets):
        cost_shock = np.random.normal(0, 0.3)
        for j in range(len(products["product_id"])):
            mc = products["marginal_cost_base"][j] + cost_shock + np.random.normal(0, 0.1)
            price_factor = np.random.uniform(0.3, 0.8)
            price = mc * (1 + price_factor)
            rows.append({
                "market_id": t,
                "product_id": products["product_id"][j],
                "product_name": products["product_name"][j],
                "sugar": products["sugar"][j],
                "xi": products["xi"][j],
                "firm_id": products["firm_id"][j],
                "nest_id": products["nest_id"][j],
                "nest_name": products["nest_name"][j],
                "price": price,
                "marginal_cost": mc,
                "cost_shifter": cost_shock,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Share computations (nested logit)
# =============================================================================

def compute_total_shares(delta: np.ndarray, nest_ids: np.ndarray,
                         sigma: float):
    """Compute s_j = s_{j|g} * s_g for all products.

    Returns (total_shares, within_nest_shares, outside_share).
    """
    J = len(delta)
    # Inclusive values D_g = sum_{k in g} exp(delta_k / (1-sigma))
    D = {}
    for gid in np.unique(nest_ids):
        mask = nest_ids == gid
        D[gid] = np.exp(delta[mask] / (1 - sigma)).sum()

    # Nest shares: s_g = D_g^{1-sigma} / [1 + sum_h D_h^{1-sigma}]
    denom = 1.0 + sum(Dg ** (1 - sigma) for Dg in D.values())
    nest_shares = {gid: (Dg ** (1 - sigma)) / denom for gid, Dg in D.items()}
    outside_share = 1.0 / denom

    # Within-nest shares: s_{j|g} = exp(delta_j / (1-sigma)) / D_g
    s_within = np.zeros(J)
    s_total = np.zeros(J)
    for j in range(J):
        gid = nest_ids[j]
        s_within[j] = np.exp(delta[j] / (1 - sigma)) / D[gid]
        s_total[j] = s_within[j] * nest_shares[gid]

    return s_total, s_within, outside_share


def compute_true_shares(df: pd.DataFrame, sigma: float = TRUE_SIGMA) -> pd.DataFrame:
    """Compute market shares from true parameters for every market."""
    df = df.copy()
    df["delta"] = (TRUE_BETA_CONST + TRUE_BETA_SUGAR * df["sugar"]
                   - TRUE_ALPHA * df["price"] + df["xi"])

    for market_id, mdf in df.groupby("market_id"):
        deltas = mdf["delta"].values
        nids = mdf["nest_id"].values
        s_total, s_within, s0 = compute_total_shares(deltas, nids, sigma)
        df.loc[mdf.index, "share"] = s_total
        df.loc[mdf.index, "within_nest_share"] = s_within
        df.loc[mdf.index, "outside_share"] = s0
    return df


# =============================================================================
# Instruments
# =============================================================================

def generate_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """Create IVs for price and within-nest share (both endogenous)."""
    df = df.copy()
    rival_sugar = []
    num_in_nest = []
    same_nest_rival_sugar = []

    for _, row in df.iterrows():
        mkt = df[df["market_id"] == row["market_id"]]
        others = mkt[mkt["product_id"] != row["product_id"]]
        rival_sugar.append(others["sugar"].sum())

        same_nest = mkt[mkt["nest_id"] == row["nest_id"]]
        num_in_nest.append(len(same_nest))

        same_nest_others = same_nest[same_nest["product_id"] != row["product_id"]]
        same_nest_rival_sugar.append(same_nest_others["sugar"].sum())

    df["rival_sugar_sum"] = rival_sugar
    df["num_in_nest"] = num_in_nest
    df["same_nest_rival_sugar"] = same_nest_rival_sugar
    return df


# =============================================================================
# 2SLS estimation
# =============================================================================

def estimate_2sls(Y, X_exog, X_endog, Z):
    """Two-stage least squares. Returns dict with coefficients, std_errors, etc."""
    n = len(Y)
    if X_endog.ndim == 1:
        X_endog = X_endog.reshape(-1, 1)
    W = np.column_stack([np.ones(n), X_exog, Z])
    X = np.column_stack([np.ones(n), X_exog, X_endog])

    # Stage 1: project endogenous onto instrument space
    WtW_inv = np.linalg.pinv(W.T @ W)
    P_W = W @ WtW_inv @ W.T
    X_endog_hat = P_W @ X_endog

    # Stage 2
    X_hat = np.column_stack([np.ones(n), X_exog, X_endog_hat])
    XhX_inv = np.linalg.pinv(X_hat.T @ X_hat)
    beta_hat = XhX_inv @ X_hat.T @ Y

    residuals = Y - X @ beta_hat
    sigma2 = (residuals @ residuals) / max(n - X.shape[1], 1)
    var_beta = sigma2 * XhX_inv
    se = np.sqrt(np.abs(np.diag(var_beta)))

    return {
        "coefficients": beta_hat,
        "std_errors": se,
        "residuals": residuals,
        "t_stats": beta_hat / np.where(se > 0, se, 1e-10),
        "n_obs": n,
    }


def estimate_logit(df):
    """Plain logit via 2SLS (no sigma, ignores nesting)."""
    Y = df["ln_share_ratio"].values
    X_exog = df[["sugar"]].values
    X_endog = df["price"].values
    Z = df[["cost_shifter", "rival_sugar_sum"]].values
    res = estimate_2sls(Y, X_exog, X_endog, Z)
    res["alpha"] = -res["coefficients"][2]
    res["beta_sugar"] = res["coefficients"][1]
    res["beta_const"] = res["coefficients"][0]
    res["sigma"] = 0.0
    return res


def estimate_nested_logit(df):
    """Nested logit via 2SLS -- instruments for price AND ln(s_{j|g})."""
    Y = df["ln_share_ratio"].values
    X_exog = df[["sugar"]].values
    X_endog = df[["price", "ln_within_share"]].values
    Z = df[["cost_shifter", "rival_sugar_sum", "num_in_nest", "same_nest_rival_sugar"]].values
    res = estimate_2sls(Y, X_exog, X_endog, Z)
    # Coefficients: [const, sugar, -alpha, sigma]
    res["alpha"] = -res["coefficients"][2]
    res["beta_sugar"] = res["coefficients"][1]
    res["beta_const"] = res["coefficients"][0]
    res["sigma"] = res["coefficients"][3]
    return res


# =============================================================================
# Elasticities
# =============================================================================

def compute_nested_elasticities(alpha, sigma, prices, shares, within_shares, nest_ids):
    """JxJ elasticity matrix for nested logit (three distinct cases)."""
    J = len(prices)
    eta = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                t1 = 1.0 / (1 - sigma)
                t2 = (1.0 / (1 - sigma) - 1) * within_shares[j]
                t3 = shares[j]
                eta[j, k] = -alpha * prices[j] * (t1 - t2 - t3)
            elif nest_ids[j] == nest_ids[k]:
                # Same nest -- HIGHER cross-elasticity (breaks IIA)
                t1 = (1.0 / (1 - sigma) - 1) * within_shares[k]
                t2 = shares[k]
                eta[j, k] = alpha * prices[k] * (t1 + t2)
            else:
                # Different nest -- same as plain logit
                eta[j, k] = alpha * prices[k] * shares[k]
    return eta


def compute_logit_elasticities(alpha, prices, shares):
    """JxJ elasticity matrix for plain logit (exhibits IIA)."""
    J = len(prices)
    eta = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                eta[j, k] = -alpha * prices[j] * (1 - shares[j])
            else:
                eta[j, k] = alpha * prices[k] * shares[k]
    return eta


# =============================================================================
# Diversion ratios
# =============================================================================

def compute_diversion_ratios(elasticity_matrix, shares, ref_idx):
    """Product-level diversion ratios from a price increase for ref_idx.

    Diversion is a derivative ratio, not an elasticity ratio:
    D_{j<-k} = - (ds_j / dp_k) / (ds_k / dp_k).
    The share terms convert elasticities back into share derivatives.
    """
    J = elasticity_matrix.shape[0]
    own = abs(elasticity_matrix[ref_idx, ref_idx])
    ratios = {}
    for j in range(J):
        if j != ref_idx:
            ratios[j] = elasticity_matrix[j, ref_idx] * shares[j] / (own * shares[ref_idx])
    return ratios


# =============================================================================
# Figures
# =============================================================================

def fig_elasticity_heatmap(eta, names, nest_ids):
    """Figure 1: Nested logit elasticity heatmap with nest-block outlines."""
    J = len(names)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    vmax = max(3.0, np.max(eta) * 1.1)
    vmin = min(-6.0, np.min(eta) * 1.1)
    im = ax.imshow(eta, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Elasticity")

    ax.set_xticks(range(J))
    ax.set_yticks(range(J))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    for i in range(J):
        for j in range(J):
            v = eta[i, j]
            color = "white" if abs(v) > 2.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    # Nest block outlines
    for gid in np.unique(nest_ids):
        idxs = np.where(nest_ids == gid)[0]
        lo, hi = idxs.min() - 0.5, idxs.max() + 0.5
        rect = plt.Rectangle((lo, lo), hi - lo, hi - lo,
                              fill=False, edgecolor="gold", linewidth=3)
        ax.add_patch(rect)

    ax.set_xlabel("Price of product (column)")
    ax.set_ylabel("Quantity of product (row)")
    ax.set_title("Price Elasticity Matrix (Nested Logit)")
    legend_el = [mpatches.Patch(facecolor="none", edgecolor="gold",
                                linewidth=3, label="Same-nest block")]
    ax.legend(handles=legend_el, loc="upper right")
    fig.tight_layout()
    return fig


def fig_cross_elasticity_comparison(logit_eta, nested_eta, true_eta, names, nest_ids, ref=0):
    """Figure 2: Bar chart -- logit vs estimated and true nested cross-elasticities."""
    ref_name = names[ref]
    ref_nest = nest_ids[ref]
    others = [i for i in range(len(names)) if i != ref]

    logit_vals = [logit_eta[i, ref] for i in others]
    nested_vals = [nested_eta[i, ref] for i in others]
    true_vals = [true_eta[i, ref] for i in others]
    other_names = [names[i] for i in others]

    x = np.arange(len(others))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_nested = ["#27ae60" if nest_ids[i] == ref_nest else "#e74c3c" for i in others]
    ax.bar(x - w, logit_vals, w, label="Plain Logit", color="#3498db",
           edgecolor="black", alpha=0.8)
    ax.bar(x, nested_vals, w, label="Nested Logit (2SLS)",
           color=colors_nested, edgecolor="black")
    ax.bar(x + w, true_vals, w, label="True Nested",
           color="#f2f2f2", edgecolor="black", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(other_names)
    ax.set_xlabel("Product gaining sales")
    ax.set_ylabel("Cross-price elasticity")
    ax.set_title(f"Where do customers go when {ref_name} raises its price?")

    legend_el = [
        mpatches.Patch(facecolor="#3498db", edgecolor="black", label="Plain Logit"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="black", label="Nested 2SLS: same nest"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", label="Nested 2SLS: different nest"),
        mpatches.Patch(facecolor="#f2f2f2", edgecolor="black", hatch="//", label="True nested"),
    ]
    ax.legend(handles=legend_el, loc="upper right")
    fig.tight_layout()
    return fig


def fig_diversion_ratios(logit_eta, nested_eta, true_eta, shares, names, nest_ids, ref=0):
    """Figure 3: Product diversion ratios for a price increase in one product."""
    ref_name = names[ref]
    others = [i for i in range(len(names)) if i != ref]
    div_logit = compute_diversion_ratios(logit_eta, shares, ref)
    div_nested = compute_diversion_ratios(nested_eta, shares, ref)
    div_true = compute_diversion_ratios(true_eta, shares, ref)

    other_names = [names[i] for i in others]
    logit_vals = [div_logit[i] * 100 for i in others]
    nested_vals = [div_nested[i] * 100 for i in others]
    true_vals = [div_true[i] * 100 for i in others]

    x = np.arange(len(others))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w, logit_vals, w, label="Plain Logit", color="#3498db",
           edgecolor="black", alpha=0.8)
    colors = ["#27ae60" if nest_ids[i] == nest_ids[ref] else "#e74c3c" for i in others]
    ax.bar(x, nested_vals, w, label="Nested Logit (2SLS)",
           color=colors, edgecolor="black")
    ax.bar(x + w, true_vals, w, label="True Nested",
           color="#f2f2f2", edgecolor="black", hatch="//")

    for xi, lv, nv, tv in zip(x, logit_vals, nested_vals, true_vals):
        ax.text(xi - w, lv + 0.5, f"{lv:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(xi, nv + 0.5, f"{nv:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w, tv + 0.5, f"{tv:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(other_names)
    ax.set_ylabel("Diversion ratio (%)")
    ax.set_title(f"Diversion ratios when {ref_name} raises its price")
    legend_el = [
        mpatches.Patch(facecolor="#3498db", edgecolor="black", label="Plain Logit"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="black", label="Nested 2SLS: same nest"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", label="Nested 2SLS: different nest"),
        mpatches.Patch(facecolor="#f2f2f2", edgecolor="black", hatch="//", label="True nested"),
    ]
    ax.legend(handles=legend_el, loc="upper right")
    fig.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    # =========================================================================
    # Step 1 -- Generate data
    # =========================================================================
    print("Generating synthetic cereal market data ...")
    df = generate_product_data(n_markets=50)
    df = compute_true_shares(df)
    df = generate_instruments(df)

    # Dependent variable for estimation
    df["ln_share_ratio"] = np.log(df["share"]) - np.log(df["outside_share"])
    df["ln_within_share"] = np.log(df["within_nest_share"])

    print(f"  {df.shape[0]} obs, {df['market_id'].nunique()} markets, "
          f"{df['product_id'].nunique()} products")

    # =========================================================================
    # Step 2 -- Estimate plain logit (for comparison)
    # =========================================================================
    print("\nEstimating plain logit (ignores nesting) ...")
    logit_res = estimate_logit(df)
    print(f"  alpha = {logit_res['alpha']:.4f}  (true {TRUE_ALPHA})")
    print(f"  beta_sugar = {logit_res['beta_sugar']:.4f}  (true {TRUE_BETA_SUGAR})")
    print(f"  beta_const = {logit_res['beta_const']:.4f}  (true {TRUE_BETA_CONST})")

    # =========================================================================
    # Step 3 -- Estimate nested logit
    # =========================================================================
    print("\nEstimating nested logit ...")
    nested_res = estimate_nested_logit(df)
    print(f"  alpha = {nested_res['alpha']:.4f}  (true {TRUE_ALPHA})")
    print(f"  beta_sugar = {nested_res['beta_sugar']:.4f}  (true {TRUE_BETA_SUGAR})")
    print(f"  beta_const = {nested_res['beta_const']:.4f}  (true {TRUE_BETA_CONST})")
    print(f"  sigma = {nested_res['sigma']:.4f}  (true {TRUE_SIGMA})")

    # =========================================================================
    # Step 4 -- Elasticities (first market)
    # =========================================================================
    print("\nComputing elasticities (market 0) ...")
    m0 = df[df["market_id"] == 0].reset_index(drop=True)
    names = m0["product_name"].tolist()
    prices = m0["price"].values
    shares = m0["share"].values
    ws = m0["within_nest_share"].values
    nids = m0["nest_id"].values

    logit_eta = compute_logit_elasticities(logit_res["alpha"], prices, shares)
    nested_eta = compute_nested_elasticities(
        nested_res["alpha"], nested_res["sigma"], prices, shares, ws, nids
    )
    true_eta = compute_nested_elasticities(TRUE_ALPHA, TRUE_SIGMA, prices, shares, ws, nids)

    # Print elasticity matrices
    header = "              " + "  ".join(f"{n[:12]:>12}" for n in names)
    print("\n  Plain Logit elasticity matrix (IIA: cross-elast depend only on share):")
    print(f"  {header}")
    for i, n in enumerate(names):
        row = "  ".join(f"{logit_eta[i, j]:>12.3f}" for j in range(len(names)))
        print(f"  {n[:12]:>12}  {row}")

    print("\n  Nested Logit elasticity matrix (breaks IIA):")
    print(f"  {header}")
    for i, n in enumerate(names):
        row = "  ".join(f"{nested_eta[i, j]:>12.3f}" for j in range(len(names)))
        print(f"  {n[:12]:>12}  {row}")

    # =========================================================================
    # Step 5 -- Diversion ratios
    # =========================================================================
    choco_idx = names.index("Choco-Bombs")
    div_logit = compute_diversion_ratios(logit_eta, shares, choco_idx)
    div_nested = compute_diversion_ratios(nested_eta, shares, choco_idx)
    div_true = compute_diversion_ratios(true_eta, shares, choco_idx)

    print("\n  Diversion ratios (Choco-Bombs raises price):")
    print(f"  {'Product':<18} {'Logit':>10} {'Nested':>10} {'True':>10} {'Nest':>12}")
    for j in sorted(div_logit.keys()):
        same = "SAME" if nids[j] == nids[choco_idx] else "different"
        print(
            f"  {names[j]:<18} {div_logit[j]*100:>9.1f}% "
            f"{div_nested[j]*100:>9.1f}% {div_true[j]*100:>9.1f}%  {same:>10}"
        )

    # =========================================================================
    # Step 6 -- Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Cereal Demand with Nested Logit Substitution",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A supermarket raises the price of Choco-Bombs. Fewer shoppers choose it. "
        "Some switch to Store-Frosted, and some leave the cereal category.\n\n"
        "The object is the diversion pattern. Nested logit groups Choco-Bombs "
        "with Store-Frosted and groups the two healthier cereals. The nesting "
        "parameter $\\sigma$ controls how much substitution stays inside a group.\n\n"
        "The computation starts from shares, prices, nests, and instruments. "
        "Berry inversion gives a linear equation. 2SLS estimates price "
        "sensitivity and $\\sigma$. Elasticities and diversion ratios then show "
        "where lost Choco-Bombs demand goes."
    )

    report.add_equations(r"""
Products $j=1,\ldots,J$ appear in markets $t=1,\ldots,T$. Product $j$ belongs
to nest $g(j)$, and $s_{0t}$ is the outside-good share. Mean utility combines a
common inside-good term, sugar content, price, and unobserved product quality:
$$\delta_{jt}=\beta_0+\beta_{\text{sugar}}\text{sugar}_j-\alpha p_{jt}+\xi_j, \qquad \alpha>0 .$$

The inclusive-value denominator aggregates the products inside one nest:
$$D_{gt}=\sum_{k:g(k)=g}\exp\left(\frac{\delta_{kt}}{1-\sigma}\right), \qquad 0\leq \sigma<1 .$$

Total share factors into a conditional share inside the nest and the nest's
overall market share (where $h$ indexes nests in the denominator sum):
$$s_{j|g,t}= \frac{\exp\left(\delta_{jt}/(1-\sigma)\right)}{D_{g(j)t}}, \qquad s_{gt}= \frac{D_{gt}^{1-\sigma}}{1+\sum_h D_{ht}^{1-\sigma}}, \qquad s_{jt}=s_{j|g,t}s_{g(j)t}.$$

The Berry inversion turns observed shares into a linear estimating equation:
$$\ln s_{jt}-\ln s_{0t} = \beta_0+\beta_{\text{sugar}}\text{sugar}_j-\alpha p_{jt} +\sigma\ln s_{j|g,t}+\xi_j .$$
Both $p_{jt}$ and $\ln s_{j|g,t}$ are endogenous in this regression.

After estimation, the substitution object is the elasticity matrix. Rows are
products whose shares change; columns are products whose prices change. For
market $t$,
$$\eta_{jk,t}=\frac{\partial\ln s_{jt}}{\partial\ln p_{kt}}= \begin{cases} -\alpha p_{jt}\left[\dfrac{1}{1-\sigma} -\dfrac{\sigma}{1-\sigma}s_{j|g,t}-s_{jt}\right], & j=k,\\[1.0em] \alpha p_{kt}\left[\dfrac{\sigma}{1-\sigma}s_{k|g,t}+s_{kt}\right], & j\neq k,\ g(j)=g(k),\\[1.0em] \alpha p_{kt}s_{kt}, & g(j)\neq g(k). \end{cases}$$
Diversion ratios convert elasticities into the share loss from product $k$ that
goes to product $j$:
$$\mathcal{D}_{j\leftarrow k}= -\frac{\partial s_{jt}/\partial p_{kt}}{\partial s_{kt}/\partial p_{kt}} = \frac{\eta_{jk,t}s_{jt}}{|\eta_{kk,t}|s_{kt}} .$$
""")

    report.add_model_setup(
        "The synthetic panel has a small cereal category across many markets. "
        "Prices move with a cost shifter. Shares come from the nested-logit model. "
        "The estimator observes prices, sugar, shares, nests, and excluded "
        "shifters.\n\n"
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        f"| Markets $T$ | {df['market_id'].nunique()} | Cross-market price and cost variation |\n"
        "| Inside products $J$ | 4 | Two sugary and two healthy cereals |\n"
        "| Outside good | Included | Pins down the Berry share ratio |\n"
        f"| True $\\alpha$ | {TRUE_ALPHA:.1f} | Price sensitivity in the data-generating model |\n"
        f"| True $\\beta_{{\\text{{sugar}}}}$ | {TRUE_BETA_SUGAR:.1f} | Taste for sugar content |\n"
        f"| True $\\beta_0$ | {TRUE_BETA_CONST:.1f} | Common inside-good utility shifter |\n"
        f"| True $\\sigma$ | {TRUE_SIGMA:.1f} | Extra same-nest substitution |\n"
        "| Nests | Sugary, healthy | Maintained grouping used by nested logit |"
    )

    report.add_solution_method(
        "Nested logit has closed-form shares. Inclusive values summarize each "
        "product group. The estimation step uses 2SLS on the Berry-inverted "
        "regression. Price and within-nest share both move with unobserved "
        "product quality. Plain logit is estimated as the IIA benchmark.\n\n"
        "```text\n"
        "Algorithm: nested-logit IV demand\n"
        "Input: markets t, products j, nests g(j), shares s_jt, outside shares s_0t\n"
        "Output: IV estimates, elasticity matrix, and diversion ratios\n\n"
        "1. For each market, compute within-nest shares s_{j|g,t} from observed shares.\n"
        "2. Form y_jt = log(s_jt) - log(s_0t) and w_jt = log(s_{j|g,t}).\n"
        "3. First stage: project price p_jt and w_jt on sugar and instruments Z_jt.\n"
        "4. Second stage: regress y_jt on sugar, fitted price, and fitted w_jt.\n"
        "5. Read alpha from the negative price coefficient and sigma from w_jt.\n"
        "6. Compute eta_jk,t and calD_{j<-k}; compare plain logit, fitted nested logit,\n"
        "   and the true synthetic nested-logit benchmark.\n"
        "```\n\n"
        "The instruments match the two endogenous variables. Cost variation moves "
        "prices. Rival characteristics and nest composition predict "
        "$\\ln s_{j|g,t}$.\n\n"
        "| Instrument | Targets | Rationale |\n"
        "|---|---|---|\n"
        "| Cost shifter | Price | Moves marginal cost without entering utility directly |\n"
        "| Rival sugar, all products | Price | Summarizes rival characteristics in the market |\n"
        "| Number of products in nest | $\\ln s_{j\\mid g,t}$ | Changes the local competitive set |\n"
        "| Same-nest rival sugar | $\\ln s_{j\\mid g,t}$ | Moves the attractiveness of close substitutes |"
    )

    # --- Figure 1: Elasticity heatmap ---
    fig1 = fig_elasticity_heatmap(nested_eta, names, nids)
    report.add_figure(
        "figures/elasticity-heatmap.png",
        "Nested logit elasticity matrix with nest blocks highlighted. "
        "Same-nest responses are higher than cross-nest responses.",
        fig1,
        description=(
            "Rows are products whose shares respond. Columns are prices that move. "
            "Gold blocks mark the nests. The Choco-Bombs column is largest for "
            "Store-Frosted, so substitution follows product similarity."
        ),
    )

    # --- Figure 2: Cross-elasticity comparison ---
    fig2 = fig_cross_elasticity_comparison(logit_eta, nested_eta, true_eta, names, nids, ref=choco_idx)
    report.add_figure(
        "figures/cross-elasticity-comparison.png",
        "Cross-elasticities when Choco-Bombs raises its price: plain logit, "
        "fitted nested logit, and the true synthetic model.",
        fig2,
        description=(
            "The blue bars show the plain-logit restriction. Cross responses "
            "follow product shares without product closeness. The green and red "
            "bars use 2SLS nested-logit estimates. The hatched bars show the "
            "true synthetic model. It ranks Store-Frosted as the close substitute."
        ),
    )

    # --- Figure 3: Diversion ratios ---
    fig3 = fig_diversion_ratios(logit_eta, nested_eta, true_eta, shares, names, nids, ref=choco_idx)
    report.add_figure(
        "figures/diversion-ratios.png",
        "Product diversion ratios from a Choco-Bombs price increase.",
        fig3,
        description=(
            "Diversion ratios convert elasticities back into share derivatives. "
            "Plain logit sends lost Choco-Bombs demand toward larger rivals. "
            "Nested logit shifts more diversion to Store-Frosted. The remaining "
            "lost demand goes to the outside good."
        ),
    )

    # --- Table: Parameter estimates ---
    table_data = {
        "Parameter": [r"alpha", r"beta_sugar", r"beta_const", r"sigma"],
        "True": [f"{TRUE_ALPHA:.3f}", f"{TRUE_BETA_SUGAR:.3f}",
                 f"{TRUE_BETA_CONST:.3f}", f"{TRUE_SIGMA:.3f}"],
        "Logit": [f"{logit_res['alpha']:.3f}", f"{logit_res['beta_sugar']:.3f}",
                  f"{logit_res['beta_const']:.3f}", "---"],
        "Nested Logit": [f"{nested_res['alpha']:.3f}", f"{nested_res['beta_sugar']:.3f}",
                         f"{nested_res['beta_const']:.3f}", f"{nested_res['sigma']:.3f}"],
    }
    tdf = pd.DataFrame(table_data)
    report.add_table("tables/parameter-estimates.csv",
                     "Parameter estimates: true values vs plain logit vs nested logit", tdf,
                     description=(
                         "The table checks whether estimation recovers the "
                         "parameters used to generate the synthetic shares. "
                         "Plain logit cannot estimate $\\sigma$. Nested logit "
                         "recovers the signs and the same-nest ranking."
                     ))

    report.add_takeaway(
        "Nested logit is useful when product groups are defensible. Here a "
        "Choco-Bombs price increase mainly sends buyers to Store-Frosted. The "
        "nests do real work. The diversion matrix is only as credible as the "
        "grouping."
    )

    report.add_references([
        "Berry, S. (1994). Estimating Discrete-Choice Models of Product "
        "Differentiation. *RAND Journal of Economics*, 25(2), 242--262.",
        "McFadden, D. (1978). Modelling the Choice of Residential Location. "
        "In A. Karlqvist et al. (Eds.), *Spatial Interaction Theory and "
        "Planning Models*. North-Holland.",
        "Train, K. (2009). *Discrete Choice Methods with Simulation*. "
        "Cambridge University Press, 2nd edition, Ch. 4.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + "
          f"{len(report._tables)} tables")


if __name__ == "__main__":
    main()
