#!/usr/bin/env python3
"""Nested Logit Demand Model: Breaking the IIA Assumption.

Estimates a nested logit demand system for a synthetic cereal market where
products are grouped into nests (sugary vs healthy). The nesting parameter
sigma controls within-nest correlation in unobserved tastes, allowing closer
substitutes to have higher cross-price elasticities -- the simplest fix for
the IIA problem of the plain logit.

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
from lib.plotting import setup_style, save_figure
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
            markup = np.random.uniform(0.3, 0.8)
            price = mc * (1 + markup)
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

def compute_diversion_ratios(elasticity_matrix, ref_idx):
    """Fraction of lost sales from product ref_idx going to each rival."""
    J = elasticity_matrix.shape[0]
    own = abs(elasticity_matrix[ref_idx, ref_idx])
    ratios = {}
    for j in range(J):
        if j != ref_idx:
            ratios[j] = elasticity_matrix[j, ref_idx] / own
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


def fig_cross_elasticity_comparison(logit_eta, nested_eta, names, nest_ids, ref=0):
    """Figure 2: Bar chart -- logit vs nested logit cross-elasticities."""
    ref_name = names[ref]
    ref_nest = nest_ids[ref]
    others = [i for i in range(len(names)) if i != ref]

    logit_vals = [logit_eta[i, ref] for i in others]
    nested_vals = [nested_eta[i, ref] for i in others]
    other_names = [names[i] for i in others]

    x = np.arange(len(others))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_nested = ["#27ae60" if nest_ids[i] == ref_nest else "#e74c3c" for i in others]
    ax.bar(x - w / 2, logit_vals, w, label="Plain Logit (IIA)", color="#3498db",
           edgecolor="black", alpha=0.8)
    bars2 = ax.bar(x + w / 2, nested_vals, w, label="Nested Logit",
                   color=colors_nested, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(other_names)
    ax.set_xlabel("Product gaining sales")
    ax.set_ylabel("Cross-price elasticity")
    ax.set_title(f"Where do customers go when {ref_name} raises its price?")

    legend_el = [
        mpatches.Patch(facecolor="#3498db", edgecolor="black", label="Plain Logit (IIA)"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="black", label="Nested: same nest"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", label="Nested: different nest"),
    ]
    ax.legend(handles=legend_el, loc="upper right")
    fig.tight_layout()
    return fig


def fig_diversion_ratios(logit_eta, nested_eta, names, nest_ids, ref=0):
    """Figure 3: Diversion ratio comparison (logit vs nested)."""
    ref_name = names[ref]
    others = [i for i in range(len(names)) if i != ref]
    div_logit = compute_diversion_ratios(logit_eta, ref)
    div_nested = compute_diversion_ratios(nested_eta, ref)

    other_names = [names[i] for i in others]
    logit_vals = [div_logit[i] * 100 for i in others]
    nested_vals = [div_nested[i] * 100 for i in others]

    x = np.arange(len(others))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, logit_vals, w, label="Plain Logit", color="#3498db",
           edgecolor="black", alpha=0.8)
    colors = ["#27ae60" if nest_ids[i] == nest_ids[ref] else "#e74c3c" for i in others]
    ax.bar(x + w / 2, nested_vals, w, label="Nested Logit",
           color=colors, edgecolor="black")

    # Value labels
    for xi, lv, nv in zip(x, logit_vals, nested_vals):
        ax.text(xi - w / 2, lv + 0.5, f"{lv:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.text(xi + w / 2, nv + 0.5, f"{nv:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(other_names)
    ax.set_ylabel("Diversion ratio (%)")
    ax.set_title(f"Diversion ratios when {ref_name} raises its price")
    legend_el = [
        mpatches.Patch(facecolor="#3498db", edgecolor="black", label="Plain Logit"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="black", label="Nested: same nest"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", label="Nested: different nest"),
    ]
    ax.legend(handles=legend_el, loc="upper right")
    fig.tight_layout()
    return fig


def fig_sigma_effect():
    """Figure 4: How sigma changes the elasticity pattern (three panels)."""
    delta = np.array([1.0, 0.8, 0.5, 0.3])
    nids = np.array([1, 1, 2, 2])
    prices = np.array([3.0, 2.0, 5.0, 4.0])
    alpha = 1.5
    labels = ["A (Nest 1)", "B (Nest 1)", "C (Nest 2)", "D (Nest 2)"]

    sigmas = [0.01, 0.5, 0.9]
    titles = [r"$\sigma = 0$ (plain logit)", r"$\sigma = 0.5$ (moderate)",
              r"$\sigma = 0.9$ (strong nesting)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, sig, title in zip(axes, sigmas, titles):
        st, sw, _ = compute_total_shares(delta, nids, sig)
        eta = compute_nested_elasticities(alpha, sig, prices, st, sw, nids)
        vmax, vmin = 3.0, -8.0
        im = ax.imshow(eta, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(4):
            for j in range(4):
                v = eta[i, j]
                c = "white" if abs(v) > 2 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=c)
        # Nest block outlines
        rect1 = plt.Rectangle((-0.5, -0.5), 2, 2, fill=False, edgecolor="gold", linewidth=2)
        rect2 = plt.Rectangle((1.5, 1.5), 2, 2, fill=False, edgecolor="gold", linewidth=2)
        ax.add_patch(rect1)
        ax.add_patch(rect2)

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Elasticity", shrink=0.8)
    fig.suptitle("Effect of nesting parameter on substitution patterns",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.22, wspace=0.35)
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
    div_logit = compute_diversion_ratios(logit_eta, choco_idx)
    div_nested = compute_diversion_ratios(nested_eta, choco_idx)

    print("\n  Diversion ratios (Choco-Bombs raises price):")
    print(f"  {'Product':<18} {'Logit':>10} {'Nested':>10} {'Nest':>12}")
    for j in sorted(div_logit.keys()):
        same = "SAME" if nids[j] == nids[choco_idx] else "different"
        print(f"  {names[j]:<18} {div_logit[j]*100:>9.1f}% {div_nested[j]*100:>9.1f}%  {same:>10}")

    # =========================================================================
    # Step 6 -- Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Nested Logit Demand Model",
        "The simplest fix for the IIA problem: grouping products into nests "
        "so that closer substitutes have higher cross-price elasticities.",
    )

    report.add_overview(
        "The plain logit imposes the Independence of Irrelevant Alternatives (IIA): "
        "the ratio of any two products' market shares is independent of the attributes "
        "of all other products. This means a price increase for Choco-Bombs sends "
        "customers to Fiber-Bran and Store-Frosted in proportion to their market shares, "
        "regardless of how similar those products are.\n\n"
        "The **nested logit** fixes this by grouping products into nests (e.g., sugary "
        "vs healthy cereals). Within a nest, products are closer substitutes. The nesting "
        "parameter $\\sigma \\in [0,1)$ controls the degree of within-nest correlation:\n"
        "- $\\sigma = 0$: collapses to plain logit (IIA holds)\n"
        "- $\\sigma \\to 1$: products within a nest are perfect substitutes\n\n"
        "This is the first step in the Berry (1994) hierarchy: logit $\\to$ nested logit "
        "$\\to$ random-coefficients logit (BLP)."
    )

    report.add_equations(r"""
$$s_j = s_{j|g} \cdot s_g$$

**Within-nest share:**
$$s_{j|g} = \frac{\exp\!\bigl(\delta_j / (1-\sigma)\bigr)}{D_g}, \qquad D_g = \sum_{k \in g} \exp\!\bigl(\delta_k / (1-\sigma)\bigr)$$

**Nest share:**
$$s_g = \frac{D_g^{\,1-\sigma}}{1 + \sum_h D_h^{\,1-\sigma}}$$

**Berry inversion (estimation equation):**
$$\ln s_j - \ln s_0 = \mathbf{x}_j \beta - \alpha \, p_j + \sigma \ln s_{j|g} + \xi_j$$

Both $p_j$ and $\ln s_{j|g}$ are endogenous; we instrument with cost shifters,
rival characteristics, number of products in nest, and same-nest rival characteristics.
""")

    report.add_model_setup(
        "| Parameter | Value | Description |\n"
        "|-----------|-------|-------------|\n"
        f"| $\\alpha$ | {TRUE_ALPHA} | Price sensitivity |\n"
        f"| $\\beta_{{\\text{{sugar}}}}$ | {TRUE_BETA_SUGAR} | Taste for sugar |\n"
        f"| $\\beta_0$ | {TRUE_BETA_CONST} | Base utility |\n"
        f"| $\\sigma$ | {TRUE_SIGMA} | Nesting parameter |\n"
        "| Products | 4 (2 nests of 2) | Sugary: Choco-Bombs, Store-Frosted; Healthy: Fiber-Bran, Granola-Crunch |\n"
        f"| Markets | {df['market_id'].nunique()} | Cross-sectional variation for IV |"
    )

    report.add_solution_method(
        "**Two-Stage Least Squares (2SLS)** on the Berry-inverted equation.\n\n"
        "The nested logit introduces a second endogenous variable, $\\ln s_{j|g}$, "
        "because within-nest shares depend on unobserved quality $\\xi_j$. We need "
        "instruments for **both** price and within-nest share:\n\n"
        "| Instrument | Targets | Rationale |\n"
        "|------------|---------|----------|\n"
        "| Cost shifter | Price | Supply-side variation |\n"
        "| Rival sugar (all) | Price | BLP-style characteristic sum |\n"
        "| Number of products in nest | $\\ln s_{j|g}$ | Affects within-nest competition |\n"
        "| Same-nest rival sugar | $\\ln s_{j|g}$ | Within-nest characteristic variation |"
    )

    # --- Figure 1: Elasticity heatmap ---
    fig1 = fig_elasticity_heatmap(nested_eta, names, nids)
    report.add_figure(
        "figures/elasticity-heatmap.png",
        "Nested logit elasticity matrix with nest blocks highlighted. "
        "Same-nest cross-elasticities (inside gold boxes) are higher than "
        "cross-nest elasticities.",
        fig1,
    )

    # --- Figure 2: Cross-elasticity comparison ---
    fig2 = fig_cross_elasticity_comparison(logit_eta, nested_eta, names, nids, ref=choco_idx)
    report.add_figure(
        "figures/cross-elasticity-comparison.png",
        "Logit vs nested logit cross-elasticities when Choco-Bombs raises its price. "
        "Nested logit sends more customers to Store-Frosted (same nest).",
        fig2,
    )

    # --- Figure 3: Diversion ratios ---
    fig3 = fig_diversion_ratios(logit_eta, nested_eta, names, nids, ref=choco_idx)
    report.add_figure(
        "figures/diversion-ratios.png",
        "Diversion ratios: fraction of Choco-Bombs' lost sales captured by each rival. "
        "Nested logit predicts much higher diversion to same-nest products.",
        fig3,
    )

    # --- Figure 4: Sigma effect ---
    fig4 = fig_sigma_effect()
    report.add_figure(
        "figures/sigma-effect.png",
        "Effect of the nesting parameter sigma on substitution patterns. "
        "As sigma increases, within-nest substitution intensifies while "
        "cross-nest substitution stays flat.",
        fig4,
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
                     "Parameter estimates: true values vs plain logit vs nested logit", tdf)

    report.add_takeaway(
        "The nested logit is the simplest departure from the IIA assumption. By "
        "grouping products into nests, it allows consumers who leave one product to "
        "disproportionately switch to similar products rather than spreading evenly "
        "across the market.\n\n"
        "**Key insights:**\n"
        "- The nesting parameter $\\sigma$ controls within-nest correlation. We "
        f"estimated $\\hat{{\\sigma}} = {nested_res['sigma']:.3f}$ vs the true "
        f"value of {TRUE_SIGMA}, confirming that products within a nest are "
        "closer substitutes.\n"
        "- Same-nest cross-price elasticities are **higher** than cross-nest "
        "elasticities. When Choco-Bombs raises its price, customers primarily "
        "switch to Store-Frosted (also sugary), not to Fiber-Bran.\n"
        "- The plain logit gets the overall price sensitivity roughly right but "
        "completely misses the substitution pattern -- this matters for merger "
        "analysis and targeted pricing.\n"
        "- Estimation requires instruments for **both** price and the within-nest "
        "share $\\ln s_{j|g}$, since both are endogenous. Standard BLP-style "
        "instruments (rival characteristics, nest size) serve this purpose."
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
