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
from lib.plotting import setup_style, save_figure, save_thumbnail


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
    """Generate a panel of cereal products across markets with nest assignments.

    Two characteristics carry genuine cross-market variation so that
    instruments built from them identify the endogenous regressors:

    * ``sugar`` is a product-specific base recipe plus a small market-level
      reformulation shock, so a product is slightly sweeter in some markets.
    * ``cost_idio`` is the firm-and-market idiosyncratic cost component (the
      part of marginal cost on top of the market-wide ``cost_shifter``).
    """
    np.random.seed(42)

    products = {
        "product_id": [1, 2, 3, 4],
        "product_name": ["Choco-Bombs", "Store-Frosted", "Fiber-Bran", "Granola-Crunch"],
        "sugar_base": [10.0, 8.0, 1.0, 2.0],
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
            # Market-level recipe shock: same product is sweeter in some markets.
            sugar = products["sugar_base"][j] + np.random.normal(0, 1.0)
            # Idiosyncratic (firm x market) cost component on top of cost_shock.
            cost_idio = np.random.normal(0, 0.1)
            mc = products["marginal_cost_base"][j] + cost_shock + cost_idio
            price_factor = np.random.uniform(0.3, 0.8)
            price = mc * (1 + price_factor)
            rows.append({
                "market_id": t,
                "product_id": products["product_id"][j],
                "product_name": products["product_name"][j],
                "sugar": sugar,
                "xi": products["xi"][j],
                "firm_id": products["firm_id"][j],
                "nest_id": products["nest_id"][j],
                "nest_name": products["nest_name"][j],
                "price": price,
                "marginal_cost": mc,
                "cost_shifter": cost_shock,
                "cost_idio": cost_idio,
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
    """Create IVs for price and within-nest share (both endogenous).

    All three excluded instruments carry genuine cross-market variation:

    * ``rival_sugar_sum`` -- total sugar of all rival products in the market.
      Sugar now varies by market, so this is a BLP-style characteristic
      instrument for price, not a product fixed effect.
    * ``same_nest_rival_sugar`` -- sugar of same-nest rivals. It shifts the
      attractiveness of close substitutes and so moves the within-nest share.
    * ``same_nest_rival_cost`` -- idiosyncratic cost of same-nest rivals. A
      rival's cost shock moves the rival's price, hence the rival's share,
      hence the conditional within-nest share of product j, while being
      excluded from product j's own utility.
    """
    df = df.copy()
    rival_sugar = []
    same_nest_rival_sugar = []
    same_nest_rival_cost = []

    for _, row in df.iterrows():
        mkt = df[df["market_id"] == row["market_id"]]
        others = mkt[mkt["product_id"] != row["product_id"]]
        rival_sugar.append(others["sugar"].sum())

        same_nest = mkt[mkt["nest_id"] == row["nest_id"]]
        same_nest_others = same_nest[same_nest["product_id"] != row["product_id"]]
        same_nest_rival_sugar.append(same_nest_others["sugar"].sum())
        same_nest_rival_cost.append(same_nest_others["cost_idio"].sum())

    df["rival_sugar_sum"] = rival_sugar
    df["same_nest_rival_sugar"] = same_nest_rival_sugar
    df["same_nest_rival_cost"] = same_nest_rival_cost
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
    Z = df[["cost_shifter", "rival_sugar_sum", "same_nest_rival_cost", "same_nest_rival_sugar"]].values
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
    # Step 6 -- Figures and tables
    # =========================================================================
    setup_style()

    # --- Figure 1: Elasticity heatmap ---
    fig1 = fig_elasticity_heatmap(nested_eta, names, nids)
    save_figure(fig1, "figures/elasticity-heatmap.png", dpi=150)

    # --- Figure 2: Cross-elasticity comparison ---
    fig2 = fig_cross_elasticity_comparison(logit_eta, nested_eta, true_eta, names, nids, ref=choco_idx)
    save_figure(fig2, "figures/cross-elasticity-comparison.png", dpi=150)

    # --- Figure 3: Diversion ratios ---
    fig3 = fig_diversion_ratios(logit_eta, nested_eta, true_eta, shares, names, nids, ref=choco_idx)
    save_figure(fig3, "figures/diversion-ratios.png", dpi=150)

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
    Path("tables").mkdir(parents=True, exist_ok=True)
    tdf.to_csv("tables/parameter-estimates.csv", index=False)

    save_thumbnail("figures/elasticity-heatmap.png", "figures/thumb.png")
    print(f"\nDone: figures/ + tables/")


if __name__ == "__main__":
    main()
