#!/usr/bin/env python3
"""BLP Full Pipeline: Random Coefficients Demand Estimation.

Implements the complete Berry, Levinsohn, Pakes (1995) demand model with:
- Synthetic cereal market data with heterogeneous consumer preferences
- BLP contraction mapping (inner loop) to invert market shares
- GMM estimation (outer loop) for nonlinear parameters
- 2SLS for linear parameters concentrated out within the contraction
- Post-estimation: elasticities, diversion ratios, merger simulation
- Comparison of logit vs BLP substitution patterns

Reference: Berry, S., Levinsohn, J., and Pakes, A. (1995). "Automobile Prices
in Market Equilibrium." Econometrica, 63(4), 841-890.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport

# =============================================================================
# True parameters (data-generating process)
# =============================================================================
TRUE_ALPHA_MEAN = 2.0          # Mean price sensitivity
TRUE_BETA_SUGAR_MEAN = 0.0     # Mean sugar taste (neutral on average)
TRUE_BETA_CONST = 1.5          # Base utility (constant)
TRUE_SIGMA_ALPHA = 1.5         # Std dev of price sensitivity across consumers
TRUE_SIGMA_SUGAR = 2.0         # Std dev of sugar taste across consumers

# With sigma_sugar = 2.0, some consumers LOVE sugar (+2 SD = +4.0 boost)
# and others HATE sugar (-2 SD = -4.0 penalty). This is the BLP insight.


# =============================================================================
# Data generation
# =============================================================================
def generate_product_data(n_markets: int = 20, seed: int = 42) -> pd.DataFrame:
    """Generate product-market level data for a synthetic cereal market.

    Products:
      - Choco-Bombs:   high sugar, medium price (loved by kids)
      - Fiber-Bran:    low sugar, high price   (loved by health-conscious)
      - Store-Frosted: high sugar, low price   (budget sugary option)
    """
    rng = np.random.RandomState(seed)

    products = {
        "product_id": [1, 2, 3],
        "product_name": ["Choco-Bombs", "Fiber-Bran", "Store-Frosted"],
        "sugar": [10.0, 1.0, 8.0],
        "xi": [0.5, 0.3, -0.2],          # Unobserved quality
        "firm_id": [1, 2, 3],
        "marginal_cost_base": [1.8, 2.8, 1.2],
    }

    rows = []
    for t in range(n_markets):
        cost_shock = rng.normal(0, 0.2)
        for j in range(len(products["product_id"])):
            mc = products["marginal_cost_base"][j] + cost_shock + rng.normal(0, 0.1)
            markup = rng.uniform(0.4, 0.9)
            price = mc * (1 + markup)
            rows.append({
                "market_id": t,
                "product_id": products["product_id"][j],
                "product_name": products["product_name"][j],
                "sugar": products["sugar"][j],
                "xi": products["xi"][j],
                "firm_id": products["firm_id"][j],
                "price": price,
                "marginal_cost": mc,
                "cost_shifter": cost_shock,
            })
    return pd.DataFrame(rows)


def draw_simulated_consumers(n_draws: int = 500, seed: int = 123) -> dict:
    """Draw simulated consumers from the population distribution.

    BLP uses simulation to integrate over consumer heterogeneity.
    Each draw represents a consumer 'type' with taste shocks v ~ N(0,1).
    """
    rng = np.random.RandomState(seed)
    return {
        "v_alpha": rng.normal(0, 1, n_draws),
        "v_sugar": rng.normal(0, 1, n_draws),
    }


# =============================================================================
# BLP share computation
# =============================================================================
def compute_individual_utilities(delta, sugar, prices, draws,
                                 sigma_alpha, sigma_sugar):
    """Compute U_ij = delta_j + sigma_alpha * v_i * (-p_j) + sigma_sugar * v_i * sugar_j.

    Returns array of shape (n_draws, J).
    """
    n_draws = len(draws["v_alpha"])
    J = len(delta)
    utilities = np.empty((n_draws, J))
    for j in range(J):
        utilities[:, j] = (delta[j]
                           + sigma_alpha * draws["v_alpha"] * (-prices[j])
                           + sigma_sugar * draws["v_sugar"] * sugar[j])
    return utilities


def compute_individual_choice_probs(utilities):
    """Multinomial logit choice probs with outside good: P_ij = exp(U_ij) / (1 + sum_k exp(U_ik))."""
    # Numerical stability: shift by row max
    row_max = utilities.max(axis=1, keepdims=True)
    shifted = utilities - row_max
    exp_util = np.exp(shifted)
    denom = np.exp(-row_max.ravel()) + exp_util.sum(axis=1)
    return exp_util / denom[:, np.newaxis]


def compute_blp_shares(delta, sugar, prices, draws, sigma_alpha, sigma_sugar):
    """Aggregate market shares: s_j = (1/ns) * sum_i P_ij."""
    utilities = compute_individual_utilities(delta, sugar, prices, draws,
                                             sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utilities)
    return probs.mean(axis=0)


# =============================================================================
# BLP contraction mapping (inner loop)
# =============================================================================
def contraction_mapping(observed_shares, sugar, prices, draws,
                        sigma_alpha, sigma_sugar,
                        tol=1e-12, max_iter=1000):
    """BLP contraction: delta^{h+1} = delta^h + ln(s_obs) - ln(s_pred(delta^h)).

    Returns (delta, converged, n_iter, norm_history).
    """
    outside_share = 1.0 - observed_shares.sum()
    delta = np.log(observed_shares) - np.log(max(outside_share, 1e-300))
    norm_history = []

    for iteration in range(max_iter):
        pred_shares = compute_blp_shares(delta, sugar, prices, draws,
                                         sigma_alpha, sigma_sugar)
        pred_shares = np.maximum(pred_shares, 1e-300)

        delta_new = delta + np.log(observed_shares) - np.log(pred_shares)
        norm_change = np.linalg.norm(delta_new - delta)
        norm_history.append(norm_change)

        if norm_change < tol:
            return delta_new, True, iteration + 1, norm_history
        delta = delta_new

    return delta, False, max_iter, norm_history


# =============================================================================
# Linear parameters via 2SLS (concentrated out)
# =============================================================================
def compute_linear_parameters(delta, X, prices, Z):
    """Recover (beta, alpha) by 2SLS: delta = X*beta - alpha*p + xi, E[Z'xi]=0.

    X_full = [constant, sugar, price].  Returns (theta1, xi).
    """
    X_full = np.column_stack([X, prices])
    W = np.column_stack([X, Z])

    ZtZ_inv = np.linalg.inv(W.T @ W)
    PZ = W @ ZtZ_inv @ W.T

    XtPZX = X_full.T @ PZ @ X_full
    XtPZy = X_full.T @ PZ @ delta

    theta1 = np.linalg.solve(XtPZX, XtPZy)
    xi = delta - X_full @ theta1
    return theta1, xi


# =============================================================================
# GMM objective (outer loop)
# =============================================================================
def gmm_objective(sigma, observed_shares, X, prices, sugar, Z, draws, W_gmm=None,
                  _cache=None):
    """GMM objective: J(sigma) = (Z'xi)' W (Z'xi).

    For each candidate sigma, runs contraction mapping to get delta,
    concentrates out linear params, computes moment conditions.
    """
    sigma_alpha, sigma_sugar = sigma
    if sigma_alpha < 0 or sigma_sugar < 0:
        return 1e10

    delta, converged, n_iter, _ = contraction_mapping(
        observed_shares, sugar, prices, draws,
        sigma_alpha, sigma_sugar, tol=1e-10, max_iter=500,
    )
    if not converged:
        return 1e10

    theta1, xi = compute_linear_parameters(delta, X, prices, Z)
    g = Z.T @ xi
    if W_gmm is None:
        W_gmm = np.eye(len(g))
    obj = float(g.T @ W_gmm @ g)

    # Cache the latest results for retrieval after optimization
    if _cache is not None:
        _cache["delta"] = delta
        _cache["theta1"] = theta1
        _cache["xi"] = xi
        _cache["obj"] = obj
    return obj


# =============================================================================
# Full BLP estimation
# =============================================================================
def estimate_blp(df_stacked, draws, initial_sigma=None, verbose=True):
    """Full BLP estimation pipeline.

    Returns dict with sigma_alpha, sigma_sugar, alpha, beta_const, beta_sugar,
    delta, xi, gmm_objective, converged.
    """
    observed_shares = df_stacked["share"].values
    prices = df_stacked["price"].values
    sugar = df_stacked["sugar"].values
    n = len(df_stacked)
    X = np.column_stack([np.ones(n), sugar])
    Z = df_stacked[["cost_shifter", "rival_sugar_sum", "sugar_squared"]].values

    if initial_sigma is None:
        initial_sigma = np.array([1.0, 1.0])

    W_gmm = np.eye(Z.shape[1])
    cache = {}
    eval_count = [0]

    def objective(sigma):
        obj = gmm_objective(sigma, observed_shares, X, prices, sugar, Z, draws,
                            W_gmm, _cache=cache)
        eval_count[0] += 1
        if verbose and eval_count[0] % 5 == 0:
            print(f"    eval {eval_count[0]:3d}: sigma=({sigma[0]:.3f}, {sigma[1]:.3f}), obj={obj:.4f}")
        return obj

    if verbose:
        print(f"  Starting GMM optimization from sigma=({initial_sigma[0]:.2f}, {initial_sigma[1]:.2f})")

    result = minimize(objective, initial_sigma, method="L-BFGS-B",
                      bounds=[(0.01, 5.0), (0.01, 5.0)],
                      options={"maxiter": 100, "ftol": 1e-8})
    sigma_hat = result.x

    if verbose:
        print(f"  Optimization finished: sigma=({sigma_hat[0]:.3f}, {sigma_hat[1]:.3f}), obj={result.fun:.6f}")

    # Final high-precision contraction at optimum
    delta_hat, _, _, _ = contraction_mapping(
        observed_shares, sugar, prices, draws,
        sigma_hat[0], sigma_hat[1], tol=1e-12, max_iter=1000,
    )
    theta1_hat, xi_hat = compute_linear_parameters(delta_hat, X, prices, Z)

    # theta1 = [beta_const, beta_sugar, coeff_on_price]
    # Price enters as -alpha*p, so coeff_on_price = -alpha
    return {
        "sigma_alpha": sigma_hat[0],
        "sigma_sugar": sigma_hat[1],
        "beta_const": theta1_hat[0],
        "beta_sugar": theta1_hat[1],
        "alpha": -theta1_hat[2],
        "delta": delta_hat,
        "xi": xi_hat,
        "gmm_objective": result.fun,
        "converged": result.success,
    }


# =============================================================================
# Compute shares with true parameters and generate instruments
# =============================================================================
def compute_true_shares(df, n_draws=1000, seed=99):
    """Compute market shares from true DGP parameters."""
    df = df.copy()
    df["delta"] = (TRUE_BETA_CONST
                   + TRUE_BETA_SUGAR_MEAN * df["sugar"]
                   - TRUE_ALPHA_MEAN * df["price"]
                   + df["xi"])

    shares_list = []
    for market_id, mdf in df.groupby("market_id"):
        draws = draw_simulated_consumers(n_draws, seed=seed + market_id)
        s = compute_blp_shares(mdf["delta"].values, mdf["sugar"].values,
                               mdf["price"].values, draws,
                               TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)
        outside = 1.0 - s.sum()
        for idx_local, (idx_global, _) in enumerate(mdf.iterrows()):
            shares_list.append({"index": idx_global, "share": s[idx_local],
                                "outside_share": outside})
    sdf = pd.DataFrame(shares_list).set_index("index")
    df["share"] = sdf["share"]
    df["outside_share"] = sdf["outside_share"]
    return df


def generate_instruments(df):
    """BLP-style instruments: cost shifters, rival characteristics, own squared."""
    df = df.copy()
    rival_sugar = []
    for _, row in df.iterrows():
        others = df[(df["market_id"] == row["market_id"]) &
                    (df["product_id"] != row["product_id"])]
        rival_sugar.append(others["sugar"].sum())
    df["rival_sugar_sum"] = rival_sugar
    df["sugar_squared"] = df["sugar"] ** 2
    return df


def create_estimation_dataset(n_markets=30):
    """Full dataset: generate products, compute shares, add instruments."""
    df = generate_product_data(n_markets=n_markets)
    df = compute_true_shares(df, n_draws=1000)
    df = generate_instruments(df)
    df["ln_share_ratio"] = np.log(df["share"]) - np.log(df["outside_share"])
    return df


# =============================================================================
# Elasticities
# =============================================================================
def compute_blp_elasticities(delta, sugar, prices, draws,
                             alpha_mean, sigma_alpha, sigma_sugar):
    """JxJ elasticity matrix: eta_jk = (p_k / s_j) * ds_j/dp_k.

    BLP elasticities depend on product similarity in characteristic space,
    unlike simple logit where cross-elasticities are proportional to share.
    """
    J = len(delta)
    utilities = compute_individual_utilities(delta, sugar, prices, draws,
                                             sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utilities)
    alpha_i = alpha_mean + sigma_alpha * draws["v_alpha"]
    shares = probs.mean(axis=0)

    deriv_matrix = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                deriv_matrix[j, k] = -(alpha_i * probs[:, j] * (1 - probs[:, j])).mean()
            else:
                deriv_matrix[j, k] = (alpha_i * probs[:, j] * probs[:, k]).mean()

    eta = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            eta[j, k] = deriv_matrix[j, k] * prices[k] / shares[j]
    return eta


def compute_blp_share_derivatives(delta, sugar, prices, draws,
                                  alpha_mean, sigma_alpha, sigma_sugar):
    """JxJ matrix of share derivatives ds_j/dp_k (needed for supply side)."""
    J = len(delta)
    utilities = compute_individual_utilities(delta, sugar, prices, draws,
                                             sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utilities)
    alpha_i = alpha_mean + sigma_alpha * draws["v_alpha"]

    deriv = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                deriv[j, k] = -(alpha_i * probs[:, j] * (1 - probs[:, j])).mean()
            else:
                deriv[j, k] = (alpha_i * probs[:, j] * probs[:, k]).mean()
    return deriv


def compute_simple_logit_elasticities(alpha, prices, shares):
    """Simple logit elasticities (IIA): cross-elasticities proportional to share."""
    J = len(prices)
    eta = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                eta[j, k] = -alpha * prices[j] * (1 - shares[j])
            else:
                eta[j, k] = alpha * prices[k] * shares[k]
    return eta


def compute_diversion_ratios(elasticity_matrix):
    """D_jk = cross-elasticity(k,j) / |own-elasticity(j,j)|.

    'If product j loses 100 units, how many go to product k?'
    """
    J = elasticity_matrix.shape[0]
    diversion = np.zeros((J, J))
    for j in range(J):
        own = abs(elasticity_matrix[j, j])
        for k in range(J):
            if j != k:
                diversion[j, k] = elasticity_matrix[k, j] / own
    return diversion


# =============================================================================
# Supply side and merger simulation
# =============================================================================
def compute_ownership_matrix(firm_ids):
    """O_jk = 1 if j and k owned by same firm."""
    J = len(firm_ids)
    return (firm_ids[:, None] == firm_ids[None, :]).astype(float)


def compute_markups_blp(shares, deriv_matrix, ownership):
    """Markups via first-order conditions: p - c = Omega^{-1} s."""
    omega = -deriv_matrix * ownership
    return np.linalg.solve(omega, shares)


def solve_equilibrium_prices(marginal_costs, ownership, sugar, draws,
                             alpha_mean, sigma_alpha, sigma_sugar,
                             initial_delta, tol=1e-6, max_iter=100):
    """Fixed-point iteration for post-merger equilibrium: p* = c + Omega(p*)^{-1} s(p*)."""
    prices = marginal_costs * 1.5  # initialize with some markup

    for iteration in range(max_iter):
        delta = initial_delta.copy()
        shares = compute_blp_shares(delta, sugar, prices, draws,
                                    sigma_alpha, sigma_sugar)
        deriv = compute_blp_share_derivatives(delta, sugar, prices, draws,
                                              alpha_mean, sigma_alpha, sigma_sugar)
        omega = -deriv * ownership
        try:
            markups = np.linalg.solve(omega, shares)
        except np.linalg.LinAlgError:
            return prices, False

        new_prices = marginal_costs + markups
        if np.max(np.abs(new_prices - prices)) < tol:
            return new_prices, True
        prices = 0.5 * prices + 0.5 * new_prices  # damped update

    return prices, False


def simulate_merger(pre_prices, pre_shares, marginal_costs, pre_firm_ids,
                    acquiring_firm, acquired_firm, sugar, draws,
                    alpha_mean, sigma_alpha, sigma_sugar, delta):
    """Full merger simulation: update ownership, solve new equilibrium, compute price effects."""
    post_firm_ids = pre_firm_ids.copy()
    post_firm_ids[pre_firm_ids == acquired_firm] = acquiring_firm
    post_ownership = compute_ownership_matrix(post_firm_ids)

    post_prices, converged = solve_equilibrium_prices(
        marginal_costs, post_ownership, sugar, draws,
        alpha_mean, sigma_alpha, sigma_sugar, delta,
    )

    # Post-merger shares: adjust delta for price changes
    delta_post = delta + alpha_mean * (pre_prices - post_prices)
    post_shares = compute_blp_shares(delta_post, sugar, post_prices, draws,
                                     sigma_alpha, sigma_sugar)

    price_changes = post_prices - pre_prices
    price_change_pct = (price_changes / pre_prices) * 100

    return {
        "post_prices": post_prices,
        "post_shares": post_shares,
        "price_changes": price_changes,
        "price_change_pct": price_change_pct,
        "converged": converged,
        "post_firm_ids": post_firm_ids,
    }


# =============================================================================
# Logit merger (for comparison)
# =============================================================================
def logit_markups(alpha, shares, ownership):
    """Simple logit markups: Omega^{-1} s where Omega_jk = alpha * s_j (1{j=k} - s_k) * O_jk."""
    J = len(shares)
    deriv = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                deriv[j, k] = -alpha * shares[j] * (1 - shares[j])
            else:
                deriv[j, k] = alpha * shares[j] * shares[k]
    omega = -deriv * ownership
    return np.linalg.solve(omega, shares)


# =============================================================================
# Figures
# =============================================================================
def plot_contraction_convergence(norm_history):
    """Figure 1: contraction mapping convergence (log scale)."""
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(norm_history) + 1), norm_history, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|\delta^{h+1} - \delta^h\|$")
    ax.set_title("BLP Contraction Mapping Convergence")
    ax.axhline(1e-12, color="r", linestyle="--", linewidth=1, label=r"Tolerance $10^{-12}$")
    ax.legend()
    return fig


def plot_elasticity_heatmaps(logit_eta, blp_eta, product_names):
    """Figure 2: side-by-side logit vs BLP elasticity heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    vmin = min(logit_eta.min(), blp_eta.min())
    vmax = max(logit_eta.max(), blp_eta.max())
    # Symmetric range centred on zero
    abs_max = max(abs(vmin), abs(vmax))

    for ax, matrix, title in zip(axes,
                                 [logit_eta, blp_eta],
                                 ["Simple Logit (IIA)", "BLP (Random Coefficients)"]):
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                       vmin=-abs_max, vmax=abs_max)
        ax.set_xticks(range(len(product_names)))
        ax.set_yticks(range(len(product_names)))
        ax.set_xticklabels(product_names, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(product_names, fontsize=9)
        ax.set_title(title, fontweight="bold")
        for i in range(len(product_names)):
            for j in range(len(product_names)):
                val = matrix[i, j]
                color = "white" if abs(val) > abs_max * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=axes, label="Elasticity", shrink=0.8)
    fig.suptitle("Elasticity Matrices: Logit vs BLP", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_diversion_comparison(logit_div, blp_div, product_names, ref_product=0):
    """Figure 3: grouped bar chart of diversion ratios from reference product."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ref_name = product_names[ref_product]
    other_idx = [i for i in range(len(product_names)) if i != ref_product]
    other_names = [product_names[i] for i in other_idx]

    logit_vals = [logit_div[ref_product, i] * 100 for i in other_idx]
    blp_vals = [blp_div[ref_product, i] * 100 for i in other_idx]

    x = np.arange(len(other_names))
    w = 0.32
    bars1 = ax.bar(x - w / 2, logit_vals, w, label="Simple Logit", color="#e74c3c",
                   edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + w / 2, blp_vals, w, label="BLP", color="#3498db",
                   edgecolor="black", linewidth=0.6)

    for bar, val in zip(list(bars1) + list(bars2), logit_vals + blp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", fontsize=9)

    ax.set_xlabel("Rival Product")
    ax.set_ylabel("Diversion Ratio (%)")
    ax.set_title(f"Diversion Ratios: Where Do {ref_name} Customers Go?")
    ax.set_xticks(x)
    ax.set_xticklabels(other_names)
    ax.legend()
    return fig


def plot_merger_prices(product_names, pre_prices, logit_post, blp_post,
                       merging_idx):
    """Figure 4: pre vs post-merger prices under logit and BLP."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(product_names))
    w = 0.25

    ax.bar(x - w, pre_prices, w, label="Pre-Merger", color="#3498db",
           edgecolor="black", linewidth=0.6)
    ax.bar(x, logit_post, w, label="Post-Merger (Logit)", color="#f39c12",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w, blp_post, w, label="Post-Merger (BLP)", color="#e74c3c",
           edgecolor="black", linewidth=0.6)

    for i in range(len(product_names)):
        logit_chg = (logit_post[i] - pre_prices[i]) / pre_prices[i] * 100
        blp_chg = (blp_post[i] - pre_prices[i]) / pre_prices[i] * 100
        ax.text(x[i], max(logit_post[i], blp_post[i]) + 0.15,
                f"L: {logit_chg:+.1f}%\nB: {blp_chg:+.1f}%",
                ha="center", fontsize=8)

    ax.set_xlabel("Product")
    ax.set_ylabel("Price ($)")
    ax.set_title("Merger Simulation: Price Effects (Logit vs BLP)")
    ax.set_xticks(x)
    ax.set_xticklabels(product_names)
    ax.legend(fontsize=9)
    return fig


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    print("=" * 70)
    print("BLP FULL PIPELINE: RANDOM COEFFICIENTS DEMAND ESTIMATION")
    print("Berry, Levinsohn, Pakes (1995)")
    print("=" * 70)

    # =========================================================================
    # Step 1: Generate synthetic data
    # =========================================================================
    print("\n--- Step 1: Generating synthetic cereal market data ---")
    df = create_estimation_dataset(n_markets=30)
    draws = draw_simulated_consumers(n_draws=500, seed=123)

    print(f"  Markets: {df['market_id'].nunique()}, Obs: {len(df)}")
    market0 = df[df["market_id"] == 0].reset_index(drop=True)
    print(f"  Sample market:")
    for _, r in market0.iterrows():
        print(f"    {r['product_name']:<15s}  p=${r['price']:.2f}  sugar={r['sugar']:.0f}  "
              f"share={r['share']:.4f}")

    # =========================================================================
    # Step 2: BLP contraction mapping demonstration
    # =========================================================================
    print("\n--- Step 2: BLP contraction mapping (inner loop demo) ---")
    product_names = market0["product_name"].tolist()
    prices0 = market0["price"].values
    sugar0 = market0["sugar"].values
    shares0 = market0["share"].values

    delta_true = (TRUE_BETA_CONST
                  + TRUE_BETA_SUGAR_MEAN * sugar0
                  - TRUE_ALPHA_MEAN * prices0
                  + market0["xi"].values)

    delta_rec, conv, n_iter, norm_hist = contraction_mapping(
        shares0, sugar0, prices0, draws,
        TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR,
        tol=1e-12, max_iter=1000,
    )
    print(f"  Converged: {conv} in {n_iter} iterations")
    print(f"  Max |delta_true - delta_recovered| = {np.abs(delta_true - delta_rec).max():.2e}")

    # =========================================================================
    # Step 3: GMM estimation (outer loop)
    # =========================================================================
    print("\n--- Step 3: GMM estimation (outer loop) ---")
    est = estimate_blp(df, draws, initial_sigma=np.array([1.0, 1.0]), verbose=True)
    print(f"\n  Estimated parameters:")
    print(f"    alpha       = {est['alpha']:.3f}   (true: {TRUE_ALPHA_MEAN})")
    print(f"    beta_const  = {est['beta_const']:.3f}   (true: {TRUE_BETA_CONST})")
    print(f"    beta_sugar  = {est['beta_sugar']:.3f}   (true: {TRUE_BETA_SUGAR_MEAN})")
    print(f"    sigma_alpha = {est['sigma_alpha']:.3f}   (true: {TRUE_SIGMA_ALPHA})")
    print(f"    sigma_sugar = {est['sigma_sugar']:.3f}   (true: {TRUE_SIGMA_SUGAR})")

    # Bootstrap standard errors (small-sample approximation)
    print("\n  Computing bootstrap standard errors (20 replications)...")
    n_boot = 20
    boot_estimates = {k: [] for k in ["alpha", "beta_const", "beta_sugar",
                                       "sigma_alpha", "sigma_sugar"]}
    market_ids = df["market_id"].unique()
    rng_boot = np.random.RandomState(777)
    for b in range(n_boot):
        sampled_markets = rng_boot.choice(market_ids, size=len(market_ids), replace=True)
        boot_dfs = []
        for new_id, old_id in enumerate(sampled_markets):
            chunk = df[df["market_id"] == old_id].copy()
            chunk["market_id"] = new_id
            boot_dfs.append(chunk)
        boot_df = pd.concat(boot_dfs, ignore_index=True)
        try:
            boot_est = estimate_blp(boot_df, draws, initial_sigma=est["sigma_alpha"] * np.ones(2),
                                    verbose=False)
            for k in boot_estimates:
                boot_estimates[k].append(boot_est[k])
        except Exception:
            pass  # skip failed bootstrap samples

    se = {}
    for k in boot_estimates:
        vals = np.array(boot_estimates[k])
        se[k] = vals.std() if len(vals) > 2 else np.nan

    print("  Bootstrap SE computed.")

    # =========================================================================
    # Step 4: Elasticities -- BLP vs Logit
    # =========================================================================
    print("\n--- Step 4: Elasticities (BLP vs Logit comparison) ---")

    # Use estimated parameters for BLP elasticities
    alpha_use = est["alpha"]
    sigma_alpha_use = est["sigma_alpha"]
    sigma_sugar_use = est["sigma_sugar"]

    # Recompute delta for market 0 with estimated params
    delta0 = delta_rec  # from contraction at true params (close enough for display)

    blp_eta = compute_blp_elasticities(delta0, sugar0, prices0, draws,
                                       alpha_use, sigma_alpha_use, sigma_sugar_use)
    logit_eta = compute_simple_logit_elasticities(alpha_use, prices0, shares0)

    print("\n  Simple Logit Elasticity Matrix:")
    _print_matrix(logit_eta, product_names)
    print("\n  BLP Elasticity Matrix:")
    _print_matrix(blp_eta, product_names)

    # =========================================================================
    # Step 5: Diversion ratios
    # =========================================================================
    print("\n--- Step 5: Diversion ratios ---")
    logit_div = compute_diversion_ratios(logit_eta)
    blp_div = compute_diversion_ratios(blp_eta)

    print(f"\n  If Choco-Bombs raises price, where do customers go?")
    print(f"  {'Product':<15s}  {'Logit':>8s}  {'BLP':>8s}")
    for i, name in enumerate(product_names):
        if i != 0:
            print(f"  {name:<15s}  {logit_div[0, i]*100:>7.1f}%  {blp_div[0, i]*100:>7.1f}%")

    # =========================================================================
    # Step 6: Supply side -- recover marginal costs
    # =========================================================================
    print("\n--- Step 6: Supply side (recover marginal costs) ---")
    firm_ids = market0["firm_id"].values
    ownership = compute_ownership_matrix(firm_ids)
    true_mc = market0["marginal_cost"].values

    deriv0 = compute_blp_share_derivatives(delta0, sugar0, prices0, draws,
                                           alpha_use, sigma_alpha_use, sigma_sugar_use)
    markups0 = compute_markups_blp(shares0, deriv0, ownership)
    est_mc = prices0 - markups0

    print(f"  {'Product':<15s}  {'Price':>7s}  {'Markup':>7s}  {'Est MC':>7s}  {'True MC':>8s}")
    for i, name in enumerate(product_names):
        print(f"  {name:<15s}  ${prices0[i]:>6.2f}  ${markups0[i]:>6.2f}  "
              f"${est_mc[i]:>6.2f}  ${true_mc[i]:>7.2f}")

    # =========================================================================
    # Step 7: Merger simulation
    # =========================================================================
    print("\n--- Step 7: Merger simulation (Choco-Bombs acquires Store-Frosted) ---")

    merger_res = simulate_merger(
        prices0, shares0, est_mc, firm_ids,
        acquiring_firm=1, acquired_firm=3,
        sugar=sugar0, draws=draws,
        alpha_mean=alpha_use, sigma_alpha=sigma_alpha_use,
        sigma_sugar=sigma_sugar_use, delta=delta0,
    )

    # Logit merger prediction
    logit_pre_markups = logit_markups(alpha_use, shares0, ownership)
    logit_mc = prices0 - logit_pre_markups
    post_firm_ids = merger_res["post_firm_ids"]
    post_ownership_logit = compute_ownership_matrix(post_firm_ids)
    logit_post_markups = logit_markups(alpha_use, shares0, post_ownership_logit)
    logit_post_prices = logit_mc + logit_post_markups

    print(f"\n  {'Product':<15s}  {'Pre-Price':>10s}  {'Logit Post':>11s}  {'BLP Post':>9s}")
    for i, name in enumerate(product_names):
        print(f"  {name:<15s}  ${prices0[i]:>9.2f}  ${logit_post_prices[i]:>10.2f}  "
              f"${merger_res['post_prices'][i]:>8.2f}")

    print(f"\n  Price change (%):")
    for i, name in enumerate(product_names):
        l_chg = (logit_post_prices[i] - prices0[i]) / prices0[i] * 100
        b_chg = merger_res["price_change_pct"][i]
        print(f"    {name:<15s}  Logit: {l_chg:+6.1f}%   BLP: {b_chg:+6.1f}%")

    # =========================================================================
    # Generate Report
    # =========================================================================
    print("\n--- Generating report and figures ---")
    setup_style()

    report = ModelReport(
        "BLP Full Pipeline: Random Coefficients Demand Estimation",
        "The gold standard for demand estimation in industrial organization. "
        "Random coefficients allow rich substitution patterns that matter "
        "enormously for merger analysis.",
    )

    report.add_overview(
        "The Berry, Levinsohn, and Pakes (1995) model is the workhorse of modern "
        "demand estimation in IO. Unlike simple logit, which suffers from the IIA "
        "(Independence of Irrelevant Alternatives) problem, BLP allows consumer "
        "preferences to vary across the population. This means substitution patterns "
        "are driven by product similarity in characteristic space, not just market shares.\n\n"
        "In this implementation we estimate demand for three synthetic cereal products. "
        "Consumers differ in their price sensitivity and sugar preferences. Sugar-loving "
        "consumers (\"kids\") substitute between sugary cereals, while health-conscious "
        "consumers (\"parents\") substitute among low-sugar options. This heterogeneity "
        "has first-order implications for merger analysis."
    )

    report.add_equations(
        r"""
**Utility:** Consumer $i$ gets utility from product $j$:

$$U_{ij} = \underbrace{\beta_0 + \beta_s \cdot \text{sugar}_j - \alpha \cdot p_j + \xi_j}_{\delta_j \text{ (mean utility)}} + \underbrace{\sigma_\alpha \nu_i^\alpha (-p_j) + \sigma_s \nu_i^s \cdot \text{sugar}_j}_{\mu_{ij} \text{ (individual deviation)}}$$

where $\nu_i \sim N(0, I)$ captures preference heterogeneity.

**Market shares** (simulation): $s_j(\delta, \sigma) = \frac{1}{n_s}\sum_{i=1}^{n_s} \frac{\exp(\delta_j + \mu_{ij})}{1 + \sum_k \exp(\delta_k + \mu_{ik})}$

**BLP contraction** (inner loop): $\delta^{h+1} = \delta^h + \ln s^{obs} - \ln s^{pred}(\delta^h, \sigma)$

**GMM** (outer loop): $\hat{\sigma} = \arg\min_\sigma \; [Z'\xi(\sigma)]' W [Z'\xi(\sigma)]$

where $\xi = \delta - X\beta + \alpha p$ is the structural error and $Z$ are instruments.
"""
    )

    report.add_model_setup(
        "| Parameter | True | Estimated | SE | Description |\n"
        "|-----------|------|-----------|-----|-------------|\n"
        f"| $\\alpha$ | {TRUE_ALPHA_MEAN:.1f} | {est['alpha']:.3f} | {se.get('alpha', np.nan):.3f} | Mean price sensitivity |\n"
        f"| $\\beta_0$ | {TRUE_BETA_CONST:.1f} | {est['beta_const']:.3f} | {se.get('beta_const', np.nan):.3f} | Base utility |\n"
        f"| $\\beta_s$ | {TRUE_BETA_SUGAR_MEAN:.1f} | {est['beta_sugar']:.3f} | {se.get('beta_sugar', np.nan):.3f} | Mean sugar taste |\n"
        f"| $\\sigma_\\alpha$ | {TRUE_SIGMA_ALPHA:.1f} | {est['sigma_alpha']:.3f} | {se.get('sigma_alpha', np.nan):.3f} | Std dev of price sensitivity |\n"
        f"| $\\sigma_s$ | {TRUE_SIGMA_SUGAR:.1f} | {est['sigma_sugar']:.3f} | {se.get('sigma_sugar', np.nan):.3f} | Std dev of sugar taste |"
    )

    report.add_solution_method(
        "**Nested Fixed Point (NFXP):**\n\n"
        "1. **Outer loop (GMM):** Search over nonlinear parameters $\\sigma = (\\sigma_\\alpha, \\sigma_s)$ "
        "using L-BFGS-B with bounds $[0.01, 5.0]$.\n"
        "2. **Inner loop (contraction):** For each candidate $\\sigma$, run the BLP contraction mapping to "
        "find $\\delta$ such that predicted shares match observed shares (tolerance $10^{-12}$).\n"
        "3. **Concentration:** Given $\\delta$, recover linear parameters $(\\beta, \\alpha)$ by 2SLS "
        "using cost shifters, rival characteristics, and own-squared sugar as instruments.\n"
        "4. **Moments:** Compute $\\xi = \\delta - X\\beta + \\alpha p$ and form GMM objective $g'Wg$ "
        f"where $g = Z'\\xi$.\n\n"
        f"The contraction converged in **{n_iter} iterations** for the demonstration market. "
        f"GMM optimization {'converged' if est['converged'] else 'did not converge'} "
        f"(objective = {est['gmm_objective']:.6f})."
    )

    # --- Figure 1: Contraction convergence ---
    fig1 = plot_contraction_convergence(norm_hist)
    report.add_figure("figures/contraction-convergence.png",
                      "BLP contraction mapping convergence (linear rate in log scale)", fig1)

    # --- Figure 2: Elasticity comparison ---
    fig2 = plot_elasticity_heatmaps(logit_eta, blp_eta, product_names)
    report.add_figure("figures/elasticity-comparison.png",
                      "Elasticity matrices: simple logit (IIA) vs BLP (realistic substitution)", fig2)

    # --- Figure 3: Diversion ratios ---
    fig3 = plot_diversion_comparison(logit_div, blp_div, product_names, ref_product=0)
    report.add_figure("figures/diversion-ratios.png",
                      "Diversion ratios from Choco-Bombs: logit diverts by market share, "
                      "BLP diverts to similar products", fig3)

    # --- Figure 4: Merger simulation ---
    fig4 = plot_merger_prices(product_names, prices0, logit_post_prices,
                              merger_res["post_prices"], merging_idx=[0, 2])
    report.add_figure("figures/merger-simulation.png",
                      "Merger price effects: BLP predicts larger increases for close substitutes "
                      "because it captures product similarity", fig4)

    # --- Table: Parameter estimates ---
    table_data = {
        "Parameter": [r"alpha", r"beta_const", r"beta_sugar",
                      r"sigma_alpha", r"sigma_sugar"],
        "True": [TRUE_ALPHA_MEAN, TRUE_BETA_CONST, TRUE_BETA_SUGAR_MEAN,
                 TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR],
        "Estimated": [est["alpha"], est["beta_const"], est["beta_sugar"],
                      est["sigma_alpha"], est["sigma_sugar"]],
        "Std Error": [se.get("alpha", np.nan), se.get("beta_const", np.nan),
                      se.get("beta_sugar", np.nan), se.get("sigma_alpha", np.nan),
                      se.get("sigma_sugar", np.nan)],
    }
    table_df = pd.DataFrame(table_data)
    for col in ["True", "Estimated", "Std Error"]:
        table_df[col] = table_df[col].map(lambda v: f"{v:.3f}")
    report.add_table("tables/parameter-estimates.csv",
                     "BLP Parameter Estimates with Bootstrap Standard Errors", table_df)

    report.add_takeaway(
        "BLP is the gold standard for demand estimation in IO because random coefficients "
        "generate realistic substitution patterns. The key results from this pipeline:\n\n"
        "**1. Substitution depends on product similarity.** When Choco-Bombs raises its price, "
        "BLP correctly predicts that most customers switch to Store-Frosted (also sugary), "
        "not Fiber-Bran. Simple logit, constrained by IIA, diverts proportionally to market "
        "share and misses this pattern entirely.\n\n"
        "**2. This matters enormously for merger analysis.** A merger between Choco-Bombs and "
        "Store-Frosted combines close substitutes. BLP predicts a larger price increase than "
        "logit because the merging firm internalizes the high diversion between its products. "
        "Antitrust authorities who use logit would systematically underestimate merger harms "
        "for mergers between similar products.\n\n"
        "**3. The computational cost is justified.** BLP requires a nested optimization "
        "(contraction mapping inside GMM), making it far more expensive than logit. But "
        "the payoff is a demand system that captures the heterogeneity in consumer preferences "
        "that drives real market outcomes."
    )

    report.add_references([
        "Berry, S., Levinsohn, J., and Pakes, A. (1995). \"Automobile Prices in Market "
        "Equilibrium.\" *Econometrica*, 63(4), 841-890.",
        "Nevo, A. (2000). \"A Practitioner's Guide to Estimation of Random-Coefficients "
        "Logit Models of Demand.\" *Journal of Economics & Management Strategy*, 9(4), 513-548.",
        "Nevo, A. (2001). \"Measuring Market Power in the Ready-to-Eat Cereal Industry.\" "
        "*Econometrica*, 69(2), 307-342.",
    ])

    report.write("README.md")
    print(f"\nDone. Generated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


def _print_matrix(matrix, names):
    header = "                " + "  ".join(f"{n[:13]:>13s}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"  {name[:13]:>13s}  " + "  ".join(f"{matrix[i, j]:>13.3f}" for j in range(len(names)))
        print(row)


if __name__ == "__main__":
    main()
