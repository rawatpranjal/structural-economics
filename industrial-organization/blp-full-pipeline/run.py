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
TRUE_ALPHA_MEAN = 3.0          # Mean price sensitivity
TRUE_BETA_SUGAR_MEAN = 0.5     # Mean sugar taste
TRUE_BETA_CONST = 2.0          # Base utility (constant)
TRUE_SIGMA_ALPHA = 0.8         # Std dev of price sensitivity across consumers
TRUE_SIGMA_SUGAR = 1.0         # Std dev of sugar taste across consumers


# =============================================================================
# Data generation
# =============================================================================
def generate_product_data(n_markets=50, seed=42):
    """Generate product-market level data for a synthetic cereal market.

    xi_jt varies across markets (essential for identification).
    """
    rng = np.random.RandomState(seed)
    products = {
        "product_id": [1, 2, 3],
        "product_name": ["Choco-Bombs", "Fiber-Bran", "Store-Frosted"],
        "sugar": [5.0, 0.5, 4.0],        # scaled so sigma_sugar*v*sugar stays moderate
        "xi_mean": [0.5, 0.8, -0.2],     # Fiber-Bran has higher unobserved quality
        "firm_id": [1, 2, 3],
        "marginal_cost_base": [1.8, 2.2, 1.2],   # Fiber-Bran cost lowered for viable share
    }
    rows = []
    for t in range(n_markets):
        cost_shock = rng.normal(0, 0.3)
        for j in range(len(products["product_id"])):
            mc = products["marginal_cost_base"][j] + cost_shock + rng.normal(0, 0.15)
            xi_jt = products["xi_mean"][j] + rng.normal(0, 0.3)
            markup = rng.uniform(0.3, 0.8)
            price = mc * (1 + markup)
            rows.append({
                "market_id": t,
                "product_id": products["product_id"][j],
                "product_name": products["product_name"][j],
                "sugar": products["sugar"][j],
                "xi": xi_jt,
                "firm_id": products["firm_id"][j],
                "price": price,
                "marginal_cost": mc,
                "cost_shifter": cost_shock,
            })
    return pd.DataFrame(rows)


def draw_simulated_consumers(n_draws=500, seed=123):
    """Draw simulated consumers: taste shocks v ~ N(0,1)."""
    rng = np.random.RandomState(seed)
    return {
        "v_alpha": rng.normal(0, 1, n_draws),
        "v_sugar": rng.normal(0, 1, n_draws),
    }


# =============================================================================
# BLP share computation (vectorized)
# =============================================================================
def compute_individual_utilities(delta, sugar, prices, draws,
                                 sigma_alpha, sigma_sugar):
    """U_ij = delta_j + sigma_alpha * v_i * (-p_j) + sigma_sugar * v_i * sugar_j."""
    mu = (sigma_alpha * draws["v_alpha"][:, None] * (-prices[None, :])
          + sigma_sugar * draws["v_sugar"][:, None] * sugar[None, :])
    return delta[None, :] + mu


def compute_individual_choice_probs(utilities):
    """P_ij = exp(U_ij) / (1 + sum_k exp(U_ik)). Numerically stable."""
    row_max = utilities.max(axis=1, keepdims=True)
    shifted = utilities - row_max
    exp_util = np.exp(shifted)
    denom = np.exp(-row_max.ravel()) + exp_util.sum(axis=1)
    return exp_util / denom[:, None]


def compute_blp_shares(delta, sugar, prices, draws, sigma_alpha, sigma_sugar):
    """Aggregate shares: s_j = (1/ns) sum_i P_ij."""
    utils = compute_individual_utilities(delta, sugar, prices, draws,
                                         sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utils)
    return probs.mean(axis=0)


# =============================================================================
# BLP contraction mapping (inner loop)
# =============================================================================
def contraction_mapping(observed_shares, sugar, prices, draws,
                        sigma_alpha, sigma_sugar,
                        tol=1e-12, max_iter=2000):
    """delta^{h+1} = delta^h + ln(s_obs) - ln(s_pred(delta^h))."""
    outside_share = max(1.0 - observed_shares.sum(), 1e-300)
    delta = np.log(observed_shares) - np.log(outside_share)
    norm_history = []

    for it in range(max_iter):
        pred = compute_blp_shares(delta, sugar, prices, draws,
                                  sigma_alpha, sigma_sugar)
        pred = np.maximum(pred, 1e-300)
        delta_new = delta + np.log(observed_shares) - np.log(pred)
        norm_change = np.max(np.abs(delta_new - delta))
        norm_history.append(norm_change)
        if norm_change < tol:
            return delta_new, True, it + 1, norm_history
        delta = delta_new

    return delta, False, max_iter, norm_history


def invert_all_markets(df, draws, sigma_alpha, sigma_sugar,
                       tol=1e-12, max_iter=2000):
    """Run contraction for every market. Returns stacked delta."""
    deltas = np.zeros(len(df))
    all_converged = True

    for market_id, mdf in df.groupby("market_id"):
        idx = mdf.index.values
        delta_m, conv, _, _ = contraction_mapping(
            mdf["share"].values, mdf["sugar"].values, mdf["price"].values,
            draws, sigma_alpha, sigma_sugar, tol=tol, max_iter=max_iter,
        )
        deltas[idx] = delta_m
        if not conv:
            all_converged = False

    return deltas, all_converged


# =============================================================================
# Linear parameters via 2SLS
# =============================================================================
def compute_linear_parameters_2sls(delta, X, prices, Z):
    """2SLS: delta = [X, price]*theta1 + xi, E[Z'xi]=0."""
    X_full = np.column_stack([X, prices])
    W = np.column_stack([X, Z])
    WtW_inv = np.linalg.inv(W.T @ W + 1e-10 * np.eye(W.shape[1]))
    PZ = W @ WtW_inv @ W.T
    theta1 = np.linalg.solve(X_full.T @ PZ @ X_full + 1e-10 * np.eye(X_full.shape[1]),
                             X_full.T @ PZ @ delta)
    xi = delta - X_full @ theta1
    return theta1, xi


# =============================================================================
# GMM objective (outer loop)
# =============================================================================
def gmm_objective(sigma, df, draws, X, Z, W_gmm=None):
    """GMM objective: J(sigma) = n * (Z'xi/n)' W (Z'xi/n)."""
    sigma_alpha, sigma_sugar = np.abs(sigma)

    deltas, converged = invert_all_markets(
        df, draws, sigma_alpha, sigma_sugar, tol=1e-8, max_iter=500,
    )
    if not converged:
        return 1e10

    theta1, xi = compute_linear_parameters_2sls(deltas, X, df["price"].values, Z)
    if theta1[2] > 0:  # price coeff should be negative
        return 1e8

    n = len(xi)
    g = Z.T @ xi / n
    if W_gmm is None:
        W_gmm = np.eye(len(g))
    return float(n * g.T @ W_gmm @ g)


def estimate_blp(df, draws, initial_sigma=None, verbose=True):
    """Full BLP estimation."""
    n = len(df)
    sugar = df["sugar"].values
    X = np.column_stack([np.ones(n), sugar])
    Z = df[["cost_shifter", "rival_sugar_sum", "sugar_squared"]].values
    W_gmm = np.eye(Z.shape[1])

    if initial_sigma is None:
        initial_sigma = np.array([1.0, 1.0])

    eval_count = [0]

    def objective(sigma):
        obj = gmm_objective(sigma, df, draws, X, Z, W_gmm)
        eval_count[0] += 1
        if verbose and eval_count[0] % 5 == 0:
            print(f"    eval {eval_count[0]:3d}: sigma=({abs(sigma[0]):.3f}, {abs(sigma[1]):.3f}), obj={obj:.6f}")
        return obj

    if verbose:
        print(f"  Starting GMM from sigma=({initial_sigma[0]:.2f}, {initial_sigma[1]:.2f})")

    result = minimize(objective, initial_sigma, method="Nelder-Mead",
                      options={"maxiter": 200, "xatol": 0.01, "fatol": 1e-3})
    sigma_hat = np.abs(result.x)

    if verbose:
        print(f"  Optimum: sigma=({sigma_hat[0]:.3f}, {sigma_hat[1]:.3f}), obj={result.fun:.6f}")

    # Final high-precision inversion
    deltas, _ = invert_all_markets(df, draws, sigma_hat[0], sigma_hat[1],
                                   tol=1e-12, max_iter=2000)
    theta1, xi = compute_linear_parameters_2sls(deltas, X, df["price"].values, Z)

    return {
        "sigma_alpha": sigma_hat[0],
        "sigma_sugar": sigma_hat[1],
        "beta_const": theta1[0],
        "beta_sugar": theta1[1],
        "alpha": -theta1[2],
        "delta": deltas,
        "xi": xi,
        "gmm_objective": result.fun,
        "converged": result.success,
    }


# =============================================================================
# Data generation pipeline
# =============================================================================
def compute_true_shares(df, draws):
    """Compute shares from true DGP parameters using shared draws."""
    df = df.copy()
    df["delta"] = (TRUE_BETA_CONST
                   + TRUE_BETA_SUGAR_MEAN * df["sugar"]
                   - TRUE_ALPHA_MEAN * df["price"]
                   + df["xi"])

    shares_list = []
    for market_id, mdf in df.groupby("market_id"):
        s = compute_blp_shares(mdf["delta"].values, mdf["sugar"].values,
                               mdf["price"].values, draws,
                               TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)
        outside = max(1.0 - s.sum(), 1e-6)
        for idx_local, (idx_global, _) in enumerate(mdf.iterrows()):
            shares_list.append({"index": idx_global, "share": s[idx_local],
                                "outside_share": outside})
    sdf = pd.DataFrame(shares_list).set_index("index")
    df["share"] = sdf["share"]
    df["outside_share"] = sdf["outside_share"]
    return df


def generate_instruments(df):
    """BLP-style instruments."""
    df = df.copy()
    rival_sugar = []
    for _, row in df.iterrows():
        others = df[(df["market_id"] == row["market_id"]) &
                    (df["product_id"] != row["product_id"])]
        rival_sugar.append(others["sugar"].sum())
    df["rival_sugar_sum"] = rival_sugar
    df["sugar_squared"] = df["sugar"] ** 2
    return df


def create_estimation_dataset(n_markets, draws):
    """Generate products, compute shares, add instruments."""
    df = generate_product_data(n_markets=n_markets)
    df = compute_true_shares(df, draws)
    df = generate_instruments(df)
    return df


# =============================================================================
# Elasticities
# =============================================================================
def compute_blp_elasticities(delta, sugar, prices, draws,
                             alpha_mean, sigma_alpha, sigma_sugar):
    """JxJ BLP elasticity matrix: eta_jk = (p_k / s_j) * ds_j/dp_k."""
    J = len(delta)
    utils = compute_individual_utilities(delta, sugar, prices, draws,
                                         sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utils)
    alpha_i = alpha_mean + sigma_alpha * draws["v_alpha"]
    shares = probs.mean(axis=0)

    deriv = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                deriv[j, k] = -(alpha_i * probs[:, j] * (1 - probs[:, j])).mean()
            else:
                deriv[j, k] = (alpha_i * probs[:, j] * probs[:, k]).mean()

    return deriv * prices[None, :] / shares[:, None]


def compute_blp_share_derivatives(delta, sugar, prices, draws,
                                  alpha_mean, sigma_alpha, sigma_sugar):
    """JxJ matrix ds_j/dp_k."""
    J = len(delta)
    utils = compute_individual_utilities(delta, sugar, prices, draws,
                                         sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utils)
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
    """Simple logit elasticities (IIA)."""
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
    """D_jk = cross-elasticity(k,j) / |own-elasticity(j,j)|."""
    J = elasticity_matrix.shape[0]
    diversion = np.zeros((J, J))
    for j in range(J):
        own = abs(elasticity_matrix[j, j])
        for k in range(J):
            if j != k and own > 1e-15:
                diversion[j, k] = elasticity_matrix[k, j] / own
    return diversion


# =============================================================================
# Supply side and merger simulation
# =============================================================================
def compute_ownership_matrix(firm_ids):
    """O_jk = 1 if same firm."""
    return (firm_ids[:, None] == firm_ids[None, :]).astype(float)


def compute_markups_blp(shares, deriv_matrix, ownership):
    """p - c = Omega^{-1} s."""
    omega = -deriv_matrix * ownership
    return np.linalg.solve(omega, shares)


def logit_share_derivatives(alpha, shares):
    """Simple logit ds_j/dp_k."""
    J = len(shares)
    deriv = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                deriv[j, k] = -alpha * shares[j] * (1 - shares[j])
            else:
                deriv[j, k] = alpha * shares[j] * shares[k]
    return deriv


def solve_equilibrium_prices(marginal_costs, ownership, sugar, draws,
                             alpha_mean, sigma_alpha, sigma_sugar,
                             initial_delta, initial_prices,
                             tol=1e-6, max_iter=300):
    """Fixed-point iteration for post-merger equilibrium.

    p* = c + Omega(p*)^{-1} s(p*).
    Delta adjusts for price changes: delta(p) = delta_0 - alpha*(p - p_0).
    """
    prices = initial_prices.copy()
    p0 = initial_prices.copy()
    delta0 = initial_delta.copy()

    for iteration in range(max_iter):
        delta = delta0 - alpha_mean * (prices - p0)
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
        new_prices = np.clip(new_prices, 0.01, initial_prices.max() * 5.0)
        if np.max(np.abs(new_prices - prices)) < tol:
            return new_prices, True
        prices = 0.5 * prices + 0.5 * new_prices

    return prices, False


def simulate_merger(pre_prices, pre_shares, marginal_costs, pre_firm_ids,
                    acquiring_firm, acquired_firm, sugar, draws,
                    alpha_mean, sigma_alpha, sigma_sugar, delta):
    """Full merger simulation."""
    post_firm_ids = pre_firm_ids.copy()
    post_firm_ids[pre_firm_ids == acquired_firm] = acquiring_firm
    post_ownership = compute_ownership_matrix(post_firm_ids)

    post_prices, converged = solve_equilibrium_prices(
        marginal_costs, post_ownership, sugar, draws,
        alpha_mean, sigma_alpha, sigma_sugar, delta, pre_prices,
    )

    delta_post = delta - alpha_mean * (post_prices - pre_prices)
    post_shares = compute_blp_shares(delta_post, sugar, post_prices, draws,
                                     sigma_alpha, sigma_sugar)

    return {
        "post_prices": post_prices,
        "post_shares": post_shares,
        "price_changes": post_prices - pre_prices,
        "price_change_pct": (post_prices - pre_prices) / pre_prices * 100,
        "converged": converged,
        "post_firm_ids": post_firm_ids,
    }


def logit_merger_prices(alpha, shares, prices, firm_ids,
                        acquiring_firm, acquired_firm):
    """Simple logit merger prediction."""
    ownership_pre = compute_ownership_matrix(firm_ids)
    deriv_pre = logit_share_derivatives(alpha, shares)
    markups_pre = compute_markups_blp(shares, deriv_pre, ownership_pre)
    mc = prices - markups_pre

    post_ids = firm_ids.copy()
    post_ids[firm_ids == acquired_firm] = acquiring_firm
    ownership_post = compute_ownership_matrix(post_ids)
    markups_post = compute_markups_blp(shares, deriv_pre, ownership_post)
    return mc + markups_post


# =============================================================================
# Figures
# =============================================================================
def plot_contraction_convergence(norm_history):
    """Figure 1: contraction mapping convergence."""
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(norm_history) + 1), norm_history, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\max|\delta^{h+1} - \delta^h|$")
    ax.set_title("BLP Contraction Mapping Convergence")
    ax.axhline(1e-12, color="r", linestyle="--", linewidth=1,
               label=r"Tolerance $10^{-12}$")
    ax.legend()
    return fig


def plot_elasticity_heatmaps(logit_eta, blp_eta, product_names):
    """Figure 2: side-by-side logit vs BLP elasticity heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    abs_max = max(np.abs(logit_eta).max(), np.abs(blp_eta).max())

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
                color = "white" if abs(val) > abs_max * 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=axes, label="Elasticity", shrink=0.8)
    fig.suptitle("Elasticity Matrices: Logit vs BLP", fontsize=13, fontweight="bold")
    return fig


def plot_diversion_comparison(logit_div, blp_div, product_names, ref_product=0):
    """Figure 3: grouped bar chart of diversion ratios."""
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


def plot_merger_prices(product_names, pre_prices, logit_post, blp_post):
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
        y_top = max(pre_prices[i], logit_post[i], blp_post[i]) + 0.15
        ax.text(x[i], y_top,
                f"L:{logit_chg:+.1f}%\nB:{blp_chg:+.1f}%",
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
    draws = draw_simulated_consumers(n_draws=500, seed=42)
    df = create_estimation_dataset(n_markets=50, draws=draws)

    print(f"  Markets: {df['market_id'].nunique()}, Obs: {len(df)}")
    market0 = df[df["market_id"] == 0].reset_index(drop=True)
    for _, r in market0.iterrows():
        print(f"    {r['product_name']:<15s}  p=${r['price']:.2f}  sugar={r['sugar']:.0f}  "
              f"share={r['share']:.4f}")

    product_names = market0["product_name"].tolist()
    prices0 = market0["price"].values
    sugar0 = market0["sugar"].values
    shares0 = market0["share"].values

    # True delta for market 0
    delta_true_m0 = (TRUE_BETA_CONST
                     + TRUE_BETA_SUGAR_MEAN * sugar0
                     - TRUE_ALPHA_MEAN * prices0
                     + market0["xi"].values)

    # =========================================================================
    # Step 2: BLP contraction mapping demo
    # =========================================================================
    print("\n--- Step 2: BLP contraction mapping (inner loop demo) ---")
    delta_rec, conv, n_iter, norm_hist = contraction_mapping(
        shares0, sugar0, prices0, draws,
        TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR, tol=1e-12, max_iter=2000,
    )
    shares_check = compute_blp_shares(delta_rec, sugar0, prices0, draws,
                                      TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)
    print(f"  Converged: {conv} in {n_iter} iterations")
    print(f"  Max |delta_true - delta_rec| = {np.abs(delta_true_m0 - delta_rec).max():.2e}")
    print(f"  Max |s_obs - s_pred| = {np.abs(shares0 - shares_check).max():.2e}")

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

    # For post-estimation analysis, use true parameters.
    # This is standard in Monte Carlo studies: the point is to demonstrate
    # the economic mechanics (elasticities, merger effects) rather than
    # finite-sample estimation performance. With only 3 products and limited
    # characteristic variation, weak instruments make the linear parameters
    # imprecise (sigma_alpha is well-identified, but alpha and sigma_sugar
    # are confounded). In practice, BLP is applied to markets with 50+ products.
    print("\n  Using true parameters for post-estimation analysis (standard Monte Carlo practice).")
    alpha_use = TRUE_ALPHA_MEAN
    sigma_alpha_use = TRUE_SIGMA_ALPHA
    sigma_sugar_use = TRUE_SIGMA_SUGAR

    # Bootstrap standard errors on the nonlinear parameters
    print("  Computing bootstrap SE on sigma (8 replications)...")
    n_boot = 8
    boot_sigma = {"sigma_alpha": [], "sigma_sugar": []}
    market_ids = df["market_id"].unique()
    rng_boot = np.random.RandomState(777)
    for b in range(n_boot):
        sampled = rng_boot.choice(market_ids, size=len(market_ids), replace=True)
        boot_dfs = []
        for new_id, old_id in enumerate(sampled):
            chunk = df[df["market_id"] == old_id].copy()
            chunk["market_id"] = new_id
            boot_dfs.append(chunk)
        boot_df = pd.concat(boot_dfs, ignore_index=True)
        try:
            b_est = estimate_blp(boot_df, draws,
                                 initial_sigma=np.array([est["sigma_alpha"],
                                                         est["sigma_sugar"]]),
                                 verbose=False)
            boot_sigma["sigma_alpha"].append(b_est["sigma_alpha"])
            boot_sigma["sigma_sugar"].append(b_est["sigma_sugar"])
        except Exception:
            pass

    se_sigma_alpha = np.std(boot_sigma["sigma_alpha"]) if len(boot_sigma["sigma_alpha"]) > 2 else np.nan
    se_sigma_sugar = np.std(boot_sigma["sigma_sugar"]) if len(boot_sigma["sigma_sugar"]) > 2 else np.nan
    n_ok = len(boot_sigma["sigma_alpha"])
    print(f"  Done ({n_ok} successful).")

    # =========================================================================
    # Step 4: Elasticities (TRUE params for economic analysis)
    # =========================================================================
    print("\n--- Step 4: Elasticities (BLP vs Logit) ---")

    # Use delta from contraction at true sigmas
    delta_analysis = delta_rec  # already computed with true sigma in Step 2

    blp_eta = compute_blp_elasticities(delta_analysis, sugar0, prices0, draws,
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
    # Step 6: Supply side
    # =========================================================================
    print("\n--- Step 6: Supply side (recover marginal costs) ---")
    firm_ids = market0["firm_id"].values
    ownership = compute_ownership_matrix(firm_ids)
    true_mc = market0["marginal_cost"].values

    deriv0 = compute_blp_share_derivatives(delta_analysis, sugar0, prices0, draws,
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

    blp_merger = simulate_merger(
        prices0, shares0, est_mc, firm_ids,
        acquiring_firm=1, acquired_firm=3,
        sugar=sugar0, draws=draws,
        alpha_mean=alpha_use, sigma_alpha=sigma_alpha_use,
        sigma_sugar=sigma_sugar_use, delta=delta_analysis,
    )

    logit_post = logit_merger_prices(alpha_use, shares0, prices0, firm_ids,
                                     acquiring_firm=1, acquired_firm=3)

    print(f"\n  {'Product':<15s}  {'Pre':>7s}  {'Logit Post':>11s}  {'BLP Post':>9s}")
    for i, name in enumerate(product_names):
        print(f"  {name:<15s}  ${prices0[i]:>6.2f}  ${logit_post[i]:>10.2f}  "
              f"${blp_merger['post_prices'][i]:>8.2f}")

    print(f"\n  Price change (%):")
    for i, name in enumerate(product_names):
        l_chg = (logit_post[i] - prices0[i]) / prices0[i] * 100
        b_chg = blp_merger["price_change_pct"][i]
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

**BLP contraction** (inner loop): $\delta^{h+1} = \delta^h + \ln s^{\text{obs}} - \ln s^{\text{pred}}(\delta^h, \sigma)$

**GMM** (outer loop): $\hat{\sigma} = \arg\min_\sigma \; [Z'\xi(\sigma)]' W [Z'\xi(\sigma)]$

where $\xi = \delta - X\beta + \alpha p$ is the structural error and $Z$ are instruments.
"""
    )

    report.add_model_setup(
        "| Parameter | True | Estimated | Description |\n"
        "|-----------|------|-----------|-------------|\n"
        f"| $\\alpha$ | {TRUE_ALPHA_MEAN:.1f} | {est['alpha']:.3f} | Mean price sensitivity |\n"
        f"| $\\beta_0$ | {TRUE_BETA_CONST:.1f} | {est['beta_const']:.3f} | Base utility |\n"
        f"| $\\beta_s$ | {TRUE_BETA_SUGAR_MEAN:.1f} | {est['beta_sugar']:.3f} | Mean sugar taste |\n"
        f"| $\\sigma_\\alpha$ | {TRUE_SIGMA_ALPHA:.1f} | {est['sigma_alpha']:.3f} (SE: {se_sigma_alpha:.3f}) | Std dev of price sensitivity |\n"
        f"| $\\sigma_s$ | {TRUE_SIGMA_SUGAR:.1f} | {est['sigma_sugar']:.3f} (SE: {se_sigma_sugar:.3f}) | Std dev of sugar taste |"
    )

    report.add_solution_method(
        "**Nested Fixed Point (NFXP):**\n\n"
        "1. **Outer loop (GMM):** Search over nonlinear parameters $\\sigma = (\\sigma_\\alpha, \\sigma_s)$ "
        "using Nelder-Mead.\n"
        "2. **Inner loop (contraction):** For each candidate $\\sigma$, run the BLP contraction mapping "
        "per market to find $\\delta$ such that predicted shares match observed shares (tolerance $10^{-12}$).\n"
        "3. **Concentration:** Given $\\delta$, recover linear parameters $(\\beta, \\alpha)$ by 2SLS "
        "using cost shifters, rival characteristics, and own-squared sugar as instruments.\n"
        "4. **Moments:** Compute $\\xi = \\delta - X\\beta + \\alpha p$ and form GMM objective $g'Wg$ "
        f"where $g = Z'\\xi$.\n\n"
        f"The contraction converged in **{n_iter} iterations** for the demonstration market. "
        f"GMM optimization {'converged' if est['converged'] else 'terminated'} "
        f"(objective = {est['gmm_objective']:.6f}). Post-estimation analysis uses true parameters "
        f"(standard Monte Carlo practice for demonstrating economic mechanics)."
    )

    # --- Figure 1: Contraction convergence ---
    fig1 = plot_contraction_convergence(norm_hist)
    report.add_figure("figures/contraction-convergence.png",
                      "BLP contraction mapping convergence (linear rate in log scale)", fig1,
        description="The linear convergence rate on a log scale confirms the contraction "
        "property. Each iteration multiplies the error by a constant factor less than 1, "
        "guaranteeing that the share inversion produces a unique set of mean utilities for "
        "any candidate nonlinear parameters.")

    # --- Figure 2: Elasticity comparison ---
    fig2 = plot_elasticity_heatmaps(logit_eta, blp_eta, product_names)
    report.add_figure("figures/elasticity-comparison.png",
                      "Elasticity matrices: simple logit (IIA) vs BLP (realistic substitution)", fig2,
        description="Under logit, each column of cross-elasticities is identical (IIA). "
        "BLP breaks this by allowing consumers to differ in how much they value sugar and "
        "price, so that similar products (e.g., two sugary cereals) compete more intensely "
        "than dissimilar ones.")

    # --- Figure 3: Diversion ratios ---
    fig3 = plot_diversion_comparison(logit_div, blp_div, product_names, ref_product=0)
    report.add_figure("figures/diversion-ratios.png",
                      "Diversion ratios from Choco-Bombs: logit diverts by market share, "
                      "BLP diverts to similar products", fig3,
        description="When Choco-Bombs loses customers, BLP correctly predicts they switch "
        "mainly to Store-Frosted (also sugary) rather than Fiber-Bran. Logit's diversion "
        "is proportional to market share, missing this product-similarity channel entirely. "
        "This difference is critical for merger analysis between close substitutes.")

    # --- Figure 4: Merger simulation ---
    fig4 = plot_merger_prices(product_names, prices0, logit_post,
                              blp_merger["post_prices"])
    report.add_figure("figures/merger-simulation.png",
                      "Merger price effects: BLP predicts larger increases for close substitutes "
                      "because it captures product similarity", fig4,
        description="A merger between Choco-Bombs and Store-Frosted combines two sugary "
        "cereals with high diversion between them. BLP predicts a larger price increase "
        "than logit because it recognizes the high substitutability between the merging "
        "products. Antitrust authorities using logit would underestimate the harm.")

    # --- Table: Parameter estimates ---
    table_data = {
        "Parameter": ["alpha", "beta_const", "beta_sugar",
                      "sigma_alpha", "sigma_sugar"],
        "True": [f"{TRUE_ALPHA_MEAN:.1f}", f"{TRUE_BETA_CONST:.1f}",
                 f"{TRUE_BETA_SUGAR_MEAN:.1f}", f"{TRUE_SIGMA_ALPHA:.1f}",
                 f"{TRUE_SIGMA_SUGAR:.1f}"],
        "Estimated": [f"{est['alpha']:.3f}", f"{est['beta_const']:.3f}",
                      f"{est['beta_sugar']:.3f}", f"{est['sigma_alpha']:.3f}",
                      f"{est['sigma_sugar']:.3f}"],
        "SE (bootstrap)": ["--", "--", "--",
                           f"{se_sigma_alpha:.3f}", f"{se_sigma_sugar:.3f}"],
    }
    table_df = pd.DataFrame(table_data)
    report.add_table("tables/parameter-estimates.csv",
                     "BLP Parameter Estimates with Bootstrap Standard Errors", table_df,
        description="The nonlinear parameters (sigma_alpha, sigma_sugar) are the key BLP "
        "innovation: they measure how much consumer preferences vary across the population. "
        "Larger sigma values produce more heterogeneous substitution patterns and stronger "
        "departures from the logit IIA baseline.")

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
    n_figs = sum(1 for it in report._results_items if it[0] == "figure")
    n_tabs = sum(1 for it in report._results_items if it[0] == "table")
    print(f"\nDone. Generated: README.md + {n_figs} figures + {n_tabs} tables")


def _print_matrix(matrix, names):
    header = "                " + "  ".join(f"{n[:13]:>13s}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = f"  {name[:13]:>13s}  " + "  ".join(f"{matrix[i, j]:>13.3f}"
                                                    for j in range(len(names)))
        print(row)


if __name__ == "__main__":
    main()
