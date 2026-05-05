#!/usr/bin/env python3
"""Logit Demand Estimation with Supply-Side Markup Recovery.

Estimates the simple logit discrete-choice demand model using Berry (1994)
inversion, then recovers firm marginal costs from the Bertrand-Nash pricing
FOC -- without observing any accounting data.

Pipeline:
  1. Generate synthetic cereal market data (5 products, 100 markets)
  2. OLS estimation (biased by price endogeneity)
  3. IV/2SLS estimation (consistent, using cost shifters as instruments)
  4. Elasticity computation (demonstrating the IIA limitation)
  5. Supply side: ownership matrix, markup recovery, marginal cost estimation

Reference: Berry, S. (1994). "Estimating Discrete-Choice Models of Product
           Differentiation." RAND Journal of Economics 25(2), 242-262.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport

# =============================================================================
# TRUE PARAMETERS (what we are trying to recover)
# =============================================================================
TRUE_ALPHA = 1.5          # Price sensitivity (enters utility negatively)
TRUE_BETA_SUGAR = 0.3     # Taste for sugar
TRUE_BETA_FIBER = 0.5     # Taste for fiber
TRUE_BETA_CONST = 1.0     # Base utility constant


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_product_data(n_markets: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic cereal market panel: 5 products x T markets.

    Products differ in sugar/fiber content, unobserved quality (xi), and
    firm ownership.  Prices are set as a markup over marginal cost with
    deliberate correlation between xi and price (creating endogeneity).

    Characteristics vary slightly across markets (regional formulations)
    and xi has a market-specific component, so that beta coefficients and
    the unobserved quality are separately identified.
    """
    rng = np.random.default_rng(seed)

    products = {
        "product_id":        [1, 2, 3, 4, 5],
        "product_name":      ["Choco-Bombs", "Fiber-Bran", "Store-Frosted",
                              "Honey-Os", "Nutri-Crunch"],
        "sugar_base":        [10.0, 1.0, 8.0, 6.0, 3.0],
        "fiber_base":        [1.0, 9.0, 2.0, 4.0, 7.0],
        "xi_base":           [0.5, 0.2, -0.3, 0.1, 0.4],
        "firm_id":           [1, 2, 1, 3, 2],       # Firms 1 & 2 multi-product
        "marginal_cost_base": [1.5, 2.5, 1.0, 1.8, 2.2],
    }

    J = len(products["product_id"])
    rows = []
    for t in range(n_markets):
        cost_shock = rng.normal(0, 0.3)
        for j in range(J):
            # Characteristics vary across markets (regional formulations)
            sugar = products["sugar_base"][j] + rng.normal(0, 0.5)
            fiber = products["fiber_base"][j] + rng.normal(0, 0.5)

            # xi has product-level base + market-specific shock
            xi_jt = products["xi_base"][j] + rng.normal(0, 0.3)

            mc = products["marginal_cost_base"][j] + cost_shock + rng.normal(0, 0.1)

            # Endogeneity: firms observe xi_jt and set higher prices for
            # higher unobserved quality => cov(price, xi) > 0
            price = mc + rng.uniform(0.3, 0.8) + 0.5 * xi_jt

            rows.append({
                "market_id": t,
                "product_id": products["product_id"][j],
                "product_name": products["product_name"][j],
                "sugar": sugar,
                "fiber": fiber,
                "xi": xi_jt,
                "firm_id": products["firm_id"][j],
                "price": price,
                "marginal_cost": mc,
                "cost_shifter": cost_shock,
            })
    return pd.DataFrame(rows)


def compute_true_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market shares using the TRUE parameters.

    s_j = exp(delta_j) / (1 + sum_k exp(delta_k))
    delta_j = beta_0 + beta_sugar*sugar + beta_fiber*fiber - alpha*price + xi
    """
    df = df.copy()
    df["delta"] = (
        TRUE_BETA_CONST
        + TRUE_BETA_SUGAR * df["sugar"]
        + TRUE_BETA_FIBER * df["fiber"]
        - TRUE_ALPHA * df["price"]
        + df["xi"]
    )
    shares_list = []
    for _, market_df in df.groupby("market_id"):
        exp_delta = np.exp(market_df["delta"].values)
        denom = 1.0 + exp_delta.sum()
        s = exp_delta / denom
        s0 = 1.0 / denom
        for idx, sj in zip(market_df.index, s):
            shares_list.append({"index": idx, "share": sj, "outside_share": s0})
    sdf = pd.DataFrame(shares_list).set_index("index")
    df["share"] = sdf["share"]
    df["outside_share"] = sdf["outside_share"]
    return df


def generate_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """Create instruments for price: cost shifters and BLP-style IVs.

    Cost shifters affect price via supply but are excluded from demand.
    BLP instruments: sum of rival characteristics (exogenous variation
    in competitive environment).
    """
    df = df.copy()
    rival_sugar = []
    rival_fiber = []
    for _, row in df.iterrows():
        mkt = df[df["market_id"] == row["market_id"]]
        others = mkt[mkt["product_id"] != row["product_id"]]
        rival_sugar.append(others["sugar"].sum())
        rival_fiber.append(others["fiber"].sum())
    df["rival_sugar_sum"] = rival_sugar
    df["rival_fiber_sum"] = rival_fiber
    return df


def create_estimation_dataset(n_markets: int = 100) -> pd.DataFrame:
    """Full pipeline: generate data, compute shares, add instruments."""
    df = generate_product_data(n_markets=n_markets)
    df = compute_true_shares(df)
    df = generate_instruments(df)
    df["ln_share_ratio"] = np.log(df["share"]) - np.log(df["outside_share"])
    return df


# =============================================================================
# DEMAND-SIDE FUNCTIONS
# =============================================================================

def compute_shares(delta: np.ndarray):
    """Logit shares from mean utilities.  Returns (inside_shares, outside_share)."""
    exp_d = np.exp(delta)
    denom = 1.0 + exp_d.sum()
    return exp_d / denom, 1.0 / denom


def invert_shares(shares: np.ndarray, outside_share: float) -> np.ndarray:
    """Berry inversion: ln(s_j) - ln(s_0) = delta_j."""
    return np.log(shares) - np.log(outside_share)


def compute_elasticities(alpha: float, prices: np.ndarray,
                         shares: np.ndarray) -> np.ndarray:
    """JxJ elasticity matrix.

    Own:   eta_jj = -alpha * p_j * (1 - s_j)
    Cross: eta_jk =  alpha * p_k * s_k   (identical across rows -- IIA)
    """
    J = len(prices)
    E = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                E[j, k] = -alpha * prices[j] * (1 - shares[j])
            else:
                E[j, k] = alpha * prices[k] * shares[k]
    return E


# =============================================================================
# ESTIMATION (OLS and 2SLS)
# =============================================================================

def estimate_ols(Y: np.ndarray, X: np.ndarray) -> dict:
    """OLS (biased baseline).  X should already contain all RHS variables."""
    X_c = np.column_stack([np.ones(len(Y)), X])
    XtX_inv = np.linalg.inv(X_c.T @ X_c)
    beta_hat = XtX_inv @ X_c.T @ Y
    resid = Y - X_c @ beta_hat
    sigma2 = (resid @ resid) / (len(Y) - X_c.shape[1])
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    r2 = 1.0 - (resid @ resid) / ((Y - Y.mean()) @ (Y - Y.mean()))
    return {"coefficients": beta_hat, "std_errors": se,
            "residuals": resid, "r_squared": r2}


def estimate_2sls(Y: np.ndarray, X_exog: np.ndarray,
                  X_endog: np.ndarray, Z: np.ndarray) -> dict:
    """Two-Stage Least Squares.

    Stage 1: project endogenous regressors onto instrument space.
    Stage 2: regress Y on fitted values.
    """
    n = len(Y)
    W = np.column_stack([np.ones(n), X_exog, Z])          # first-stage regressors
    X = np.column_stack([np.ones(n), X_exog, X_endog])    # second-stage regressors

    # Stage 1
    WtW_inv = np.linalg.pinv(W.T @ W)
    P_W = W @ WtW_inv @ W.T
    X_endog_hat = P_W @ X_endog

    # Stage 2
    X_hat = np.column_stack([np.ones(n), X_exog, X_endog_hat])
    XhX_inv = np.linalg.pinv(X_hat.T @ X_hat)
    beta_hat = XhX_inv @ X_hat.T @ Y

    resid = Y - X @ beta_hat
    sigma2 = (resid @ resid) / max(n - X.shape[1], 1)
    se = np.sqrt(np.abs(np.diag(sigma2 * XhX_inv)))

    # First-stage F-statistic
    gamma_hat = WtW_inv @ W.T @ X_endog
    ss_resid = (X_endog - W @ gamma_hat) @ (X_endog - W @ gamma_hat)
    ss_total = (X_endog - X_endog.mean()) @ (X_endog - X_endog.mean())
    r2_fs = 1.0 - ss_resid / max(ss_total, 1e-10)
    f_stat = (r2_fs / max(Z.shape[1], 1)) / max((1 - r2_fs) / max(n - W.shape[1], 1), 1e-10)

    return {"coefficients": beta_hat, "std_errors": se, "residuals": resid,
            "t_stats": beta_hat / se, "first_stage_f": f_stat, "n_obs": n}


def estimate_logit_demand(df: pd.DataFrame) -> dict:
    """Full 2SLS logit demand estimation.

    LHS: ln(s_j) - ln(s_0) = beta_0 + beta_sugar*sugar + beta_fiber*fiber - alpha*price + xi
    Instruments for price: cost_shifter, rival_sugar_sum, rival_fiber_sum
    """
    Y = df["ln_share_ratio"].values
    X_exog = df[["sugar", "fiber"]].values
    X_endog = df["price"].values
    Z = df[["cost_shifter", "rival_sugar_sum", "rival_fiber_sum"]].values

    results = estimate_2sls(Y, X_exog, X_endog, Z)
    # coefficients: [const, sugar, fiber, price]
    results["alpha"] = -results["coefficients"][3]
    results["beta_sugar"] = results["coefficients"][1]
    results["beta_fiber"] = results["coefficients"][2]
    results["beta_const"] = results["coefficients"][0]
    return results


# =============================================================================
# SUPPLY SIDE
# =============================================================================

def compute_ownership_matrix(firm_ids: np.ndarray) -> np.ndarray:
    """O_jk = 1 if products j and k are owned by the same firm."""
    return (firm_ids[:, None] == firm_ids[None, :]).astype(float)


def compute_share_derivatives(alpha: float, shares: np.ndarray) -> np.ndarray:
    """ds_j/dp_k matrix.  Own: -alpha*s_j*(1-s_j).  Cross: alpha*s_j*s_k."""
    J = len(shares)
    D = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                D[j, k] = -alpha * shares[j] * (1.0 - shares[j])
            else:
                D[j, k] = alpha * shares[j] * shares[k]
    return D


def compute_markups(alpha: float, shares: np.ndarray,
                    ownership: np.ndarray) -> np.ndarray:
    """Recover markups from Bertrand-Nash FOC: p - mc = Omega^{-1} s.

    Omega_jk = -ds_k/dp_j * O_jk  (only internalise own-firm products).
    """
    D = compute_share_derivatives(alpha, shares)
    omega = -D * ownership
    return np.linalg.solve(omega, shares)


def recover_marginal_costs(prices: np.ndarray,
                           markups: np.ndarray) -> np.ndarray:
    """mc = p - markup.  The key: no accounting data needed."""
    return prices - markups


# =============================================================================
# MAIN
# =============================================================================

def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    N_MARKETS = 100
    N_PRODUCTS = 5

    # =========================================================================
    # Step 1: Generate data
    # =========================================================================
    print("Step 1: Generating synthetic cereal market data "
          f"({N_PRODUCTS} products, {N_MARKETS} markets) ...")
    df = create_estimation_dataset(n_markets=N_MARKETS)
    print(f"  Dataset: {len(df)} obs, {df['market_id'].nunique()} markets, "
          f"{df['product_id'].nunique()} products")

    # =========================================================================
    # Step 2: OLS estimation (biased)
    # =========================================================================
    print("\nStep 2: OLS estimation (biased due to price endogeneity) ...")
    Y = df["ln_share_ratio"].values
    X_ols = df[["sugar", "fiber", "price"]].values
    ols = estimate_ols(Y, X_ols)
    # coefficients: [const, sugar, fiber, price]
    ols_alpha = -ols["coefficients"][3]
    ols_beta_sugar = ols["coefficients"][1]
    ols_beta_fiber = ols["coefficients"][2]
    ols_beta_const = ols["coefficients"][0]
    print(f"  alpha  = {ols_alpha:.4f}  (true {TRUE_ALPHA})")
    print(f"  sugar  = {ols_beta_sugar:.4f}  (true {TRUE_BETA_SUGAR})")
    print(f"  fiber  = {ols_beta_fiber:.4f}  (true {TRUE_BETA_FIBER})")
    print(f"  const  = {ols_beta_const:.4f}  (true {TRUE_BETA_CONST})")
    print(f"  Bias in alpha: {(ols_alpha - TRUE_ALPHA) / TRUE_ALPHA * 100:+.1f}%")

    # =========================================================================
    # Step 3: IV / 2SLS estimation (consistent)
    # =========================================================================
    print("\nStep 3: IV/2SLS estimation (consistent) ...")
    iv = estimate_logit_demand(df)
    print(f"  alpha  = {iv['alpha']:.4f}  (true {TRUE_ALPHA})")
    print(f"  sugar  = {iv['beta_sugar']:.4f}  (true {TRUE_BETA_SUGAR})")
    print(f"  fiber  = {iv['beta_fiber']:.4f}  (true {TRUE_BETA_FIBER})")
    print(f"  const  = {iv['beta_const']:.4f}  (true {TRUE_BETA_CONST})")
    print(f"  First-stage F = {iv['first_stage_f']:.1f}")

    # =========================================================================
    # Step 4: Elasticities (market 0)
    # =========================================================================
    print("\nStep 4: Computing elasticities (market 0) ...")
    m0 = df[df["market_id"] == 0].copy()
    product_names = m0["product_name"].tolist()
    prices_m0 = m0["price"].values
    shares_m0 = m0["share"].values

    elast = compute_elasticities(iv["alpha"], prices_m0, shares_m0)
    print("  Own-price elasticities:")
    for i, name in enumerate(product_names):
        print(f"    {name}: {elast[i, i]:.3f}")

    # =========================================================================
    # Step 5: Supply side -- markup recovery
    # =========================================================================
    print("\nStep 5: Supply-side markup recovery ...")
    firm_ids = m0["firm_id"].values
    ownership = compute_ownership_matrix(firm_ids)
    markups = compute_markups(iv["alpha"], shares_m0, ownership)
    est_mc = recover_marginal_costs(prices_m0, markups)
    true_mc = m0["marginal_cost"].values

    print("  Price decomposition (market 0):")
    print(f"  {'Product':<16} {'Price':>7} {'Markup':>7} {'Est MC':>7} {'True MC':>7}")
    for i, name in enumerate(product_names):
        print(f"  {name:<16} ${prices_m0[i]:>6.2f} ${markups[i]:>6.2f} "
              f"${est_mc[i]:>6.2f} ${true_mc[i]:>6.2f}")
    print(f"  Mean |est MC - true MC| = ${np.abs(est_mc - true_mc).mean():.3f}")

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Logit Demand and Markup Recovery",
        "Berry inversion, price endogeneity, and the supply-side recovery of marginal costs.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A differentiated-products demand estimate becomes economically useful when it "
        "can say something about markups and marginal costs. In many IO applications, "
        "the researcher observes prices, shares, product characteristics, and firm "
        "ownership, but not accounting marginal cost. The supply side uses the firm's "
        "pricing first-order condition to recover those costs from demand curvature.\n\n"
        "This tutorial builds a synthetic cereal market with five products and three "
        "firms. Prices are deliberately endogenous because firms charge more for "
        "products with high unobserved quality. OLS therefore understates price "
        "sensitivity. IV/2SLS uses cost shifters and rival-characteristic instruments "
        "to recover demand, then the Bertrand-Nash FOC decomposes each observed price "
        "into a markup and marginal cost. The point is not that simple logit is a rich "
        "substitution model; the elasticity figures make its IIA restriction visible."
    )

    report.add_equations(r"""
There are markets $t$, products $j$, and an outside option. Mean utility is
$$
\delta_{jt}
=\beta_0+\beta_{\text{sugar}}x^{\text{sugar}}_{jt}
+\beta_{\text{fiber}}x^{\text{fiber}}_{jt}
-\alpha p_{jt}+\xi_{jt}.
$$

Simple logit shares satisfy
$$
s_{jt}=\frac{\exp(\delta_{jt})}{1+\sum_k \exp(\delta_{kt})},
\qquad
s_{0t}=\frac{1}{1+\sum_k \exp(\delta_{kt})}.
$$
Berry's inversion turns observed shares into a linear estimating equation:
$$
\log s_{jt}-\log s_{0t}
=\beta_0+\beta_{\text{sugar}}x^{\text{sugar}}_{jt}
+\beta_{\text{fiber}}x^{\text{fiber}}_{jt}
-\alpha p_{jt}+\xi_{jt}.
$$
The price coefficient is identified from price variation that is excluded from
$\xi_{jt}$, here cost shifters and rival characteristics.

The logit elasticity matrix is
$$
\eta_{jj}=-\alpha p_j(1-s_j), \qquad
\eta_{jk}=\alpha p_k s_k, \quad j\neq k.
$$
The cross-elasticity $\eta_{jk}$ depends on product $k$'s price and share, but
not on how close products $j$ and $k$ are. That is the IIA restriction.

On the supply side, firm $f$ chooses prices for its products. Product $j$'s FOC is
$$
0=s_j(p)+\sum_k
\mathbf 1[f(j)=f(k)](p_k-c_k)\frac{\partial s_k(p)}{\partial p_j}.
$$
Let $\Delta_{jk}=\partial s_j/\partial p_k$ and let $O_{jk}=1$ when products
$j$ and $k$ are owned by the same firm. The markup equation is
$$
p-c=-(O\circ \Delta')^{-1}s.
$$
Multi-product firms internalize business stolen from their own products, so the
ownership matrix is part of the cost recovery exercise.
""")

    report.add_model_setup(
        "The data are simulated so the true demand parameters and marginal costs are "
        "known. This lets the tutorial separate two errors that are often mixed in real "
        "applications: demand bias from endogenous prices and cost-recovery error from "
        "using the wrong demand curvature.\n\n"
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\alpha$ | {TRUE_ALPHA} | Price sensitivity |\n"
        f"| $\\beta_{{\\text{{sugar}}}}$ | {TRUE_BETA_SUGAR} | Sugar taste |\n"
        f"| $\\beta_{{\\text{{fiber}}}}$ | {TRUE_BETA_FIBER} | Fiber taste |\n"
        f"| $\\beta_0$ | {TRUE_BETA_CONST} | Base utility |\n"
        f"| Products | {N_PRODUCTS} | Choco-Bombs, Fiber-Bran, Store-Frosted, Honey-Os, Nutri-Crunch |\n"
        f"| Markets | {N_MARKETS} | Cross-sectional variation in costs |\n"
        f"| Firms | 3 | Firms 1 and 2 own 2 products each (multi-product) |"
    )

    report.add_solution_method(
        "The computation keeps demand estimation and supply inversion separate. First "
        "recover mean utilities from shares and estimate demand. Then take the estimated "
        "price coefficient into the Bertrand-Nash FOC for one market.\n\n"
        "```text\n"
        "Inputs: product characteristics, prices, shares, instruments, firm labels\n"
        "Outputs: demand estimates, elasticities, markups, recovered marginal costs\n\n"
        "1. Compute delta_jt = log(s_jt) - log(s_0t).\n"
        "2. Regress delta_jt on characteristics and price by OLS.\n"
        "3. Re-estimate by IV/2SLS using excluded cost shifters for price.\n"
        "4. In a representative market, form the logit derivative matrix Delta.\n"
        "5. Build ownership O from firm labels.\n"
        "6. Recover markups from p - c = -[(O .* Delta')]^{-1}s.\n"
        "7. Compare recovered marginal costs with the simulated truth.\n"
        "```\n\n"
        f"The first-stage F-statistic is {iv['first_stage_f']:.1f}, so the instrument "
        "set is strong in this synthetic design. That strength is deliberately built "
        "in; the exercise is about the mechanics of demand-side IV and supply-side "
        "markup recovery, not weak-instrument diagnostics."
    )

    report.add_results(
        f"OLS recovers a price-sensitivity estimate of {ols_alpha:.3f}, far below the "
        f"true value {TRUE_ALPHA:.3f}, because high-$\\xi$ products are both more popular "
        "and more expensive. IV/2SLS moves the estimate to "
        f"{iv['alpha']:.3f}. In market 0, the recovered marginal costs have mean absolute "
        f"error {np.abs(est_mc - true_mc).mean():.3f} dollars. The remaining figures show "
        "why that recovery is informative and where simple logit remains restrictive."
    )

    # --- Figure 1: OLS vs IV parameter estimates ---
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    param_names = [r"$\alpha$", r"$\beta_{\mathrm{sugar}}$",
                   r"$\beta_{\mathrm{fiber}}$", r"$\beta_0$"]
    true_vals = [TRUE_ALPHA, TRUE_BETA_SUGAR, TRUE_BETA_FIBER, TRUE_BETA_CONST]
    ols_vals = [ols_alpha, ols_beta_sugar, ols_beta_fiber, ols_beta_const]
    iv_vals = [iv["alpha"], iv["beta_sugar"], iv["beta_fiber"], iv["beta_const"]]
    iv_se = [iv["std_errors"][3], iv["std_errors"][1],
             iv["std_errors"][2], iv["std_errors"][0]]

    x_pos = np.arange(len(param_names))
    w = 0.25
    ax1.bar(x_pos - w, true_vals, w, label="True", color="#27ae60", edgecolor="black")
    ax1.bar(x_pos, ols_vals, w, label="OLS (biased)", color="#e74c3c",
            edgecolor="black", alpha=0.85)
    ax1.bar(x_pos + w, iv_vals, w, label="IV/2SLS", color="#3498db", edgecolor="black")
    ax1.errorbar(x_pos + w, iv_vals, yerr=[s * 1.96 for s in iv_se],
                 fmt="none", color="black", capsize=5, capthick=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names, fontsize=12)
    ax1.set_ylabel("Parameter value")
    ax1.set_title("OLS vs IV/2SLS Estimates\n(OLS biased by price endogeneity)")
    ax1.legend()
    ax1.axhline(0, color="black", linewidth=0.5)
    report.add_figure(
        "figures/estimation-comparison.png",
        "Parameter estimates: True vs OLS (biased) vs IV/2SLS (consistent). "
        "OLS attenuates price sensitivity because high-xi products command higher prices.",
        fig1,
        description="The gap between OLS (red) and the true value (green) for alpha is "
        "the endogeneity bias: OLS understates price sensitivity because unobserved quality "
        "raises both demand and price simultaneously. IV/2SLS (blue) recovers the correct "
        "parameter by isolating exogenous cost-driven price variation.",
    )

    # --- Figure 2: Elasticity heatmap ---
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    vmax = max(abs(elast.min()), abs(elast.max()))
    im = ax2.imshow(elast, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax2, label="Elasticity")
    short_names = [n[:12] for n in product_names]
    ax2.set_xticks(range(N_PRODUCTS))
    ax2.set_yticks(range(N_PRODUCTS))
    ax2.set_xticklabels(short_names, rotation=45, ha="right")
    ax2.set_yticklabels(short_names)
    for i in range(N_PRODUCTS):
        for j in range(N_PRODUCTS):
            color = "white" if abs(elast[i, j]) > vmax * 0.5 else "black"
            ax2.text(j, i, f"{elast[i, j]:.2f}", ha="center", va="center",
                     color=color, fontsize=9, fontweight="bold")
    ax2.set_xlabel("Price of product (column)")
    ax2.set_ylabel("Quantity of product (row)")
    ax2.set_title("Price Elasticity Matrix (Logit)\nOff-diagonal columns are identical (IIA)")
    report.add_figure(
        "figures/elasticity-heatmap.png",
        "Elasticity matrix. Cross-elasticities in each column are identical -- "
        "the IIA limitation of the simple logit.",
        fig2,
        description="Each column of cross-elasticities is identical because the logit model "
        "forces all products to be equally substitutable. This means that when a sugary cereal "
        "raises its price, the model predicts equal substitution to both a similar sugary cereal "
        "and a dissimilar fiber cereal -- an unrealistic restriction.",
    )

    # --- Figure 3: Price decomposition (stacked bar: MC + markup) ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x_prod = np.arange(N_PRODUCTS)
    bar_w = 0.3
    ax3.bar(x_prod - bar_w / 2, est_mc, bar_w, label="Estimated MC",
            color="#2ecc71", edgecolor="black")
    ax3.bar(x_prod - bar_w / 2, markups, bar_w, bottom=est_mc,
            label="Markup", color="#e74c3c", edgecolor="black")
    ax3.bar(x_prod + bar_w / 2 + 0.05, true_mc, bar_w, label="True MC",
            color="#3498db", edgecolor="black", alpha=0.7)
    ax3.scatter(x_prod - bar_w / 2, prices_m0, s=80, color="black",
                marker="_", linewidths=2.5, zorder=5, label="Observed price")
    ax3.set_xticks(x_prod)
    ax3.set_xticklabels(product_names, fontsize=10, rotation=20, ha="right")
    ax3.set_ylabel("Dollars ($)")
    ax3.set_title("Price Decomposition: Marginal Cost + Markup")
    ax3.legend(loc="upper right")
    # Value labels
    for i in range(N_PRODUCTS):
        if est_mc[i] > 0.3:
            ax3.text(i - bar_w / 2, est_mc[i] / 2, f"${est_mc[i]:.2f}",
                     ha="center", va="center", fontsize=8, color="white",
                     fontweight="bold")
        ax3.text(i - bar_w / 2, est_mc[i] + markups[i] / 2,
                 f"${markups[i]:.2f}", ha="center", va="center", fontsize=8,
                 color="white", fontweight="bold")
    report.add_figure(
        "figures/price-decomposition.png",
        "Price = marginal cost + markup. Estimated MC (green, from Bertrand-Nash FOC) "
        "compared with true MC (blue). No accounting data required.",
        fig3,
        description="The decomposition is the supply-side accounting exercise: demand estimates "
        "plus the Bertrand-Nash FOC turn observed prices into markup and marginal-cost "
        "components. Estimated costs are imperfect product by product because the estimated "
        "demand slope is not exactly the true slope, but the exercise recovers the economic "
        "object needed for counterfactual pricing. Multi-product firms (Choco-Bombs and "
        "Store-Frosted, both owned by Firm 1) charge higher markups because they internalize "
        "cannibalization between their own products.",
    )

    # --- Figure 4: IIA demonstration (cross-elasticity bar charts) ---
    fig4, axes4 = plt.subplots(1, N_PRODUCTS, figsize=(16, 5), sharey=True)
    bar_colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]
    for j, ax in enumerate(axes4):
        cross_e = elast[:, j].copy()
        cross_e[j] = 0.0
        other_idx = [i for i in range(N_PRODUCTS) if i != j]
        other_names_short = [product_names[i][:10] for i in other_idx]
        other_vals = [cross_e[i] for i in other_idx]
        colors_j = [bar_colors[i] for i in other_idx]
        ax.barh(other_names_short, other_vals, color=colors_j, edgecolor="black")
        ax.set_xlabel("Cross-elasticity")
        ax.set_title(f"If {product_names[j][:10]}\nraises price", fontsize=10,
                     fontweight="bold")
        for bar_patch, val in zip(ax.patches, other_vals):
            ax.text(val + 0.005, bar_patch.get_y() + bar_patch.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    fig4.suptitle(
        "IIA Problem: Cross-Elasticities Depend Only on Market Share, "
        "Not Product Similarity",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig4.tight_layout()
    report.add_figure(
        "figures/iia-demonstration.png",
        "IIA demonstration. When any product raises its price, substitution to "
        "each rival is proportional to that rival's market share -- not to how "
        "similar the products are.",
        fig4,
        description="Within each panel, notice that all bars have the same height -- every rival "
        "gains the same cross-elasticity regardless of product similarity. This is the IIA "
        "property in action. The BLP random coefficients model (see blp-random-coefficients/) "
        "breaks this restriction by allowing consumer heterogeneity.",
    )

    # --- Table: Estimation results ---
    table_data = {
        "Parameter": ["alpha", "beta_sugar", "beta_fiber", "beta_const"],
        "True": [f"{v:.3f}" for v in true_vals],
        "OLS": [f"{v:.3f}" for v in ols_vals],
        "IV/2SLS": [f"{v:.3f}" for v in iv_vals],
        "IV s.e.": [f"{v:.3f}" for v in iv_se],
    }
    df_table = pd.DataFrame(table_data)
    report.add_table(
        "tables/estimation-results.csv",
        "Estimation Results: True vs OLS vs IV/2SLS",
        df_table,
        description="Compare the OLS and IV/2SLS columns: the bias is concentrated in alpha "
        "(price sensitivity) because price is the endogenous variable. The characteristic "
        "coefficients (sugar, fiber) are less affected because product attributes have weaker "
        "correlation with the unobserved quality term.",
    )

    report.add_takeaway(
        "The supply-side object is not observed cost; it is the marginal cost that "
        "rationalizes observed prices under the estimated demand system and an ownership "
        "matrix. That makes the demand estimate consequential: attenuating price "
        "sensitivity also distorts markups and recovered costs. Simple logit is a clean "
        "benchmark because Berry inversion and the markup equation are transparent, but "
        "its IIA substitution pattern is too rigid for many merger and product-space "
        "applications. The natural next step is random-coefficients demand in "
        "[BLP](../blp-random-coefficients/), where substitution can vary with consumer "
        "heterogeneity and product characteristics."
    )

    report.add_references([
        "Berry, S. (1994). \"Estimating Discrete-Choice Models of Product "
        "Differentiation.\" *RAND Journal of Economics* 25(2), 242-262.",
        "Berry, S., Levinsohn, J., and Pakes, A. (1995). \"Automobile Prices in "
        "Market Equilibrium.\" *Econometrica* 63(4), 841-890.",
        "Nevo, A. (2001). \"Measuring Market Power in the Ready-to-Eat Cereal "
        "Industry.\" *Econometrica* 69(2), 307-342.",
        "Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge "
        "University Press, 2nd edition, Ch. 3.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures "
          f"+ {len(report._tables)} tables")


if __name__ == "__main__":
    main()
