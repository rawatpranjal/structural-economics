#!/usr/bin/env python3
"""Logit Discrete Choice Model: Maximum Likelihood Estimation.

Estimates a multinomial logit model of consumer choice over differentiated
products using simulated data. Demonstrates MLE, standard error computation
from the Hessian, and the IIA property of the logit model.

Reference: McFadden (1974), Train (2009) "Discrete Choice Methods with Simulation"
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# Add repo root to path for lib/ imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure
from lib.output import ModelReport


def logit_probabilities(V):
    """Compute logit choice probabilities from deterministic utilities.

    Parameters
    ----------
    V : ndarray, shape (N, J)
        Deterministic utility for each consumer-alternative pair.

    Returns
    -------
    probs : ndarray, shape (N, J)
        Choice probabilities P(j|x_i) for each consumer and alternative.
    """
    # Subtract max for numerical stability (log-sum-exp trick)
    V_shifted = V - V.max(axis=1, keepdims=True)
    exp_V = np.exp(V_shifted)
    return exp_V / exp_V.sum(axis=1, keepdims=True)


def log_likelihood(params, X_price, X_quality, choices, N, J):
    """Negative log-likelihood for the multinomial logit model.

    Parameters
    ----------
    params : array-like, shape (2,)
        [beta_price, beta_quality]
    X_price : ndarray, shape (J,)
        Price of each alternative.
    X_quality : ndarray, shape (J,)
        Quality of each alternative.
    choices : ndarray, shape (N,)
        Observed choice index for each consumer.
    N : int
        Number of consumers.
    J : int
        Number of alternatives.

    Returns
    -------
    neg_ll : float
        Negative log-likelihood (for minimization).
    """
    beta_price, beta_quality = params

    # Deterministic utility: V_j = beta_price * price_j + beta_quality * quality_j
    V_j = beta_price * X_price + beta_quality * X_quality  # shape (J,)
    V = np.tile(V_j, (N, 1))  # shape (N, J)

    probs = logit_probabilities(V)

    # Log-likelihood: sum of log P(chosen alternative)
    chosen_probs = probs[np.arange(N), choices]
    ll = np.sum(np.log(np.maximum(chosen_probs, 1e-300)))

    return -ll


def main():
    # =========================================================================
    # Parameters
    # =========================================================================
    np.random.seed(42)

    N = 5000          # Number of consumers
    J = 5             # Number of alternatives (products)

    # True parameters
    beta_price_true = -0.5    # Negative: higher price -> lower utility
    beta_quality_true = 1.2   # Positive: higher quality -> higher utility

    # =========================================================================
    # Generate product characteristics
    # =========================================================================
    product_names = [f"Product {j+1}" for j in range(J)]
    X_price = np.array([2.0, 3.5, 5.0, 7.0, 10.0])       # Prices
    X_quality = np.array([1.0, 2.0, 3.5, 4.0, 5.0])       # Quality levels

    # =========================================================================
    # Simulate choices
    # =========================================================================
    # Deterministic utility
    V_true = beta_price_true * X_price + beta_quality_true * X_quality  # shape (J,)
    V_all = np.tile(V_true, (N, 1))  # shape (N, J)

    # Add Type I Extreme Value (Gumbel) errors
    epsilon = np.random.gumbel(loc=0, scale=1, size=(N, J))
    U = V_all + epsilon

    # Each consumer chooses the alternative with highest total utility
    choices = np.argmax(U, axis=1)

    # Observed market shares
    actual_shares = np.bincount(choices, minlength=J) / N

    # True predicted shares (from the logit formula, without error)
    true_probs = logit_probabilities(V_all)
    true_shares = true_probs[0, :]  # Same for all consumers (no individual variation)

    print("Product characteristics and observed shares:")
    for j in range(J):
        print(f"  {product_names[j]}: price={X_price[j]:.1f}, "
              f"quality={X_quality[j]:.1f}, share={actual_shares[j]:.3f}")

    # =========================================================================
    # Maximum Likelihood Estimation
    # =========================================================================
    print("\nEstimating logit model via MLE...")

    # Starting values
    x0 = np.array([0.0, 0.0])

    result = minimize(
        log_likelihood,
        x0,
        args=(X_price, X_quality, choices, N, J),
        method="BFGS",
    )

    beta_hat = result.x
    beta_price_hat, beta_quality_hat = beta_hat

    # Standard errors from inverse Hessian
    se = np.sqrt(np.diag(result.hess_inv))
    se_price, se_quality = se

    # t-statistics
    t_price = beta_price_hat / se_price
    t_quality = beta_quality_hat / se_quality

    print(f"\n  beta_price:   {beta_price_hat:+.4f}  (SE = {se_price:.4f}, t = {t_price:.2f})")
    print(f"  beta_quality: {beta_quality_hat:+.4f}  (SE = {se_quality:.4f}, t = {t_quality:.2f})")
    print(f"  Log-likelihood: {-result.fun:.2f}")
    print(f"  True values: beta_price = {beta_price_true}, beta_quality = {beta_quality_true}")

    # =========================================================================
    # Predicted shares from estimated model
    # =========================================================================
    V_hat = beta_price_hat * X_price + beta_quality_hat * X_quality
    V_hat_all = np.tile(V_hat, (N, 1))
    predicted_probs = logit_probabilities(V_hat_all)
    predicted_shares = predicted_probs[0, :]

    # =========================================================================
    # Own-price elasticities
    # =========================================================================
    # For the logit model:
    #   Own-price elasticity:  eta_jj = beta_price * price_j * (1 - s_j)
    #   Cross-price elasticity: eta_jk = -beta_price * price_k * s_k
    own_elasticities = beta_price_hat * X_price * (1 - predicted_shares)
    cross_elasticity_matrix = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if j == k:
                cross_elasticity_matrix[j, k] = own_elasticities[j]
            else:
                cross_elasticity_matrix[j, k] = -beta_price_hat * X_price[k] * predicted_shares[k]

    # =========================================================================
    # IIA illustration: remove product 3 and check share ratios
    # =========================================================================
    # IIA predicts that the ratio of probabilities between any two remaining
    # alternatives is unchanged when a third alternative is removed.
    remove_j = 2  # Remove product 3 (index 2)
    remaining = [j for j in range(J) if j != remove_j]

    # Full-choice-set ratios
    full_ratios_01 = predicted_shares[0] / predicted_shares[1]
    full_ratios_03 = predicted_shares[0] / predicted_shares[3]

    # Restricted choice set
    V_restricted = V_hat[remaining]
    V_restricted_all = np.tile(V_restricted, (N, 1))
    restricted_probs = logit_probabilities(V_restricted_all)
    restricted_shares = restricted_probs[0, :]

    # Map back: remaining[0]=0, remaining[1]=1, remaining[2]=3, remaining[3]=4
    restricted_ratios_01 = restricted_shares[0] / restricted_shares[1]
    restricted_ratios_03 = restricted_shares[0] / restricted_shares[2]  # index 2 maps to product 4 (orig index 3)

    # =========================================================================
    # Generate Report
    # =========================================================================
    setup_style()

    report = ModelReport(
        "Logit Discrete Choice Model",
        "Maximum likelihood estimation of a multinomial logit demand model with simulated consumer choice data.",
    )

    report.add_overview(
        "The multinomial logit model is the workhorse of discrete choice analysis in "
        "industrial organization. Consumers choose among $J$ differentiated products to "
        "maximize utility, which depends on observed product characteristics (price, quality) "
        "and an unobserved idiosyncratic taste shock drawn from a Type I Extreme Value "
        "distribution.\n\n"
        "This distributional assumption yields the elegant logit choice probability formula "
        "and makes maximum likelihood estimation tractable. The model is the foundation for "
        "BLP (1995) and virtually all modern demand estimation in IO."
    )

    report.add_equations(
        r"""
**Utility:**
$$U_{ij} = \beta_{\text{price}} \, p_j + \beta_{\text{quality}} \, q_j + \varepsilon_{ij}$$

where $\varepsilon_{ij} \sim$ Type I Extreme Value (Gumbel) i.i.d. across consumers and products.

**Choice probability (McFadden, 1974):**
$$P(i \text{ chooses } j) = \frac{\exp(V_j)}{\sum_{k=1}^{J} \exp(V_k)}, \qquad V_j = \beta_{\text{price}} \, p_j + \beta_{\text{quality}} \, q_j$$

**Log-likelihood:**
$$\ell(\beta) = \sum_{i=1}^{N} \ln P(y_i \mid x; \beta)$$

**Own-price elasticity:**
$$\eta_{jj} = \beta_{\text{price}} \, p_j \, (1 - s_j)$$

**Cross-price elasticity (IIA):**
$$\eta_{jk} = -\beta_{\text{price}} \, p_k \, s_k$$
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $N$ | {N} | Number of consumers |\n"
        f"| $J$ | {J} | Number of alternatives |\n"
        f"| $\\beta_{{\\text{{price}}}}$ | {beta_price_true} | True price coefficient |\n"
        f"| $\\beta_{{\\text{{quality}}}}$ | {beta_quality_true} | True quality coefficient |\n"
        f"| Estimation | MLE via BFGS | scipy.optimize.minimize |"
    )

    report.add_solution_method(
        "**Maximum Likelihood Estimation (MLE):** Given observed choices $\\{y_i\\}_{i=1}^N$ "
        "and product characteristics, we maximize the log-likelihood function:\n\n"
        "$$\\hat{\\beta} = \\arg\\max_{\\beta} \\sum_{i=1}^N \\ln P(y_i \\mid x; \\beta)$$\n\n"
        "The logit log-likelihood is globally concave (McFadden, 1974), so any gradient-based "
        "optimizer converges to the unique global maximum. We use BFGS, which also produces an "
        "approximation to the inverse Hessian for standard error computation.\n\n"
        f"Converged in **{result.nit} iterations** "
        f"(log-likelihood = {-result.fun:.2f})."
    )

    # --- Figure 1: Log-Likelihood Surface (Contour Plot) ---
    n_grid = 80
    bp_grid = np.linspace(beta_price_true - 0.3, beta_price_true + 0.3, n_grid)
    bq_grid = np.linspace(beta_quality_true - 0.3, beta_quality_true + 0.3, n_grid)
    BP, BQ = np.meshgrid(bp_grid, bq_grid)
    LL = np.zeros_like(BP)

    for i in range(n_grid):
        for j_idx in range(n_grid):
            LL[i, j_idx] = -log_likelihood(
                [BP[i, j_idx], BQ[i, j_idx]],
                X_price, X_quality, choices, N, J,
            )

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    cs = ax1.contourf(BP, BQ, LL, levels=30, cmap="RdYlBu_r")
    ax1.contour(BP, BQ, LL, levels=15, colors="k", linewidths=0.3, alpha=0.4)
    plt.colorbar(cs, ax=ax1, label="Log-likelihood")
    ax1.plot(beta_price_true, beta_quality_true, "w*", markersize=15, markeredgecolor="k",
             markeredgewidth=1.2, label="True parameters")
    ax1.plot(beta_price_hat, beta_quality_hat, "r^", markersize=12, markeredgecolor="k",
             markeredgewidth=1.0, label="MLE estimate")
    ax1.set_xlabel(r"$\beta_{\mathrm{price}}$")
    ax1.set_ylabel(r"$\beta_{\mathrm{quality}}$")
    ax1.set_title("Log-Likelihood Surface")
    ax1.legend(loc="lower left")
    report.add_figure("figures/log-likelihood-surface.png",
                       "Log-likelihood surface: the logit likelihood is globally concave, with the MLE close to the true parameters",
                       fig1,
                       description="The single peak confirms global concavity of the logit likelihood -- any gradient-based optimizer will find the same maximum. "
                       "Notice how tightly the MLE estimate clusters near the true parameters, illustrating the statistical precision achievable with N=5000 observations.")

    # --- Figure 2: Predicted vs Actual Market Shares ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(J)
    width = 0.25
    bars1 = ax2.bar(x_pos - width, actual_shares, width, label="Observed", color="#2196F3", edgecolor="black", linewidth=0.5)
    bars2 = ax2.bar(x_pos, predicted_shares, width, label="Predicted (MLE)", color="#FF9800", edgecolor="black", linewidth=0.5)
    bars3 = ax2.bar(x_pos + width, true_shares, width, label="True model", color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Product")
    ax2.set_ylabel("Market share")
    ax2.set_title("Predicted vs Actual Market Shares")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(product_names, rotation=15)
    ax2.legend()
    report.add_figure("figures/market-shares.png",
                       "Predicted vs actual market shares: the estimated model closely recovers observed choice frequencies",
                       fig2,
                       description="Close agreement between predicted and observed shares validates the model specification. "
                       "Any systematic gap would signal omitted product characteristics or misspecification of the utility function.")

    # --- Figure 3: Own-Price Elasticities ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    bars = ax3.bar(x_pos, own_elasticities, color=colors, edgecolor="black", linewidth=0.5)
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.set_xlabel("Product")
    ax3.set_ylabel("Own-price elasticity")
    ax3.set_title("Own-Price Elasticities by Product")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{name}\n(p={X_price[j]:.1f}, s={predicted_shares[j]:.2f})"
                          for j, name in enumerate(product_names)], fontsize=9)
    # Annotate values
    for bar_item, val in zip(bars, own_elasticities):
        ax3.text(bar_item.get_x() + bar_item.get_width() / 2, val - 0.15,
                 f"{val:.2f}", ha="center", va="top", fontsize=9, fontweight="bold")
    report.add_figure("figures/own-price-elasticities.png",
                       "Own-price elasticities: higher-priced products have more elastic demand in the logit model",
                       fig3,
                       description="In the logit, own-price elasticity is driven by both the price level and market share through the formula "
                       "$\\eta_{jj} = \\beta_p \\, p_j (1 - s_j)$. Higher-priced products lose a larger fraction of their customers for a given percentage price increase, "
                       "which has direct implications for optimal pricing and merger simulation.")

    # --- Figure 4: IIA Illustration ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Share reallocation after removing product 3
    remaining_names = [product_names[j] for j in remaining]
    ax4a.bar(np.arange(len(remaining)) - 0.15, predicted_shares[remaining],
             0.3, label="Full choice set", color="#2196F3", edgecolor="black", linewidth=0.5)
    ax4a.bar(np.arange(len(remaining)) + 0.15, restricted_shares,
             0.3, label=f"Without {product_names[remove_j]}", color="#FF5722", edgecolor="black", linewidth=0.5)
    ax4a.set_xlabel("Product")
    ax4a.set_ylabel("Market share")
    ax4a.set_title(f"Share Reallocation after Removing {product_names[remove_j]}")
    ax4a.set_xticks(np.arange(len(remaining)))
    ax4a.set_xticklabels(remaining_names, rotation=15)
    ax4a.legend()

    # Panel B: Ratio preservation (IIA)
    pairs = [
        (f"P1/P2", full_ratios_01, restricted_ratios_01),
        (f"P1/P4", full_ratios_03, restricted_ratios_03),
    ]
    pair_labels = [p[0] for p in pairs]
    full_vals = [p[1] for p in pairs]
    rest_vals = [p[2] for p in pairs]
    x_pairs = np.arange(len(pairs))
    ax4b.bar(x_pairs - 0.15, full_vals, 0.3, label="Full choice set",
             color="#2196F3", edgecolor="black", linewidth=0.5)
    ax4b.bar(x_pairs + 0.15, rest_vals, 0.3, label=f"Without {product_names[remove_j]}",
             color="#FF5722", edgecolor="black", linewidth=0.5)
    ax4b.set_xlabel("Share ratio")
    ax4b.set_ylabel("Ratio value")
    ax4b.set_title("IIA: Share Ratios Unchanged")
    ax4b.set_xticks(x_pairs)
    ax4b.set_xticklabels(pair_labels)
    ax4b.legend()
    fig4.tight_layout()
    report.add_figure("figures/iia-illustration.png",
                       "IIA property: removing an alternative does not change the ratio of choice probabilities between remaining alternatives",
                       fig4,
                       description="The left panel shows that removing a product reallocates its share proportionally to all remaining products, regardless of similarity. "
                       "The right panel confirms that pairwise share ratios are exactly preserved -- this is the IIA property, and it is the main limitation motivating the nested logit and BLP models.")

    # --- Table: Estimation Results ---
    table_data = {
        "Parameter": [r"beta_price", r"beta_quality"],
        "True": [f"{beta_price_true:.4f}", f"{beta_quality_true:.4f}"],
        "Estimate": [f"{beta_price_hat:.4f}", f"{beta_quality_hat:.4f}"],
        "Std Error": [f"{se_price:.4f}", f"{se_quality:.4f}"],
        "t-stat": [f"{t_price:.2f}", f"{t_quality:.2f}"],
        "p-value": [f"{2*(1-norm.cdf(abs(t_price))):.4f}",
                     f"{2*(1-norm.cdf(abs(t_quality))):.4f}"],
    }
    df_results = pd.DataFrame(table_data)
    report.add_table("tables/estimation-results.csv",
                      "MLE Estimation Results: Estimated vs True Parameters",
                      df_results,
                      description="Both coefficients are estimated with high precision and are statistically significant. "
                      "The negative price coefficient and positive quality coefficient confirm that consumers trade off price against quality as expected.")

    # --- Table: Elasticity Matrix ---
    elas_data = {"Product": product_names}
    for k in range(J):
        elas_data[product_names[k]] = [f"{cross_elasticity_matrix[j, k]:.3f}" for j in range(J)]
    df_elas = pd.DataFrame(elas_data)
    report.add_table("tables/elasticity-matrix.csv",
                      "Price Elasticity Matrix (row = product whose share changes, column = product whose price changes)",
                      df_elas,
                      description="The IIA property is visible in the cross-elasticity columns: all off-diagonal entries in a given column are identical, meaning every rival product "
                      "gains the same share when one product raises its price. This unrealistic substitution pattern is the key motivation for richer models like nested logit and BLP.")

    report.add_takeaway(
        "The multinomial logit is the workhorse of discrete choice demand estimation, "
        "but its elegance comes at a cost: the **Independence of Irrelevant Alternatives "
        "(IIA)** property.\n\n"
        "**Key insights:**\n"
        "- MLE recovers the true parameters precisely with N=5000 observations. The logit "
        "likelihood is globally concave, so estimation is fast and reliable.\n"
        "- Own-price elasticities depend on price level and market share: "
        "$\\eta_{jj} = \\beta_p \\, p_j (1 - s_j)$. Higher-priced products are more elastic.\n"
        "- **Cross-elasticities are proportional to market shares**, not to product similarity. "
        "When a product is removed, its share is reallocated to all remaining products in "
        "proportion to their existing shares. This is the same substitution pattern as in "
        "symmetric Bertrand competition.\n"
        "- IIA is unrealistic: if a luxury product exits the market, the logit predicts its "
        "share goes equally (proportionally) to budget and premium products. The nested logit "
        "and mixed logit (BLP) relax this restriction.\n"
        "- Despite its limitations, the logit remains the starting point for demand estimation "
        "because of its computational tractability and clean closed-form expressions."
    )

    report.add_references([
        "McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice Behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics*. Academic Press.",
        "Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition.",
        "Berry, S., Levinsohn, J., and Pakes, A. (1995). Automobile Prices in Market Equilibrium. *Econometrica*, 63(4), 841-890.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
