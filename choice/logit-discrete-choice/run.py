#!/usr/bin/env python3
"""Product demand with plain logit and the IIA restriction.

Simulates choices in a small differentiated-products market, estimates price
and quality tastes by maximum likelihood, and shows how plain logit maps those
tastes into substitution patterns.

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
from lib.plotting import setup_style
from lib.output import ModelReport


def logit_probabilities(V: np.ndarray) -> np.ndarray:
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


def log_likelihood(
    params: np.ndarray,
    X_price: np.ndarray,
    X_quality: np.ndarray,
    choices: np.ndarray,
    N: int,
    J: int,
) -> float:
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

    return float(-ll)


def p_value_label(t_stat: float) -> str:
    """Format a two-sided normal p-value for a compact results table."""
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def main() -> None:
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
        "Product Demand with Plain Logit and IIA",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Five products sit on one shelf. Each product has a price and a quality "
        "level, and each buyer chooses one product.\n\n"
        "The object is product demand in this market. We want price and quality "
        "tastes, fitted shares, and buyer reallocation after removal.\n\n"
        "The computational task is maximum likelihood. Each trial coefficient "
        "vector gives logit probabilities, and the observed choices select the "
        "best-fitting vector."
    )

    report.add_equations(
        r"""
Consumers $i=1,\ldots,N$ choose one product $j\in\{1,\ldots,J\}$.
Product $j$ has price $p_j$ and quality $q_j$. The deterministic part of
utility is common across consumers in this simple market.

**Utility:**
$$U_{ij}=V_j+\varepsilon_{ij}, \qquad
V_j=\beta_p p_j+\beta_q q_j,$$

with $\varepsilon_{ij}$ i.i.d. Type I extreme value. The expected sign is
$\beta_p<0$ and $\beta_q>0$.

**Choice probability:**
$$P_j(\beta)=\Pr(y_i=j\mid p,q;\beta)
=\frac{\exp(V_j)}{\sum_{k=1}^J \exp(V_k)}.$$

If $d_{ij}=1\{y_i=j\}$ records the observed purchase, the sample
log-likelihood is
$$\ell(\beta)=\sum_{i=1}^N\sum_{j=1}^J d_{ij}\log P_j(\beta).$$

Because this one-market example has no individual covariates, fitted market
shares are $s_j=P_j(\hat\beta)$. The implied price elasticities are
$$\eta_{jj}=\beta_p p_j(1-s_j), \qquad
\eta_{jk}=-\beta_p p_k s_k \quad (j\neq k).$$

IIA follows from the odds ratio
$$\frac{P_j}{P_k}=\exp(V_j-V_k),$$
which does not depend on any third product in the choice set.
"""
    )

    report.add_model_setup(
        "The market is small enough to see each estimate. Five products trade off "
        "price against quality. The sample is synthetic, so true coefficients and "
        "population shares are available after estimation.\n\n"
        f"| Object | Value | Role |\n"
        f"|-----------|-------|-------------|\n"
        f"| Consumers | {N} | Independent choice draws |\n"
        f"| Products | {J} | Fixed alternatives in one market |\n"
        f"| Prices | {', '.join(f'{p:.1f}' for p in X_price)} | Utility shifter with negative coefficient |\n"
        f"| Quality | {', '.join(f'{q:.1f}' for q in X_quality)} | Utility shifter with positive coefficient |\n"
        f"| True $\\beta_p$ | {beta_price_true} | Price coefficient used to simulate choices |\n"
        f"| True $\\beta_q$ | {beta_quality_true} | Quality coefficient used to simulate choices |"
    )

    report.add_solution_method(
        "The likelihood turns the demand model into a two-parameter optimization "
        "problem. Each candidate $\\beta=(\\beta_p,\\beta_q)$ implies utilities. "
        "Those utilities imply probabilities, and the observed choices score the "
        "candidate through the log-likelihood:\n\n"
        "$$\\hat\\beta=\\arg\\max_\\beta \\ell(\\beta).$$\n\n"
        "```text\n"
        "Inputs: prices p_j, qualities q_j, choices y_i, starting value beta^(0)\n"
        "For each trial beta proposed by the optimizer:\n"
        "    1. Form V_j(beta) = beta_p p_j + beta_q q_j for every product j.\n"
        "    2. Convert V into logit probabilities P_j(beta).\n"
        "    3. Evaluate ell(beta) = sum_i log P_{y_i}(beta).\n"
        "Choose beta_hat that maximizes ell(beta).\n"
        "At beta_hat: compute fitted shares, elasticities, and IIA share ratios.\n"
        "```"
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
    ax1.set_title("Likelihood over price and quality tastes")
    ax1.legend(loc="lower left")
    report.add_figure("figures/log-likelihood-surface.png",
                       "Log-likelihood surface with true and estimated coefficients marked",
                       fig1,
                       description="The contour plot shows the two-parameter likelihood. "
                       "The MLE sits near the true coefficients used to generate the choices.")

    # --- Figure 2: Predicted vs Actual Market Shares ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(J)
    width = 0.25
    ax2.bar(x_pos - width, actual_shares, width, label="Observed", color="#2196F3", edgecolor="black", linewidth=0.5)
    ax2.bar(x_pos, predicted_shares, width, label="Predicted (MLE)", color="#FF9800", edgecolor="black", linewidth=0.5)
    ax2.bar(x_pos + width, true_shares, width, label="True model", color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Product")
    ax2.set_ylabel("Market share")
    ax2.set_title("Observed, fitted, and population shares")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(product_names, rotation=15)
    ax2.legend()
    report.add_figure("figures/market-shares.png",
                       "Observed, fitted, and true logit market shares",
                       fig2,
                       description="Observed shares are sample purchase frequencies. "
                       "Fitted shares are close to both the sample and population shares.")

    # --- Figure 3: Own-Price Elasticities ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    bars = ax3.bar(x_pos, own_elasticities, color=colors, edgecolor="black", linewidth=0.5)
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.set_xlabel("Product")
    ax3.set_ylabel("Own-price elasticity")
    ax3.set_title("Own-price elasticities")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{name}\n(p={X_price[j]:.1f}, s={predicted_shares[j]:.2f})"
                          for j, name in enumerate(product_names)], fontsize=9)
    # Annotate values
    for bar_item, val in zip(bars, own_elasticities):
        ax3.text(bar_item.get_x() + bar_item.get_width() / 2, val - 0.15,
                 f"{val:.2f}", ha="center", va="top", fontsize=9, fontweight="bold")
    report.add_figure("figures/own-price-elasticities.png",
                       "Own-price elasticities implied by the estimated logit",
                       fig3,
                       description="The own-price elasticities combine the estimated price coefficient with each product's price and fitted share. "
                       "Higher prices make demand more elastic in absolute value here.")

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
    ax4a.set_title(f"Shares after removing {product_names[remove_j]}")
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
    ax4b.set_title("IIA preserves odds ratios")
    ax4b.set_xticks(x_pairs)
    ax4b.set_xticklabels(pair_labels)
    ax4b.legend()
    fig4.tight_layout()
    report.add_figure("figures/iia-illustration.png",
                       "IIA reallocation after removing one alternative",
                       fig4,
                       description="Removing Product 3 raises every remaining share. "
                       "The pairwise odds ratios stay fixed, which is the IIA restriction.")

    # --- Table: Estimation Results ---
    table_data = {
        "Parameter": [r"beta_p", r"beta_q"],
        "True": [f"{beta_price_true:.4f}", f"{beta_quality_true:.4f}"],
        "Estimate": [f"{beta_price_hat:.4f}", f"{beta_quality_hat:.4f}"],
        "Std. error": [f"{se_price:.4f}", f"{se_quality:.4f}"],
        "t-stat": [f"{t_price:.2f}", f"{t_quality:.2f}"],
        "p-value": [p_value_label(t_price), p_value_label(t_quality)],
    }
    df_results = pd.DataFrame(table_data)
    report.add_table("tables/estimation-results.csv",
                      "MLE estimates and true coefficients",
                      df_results,
                      description="The estimated signs match the simulation: consumers dislike price and value quality. "
                      "The estimates are close to the true coefficients.")

    # --- Table: Elasticity Matrix ---
    elas_data = {"Product": product_names}
    for k in range(J):
        elas_data[product_names[k]] = [f"{cross_elasticity_matrix[j, k]:.3f}" for j in range(J)]
    df_elas = pd.DataFrame(elas_data)
    report.add_table("tables/elasticity-matrix.csv",
                      "Price elasticity matrix",
                      df_elas,
                      description="Rows are products whose shares change. Columns are products whose prices change. "
                      "Off-diagonal entries repeat within each column because substitution is proportional to rival shares.")

    report.add_takeaway(
        "Plain logit turns utility coefficients into shares, elasticities, and "
        "removal counterfactuals. That simplicity imposes IIA. After one product "
        "disappears, remaining buyers are reassigned by existing shares, not "
        "measured product closeness."
    )

    report.add_references([
        "McFadden, D. (1974). Conditional Logit Analysis of Qualitative Choice Behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics*. Academic Press.",
        "Train, K. (2009). *Discrete Choice Methods with Simulation*. Cambridge University Press, 2nd edition.",
    ])

    report.write("README.md")
    print(f"\nGenerated: README.md + {len(report._figures)} figures + {len(report._tables)} tables")


if __name__ == "__main__":
    main()
