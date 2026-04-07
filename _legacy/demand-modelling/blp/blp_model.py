"""
BLP Model Core Functions
========================
Main functions for:
1. Computing BLP market shares with random coefficients
2. Computing elasticities and derivatives
3. Computing diversion ratios
"""

import numpy as np
from typing import Dict, Tuple


def compute_individual_utilities(delta: np.ndarray, sugar: np.ndarray,
                                 prices: np.ndarray, draws: dict,
                                 sigma_alpha: float, sigma_sugar: float) -> np.ndarray:
    """
    Compute utility for each simulated consumer and product.

    U_ij = delta_j + mu_ij
    where mu_ij = sigma_alpha * v_i * (-p_j) + sigma_sugar * v_i * sugar_j

    Parameters
    ----------
    delta : np.ndarray
        Mean utilities (J,)
    sugar : np.ndarray
        Sugar content (J,)
    prices : np.ndarray
        Prices (J,)
    draws : dict
        Consumer draws (v_alpha, v_sugar)
    sigma_alpha : float
        Std dev of price sensitivity
    sigma_sugar : float
        Std dev of sugar taste

    Returns
    -------
    np.ndarray
        Utilities (n_draws x J)
    """
    n_draws = len(draws['v_alpha'])
    J = len(delta)

    utilities = np.zeros((n_draws, J))

    for j in range(J):
        utilities[:, j] = delta[j]
        utilities[:, j] += sigma_alpha * draws['v_alpha'] * (-prices[j])
        utilities[:, j] += sigma_sugar * draws['v_sugar'] * sugar[j]

    return utilities


def compute_individual_choice_probs(utilities: np.ndarray) -> np.ndarray:
    """
    Compute choice probabilities for each consumer and product.

    P_ij = exp(U_ij) / (1 + sum_k exp(U_ik))

    Parameters
    ----------
    utilities : np.ndarray
        Utilities (n_draws x J)

    Returns
    -------
    np.ndarray
        Choice probabilities (n_draws x J)
    """
    exp_util = np.exp(utilities)
    denominators = 1 + exp_util.sum(axis=1)  # 1 = outside good
    probs = exp_util / denominators[:, np.newaxis]
    return probs


def compute_blp_shares(delta: np.ndarray, sugar: np.ndarray,
                       prices: np.ndarray, draws: dict,
                       sigma_alpha: float, sigma_sugar: float) -> np.ndarray:
    """
    Compute aggregate market shares by integrating over consumers.

    s_j = E[P_ij] = (1/ns) * sum_i P_ij

    Parameters
    ----------
    delta : np.ndarray
        Mean utilities (J,)
    sugar : np.ndarray
        Sugar content (J,)
    prices : np.ndarray
        Prices (J,)
    draws : dict
        Consumer draws
    sigma_alpha : float
        Std dev of price sensitivity
    sigma_sugar : float
        Std dev of sugar taste

    Returns
    -------
    np.ndarray
        Market shares (J,)
    """
    utilities = compute_individual_utilities(delta, sugar, prices, draws,
                                             sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utilities)
    shares = probs.mean(axis=0)
    return shares


def compute_blp_elasticities(delta: np.ndarray, sugar: np.ndarray,
                             prices: np.ndarray, draws: dict,
                             alpha_mean: float, sigma_alpha: float,
                             sigma_sugar: float) -> np.ndarray:
    """
    Compute the JxJ elasticity matrix for BLP.

    The key innovation: elasticities now depend on product SIMILARITY
    in the characteristic space, not just market shares!

    eta_jk = (p_k / s_j) * ds_j/dp_k

    For BLP:
    ds_j/dp_k = -(1/ns) * sum_i alpha_i * P_ij * (1{j=k} - P_ik)

    Parameters
    ----------
    delta : np.ndarray
        Mean utilities (J,)
    sugar : np.ndarray
        Sugar content (J,)
    prices : np.ndarray
        Prices (J,)
    draws : dict
        Consumer draws
    alpha_mean : float
        Mean price sensitivity
    sigma_alpha : float
        Std dev of price sensitivity
    sigma_sugar : float
        Std dev of sugar taste

    Returns
    -------
    np.ndarray
        Elasticity matrix (J x J)
    """
    J = len(delta)
    n_draws = len(draws['v_alpha'])

    # Compute utilities and choice probabilities
    utilities = compute_individual_utilities(delta, sugar, prices, draws,
                                             sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utilities)

    # Individual price sensitivities
    alpha_i = alpha_mean + sigma_alpha * draws['v_alpha']

    # Market shares
    shares = probs.mean(axis=0)

    # Compute derivatives ds_j/dp_k
    deriv_matrix = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if j == k:
                # Own derivative: -E[alpha_i * P_ij * (1 - P_ij)]
                deriv = -(alpha_i * probs[:, j] * (1 - probs[:, j])).mean()
            else:
                # Cross derivative: E[alpha_i * P_ij * P_ik]
                deriv = (alpha_i * probs[:, j] * probs[:, k]).mean()

            deriv_matrix[j, k] = deriv

    # Convert to elasticities
    elasticity_matrix = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            elasticity_matrix[j, k] = deriv_matrix[j, k] * prices[k] / shares[j]

    return elasticity_matrix


def compute_blp_share_derivatives(delta: np.ndarray, sugar: np.ndarray,
                                  prices: np.ndarray, draws: dict,
                                  alpha_mean: float, sigma_alpha: float,
                                  sigma_sugar: float) -> np.ndarray:
    """
    Compute the matrix of share derivatives (for supply side).

    ds_j/dp_k

    Parameters
    ----------
    (same as compute_blp_elasticities)

    Returns
    -------
    np.ndarray
        Derivative matrix (J x J)
    """
    J = len(delta)

    utilities = compute_individual_utilities(delta, sugar, prices, draws,
                                             sigma_alpha, sigma_sugar)
    probs = compute_individual_choice_probs(utilities)
    alpha_i = alpha_mean + sigma_alpha * draws['v_alpha']

    deriv_matrix = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if j == k:
                deriv = -(alpha_i * probs[:, j] * (1 - probs[:, j])).mean()
            else:
                deriv = (alpha_i * probs[:, j] * probs[:, k]).mean()
            deriv_matrix[j, k] = deriv

    return deriv_matrix


def compute_diversion_ratios(elasticity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute diversion ratios from elasticity matrix.

    D_jk = (ds_k/dp_j) / |ds_j/dp_j|
         = "If j loses 100 units, how many go to k?"

    This is KEY for merger analysis!

    Parameters
    ----------
    elasticity_matrix : np.ndarray
        JxJ elasticity matrix

    Returns
    -------
    np.ndarray
        Diversion ratio matrix (J x J)
    """
    J = elasticity_matrix.shape[0]
    diversion = np.zeros((J, J))

    for j in range(J):
        own_effect = abs(elasticity_matrix[j, j])
        for k in range(J):
            if j != k:
                # Cross-elasticity / own-elasticity
                diversion[j, k] = elasticity_matrix[k, j] / own_effect

    return diversion


def compute_simple_logit_elasticities(alpha: float, prices: np.ndarray,
                                       shares: np.ndarray) -> np.ndarray:
    """
    Compute simple logit elasticities for comparison.

    Shows the IIA problem clearly.
    """
    J = len(prices)
    elasticity_matrix = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if j == k:
                elasticity_matrix[j, k] = -alpha * prices[j] * (1 - shares[j])
            else:
                elasticity_matrix[j, k] = alpha * prices[k] * shares[k]

    return elasticity_matrix


if __name__ == '__main__':
    print("Testing BLP Model Functions")
    print("=" * 50)

    from blp.synthetic_data import (
        generate_product_data, draw_simulated_consumers, compute_blp_shares,
        TRUE_ALPHA_MEAN, TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR,
        TRUE_BETA_CONST, TRUE_BETA_SUGAR_MEAN
    )

    # Generate test data
    df = generate_product_data(n_markets=1)
    draws = draw_simulated_consumers(n_draws=1000)

    # Compute delta
    delta = (TRUE_BETA_CONST
             + TRUE_BETA_SUGAR_MEAN * df['sugar'].values
             - TRUE_ALPHA_MEAN * df['price'].values
             + df['xi'].values)

    sugar = df['sugar'].values
    prices = df['price'].values

    # Compute shares
    shares = compute_blp_shares(delta, sugar, prices, draws,
                                TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)

    print(f"\nProducts: {df['product_name'].tolist()}")
    print(f"Shares: {shares}")

    # Compute elasticities
    blp_eta = compute_blp_elasticities(
        delta, sugar, prices, draws,
        TRUE_ALPHA_MEAN, TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR
    )

    logit_eta = compute_simple_logit_elasticities(TRUE_ALPHA_MEAN, prices, shares)

    print("\nBLP Elasticity Matrix:")
    print(blp_eta)

    print("\nSimple Logit Elasticity Matrix (for comparison):")
    print(logit_eta)

    print("\n" + "=" * 50)
    print("KEY OBSERVATION:")
    print("=" * 50)
    print("In BLP, cross-elasticities between SIMILAR products (high sugar)")
    print("are HIGHER than between dissimilar products.")
    print("This is because consumers who like sugar are more likely to")
    print("switch between sugary products!")
