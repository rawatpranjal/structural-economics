"""
Nested Logit Demand Model Implementation
========================================
Core functions for:
1. Computing within-nest and nest shares
2. Berry (1994) inversion for nested logit
3. Elasticities (own, same-nest cross, different-nest cross)
4. IV estimation with additional instrument for within-nest share

Reference: Berry (1994), "Estimating Discrete-Choice Models of Product Differentiation"
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# =============================================================================
# SHARE COMPUTATIONS
# =============================================================================

def compute_inclusive_value(delta: np.ndarray, nest_ids: np.ndarray,
                            sigma: float) -> Dict[int, float]:
    """
    Compute the inclusive value D_g for each nest.

    D_g = sum_{k in g} exp(delta_k / (1 - sigma))

    The inclusive value measures the "attractiveness" of a nest.

    Parameters
    ----------
    delta : np.ndarray
        Mean utilities (J,)
    nest_ids : np.ndarray
        Nest assignment for each product (J,)
    sigma : float
        Nesting parameter

    Returns
    -------
    Dict[int, float]
        Inclusive value for each nest
    """
    D = {}
    for nest_id in np.unique(nest_ids):
        mask = nest_ids == nest_id
        D[nest_id] = np.exp(delta[mask] / (1 - sigma)).sum()
    return D


def compute_within_nest_share(delta: np.ndarray, nest_ids: np.ndarray,
                              sigma: float) -> np.ndarray:
    """
    Compute within-nest shares s_{j|g}.

    s_{j|g} = exp(delta_j / (1-sigma)) / D_g

    This is the probability of choosing j GIVEN that you chose nest g.

    Parameters
    ----------
    delta : np.ndarray
        Mean utilities (J,)
    nest_ids : np.ndarray
        Nest assignments (J,)
    sigma : float
        Nesting parameter

    Returns
    -------
    np.ndarray
        Within-nest shares (J,)
    """
    D = compute_inclusive_value(delta, nest_ids, sigma)
    J = len(delta)
    s_within = np.zeros(J)

    for j in range(J):
        nest_id = nest_ids[j]
        s_within[j] = np.exp(delta[j] / (1 - sigma)) / D[nest_id]

    return s_within


def compute_nest_share(inclusive_values: Dict[int, float],
                       sigma: float) -> Tuple[Dict[int, float], float]:
    """
    Compute nest shares s_g.

    s_g = D_g^(1-sigma) / [1 + sum_h D_h^(1-sigma)]

    The "1" represents the outside good (nest 0).

    Parameters
    ----------
    inclusive_values : Dict[int, float]
        Inclusive value D_g for each nest
    sigma : float
        Nesting parameter

    Returns
    -------
    Tuple[Dict[int, float], float]
        (nest_shares, outside_share)
    """
    # Denominator includes outside good (D_0 = 1)
    denominator = 1 + sum(D ** (1 - sigma) for D in inclusive_values.values())

    nest_shares = {nest_id: (D ** (1 - sigma)) / denominator
                   for nest_id, D in inclusive_values.items()}
    outside_share = 1 / denominator

    return nest_shares, outside_share


def compute_total_shares(delta: np.ndarray, nest_ids: np.ndarray,
                         sigma: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute total market shares: s_j = s_{j|g} * s_g

    Parameters
    ----------
    delta : np.ndarray
        Mean utilities (J,)
    nest_ids : np.ndarray
        Nest assignments (J,)
    sigma : float
        Nesting parameter

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (total_shares, within_nest_shares, outside_share)
    """
    D = compute_inclusive_value(delta, nest_ids, sigma)
    nest_shares, outside_share = compute_nest_share(D, sigma)
    s_within = compute_within_nest_share(delta, nest_ids, sigma)

    J = len(delta)
    s_total = np.zeros(J)

    for j in range(J):
        nest_id = nest_ids[j]
        s_total[j] = s_within[j] * nest_shares[nest_id]

    return s_total, s_within, outside_share


# =============================================================================
# BERRY INVERSION
# =============================================================================

def berry_inversion_nested(shares: np.ndarray, within_shares: np.ndarray,
                           outside_share: float, sigma: float) -> np.ndarray:
    """
    Berry (1994) inversion for nested logit.

    The estimation equation is:
        ln(s_j) - ln(s_0) = delta_j + sigma * ln(s_{j|g})

    Rearranging:
        delta_j = ln(s_j) - ln(s_0) - sigma * ln(s_{j|g})

    This linearizes the nested logit for IV regression.

    Parameters
    ----------
    shares : np.ndarray
        Observed total shares (J,)
    within_shares : np.ndarray
        Within-nest shares (J,)
    outside_share : float
        Share of outside good
    sigma : float
        Nesting parameter (known or estimated)

    Returns
    -------
    np.ndarray
        Recovered mean utilities (J,)
    """
    delta = np.log(shares) - np.log(outside_share) - sigma * np.log(within_shares)
    return delta


def compute_ln_share_ratio(shares: np.ndarray, outside_share: float) -> np.ndarray:
    """
    Compute the LHS of the estimation equation.

    Y = ln(s_j) - ln(s_0)

    For nested logit, this equals:
        Y = delta + sigma * ln(s_{j|g})

    Parameters
    ----------
    shares : np.ndarray
        Total shares (J,)
    outside_share : float
        Outside good share

    Returns
    -------
    np.ndarray
        Log share ratio (J,)
    """
    return np.log(shares) - np.log(outside_share)


# =============================================================================
# ELASTICITIES
# =============================================================================

def compute_nested_elasticities(alpha: float, sigma: float,
                                prices: np.ndarray, shares: np.ndarray,
                                within_shares: np.ndarray,
                                nest_ids: np.ndarray) -> np.ndarray:
    """
    Compute the JxJ elasticity matrix for nested logit.

    THREE types of elasticities (this breaks IIA!):

    1. Own-price:
       eta_jj = -alpha * p_j * [1/(1-sigma) - (1/(1-sigma) - 1)*s_{j|g} - s_j]

    2. Cross-price (SAME nest):
       eta_jk = alpha * p_k * [(1/(1-sigma) - 1)*s_{k|g} + s_k]

    3. Cross-price (DIFFERENT nest):
       eta_jk = alpha * p_k * s_k

    The key insight: products in the same nest have HIGHER cross-elasticities
    than products in different nests!

    Parameters
    ----------
    alpha : float
        Price sensitivity
    sigma : float
        Nesting parameter
    prices : np.ndarray
        Prices (J,)
    shares : np.ndarray
        Total market shares (J,)
    within_shares : np.ndarray
        Within-nest shares (J,)
    nest_ids : np.ndarray
        Nest assignments (J,)

    Returns
    -------
    np.ndarray
        Elasticity matrix (J x J)
    """
    J = len(prices)
    elasticity_matrix = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if j == k:
                # Own-price elasticity
                term1 = 1 / (1 - sigma)
                term2 = (1 / (1 - sigma) - 1) * within_shares[j]
                term3 = shares[j]
                elasticity_matrix[j, k] = -alpha * prices[j] * (term1 - term2 - term3)

            elif nest_ids[j] == nest_ids[k]:
                # Cross-price, SAME nest (high substitution)
                term1 = (1 / (1 - sigma) - 1) * within_shares[k]
                term2 = shares[k]
                elasticity_matrix[j, k] = alpha * prices[k] * (term1 + term2)

            else:
                # Cross-price, DIFFERENT nest (low substitution)
                # Same as simple logit!
                elasticity_matrix[j, k] = alpha * prices[k] * shares[k]

    return elasticity_matrix


def compute_simple_logit_elasticities(alpha: float, prices: np.ndarray,
                                       shares: np.ndarray) -> np.ndarray:
    """
    Compute simple logit elasticities for comparison.

    This shows the IIA problem: all cross-elasticities are identical.
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


# =============================================================================
# ESTIMATION (2SLS)
# =============================================================================

def estimate_2sls_nested(Y: np.ndarray, X_exog: np.ndarray,
                         X_endog: np.ndarray, Z: np.ndarray) -> dict:
    """
    Two-Stage Least Squares for Nested Logit.

    Model: Y = X_exog * beta_exog + X_endog * beta_endog + error

    For nested logit:
    - Y = ln(s_j) - ln(s_0)
    - X_exog = [constant, sugar, ...]
    - X_endog = [price, ln(s_{j|g})]  <- BOTH endogenous!

    Parameters
    ----------
    Y : np.ndarray
        Dependent variable: ln(s_j) - ln(s_0)
    X_exog : np.ndarray
        Exogenous variables (characteristics)
    X_endog : np.ndarray
        Endogenous variables (price, ln(within_share))
    Z : np.ndarray
        Instruments for endogenous variables

    Returns
    -------
    dict
        Estimation results
    """
    n = len(Y)

    # All exogenous instruments
    W = np.column_stack([np.ones(n), X_exog, Z])
    X = np.column_stack([np.ones(n), X_exog, X_endog])

    # Stage 1: Project endogenous onto instruments (use pseudo-inverse for stability)
    WtW_inv = np.linalg.pinv(W.T @ W)
    P_W = W @ WtW_inv @ W.T

    # Predict each endogenous variable
    X_endog_hat = P_W @ X_endog

    # Stage 2: Regress Y on predicted X
    X_hat = np.column_stack([np.ones(n), X_exog, X_endog_hat])
    XhX_inv = np.linalg.pinv(X_hat.T @ X_hat)
    beta_hat = XhX_inv @ X_hat.T @ Y

    # Standard errors
    residuals = Y - X @ beta_hat
    sigma2 = (residuals @ residuals) / max(n - X.shape[1], 1)
    var_beta = sigma2 * XhX_inv
    se = np.sqrt(np.abs(np.diag(var_beta)))

    return {
        'coefficients': beta_hat,
        'std_errors': se,
        'residuals': residuals,
        't_stats': beta_hat / se,
        'n_obs': n
    }


def estimate_nested_logit_demand(df: pd.DataFrame) -> dict:
    """
    Full nested logit demand estimation.

    Estimates:
        ln(s_j) - ln(s_0) = beta_0 + beta_sugar * sugar - alpha * price
                           + sigma * ln(s_{j|g}) + xi

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with required columns

    Returns
    -------
    dict
        Estimation results with alpha, beta, sigma
    """
    Y = df['ln_share_ratio'].values
    X_exog = df[['sugar']].values
    X_endog = df[['price', 'ln_within_share']].values
    Z = df[['cost_shifter', 'rival_sugar_sum', 'num_in_nest', 'same_nest_rival_sugar']].values

    results = estimate_2sls_nested(Y, X_exog, X_endog, Z)

    # Parse coefficients: [const, sugar, -alpha, sigma]
    results['alpha'] = -results['coefficients'][2]  # Price enters negatively
    results['beta_sugar'] = results['coefficients'][1]
    results['beta_const'] = results['coefficients'][0]
    results['sigma'] = results['coefficients'][3]  # Nesting parameter

    return results


# =============================================================================
# SUPPLY SIDE (Similar to simple logit but with nested derivatives)
# =============================================================================

def compute_nested_share_derivatives(alpha: float, sigma: float,
                                     shares: np.ndarray, within_shares: np.ndarray,
                                     nest_ids: np.ndarray) -> np.ndarray:
    """
    Compute share derivatives for nested logit supply side.

    ds_j/dp_k has three cases:
    1. j = k (own derivative)
    2. j != k, same nest
    3. j != k, different nest

    Parameters
    ----------
    alpha : float
        Price sensitivity
    sigma : float
        Nesting parameter
    shares : np.ndarray
        Total shares (J,)
    within_shares : np.ndarray
        Within-nest shares (J,)
    nest_ids : np.ndarray
        Nest assignments (J,)

    Returns
    -------
    np.ndarray
        Derivative matrix (J x J)
    """
    J = len(shares)
    deriv = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if j == k:
                # Own derivative
                term1 = 1 / (1 - sigma)
                term2 = (1 / (1 - sigma) - 1) * within_shares[j]
                term3 = shares[j]
                deriv[j, k] = -alpha * shares[j] * (term1 - term2 - term3)

            elif nest_ids[j] == nest_ids[k]:
                # Same nest cross-derivative
                term1 = (1 / (1 - sigma) - 1) * within_shares[k]
                term2 = shares[k]
                deriv[j, k] = alpha * shares[j] * (term1 + term2)

            else:
                # Different nest cross-derivative
                deriv[j, k] = alpha * shares[j] * shares[k]

    return deriv


def compute_nested_markups(alpha: float, sigma: float,
                           shares: np.ndarray, within_shares: np.ndarray,
                           nest_ids: np.ndarray, ownership: np.ndarray) -> np.ndarray:
    """
    Compute markups using nested logit derivatives.

    p - c = Omega^{-1} * s

    where Omega uses nested logit derivatives.
    """
    deriv = compute_nested_share_derivatives(alpha, sigma, shares, within_shares, nest_ids)
    omega = -deriv * ownership
    markups = np.linalg.solve(omega, shares)
    return markups


if __name__ == '__main__':
    print("Testing Nested Logit Functions")
    print("=" * 50)

    # Example: 4 products in 2 nests
    delta = np.array([1.0, 0.8, 0.5, 0.3])
    nest_ids = np.array([1, 1, 2, 2])
    sigma = 0.6

    s_total, s_within, s_outside = compute_total_shares(delta, nest_ids, sigma)

    print(f"\nMean utilities: {delta}")
    print(f"Nest assignments: {nest_ids}")
    print(f"Sigma: {sigma}")
    print(f"\nTotal shares: {s_total}")
    print(f"Within-nest shares: {s_within}")
    print(f"Outside share: {s_outside:.4f}")

    # Elasticities
    prices = np.array([3.0, 2.0, 5.0, 4.0])
    alpha = 1.5

    nested_eta = compute_nested_elasticities(alpha, sigma, prices, s_total, s_within, nest_ids)
    logit_eta = compute_simple_logit_elasticities(alpha, prices, s_total)

    print("\nNested Logit Elasticity Matrix:")
    print(nested_eta)

    print("\nSimple Logit Elasticity Matrix (for comparison):")
    print(logit_eta)

    print("\nKey observation:")
    print("- Nested: Same-nest cross-elasticities are HIGHER")
    print("- Logit: All cross-elasticities in a column are identical")
