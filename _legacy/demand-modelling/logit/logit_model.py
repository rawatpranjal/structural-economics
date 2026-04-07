"""
Simple Logit Demand Model Implementation
========================================
Core functions for:
1. Computing mean utility and market shares
2. Inverting shares to recover delta (Berry inversion)
3. Computing own-price and cross-price elasticities
4. IV/2SLS estimation
5. Supply-side: markup recovery and marginal cost estimation

Reference: Berry (1994), "Estimating Discrete-Choice Models of Product Differentiation"
"""

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# DEMAND SIDE
# =============================================================================

def compute_mean_utility(X: np.ndarray, beta: np.ndarray, alpha: float,
                         prices: np.ndarray, xi: np.ndarray = None) -> np.ndarray:
    """
    Compute mean utility: delta = X*beta - alpha*p + xi

    Parameters
    ----------
    X : np.ndarray
        Product characteristics matrix (J x K), excludes price
    beta : np.ndarray
        Taste parameters (K,)
    alpha : float
        Price sensitivity (positive value, enters negatively)
    prices : np.ndarray
        Prices (J,)
    xi : np.ndarray, optional
        Unobserved quality (J,), defaults to zeros

    Returns
    -------
    np.ndarray
        Mean utility vector (J,)
    """
    if xi is None:
        xi = np.zeros(len(prices))

    delta = X @ beta - alpha * prices + xi
    return delta


def compute_shares(delta: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute market shares from mean utility.

    s_j = exp(delta_j) / (1 + sum_k exp(delta_k))

    The "1" in the denominator is the outside good (normalized delta_0 = 0).

    Parameters
    ----------
    delta : np.ndarray
        Mean utility vector (J,)

    Returns
    -------
    tuple
        (inside_shares, outside_share)
        inside_shares: np.ndarray (J,)
        outside_share: float
    """
    exp_delta = np.exp(delta)
    denominator = 1 + exp_delta.sum()

    inside_shares = exp_delta / denominator
    outside_share = 1 / denominator

    return inside_shares, outside_share


def invert_shares(shares: np.ndarray, outside_share: float) -> np.ndarray:
    """
    Berry inversion: recover mean utility from observed shares.

    ln(s_j) - ln(s_0) = delta_j

    This is the KEY insight that allows linear IV estimation.

    Parameters
    ----------
    shares : np.ndarray
        Observed market shares (J,)
    outside_share : float
        Share of the outside good

    Returns
    -------
    np.ndarray
        Recovered mean utility vector (J,)
    """
    delta = np.log(shares) - np.log(outside_share)
    return delta


def compute_elasticities(alpha: float, prices: np.ndarray,
                         shares: np.ndarray) -> np.ndarray:
    """
    Compute the JxJ elasticity matrix.

    Own-price elasticity: eta_jj = -alpha * p_j * (1 - s_j)
    Cross-price elasticity: eta_jk = alpha * p_k * s_k

    THE IIA PROBLEM: Cross-elasticities depend ONLY on the rival's
    price and share, not on how similar products are!

    Parameters
    ----------
    alpha : float
        Estimated price sensitivity (positive value)
    prices : np.ndarray
        Prices (J,)
    shares : np.ndarray
        Market shares (J,)

    Returns
    -------
    np.ndarray
        Elasticity matrix (J x J)
        Entry (j, k) = percent change in s_j from 1% price increase in k
    """
    J = len(prices)
    elasticity_matrix = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if j == k:
                # Own-price elasticity (negative)
                elasticity_matrix[j, k] = -alpha * prices[j] * (1 - shares[j])
            else:
                # Cross-price elasticity (positive)
                elasticity_matrix[j, k] = alpha * prices[k] * shares[k]

    return elasticity_matrix


def compute_share_derivatives(alpha: float, shares: np.ndarray) -> np.ndarray:
    """
    Compute the matrix of share derivatives with respect to price.

    ds_j/dp_j = -alpha * s_j * (1 - s_j)
    ds_j/dp_k = alpha * s_j * s_k

    Used for supply-side markup calculations.

    Parameters
    ----------
    alpha : float
        Price sensitivity
    shares : np.ndarray
        Market shares (J,)

    Returns
    -------
    np.ndarray
        Derivative matrix (J x J)
    """
    J = len(shares)
    deriv_matrix = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if j == k:
                deriv_matrix[j, k] = -alpha * shares[j] * (1 - shares[j])
            else:
                deriv_matrix[j, k] = alpha * shares[j] * shares[k]

    return deriv_matrix


# =============================================================================
# ESTIMATION (2SLS / IV)
# =============================================================================

def estimate_ols(Y: np.ndarray, X: np.ndarray) -> dict:
    """
    Simple OLS estimation (biased due to endogeneity).

    Used as a baseline to show why IV is needed.
    """
    # Add constant
    X_const = np.column_stack([np.ones(len(Y)), X])

    # OLS: beta = (X'X)^-1 X'Y
    XtX_inv = np.linalg.inv(X_const.T @ X_const)
    beta_hat = XtX_inv @ X_const.T @ Y

    # Residuals and standard errors
    residuals = Y - X_const @ beta_hat
    sigma2 = (residuals @ residuals) / (len(Y) - X_const.shape[1])
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))

    return {
        'coefficients': beta_hat,
        'std_errors': se,
        'residuals': residuals,
        'r_squared': 1 - (residuals @ residuals) / ((Y - Y.mean()) @ (Y - Y.mean()))
    }


def estimate_2sls(Y: np.ndarray, X_exog: np.ndarray, X_endog: np.ndarray,
                  Z: np.ndarray) -> dict:
    """
    Two-Stage Least Squares (2SLS) estimation.

    Stage 1: Regress endogenous variables on instruments
    Stage 2: Regress Y on predicted endogenous variables

    Parameters
    ----------
    Y : np.ndarray
        Dependent variable: ln(s_j) - ln(s_0) = delta
    X_exog : np.ndarray
        Exogenous regressors (characteristics like sugar)
    X_endog : np.ndarray
        Endogenous regressors (price)
    Z : np.ndarray
        Instruments for endogenous variables

    Returns
    -------
    dict
        Estimation results including coefficients, standard errors, etc.
    """
    n = len(Y)

    # Combine all exogenous variables (used in both stages)
    # Instruments include: exogenous X + excluded instruments Z
    W = np.column_stack([np.ones(n), X_exog, Z])  # First stage regressors
    X = np.column_stack([np.ones(n), X_exog, X_endog])  # Second stage regressors

    # ===================
    # STAGE 1: X_endog = W * gamma + error
    # ===================
    # Project endogenous variables onto instrument space
    # Use pseudo-inverse for numerical stability
    WtW_inv = np.linalg.pinv(W.T @ W)
    P_W = W @ WtW_inv @ W.T  # Projection matrix

    # Predicted endogenous variables
    X_endog_hat = P_W @ X_endog

    # ===================
    # STAGE 2: Y = X_hat * beta + error
    # ===================
    X_hat = np.column_stack([np.ones(n), X_exog, X_endog_hat])

    # 2SLS estimator (use pseudo-inverse for stability)
    XhX_inv = np.linalg.pinv(X_hat.T @ X_hat)
    beta_hat = XhX_inv @ X_hat.T @ Y

    # Residuals (using original X, not X_hat)
    residuals = Y - X @ beta_hat

    # Standard errors (using original X for variance)
    sigma2 = (residuals @ residuals) / max(n - X.shape[1], 1)
    # Correct variance: (X'P_W X)^-1 * sigma^2
    var_beta = sigma2 * XhX_inv
    se = np.sqrt(np.abs(np.diag(var_beta)))

    # First stage F-statistic (test instrument strength)
    # Regress price on instruments, check F-stat
    gamma_hat = WtW_inv @ W.T @ X_endog
    X_endog_resid = X_endog - W @ gamma_hat
    ss_resid = X_endog_resid @ X_endog_resid
    ss_total = (X_endog - X_endog.mean()) @ (X_endog - X_endog.mean())
    r2_first_stage = 1 - ss_resid / max(ss_total, 1e-10)
    denom = max((1 - r2_first_stage) / max(n - W.shape[1], 1), 1e-10)
    f_stat = (r2_first_stage / max(Z.shape[1], 1)) / denom

    return {
        'coefficients': beta_hat,
        'std_errors': se,
        'residuals': residuals,
        't_stats': beta_hat / se,
        'first_stage_f': f_stat,
        'n_obs': n
    }


def estimate_logit_demand(df: pd.DataFrame) -> dict:
    """
    Full logit demand estimation pipeline.

    Estimates: ln(s_j) - ln(s_0) = beta_0 + beta_sugar * sugar - alpha * price + xi

    Uses 2SLS with cost shifters as instruments for price.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns: ln_share_ratio, price, sugar, cost_shifter, rival_sugar_sum

    Returns
    -------
    dict
        Estimation results
    """
    Y = df['ln_share_ratio'].values
    X_exog = df[['sugar']].values
    X_endog = df['price'].values
    Z = df[['cost_shifter', 'rival_sugar_sum']].values

    # Run 2SLS
    results = estimate_2sls(Y, X_exog, X_endog, Z)

    # Parse coefficients
    # beta_hat = [constant, sugar, -alpha]
    results['alpha'] = -results['coefficients'][2]  # Price enters negatively
    results['beta_sugar'] = results['coefficients'][1]
    results['beta_const'] = results['coefficients'][0]

    return results


# =============================================================================
# SUPPLY SIDE
# =============================================================================

def compute_ownership_matrix(firm_ids: np.ndarray) -> np.ndarray:
    """
    Create ownership matrix indicating which products are owned by same firm.

    O_jk = 1 if products j and k are owned by the same firm, else 0

    Parameters
    ----------
    firm_ids : np.ndarray
        Firm identifier for each product (J,)

    Returns
    -------
    np.ndarray
        Ownership matrix (J x J)
    """
    J = len(firm_ids)
    ownership = np.zeros((J, J))

    for j in range(J):
        for k in range(J):
            if firm_ids[j] == firm_ids[k]:
                ownership[j, k] = 1

    return ownership


def compute_omega_matrix(alpha: float, shares: np.ndarray,
                         ownership: np.ndarray) -> np.ndarray:
    """
    Compute the Omega matrix for multi-product firm markup calculation.

    Omega_jk = -ds_k/dp_j if same firm, else 0

    This captures how firm f internalizes cross-product effects.

    Parameters
    ----------
    alpha : float
        Price sensitivity
    shares : np.ndarray
        Market shares (J,)
    ownership : np.ndarray
        Ownership matrix (J x J)

    Returns
    -------
    np.ndarray
        Omega matrix (J x J)
    """
    deriv = compute_share_derivatives(alpha, shares)

    # Omega = -deriv * ownership (element-wise)
    # Note: We negate because the FOC has s + (p-c) * ds/dp = 0
    omega = -deriv * ownership

    return omega


def compute_markups(alpha: float, shares: np.ndarray,
                    ownership: np.ndarray) -> np.ndarray:
    """
    Recover markups using the supply-side first order conditions.

    For multi-product firms:
        p - c = Omega^(-1) * s

    For single-product firms (special case):
        p - c = 1 / (alpha * (1 - s))

    Parameters
    ----------
    alpha : float
        Price sensitivity
    shares : np.ndarray
        Market shares (J,)
    ownership : np.ndarray
        Ownership matrix (J x J)

    Returns
    -------
    np.ndarray
        Markups (J,)
    """
    omega = compute_omega_matrix(alpha, shares, ownership)

    # Solve: Omega * (p - c) = s
    # => (p - c) = Omega^(-1) * s
    markups = np.linalg.solve(omega, shares)

    return markups


def recover_marginal_costs(prices: np.ndarray, markups: np.ndarray) -> np.ndarray:
    """
    Recover marginal costs from prices and markups.

    c = p - markup

    This is the key insight: we can back out costs without
    ever seeing the firm's accounting books!

    Parameters
    ----------
    prices : np.ndarray
        Observed prices (J,)
    markups : np.ndarray
        Estimated markups (J,)

    Returns
    -------
    np.ndarray
        Marginal costs (J,)
    """
    return prices - markups


def compute_single_product_markups(alpha: float, shares: np.ndarray) -> np.ndarray:
    """
    Compute markups assuming single-product firms (no portfolio effects).

    p - c = 1 / (alpha * (1 - s))

    This is the simplified formula from basic IO theory.

    Parameters
    ----------
    alpha : float
        Price sensitivity
    shares : np.ndarray
        Market shares (J,)

    Returns
    -------
    np.ndarray
        Markups (J,)
    """
    return 1 / (alpha * (1 - shares))


if __name__ == '__main__':
    # Test the functions with simple example
    print("Testing Logit Model Functions")
    print("=" * 50)

    # Example: 3 products
    delta = np.array([1.0, 0.5, -0.2])

    shares, s0 = compute_shares(delta)
    print(f"\nMean utilities: {delta}")
    print(f"Shares: {shares}")
    print(f"Outside share: {s0:.4f}")
    print(f"Sum of all shares: {shares.sum() + s0:.4f}")

    # Inversion test
    delta_recovered = invert_shares(shares, s0)
    print(f"\nRecovered delta: {delta_recovered}")
    print(f"Original delta:  {delta}")
    print(f"Difference: {np.abs(delta - delta_recovered).max():.2e}")

    # Elasticities
    prices = np.array([3.0, 5.0, 2.0])
    alpha = 1.5
    eta = compute_elasticities(alpha, prices, shares)
    print(f"\nElasticity matrix:\n{eta}")
    print(f"\nOwn-price elasticities: {np.diag(eta)}")
    print("Note: Cross-elasticities are IDENTICAL across columns (IIA problem)")
