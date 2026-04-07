"""
BLP GMM Estimation (Outer Loop)
===============================
The outer loop searches for the non-linear parameters (sigma)
that minimize the GMM objective function.

For each candidate sigma:
1. Run contraction mapping to get delta
2. Compute structural error xi = delta - X*beta + alpha*p
3. Check if xi is orthogonal to instruments (GMM condition)
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple


def compute_linear_parameters(delta: np.ndarray, X: np.ndarray,
                              prices: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concentrate out linear parameters using IV/GMM.

    Given delta (from contraction), recover (beta, alpha) by regressing
    delta on characteristics and price using instruments.

    delta = X*beta - alpha*p + xi
    E[xi * Z] = 0  (moment condition)

    Parameters
    ----------
    delta : np.ndarray
        Mean utilities from contraction (J*T,)
    X : np.ndarray
        Characteristics matrix with constant (J*T x K)
    prices : np.ndarray
        Prices (J*T,)
    Z : np.ndarray
        Instruments (J*T x L)

    Returns
    -------
    Tuple
        (theta1, xi) where theta1 = [beta_0, beta_1, ..., -alpha]
    """
    n = len(delta)

    # Full regressor matrix: [constant, characteristics, price]
    X_full = np.column_stack([X, prices])

    # Instrument matrix (includes exogenous X and excluded instruments)
    W = np.column_stack([X, Z])

    # GMM estimator: (X'Z(Z'Z)^-1 Z'X)^-1 X'Z(Z'Z)^-1 Z'y
    ZtZ_inv = np.linalg.inv(W.T @ W)

    # Projection matrix
    PZ = W @ ZtZ_inv @ W.T

    # 2SLS estimator
    XtPZX = X_full.T @ PZ @ X_full
    XtPZy = X_full.T @ PZ @ delta

    theta1 = np.linalg.solve(XtPZX, XtPZy)

    # Compute residuals (structural error xi)
    xi = delta - X_full @ theta1

    return theta1, xi


def gmm_objective(sigma: np.ndarray, observed_shares: np.ndarray,
                  X: np.ndarray, prices: np.ndarray, sugar: np.ndarray,
                  Z: np.ndarray, draws: dict, W: np.ndarray = None) -> float:
    """
    GMM objective function: J(sigma) = xi'Z * W * Z'xi

    This is what the optimizer minimizes over sigma.

    Parameters
    ----------
    sigma : np.ndarray
        Non-linear parameters [sigma_alpha, sigma_sugar]
    observed_shares : np.ndarray
        Observed market shares (stacked across markets)
    X : np.ndarray
        Characteristics (with constant)
    prices : np.ndarray
        Prices
    sugar : np.ndarray
        Sugar content (for heterogeneity)
    Z : np.ndarray
        Instruments
    draws : dict
        Simulated consumer draws
    W : np.ndarray, optional
        GMM weighting matrix. If None, uses identity.

    Returns
    -------
    float
        GMM objective value
    """
    from blp.contraction import contraction_mapping_nfxp

    sigma_alpha, sigma_sugar = sigma

    # Ensure sigma is non-negative
    if sigma_alpha < 0 or sigma_sugar < 0:
        return 1e10

    # Run contraction to get delta
    delta, converged, n_iter = contraction_mapping_nfxp(
        observed_shares, sugar, prices, draws,
        sigma_alpha, sigma_sugar,
        tol=1e-10, max_iter=500
    )

    if not converged:
        return 1e10  # Penalize non-convergence

    # Concentrate out linear parameters
    theta1, xi = compute_linear_parameters(delta, X, prices, Z)

    # GMM objective
    # Moments: g = Z' * xi
    g = Z.T @ xi

    # Weighting matrix
    if W is None:
        W = np.eye(len(g))

    # Objective: g' W g
    obj = g.T @ W @ g

    return obj


def estimate_blp(df_stacked, draws: dict, initial_sigma: np.ndarray = None,
                 verbose: bool = True) -> dict:
    """
    Full BLP estimation.

    Parameters
    ----------
    df_stacked : pd.DataFrame
        Stacked market data
    draws : dict
        Simulated consumer draws (shared across markets for efficiency)
    initial_sigma : np.ndarray, optional
        Starting values for [sigma_alpha, sigma_sugar]
    verbose : bool
        Print optimization progress

    Returns
    -------
    dict
        Estimation results
    """
    # Extract data
    observed_shares = df_stacked['share'].values
    prices = df_stacked['price'].values
    sugar = df_stacked['sugar'].values

    # Characteristics matrix (constant + sugar)
    n = len(df_stacked)
    X = np.column_stack([np.ones(n), sugar])

    # Instruments
    Z = df_stacked[['cost_shifter', 'rival_sugar_sum', 'sugar_squared']].values

    # Initial sigma
    if initial_sigma is None:
        initial_sigma = np.array([1.0, 1.0])

    # Weighting matrix (identity for first stage)
    W = np.eye(Z.shape[1])

    # Optimization
    if verbose:
        print("Starting BLP optimization...")
        print(f"Initial sigma: {initial_sigma}")

    def objective(sigma):
        obj = gmm_objective(sigma, observed_shares, X, prices, sugar, Z, draws, W)
        if verbose:
            print(f"  sigma = {sigma}, objective = {obj:.4f}")
        return obj

    result = minimize(
        objective,
        initial_sigma,
        method='L-BFGS-B',
        bounds=[(0.01, 5.0), (0.01, 5.0)],
        options={'maxiter': 100, 'disp': False}
    )

    sigma_hat = result.x

    if verbose:
        print(f"\nOptimization finished.")
        print(f"Final sigma: {sigma_hat}")

    # Get final delta and linear parameters
    from blp.contraction import contraction_mapping_nfxp

    delta_hat, _, _ = contraction_mapping_nfxp(
        observed_shares, sugar, prices, draws,
        sigma_hat[0], sigma_hat[1],
        tol=1e-12, max_iter=1000
    )

    theta1_hat, xi_hat = compute_linear_parameters(delta_hat, X, prices, Z)

    # Parse results
    # theta1 = [beta_const, beta_sugar, -alpha]
    results = {
        'sigma_alpha': sigma_hat[0],
        'sigma_sugar': sigma_hat[1],
        'beta_const': theta1_hat[0],
        'beta_sugar': theta1_hat[1],
        'alpha': -theta1_hat[2],  # Price enters negatively
        'delta': delta_hat,
        'xi': xi_hat,
        'gmm_objective': result.fun,
        'converged': result.success
    }

    return results


if __name__ == '__main__':
    print("Testing GMM Estimation")
    print("=" * 50)

    from blp.synthetic_data import (
        create_estimation_dataset, draw_simulated_consumers,
        TRUE_ALPHA_MEAN, TRUE_BETA_SUGAR_MEAN, TRUE_BETA_CONST,
        TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR
    )

    # Generate data
    df = create_estimation_dataset()
    draws = draw_simulated_consumers(n_draws=200)  # Fewer draws for speed

    # Use one market for quick test
    market0 = df[df['market_id'] == 0].copy()

    print("\nTrue Parameters:")
    print(f"  sigma_alpha: {TRUE_SIGMA_ALPHA}")
    print(f"  sigma_sugar: {TRUE_SIGMA_SUGAR}")
    print(f"  alpha:       {TRUE_ALPHA_MEAN}")
    print(f"  beta_sugar:  {TRUE_BETA_SUGAR_MEAN}")
    print(f"  beta_const:  {TRUE_BETA_CONST}")

    # Quick estimation (would need more markets and draws for accuracy)
    print("\n(Using reduced data for demonstration)")
