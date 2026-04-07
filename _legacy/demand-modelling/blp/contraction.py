"""
BLP Contraction Mapping (Inner Loop)
====================================
The most famous part of the BLP paper.

The contraction mapping inverts market shares to find the mean utility
vector (delta) that makes predicted shares match observed shares.

This is necessary because with random coefficients, we can't analytically
invert shares like in simple/nested logit.
"""

import numpy as np
from typing import Callable


def contraction_mapping(observed_shares: np.ndarray,
                        predict_shares_fn: Callable,
                        initial_delta: np.ndarray = None,
                        tol: float = 1e-12,
                        max_iter: int = 1000,
                        verbose: bool = False) -> tuple:
    """
    BLP Contraction Mapping to invert shares and find delta.

    The iteration:
        delta^{h+1} = delta^h + ln(s_observed) - ln(s_predicted(delta^h))

    This converges to the unique delta that makes predicted shares
    equal observed shares (for a given sigma).

    Parameters
    ----------
    observed_shares : np.ndarray
        Observed market shares (J,)
    predict_shares_fn : Callable
        Function that takes delta and returns predicted shares
        Signature: predict_shares_fn(delta) -> shares
    initial_delta : np.ndarray, optional
        Starting guess for delta. If None, uses log(s) - log(s0)
    tol : float
        Convergence tolerance (on norm of delta change)
    max_iter : int
        Maximum iterations
    verbose : bool
        Print convergence info

    Returns
    -------
    tuple
        (delta, converged, n_iter, norm_history)
    """
    J = len(observed_shares)

    # Initialize delta
    if initial_delta is None:
        # Standard initialization: simple logit inversion
        outside_share = 1 - observed_shares.sum()
        initial_delta = np.log(observed_shares) - np.log(outside_share)

    delta = initial_delta.copy()
    norm_history = []

    for iteration in range(max_iter):
        # Predict shares with current delta
        predicted_shares = predict_shares_fn(delta)

        # Avoid log(0)
        predicted_shares = np.maximum(predicted_shares, 1e-300)

        # Contraction update
        delta_new = delta + np.log(observed_shares) - np.log(predicted_shares)

        # Check convergence
        norm_change = np.linalg.norm(delta_new - delta)
        norm_history.append(norm_change)

        if verbose and iteration % 100 == 0:
            print(f"  Iteration {iteration}: ||delta_change|| = {norm_change:.2e}")

        if norm_change < tol:
            if verbose:
                print(f"  Converged in {iteration + 1} iterations")
            return delta_new, True, iteration + 1, norm_history

        delta = delta_new

    if verbose:
        print(f"  WARNING: Did not converge in {max_iter} iterations")
        print(f"  Final ||delta_change|| = {norm_history[-1]:.2e}")

    return delta, False, max_iter, norm_history


def contraction_mapping_nfxp(observed_shares: np.ndarray,
                             sugar: np.ndarray,
                             prices: np.ndarray,
                             draws: dict,
                             sigma_alpha: float,
                             sigma_sugar: float,
                             tol: float = 1e-12,
                             max_iter: int = 1000) -> tuple:
    """
    Full NFXP-style contraction mapping for BLP.

    Given sigma (non-linear parameters), finds delta that matches shares.

    Parameters
    ----------
    observed_shares : np.ndarray
        Observed market shares (J,)
    sugar : np.ndarray
        Sugar content (J,)
    prices : np.ndarray
        Prices (J,)
    draws : dict
        Simulated consumer draws
    sigma_alpha : float
        Std dev of price sensitivity
    sigma_sugar : float
        Std dev of sugar taste
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    tuple
        (delta, converged, n_iter)
    """
    from blp.synthetic_data import compute_blp_shares

    J = len(observed_shares)

    # Initialize delta
    outside_share = 1 - observed_shares.sum()
    delta = np.log(observed_shares) - np.log(outside_share)

    for iteration in range(max_iter):
        # Predict shares
        predicted_shares = compute_blp_shares(delta, sugar, prices, draws,
                                              sigma_alpha, sigma_sugar)

        # Avoid numerical issues
        predicted_shares = np.maximum(predicted_shares, 1e-300)

        # Contraction update
        delta_new = delta + np.log(observed_shares) - np.log(predicted_shares)

        # Check convergence
        norm_change = np.linalg.norm(delta_new - delta)

        if norm_change < tol:
            return delta_new, True, iteration + 1

        delta = delta_new

    return delta, False, max_iter


def test_contraction_convergence():
    """
    Test that the contraction mapping converges and is accurate.
    """
    print("Testing Contraction Mapping Convergence")
    print("=" * 50)

    # Generate test data
    from blp.synthetic_data import (
        generate_product_data, draw_simulated_consumers, compute_blp_shares,
        TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR, TRUE_ALPHA_MEAN, TRUE_BETA_SUGAR_MEAN, TRUE_BETA_CONST
    )

    df = generate_product_data(n_markets=1)
    draws = draw_simulated_consumers(n_draws=500)

    # True delta (what we're trying to recover)
    delta_true = (TRUE_BETA_CONST
                  + TRUE_BETA_SUGAR_MEAN * df['sugar'].values
                  - TRUE_ALPHA_MEAN * df['price'].values
                  + df['xi'].values)

    sugar = df['sugar'].values
    prices = df['price'].values

    # Compute observed shares from true delta
    observed_shares = compute_blp_shares(delta_true, sugar, prices, draws,
                                         TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)

    print(f"\nTrue delta: {delta_true}")
    print(f"Observed shares: {observed_shares}")

    # Run contraction mapping
    delta_recovered, converged, n_iter = contraction_mapping_nfxp(
        observed_shares, sugar, prices, draws,
        TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR,
        tol=1e-12, max_iter=1000
    )

    print(f"\nRecovered delta: {delta_recovered}")
    print(f"Converged: {converged} in {n_iter} iterations")
    print(f"Max error: {np.abs(delta_true - delta_recovered).max():.2e}")

    # Verify shares match
    recovered_shares = compute_blp_shares(delta_recovered, sugar, prices, draws,
                                          TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)
    print(f"Max share error: {np.abs(observed_shares - recovered_shares).max():.2e}")


if __name__ == '__main__':
    test_contraction_convergence()
