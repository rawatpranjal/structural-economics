"""
BLP Merger Simulation and Counterfactuals
=========================================
The real payoff of BLP: answering "What if?" questions.

Key applications:
1. Merger simulation (price effects)
2. Consumer surplus calculation
3. New product introduction
"""

import numpy as np
from typing import Dict, Tuple


def compute_ownership_matrix(firm_ids: np.ndarray) -> np.ndarray:
    """
    Create ownership matrix.

    O_jk = 1 if j and k owned by same firm, else 0
    """
    J = len(firm_ids)
    ownership = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            if firm_ids[j] == firm_ids[k]:
                ownership[j, k] = 1
    return ownership


def update_ownership_for_merger(firm_ids: np.ndarray,
                                acquiring_firm: int,
                                acquired_firm: int) -> np.ndarray:
    """
    Update firm IDs to reflect a merger.

    Parameters
    ----------
    firm_ids : np.ndarray
        Current firm ownership
    acquiring_firm : int
        ID of acquiring firm
    acquired_firm : int
        ID of firm being acquired

    Returns
    -------
    np.ndarray
        Updated firm IDs (acquired firm's products now owned by acquirer)
    """
    new_firm_ids = firm_ids.copy()
    new_firm_ids[firm_ids == acquired_firm] = acquiring_firm
    return new_firm_ids


def compute_omega_blp(deriv_matrix: np.ndarray,
                      ownership: np.ndarray) -> np.ndarray:
    """
    Compute Omega matrix for BLP supply side.

    Omega_jk = -ds_k/dp_j if same firm, else 0

    Parameters
    ----------
    deriv_matrix : np.ndarray
        Share derivative matrix (J x J)
    ownership : np.ndarray
        Ownership matrix (J x J)

    Returns
    -------
    np.ndarray
        Omega matrix (J x J)
    """
    return -deriv_matrix * ownership


def compute_markups_blp(shares: np.ndarray, deriv_matrix: np.ndarray,
                        ownership: np.ndarray) -> np.ndarray:
    """
    Compute markups: p - c = Omega^{-1} * s

    Parameters
    ----------
    shares : np.ndarray
        Market shares (J,)
    deriv_matrix : np.ndarray
        Share derivative matrix (J x J)
    ownership : np.ndarray
        Ownership matrix (J x J)

    Returns
    -------
    np.ndarray
        Markups (J,)
    """
    omega = compute_omega_blp(deriv_matrix, ownership)
    markups = np.linalg.solve(omega, shares)
    return markups


def recover_marginal_costs(prices: np.ndarray, markups: np.ndarray) -> np.ndarray:
    """Recover marginal costs: c = p - markup"""
    return prices - markups


def solve_equilibrium_prices(marginal_costs: np.ndarray,
                             ownership: np.ndarray,
                             sugar: np.ndarray,
                             draws: dict,
                             alpha_mean: float,
                             sigma_alpha: float,
                             sigma_sugar: float,
                             initial_delta: np.ndarray,
                             tol: float = 1e-6,
                             max_iter: int = 100) -> Tuple[np.ndarray, bool]:
    """
    Solve for new equilibrium prices after ownership change.

    Uses fixed-point iteration:
        p* = c + Omega(p*)^{-1} * s(p*)

    This is complex because shares and derivatives depend on prices!

    Parameters
    ----------
    marginal_costs : np.ndarray
        Marginal costs (J,)
    ownership : np.ndarray
        New ownership matrix (J x J)
    sugar : np.ndarray
        Sugar content (J,)
    draws : dict
        Consumer draws
    alpha_mean : float
        Mean price sensitivity
    sigma_alpha : float
        Std dev of price sensitivity
    sigma_sugar : float
        Std dev of sugar taste
    initial_delta : np.ndarray
        Initial mean utilities
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    Tuple
        (new_prices, converged)
    """
    from blp.blp_model import compute_blp_shares, compute_blp_share_derivatives

    J = len(marginal_costs)

    # Initialize with original prices
    # delta = const + beta*sugar - alpha*p + xi
    # p_init = (delta_init - xi) / (-alpha) approximately
    prices = marginal_costs * 1.5  # Start with some markup

    for iteration in range(max_iter):
        # Compute delta at current prices
        # We need to adjust delta for the new prices
        # delta(p) = delta_0 + alpha_mean * (p_0 - p)
        # This is an approximation; full solution would use contraction

        delta = initial_delta.copy()

        # Compute shares at current prices
        shares = compute_blp_shares(delta, sugar, prices, draws,
                                    sigma_alpha, sigma_sugar)

        # Compute derivatives
        deriv = compute_blp_share_derivatives(
            delta, sugar, prices, draws,
            alpha_mean, sigma_alpha, sigma_sugar
        )

        # Compute markups
        omega = compute_omega_blp(deriv, ownership)
        try:
            markups = np.linalg.solve(omega, shares)
        except np.linalg.LinAlgError:
            return prices, False

        # Update prices
        new_prices = marginal_costs + markups

        # Check convergence
        if np.max(np.abs(new_prices - prices)) < tol:
            return new_prices, True

        # Damped update for stability
        prices = 0.5 * prices + 0.5 * new_prices

    return prices, False


def simulate_merger(pre_prices: np.ndarray,
                    pre_shares: np.ndarray,
                    marginal_costs: np.ndarray,
                    pre_firm_ids: np.ndarray,
                    acquiring_firm: int,
                    acquired_firm: int,
                    sugar: np.ndarray,
                    draws: dict,
                    alpha_mean: float,
                    sigma_alpha: float,
                    sigma_sugar: float,
                    delta: np.ndarray) -> Dict:
    """
    Full merger simulation.

    1. Update ownership matrix
    2. Solve for new equilibrium prices
    3. Compute price changes

    Parameters
    ----------
    pre_prices : np.ndarray
        Pre-merger prices
    pre_shares : np.ndarray
        Pre-merger shares
    marginal_costs : np.ndarray
        Marginal costs (assumed unchanged)
    pre_firm_ids : np.ndarray
        Pre-merger firm ownership
    acquiring_firm : int
        Acquiring firm ID
    acquired_firm : int
        Acquired firm ID
    sugar, draws, alpha_mean, sigma_alpha, sigma_sugar, delta :
        Model parameters

    Returns
    -------
    Dict
        Merger simulation results
    """
    # Update ownership
    post_firm_ids = update_ownership_for_merger(pre_firm_ids, acquiring_firm, acquired_firm)
    post_ownership = compute_ownership_matrix(post_firm_ids)
    pre_ownership = compute_ownership_matrix(pre_firm_ids)

    # Solve for post-merger prices
    post_prices, converged = solve_equilibrium_prices(
        marginal_costs, post_ownership, sugar, draws,
        alpha_mean, sigma_alpha, sigma_sugar, delta
    )

    # Compute post-merger shares
    from blp.blp_model import compute_blp_shares

    # Update delta for new prices (approximation)
    delta_post = delta + alpha_mean * (pre_prices - post_prices)
    post_shares = compute_blp_shares(delta_post, sugar, post_prices, draws,
                                     sigma_alpha, sigma_sugar)

    # Price changes
    price_changes = post_prices - pre_prices
    price_change_pct = (price_changes / pre_prices) * 100

    return {
        'post_prices': post_prices,
        'post_shares': post_shares,
        'price_changes': price_changes,
        'price_change_pct': price_change_pct,
        'converged': converged,
        'post_firm_ids': post_firm_ids
    }


def compute_consumer_surplus(utilities: np.ndarray,
                             alpha_i: np.ndarray) -> float:
    """
    Compute consumer surplus using the log-sum formula.

    CS_i = (1/alpha_i) * ln(1 + sum_j exp(U_ij))

    Parameters
    ----------
    utilities : np.ndarray
        Utilities (n_draws x J)
    alpha_i : np.ndarray
        Individual price sensitivities (n_draws,)

    Returns
    -------
    float
        Average consumer surplus
    """
    # Inclusive value (log-sum)
    inclusive_value = np.log(1 + np.exp(utilities).sum(axis=1))

    # Consumer surplus
    cs_i = inclusive_value / alpha_i

    return cs_i.mean()


def compute_welfare_change(pre_utilities: np.ndarray,
                           post_utilities: np.ndarray,
                           alpha_i: np.ndarray,
                           market_size: float = 1000000) -> Dict:
    """
    Compute change in consumer welfare from policy/merger.

    Parameters
    ----------
    pre_utilities : np.ndarray
        Pre-change utilities (n_draws x J)
    post_utilities : np.ndarray
        Post-change utilities (n_draws x J)
    alpha_i : np.ndarray
        Individual price sensitivities (n_draws,)
    market_size : float
        Total number of consumers in market

    Returns
    -------
    Dict
        Welfare analysis results
    """
    cs_pre = compute_consumer_surplus(pre_utilities, alpha_i)
    cs_post = compute_consumer_surplus(post_utilities, alpha_i)

    delta_cs = cs_post - cs_pre
    total_welfare_change = delta_cs * market_size

    return {
        'cs_pre': cs_pre,
        'cs_post': cs_post,
        'delta_cs': delta_cs,
        'total_welfare_change': total_welfare_change
    }


if __name__ == '__main__':
    print("Testing Merger Simulation")
    print("=" * 50)

    from blp.synthetic_data import (
        generate_product_data, draw_simulated_consumers,
        TRUE_ALPHA_MEAN, TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR,
        TRUE_BETA_CONST, TRUE_BETA_SUGAR_MEAN
    )
    from blp.blp_model import compute_blp_shares, compute_blp_share_derivatives

    # Generate data
    df = generate_product_data(n_markets=1)
    draws = draw_simulated_consumers(n_draws=500)

    # Compute delta
    delta = (TRUE_BETA_CONST
             + TRUE_BETA_SUGAR_MEAN * df['sugar'].values
             - TRUE_ALPHA_MEAN * df['price'].values
             + df['xi'].values)

    prices = df['price'].values
    sugar = df['sugar'].values
    firm_ids = df['firm_id'].values
    mc = df['marginal_cost'].values

    # Pre-merger shares
    shares = compute_blp_shares(delta, sugar, prices, draws,
                                TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)

    print(f"\nProducts: {df['product_name'].tolist()}")
    print(f"Pre-merger prices: {prices}")
    print(f"Pre-merger shares: {shares}")
    print(f"Firm ownership: {firm_ids}")

    # Simulate merger: Firm 1 (Choco-Bombs) acquires Firm 3 (Store-Frosted)
    print("\n" + "=" * 50)
    print("MERGER SIMULATION: Choco-Bombs acquires Store-Frosted")
    print("=" * 50)

    results = simulate_merger(
        prices, shares, mc, firm_ids,
        acquiring_firm=1, acquired_firm=3,
        sugar=sugar, draws=draws,
        alpha_mean=TRUE_ALPHA_MEAN,
        sigma_alpha=TRUE_SIGMA_ALPHA,
        sigma_sugar=TRUE_SIGMA_SUGAR,
        delta=delta
    )

    print(f"\nPost-merger prices: {results['post_prices']}")
    print(f"Price changes ($): {results['price_changes']}")
    print(f"Price changes (%): {results['price_change_pct']}")
    print(f"Converged: {results['converged']}")
