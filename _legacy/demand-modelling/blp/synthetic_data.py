"""
Synthetic Data Generation for BLP Model
========================================
Generates a fake cereal market with HETEROGENEOUS consumer preferences.

The key innovation: Different consumers have different tastes!
- Kids: Love sugar, don't care about price
- Parents: Hate sugar, very price sensitive

This heterogeneity creates realistic substitution patterns that
neither Simple Logit nor Nested Logit can capture.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# =============================================================================
# TRUE PARAMETERS
# =============================================================================
# Mean preferences (the "average" consumer)
TRUE_ALPHA_MEAN = 2.0        # Average price sensitivity
TRUE_BETA_SUGAR_MEAN = 0.0   # Average sugar taste (neutral on average)
TRUE_BETA_CONST = 1.5        # Base utility

# Standard deviations (heterogeneity across consumers)
TRUE_SIGMA_ALPHA = 1.5       # Variance in price sensitivity
TRUE_SIGMA_SUGAR = 2.0       # Variance in sugar taste (POLARIZING!)

# Note: With sigma_sugar = 2.0, some people LOVE sugar (+2SD = +4.0 boost)
# and others HATE sugar (-2SD = -4.0 penalty). This is the BLP insight!


def generate_product_data(n_markets: int = 20) -> pd.DataFrame:
    """
    Generate product-market level data.

    Products (like the cereal example):
    - Choco-Bombs: High sugar, medium price (loved by kids)
    - Fiber-Bran: Low sugar, high price (loved by health-conscious)
    - Store-Frosted: High sugar, low price (budget option)
    """
    products = {
        'product_id': [1, 2, 3],
        'product_name': ['Choco-Bombs', 'Fiber-Bran', 'Store-Frosted'],
        'sugar': [10.0, 1.0, 8.0],
        'xi': [0.5, 0.3, -0.2],  # Unobserved quality
        'firm_id': [1, 2, 3],
        'marginal_cost_base': [1.8, 2.8, 1.2]
    }

    rows = []
    for t in range(n_markets):
        cost_shock = np.random.normal(0, 0.2)

        for j in range(len(products['product_id'])):
            mc = products['marginal_cost_base'][j] + cost_shock + np.random.normal(0, 0.1)
            markup = np.random.uniform(0.4, 0.9)
            price = mc * (1 + markup)

            rows.append({
                'market_id': t,
                'product_id': products['product_id'][j],
                'product_name': products['product_name'][j],
                'sugar': products['sugar'][j],
                'xi': products['xi'][j],
                'firm_id': products['firm_id'][j],
                'price': price,
                'marginal_cost': mc,
                'cost_shifter': cost_shock
            })

    return pd.DataFrame(rows)


def draw_simulated_consumers(n_draws: int = 500) -> dict:
    """
    Draw "fake" consumers from the population distribution.

    BLP uses simulation to integrate over consumer heterogeneity.
    We draw random taste shocks (v) from standard normal.

    Parameters
    ----------
    n_draws : int
        Number of simulated consumers

    Returns
    -------
    dict
        Random draws for each random coefficient
    """
    draws = {
        'v_alpha': np.random.normal(0, 1, n_draws),   # Shocks to price sensitivity
        'v_sugar': np.random.normal(0, 1, n_draws),  # Shocks to sugar taste
    }
    return draws


def compute_individual_utility(delta: np.ndarray, sugar: np.ndarray,
                               prices: np.ndarray, draws: dict,
                               sigma_alpha: float, sigma_sugar: float) -> np.ndarray:
    """
    Compute utility for each simulated consumer and product.

    U_ij = delta_j + mu_ij
    where mu_ij = sigma_alpha * v_i^alpha * (-p_j) + sigma_sugar * v_i^sugar * sugar_j

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
        Utilities (n_draws x J)
    """
    n_draws = len(draws['v_alpha'])
    J = len(delta)

    utilities = np.zeros((n_draws, J))

    for j in range(J):
        # Mean utility
        utilities[:, j] = delta[j]

        # Add heterogeneity
        # Price sensitivity: people with high v_alpha are MORE price sensitive
        utilities[:, j] += sigma_alpha * draws['v_alpha'] * (-prices[j])

        # Sugar taste: people with high v_sugar LOVE sugar
        utilities[:, j] += sigma_sugar * draws['v_sugar'] * sugar[j]

    return utilities


def compute_blp_shares(delta: np.ndarray, sugar: np.ndarray,
                       prices: np.ndarray, draws: dict,
                       sigma_alpha: float, sigma_sugar: float) -> np.ndarray:
    """
    Compute market shares by integrating over consumer distribution.

    s_j = (1/ns) * sum_i [ exp(U_ij) / (1 + sum_k exp(U_ik)) ]

    This is the key BLP insight: we simulate individual choices
    and average to get aggregate shares.

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
    utilities = compute_individual_utility(delta, sugar, prices, draws,
                                           sigma_alpha, sigma_sugar)

    n_draws, J = utilities.shape

    # Compute individual choice probabilities
    exp_util = np.exp(utilities)
    denominators = 1 + exp_util.sum(axis=1)  # 1 = outside good

    individual_probs = exp_util / denominators[:, np.newaxis]

    # Average across consumers to get market shares
    shares = individual_probs.mean(axis=0)

    return shares


def compute_true_shares_blp(df: pd.DataFrame, n_draws: int = 1000) -> pd.DataFrame:
    """
    Compute market shares using the true BLP model with heterogeneity.
    """
    df = df.copy()

    # Compute mean utility (using true mean parameters)
    df['delta'] = (TRUE_BETA_CONST
                   + TRUE_BETA_SUGAR_MEAN * df['sugar']
                   - TRUE_ALPHA_MEAN * df['price']
                   + df['xi'])

    shares_list = []

    for market_id, market_df in df.groupby('market_id'):
        # Draw consumers for this market
        draws = draw_simulated_consumers(n_draws)

        delta = market_df['delta'].values
        sugar = market_df['sugar'].values
        prices = market_df['price'].values

        shares = compute_blp_shares(delta, sugar, prices, draws,
                                    TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)

        outside_share = 1 - shares.sum()

        for idx, (_, row) in enumerate(market_df.iterrows()):
            shares_list.append({
                'index': row.name,
                'share': shares[idx],
                'outside_share': outside_share
            })

    shares_df = pd.DataFrame(shares_list).set_index('index')
    df['share'] = shares_df['share']
    df['outside_share'] = shares_df['outside_share']

    return df


def generate_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """Generate instruments for BLP estimation."""
    df = df.copy()

    # Instrument 1: Cost shifter
    # (already in data)

    # Instrument 2: Sum of rival characteristics (BLP instruments)
    rival_sugar = []
    for _, row in df.iterrows():
        market_df = df[df['market_id'] == row['market_id']]
        other_products = market_df[market_df['product_id'] != row['product_id']]
        rival_sugar.append(other_products['sugar'].sum())
    df['rival_sugar_sum'] = rival_sugar

    # Instrument 3: Number of products
    df['num_products'] = df.groupby('market_id')['product_id'].transform('count')

    # Instrument 4: Squared characteristics (Hausman-style)
    df['sugar_squared'] = df['sugar'] ** 2

    return df


def create_estimation_dataset() -> pd.DataFrame:
    """
    Create full dataset ready for BLP estimation.
    """
    df = generate_product_data(n_markets=30)
    df = compute_true_shares_blp(df, n_draws=1000)
    df = generate_instruments(df)

    # Dependent variable (for the linear step)
    df['ln_share_ratio'] = np.log(df['share']) - np.log(df['outside_share'])

    return df


def get_consumer_type_analysis(n_draws: int = 5000) -> pd.DataFrame:
    """
    Analyze the simulated consumer types to understand heterogeneity.

    Classifies consumers as:
    - "Kids" (high sugar taste, low price sensitivity)
    - "Parents" (low sugar taste, high price sensitivity)
    """
    draws = draw_simulated_consumers(n_draws)

    # Compute actual taste parameters for each consumer
    alpha_i = TRUE_ALPHA_MEAN + TRUE_SIGMA_ALPHA * draws['v_alpha']
    beta_sugar_i = TRUE_BETA_SUGAR_MEAN + TRUE_SIGMA_SUGAR * draws['v_sugar']

    consumer_df = pd.DataFrame({
        'alpha': alpha_i,
        'beta_sugar': beta_sugar_i
    })

    # Classify consumers
    consumer_df['type'] = 'Average'
    consumer_df.loc[(consumer_df['beta_sugar'] > 1) & (consumer_df['alpha'] < 1.5), 'type'] = 'Kids (sugar-lover)'
    consumer_df.loc[(consumer_df['beta_sugar'] < -1) & (consumer_df['alpha'] > 2.5), 'type'] = 'Parents (health-conscious)'

    return consumer_df


if __name__ == '__main__':
    df = create_estimation_dataset()

    print("=" * 60)
    print("SYNTHETIC BLP CEREAL MARKET DATA")
    print("=" * 60)

    print(f"\nTrue Parameters (Mean):")
    print(f"  Alpha (price sensitivity): {TRUE_ALPHA_MEAN}")
    print(f"  Beta (sugar taste):        {TRUE_BETA_SUGAR_MEAN}")
    print(f"  Beta (constant):           {TRUE_BETA_CONST}")

    print(f"\nTrue Parameters (Std Dev - HETEROGENEITY):")
    print(f"  Sigma_alpha: {TRUE_SIGMA_ALPHA}")
    print(f"  Sigma_sugar: {TRUE_SIGMA_SUGAR}")

    print(f"\nDataset: {df.shape[0]} observations, {df['market_id'].nunique()} markets")

    print("\nSample data (Market 0):")
    cols = ['product_name', 'price', 'sugar', 'share', 'xi']
    print(df[df['market_id'] == 0][cols].to_string(index=False))

    print("\n" + "=" * 50)
    print("CONSUMER HETEROGENEITY ANALYSIS")
    print("=" * 50)

    consumer_df = get_consumer_type_analysis()
    print("\nConsumer Type Distribution:")
    print(consumer_df['type'].value_counts())

    print("\nTaste Parameter Statistics:")
    print(f"  Price sensitivity (alpha): mean={consumer_df['alpha'].mean():.2f}, "
          f"std={consumer_df['alpha'].std():.2f}")
    print(f"  Sugar taste (beta_sugar):  mean={consumer_df['beta_sugar'].mean():.2f}, "
          f"std={consumer_df['beta_sugar'].std():.2f}")
