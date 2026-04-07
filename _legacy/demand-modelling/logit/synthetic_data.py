"""
Synthetic Data Generation for Simple Logit Model
================================================
Generates a fake cereal market with known parameters to demonstrate
the logit demand estimation pipeline.

Products:
- Choco-Bombs (high sugar, medium price)
- Fiber-Bran (low sugar, high price)
- Store-Frosted (high sugar, low price)
- Outside Good (consumers who buy toast/nothing)
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# TRUE PARAMETERS (What we're trying to recover)
# =============================================================================
TRUE_ALPHA = 1.5          # Price sensitivity (negative effect on utility)
TRUE_BETA_SUGAR = 0.3     # Taste for sugar (positive effect)
TRUE_BETA_CONST = 1.0     # Base utility constant


def generate_product_data(n_markets: int = 10) -> pd.DataFrame:
    """
    Generate synthetic market data for J products across T markets.

    The data mimics a cereal market where:
    - Sugar content varies by product but is constant across markets
    - Prices vary across markets (due to cost shifters)
    - Unobserved quality (xi) is product-specific

    Parameters
    ----------
    n_markets : int
        Number of markets (e.g., different cities/time periods)

    Returns
    -------
    pd.DataFrame
        Panel data with columns: market_id, product_id, product_name,
        price, sugar, xi, firm_id, cost_shifter
    """

    # Product characteristics (fixed across markets)
    products = {
        'product_id': [1, 2, 3],
        'product_name': ['Choco-Bombs', 'Fiber-Bran', 'Store-Frosted'],
        'sugar': [10.0, 1.0, 8.0],           # Sugar content (1-10 scale)
        'xi': [0.5, 0.2, -0.3],              # Unobserved quality
        'firm_id': [1, 2, 3],                # Each product owned by different firm
        'marginal_cost_base': [1.5, 2.5, 1.0]  # Base marginal costs
    }

    # Create panel data
    rows = []
    for t in range(n_markets):
        # Cost shifters vary by market (e.g., regional input costs)
        cost_shock = np.random.normal(0, 0.3)

        for j in range(len(products['product_id'])):
            # Marginal cost = base + market shock
            mc = products['marginal_cost_base'][j] + cost_shock + np.random.normal(0, 0.1)

            # Price is set with some markup over marginal cost
            # In reality this comes from supply equilibrium, but we'll simplify
            markup = np.random.uniform(0.3, 0.8)
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
                'cost_shifter': cost_shock  # This will be our instrument
            })

    return pd.DataFrame(rows)


def compute_true_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market shares using the TRUE parameters.

    This simulates what we would observe in the real world:
    s_j = exp(delta_j) / (1 + sum_k exp(delta_k))

    where delta_j = beta_0 + beta_sugar * sugar - alpha * price + xi
    """

    df = df.copy()

    # Compute mean utility delta
    df['delta'] = (TRUE_BETA_CONST
                   + TRUE_BETA_SUGAR * df['sugar']
                   - TRUE_ALPHA * df['price']
                   + df['xi'])

    # Compute shares within each market
    shares = []
    for market_id, market_df in df.groupby('market_id'):
        # exp(delta) for each product
        exp_delta = np.exp(market_df['delta'].values)

        # Denominator includes outside good (exp(0) = 1)
        denominator = 1 + exp_delta.sum()

        # Market shares
        market_shares = exp_delta / denominator

        # Outside good share
        outside_share = 1 / denominator

        for idx, share in zip(market_df.index, market_shares):
            shares.append({
                'index': idx,
                'share': share,
                'outside_share': outside_share
            })

    shares_df = pd.DataFrame(shares).set_index('index')
    df['share'] = shares_df['share']
    df['outside_share'] = shares_df['outside_share']

    return df


def generate_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate instrumental variables for price.

    Price is endogenous because xi (unobserved quality) affects both
    consumer demand and firm pricing decisions.

    Instruments must be:
    1. Correlated with price (relevance)
    2. Uncorrelated with xi (validity)

    We use:
    - Cost shifters (affects price through supply, not demand)
    - BLP instruments: sum of characteristics of OTHER products
    """

    df = df.copy()

    # Instrument 1: Cost shifter (already in data)
    # This is a valid instrument because it affects price but not xi

    # Instrument 2: Sum of rival sugar content
    # (More sugar from rivals -> more competition -> lower price)
    rival_sugar = []
    for _, row in df.iterrows():
        market_df = df[df['market_id'] == row['market_id']]
        other_products = market_df[market_df['product_id'] != row['product_id']]
        rival_sugar.append(other_products['sugar'].sum())
    df['rival_sugar_sum'] = rival_sugar

    # Instrument 3: Number of products (market structure)
    df['num_products'] = df.groupby('market_id')['product_id'].transform('count')

    return df


def create_estimation_dataset() -> pd.DataFrame:
    """
    Create the full dataset ready for estimation.

    Returns a DataFrame with:
    - Observed data: shares, prices, characteristics
    - Instruments for price
    - The LHS variable: ln(s_j) - ln(s_0)
    """

    # Generate base data
    df = generate_product_data(n_markets=20)

    # Compute shares using true model
    df = compute_true_shares(df)

    # Add instruments
    df = generate_instruments(df)

    # Create the dependent variable for logit estimation
    # Y = ln(s_j) - ln(s_0) = delta_j
    df['ln_share_ratio'] = np.log(df['share']) - np.log(df['outside_share'])

    return df


if __name__ == '__main__':
    # Generate and preview the data
    df = create_estimation_dataset()

    print("=" * 60)
    print("SYNTHETIC CEREAL MARKET DATA")
    print("=" * 60)
    print(f"\nTrue Parameters:")
    print(f"  Alpha (price sensitivity): {TRUE_ALPHA}")
    print(f"  Beta (sugar taste):        {TRUE_BETA_SUGAR}")
    print(f"  Beta (constant):           {TRUE_BETA_CONST}")

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of markets: {df['market_id'].nunique()}")
    print(f"Products per market: {df.groupby('market_id').size().iloc[0]}")

    print("\nSample data (first market):")
    print(df[df['market_id'] == 0][['product_name', 'price', 'sugar', 'share', 'xi']].to_string(index=False))

    print("\nAverage shares across markets:")
    print(df.groupby('product_name')['share'].mean().to_string())
