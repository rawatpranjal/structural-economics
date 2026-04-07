"""
Synthetic Data Generation for Nested Logit Model
=================================================
Generates a fake cereal market with products organized into nests
(groups of similar products).

Nests:
- Nest 1 (Sugary): Choco-Bombs, Store-Frosted
- Nest 2 (Healthy): Fiber-Bran, Granola-Crunch
- Outside Good (Nest 0)

The nesting structure captures that consumers substitute MORE
within a nest than across nests.
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# TRUE PARAMETERS (What we're trying to recover)
# =============================================================================
TRUE_ALPHA = 1.5          # Price sensitivity
TRUE_BETA_SUGAR = 0.3     # Taste for sugar
TRUE_BETA_CONST = 1.0     # Base utility constant
TRUE_SIGMA = 0.7          # Nesting parameter (0 = logit, approaching 1 = perfect substitutes within nest)


def generate_product_data(n_markets: int = 20) -> pd.DataFrame:
    """
    Generate synthetic market data with nest assignments.

    Products are grouped into nests:
    - Sugary cereals (high substitution among sweet options)
    - Healthy cereals (high substitution among healthy options)

    Parameters
    ----------
    n_markets : int
        Number of markets

    Returns
    -------
    pd.DataFrame
        Panel data with nest assignments
    """

    # Product characteristics with nest assignments
    products = {
        'product_id': [1, 2, 3, 4],
        'product_name': ['Choco-Bombs', 'Store-Frosted', 'Fiber-Bran', 'Granola-Crunch'],
        'sugar': [10.0, 8.0, 1.0, 2.0],           # Sugar content (1-10)
        'xi': [0.5, -0.1, 0.3, 0.1],              # Unobserved quality
        'firm_id': [1, 2, 3, 4],                  # Each owned by different firm
        'nest_id': [1, 1, 2, 2],                  # 1=Sugary, 2=Healthy
        'nest_name': ['Sugary', 'Sugary', 'Healthy', 'Healthy'],
        'marginal_cost_base': [1.5, 1.0, 2.5, 2.0]
    }

    # Create panel data
    rows = []
    for t in range(n_markets):
        # Cost shock varies by market
        cost_shock = np.random.normal(0, 0.3)

        for j in range(len(products['product_id'])):
            mc = products['marginal_cost_base'][j] + cost_shock + np.random.normal(0, 0.1)
            markup = np.random.uniform(0.3, 0.8)
            price = mc * (1 + markup)

            rows.append({
                'market_id': t,
                'product_id': products['product_id'][j],
                'product_name': products['product_name'][j],
                'sugar': products['sugar'][j],
                'xi': products['xi'][j],
                'firm_id': products['firm_id'][j],
                'nest_id': products['nest_id'][j],
                'nest_name': products['nest_name'][j],
                'price': price,
                'marginal_cost': mc,
                'cost_shifter': cost_shock
            })

    return pd.DataFrame(rows)


def compute_true_shares_nested(df: pd.DataFrame, sigma: float = TRUE_SIGMA) -> pd.DataFrame:
    """
    Compute market shares using the Nested Logit formula.

    The share of product j in nest g is:
        s_j = s_{j|g} * s_g

    Where:
        s_{j|g} = exp(delta_j / (1-sigma)) / D_g  (within-nest share)
        D_g = sum_k exp(delta_k / (1-sigma))      (inclusive value)
        s_g = D_g^(1-sigma) / sum_h D_h^(1-sigma) (nest share)

    Parameters
    ----------
    df : pd.DataFrame
        Product data with nest assignments
    sigma : float
        Nesting parameter (0 to 1)

    Returns
    -------
    pd.DataFrame
        Data with computed shares
    """
    df = df.copy()

    # Compute mean utility
    df['delta'] = (TRUE_BETA_CONST
                   + TRUE_BETA_SUGAR * df['sugar']
                   - TRUE_ALPHA * df['price']
                   + df['xi'])

    shares_list = []

    for market_id, market_df in df.groupby('market_id'):
        market_results = []

        # Compute inclusive value D_g for each nest
        nest_D = {}
        for nest_id, nest_df in market_df.groupby('nest_id'):
            D_g = np.exp(nest_df['delta'].values / (1 - sigma)).sum()
            nest_D[nest_id] = D_g

        # Compute nest shares s_g
        # Include outside good: D_0 = 1 (exp(0/(1-sigma)) = 1)
        D_sum = 1 + sum(D ** (1 - sigma) for D in nest_D.values())  # 1 is outside good
        outside_share = 1 / D_sum

        nest_shares = {nest_id: (D ** (1 - sigma)) / D_sum
                       for nest_id, D in nest_D.items()}

        # Compute within-nest shares and total shares
        for idx, row in market_df.iterrows():
            nest_id = row['nest_id']
            D_g = nest_D[nest_id]

            # Within-nest share
            s_within = np.exp(row['delta'] / (1 - sigma)) / D_g

            # Nest share
            s_nest = nest_shares[nest_id]

            # Total share
            s_total = s_within * s_nest

            market_results.append({
                'index': idx,
                'share': s_total,
                'within_nest_share': s_within,
                'nest_share': s_nest,
                'outside_share': outside_share,
                'inclusive_value': D_g
            })

        shares_list.extend(market_results)

    shares_df = pd.DataFrame(shares_list).set_index('index')
    for col in shares_df.columns:
        df[col] = shares_df[col]

    return df


def generate_instruments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate instruments for Nested Logit estimation.

    For nested logit, we need instruments for:
    1. Price (endogenous due to xi)
    2. Within-nest share ln(s_{j|g}) (endogenous because it contains xi)

    Instruments:
    - Cost shifters (for price)
    - Number of products in nest (for within-nest share)
    - Sum of characteristics of OTHER products in SAME nest
    """
    df = df.copy()

    # Instrument 1: Cost shifter (already in data)

    # Instrument 2: Sum of rival sugar content
    rival_sugar = []
    for _, row in df.iterrows():
        market_df = df[df['market_id'] == row['market_id']]
        other_products = market_df[market_df['product_id'] != row['product_id']]
        rival_sugar.append(other_products['sugar'].sum())
    df['rival_sugar_sum'] = rival_sugar

    # Instrument 3: Number of products in the same nest
    nest_count = []
    for _, row in df.iterrows():
        market_df = df[df['market_id'] == row['market_id']]
        same_nest = market_df[market_df['nest_id'] == row['nest_id']]
        nest_count.append(len(same_nest))
    df['num_in_nest'] = nest_count

    # Instrument 4: Sum of sugar of OTHER products in SAME nest (BLP-style)
    same_nest_rival_sugar = []
    for _, row in df.iterrows():
        market_df = df[df['market_id'] == row['market_id']]
        same_nest_others = market_df[(market_df['nest_id'] == row['nest_id']) &
                                      (market_df['product_id'] != row['product_id'])]
        same_nest_rival_sugar.append(same_nest_others['sugar'].sum())
    df['same_nest_rival_sugar'] = same_nest_rival_sugar

    return df


def create_estimation_dataset() -> pd.DataFrame:
    """
    Create the full dataset ready for nested logit estimation.

    The estimation equation is:
        ln(s_j) - ln(s_0) = x*beta - alpha*p + sigma*ln(s_{j|g}) + xi

    Returns
    -------
    pd.DataFrame
        Dataset ready for estimation
    """
    df = generate_product_data(n_markets=30)
    df = compute_true_shares_nested(df)
    df = generate_instruments(df)

    # Dependent variable
    df['ln_share_ratio'] = np.log(df['share']) - np.log(df['outside_share'])

    # Additional regressor for nested logit: ln(within-nest share)
    df['ln_within_share'] = np.log(df['within_nest_share'])

    return df


if __name__ == '__main__':
    df = create_estimation_dataset()

    print("=" * 60)
    print("SYNTHETIC NESTED CEREAL MARKET DATA")
    print("=" * 60)

    print(f"\nTrue Parameters:")
    print(f"  Alpha (price sensitivity): {TRUE_ALPHA}")
    print(f"  Beta (sugar taste):        {TRUE_BETA_SUGAR}")
    print(f"  Beta (constant):           {TRUE_BETA_CONST}")
    print(f"  Sigma (nesting):           {TRUE_SIGMA}")

    print(f"\nDataset: {df.shape[0]} observations, {df['market_id'].nunique()} markets")
    print(f"Products per market: {df.groupby('market_id').size().iloc[0]}")

    print("\nNest Structure:")
    for nest_id, nest_df in df[df['market_id'] == 0].groupby('nest_id'):
        print(f"  Nest {nest_id} ({nest_df['nest_name'].iloc[0]}): {nest_df['product_name'].tolist()}")

    print("\nSample data (Market 0):")
    cols = ['product_name', 'nest_name', 'price', 'sugar', 'share', 'within_nest_share']
    print(df[df['market_id'] == 0][cols].to_string(index=False))

    print(f"\nAverage within-nest shares by product:")
    print(df.groupby('product_name')['within_nest_share'].mean().to_string())
