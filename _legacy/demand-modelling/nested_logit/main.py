"""
Nested Logit Demand Estimation - Full Pipeline
==============================================
This script demonstrates the nested logit demand estimation workflow:

1. Generate synthetic cereal market data with nest structure
2. Estimate demand using IV/2SLS (instruments for price AND within-share)
3. Compute elasticities (showing how nested logit breaks IIA)
4. Compare with simple logit to demonstrate the improvement
5. Generate visualizations

Run this script:
    python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nested_logit.synthetic_data import (
    create_estimation_dataset,
    TRUE_ALPHA, TRUE_BETA_SUGAR, TRUE_BETA_CONST, TRUE_SIGMA
)
from nested_logit.nested_logit_model import (
    compute_total_shares, compute_nested_elasticities,
    compute_simple_logit_elasticities, estimate_nested_logit_demand,
    compute_nested_markups
)
from nested_logit.visualizations import (
    plot_nested_elasticity_heatmap, plot_substitution_comparison,
    plot_demand_curves_by_nest, plot_nesting_parameter_effect,
    plot_estimation_comparison_nested
)

# Also import simple logit for comparison
from logit.logit_model import (
    estimate_logit_demand, compute_ownership_matrix,
    compute_markups as compute_logit_markups
)

OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("NESTED LOGIT DEMAND ESTIMATION")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Generate Data
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: GENERATING SYNTHETIC DATA WITH NESTS")
    print("=" * 50)

    df = create_estimation_dataset()

    print(f"\nDataset: {df.shape[0]} observations, {df['market_id'].nunique()} markets")

    print("\nTrue Parameters:")
    print(f"  Alpha (price sensitivity): {TRUE_ALPHA}")
    print(f"  Beta_sugar:                {TRUE_BETA_SUGAR}")
    print(f"  Beta_const:                {TRUE_BETA_CONST}")
    print(f"  Sigma (nesting):           {TRUE_SIGMA}")

    print("\nNest Structure:")
    market0 = df[df['market_id'] == 0]
    for nest_id, nest_df in market0.groupby('nest_id'):
        nest_name = nest_df['nest_name'].iloc[0]
        products = nest_df['product_name'].tolist()
        print(f"  Nest {nest_id} ({nest_name}): {products}")

    print("\nSample data (Market 0):")
    cols = ['product_name', 'nest_name', 'price', 'sugar', 'share', 'within_nest_share']
    print(market0[cols].to_string(index=False))

    # =========================================================================
    # STEP 2: Estimate Nested Logit
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: ESTIMATING NESTED LOGIT DEMAND")
    print("=" * 50)

    # --- Simple Logit (for comparison) ---
    print("\n--- Simple Logit Estimation (ignores nesting) ---")

    # Prepare data for simple logit
    df['ln_share_ratio_simple'] = np.log(df['share']) - np.log(df['outside_share'])

    # Simple 2SLS estimation
    Y = df['ln_share_ratio_simple'].values
    X_exog = df[['sugar']].values
    X_endog = df['price'].values
    Z = df[['cost_shifter', 'rival_sugar_sum']].values

    from logit.logit_model import estimate_2sls
    logit_results = estimate_2sls(Y, X_exog, X_endog, Z)
    logit_alpha = -logit_results['coefficients'][2]
    logit_beta_sugar = logit_results['coefficients'][1]
    logit_beta_const = logit_results['coefficients'][0]

    print(f"Alpha (Logit):      {logit_alpha:.4f}  (True: {TRUE_ALPHA})")
    print(f"Beta_sugar (Logit): {logit_beta_sugar:.4f}  (True: {TRUE_BETA_SUGAR})")

    # --- Nested Logit ---
    print("\n--- Nested Logit Estimation ---")
    nested_results = estimate_nested_logit_demand(df)

    print(f"Alpha (Nested):      {nested_results['alpha']:.4f}  (True: {TRUE_ALPHA})")
    print(f"Beta_sugar (Nested): {nested_results['beta_sugar']:.4f}  (True: {TRUE_BETA_SUGAR})")
    print(f"Beta_const (Nested): {nested_results['beta_const']:.4f}  (True: {TRUE_BETA_CONST})")
    print(f"Sigma (Nested):      {nested_results['sigma']:.4f}  (True: {TRUE_SIGMA})")

    print(f"\nKey Insight: We recovered sigma = {nested_results['sigma']:.3f}")
    print("This captures the fact that products in the same nest are closer substitutes!")

    # =========================================================================
    # STEP 3: Compute Elasticities
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: COMPUTING ELASTICITIES - THE KEY DIFFERENCE")
    print("=" * 50)

    # Use first market for calculations
    market0 = df[df['market_id'] == 0].reset_index(drop=True)
    product_names = market0['product_name'].tolist()
    prices = market0['price'].values
    shares = market0['share'].values
    within_shares = market0['within_nest_share'].values
    nest_ids = market0['nest_id'].values

    # Simple Logit elasticities
    logit_eta = compute_simple_logit_elasticities(logit_alpha, prices, shares)

    # Nested Logit elasticities
    nested_eta = compute_nested_elasticities(
        nested_results['alpha'], nested_results['sigma'],
        prices, shares, within_shares, nest_ids
    )

    print("\n--- Simple Logit Elasticity Matrix ---")
    print("(Shows IIA problem: cross-elasticities depend only on market share)")
    print("-" * 60)
    header = "              " + "  ".join([f"{n[:12]:>12}" for n in product_names])
    print(header)
    for i, name in enumerate(product_names):
        row = f"{name[:12]:>12}  " + "  ".join([f"{logit_eta[i,j]:>12.3f}"
                                                 for j in range(len(product_names))])
        print(row)

    print("\n--- Nested Logit Elasticity Matrix ---")
    print("(Breaks IIA: same-nest cross-elasticities are HIGHER)")
    print("-" * 60)
    print(header)
    for i, name in enumerate(product_names):
        row = f"{name[:12]:>12}  " + "  ".join([f"{nested_eta[i,j]:>12.3f}"
                                                 for j in range(len(product_names))])
        print(row)

    # Highlight the key difference
    print("\n" + "=" * 50)
    print("THE KEY INSIGHT: How customers switch when Choco-Bombs raises price")
    print("=" * 50)

    choco_idx = product_names.index('Choco-Bombs')

    print("\n| Substitute For  | Simple Logit | Nested Logit | Difference |")
    print("|-----------------|--------------|--------------|------------|")

    for i, name in enumerate(product_names):
        if i != choco_idx:
            logit_val = logit_eta[i, choco_idx]
            nested_val = nested_eta[i, choco_idx]
            diff = nested_val - logit_val
            same_nest = "SAME NEST" if nest_ids[i] == nest_ids[choco_idx] else "diff nest"
            print(f"| {name:<15} | {logit_val:>12.4f} | {nested_val:>12.4f} | {diff:>+10.4f} | {same_nest}")

    print("""
KEY TAKEAWAY:
- Simple Logit: If Choco-Bombs raises price, customers switch to Fiber-Bran
  and Store-Frosted proportionally to their market shares (IIA problem)

- Nested Logit: Customers PRIMARILY switch to Store-Frosted (same nest = Sugary)
  because they're closer substitutes. Much less switching to Fiber-Bran.

This is realistic! Sugary cereal buyers don't suddenly want Fiber-Bran!
    """)

    # =========================================================================
    # STEP 4: Diversion Ratios
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: DIVERSION RATIOS")
    print("=" * 50)

    print("\nIf Choco-Bombs raises price, what % of lost sales go to each rival?")

    # Compute diversion ratios from elasticities
    choco_own_logit = abs(logit_eta[choco_idx, choco_idx])
    choco_own_nested = abs(nested_eta[choco_idx, choco_idx])

    print("\n| Diversion To    | Simple Logit | Nested Logit |")
    print("|-----------------|--------------|--------------|")

    for i, name in enumerate(product_names):
        if i != choco_idx:
            # Diversion = cross-elasticity / |own-elasticity|
            div_logit = logit_eta[i, choco_idx] / choco_own_logit * 100
            div_nested = nested_eta[i, choco_idx] / choco_own_nested * 100
            print(f"| {name:<15} | {div_logit:>10.1f}% | {div_nested:>10.1f}% |")

    # =========================================================================
    # STEP 5: Generate Visualizations
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 50)

    # 1. Nested Elasticity Heatmap
    fig1 = plot_nested_elasticity_heatmap(
        nested_eta, product_names, nest_ids,
        title="Price Elasticity Matrix (Nested Logit)",
        save_path=OUTPUT_DIR / "nested_elasticity_heatmap.png"
    )

    # 2. Substitution Comparison
    fig2 = plot_substitution_comparison(
        logit_eta, nested_eta, product_names, nest_ids,
        reference_product=0,  # Choco-Bombs
        save_path=OUTPUT_DIR / "nested_substitution_comparison.png"
    )

    # 3. Demand Curves by Nest
    products_for_curves = market0[['product_name', 'price', 'sugar', 'nest_id', 'nest_name']].copy()
    fig3 = plot_demand_curves_by_nest(
        nested_results['alpha'], nested_results['sigma'],
        nested_results['beta_sugar'], nested_results['beta_const'],
        products_for_curves,
        save_path=OUTPUT_DIR / "nested_demand_curves.png"
    )

    # 4. Nesting Parameter Effect
    fig4 = plot_nesting_parameter_effect(
        save_path=OUTPUT_DIR / "nested_sigma_effect.png"
    )

    # 5. Estimation Comparison
    true_params = {
        'alpha': TRUE_ALPHA, 'beta_sugar': TRUE_BETA_SUGAR,
        'beta_const': TRUE_BETA_CONST, 'sigma': TRUE_SIGMA
    }
    logit_params = {
        'alpha': logit_alpha, 'beta_sugar': logit_beta_sugar,
        'beta_const': logit_beta_const, 'sigma': 0  # Logit has no sigma
    }
    nested_params = {
        'alpha': nested_results['alpha'],
        'beta_sugar': nested_results['beta_sugar'],
        'beta_const': nested_results['beta_const'],
        'sigma': nested_results['sigma']
    }
    fig5 = plot_estimation_comparison_nested(
        true_params, logit_params, nested_params,
        save_path=OUTPUT_DIR / "nested_estimation_comparison.png"
    )

    print("\nAll figures saved to:", OUTPUT_DIR)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("""
    KEY TAKEAWAYS:

    1. NESTED LOGIT BREAKS IIA:
       - Products in the SAME nest have HIGHER cross-price elasticities
       - Products in DIFFERENT nests have LOWER cross-elasticities
       - Simple Logit treats all products equally (unrealistic!)

    2. THE NESTING PARAMETER (sigma):
       - sigma = 0: Collapses to Simple Logit (IIA holds)
       - sigma -> 1: Perfect substitutes within nest
       - We estimated sigma = {:.3f} (true = {:.3f})

    3. NEW ENDOGENEITY:
       - ln(s_{{j|g}}) is endogenous (contains xi)
       - Need additional instruments (# products in nest, rival characteristics)

    4. REALISTIC SUBSTITUTION:
       - If sugary Choco-Bombs raises price...
       - Nested Logit: Customers mostly switch to other sugary cereals
       - Simple Logit: Customers switch proportionally to ALL products

    NEXT STEP: BLP (allows heterogeneous preferences across consumers)
    """.format(nested_results['sigma'], TRUE_SIGMA))

    plt.show()


if __name__ == '__main__':
    main()
