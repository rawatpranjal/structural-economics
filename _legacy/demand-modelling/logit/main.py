"""
Simple Logit Demand Estimation - Full Pipeline
==============================================
This script demonstrates the complete logit demand estimation workflow:

1. Generate synthetic cereal market data
2. Estimate demand using OLS (biased) and IV/2SLS (consistent)
3. Compute elasticities (showing the IIA problem)
4. Recover markups and marginal costs (supply side)
5. Generate visualizations

Run this script to see the full pipeline in action:
    python main.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logit.synthetic_data import (
    create_estimation_dataset,
    TRUE_ALPHA, TRUE_BETA_SUGAR, TRUE_BETA_CONST
)
from logit.logit_model import (
    compute_shares, invert_shares, compute_elasticities,
    estimate_ols, estimate_logit_demand,
    compute_ownership_matrix, compute_markups, recover_marginal_costs
)
from logit.visualizations import (
    plot_elasticity_heatmap, plot_demand_curves, plot_markup_comparison,
    plot_estimation_results, plot_iia_demonstration
)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("SIMPLE LOGIT DEMAND ESTIMATION")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Generate Data
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("=" * 50)

    df = create_estimation_dataset()

    print(f"\nDataset: {df.shape[0]} observations, {df['market_id'].nunique()} markets")
    print(f"Products: {df['product_name'].unique().tolist()}")

    print("\nTrue Parameters (what we're trying to recover):")
    print(f"  Alpha (price sensitivity): {TRUE_ALPHA}")
    print(f"  Beta_sugar:                {TRUE_BETA_SUGAR}")
    print(f"  Beta_const:                {TRUE_BETA_CONST}")

    print("\nSample data (Market 0):")
    sample = df[df['market_id'] == 0][['product_name', 'price', 'sugar', 'share', 'xi']]
    print(sample.to_string(index=False))

    # =========================================================================
    # STEP 2: Estimate Demand
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: ESTIMATING DEMAND")
    print("=" * 50)

    # --- OLS (biased due to endogeneity) ---
    print("\n--- OLS Estimation (BIASED) ---")
    Y = df['ln_share_ratio'].values
    X = df[['sugar', 'price']].values

    ols_results = estimate_ols(Y, X)
    ols_alpha = -ols_results['coefficients'][2]
    ols_beta_sugar = ols_results['coefficients'][1]
    ols_beta_const = ols_results['coefficients'][0]

    print(f"Alpha (OLS):      {ols_alpha:.4f}  (True: {TRUE_ALPHA})")
    print(f"Beta_sugar (OLS): {ols_beta_sugar:.4f}  (True: {TRUE_BETA_SUGAR})")
    print(f"Beta_const (OLS): {ols_beta_const:.4f}  (True: {TRUE_BETA_CONST})")
    print(f"\nBias in alpha: {(ols_alpha - TRUE_ALPHA) / TRUE_ALPHA * 100:.1f}%")
    print("=> OLS UNDERESTIMATES price sensitivity (endogeneity bias)")

    # --- IV/2SLS (consistent) ---
    print("\n--- IV/2SLS Estimation (CONSISTENT) ---")
    iv_results = estimate_logit_demand(df)

    print(f"Alpha (IV):      {iv_results['alpha']:.4f}  (True: {TRUE_ALPHA})")
    print(f"Beta_sugar (IV): {iv_results['beta_sugar']:.4f}  (True: {TRUE_BETA_SUGAR})")
    print(f"Beta_const (IV): {iv_results['beta_const']:.4f}  (True: {TRUE_BETA_CONST})")
    print(f"\nFirst Stage F-statistic: {iv_results['first_stage_f']:.2f}")
    print("=> F > 10 suggests strong instruments")

    # =========================================================================
    # STEP 3: Compute Elasticities
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: COMPUTING ELASTICITIES")
    print("=" * 50)

    # Use first market for elasticity calculations
    market0 = df[df['market_id'] == 0].copy()
    product_names = market0['product_name'].tolist()
    prices = market0['price'].values
    shares = market0['share'].values

    elasticity_matrix = compute_elasticities(iv_results['alpha'], prices, shares)

    print("\nElasticity Matrix:")
    print("(Row = quantity response, Column = price change)")
    print("-" * 50)

    # Pretty print matrix
    header = "            " + "  ".join([f"{n[:10]:>10}" for n in product_names])
    print(header)
    for i, name in enumerate(product_names):
        row = f"{name[:10]:>10}  " + "  ".join([f"{elasticity_matrix[i,j]:>10.3f}"
                                                 for j in range(len(product_names))])
        print(row)

    print("\nOwn-Price Elasticities:")
    for i, name in enumerate(product_names):
        print(f"  {name}: {elasticity_matrix[i,i]:.3f}")

    print("\nIIA PROBLEM DEMONSTRATION:")
    print("Notice: Cross-elasticities in each COLUMN are nearly identical!")
    print("This means if Fiber-Bran raises price, customers switch to")
    print("Choco-Bombs and Store-Frosted proportionally to their shares,")
    print("NOT based on how similar the products are!")

    # =========================================================================
    # STEP 4: Supply Side - Recover Marginal Costs
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: SUPPLY SIDE - RECOVERING MARGINAL COSTS")
    print("=" * 50)

    firm_ids = market0['firm_id'].values
    ownership = compute_ownership_matrix(firm_ids)

    print("\nOwnership Matrix (1 = same firm):")
    print(ownership)

    markups = compute_markups(iv_results['alpha'], shares, ownership)
    estimated_mc = recover_marginal_costs(prices, markups)
    true_mc = market0['marginal_cost'].values

    print("\nPrice Decomposition (Market 0):")
    print("-" * 60)
    print(f"{'Product':<15} {'Price':>8} {'Markup':>8} {'Est MC':>8} {'True MC':>8}")
    print("-" * 60)
    for i, name in enumerate(product_names):
        print(f"{name:<15} ${prices[i]:>7.2f} ${markups[i]:>7.2f} "
              f"${estimated_mc[i]:>7.2f} ${true_mc[i]:>7.2f}")

    print("\nKey Insight: We recovered marginal costs WITHOUT seeing accounting data!")
    print(f"Average cost estimation error: ${np.abs(estimated_mc - true_mc).mean():.3f}")

    # =========================================================================
    # STEP 5: Generate Visualizations
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 50)

    # 1. Elasticity Heatmap
    fig1 = plot_elasticity_heatmap(
        elasticity_matrix, product_names,
        title="Price Elasticity Matrix (Simple Logit)",
        save_path=OUTPUT_DIR / "logit_elasticity_heatmap.png"
    )

    # 2. Demand Curves
    products_for_curves = market0[['product_name', 'price', 'sugar']].copy()
    fig2 = plot_demand_curves(
        iv_results['alpha'], iv_results['beta_sugar'], iv_results['beta_const'],
        products_for_curves,
        save_path=OUTPUT_DIR / "logit_demand_curves.png"
    )

    # 3. Markup Comparison
    fig3 = plot_markup_comparison(
        product_names, prices, markups, estimated_mc, true_mc,
        save_path=OUTPUT_DIR / "logit_markup_comparison.png"
    )

    # 4. Estimation Results (OLS vs IV)
    true_params = {'alpha': TRUE_ALPHA, 'beta_sugar': TRUE_BETA_SUGAR, 'beta_const': TRUE_BETA_CONST}
    ols_params = {'alpha': ols_alpha, 'beta_sugar': ols_beta_sugar, 'beta_const': ols_beta_const}
    iv_params = {
        'alpha': iv_results['alpha'],
        'beta_sugar': iv_results['beta_sugar'],
        'beta_const': iv_results['beta_const'],
        'alpha_se': iv_results['std_errors'][2],
        'beta_sugar_se': iv_results['std_errors'][1],
        'beta_const_se': iv_results['std_errors'][0]
    }
    fig4 = plot_estimation_results(
        true_params, ols_params, iv_params,
        save_path=OUTPUT_DIR / "logit_estimation_comparison.png"
    )

    # 5. IIA Demonstration
    fig5 = plot_iia_demonstration(
        product_names, elasticity_matrix,
        save_path=OUTPUT_DIR / "logit_iia_problem.png"
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

    1. ENDOGENEITY MATTERS:
       - OLS underestimates price sensitivity (alpha)
       - High-quality products have high xi AND high prices
       - IV/2SLS using cost shifters corrects this bias

    2. THE IIA PROBLEM:
       - Logit cross-elasticities depend only on market shares
       - Not on how similar products are!
       - If sugary Choco-Bombs raises price, customers "unrealistically"
         switch to healthy Fiber-Bran at the same rate as sugary Store-Frosted

    3. SUPPLY SIDE:
       - We can back out marginal costs from demand estimates
       - No need for accounting data!
       - Key insight for merger analysis

    NEXT STEP: Nested Logit (fixes IIA by allowing within-group substitution)
    """)

    plt.show()


if __name__ == '__main__':
    main()
