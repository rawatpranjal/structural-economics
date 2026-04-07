"""
BLP Demand Estimation - Full Pipeline
=====================================
This script demonstrates the complete BLP workflow:

1. Generate synthetic data with consumer heterogeneity
2. Compare Simple Logit vs BLP elasticities
3. Run merger simulation
4. Generate all visualizations

This is the culmination of the demand estimation journey:
Simple Logit -> Nested Logit -> BLP

Run: python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from blp.synthetic_data import (
    create_estimation_dataset, draw_simulated_consumers,
    get_consumer_type_analysis,
    TRUE_ALPHA_MEAN, TRUE_BETA_SUGAR_MEAN, TRUE_BETA_CONST,
    TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR
)
from blp.blp_model import (
    compute_blp_shares, compute_blp_elasticities,
    compute_simple_logit_elasticities, compute_diversion_ratios
)
from blp.simulation import (
    compute_ownership_matrix, compute_markups_blp, recover_marginal_costs,
    simulate_merger, compute_blp_share_derivatives
)
from blp.visualizations import (
    plot_elasticity_comparison, plot_diversion_comparison,
    plot_merger_simulation, plot_consumer_heterogeneity,
    plot_comparison_table
)

OUTPUT_DIR = Path(__file__).parent.parent / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("BLP DEMAND ESTIMATION - FULL PIPELINE")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Generate Data and Consumer Draws
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: GENERATING DATA WITH CONSUMER HETEROGENEITY")
    print("=" * 50)

    df = create_estimation_dataset()
    draws = draw_simulated_consumers(n_draws=1000)

    print(f"\nTrue Parameters (Mean):")
    print(f"  Alpha (price sensitivity): {TRUE_ALPHA_MEAN}")
    print(f"  Beta_sugar:                {TRUE_BETA_SUGAR_MEAN}")
    print(f"  Beta_const:                {TRUE_BETA_CONST}")

    print(f"\nTrue Parameters (Heterogeneity):")
    print(f"  Sigma_alpha (price variance): {TRUE_SIGMA_ALPHA}")
    print(f"  Sigma_sugar (sugar variance): {TRUE_SIGMA_SUGAR}")

    print("\nConsumer Type Analysis:")
    consumer_df = get_consumer_type_analysis(n_draws=5000)
    print(consumer_df['type'].value_counts())

    print("\nSample Market Data:")
    market0 = df[df['market_id'] == 0].reset_index(drop=True)
    print(market0[['product_name', 'price', 'sugar', 'share', 'xi']].to_string(index=False))

    # =========================================================================
    # STEP 2: Compute Elasticities - BLP vs Logit
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: COMPUTING ELASTICITIES - THE KEY COMPARISON")
    print("=" * 50)

    product_names = market0['product_name'].tolist()
    prices = market0['price'].values
    sugar = market0['sugar'].values
    shares = market0['share'].values

    # Compute mean utility (delta)
    delta = (TRUE_BETA_CONST
             + TRUE_BETA_SUGAR_MEAN * sugar
             - TRUE_ALPHA_MEAN * prices
             + market0['xi'].values)

    # BLP Elasticities
    blp_eta = compute_blp_elasticities(
        delta, sugar, prices, draws,
        TRUE_ALPHA_MEAN, TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR
    )

    # Simple Logit Elasticities (for comparison)
    logit_eta = compute_simple_logit_elasticities(TRUE_ALPHA_MEAN, prices, shares)

    print("\n--- Simple Logit Elasticity Matrix (IIA Problem) ---")
    print_matrix(logit_eta, product_names)

    print("\n--- BLP Elasticity Matrix (Realistic Substitution) ---")
    print_matrix(blp_eta, product_names)

    # =========================================================================
    # STEP 3: Diversion Ratios - The Key Insight
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: DIVERSION RATIOS - WHERE DO CUSTOMERS GO?")
    print("=" * 50)

    logit_div = compute_diversion_ratios(logit_eta)
    blp_div = compute_diversion_ratios(blp_eta)

    print("\nIf CHOCO-BOMBS raises price, what % of lost sales go to each rival?")
    print("\n| Product         | Logit (IIA) | BLP (Realistic) |")
    print("|-----------------|-------------|-----------------|")

    for i, name in enumerate(product_names):
        if i != 0:  # Skip Choco-Bombs itself
            logit_val = logit_div[0, i] * 100
            blp_val = blp_div[0, i] * 100
            print(f"| {name:<15} | {logit_val:>9.1f}%  | {blp_val:>13.1f}%  |")

    print("""
    KEY INSIGHT:
    - Logit says: Customers switch to Fiber-Bran at high rate (large market share)
    - BLP says: Customers PRIMARILY switch to Store-Frosted (also sugary!)

    This is because BLP captures that sugar-lovers are more likely to
    substitute to another sugary product, not a healthy one!
    """)

    # =========================================================================
    # STEP 4: Supply Side - Recover Marginal Costs
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: SUPPLY SIDE - RECOVERING MARGINAL COSTS")
    print("=" * 50)

    firm_ids = market0['firm_id'].values
    ownership = compute_ownership_matrix(firm_ids)
    true_mc = market0['marginal_cost'].values

    # Compute share derivatives
    deriv = compute_blp_share_derivatives(
        delta, sugar, prices, draws,
        TRUE_ALPHA_MEAN, TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR
    )

    markups = compute_markups_blp(shares, deriv, ownership)
    estimated_mc = recover_marginal_costs(prices, markups)

    print("\nPrice Decomposition:")
    print("-" * 60)
    print(f"{'Product':<15} {'Price':>8} {'Markup':>8} {'Est MC':>8} {'True MC':>8}")
    print("-" * 60)
    for i, name in enumerate(product_names):
        print(f"{name:<15} ${prices[i]:>7.2f} ${markups[i]:>7.2f} "
              f"${estimated_mc[i]:>7.2f} ${true_mc[i]:>7.2f}")

    # =========================================================================
    # STEP 5: Merger Simulation
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 5: MERGER SIMULATION")
    print("=" * 50)

    print("\nScenario: Choco-Bombs (Firm 1) acquires Store-Frosted (Firm 3)")
    print("These are CLOSE SUBSTITUTES in BLP (both sugary)")
    print("This merger should raise prices significantly!")

    merger_results = simulate_merger(
        pre_prices=prices,
        pre_shares=shares,
        marginal_costs=estimated_mc,
        pre_firm_ids=firm_ids,
        acquiring_firm=1,
        acquired_firm=3,
        sugar=sugar,
        draws=draws,
        alpha_mean=TRUE_ALPHA_MEAN,
        sigma_alpha=TRUE_SIGMA_ALPHA,
        sigma_sugar=TRUE_SIGMA_SUGAR,
        delta=delta
    )

    # Also compute logit merger prediction for comparison
    from logit.logit_model import compute_markups, compute_share_derivatives

    logit_deriv = compute_share_derivatives(TRUE_ALPHA_MEAN, shares)
    logit_markups = compute_markups(TRUE_ALPHA_MEAN, shares, ownership)
    logit_mc = recover_marginal_costs(prices, logit_markups)

    # Simple logit merger (approximation)
    post_ownership_logit = compute_ownership_matrix(merger_results['post_firm_ids'])
    logit_post_markups = compute_markups(TRUE_ALPHA_MEAN, shares, post_ownership_logit)
    logit_post_prices = logit_mc + logit_post_markups

    print("\n| Product        | Pre-Price | Logit Post | BLP Post  |")
    print("|----------------|-----------|------------|-----------|")
    for i, name in enumerate(product_names):
        pre = prices[i]
        logit_post = logit_post_prices[i]
        blp_post = merger_results['post_prices'][i]
        print(f"| {name:<14} | ${pre:>8.2f} | ${logit_post:>9.2f} | ${blp_post:>8.2f} |")

    print("\nPrice Change Summary:")
    print("-" * 50)
    for i, name in enumerate(product_names):
        logit_chg = (logit_post_prices[i] - prices[i]) / prices[i] * 100
        blp_chg = merger_results['price_change_pct'][i]
        print(f"{name}: Logit +{logit_chg:.1f}%, BLP +{blp_chg:.1f}%")

    # =========================================================================
    # STEP 6: Generate Visualizations
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 50)

    # 1. Elasticity Comparison
    fig1 = plot_elasticity_comparison(
        logit_eta, blp_eta, product_names,
        save_path=OUTPUT_DIR / "blp_elasticity_comparison.png"
    )

    # 2. Diversion Comparison
    fig2 = plot_diversion_comparison(
        logit_div, blp_div, product_names, reference_product=0,
        save_path=OUTPUT_DIR / "blp_diversion_comparison.png"
    )

    # 3. Merger Simulation
    fig3 = plot_merger_simulation(
        product_names, prices, merger_results['post_prices'],
        merging_products=[0, 2],  # Choco-Bombs and Store-Frosted
        save_path=OUTPUT_DIR / "blp_merger_simulation.png"
    )

    # 4. Consumer Heterogeneity
    fig4 = plot_consumer_heterogeneity(
        draws, TRUE_ALPHA_MEAN, TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR,
        save_path=OUTPUT_DIR / "blp_consumer_heterogeneity.png"
    )

    # 5. Summary Comparison Table
    logit_results = {
        'alpha': TRUE_ALPHA_MEAN,
        'beta_sugar': TRUE_BETA_SUGAR_MEAN,
        'merger_effect': (logit_post_prices[0] - prices[0]) / prices[0] * 100
    }
    blp_results = {
        'alpha': TRUE_ALPHA_MEAN,
        'beta_sugar': TRUE_BETA_SUGAR_MEAN,
        'sigma_alpha': TRUE_SIGMA_ALPHA,
        'sigma_sugar': TRUE_SIGMA_SUGAR,
        'merger_effect': merger_results['price_change_pct'][0]
    }
    fig5 = plot_comparison_table(
        logit_results, blp_results, {}, product_names,
        save_path=OUTPUT_DIR / "blp_comparison_table.png"
    )

    print("\nAll figures saved to:", OUTPUT_DIR)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: THE BLP JOURNEY")
    print("=" * 70)

    print("""
    We've now covered the three main demand models:

    1. SIMPLE LOGIT:
       - Easy to estimate (IV regression)
       - IIA Problem: Unrealistic substitution patterns
       - All cross-elasticities proportional to market share

    2. NESTED LOGIT:
       - Groups products into nests
       - Higher substitution within nests
       - Still somewhat restrictive

    3. BLP (RANDOM COEFFICIENTS):
       - Full consumer heterogeneity
       - Substitution based on product similarity
       - The industry standard for merger analysis

    KEY INSIGHT FROM THIS EXAMPLE:
    ------------------------------------
    When Choco-Bombs merges with Store-Frosted...

    - Logit predicts: Small price increase (~{:.1f}%)
      (Because they're "not close competitors" by market share)

    - BLP predicts: Large price increase (~{:.1f}%)
      (Because they're VERY close competitors in taste space!)

    BLP correctly identifies that sugar-loving consumers view these
    products as close substitutes. The merger gives the combined firm
    pricing power over this consumer segment.

    This is why BLP is the standard for antitrust analysis!
    """.format(
        (logit_post_prices[0] - prices[0]) / prices[0] * 100,
        merger_results['price_change_pct'][0]
    ))

    plt.show()


def print_matrix(matrix: np.ndarray, names: list):
    """Helper function to print matrices nicely."""
    header = "              " + "  ".join([f"{n[:12]:>12}" for n in names])
    print(header)
    for i, name in enumerate(names):
        row = f"{name[:12]:>12}  " + "  ".join([f"{matrix[i,j]:>12.3f}"
                                                 for j in range(len(names))])
        print(row)


if __name__ == '__main__':
    main()
