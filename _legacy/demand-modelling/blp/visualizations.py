"""
Visualizations for BLP Model
============================
Creates publication-quality figures for:
1. Elasticity comparison (Logit vs BLP)
2. Diversion ratio comparison
3. Merger simulation results
4. Consumer heterogeneity plots
5. Parameter comparison tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def plot_elasticity_comparison(logit_eta: np.ndarray,
                               blp_eta: np.ndarray,
                               product_names: list,
                               save_path: str = None) -> plt.Figure:
    """
    Side-by-side comparison of Logit vs BLP elasticity matrices.

    Shows how BLP captures realistic substitution patterns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    matrices = [logit_eta, blp_eta]
    titles = ['Simple Logit (IIA Problem)', 'BLP (Realistic Substitution)']

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=2)

        ax.set_xticks(range(len(product_names)))
        ax.set_yticks(range(len(product_names)))
        ax.set_xticklabels(product_names, rotation=45, ha='right')
        ax.set_yticklabels(product_names)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # Add values
        for i in range(len(product_names)):
            for j in range(len(product_names)):
                val = matrix[i, j]
                color = 'white' if abs(val) > 2 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')

    fig.colorbar(im, ax=axes, label='Elasticity', shrink=0.8)

    fig.suptitle('BLP vs Logit: Elasticity Matrices\n'
                 'BLP shows higher substitution between similar products!',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add explanatory text box
    fig.text(0.5, -0.05,
             "Interpretation:\n"
             "• Logit (Left): Columns are identical (IIA). If Product A raises price, customers switch to B and C equally.\n"
             "• BLP (Right): Realistic substitution. If a sugary cereal raises price, customers switch to other sugary cereals.\n"
             "  (Notice the darker red values on the diagonal and between similar products)",
             ha='center', fontsize=11, bbox=dict(facecolor='#f0f0f0', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_diversion_comparison(logit_diversion: np.ndarray,
                              blp_diversion: np.ndarray,
                              product_names: list,
                              reference_product: int = 0,
                              save_path: str = None) -> plt.Figure:
    """
    Bar chart comparing diversion ratios between Logit and BLP.

    The key insight: BLP shows realistic diversion to similar products,
    while Logit diverts proportionally to market share (IIA).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ref_name = product_names[reference_product]
    other_indices = [i for i in range(len(product_names)) if i != reference_product]
    other_names = [product_names[i] for i in other_indices]

    logit_div = [logit_diversion[reference_product, i] * 100 for i in other_indices]
    blp_div = [blp_diversion[reference_product, i] * 100 for i in other_indices]

    x = np.arange(len(other_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, logit_div, width, label='Simple Logit',
                   color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, blp_div, width, label='BLP',
                   color='#3498db', edgecolor='black')

    # Add value labels
    for rect in bars1 + bars2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Product Gaining Sales', fontsize=12)
    ax.set_ylabel('Diversion Ratio (%)', fontsize=12)
    
    # Add explanatory text box
    ax.text(0.5, -0.25,
            "Interpretation:
"
            "• Logit: Diversion is proportional to market share (IIA).
"
            "• BLP: Diversion is higher to products with similar characteristics (e.g., similar sugar content).
"
            "  This captures that if 'Choco-Bombs' gets expensive, kids switch to 'Store-Frosted', not 'Fiber-Bran'.",
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(facecolor='#f0f0f0', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
            
    plt.tight_layout()
    ax.set_title(f'Diversion Ratios: Where Do {ref_name} Customers Go?\n'
                 f'(If {ref_name} exits market or raises price)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(other_names, fontsize=11)
    ax.legend(fontsize=11)

    # Add value labels
    for bar, val in zip(bars1, logit_div):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10)
    for bar, val in zip(bars2, blp_div):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=10)

    # Annotation
    ax.text(0.02, 0.98, 'Logit: Diversion proportional to market share\n'
                        'BLP: Diversion based on product similarity',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_merger_simulation(product_names: list,
                           pre_prices: np.ndarray,
                           post_prices: np.ndarray,
                           merging_products: list,
                           save_path: str = None) -> plt.Figure:
    """
    Bar chart showing pre vs post merger prices.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(product_names))
    width = 0.35

    colors_pre = ['#3498db'] * len(product_names)
    colors_post = ['#e74c3c' if i in merging_products else '#27ae60'
                   for i in range(len(product_names))]

    bars1 = ax.bar(x - width/2, pre_prices, width, label='Pre-Merger',
                   color=colors_pre, edgecolor='black')
    bars2 = ax.bar(x + width/2, post_prices, width, label='Post-Merger',
                   color=colors_post, edgecolor='black')

    ax.set_xlabel('Product', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('Merger Simulation: Price Effects\n'
                 '(Red = Merging Products)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(product_names, fontsize=11)
    ax.legend(fontsize=11)

    # Add price change labels
    for i, (pre, post) in enumerate(zip(pre_prices, post_prices)):
        change = (post - pre) / pre * 100
        color = 'red' if change > 0 else 'green'
        ax.text(i + width/2, post + 0.1, f'{change:+.1f}%',
                ha='center', fontsize=10, color=color, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_consumer_heterogeneity(draws: dict, alpha_mean: float,
                                sigma_alpha: float, sigma_sugar: float,
                                save_path: str = None) -> plt.Figure:
    """
    Scatter plot showing consumer heterogeneity in taste space.

    Shows how some consumers love sugar (Kids) while others hate it (Parents).
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    n = len(draws['v_alpha'])

    # Compute actual preferences
    alpha_i = alpha_mean + sigma_alpha * draws['v_alpha']
    beta_sugar_i = sigma_sugar * draws['v_sugar']  # Mean sugar taste = 0

    # Color by consumer type
    colors = np.zeros(n, dtype=object)
    colors[:] = '#95a5a6'  # Gray for average
    colors[(beta_sugar_i > 1) & (alpha_i < 1.5)] = '#e74c3c'  # Red for sugar-lovers
    colors[(beta_sugar_i < -1) & (alpha_i > 2.5)] = '#3498db'  # Blue for health-conscious

    ax.scatter(alpha_i, beta_sugar_i, c=colors, alpha=0.5, s=20)

    # Add type regions
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=alpha_mean, color='black', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Price Sensitivity (alpha_i)\n(Higher = More Sensitive)',
                  fontsize=12)
    ax.set_ylabel('Sugar Preference (beta_sugar_i)\n(+ = Loves Sugar, - = Hates Sugar)',
                  fontsize=12)
    ax.set_title('Consumer Heterogeneity in BLP\n'
                 'Each point is a simulated consumer',
                 fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='Sugar-Lovers (like Kids)'),
        mpatches.Patch(facecolor='#3498db', label='Health-Conscious (like Parents)'),
        mpatches.Patch(facecolor='#95a5a6', label='Average Consumer')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Annotations
    ax.text(0.5, 3.5, 'LOVES SUGAR\nLow Price Sensitivity\n(Kids)',
            fontsize=10, ha='center', color='#e74c3c', fontweight='bold')
    ax.text(3.5, -3.5, 'HATES SUGAR\nHigh Price Sensitivity\n(Parents)',
            fontsize=10, ha='center', color='#3498db', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_comparison_table(logit_results: dict, blp_results: dict,
                          true_params: dict, product_names: list,
                          save_path: str = None) -> plt.Figure:
    """
    Create a visual comparison table like in the fake case study.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Table data
    columns = ['Metric', 'Simple Logit', 'BLP', 'Conclusion']

    data = [
        ['Price Sensitivity (alpha)',
         f"{logit_results.get('alpha', 'N/A'):.2f}",
         f"{blp_results.get('alpha', 'N/A'):.2f} (mean)\n+/- {blp_results.get('sigma_alpha', 0):.2f} (std)",
         'BLP captures variance'],
        ['Sugar Taste (beta)',
         f"{logit_results.get('beta_sugar', 'N/A'):.2f}",
         f"{blp_results.get('beta_sugar', 'N/A'):.2f} (mean)\n+/- {blp_results.get('sigma_sugar', 0):.2f} (std)",
         'BLP shows polarization'],
        ['Substitution Pattern',
         'Proportional to\nmarket share (IIA)',
         'Based on product\nsimilarity',
         'BLP is realistic'],
        ['Merger Effect\n(Choco + Store)',
         f"+{logit_results.get('merger_effect', 1):.1f}%",
         f"+{blp_results.get('merger_effect', 5):.1f}%",
         'BLP predicts larger\nprice increase']
    ]

    # Create table
    table = ax.table(cellText=data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colColours=['#3498db', '#e74c3c', '#27ae60', '#f39c12'],
                     colWidths=[0.25, 0.25, 0.25, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold', color='white')
        table[(0, i)].set_facecolor('#2c3e50')

    ax.set_title('Logit vs BLP: Summary Comparison\n',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_demand_by_consumer_type(product_names: list, sugar: np.ndarray,
                                 prices: np.ndarray, draws: dict,
                                 alpha_mean: float, sigma_alpha: float,
                                 sigma_sugar: float,
                                 save_path: str = None) -> plt.Figure:
    """
    Show demand curves for different consumer types.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Classify consumers
    alpha_i = alpha_mean + sigma_alpha * draws['v_alpha']
    beta_sugar_i = sigma_sugar * draws['v_sugar']

    # Sugar lovers (kids)
    sugar_lover_mask = (beta_sugar_i > 1) & (alpha_i < 2)
    # Health conscious (parents)
    health_mask = (beta_sugar_i < -1) & (alpha_i > 2)

    from blp.blp_model import compute_individual_utilities, compute_individual_choice_probs

    price_range = np.linspace(1.5, 5.0, 50)
    titles = ['Sugar-Lovers (Kids)', 'Health-Conscious (Parents)']
    masks = [sugar_lover_mask, health_mask]

    for ax, title, mask in zip(axes, titles, masks):
        if mask.sum() < 10:
            ax.text(0.5, 0.5, 'Not enough consumers\nof this type',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Create subset draws
        subset_draws = {
            'v_alpha': draws['v_alpha'][mask],
            'v_sugar': draws['v_sugar'][mask]
        }

        for j, name in enumerate(product_names):
            shares = []
            base_delta = np.zeros(len(product_names))

            for p in price_range:
                # Vary price of this product
                test_prices = prices.copy()
                test_prices[j] = p

                # Compute utilities for subset
                utils = compute_individual_utilities(
                    base_delta, sugar, test_prices, subset_draws,
                    sigma_alpha, sigma_sugar
                )
                probs = compute_individual_choice_probs(utils)
                shares.append(probs[:, j].mean() * 100)

            ax.plot(price_range, shares, linewidth=2.5, label=name)

        ax.set_xlabel('Price ($)', fontsize=11)
        ax.set_ylabel('Market Share (%)', fontsize=11)
        ax.set_title(f'Demand by {title}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


if __name__ == '__main__':
    print("Testing BLP visualizations...")

    from blp.synthetic_data import draw_simulated_consumers, TRUE_ALPHA_MEAN, TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR

    draws = draw_simulated_consumers(n_draws=2000)
    fig = plot_consumer_heterogeneity(draws, TRUE_ALPHA_MEAN, TRUE_SIGMA_ALPHA, TRUE_SIGMA_SUGAR)
    plt.show()
