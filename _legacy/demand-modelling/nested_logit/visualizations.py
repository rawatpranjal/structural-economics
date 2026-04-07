"""
Visualizations for Nested Logit Model
=====================================
Creates publication-quality figures for:
1. Elasticity heatmaps (showing how nested logit breaks IIA)
2. Within-nest vs cross-nest substitution comparison
3. Demand curves by nest
4. Logit vs Nested Logit comparison
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


def plot_nested_elasticity_heatmap(elasticity_matrix: np.ndarray,
                                   product_names: list,
                                   nest_ids: np.ndarray,
                                   nest_names: dict = None,
                                   title: str = "Price Elasticity Matrix (Nested Logit)",
                                   save_path: str = None) -> plt.Figure:
    """
    Create a heatmap with nest boundaries highlighted.

    Shows that same-nest cross-elasticities are higher than
    different-nest cross-elasticities.

    Parameters
    ----------
    elasticity_matrix : np.ndarray
        JxJ elasticity matrix
    product_names : list
        Product names
    nest_ids : np.ndarray
        Nest assignment for each product
    nest_names : dict, optional
        Mapping from nest_id to name
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(11, 9))

    # Create heatmap
    im = ax.imshow(elasticity_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-6, vmax=3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Elasticity')

    # Set ticks and labels
    J = len(product_names)
    ax.set_xticks(range(J))
    ax.set_yticks(range(J))
    ax.set_xticklabels(product_names, rotation=45, ha='right')
    ax.set_yticklabels(product_names)

    # Add text annotations
    for i in range(J):
        for j in range(J):
            value = elasticity_matrix[i, j]
            color = 'white' if abs(value) > 2.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

    # Draw nest boundaries
    unique_nests = np.unique(nest_ids)
    for nest_id in unique_nests:
        indices = np.where(nest_ids == nest_id)[0]
        if len(indices) > 0:
            min_idx = indices.min() - 0.5
            max_idx = indices.max() + 0.5

            # Draw rectangle around nest block
            rect = plt.Rectangle((min_idx, min_idx), max_idx - min_idx, max_idx - min_idx,
                                  fill=False, edgecolor='yellow', linewidth=3)
            ax.add_patch(rect)

    ax.set_xlabel('Price of Product (column)', fontsize=12)
    ax.set_ylabel('Quantity of Product (row)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend for nest boundaries
    legend_elements = [mpatches.Patch(facecolor='none', edgecolor='yellow',
                                       linewidth=3, label='Same Nest Block')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add explanation
    ax.text(0.5, -0.15, 
            "Interpretation:\n"
            "• Yellow Blocks: Products in the same nest (e.g., all Sugary cereals).\n"
            "• Redder Colors: Higher substitution (cross-elasticity).\n"
            "• Result: Consumers are more likely to switch to another product in the SAME nest\n"
            "  than to a product in a different nest. This breaks the IIA assumption!",
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(facecolor='#f0f0f0', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    # Add annotation
    ax.text(0.5, -0.18, 'Note: Same-nest cross-elasticities (yellow blocks) are HIGHER\n'
                        'than different-nest cross-elasticities (outside yellow)',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic',
            color='darkgreen')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_substitution_comparison(logit_elasticity: np.ndarray,
                                 nested_elasticity: np.ndarray,
                                 product_names: list,
                                 nest_ids: np.ndarray,
                                 reference_product: int = 0,
                                 save_path: str = None) -> plt.Figure:
    """
    Bar chart comparing substitution patterns: Logit vs Nested Logit.

    Shows where customers go when a product raises its price.

    Parameters
    ----------
    logit_elasticity : np.ndarray
        Simple logit elasticity matrix
    nested_elasticity : np.ndarray
        Nested logit elasticity matrix
    product_names : list
        Product names
    nest_ids : np.ndarray
        Nest assignments
    reference_product : int
        Index of product that raises price
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ref_name = product_names[reference_product]
    ref_nest = nest_ids[reference_product]

    # Get cross-elasticities (where customers switch TO)
    other_indices = [i for i in range(len(product_names)) if i != reference_product]
    other_names = [product_names[i] for i in other_indices]
    other_nests = [nest_ids[i] for i in other_indices]

    logit_cross = [logit_elasticity[i, reference_product] for i in other_indices]
    nested_cross = [nested_elasticity[i, reference_product] for i in other_indices]

    x = np.arange(len(other_names))
    width = 0.35

    # Color bars by nest
    colors_logit = ['#3498db'] * len(other_names)
    colors_nested = ['#27ae60' if nest_ids[i] == ref_nest else '#e74c3c'
                     for i in other_indices]

    bars1 = ax.bar(x - width/2, logit_cross, width, label='Simple Logit',
                   color='#3498db', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, nested_cross, width, label='Nested Logit',
                   color=colors_nested, edgecolor='black')

    ax.set_xlabel('Product Gaining Sales', fontsize=12)
    ax.set_ylabel('Cross-Price Elasticity', fontsize=12)
    ax.set_title(f'Where Do Customers Go When {ref_name} Raises Price?\n'
                 f'Logit (IIA) vs Nested Logit', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(other_names, fontsize=11)

    # Custom legend
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Simple Logit (IIA)'),
        mpatches.Patch(facecolor='#27ae60', edgecolor='black', label='Nested: Same Nest'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Nested: Different Nest')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add value labels
    for bar, val in zip(bars1, logit_cross):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, nested_cross):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Annotation
    ax.text(0.02, 0.98, f'{ref_name} is in the Sugary nest\n'
                        'Nested Logit shows MORE substitution\n'
                        'to same-nest products!',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            style='italic', color='darkgreen',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_demand_curves_by_nest(alpha: float, sigma: float,
                               beta_sugar: float, beta_const: float,
                               products_df: pd.DataFrame,
                               save_path: str = None) -> plt.Figure:
    """
    Plot demand curves colored by nest membership.

    Parameters
    ----------
    alpha : float
        Price sensitivity
    sigma : float
        Nesting parameter
    beta_sugar : float
        Sugar taste parameter
    beta_const : float
        Constant
    products_df : pd.DataFrame
        Product data with nest assignments
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    prices_range = np.linspace(1.0, 7.0, 100)

    # Color by nest
    nest_colors = {1: '#e74c3c', 2: '#3498db'}  # Sugary=red, Healthy=blue

    for _, row in products_df.iterrows():
        shares = []

        for p in prices_range:
            # Compute all deltas
            deltas = []
            for _, other_row in products_df.iterrows():
                if other_row['product_name'] == row['product_name']:
                    d = beta_const + beta_sugar * row['sugar'] - alpha * p
                else:
                    d = beta_const + beta_sugar * other_row['sugar'] - alpha * other_row['price']
                deltas.append(d)

            deltas = np.array(deltas)
            nest_ids = products_df['nest_id'].values

            # Compute nested logit shares
            from nested_logit.nested_logit_model import compute_total_shares
            s_total, _, _ = compute_total_shares(deltas, nest_ids, sigma)

            # Get share of this product
            idx = products_df[products_df['product_name'] == row['product_name']].index[0]
            local_idx = list(products_df.index).index(idx)
            shares.append(s_total[local_idx] * 100)

        color = nest_colors[row['nest_id']]
        linestyle = '-' if row['nest_id'] == 1 else '--'

        ax.plot(prices_range, shares, linewidth=2.5, color=color, linestyle=linestyle,
                label=f"{row['product_name']} ({row['nest_name']})")

        # Mark current price
        current_idx = np.argmin(np.abs(prices_range - row['price']))
        ax.scatter(row['price'], shares[current_idx], s=100, color=color,
                   edgecolor='black', linewidth=2, zorder=5)

    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Market Share (%)', fontsize=12)
    ax.set_title('Demand Curves by Nest\n(Nested Logit)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(1.0, 7.0)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_nesting_parameter_effect(save_path: str = None) -> plt.Figure:
    """
    Show how the nesting parameter sigma affects substitution.

    sigma = 0: Collapses to simple logit (IIA)
    sigma -> 1: Products in nest are perfect substitutes

    Parameters
    ----------
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sigmas = [0.0, 0.5, 0.9]
    titles = ['sigma = 0 (Simple Logit)', 'sigma = 0.5 (Moderate Nesting)',
              'sigma = 0.9 (Strong Nesting)']

    delta = np.array([1.0, 0.8, 0.5, 0.3])
    nest_ids = np.array([1, 1, 2, 2])
    prices = np.array([3.0, 2.0, 5.0, 4.0])
    alpha = 1.5
    product_names = ['A (Nest 1)', 'B (Nest 1)', 'C (Nest 2)', 'D (Nest 2)']

    from nested_logit.nested_logit_model import (
        compute_total_shares, compute_nested_elasticities
    )

    for ax, sigma, title in zip(axes, sigmas, titles):
        if sigma == 0:
            sigma = 0.01  # Avoid division by zero

        s_total, s_within, _ = compute_total_shares(delta, nest_ids, sigma)
        eta = compute_nested_elasticities(alpha, sigma, prices, s_total, s_within, nest_ids)

        im = ax.imshow(eta, cmap='RdBu_r', aspect='auto', vmin=-8, vmax=3)

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(product_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(product_names, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add values
        for i in range(4):
            for j in range(4):
                val = eta[i, j]
                color = 'white' if abs(val) > 2 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, color=color)

        # Draw nest boundary
        rect = plt.Rectangle((-0.5, -0.5), 2, 2, fill=False,
                             edgecolor='yellow', linewidth=2)
        ax.add_patch(rect)
        rect2 = plt.Rectangle((1.5, 1.5), 2, 2, fill=False,
                              edgecolor='yellow', linewidth=2)
        ax.add_patch(rect2)

    # Shared colorbar
    fig.colorbar(im, ax=axes, label='Elasticity', shrink=0.8)

    fig.suptitle('Effect of Nesting Parameter on Substitution Patterns',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_estimation_comparison_nested(true_params: dict, logit_params: dict,
                                       nested_params: dict,
                                       save_path: str = None) -> plt.Figure:
    """
    Compare true parameters with Logit and Nested Logit estimates.

    Parameters
    ----------
    true_params : dict
        True parameter values
    logit_params : dict
        Simple logit estimates
    nested_params : dict
        Nested logit estimates
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    params = ['alpha', 'beta_sugar', 'beta_const', 'sigma']
    param_labels = ['Price Sensitivity\n(alpha)', 'Sugar Taste\n(beta)',
                    'Constant\n(beta_0)', 'Nesting\n(sigma)']

    x = np.arange(len(params))
    width = 0.25

    true_vals = [true_params.get(p, 0) for p in params]
    logit_vals = [logit_params.get(p, 0) for p in params]
    nested_vals = [nested_params.get(p, 0) for p in params]

    bars1 = ax.bar(x - width, true_vals, width, label='True', color='#27ae60',
                   edgecolor='black')
    bars2 = ax.bar(x, logit_vals, width, label='Simple Logit', color='#e74c3c',
                   edgecolor='black', alpha=0.8)
    bars3 = ax.bar(x + width, nested_vals, width, label='Nested Logit', color='#3498db',
                   edgecolor='black')

    ax.set_xlabel('Parameter', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Parameter Estimates: Logit vs Nested Logit',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add annotation
    ax.text(0.02, 0.98, 'Note: Nested Logit captures sigma,\n'
                        'improving substitution patterns',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            style='italic', color='darkblue')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


if __name__ == '__main__':
    print("Testing nested logit visualization functions...")
    plot_nesting_parameter_effect()
    plt.show()
