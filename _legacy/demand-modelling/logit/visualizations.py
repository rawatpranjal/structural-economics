"""
Visualizations for Simple Logit Model
=====================================
Creates publication-quality figures for:
1. Elasticity heatmaps (showing the IIA problem)
2. Demand curves
3. Markup comparisons
4. Estimation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style for clean, professional plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def plot_elasticity_heatmap(elasticity_matrix: np.ndarray,
                            product_names: list,
                            title: str = "Price Elasticity Matrix (Logit)",
                            save_path: str = None) -> plt.Figure:
    """
    Create a heatmap of the elasticity matrix.

    Highlights the IIA problem: all cross-elasticities in a column are identical.

    Parameters
    ----------
    elasticity_matrix : np.ndarray
        JxJ elasticity matrix
    product_names : list
        Names of products for labels
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(elasticity_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-5, vmax=2)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Elasticity')

    # Set ticks and labels
    ax.set_xticks(range(len(product_names)))
    ax.set_yticks(range(len(product_names)))
    ax.set_xticklabels(product_names, rotation=45, ha='right')
    ax.set_yticklabels(product_names)

    # Add text annotations
    for i in range(len(product_names)):
        for j in range(len(product_names)):
            value = elasticity_matrix[i, j]
            color = 'white' if abs(value) > 2 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    ax.set_xlabel('Price of Product (column)', fontsize=12)
    ax.set_ylabel('Quantity of Product (row)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add annotation about IIA
    ax.text(0.5, -0.15, 
            "Interpretation (The IIA Problem):\n"
            "• Notice that every value in a column (excluding the diagonal) is IDENTICAL.\n"
            "• This means if Product J raises its price, it loses customers to ALL other products\n"
            "  in exact proportion to their market shares, regardless of similarity.\n"
            "• This is unrealistic! (e.g., a luxury car buyer shouldn't switch to a bus pass)",
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(facecolor='#f0f0f0', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_demand_curves(alpha: float, beta_sugar: float, beta_const: float,
                       products_df: pd.DataFrame,
                       price_range: tuple = (1.0, 6.0),
                       save_path: str = None) -> plt.Figure:
    """
    Plot demand curves: market share vs price for each product.

    Shows how quantity demanded responds to price changes,
    holding other products' prices fixed.

    Parameters
    ----------
    alpha : float
        Price sensitivity parameter
    beta_sugar : float
        Sugar taste parameter
    beta_const : float
        Constant term
    products_df : pd.DataFrame
        Product characteristics (sugar, base price)
    price_range : tuple
        (min_price, max_price) for the curves
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    prices = np.linspace(price_range[0], price_range[1], 100)
    colors = plt.cm.Set1(np.linspace(0, 1, len(products_df)))

    for idx, (_, row) in enumerate(products_df.iterrows()):
        shares = []

        # Hold other products at their base prices
        other_products = products_df[products_df['product_name'] != row['product_name']]

        for p in prices:
            # Compute deltas
            delta_j = beta_const + beta_sugar * row['sugar'] - alpha * p

            other_deltas = []
            for _, other in other_products.iterrows():
                delta_k = beta_const + beta_sugar * other['sugar'] - alpha * other['price']
                other_deltas.append(delta_k)

            # Compute share
            exp_sum = np.exp(delta_j) + sum(np.exp(d) for d in other_deltas)
            share = np.exp(delta_j) / (1 + exp_sum)
            shares.append(share * 100)  # Convert to percentage

        ax.plot(prices, shares, linewidth=2.5, color=colors[idx],
                label=f"{row['product_name']} (Sugar={row['sugar']:.0f})")

        # Mark current price point
        current_idx = np.argmin(np.abs(prices - row['price']))
        ax.scatter(row['price'], shares[current_idx], s=100, color=colors[idx],
                   edgecolor='black', linewidth=2, zorder=5)

    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Market Share (%)', fontsize=12)
    ax.set_title('Logit Demand Curves\nSteeper slope indicates higher price sensitivity',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9)
    ax.set_xlim(price_range)
    ax.set_ylim(0, None)

    # Add grid
    ax.grid(True, alpha=0.5, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_markup_comparison(product_names: list,
                           prices: np.ndarray,
                           markups: np.ndarray,
                           marginal_costs: np.ndarray,
                           true_costs: np.ndarray = None,
                           save_path: str = None) -> plt.Figure:
    """
    Bar chart comparing prices, markups, and recovered marginal costs.

    Demonstrates that we can back out costs from demand estimation.

    Parameters
    ----------
    product_names : list
        Product names
    prices : np.ndarray
        Observed prices
    markups : np.ndarray
        Estimated markups
    marginal_costs : np.ndarray
        Recovered marginal costs (price - markup)
    true_costs : np.ndarray, optional
        True marginal costs (if known from simulation)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(product_names))
    width = 0.22

    # Stacked bar for price decomposition
    bars1 = ax.bar(x - width, marginal_costs, width, label='Estimated MC',
                   color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x - width, markups, width, bottom=marginal_costs,
                   label='Markup', color='#e74c3c', edgecolor='black')

    # True costs comparison
    if true_costs is not None:
        bars3 = ax.bar(x + width, true_costs, width, label='True MC',
                       color='#3498db', edgecolor='black', alpha=0.7)

    # Price line
    ax.scatter(x - width, prices, s=100, color='black', marker='_',
               linewidths=3, zorder=5, label='Observed Price')

    # Labels
    ax.set_xlabel('Product', fontsize=12)
    ax.set_ylabel('Dollars ($)', fontsize=12)
    ax.set_title('Price Decomposition: Marginal Cost + Markup = Price',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(product_names, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)

    # Add value labels on bars
    for bar, mc, mu in zip(bars1, marginal_costs, markups):
        ax.text(bar.get_x() + bar.get_width()/2, mc/2,
                f'${mc:.2f}', ha='center', va='center', fontsize=9, color='white')
        ax.text(bar.get_x() + bar.get_width()/2, mc + mu/2,
                f'${mu:.2f}', ha='center', va='center', fontsize=9, color='white')

    ax.set_ylim(0, max(prices) * 1.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_estimation_results(true_params: dict, ols_params: dict,
                            iv_params: dict, save_path: str = None) -> plt.Figure:
    """
    Compare true parameters with OLS and IV estimates.

    Demonstrates that OLS is biased (price endogeneity) while IV recovers truth.

    Parameters
    ----------
    true_params : dict
        True parameter values
    ols_params : dict
        OLS estimates (biased)
    iv_params : dict
        IV/2SLS estimates (consistent)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    params = ['alpha', 'beta_sugar', 'beta_const']
    param_labels = ['Price Sensitivity\n(alpha)', 'Sugar Taste\n(beta_sugar)', 'Constant\n(beta_0)']

    x = np.arange(len(params))
    width = 0.25

    true_vals = [true_params.get(p, 0) for p in params]
    ols_vals = [ols_params.get(p, 0) for p in params]
    iv_vals = [iv_params.get(p, 0) for p in params]
    iv_se = [iv_params.get(f'{p}_se', 0) for p in params]

    # Bars
    bars1 = ax.bar(x - width, true_vals, width, label='True', color='#27ae60',
                   edgecolor='black')
    bars2 = ax.bar(x, ols_vals, width, label='OLS (Biased)', color='#e74c3c',
                   edgecolor='black', alpha=0.8)
    bars3 = ax.bar(x + width, iv_vals, width, label='IV/2SLS', color='#3498db',
                   edgecolor='black')

    # Error bars for IV
    ax.errorbar(x + width, iv_vals, yerr=[s * 1.96 for s in iv_se],
                fmt='none', color='black', capsize=5, capthick=2)

    ax.set_xlabel('Parameter', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Parameter Estimation: True vs OLS vs IV\n(OLS biased due to price endogeneity)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Add bias annotation
    ax.text(0.02, 0.98, 'Note: OLS underestimates alpha\n(price is correlated with xi)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            style='italic', color='darkred')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_iia_demonstration(product_names: list, elasticity_matrix: np.ndarray,
                           save_path: str = None) -> plt.Figure:
    """
    Visualize the IIA problem with cross-price elasticities.

    Shows that if Product A raises price, substitution to B and C
    is proportional to their market shares (not similarity).

    Parameters
    ----------
    product_names : list
        Product names
    elasticity_matrix : np.ndarray
        Elasticity matrix
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, axes = plt.subplots(1, len(product_names), figsize=(15, 5))

    colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12']

    for j, (ax, name) in enumerate(zip(axes, product_names)):
        # Cross-elasticities when product j raises price
        cross_elast = elasticity_matrix[:, j].copy()
        cross_elast[j] = 0  # Remove own-price

        other_names = [n for i, n in enumerate(product_names) if i != j]
        other_elast = [cross_elast[i] for i in range(len(product_names)) if i != j]

        bars = ax.barh(other_names, other_elast, color=colors[:len(other_names)],
                       edgecolor='black')

        ax.set_xlabel('Cross-Price Elasticity', fontsize=10)
        ax.set_title(f'If {name} raises price\nwho gains?', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, other_elast):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)

    fig.suptitle('The IIA Problem: Cross-Elasticities Depend Only on Market Share\n'
                 '(Not product similarity!)', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


if __name__ == '__main__':
    # Test with dummy data
    print("Testing visualization functions...")

    product_names = ['Choco-Bombs', 'Fiber-Bran', 'Store-Frosted']
    elasticity_matrix = np.array([
        [-2.5, 0.15, 0.15],
        [0.08, -3.2, 0.08],
        [0.05, 0.05, -1.8]
    ])

    fig = plot_elasticity_heatmap(elasticity_matrix, product_names)
    plt.show()
