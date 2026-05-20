#!/usr/bin/env python3
"""Mixed logit demand with simulated maximum likelihood."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


PRODUCTS = ["Budget", "Mainstream", "Premium", "Niche"]
LIKELIHOOD_FLOOR = 1e-14
MIXED_START = np.array([-0.75, 0.85, np.log(0.25), np.log(0.35)])
MIXED_BOUNDS = [
    (-3.0, -0.05),
    (0.05, 2.5),
    (np.log(0.03), np.log(1.3)),
    (np.log(0.03), np.log(1.3)),
]
MIXED_MAXITER = 220
MIXED_FTOL = 1e-10
LOGIT_START = np.array([-0.75, 0.85])
LOGIT_MAXITER = 160
PROFILE_GRID_SIZE = 21
PRICE_SUBSTITUTION_STEP = 0.10


def softmax(utility: np.ndarray) -> np.ndarray:
    """Stable logit probabilities along the product axis."""
    shifted = utility - utility.max(axis=-1, keepdims=True)
    exp_utility = np.exp(shifted)
    return exp_utility / exp_utility.sum(axis=-1, keepdims=True)


def make_product_panel(seed: int, n_consumers: int) -> dict[str, np.ndarray]:
    """Create repeated differentiated-product choice occasions."""
    rng = np.random.default_rng(seed)
    base_price = np.array([1.8, 2.6, 3.7, 3.2])
    base_quality = np.array([1.0, 1.7, 2.6, 2.0])
    promo = rng.normal(0.0, 0.28, size=(n_consumers, len(PRODUCTS)))
    match = rng.normal(0.0, 0.35, size=(n_consumers, len(PRODUCTS)))
    price = np.maximum(0.7, base_price + promo)
    quality = base_quality + match
    return {"price": price, "quality": quality}


def draw_choices(
    theta: np.ndarray,
    price: np.ndarray,
    quality: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Simulate choices from the mixed-logit data-generating process."""
    rng = np.random.default_rng(seed)
    alpha, beta, sigma_alpha, sigma_beta = theta
    n_consumers = price.shape[0]
    price_coef = alpha + sigma_alpha * rng.normal(size=n_consumers)
    quality_coef = beta + sigma_beta * rng.normal(size=n_consumers)
    utility = price_coef[:, None] * price + quality_coef[:, None] * quality
    utility += rng.gumbel(size=utility.shape)
    return utility.argmax(axis=1)


def mixed_logit_probabilities(
    theta: np.ndarray,
    price: np.ndarray,
    quality: np.ndarray,
    draws: np.ndarray,
    available: np.ndarray | None = None,
) -> np.ndarray:
    """Simulated choice probabilities for each observed choice occasion."""
    alpha, beta, log_sigma_alpha, log_sigma_beta = theta
    sigma_alpha = np.exp(log_sigma_alpha)
    sigma_beta = np.exp(log_sigma_beta)
    price_coef = alpha + sigma_alpha * draws[:, 0]
    quality_coef = beta + sigma_beta * draws[:, 1]
    utility = (
        price_coef[:, None, None] * price[None, :, :]
        + quality_coef[:, None, None] * quality[None, :, :]
    )
    if available is not None:
        utility = np.where(available[None, None, :], utility, -1e9)
    return softmax(utility).mean(axis=0)


def logit_probabilities(
    theta: np.ndarray,
    price: np.ndarray,
    quality: np.ndarray,
    available: np.ndarray | None = None,
) -> np.ndarray:
    """Plain-logit probabilities with no random coefficients."""
    alpha, beta = theta
    utility = alpha * price + beta * quality
    if available is not None:
        utility = np.where(available[None, :], utility, -1e9)
    return softmax(utility)


def mixed_negative_log_likelihood(
    theta: np.ndarray,
    price: np.ndarray,
    quality: np.ndarray,
    choices: np.ndarray,
    draws: np.ndarray,
) -> float:
    """Average negative simulated log likelihood."""
    probabilities = mixed_logit_probabilities(theta, price, quality, draws)
    chosen = probabilities[np.arange(len(choices)), choices]
    return float(-np.log(np.maximum(chosen, LIKELIHOOD_FLOOR)).mean())


def logit_negative_log_likelihood(
    theta: np.ndarray,
    price: np.ndarray,
    quality: np.ndarray,
    choices: np.ndarray,
) -> float:
    """Average negative log likelihood for the homogeneous logit."""
    probabilities = logit_probabilities(theta, price, quality)
    chosen = probabilities[np.arange(len(choices)), choices]
    return float(-np.log(np.maximum(chosen, LIKELIHOOD_FLOOR)).mean())


def estimate_mixed_logit(
    price: np.ndarray,
    quality: np.ndarray,
    choices: np.ndarray,
    draws: np.ndarray,
) -> dict[str, object]:
    """Estimate mean and dispersion parameters by simulated likelihood."""
    result = minimize(
        mixed_negative_log_likelihood,
        MIXED_START,
        args=(price, quality, choices, draws),
        method="L-BFGS-B",
        bounds=MIXED_BOUNDS,
        options={"maxiter": MIXED_MAXITER, "ftol": MIXED_FTOL},
    )
    alpha, beta, log_sigma_alpha, log_sigma_beta = result.x
    theta = np.array([alpha, beta, np.exp(log_sigma_alpha), np.exp(log_sigma_beta)])
    return {
        "theta": theta,
        "raw_theta": np.asarray(result.x, dtype=float),
        "criterion": float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
    }


def estimate_plain_logit(
    price: np.ndarray,
    quality: np.ndarray,
    choices: np.ndarray,
) -> dict[str, object]:
    """Estimate the homogeneous logit restriction."""
    result = minimize(
        logit_negative_log_likelihood,
        LOGIT_START,
        args=(price, quality, choices),
        method="BFGS",
        options={"maxiter": LOGIT_MAXITER},
    )
    return {
        "theta": np.asarray(result.x, dtype=float),
        "criterion": float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
    }


def observed_shares(choices: np.ndarray) -> np.ndarray:
    """Observed product choice shares."""
    return np.bincount(choices, minlength=len(PRODUCTS)) / len(choices)


def diversion_rates(full_shares: np.ndarray, restricted_shares: np.ndarray, removed: int) -> np.ndarray:
    """Share recapture after removing one product."""
    diversion = np.full_like(full_shares, np.nan, dtype=float)
    lost_share = full_shares[removed]
    for product in range(len(full_shares)):
        if product != removed:
            diversion[product] = (restricted_shares[product] - full_shares[product]) / lost_share
    return diversion


def price_substitution_table(
    logit_theta: np.ndarray,
    mixed_theta: np.ndarray,
    price: np.ndarray,
    quality: np.ndarray,
    draws: np.ndarray,
) -> pd.DataFrame:
    """Share recapture from a finite price increase in each product."""
    rows: list[dict[str, object]] = []
    models = [
        ("Plain logit", lambda p: logit_probabilities(logit_theta, p, quality).mean(axis=0)),
        ("Mixed logit", lambda p: mixed_logit_probabilities(mixed_theta, p, quality, draws).mean(axis=0)),
    ]
    for model_name, share_func in models:
        base_shares = share_func(price)
        matrix = np.zeros((len(PRODUCTS), len(PRODUCTS)))
        for shocked in range(len(PRODUCTS)):
            perturbed_price = price.copy()
            perturbed_price[:, shocked] += PRICE_SUBSTITUTION_STEP
            perturbed_shares = share_func(perturbed_price)
            lost_share = base_shares[shocked] - perturbed_shares[shocked]
            for receiver in range(len(PRODUCTS)):
                if receiver == shocked:
                    matrix[receiver, shocked] = -1.0
                else:
                    matrix[receiver, shocked] = (perturbed_shares[receiver] - base_shares[receiver]) / lost_share
        for receiver, product in enumerate(PRODUCTS):
            row: dict[str, object] = {"Model": model_name, "Receiving product": product}
            for shocked, shocked_product in enumerate(PRODUCTS):
                row[f"Price up: {shocked_product}"] = matrix[receiver, shocked]
            rows.append(row)
    return pd.DataFrame(rows)


def substitution_matrices_from_table(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert the price-substitution table into model-specific matrices."""
    columns = [f"Price up: {product}" for product in PRODUCTS]
    matrices: dict[str, np.ndarray] = {}
    for model_name in table["Model"].drop_duplicates():
        model_rows = table.loc[table["Model"] == model_name].set_index("Receiving product")
        matrices[str(model_name)] = model_rows.loc[PRODUCTS, columns].to_numpy(dtype=float)
    return matrices


def parameter_table(theta_true: np.ndarray, logit: dict[str, object], mixed: dict[str, object]) -> pd.DataFrame:
    """Parameter recovery table."""
    logit_theta = np.asarray(logit["theta"], dtype=float)
    mixed_theta = np.asarray(mixed["theta"], dtype=float)
    return pd.DataFrame(
        {
            "Parameter": ["Mean price taste", "Mean quality taste", "SD price taste", "SD quality taste"],
            "True": [f"{value:.4f}" for value in theta_true],
            "Plain logit": [f"{logit_theta[0]:.4f}", f"{logit_theta[1]:.4f}", "not estimated", "not estimated"],
            "Mixed logit": [f"{value:.4f}" for value in mixed_theta],
            "Mixed error": [f"{value:.4f}" for value in mixed_theta - theta_true],
        }
    )


def share_fit_table(
    observed: np.ndarray,
    true_fit: np.ndarray,
    logit_fit: np.ndarray,
    mixed_fit: np.ndarray,
) -> pd.DataFrame:
    """Observed and fitted shares by product."""
    return pd.DataFrame(
        {
            "Product": PRODUCTS,
            "Observed share": observed,
            "True probability": true_fit,
            "Plain logit": logit_fit,
            "Mixed logit": mixed_fit,
            "Mixed error": mixed_fit - observed,
        }
    )


def profile_surface(
    alpha: float,
    beta: float,
    price: np.ndarray,
    quality: np.ndarray,
    choices: np.ndarray,
    draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Profile the simulated likelihood over random-coefficient dispersions."""
    sigma_alpha_grid = np.linspace(0.05, 0.80, PROFILE_GRID_SIZE)
    sigma_beta_grid = np.linspace(0.05, 1.05, PROFILE_GRID_SIZE)
    values = np.zeros((len(sigma_beta_grid), len(sigma_alpha_grid)))
    for i, sigma_beta in enumerate(sigma_beta_grid):
        for j, sigma_alpha in enumerate(sigma_alpha_grid):
            theta = np.array([alpha, beta, np.log(sigma_alpha), np.log(sigma_beta)])
            values[i, j] = mixed_negative_log_likelihood(theta, price, quality, choices, draws)
    values -= values.min()
    return sigma_alpha_grid, sigma_beta_grid, values


def plot_price_substitution_matrices(matrices: dict[str, np.ndarray]) -> plt.Figure:
    """Plot comparable plain-logit and mixed-logit substitution matrices."""
    model_names = ["Plain logit", "Mixed logit"]
    off_diagonal_values = np.concatenate(
        [
            matrix[~np.eye(matrix.shape[0], dtype=bool)]
            for matrix in (matrices[name] for name in model_names)
        ]
    )
    vmin = float(off_diagonal_values.min())
    vmax = float(off_diagonal_values.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), constrained_layout=True)
    image = None
    diagonal_mask = np.eye(len(PRODUCTS), dtype=bool)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#eeeeee")

    for ax, model_name in zip(axes, model_names, strict=True):
        matrix = matrices[model_name]
        display_matrix = np.ma.array(matrix, mask=diagonal_mask)
        image = ax.imshow(display_matrix, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(model_name)
        ax.set_xlabel("Shocked product")
        ax.set_xticks(np.arange(len(PRODUCTS)))
        ax.set_xticklabels(PRODUCTS, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(PRODUCTS)))
        ax.set_yticklabels(PRODUCTS)
        for receiver in range(len(PRODUCTS)):
            for shocked in range(len(PRODUCTS)):
                if receiver == shocked:
                    label = "-1"
                    color = "black"
                else:
                    label = f"{matrix[receiver, shocked]:.2f}"
                    midpoint = (vmin + vmax) / 2.0
                    color = "white" if matrix[receiver, shocked] < midpoint else "black"
                ax.text(shocked, receiver, label, ha="center", va="center", color=color, fontsize=9)

    axes[0].set_ylabel("Receiving product")
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.03, label="Off-diagonal recapture share")
    return fig


def main() -> None:
    n_consumers = 1_500
    n_draws = 120
    theta_true = np.array([-1.00, 1.10, 0.36, 0.55])
    panel = make_product_panel(431, n_consumers)
    price = panel["price"]
    quality = panel["quality"]
    choices = draw_choices(theta_true, price, quality, 432)
    simulation_draws = np.random.default_rng(433).normal(size=(n_draws, 2))

    logit = estimate_plain_logit(price, quality, choices)
    mixed = estimate_mixed_logit(price, quality, choices, simulation_draws)
    true_raw = np.array([theta_true[0], theta_true[1], np.log(theta_true[2]), np.log(theta_true[3])])

    observed = observed_shares(choices)
    true_probs = mixed_logit_probabilities(true_raw, price, quality, simulation_draws).mean(axis=0)
    logit_probs = logit_probabilities(np.asarray(logit["theta"], dtype=float), price, quality).mean(axis=0)
    mixed_probs = mixed_logit_probabilities(np.asarray(mixed["raw_theta"], dtype=float), price, quality, simulation_draws).mean(axis=0)

    removed_product = 2
    available = np.ones(len(PRODUCTS), dtype=bool)
    available[removed_product] = False
    logit_restricted = logit_probabilities(np.asarray(logit["theta"], dtype=float), price, quality, available).mean(axis=0)
    mixed_restricted = mixed_logit_probabilities(np.asarray(mixed["raw_theta"], dtype=float), price, quality, simulation_draws, available).mean(axis=0)
    logit_diversion = diversion_rates(logit_probs, logit_restricted, removed_product)
    mixed_diversion = diversion_rates(mixed_probs, mixed_restricted, removed_product)
    substitution_table = price_substitution_table(
        np.asarray(logit["theta"], dtype=float),
        np.asarray(mixed["raw_theta"], dtype=float),
        price,
        quality,
        simulation_draws,
    )
    substitution_matrices = substitution_matrices_from_table(substitution_table)

    sigma_alpha_grid, sigma_beta_grid, surface = profile_surface(
        np.asarray(mixed["raw_theta"], dtype=float)[0],
        np.asarray(mixed["raw_theta"], dtype=float)[1],
        price,
        quality,
        choices,
        simulation_draws,
    )

    print("Mixed-logit simulation tutorial")
    print(f"  True theta: {theta_true}")
    print(f"  Plain logit theta: {np.asarray(logit['theta'])}")
    print(f"  Mixed logit theta: {np.asarray(mixed['theta'])}")
    print(f"  Mixed likelihood success: {mixed['success']}, criterion: {mixed['criterion']:.4f}")

    setup_style()

    fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(PRODUCTS))
    width = 0.2
    ax1.bar(x - 1.5 * width, observed, width, label="Observed")
    ax1.bar(x - 0.5 * width, true_probs, width, label="True")
    ax1.bar(x + 0.5 * width, logit_probs, width, label="Plain logit")
    ax1.bar(x + 1.5 * width, mixed_probs, width, label="Mixed logit")
    ax1.set_xticks(x)
    ax1.set_xticklabels(PRODUCTS)
    ax1.set_ylabel("Choice share")
    ax1.set_title("Observed and Fitted Shares")
    ax1.legend(ncol=2)
    save_figure(fig1, "figures/choice-fit.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4.8))
    extent = [sigma_alpha_grid.min(), sigma_alpha_grid.max(), sigma_beta_grid.min(), sigma_beta_grid.max()]
    image = ax2.imshow(surface, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax2.scatter(theta_true[2], theta_true[3], color="white", edgecolor="black", label="True")
    ax2.scatter(np.asarray(mixed["theta"])[2], np.asarray(mixed["theta"])[3], color="tab:red", edgecolor="black", label="Estimate")
    ax2.set_xlabel("SD price taste")
    ax2.set_ylabel("SD quality taste")
    ax2.set_title("Profile Simulated Negative Log Likelihood")
    ax2.legend(loc="upper left")
    fig2.colorbar(image, ax=ax2, fraction=0.046, label="Distance from minimum")
    save_figure(fig2, "figures/heterogeneity-profile.png", dpi=150)

    fig3, ax3 = plt.subplots(figsize=(7.5, 4.8))
    keep = [j for j in range(len(PRODUCTS)) if j != removed_product]
    x_keep = np.arange(len(keep))
    ax3.bar(x_keep - width / 2, logit_diversion[keep], width, label="Plain logit")
    ax3.bar(x_keep + width / 2, mixed_diversion[keep], width, label="Mixed logit")
    ax3.set_xticks(x_keep)
    ax3.set_xticklabels([PRODUCTS[j] for j in keep])
    ax3.set_ylabel("Diversion from removed product")
    ax3.set_title(f"Remove {PRODUCTS[removed_product]}")
    ax3.legend()
    save_figure(fig3, "figures/substitution-patterns.png", dpi=150)

    fig4 = plot_price_substitution_matrices(substitution_matrices)
    save_figure(fig4, "figures/price-substitution-matrix.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    substitution_table.round(4).to_csv("tables/price-substitution-matrix.csv", index=False)
    parameter_table(theta_true, logit, mixed).to_csv("tables/parameter-recovery.csv", index=False)
    share_fit_table(observed, true_probs, logit_probs, mixed_probs).round(4).to_csv("tables/share-fit.csv", index=False)

    save_thumbnail("figures/choice-fit.png", "figures/thumb.png")
    print(f"\nDone: figures/ + tables/")


if __name__ == "__main__":
    main()
