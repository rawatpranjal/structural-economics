#!/usr/bin/env python3
"""Mixed logit demand with simulated maximum likelihood."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


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
    report = ModelReport(
        "Mixed Logit Demand with Simulated Likelihood",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Consumers choose among differentiated products. The econometrician observes "
        "prices, qualities, and choices, but not each consumer's price sensitivity or "
        "quality taste.\n\n"
        "Plain logit compresses everyone into one representative taste vector. Mixed "
        "logit lets those coefficients vary across consumers. The price of that extra "
        "flexibility is numerical integration: each candidate parameter vector requires "
        "simulated choice probabilities.\n\n"
        "The key economic issue is substitution. A plain logit can match average shares "
        "and still say that all products are equally close substitutes after conditioning "
        "on their shares. Mixed logit keeps the logit formula for a simulated consumer "
        "with fixed tastes, then averages across consumers with different tastes."
    )

    report.add_equations(
        r"""
Consumer $i$ chooses product $j$ with utility

$$
u_{ij}
= \alpha_i p_{ij} + \beta_i q_{ij} + \varepsilon_{ij},
\qquad
\varepsilon_{ij}\sim\text{Type I EV}.
$$

Random coefficients are

$$
\alpha_i = \bar\alpha + \sigma_\alpha \nu_{i\alpha},
\qquad
\beta_i = \bar\beta + \sigma_\beta \nu_{i\beta},
\qquad
\nu_i\sim N(0,I).
$$

Conditional on a draw $\nu_r$, the logit probability is

$$
\begin{aligned}
P_{ij}(\theta,\nu_r)
&=
\frac{\exp(\alpha_r p_{ij}+\beta_r q_{ij})}
{\sum_{k\in\mathcal{J}} \exp(\alpha_r p_{ik}+\beta_r q_{ik})}.
\end{aligned}
$$

The mixed-logit probability integrates over random tastes. The code approximates
that integral with fixed simulation draws. The exact integral
$\int P_{ij}(\theta,\nu)\phi(\nu)d\nu$ has no closed form here because the
logit probability is nonlinear in the random coefficients. The code draws
$\nu_1,\ldots,\nu_R$ once and replaces the integral with the same finite
average at every candidate $\theta$:

$$
\begin{aligned}
\widehat P_{ij}(\theta)
&=
\frac{1}{R}\sum_{r=1}^R P_{ij}(\theta,\nu_r).
\end{aligned}
$$

Simulated maximum likelihood picks the parameter vector that assigns high
probability to the observed choices $y_i$ after averaging over simulated taste
heterogeneity:

$$
\begin{aligned}
\hat\theta
&=
\arg\max_\theta
\sum_{i=1}^N \log \widehat P_{i y_i}(\theta).
\end{aligned}
$$
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| Products | {len(PRODUCTS)} | Differentiated alternatives in each choice set |\n"
        f"| Choice occasions | {n_consumers:,} | Synthetic individual-level choices |\n"
        f"| Simulation draws | {n_draws:,} | Fixed normal draws for simulated likelihood |\n"
        f"| True $\\bar\\alpha$ | {theta_true[0]:.2f} | Mean price taste |\n"
        f"| True $\\bar\\beta$ | {theta_true[1]:.2f} | Mean quality taste |\n"
        f"| True $\\sigma_\\alpha$ | {theta_true[2]:.2f} | Heterogeneity in price sensitivity |\n"
        f"| True $\\sigma_\\beta$ | {theta_true[3]:.2f} | Heterogeneity in quality taste |\n\n"
        f"**Numerical settings**\n\n"
        f"| Setting | Value | Role |\n"
        f"|---------|-------|------|\n"
        f"| Mixed-logit optimizer | L-BFGS-B | Handles simple bounds on mean tastes and log standard deviations |\n"
        f"| Mixed-logit start | ({MIXED_START[0]:.2f}, {MIXED_START[1]:.2f}, "
        f"log {np.exp(MIXED_START[2]):.2f}, log {np.exp(MIXED_START[3]):.2f}) | Initial mean tastes and heterogeneity |\n"
        f"| Price-taste bound | [{MIXED_BOUNDS[0][0]:.2f}, {MIXED_BOUNDS[0][1]:.2f}] | Keeps price sensitivity negative |\n"
        f"| Quality-taste bound | [{MIXED_BOUNDS[1][0]:.2f}, {MIXED_BOUNDS[1][1]:.2f}] | Keeps quality taste positive |\n"
        f"| SD bounds | [{np.exp(MIXED_BOUNDS[2][0]):.2f}, {np.exp(MIXED_BOUNDS[2][1]):.2f}] | Applied to both random-coefficient standard deviations |\n"
        f"| Probability floor | {LIKELIHOOD_FLOOR:.0e} | Prevents log zero during likelihood evaluation |\n"
        f"| Max iterations | {MIXED_MAXITER} | L-BFGS-B iteration cap |\n"
        f"| Profile grid | {PROFILE_GRID_SIZE} x {PROFILE_GRID_SIZE} | Grid over $\\sigma_\\alpha$ and $\\sigma_\\beta$ for the likelihood surface |"
    )

    report.add_solution_method(
        r"""
The estimator uses common random numbers. Draws are made once and then held fixed while the optimizer moves $\theta$. This turns the population integral into the same finite average at every trial parameter vector. Without common draws, fresh simulation noise would move the likelihood surface while the optimizer is trying to climb it.

The standard deviations are optimized in logs. The optimizer can move freely over log standard deviations, while the model sees positive values after exponentiation. The bounds are not an economic restriction in this example. They keep the teaching likelihood away from numerically irrelevant regions.

### Algorithm 1. Simulated likelihood at a trial $\theta$

**Inputs.** Observed choices and characteristics $\{y_i,p_{ij},q_{ij}\}_{i=1,j=1}^{N,J}$, fixed draws $\nu_r=(\nu_{r\alpha},\nu_{r\beta})$ for $r=1,\ldots,R$, and a trial parameter vector $\theta=(\bar\alpha,\bar\beta,\ell_\alpha,\ell_\beta)$.

**Output.** The simulated objective $Q_R(\theta)$.

1. Convert log standard deviations into positive standard deviations:

$$
\sigma_\alpha=\exp(\ell_\alpha),
\qquad
\sigma_\beta=\exp(\ell_\beta).
$$

2. For each draw $r$, construct simulated tastes:

$$
\alpha_r=\bar\alpha+\sigma_\alpha\nu_{r\alpha},
\qquad
\beta_r=\bar\beta+\sigma_\beta\nu_{r\beta}.
$$

3. For each consumer-product pair $(i,j)$, compute the draw-specific logit probability:

$$
P_{ijr}(\theta)=
\frac{\exp(\alpha_r p_{ij}+\beta_r q_{ij})}
{\sum_{k=1}^J \exp(\alpha_r p_{ik}+\beta_r q_{ik})}.
$$

4. Average those probabilities over the fixed simulation draws:

$$
\widehat P_{ij}(\theta)=\frac{1}{R}\sum_{r=1}^R P_{ijr}(\theta).
$$

5. Score the observed choice $y_i$ with the simulated probability $\widehat P_{i y_i}(\theta)$.

6. Return the simulated log likelihood and the minimized objective:

$$
\ell_R(\theta)=
\sum_{i=1}^N \log \max\{\widehat P_{i y_i}(\theta),\eta\},
\qquad
Q_R(\theta)=-\ell_R(\theta)/N.
$$

### Algorithm 2. Optimization and price substitution

**Inputs.** Starting value $\theta_0$, bounds $B$, common draws $\{\nu_r\}_{r=1}^R$, data $\{y_i,p_{ij},q_{ij}\}$, and price step $\Delta p$.

**Outputs.** Estimate $\hat\theta$, fitted shares $\hat s_j$, and substitution matrix $D$.

1. Start L-BFGS-B at $\theta_0$ within bounds $B$.

2. At each candidate $\theta^m\in B$, evaluate $Q_R(\theta^m)$ using Algorithm 1.

3. Continue until the optimizer stops and set

$$
\hat\theta=\arg\min_{\theta\in B} Q_R(\theta),
$$

which is the same as maximizing $\ell_R(\theta)$.

4. Compute fitted shares from the estimated simulated probabilities:

$$
\hat s_j=\frac{1}{N}\sum_{i=1}^N \widehat P_{ij}(\hat\theta).
$$

5. For each shocked product $k$, raise $p_{ik}$ by $\Delta p$ for every consumer.

6. Recompute shares $\hat s_j^{+k}$ using the same $\hat\theta$ and the same draws $\nu_r$.

7. Fill column $k$ of the substitution matrix:

$$
D_{jk}=
\frac{\hat s_j^{+k}-\hat s_j}{\hat s_k-\hat s_k^{+k}}
\quad\text{for }j\neq k,
\qquad
D_{kk}=-1.
$$

8. Repeat steps 5-7 for every shocked product $k=1,\ldots,J$.

The homogeneous logit is estimated on the same data. Its likelihood is easier because it does not integrate over tastes. The comparison is useful because the homogeneous model can fit mean shares while still forcing diversion to follow existing market shares.
"""
    )

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
    report.add_results(
        "The mixed-logit fit tracks the product shares closely. The homogeneous logit "
        "also fits average shares reasonably well, so share fit alone is not enough to "
        "show why heterogeneity matters."
    )
    report.add_figure("figures/choice-fit.png", "Observed and fitted product shares", fig1)

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
    report.add_results(
        "The profiled likelihood is lowest near positive taste dispersion. Setting the "
        "standard deviations close to zero collapses the model toward plain logit and "
        "loses the substitution patterns generated by heterogeneous consumers."
    )
    report.add_figure("figures/heterogeneity-profile.png", "Profile likelihood over taste heterogeneity", fig2)

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
    report.add_results(
        f"When {PRODUCTS[removed_product]} is removed, the two models make different "
        "recapture predictions. Plain logit reallocates the lost demand according to "
        "average shares. Mixed logit moves more demand toward products that appeal to "
        "similar simulated consumers."
    )
    report.add_figure("figures/substitution-patterns.png", "Diversion after removing one product", fig3)

    report.add_results(
        "The price-substitution matrix asks where demand goes when one product becomes "
        f"{PRICE_SUBSTITUTION_STEP:.2f} price units more expensive. Each column is the "
        "product whose price is raised. Each off-diagonal entry is the share gain for "
        "the receiving product divided by the shocked product's lost share."
    )
    fig4 = plot_price_substitution_matrices(substitution_matrices)
    report.add_figure(
        "figures/price-substitution-matrix.png",
        "Price-substitution recapture matrix",
        fig4,
    )
    report.add_results(
        "Read each heatmap column as the product whose price is shocked and each row as "
        "the product receiving lost demand. The off-diagonal cells are recapture shares. "
        "The diagonal is labeled -1 because the shocked product is the losing product."
    )
    report.add_table(
        "tables/price-substitution-matrix.csv",
        "Plain and mixed logit price substitution matrix",
        substitution_table.round(4),
    )

    report.add_results(
        "The parameter and fit tables separate two diagnostics. The parameter table "
        "checks known-truth recovery. The share table checks whether the fitted model "
        "matches observed product choices."
    )
    report.add_table(
        "tables/parameter-recovery.csv",
        "Known-truth parameter recovery",
        parameter_table(theta_true, logit, mixed),
    )
    report.add_table(
        "tables/share-fit.csv",
        "Observed and fitted product shares",
        share_fit_table(observed, true_probs, logit_probs, mixed_probs).round(4),
    )

    report.add_takeaway(
        r"""
Mixed logit is a simulation estimator because choice probabilities require an integral over unobserved tastes. Fixed draws turn that integral into a smooth sample average. The payoff is economic: aggregate substitution is no longer forced to satisfy IIA, even though each simulated consumer still has a logit choice rule conditional on tastes.
"""
    )

    report.add_references(
        [
            "[Train, K. (2009). *Discrete Choice Methods with Simulation* (2nd ed.). Cambridge University Press.](https://eml.berkeley.edu/books/choice2.html)",
            "[McFadden, D., and Train, K. (2000). Mixed MNL Models for Discrete Response. *Journal of Applied Econometrics*, 15(5), 447-470.](https://doi.org/10.1002/1099-1255%28200009/10%2915:5%3C447::AID-JAE570%3E3.0.CO;2-1)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
