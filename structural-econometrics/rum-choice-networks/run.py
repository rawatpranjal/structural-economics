#!/usr/bin/env python3
"""Choice prediction with a small RUMnet-style estimator.

The tutorial simulates product choices from a nonlinear random-utility model,
then compares a plain logit with a neural utility model that averages over
fixed latent taste draws. It follows the RUMnet idea in Aouad and Desir (2023)
without adding a deep-learning dependency.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import save_figure, save_thumbnail, setup_style


PRODUCTS = ["Saver", "Standard", "Premium"]
N_PRODUCTS = len(PRODUCTS)
N_TRAIN = 3_000
N_TEST = 1_500
N_LATENT_DRAWS = 9
HIDDEN_UNITS = 6
N_FEATURES = 12
RUMNET_L2 = 1.2e-2
LEARNING_CURVE_SIZES = (300, 600, 1_200, N_TRAIN)
PRICE_SHOCK_PRODUCT = 2
PRICE_SHOCK = 0.25
SEED = 20260508


@dataclass
class ChoiceData:
    """Choice occasions with product-level attributes."""

    price: np.ndarray
    quality: np.ndarray
    context: np.ndarray
    choice: np.ndarray


def observation_count(data: ChoiceData) -> int:
    """Return the number of choice occasions in a sample."""
    return int(data.choice.shape[0])


def subset_data(data: ChoiceData, n_obs: int) -> ChoiceData:
    """Return the first n observations from a choice sample."""
    return ChoiceData(
        price=data.price[:n_obs].copy(),
        quality=data.quality[:n_obs].copy(),
        context=data.context[:n_obs].copy(),
        choice=data.choice[:n_obs].copy(),
    )


def softmax_np(utility: np.ndarray) -> np.ndarray:
    """Stable softmax along the product axis."""
    shifted = utility - utility.max(axis=-1, keepdims=True)
    exp_utility = np.exp(shifted)
    return exp_utility / exp_utility.sum(axis=-1, keepdims=True)


def systematic_utility(
    price: np.ndarray,
    quality: np.ndarray,
    context: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    """Nonlinear random utility used to generate synthetic choices."""
    intercept = np.array([0.0, 0.12, -0.02])
    context_tilt = np.tanh(1.20 * context)
    context_hump = context ** 2 - 1.0
    product_context = np.column_stack([
        -0.35 * context_tilt + 0.10 * context_hump,
        0.10 * context_tilt - 0.08 * context_hump,
        0.55 * context_tilt - 0.28 * context_hump,
    ])
    price_taste = -(
        1.05
        + 0.22 * np.tanh(context)
        + 0.12 * eta
        + 0.08 * eta * np.tanh(context)
    )
    quality_taste = 0.78 + 0.30 * np.tanh(1.10 * context + 0.45 * eta)
    nonlinear_quality = (
        np.array([-0.15, 0.20, 0.62])[None, :]
        * np.tanh(1.15 * (quality - 1.25) * context[:, None])
    )
    latent_match = 0.18 * eta[:, None] * np.tanh(quality * context[:, None])
    premium_switch = np.column_stack([
        np.zeros_like(context),
        0.12 * np.tanh(1.80 * context) ** 2,
        0.35 * np.tanh(1.50 * context) ** 2,
    ])
    return (
        intercept
        + price_taste[:, None] * price
        + quality_taste[:, None] * quality
        + product_context
        + nonlinear_quality
        + latent_match
        + premium_switch
    )


def simulate_choices(seed: int, n_obs: int) -> ChoiceData:
    """Simulate product attributes, context, latent tastes, and choices."""
    rng = np.random.default_rng(seed)
    context = rng.normal(size=n_obs)
    base_price = np.array([1.75, 2.45, 3.35])
    base_quality = np.array([0.75, 1.45, 2.35])
    context_price_slope = np.array([-1.00, 0.15, 0.55])
    context_quality_slope = np.array([0.05, 0.45, 0.95])

    price = np.maximum(
        0.60,
        base_price
        + 0.10 * context[:, None] * context_price_slope
        + rng.normal(0.0, 0.18, size=(n_obs, N_PRODUCTS)),
    )
    quality = (
        base_quality
        + 0.12 * context[:, None] * context_quality_slope
        + rng.normal(0.0, 0.22, size=(n_obs, N_PRODUCTS))
    )
    eta = rng.normal(size=n_obs)
    utility = systematic_utility(price, quality, context, eta)
    choice = (utility + rng.gumbel(size=utility.shape)).argmax(axis=1)

    return ChoiceData(
        price=price.astype(np.float32),
        quality=quality.astype(np.float32),
        context=context.astype(np.float32),
        choice=choice.astype(np.int32),
    )


def true_probabilities(data: ChoiceData, eta_draws: np.ndarray) -> np.ndarray:
    """Approximate true choice probabilities under the simulated RUM."""
    probabilities = []
    for eta in eta_draws:
        eta_vec = np.full(observation_count(data), eta, dtype=float)
        utility = systematic_utility(data.price, data.quality, data.context, eta_vec)
        probabilities.append(softmax_np(utility))
    return np.mean(probabilities, axis=0)


def logit_features(data: ChoiceData) -> np.ndarray:
    """Product features used by the plain logit baseline."""
    z = data.context[:, None]
    return np.stack(
        [
            data.price,
            data.quality,
            data.price * z,
            data.quality * z,
        ],
        axis=-1,
    ).astype(np.float32)


def unpack_logit(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return product intercepts and feature slopes for the baseline."""
    intercept = jnp.concatenate([jnp.zeros(1), theta[: N_PRODUCTS - 1]])
    slopes = theta[N_PRODUCTS - 1 :]
    return intercept, slopes


def logit_nll(theta: jnp.ndarray, features: jnp.ndarray, choice: jnp.ndarray) -> jnp.ndarray:
    """Average negative log likelihood for plain logit."""
    intercept, slopes = unpack_logit(theta)
    utility = intercept + jnp.einsum("njd,d->nj", features, slopes)
    chosen = utility[jnp.arange(choice.shape[0]), choice]
    return -jnp.mean(chosen - jax.nn.logsumexp(utility, axis=1))


LOGIT_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(logit_nll))


def estimate_logit(data: ChoiceData) -> dict[str, object]:
    """Estimate the plain logit baseline by maximum likelihood."""
    features = jnp.asarray(logit_features(data))
    choice = jnp.asarray(data.choice)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = LOGIT_VALUE_AND_GRAD(jnp.asarray(theta, dtype=jnp.float32), features, choice)
        return float(value), np.asarray(gradient, dtype=float)

    result = minimize(
        objective,
        np.zeros((N_PRODUCTS - 1) + 4, dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-9},
    )
    return {
        "theta": np.asarray(result.x, dtype=float),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "success": bool(result.success),
    }


def rum_features(data: ChoiceData, latent_draws: np.ndarray) -> np.ndarray:
    """Features entering the RUMnet utility for each latent draw."""
    n_obs = observation_count(data)
    n_draws = len(latent_draws)
    price = np.broadcast_to(data.price[:, None, :, None], (n_obs, n_draws, N_PRODUCTS, 1))
    quality = np.broadcast_to(data.quality[:, None, :, None], (n_obs, n_draws, N_PRODUCTS, 1))
    context = np.broadcast_to(data.context[:, None, None, None], (n_obs, n_draws, N_PRODUCTS, 1))
    draw = np.broadcast_to(latent_draws[None, :, None, None], (n_obs, n_draws, N_PRODUCTS, 1))
    context_sq = context * context
    return np.concatenate(
        [
            price,
            quality,
            context,
            price * context,
            quality * context,
            context_sq,
            quality * context_sq,
            draw,
            price * draw,
            quality * draw,
            context * draw,
            quality * context * draw,
        ],
        axis=-1,
    ).astype(np.float32)


def unpack_rumnet(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return RUMnet intercept, linear, hidden-layer, and output parameters."""
    cursor = 0
    intercept = jnp.concatenate([jnp.zeros(1), theta[cursor : cursor + N_PRODUCTS - 1]])
    cursor += N_PRODUCTS - 1
    slopes = theta[cursor : cursor + 4]
    cursor += 4
    hidden_weights = theta[cursor : cursor + N_FEATURES * HIDDEN_UNITS].reshape(N_FEATURES, HIDDEN_UNITS)
    cursor += N_FEATURES * HIDDEN_UNITS
    hidden_bias = theta[cursor : cursor + HIDDEN_UNITS]
    cursor += HIDDEN_UNITS
    output_weights = theta[cursor : cursor + HIDDEN_UNITS]
    return intercept, slopes, hidden_weights, hidden_bias, output_weights


def rumnet_nll(theta: jnp.ndarray, features: jnp.ndarray, choice: jnp.ndarray) -> jnp.ndarray:
    """Average negative simulated log likelihood for the RUMnet."""
    intercept, slopes, hidden_weights, hidden_bias, output_weights = unpack_rumnet(theta)
    linear_utility = intercept + jnp.einsum("nkjd,d->nkj", features[:, :, :, [0, 1, 3, 4]], slopes)
    hidden = jnp.tanh(jnp.einsum("nkjd,dh->nkjh", features, hidden_weights) + hidden_bias)
    utility = linear_utility + jnp.einsum("nkjh,h->nkj", hidden, output_weights)
    log_probability = utility - jax.nn.logsumexp(utility, axis=2, keepdims=True)
    chosen_log_probability = log_probability[
        jnp.arange(choice.shape[0])[:, None],
        jnp.arange(features.shape[1])[None, :],
        choice[:, None],
    ]
    probability = jnp.mean(jnp.exp(chosen_log_probability), axis=1)
    penalty = RUMNET_L2 * jnp.mean(theta[(N_PRODUCTS - 1) + 4 :] ** 2)
    return -jnp.mean(jnp.log(jnp.maximum(probability, 1e-12))) + penalty


RUMNET_VALUE_AND_GRAD = jax.jit(jax.value_and_grad(rumnet_nll))


def initial_rumnet_parameters(logit_theta: np.ndarray, seed: int, scale: float) -> np.ndarray:
    """Start the RUMnet near the logit estimate with a small neural component."""
    rng = np.random.default_rng(seed)
    n_parameters = (
        (N_PRODUCTS - 1)
        + 4
        + N_FEATURES * HIDDEN_UNITS
        + HIDDEN_UNITS
        + HIDDEN_UNITS
    )
    theta = np.zeros(n_parameters, dtype=float)
    theta[: (N_PRODUCTS - 1) + 4] = logit_theta
    cursor = (N_PRODUCTS - 1) + 4
    theta[cursor : cursor + N_FEATURES * HIDDEN_UNITS] = rng.normal(
        0.0,
        scale,
        size=N_FEATURES * HIDDEN_UNITS,
    )
    cursor += N_FEATURES * HIDDEN_UNITS + HIDDEN_UNITS
    theta[cursor : cursor + HIDDEN_UNITS] = rng.normal(0.0, scale, size=HIDDEN_UNITS)
    return theta


def estimate_rumnet(data: ChoiceData, logit_theta: np.ndarray, latent_draws: np.ndarray) -> dict[str, object]:
    """Estimate the RUMnet by simulated maximum likelihood."""
    features = jnp.asarray(rum_features(data, latent_draws))
    choice = jnp.asarray(data.choice)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = RUMNET_VALUE_AND_GRAD(jnp.asarray(theta, dtype=jnp.float32), features, choice)
        return float(value), np.asarray(gradient, dtype=float)

    starts = [
        initial_rumnet_parameters(logit_theta, seed=123, scale=0.04),
        initial_rumnet_parameters(logit_theta, seed=123, scale=0.08),
    ]
    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 120, "ftol": 1e-8, "maxls": 30},
        )
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None
    return {
        "theta": np.asarray(best.x, dtype=float),
        "objective": float(best.fun),
        "iterations": int(best.nit),
        "success": bool(best.success),
    }


def logit_probabilities(model: dict[str, object], data: ChoiceData) -> np.ndarray:
    """Choice probabilities from the fitted plain logit."""
    theta = np.asarray(model["theta"], dtype=float)
    intercept = np.r_[0.0, theta[: N_PRODUCTS - 1]]
    slopes = theta[N_PRODUCTS - 1 :]
    utility = intercept + np.einsum("njd,d->nj", logit_features(data), slopes)
    return softmax_np(utility)


def rumnet_probabilities(
    model: dict[str, object],
    data: ChoiceData,
    latent_draws: np.ndarray,
) -> np.ndarray:
    """Choice probabilities from the fitted RUMnet."""
    theta = np.asarray(model["theta"], dtype=float)
    cursor = 0
    intercept = np.r_[0.0, theta[cursor : cursor + N_PRODUCTS - 1]]
    cursor += N_PRODUCTS - 1
    slopes = theta[cursor : cursor + 4]
    cursor += 4
    hidden_weights = theta[cursor : cursor + N_FEATURES * HIDDEN_UNITS].reshape(N_FEATURES, HIDDEN_UNITS)
    cursor += N_FEATURES * HIDDEN_UNITS
    hidden_bias = theta[cursor : cursor + HIDDEN_UNITS]
    cursor += HIDDEN_UNITS
    output_weights = theta[cursor : cursor + HIDDEN_UNITS]

    features = rum_features(data, latent_draws)
    linear_utility = intercept + np.einsum("nkjd,d->nkj", features[:, :, :, [0, 1, 3, 4]], slopes)
    hidden = np.tanh(np.einsum("nkjd,dh->nkjh", features, hidden_weights) + hidden_bias)
    utility = linear_utility + np.einsum("nkjh,h->nkj", hidden, output_weights)
    return softmax_np(utility).mean(axis=1)


def negative_log_likelihood(probabilities: np.ndarray, choice: np.ndarray) -> float:
    """Average negative log probability of observed choices."""
    chosen = probabilities[np.arange(len(choice)), choice]
    return float(-np.log(np.maximum(chosen, 1e-12)).mean())


def accuracy(probabilities: np.ndarray, choice: np.ndarray) -> float:
    """Share of choices predicted by the largest fitted probability."""
    return float((probabilities.argmax(axis=1) == choice).mean())


def observed_shares(data: ChoiceData) -> np.ndarray:
    """Observed product shares."""
    return np.bincount(data.choice, minlength=N_PRODUCTS) / observation_count(data)


def prediction_summary(
    train: ChoiceData,
    test: ChoiceData,
    logit_model: dict[str, object],
    rumnet_model: dict[str, object],
    latent_draws: np.ndarray,
) -> pd.DataFrame:
    """Compare fit on train and test samples."""
    rows = []
    for model_name, probability_function in [
        ("Plain logit", lambda sample: logit_probabilities(logit_model, sample)),
        ("RUMnet", lambda sample: rumnet_probabilities(rumnet_model, sample, latent_draws)),
    ]:
        train_prob = probability_function(train)
        test_prob = probability_function(test)
        rows.append({
            "Model": model_name,
            "Train NLL": negative_log_likelihood(train_prob, train.choice),
            "Test NLL": negative_log_likelihood(test_prob, test.choice),
            "Test accuracy": accuracy(test_prob, test.choice),
        })
    return pd.DataFrame(rows).round(4)


def share_fit_table(
    test: ChoiceData,
    logit_prob: np.ndarray,
    rumnet_prob: np.ndarray,
) -> pd.DataFrame:
    """Observed and predicted product shares on the test sample."""
    rows = []
    observed = observed_shares(test)
    for product_id, product in enumerate(PRODUCTS):
        rows.append({
            "Product": product,
            "Observed share": observed[product_id],
            "Plain logit": logit_prob[:, product_id].mean(),
            "RUMnet": rumnet_prob[:, product_id].mean(),
        })
    return pd.DataFrame(rows).round(4)


def apply_price_shock(data: ChoiceData, product_id: int, shock: float) -> ChoiceData:
    """Return a copy with one product's price increased."""
    price = data.price.copy()
    price[:, product_id] += shock
    return ChoiceData(
        price=price.astype(np.float32),
        quality=data.quality.copy(),
        context=data.context.copy(),
        choice=data.choice.copy(),
    )


def recapture_table(
    test: ChoiceData,
    logit_model: dict[str, object],
    rumnet_model: dict[str, object],
    latent_draws: np.ndarray,
) -> pd.DataFrame:
    """Recapture rates after raising the Premium price."""
    shocked = apply_price_shock(test, PRICE_SHOCK_PRODUCT, PRICE_SHOCK)
    rows = []
    for model_name, probability_function in [
        ("Plain logit", lambda sample: logit_probabilities(logit_model, sample)),
        ("RUMnet", lambda sample: rumnet_probabilities(rumnet_model, sample, latent_draws)),
    ]:
        base_shares = probability_function(test).mean(axis=0)
        shocked_shares = probability_function(shocked).mean(axis=0)
        lost_share = base_shares[PRICE_SHOCK_PRODUCT] - shocked_shares[PRICE_SHOCK_PRODUCT]
        for product_id, product in enumerate(PRODUCTS):
            if product_id == PRICE_SHOCK_PRODUCT:
                recapture = -1.0
            else:
                recapture = (shocked_shares[product_id] - base_shares[product_id]) / lost_share
            rows.append({
                "Model": model_name,
                "Product": product,
                "Base share": base_shares[product_id],
                "After shock": shocked_shares[product_id],
                "Recapture": recapture,
            })
    return pd.DataFrame(rows).round(4)


def learning_curve_table(
    train: ChoiceData,
    test: ChoiceData,
    logit_model: dict[str, object],
    rumnet_model: dict[str, object],
    latent_draws: np.ndarray,
) -> pd.DataFrame:
    """Estimate both models on increasing sample sizes and evaluate on test data."""
    rows = []
    full_n = observation_count(train)
    for n_obs in LEARNING_CURVE_SIZES:
        if n_obs == full_n:
            logit_n = logit_model
            rumnet_n = rumnet_model
        else:
            train_n = subset_data(train, n_obs)
            logit_n = estimate_logit(train_n)
            rumnet_n = estimate_rumnet(train_n, np.asarray(logit_n["theta"]), latent_draws)

        for model_name, probabilities in [
            ("Plain logit", logit_probabilities(logit_n, test)),
            ("RUMnet", rumnet_probabilities(rumnet_n, test, latent_draws)),
        ]:
            rows.append({
                "Training choices": n_obs,
                "Model": model_name,
                "Test NLL": negative_log_likelihood(probabilities, test.choice),
                "Test accuracy": accuracy(probabilities, test.choice),
            })
    return pd.DataFrame(rows).round(4)


def representative_context_panel(context_grid: np.ndarray) -> ChoiceData:
    """Build a representative choice panel varying only the customer context."""
    n_obs = len(context_grid)
    base_price = np.array([1.75, 2.45, 3.35], dtype=np.float32)
    base_quality = np.array([0.75, 1.45, 2.35], dtype=np.float32)
    price = np.broadcast_to(base_price[None, :], (n_obs, N_PRODUCTS)).copy()
    quality = np.broadcast_to(base_quality[None, :], (n_obs, N_PRODUCTS)).copy()
    return ChoiceData(
        price=price,
        quality=quality,
        context=context_grid.astype(np.float32),
        choice=np.zeros(n_obs, dtype=np.int32),
    )


def plot_share_fit(share_table: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing observed and predicted test shares."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(N_PRODUCTS)
    width = 0.25
    ax.bar(x - width, share_table["Observed share"], width, label="Observed", color="#4c78a8")
    ax.bar(x, share_table["Plain logit"], width, label="Plain logit", color="#f58518")
    ax.bar(x + width, share_table["RUMnet"], width, label="RUMnet", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(PRODUCTS)
    ax.set_ylabel("Share")
    ax.set_title("Choice Shares in the Test Sample")
    ax.legend(frameon=False)
    ax.set_ylim(0.0, max(0.55, float(share_table[["Observed share", "Plain logit", "RUMnet"]].to_numpy().max()) + 0.08))
    return fig


def plot_likelihoods(fit_table: pd.DataFrame) -> plt.Figure:
    """Grouped bars for train and test negative log likelihood."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(fit_table))
    width = 0.32
    ax.bar(x - width / 2, fit_table["Train NLL"], width, label="Train", color="#4c78a8")
    ax.bar(x + width / 2, fit_table["Test NLL"], width, label="Test", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(fit_table["Model"])
    ax.set_ylabel("Negative log likelihood")
    ax.set_title("Held-Out Fit")
    ax.legend(frameon=False)
    return fig


def plot_learning_curve(learning_curve: pd.DataFrame) -> plt.Figure:
    """Held-out likelihood as the training sample grows."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for model_name, color, marker in [
        ("Plain logit", "#f58518", "o"),
        ("RUMnet", "#54a24b", "s"),
    ]:
        series = learning_curve[learning_curve["Model"] == model_name]
        ax.plot(
            series["Training choices"],
            series["Test NLL"],
            label=model_name,
            color=color,
            marker=marker,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training choices")
    ax.set_ylabel("Test negative log likelihood")
    ax.set_title("Learning Curve")
    ax.legend(frameon=False)
    return fig


def plot_recapture(recapture: pd.DataFrame) -> plt.Figure:
    """Recapture shares after increasing Premium's price."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    receiving = [product for product in PRODUCTS if product != PRODUCTS[PRICE_SHOCK_PRODUCT]]
    x = np.arange(len(receiving))
    width = 0.32
    for offset, model_name, color in [
        (-width / 2, "Plain logit", "#f58518"),
        (width / 2, "RUMnet", "#54a24b"),
    ]:
        values = [
            float(recapture[(recapture["Model"] == model_name) & (recapture["Product"] == product)]["Recapture"].iloc[0])
            for product in receiving
        ]
        ax.bar(x + offset, values, width, label=model_name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(receiving)
    ax.set_ylabel("Recapture share")
    ax.set_title(f"Where Demand Goes After a {PRICE_SHOCK:.2f} Premium Price Increase")
    ax.legend(frameon=False)
    ax.set_ylim(0.0, 1.0)
    return fig


def plot_context_curve(
    logit_model: dict[str, object],
    rumnet_model: dict[str, object],
    latent_draws: np.ndarray,
    truth_draws: np.ndarray,
) -> plt.Figure:
    """Premium choice probability as customer context varies."""
    context_grid = np.linspace(-2.2, 2.2, 120)
    panel = representative_context_panel(context_grid)
    true_prob = true_probabilities(panel, truth_draws)[:, PRICE_SHOCK_PRODUCT]
    logit_prob = logit_probabilities(logit_model, panel)[:, PRICE_SHOCK_PRODUCT]
    rumnet_prob = rumnet_probabilities(rumnet_model, panel, latent_draws)[:, PRICE_SHOCK_PRODUCT]

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(context_grid, true_prob, label="True RUM", color="#4c78a8")
    ax.plot(context_grid, logit_prob, label="Plain logit", color="#f58518", linestyle="--")
    ax.plot(context_grid, rumnet_prob, label="RUMnet", color="#54a24b")
    ax.set_xlabel("Customer context")
    ax.set_ylabel("Premium probability")
    ax.set_title("Context Changes Premium Demand")
    ax.legend(frameon=False)
    ax.set_ylim(0.0, 1.0)
    return fig


def main() -> None:
    """Run the tutorial and regenerate all outputs."""
    setup_style()
    train = simulate_choices(SEED, N_TRAIN)
    test = simulate_choices(SEED + 1, N_TEST)
    latent_draws = norm.ppf((np.arange(N_LATENT_DRAWS) + 0.5) / N_LATENT_DRAWS).astype(np.float32)
    truth_draws = norm.ppf((np.arange(41) + 0.5) / 41).astype(np.float32)

    logit_model = estimate_logit(train)
    rumnet_model = estimate_rumnet(train, np.asarray(logit_model["theta"]), latent_draws)
    logit_test_prob = logit_probabilities(logit_model, test)
    rumnet_test_prob = rumnet_probabilities(rumnet_model, test, latent_draws)

    fit_table = prediction_summary(train, test, logit_model, rumnet_model, latent_draws)
    shares = share_fit_table(test, logit_test_prob, rumnet_test_prob)
    recapture = recapture_table(test, logit_model, rumnet_model, latent_draws)
    learning_curve = learning_curve_table(train, test, logit_model, rumnet_model, latent_draws)

    print("RUMnet choice tutorial")
    print(f"  Train choices: {observation_count(train)}")
    print(f"  Test choices: {observation_count(test)}")
    print(f"  Logit test NLL: {fit_table.loc[fit_table['Model'] == 'Plain logit', 'Test NLL'].iloc[0]:.4f}")
    print(f"  RUMnet test NLL: {fit_table.loc[fit_table['Model'] == 'RUMnet', 'Test NLL'].iloc[0]:.4f}")
    print(f"  Latent draws: {N_LATENT_DRAWS}")

    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("tables").mkdir(parents=True, exist_ok=True)

    save_figure(plot_share_fit(shares), "figures/choice-fit.png", dpi=150)
    save_figure(plot_likelihoods(fit_table), "figures/likelihood-comparison.png", dpi=150)
    save_figure(plot_learning_curve(learning_curve), "figures/learning-curve.png", dpi=150)
    save_figure(plot_recapture(recapture), "figures/premium-price-shock.png", dpi=150)
    save_figure(
        plot_context_curve(logit_model, rumnet_model, latent_draws, truth_draws),
        "figures/context-curve.png",
        dpi=150,
    )

    fit_table.to_csv("tables/fit-comparison.csv", index=False)
    shares.to_csv("tables/test-share-fit.csv", index=False)
    recapture.to_csv("tables/premium-price-shock.csv", index=False)
    learning_curve.to_csv("tables/learning-curve.csv", index=False)

    save_thumbnail("figures/choice-fit.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
