#!/usr/bin/env python3
"""Feedforward Neural Networks for Regression and Density Approximation.

One-hidden-layer tanh network fits a Cobb-Douglas production surface with
Gaussian noise. Trained by Adam via JAX autodiff with L2 weight decay.
Compared against linear and degree-3 polynomial baselines.
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import grad, jit, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H: int = 16
ALPHA_CD: float = 0.4
A_CD: float = 1.0
SIGMA_EPS: float = 0.05
N_TRAIN: int = 400
N_TEST: int = 1000
N_STEPS: int = 4000
LR: float = 0.01
LAMBDA_GRID: tuple[float, ...] = (0.0, 1e-3, 1e-2)
GRID_RES: int = 30
SEED: int = 0

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def cobb_douglas(k: jnp.ndarray, l: jnp.ndarray) -> jnp.ndarray:
    """Cobb-Douglas production function A * k^alpha * l^(1-alpha)."""
    return A_CD * (k ** ALPHA_CD) * (l ** (1.0 - ALPHA_CD))


def make_dataset(key: jax.Array, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw n uniform samples on [0.5, 2.5]^2 with additive Gaussian noise.

    Returns x of shape (n, 2) and y of shape (n,).
    """
    key_kl, key_noise = random.split(key)
    kl = random.uniform(key_kl, shape=(n, 2), minval=0.5, maxval=2.5)
    noise = SIGMA_EPS * random.normal(key_noise, shape=(n,))
    y = cobb_douglas(kl[:, 0], kl[:, 1]) + noise
    return kl, y


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def init_params(key: jax.Array, H: int) -> tuple[jnp.ndarray, ...]:
    """He-scaled weights for a (input=2, H, output=1) network.

    He (Kaiming) scaling: W ~ N(0, 2/fan_in). Originally derived for ReLU
    units; used here for tanh as a slightly heavier-tailed alternative to
    classical Xavier/Glorot (which uses 1/fan_in).
    """
    k1, k2, k3, k4 = random.split(key, 4)
    W = random.normal(k1, shape=(H, 2)) * jnp.sqrt(2.0 / 2.0)
    b = jnp.zeros((H,))
    W_out = random.normal(k3, shape=(1, H)) * jnp.sqrt(2.0 / H)
    b_out = jnp.zeros((1,))
    return (W, b, W_out, b_out)


def forward(params: tuple, x: jnp.ndarray) -> jnp.ndarray:
    """One hidden-layer tanh forward pass.

    x has shape (n, 2); returns y_hat of shape (n,).
    """
    W, b, W_out, b_out = params
    h = jnp.tanh(x @ W.T + b)
    y = h @ W_out.T + b_out
    return y[:, 0]


def loss_fn(params: tuple, x: jnp.ndarray, y: jnp.ndarray, lam: float) -> jnp.ndarray:
    """MSE plus L2 penalty on W and W_out. Biases are not penalised."""
    y_hat = forward(params, x)
    mse = jnp.mean((y - y_hat) ** 2)
    W, b, W_out, b_out = params
    reg = lam * (jnp.sum(W ** 2) + jnp.sum(W_out ** 2))
    return mse + reg


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

AdamState = tuple[tuple, tuple, int]


def adam_init(params: tuple) -> AdamState:
    """Initialise first and second moment accumulators to zero."""
    m = tuple(jnp.zeros_like(p) for p in params)
    v = tuple(jnp.zeros_like(p) for p in params)
    return (m, v, 0)


def adam_step(
    params: tuple,
    grads: tuple,
    state: AdamState,
    lr: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[tuple, AdamState]:
    """One bias-corrected Adam update. Returns (new_params, new_state)."""
    m, v, t = state
    t = t + 1
    new_m = tuple(b1 * mi + (1.0 - b1) * gi for mi, gi in zip(m, grads))
    new_v = tuple(b2 * vi + (1.0 - b2) * gi ** 2 for vi, gi in zip(v, grads))
    m_hat = tuple(mi / (1.0 - b1 ** t) for mi in new_m)
    v_hat = tuple(vi / (1.0 - b2 ** t) for vi in new_v)
    new_params = tuple(
        p - lr * mh / (jnp.sqrt(vh) + eps)
        for p, mh, vh in zip(params, m_hat, v_hat)
    )
    return new_params, (new_m, new_v, t)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


def train_adam(
    params: tuple,
    x: jnp.ndarray,
    y: jnp.ndarray,
    lam: float,
    n_steps: int,
    record_every: int = 50,
) -> tuple[tuple, jnp.ndarray]:
    """Train with Adam; return (final_params, recorded_loss_array)."""
    grad_fn = jit(grad(loss_fn))
    loss_jit = jit(loss_fn)
    state = adam_init(params)
    losses = []
    for step in range(n_steps):
        g = grad_fn(params, x, y, lam)
        params, state = adam_step(params, g, state, LR)
        if step % record_every == 0:
            losses.append(float(loss_jit(params, x, y, lam)))
    return params, jnp.array(losses)


def train_sgd(
    params: tuple,
    x: jnp.ndarray,
    y: jnp.ndarray,
    lam: float,
    n_steps: int,
    lr: float,
    record_every: int = 50,
) -> tuple[tuple, jnp.ndarray]:
    """Train with plain gradient descent (constant lr). Returns same shape as train_adam."""
    grad_fn = jit(grad(loss_fn))
    loss_jit = jit(loss_fn)
    losses = []
    for step in range(n_steps):
        g = grad_fn(params, x, y, lam)
        params = tuple(p - lr * gi for p, gi in zip(params, g))
        if step % record_every == 0:
            losses.append(float(loss_jit(params, x, y, lam)))
    return params, jnp.array(losses)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial features in two variables up to total degree `degree`."""
    k, l = x[:, 0], x[:, 1]
    cols = [np.ones(len(k))]
    for total in range(1, degree + 1):
        for pk in range(total + 1):
            pl = total - pk
            cols.append(k ** pk * l ** pl)
    return np.column_stack(cols)


def linear_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Linear regression on (1, k, l).

    Returns (coefficients, train_mse, test_mse).
    """
    X_tr = np.column_stack([np.ones(len(x_train)), x_train])
    X_te = np.column_stack([np.ones(len(x_test)), x_test])
    coef, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)
    train_mse = float(np.mean((y_train - X_tr @ coef) ** 2))
    test_mse = float(np.mean((y_test - X_te @ coef) ** 2))
    return coef, train_mse, test_mse


def polynomial_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    degree: int = 3,
) -> tuple[np.ndarray, float, float]:
    """Polynomial regression up to total degree `degree` in (k, l).

    Returns (coefficients, train_mse, test_mse).
    """
    X_tr = _poly_features(x_train, degree)
    X_te = _poly_features(x_test, degree)
    coef, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)
    train_mse = float(np.mean((y_train - X_tr @ coef) ** 2))
    test_mse = float(np.mean((y_test - X_te @ coef) ** 2))
    return coef, train_mse, test_mse


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _make_grid(res: int = GRID_RES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (K, L, Z_true) meshgrids over [0.5, 2.5]^2."""
    kv = np.linspace(0.5, 2.5, res)
    lv = np.linspace(0.5, 2.5, res)
    K, L = np.meshgrid(kv, lv)
    Z_true = A_CD * (K ** ALPHA_CD) * (L ** (1.0 - ALPHA_CD))
    return K, L, Z_true


def _eval_nn_grid(params: tuple, K: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Evaluate the trained network on a 2-D meshgrid."""
    pts = jnp.column_stack([K.ravel(), L.ravel()])
    z_hat = np.array(forward(params, pts))
    return z_hat.reshape(K.shape)


def _eval_poly_grid(coef: np.ndarray, K: np.ndarray, L: np.ndarray, degree: int = 3) -> np.ndarray:
    """Evaluate a fitted polynomial on a 2-D meshgrid."""
    pts = np.column_stack([K.ravel(), L.ravel()])
    X = _poly_features(pts, degree)
    return (X @ coef).reshape(K.shape)


def plot_fitted_surface(
    params_best: tuple,
    coef_lin: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    path: str,
) -> None:
    """Three-panel surface plot: true, NN fit, and residual."""
    K, L, Z_true = _make_grid()
    Z_nn = _eval_nn_grid(params_best, K, L)
    residual = Z_nn - Z_true

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={"projection": "3d"})
    vmin = float(Z_true.min())
    vmax = float(Z_true.max())

    for ax, Z, title in [
        (axes[0], Z_true, "True surface"),
        (axes[1], Z_nn,   "Neural network fit"),
        (axes[2], residual, "Residual (fit - true)"),
    ]:
        if title.startswith("Residual"):
            surf = ax.plot_surface(K, L, Z, cmap="RdBu_r", alpha=0.85)
        else:
            surf = ax.plot_surface(K, L, Z, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.85)
        ax.set_xlabel("k")
        ax.set_ylabel("l")
        ax.set_zlabel("y")
        ax.set_title(title)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08)

    fig.suptitle("Cobb-Douglas Surface: True vs Neural Network Fit", fontsize=13)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


def plot_bias_variance_vs_lambda(
    train_mses: list[float],
    test_mses: list[float],
    lin_train: float,
    lin_test: float,
    poly_train: float,
    poly_test: float,
    path: str,
) -> None:
    """In-sample and out-of-sample MSE vs weight-decay strength lambda."""
    lambdas = list(LAMBDA_GRID)
    x_pos = np.arange(len(lambdas))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_pos, train_mses, "o-", color="#4878CF", label="NN train MSE")
    ax.plot(x_pos, test_mses,  "s-", color="#D65F5F", label="NN test MSE")
    ax.axhline(lin_train,  color="#6ACC65", linestyle="--", linewidth=1.2, label="Linear train MSE")
    ax.axhline(lin_test,   color="#6ACC65", linestyle=":",  linewidth=1.2, label="Linear test MSE")
    ax.axhline(poly_train, color="#E07B39", linestyle="--", linewidth=1.2, label="Poly-3 train MSE")
    ax.axhline(poly_test,  color="#E07B39", linestyle=":",  linewidth=1.2, label="Poly-3 test MSE")
    noise_var = SIGMA_EPS ** 2
    ax.axhline(noise_var, color="gray", linestyle="-.", linewidth=1.0, label=f"Noise variance ({noise_var:.4f})")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(lam) for lam in lambdas])
    ax.set_xlabel("Weight decay lambda")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Bias-Variance Tradeoff vs Weight Decay")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


def plot_training_curves(
    curves_adam: dict[float, np.ndarray],
    curve_sgd: np.ndarray,
    n_steps: int,
    record_every: int,
    path: str,
) -> None:
    """Training loss vs steps for Adam (three lambdas) and plain SGD."""
    steps = np.arange(0, n_steps, record_every)
    colors = ["#4878CF", "#D65F5F", "#6ACC65"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (lam, curve), color in zip(curves_adam.items(), colors):
        ax.semilogy(steps, np.array(curve), color=color, label=f"Adam lambda={lam}")
    ax.semilogy(steps, np.array(curve_sgd), color="#E07B39", linestyle="--", label="SGD lambda=0")
    ax.axhline(SIGMA_EPS ** 2, color="gray", linestyle="-.", linewidth=1.0, label="Noise variance")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss (log scale)")
    ax.set_title("Training Curves: Adam vs Plain SGD")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, path, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_style()
    t0 = time.perf_counter()

    Path("figures").mkdir(parents=True, exist_ok=True)

    key_data = random.PRNGKey(0)
    key_net = random.PRNGKey(1)
    key_train, key_test = random.split(key_data, 2)

    x_train_jax, y_train_jax = make_dataset(key_train, N_TRAIN)
    x_test_jax,  y_test_jax  = make_dataset(key_test,  N_TEST)
    x_train = np.array(x_train_jax)
    y_train = np.array(y_train_jax)
    x_test  = np.array(x_test_jax)
    y_test  = np.array(y_test_jax)

    coef_lin, lin_train_mse, lin_test_mse = linear_baseline(x_train, y_train, x_test, y_test)
    coef_poly, poly_train_mse, poly_test_mse = polynomial_baseline(x_train, y_train, x_test, y_test)

    params_init = init_params(key_net, H)

    train_mses: list[float] = []
    test_mses:  list[float] = []
    curves_adam: dict[float, np.ndarray] = {}
    best_params = None
    best_test_mse = float("inf")

    record_every = 50
    for lam in LAMBDA_GRID:
        params_lam, curve = train_adam(
            params_init, x_train_jax, y_train_jax, lam, N_STEPS, record_every
        )
        y_hat_tr = np.array(forward(params_lam, x_train_jax))
        y_hat_te = np.array(forward(params_lam, x_test_jax))
        tr_mse = float(np.mean((y_train - y_hat_tr) ** 2))
        te_mse = float(np.mean((y_test  - y_hat_te) ** 2))
        train_mses.append(tr_mse)
        test_mses.append(te_mse)
        curves_adam[lam] = curve
        if te_mse < best_test_mse:
            best_test_mse = te_mse
            best_params = params_lam
        print(f"  lambda={lam:.0e}  train MSE={tr_mse:.5f}  test MSE={te_mse:.5f}")

    _, curve_sgd = train_sgd(
        params_init, x_train_jax, y_train_jax, 0.0, N_STEPS, LR, record_every
    )

    print()
    col = 14
    header = f"{'Model':<22} {'Train MSE':>{col}} {'Test MSE':>{col}}"
    print(header)
    print("-" * len(header))
    print(f"{'Linear':<22} {lin_train_mse:>{col}.5f} {lin_test_mse:>{col}.5f}")
    print(f"{'Polynomial deg-3':<22} {poly_train_mse:>{col}.5f} {poly_test_mse:>{col}.5f}")
    for lam, tr, te in zip(LAMBDA_GRID, train_mses, test_mses):
        label = f"NN lambda={lam:.0e}"
        print(f"{label:<22} {tr:>{col}.5f} {te:>{col}.5f}")
    noise_var = SIGMA_EPS ** 2
    print(f"\nNoise variance: {noise_var:.5f}")
    print()

    plot_fitted_surface(best_params, coef_lin, x_train, y_train, "figures/fitted-surface.png")
    plot_bias_variance_vs_lambda(
        train_mses, test_mses,
        lin_train_mse, lin_test_mse,
        poly_train_mse, poly_test_mse,
        "figures/bias-variance-vs-lambda.png",
    )
    plot_training_curves(curves_adam, curve_sgd, N_STEPS, record_every, "figures/training-curves.png")
    save_thumbnail("figures/fitted-surface.png", "figures/thumb.png")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s. Figures written to figures/")


if __name__ == "__main__":
    main()
