#!/usr/bin/env python3
"""Deep learning policy approximation for deterministic optimal growth.

The model is the log Cobb-Douglas Brock-Mirman planner with full
depreciation. A tiny JAX neural network approximates the saving share and is
trained by minimizing squared Euler residuals on simulated capital states. The
closed-form policy provides a point-by-point audit.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

jax.config.update("jax_platform_name", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.plotting import setup_style, save_figure, save_thumbnail


ALPHA = 0.36
BETA = 0.95
A_TFP = 2.0

HIDDEN_UNITS = 16
TRAIN_STEPS = 6_000
BATCH_SIZE = 256
LEARNING_RATE = 2.5e-3
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPS = 1e-8

SHARE_MIN = 0.02
SHARE_MAX = 0.95


def production(k: jnp.ndarray) -> jnp.ndarray:
    """Cobb-Douglas output."""
    return A_TFP * jnp.maximum(k, 1e-10) ** ALPHA


def exact_policy(k: jnp.ndarray) -> jnp.ndarray:
    """Closed-form next-period capital."""
    return ALPHA * BETA * production(k)


def exact_consumption(k: jnp.ndarray) -> jnp.ndarray:
    """Closed-form consumption."""
    return (1.0 - ALPHA * BETA) * production(k)


def steady_state() -> tuple[float, float]:
    """Closed-form steady-state capital and consumption."""
    k_ss = (ALPHA * BETA * A_TFP) ** (1.0 / (1.0 - ALPHA))
    c_ss = (1.0 - ALPHA * BETA) * A_TFP * k_ss ** ALPHA
    return float(k_ss), float(c_ss)


def inverse_sigmoid(x: float) -> float:
    """Logit transform for a scalar probability."""
    return float(np.log(x / (1.0 - x)))


def init_params(key: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
    """Initialize a small tanh MLP for the saving share."""
    keys = random.split(key, 3)
    w1 = random.normal(keys[0], (1, HIDDEN_UNITS)) * np.sqrt(2.0 / (1 + HIDDEN_UNITS))
    b1 = jnp.zeros((HIDDEN_UNITS,))
    w2 = random.normal(keys[1], (HIDDEN_UNITS, HIDDEN_UNITS)) * np.sqrt(
        2.0 / (2 * HIDDEN_UNITS)
    )
    b2 = jnp.zeros((HIDDEN_UNITS,))
    w3 = random.normal(keys[2], (HIDDEN_UNITS, 1)) * 0.03
    start_share = 0.45
    start_prob = (start_share - SHARE_MIN) / (SHARE_MAX - SHARE_MIN)
    b3 = jnp.array([inverse_sigmoid(start_prob)])
    return [(w1, b1), (w2, b2), (w3, b3)]


def saving_share(
    params: list[tuple[jax.Array, jax.Array]], k: jax.Array, k_ss: float
) -> jax.Array:
    """Map capital into a feasible saving share."""
    x = jnp.log(jnp.maximum(k, 1e-10) / k_ss).reshape((-1, 1))
    w1, b1 = params[0]
    w2, b2 = params[1]
    w3, b3 = params[2]
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    raw = (h @ w3 + b3).reshape(k.shape)
    return SHARE_MIN + (SHARE_MAX - SHARE_MIN) * jax.nn.sigmoid(raw)


def neural_policy(
    params: list[tuple[jax.Array, jax.Array]], k: jax.Array, k_ss: float
) -> jax.Array:
    """Neural next-period capital policy."""
    return saving_share(params, k, k_ss) * production(k)


def neural_consumption(
    params: list[tuple[jax.Array, jax.Array]], k: jax.Array, k_ss: float
) -> jax.Array:
    """Consumption implied by the neural saving share."""
    return (1.0 - saving_share(params, k, k_ss)) * production(k)


def euler_log_residual(
    params: list[tuple[jax.Array, jax.Array]], k: jax.Array, k_ss: float
) -> jax.Array:
    """Log Euler residual; zero means the Euler equation holds exactly."""
    kp = neural_policy(params, k, k_ss)
    c = neural_consumption(params, k, k_ss)
    c_next = neural_consumption(params, kp, k_ss)
    mpk_next = ALPHA * A_TFP * jnp.maximum(kp, 1e-10) ** (ALPHA - 1.0)
    return jnp.log(BETA * mpk_next * c / c_next)


def loss_fn(
    params: list[tuple[jax.Array, jax.Array]],
    k_batch: jax.Array,
    k_ss: float,
    k_min: float,
    k_max: float,
) -> jax.Array:
    """Empirical risk over simulated states plus a light stability guard."""
    residual = euler_log_residual(params, k_batch, k_ss)
    kp = neural_policy(params, k_batch, k_ss)
    lower_guard = jax.nn.relu(0.5 * k_min - kp) / k_ss
    upper_guard = jax.nn.relu(kp - 1.15 * k_max) / k_ss
    return jnp.mean(residual**2) + 1e-3 * jnp.mean(lower_guard**2 + upper_guard**2)


@jax.jit
def adam_step(
    params: list[tuple[jax.Array, jax.Array]],
    m: list[tuple[jax.Array, jax.Array]],
    v: list[tuple[jax.Array, jax.Array]],
    key: jax.Array,
    step: jax.Array,
    k_ss: float,
    k_min: float,
    k_max: float,
) -> tuple[
    list[tuple[jax.Array, jax.Array]],
    list[tuple[jax.Array, jax.Array]],
    list[tuple[jax.Array, jax.Array]],
    jax.Array,
    jax.Array,
]:
    """One Adam update on a fresh batch of capital states."""
    key, batch_key = random.split(key)
    k_batch = random.uniform(batch_key, (BATCH_SIZE,), minval=k_min, maxval=k_max)
    loss, grads = jax.value_and_grad(loss_fn)(params, k_batch, k_ss, k_min, k_max)

    m = jax.tree_util.tree_map(
        lambda old, g: ADAM_BETA_1 * old + (1 - ADAM_BETA_1) * g,
        m,
        grads,
    )
    v = jax.tree_util.tree_map(
        lambda old, g: ADAM_BETA_2 * old + (1 - ADAM_BETA_2) * (g * g),
        v,
        grads,
    )
    step_float = step.astype(jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - ADAM_BETA_1**step_float), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - ADAM_BETA_2**step_float), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - LEARNING_RATE * mh / (jnp.sqrt(vh) + ADAM_EPS),
        params,
        m_hat,
        v_hat,
    )
    return params, m, v, key, loss


def train_policy(k_ss: float, k_min: float, k_max: float) -> tuple[
    list[tuple[jax.Array, jax.Array]], dict[str, np.ndarray | float]
]:
    """Train the neural policy on random capital draws."""
    key = random.PRNGKey(2026)
    params = init_params(key)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    log_steps: list[int] = []
    log_losses: list[float] = []
    log_policy_errors: list[float] = []
    audit_grid = jnp.linspace(k_min, k_max, 250)

    for step in range(1, TRAIN_STEPS + 1):
        params, m, v, key, loss = adam_step(
            params, m, v, key, jnp.array(step), k_ss, k_min, k_max
        )
        if step == 1 or step % 200 == 0:
            policy_error = jnp.mean(
                jnp.abs(neural_policy(params, audit_grid, k_ss) - exact_policy(audit_grid))
            )
            log_steps.append(step)
            log_losses.append(float(loss))
            log_policy_errors.append(float(policy_error))

    return params, {
        "steps": np.array(log_steps),
        "loss": np.array(log_losses),
        "mean_policy_error": np.array(log_policy_errors),
        "initial_loss": float(log_losses[0]),
        "initial_policy_error": float(log_policy_errors[0]),
        "final_loss": float(log_losses[-1]),
    }


def simulate_path(
    params: list[tuple[jax.Array, jax.Array]], k0: float, periods: int, k_ss: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate neural and exact capital paths from the same initial capital."""
    neural = np.empty(periods)
    exact = np.empty(periods)
    neural[0] = k0
    exact[0] = k0
    for t in range(periods - 1):
        neural[t + 1] = float(neural_policy(params, jnp.array([neural[t]]), k_ss)[0])
        exact[t + 1] = float(exact_policy(jnp.array([exact[t]]))[0])
    return neural, exact


def main() -> None:
    k_ss, c_ss = steady_state()
    k_min = 0.25 * k_ss
    k_max = 2.5 * k_ss
    train_interval_text = f"[{k_min:.3f}, {k_max:.3f}]"

    print("Training a JAX neural policy for deterministic optimal growth...")
    print(f"  exact steady state: k_ss={k_ss:.4f}, c_ss={c_ss:.4f}")
    print(f"  training interval: {train_interval_text}")
    params, train_log = train_policy(k_ss, k_min, k_max)

    k_grid = jnp.linspace(k_min, k_max, 400)
    k_grid_np = np.asarray(k_grid)
    neural_kp = np.asarray(neural_policy(params, k_grid, k_ss))
    exact_kp = np.asarray(exact_policy(k_grid))
    policy_error = neural_kp - exact_kp
    euler_residual = np.asarray(euler_log_residual(params, k_grid, k_ss))
    share_grid = np.asarray(saving_share(params, k_grid, k_ss))

    periods = 40
    k0 = 0.45 * k_ss
    neural_path, exact_path = simulate_path(params, k0, periods, k_ss)
    final_path_error = float(abs(neural_path[-1] - exact_path[-1]))

    mean_policy_error = float(np.mean(np.abs(policy_error)))
    max_policy_error = float(np.max(np.abs(policy_error)))
    max_euler_residual = float(np.max(np.abs(euler_residual)))
    mean_saving_share = float(share_grid.mean())
    exact_saving_share = ALPHA * BETA

    summary = pd.DataFrame([{
        "Initial loss": float(train_log["initial_loss"]),
        "Final loss": float(train_log["final_loss"]),
        "Mean policy error": mean_policy_error,
        "Max policy error": max_policy_error,
        "Max Euler residual": max_euler_residual,
        "Terminal path error": final_path_error,
        "Mean saving share": mean_saving_share,
        "Gradient steps": TRAIN_STEPS,
    }])

    print(f"  final loss: {summary.loc[0, 'Final loss']:.3e}")
    print(f"  max policy error: {summary.loc[0, 'Max policy error']:.3e}")
    print(f"  max log Euler residual: {summary.loc[0, 'Max Euler residual']:.3e}")

    setup_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_grid_np / k_ss, exact_kp / k_ss, label="Closed form", color="#1b5e20")
    ax.plot(k_grid_np / k_ss, neural_kp / k_ss, "--", label="Neural policy", color="#0d47a1")
    ax.axline((1.0, 1.0), slope=1.0, color="0.4", linewidth=1.0, linestyle=":", label="45-degree line")
    ax.set_xlabel("$k/k_{ss}$")
    ax.set_ylabel("$k'(k)/k_{ss}$")
    ax.set_title("Policy Function")
    ax.legend()
    save_figure(fig, "figures/policy-comparison.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    train_steps = np.asarray(train_log["steps"])
    train_losses = np.asarray(train_log["loss"])
    train_policy_errors = np.asarray(train_log["mean_policy_error"])
    axes[0].plot(train_steps, train_losses, color="#4e79a7", linewidth=2.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Gradient step")
    axes[0].set_ylabel("Euler-residual loss")
    axes[0].set_title("Training Loss")
    axes[1].plot(train_steps, train_policy_errors, color="#f28e2b", linewidth=2.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Gradient step")
    axes[1].set_ylabel("Mean policy error")
    axes[1].set_title("Policy Error")
    fig.tight_layout()
    save_figure(fig, "figures/training-curves.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(k_grid_np / k_ss, np.abs(euler_residual), color="#7b1fa2")
    axes[0].set_xlabel("$k/k_{ss}$")
    axes[0].set_ylabel("$|r(k;\\theta)|$")
    axes[0].set_title("Euler Residual")
    axes[1].plot(k_grid_np / k_ss, policy_error / k_ss, color="#bf360c")
    axes[1].axhline(0.0, color="0.4", linewidth=1.0)
    axes[1].set_xlabel("$k/k_{ss}$")
    axes[1].set_ylabel("$(k'_{NN}-k'_{exact})/k_{ss}$")
    axes[1].set_title("Policy Error")
    fig.tight_layout()
    save_figure(fig, "figures/euler-residuals.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    t_grid = np.arange(periods)
    ax.plot(t_grid, exact_path / k_ss, label="Closed form", color="#1b5e20")
    ax.plot(t_grid, neural_path / k_ss, "--", label="Neural policy", color="#0d47a1")
    ax.axhline(1.0, color="0.4", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Period")
    ax.set_ylabel("$k_t/k_{ss}$")
    ax.set_title("Simulated Capital Path")
    ax.legend()
    save_figure(fig, "figures/simulated-path.png", dpi=150)

    Path("tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv("tables/training-summary.csv", index=False)

    save_thumbnail("figures/policy-comparison.png", "figures/thumb.png")
    print("Done: 4 figures + tables/training-summary.csv")


if __name__ == "__main__":
    main()
