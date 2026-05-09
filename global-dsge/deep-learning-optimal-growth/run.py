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
from lib.output import ModelReport
from lib.plotting import setup_style


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
        "Final loss": float(train_log["final_loss"]),
        "Mean policy error": mean_policy_error,
        "Max policy error": max_policy_error,
        "Max Euler residual": max_euler_residual,
        "Terminal path error": final_path_error,
        "Gradient steps": TRAIN_STEPS,
    }])

    print(f"  final loss: {summary.loc[0, 'Final loss']:.3e}")
    print(f"  max policy error: {summary.loc[0, 'Max policy error']:.3e}")
    print(f"  max log Euler residual: {summary.loc[0, 'Max Euler residual']:.3e}")

    setup_style()

    report = ModelReport(
        "Deep Learning for Optimal Growth",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A planner chooses how much output to consume today. "
        "Remaining output becomes capital tomorrow. "
        "The log Cobb-Douglas case has an exact saving rule. "
        "That exact rule makes the neural solver easy to audit.\n\n"
        "Deep-learning macro methods often rewrite dynamic models as empirical-risk problems. "
        "Some solvers optimize lifetime rewards. "
        "Some solvers minimize Euler residuals. "
        "Some solvers minimize Bellman residuals.\n\n"
        "This tutorial uses an Euler-residual loss. "
        "The neural net proposes a feasible saving share. "
        "Gradient steps fit the policy on simulated capital states. "
        "The Brock-Mirman formula audits the trained rule point by point."
    )

    report.add_equations(
        r"""
Capital fully depreciates each period. With state $k_t$, output is

$$
y_t = A k_t^{\alpha}, \qquad 0<\alpha<1,
$$

Feasibility requires

$$
c_t + k_{t+1} = A k_t^{\alpha},
\qquad c_t>0, k_{t+1}>0.
$$

The planner maximizes

$$
\sum_{t=0}^{\infty}\beta^t \log c_t,
\qquad 0<\beta<1.
$$

The Euler equation is

$$
\frac{1}{c_t}
= \beta \frac{\alpha A k_{t+1}^{\alpha-1}}{c_{t+1}}.
$$

The code evaluates the Euler equation as this log residual:

$$
r(k;\theta)
= \log\left[
\beta \alpha A k'(k;\theta)^{\alpha-1}
\frac{c(k;\theta)}{c(k'(k;\theta);\theta)}
\right].
$$

The residual is zero when the Euler equation holds. Training chooses parameters that make this residual small.

The population risk is

$$
\Xi(\theta) = E\left[r(k;\theta)^2\right].
$$

The program replaces that expectation with simulated capital draws. With draws $k_1,\ldots,k_n$, it solves the empirical problem

$$
\Xi_n(\theta) = \frac{1}{n}\sum_{i=1}^{n} r(k_i;\theta)^2,
\qquad
\hat{\theta} = \arg\min_{\theta} \Xi_n(\theta).
$$

The neural policy first chooses a saving share:

$$
s(k;\theta)
= s_{\min} + (s_{\max}-s_{\min})
\sigma\left(N_{\theta}(\log(k/k_{ss}))\right),
$$

It then imposes feasibility by construction:

$$
k'(k;\theta)=s(k;\theta)A k^{\alpha},
\qquad
c(k;\theta)=(1-s(k;\theta))A k^{\alpha}.
$$

This special case has an exact policy:

$$
k'(k)=\alpha\beta A k^{\alpha},
\qquad
c(k)=(1-\alpha\beta)A k^{\alpha},
$$

The steady state is

$$
k_{ss}=(\alpha\beta A)^{1/(1-\alpha)}.
$$
"""
    )

    report.add_model_setup(
        "| Symbol | Value | Role |\n"
        "|--------|-------|------|\n"
        f"| $\\alpha$ | {ALPHA} | Capital share in $A k^{{\\alpha}}$ |\n"
        f"| $\\beta$ | {BETA} | Discount factor |\n"
        f"| $A$ | {A_TFP} | Total factor productivity |\n"
        f"| $k_{{ss}}$ | {k_ss:.4f} | Closed-form steady-state capital |\n"
        f"| $c_{{ss}}$ | {c_ss:.4f} | Closed-form steady-state consumption |\n"
        f"| Training states | {train_interval_text} | Uniform random draws around $k_{{ss}}$ |\n"
        f"| Neural net | 1-16-16-1 tanh MLP | Maps $\\log(k/k_{{ss}})$ to a saving share |\n"
        f"| Saving-share bounds | [{SHARE_MIN:.2f}, {SHARE_MAX:.2f}] | Keep $c$ and $k'$ feasible |\n"
        f"| Batch size | {BATCH_SIZE} | States per gradient step |\n"
        f"| Gradient steps | {TRAIN_STEPS} | Adam updates using JAX autodiff |"
    )

    report.add_solution_method(
        "The method starts with a parameterized policy. "
        "It draws states from a training interval. "
        "It evaluates an economic residual at those states. "
        "It minimizes the average squared residual.\n\n"
        "Richer DSGE examples can optimize lifetime rewards. "
        "Other DSGE examples can minimize Bellman residuals. "
        "This model only needs Euler residuals. "
        "JAX autodiff provides the gradient. "
        "Adam updates the weights.\n\n"
        "```text\n"
        "Algorithm: simulated-state Euler-residual training\n"
        "Inputs:\n"
        "    alpha, beta, A\n"
        "    training interval K\n"
        "    saving-share bounds s_min, s_max\n"
        "    batch size n\n"
        "    steps T\n"
        "Output:\n"
        "    feasible neural policy k'(k; theta)\n"
        "Initialize theta\n"
        "Initialize Adam moments\n"
        "Compute the exact steady state k_ss\n"
        "Use sigmoid(z) = 1 / (1 + exp(-z))\n"
        "For t = 1,...,T:\n"
        "    Draw minibatch states k_1,...,k_n from K\n"
        "    Evaluate saving shares s(k_i; theta):\n"
        "        x_i = log(k_i / k_ss)\n"
        "        q_i = N_theta(x_i)\n"
        "        s_i = s_min + (s_max - s_min) * sigmoid(q_i)\n"
        "    Evaluate consumption c(k_i; theta):\n"
        "        y_i = A * k_i^alpha\n"
        "        c_i = (1 - s_i) * y_i\n"
        "    Evaluate next capital k'(k_i; theta):\n"
        "        k_i_next = s_i * y_i\n"
        "    Evaluate c(k'(k_i; theta); theta):\n"
        "        x_i_next = log(k_i_next / k_ss)\n"
        "        q_i_next = N_theta(x_i_next)\n"
        "        s_i_next = s_min + (s_max - s_min) * sigmoid(q_i_next)\n"
        "        y_i_next = A * k_i_next^alpha\n"
        "        c_i_next = (1 - s_i_next) * y_i_next\n"
        "    Compute residuals r(k_i; theta):\n"
        "        r_i = log(beta * alpha * A * k_i_next^(alpha - 1) * c_i / c_i_next)\n"
        "    Set Xi_n(theta) = (1/n) sum_i r_i^2\n"
        "    Update theta with one Adam step using the gradient of Xi_n(theta)\n"
        "Audit k'(k; theta) on a holdout grid against the exact rule\n"
        "```\n\n"
        "The audit is not part of the training loss. "
        "It exists because this Brock-Mirman case has the closed-form rule "
        "$k'=\\alpha\\beta A k^\\alpha$. "
        "The comparison shows what the residual-trained neural policy learned."
    )

    report.add_results(
        "The trained policy is almost the exact constant-saving rule. "
        "The x-axis uses capital relative to the steady state. "
        "The main object is the saving rule away from $k_{ss}$. "
        "The exact and neural policy functions lie nearly on top of each other."
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_grid_np / k_ss, exact_kp / k_ss, label="Closed form", color="#1b5e20")
    ax.plot(k_grid_np / k_ss, neural_kp / k_ss, "--", label="Neural policy", color="#0d47a1")
    ax.axline((1.0, 1.0), slope=1.0, color="0.4", linewidth=1.0, linestyle=":", label="45-degree line")
    ax.set_xlabel("$k/k_{ss}$")
    ax.set_ylabel("$k'(k)/k_{ss}$")
    ax.set_title("Policy Function")
    ax.legend()
    report.add_figure(
        "figures/policy-comparison.png",
        "Neural and closed-form capital policy",
        fig,
    )

    report.add_results(
        "The training curves show the fitting process. "
        "The Euler-residual loss falls sharply. "
        "The mean policy error falls with it."
    )
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
    report.add_figure(
        "figures/training-curves.png",
        "Training loss and mean policy error",
        fig,
        description=(
            "Both series drop by several orders of magnitude. "
            "The loss curve uses fresh minibatches. "
            "The error curve uses a fixed audit grid."
        ),
    )

    report.add_results(
        "The residual plot is the numerical check. "
        "A small log Euler residual means the Euler equation is nearly satisfied. "
        "The policy-error panel compares the neural rule with the closed-form rule. "
        "That direct comparison is available only in this teaching example."
    )
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
    report.add_figure(
        "figures/euler-residuals.png",
        "Euler residuals and policy errors over the audit grid",
        fig,
        description=(
            "Residuals stay small across the grid. "
            "The remaining policy error is centered near zero."
        ),
    )

    report.add_results(
        "Starting below the steady state gives a transition path. "
        "The exact path converges to the steady state. "
        "The neural path follows the same transition. "
        "Both paths start from the same initial capital."
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t_grid = np.arange(periods)
    ax.plot(t_grid, exact_path / k_ss, label="Closed form", color="#1b5e20")
    ax.plot(t_grid, neural_path / k_ss, "--", label="Neural policy", color="#0d47a1")
    ax.axhline(1.0, color="0.4", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Period")
    ax.set_ylabel("$k_t/k_{ss}$")
    ax.set_title("Simulated Capital Path")
    ax.legend()
    report.add_figure(
        "figures/simulated-path.png",
        "Capital transition under neural and closed-form policies",
        fig,
    )

    report.add_results(
        "The table reports the holdout audit on the plotted grid. "
        "Policy errors are in capital units. "
        "The Euler residual column is the maximum absolute log residual. "
        "Values near zero mean the Euler equation is nearly satisfied."
    )
    report.add_table(
        "tables/training-summary.csv",
        "Policy Approximation Accuracy",
        summary,
    )

    report.add_results(
        "The estimated policy is nearly identical to the exact Brock-Mirman policy. "
        "The learned saving share is nearly flat. "
        f"Its mean is {mean_saving_share:.4f}. "
        f"The exact saving share is $\\alpha\\beta={exact_saving_share:.4f}$.\n\n"
        "The policy figure is the main evidence. "
        "The table records the diagnostics behind the plot."
    )

    report.add_takeaway(
        "Deep learning is not needed to solve this Brock-Mirman model. "
        "The example is useful because the answer is known. "
        "The same residual-minimization machinery appears in larger nonlinear macro models.\n\n"
        "Feasibility comes from the policy parameterization. "
        "Training comes from Euler residuals on simulated states. "
        "Credibility comes from the out-of-sample closed-form audit."
    )

    report.add_references([
        "Brock, W. A., and Mirman, L. J. (1972). *Optimal Economic Growth and Uncertainty: The Discounted Case*. Journal of Economic Theory.",
        "Maliar, L., Maliar, S., and Winant, P. (2022). *Deep Learning for Solving Dynamic Economic Models*. Lecture slides.",
    ])

    report.write("README.md")


if __name__ == "__main__":
    main()
