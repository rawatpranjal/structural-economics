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

    summary = pd.DataFrame([{
        "final_training_loss": float(train_log["final_loss"]),
        "mean_policy_error": float(np.mean(np.abs(policy_error))),
        "max_policy_error": float(np.max(np.abs(policy_error))),
        "max_log_euler_residual": float(np.max(np.abs(euler_residual))),
        "final_path_error": final_path_error,
        "gradient_steps": TRAIN_STEPS,
    }])

    print(f"  final loss: {summary.loc[0, 'final_training_loss']:.3e}")
    print(f"  max policy error: {summary.loc[0, 'max_policy_error']:.3e}")
    print(f"  max log Euler residual: {summary.loc[0, 'max_log_euler_residual']:.3e}")

    setup_style()

    report = ModelReport(
        "Deep Learning for Optimal Growth",
        "A tiny JAX neural net learns the Brock-Mirman saving rule from Euler residuals, then faces a closed-form audit.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A planner allocates output between consumption today and capital tomorrow. "
        "The log Cobb-Douglas version is useful because the exact saving rule is known. "
        "That makes it a clean place to study a neural policy solver without hiding the economics.\n\n"
        "Recent deep-learning macro methods turn dynamic models into empirical-risk "
        "minimization problems over simulated states. The loss can use lifetime rewards, "
        "Euler residuals, or Bellman residuals. This first example uses Euler residuals. "
        "The neural net proposes a feasible saving share, gradients update its weights, "
        "and the closed-form Brock-Mirman policy audits the result point by point."
    )

    report.add_equations(
        r"""
Capital fully depreciates each period. With state $k_t$, output is

$$
y_t = A k_t^{\alpha}, \qquad 0<\alpha<1,
$$

and feasibility is

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

The code evaluates it as the log residual

$$
r(k;\theta)
= \log\left[
\beta \alpha A k'(k;\theta)^{\alpha-1}
\frac{c(k;\theta)}{c(k'(k;\theta);\theta)}
\right].
$$

Zero means the Euler equation holds.

The neural policy first chooses a saving share,

$$
s(k;\theta)
= s_{\min} + (s_{\max}-s_{\min})
\sigma\left(N_{\theta}(\log(k/k_{ss}))\right),
$$

then imposes feasibility by construction:

$$
k'(k;\theta)=s(k;\theta)A k^{\alpha},
\qquad
c(k;\theta)=(1-s(k;\theta))A k^{\alpha}.
$$

For this special case, the exact policy is

$$
k'(k)=\alpha\beta A k^{\alpha},
\qquad
c(k)=(1-\alpha\beta)A k^{\alpha},
$$

and the steady state is

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
        "The broader deep-learning recipe is to choose a parameterized policy, "
        "simulate or draw states, build an economic residual at those states, "
        "and minimize the average squared residual. In richer DSGE models the "
        "same template can use lifetime rewards or Bellman residuals. Here the "
        "Euler equation is enough because the deterministic growth model has one "
        "state and one policy.\n\n"
        "```text\n"
        "Algorithm: simulated-state Euler-residual training\n"
        "Input: primitives alpha, beta, A; training interval K; neural weights theta\n"
        "Output: feasible neural policy k'(k; theta)\n"
        "Compute the exact steady state k_ss and draw minibatches k_i from K\n"
        "For each k_i, map log(k_i/k_ss) through the MLP and sigmoid saving-share transform\n"
        "Construct c_i, k'_i, and c(k'_i; theta) so feasibility holds by construction\n"
        "Evaluate the log Euler residual r(k_i; theta)\n"
        "Minimize mean_i r(k_i; theta)^2 with Adam and JAX autodiff\n"
        "Audit the trained policy on an out-of-sample capital grid against the exact rule\n"
        "```\n\n"
        "The audit is not part of the training loss. It exists because this "
        "Brock-Mirman case has the closed-form rule $k'=\\alpha\\beta A k^\\alpha$. "
        "The comparison shows what the residual-trained neural policy learned."
    )

    report.add_results(
        "The trained policy is almost the exact constant-saving rule. The x-axis "
        "uses capital relative to the steady state, so the main economic object is "
        "the slope of the saving rule away from $k_{ss}$."
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
        description="The exact and neural policy functions lie nearly on top of each other.",
    )

    report.add_results(
        "The residual plot is the numerical check. A small log Euler residual means "
        "the neural policy makes marginal utility today match discounted marginal "
        "utility tomorrow. The policy-error panel measures the same approximation "
        "directly against the closed-form rule, which is available only in this "
        "teaching example."
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
        description="Residuals stay small across the grid, and the remaining policy error is centered near zero.",
    )

    report.add_results(
        "Starting below the steady state gives a transition path. The exact and "
        "neural paths converge to the same steady state because the learned saving "
        "share reproduces the Brock-Mirman policy."
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
        description="The neural policy tracks the closed-form transition path from the same initial capital.",
    )

    report.add_results(
        "The table reports the out-of-sample audit on the plotted grid. Policy "
        "errors are in capital units. The Euler residual is the maximum absolute "
        "log residual, so values near zero mean the intertemporal first-order "
        "condition is nearly satisfied."
    )
    report.add_table(
        "tables/training-summary.csv",
        "Neural Policy Audit",
        summary,
    )

    report.add_results(
        "The learned saving share is nearly flat, as theory predicts. Across the "
        f"audit grid its mean is {share_grid.mean():.4f}; the exact saving share "
        f"is $\\alpha\\beta={ALPHA * BETA:.4f}$."
    )

    report.add_takeaway(
        "Deep learning is not needed to solve this Brock-Mirman model. Its value "
        "here is pedagogical: the same residual-minimization machinery used in "
        "larger nonlinear macro models can be inspected in a case where the answer "
        "is known. Feasibility comes from the policy parameterization, training "
        "comes from Euler residuals on simulated states, and credibility comes "
        "from an out-of-sample closed-form audit."
    )

    report.add_references([
        "Brock, W. A., and Mirman, L. J. (1972). *Optimal Economic Growth and Uncertainty: The Discounted Case*. Journal of Economic Theory.",
        "Maliar, L., Maliar, S., and Winant, P. (2022). *Deep Learning for Solving Dynamic Economic Models*. Lecture slides.",
    ])

    report.write("README.md")


if __name__ == "__main__":
    main()
