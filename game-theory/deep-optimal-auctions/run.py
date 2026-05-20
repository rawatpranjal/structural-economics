#!/usr/bin/env python3
"""Deep optimal auction design with a small RegretNet-style mechanism.

The tutorial trains a tiny JAX neural auction for two bidders and one item.
The benchmark is Myerson's reserve-price auction for IID uniform values.
"""

from __future__ import annotations

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


N_BIDDERS = 2
RESERVE_PRICE = 0.5
HIDDEN_UNITS = 32
TRAIN_STEPS = 5_000
BATCH_SIZE = 512
LEARNING_RATE = 1.5e-3
MISREPORT_POINTS = 41
AUDIT_MISREPORT_POINTS = 81
AUDIT_SAMPLE_SIZE = 20_000
LOG_EVERY = 100

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPS = 1e-8

RHO_START = 10.0
RHO_GROWTH = 1.35
RHO_MAX = 200.0
AUGMENT_UPDATE_EVERY = 250

ParamTree = list[tuple[jax.Array, jax.Array]]


def myerson_expected_revenue(reserve: float = RESERVE_PRICE) -> float:
    """Expected revenue for two uniform bidders and one reserve-price item."""
    return float(1.0 / 3.0 + reserve**2 - (4.0 / 3.0) * reserve**3)


def second_price_expected_revenue() -> float:
    """Expected second-highest value with two IID U[0,1] bidders."""
    return 1.0 / 3.0


def init_params(key: jax.Array) -> ParamTree:
    """Initialize a small tanh MLP with allocation and payment heads."""
    keys = random.split(key, 3)
    w1 = random.normal(keys[0], (N_BIDDERS, HIDDEN_UNITS)) * np.sqrt(
        2.0 / (N_BIDDERS + HIDDEN_UNITS)
    )
    b1 = jnp.zeros((HIDDEN_UNITS,))
    w2 = random.normal(keys[1], (HIDDEN_UNITS, HIDDEN_UNITS)) * np.sqrt(
        2.0 / (2 * HIDDEN_UNITS)
    )
    b2 = jnp.zeros((HIDDEN_UNITS,))
    w3 = random.normal(keys[2], (HIDDEN_UNITS, N_BIDDERS + 3)) * 0.05

    # Output order: bidder-1 allocation logit, bidder-2 allocation logit,
    # no-sale logit, bidder-1 payment fraction, bidder-2 payment fraction.
    b3 = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])
    return [(w1, b1), (w2, b2), (w3, b3)]


def neural_auction(params: ParamTree, bids: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Map bid profiles into feasible allocations and payments."""
    x = 2.0 * bids - 1.0
    for weights, bias in params[:-1]:
        x = jnp.tanh(x @ weights + bias)

    weights, bias = params[-1]
    output = x @ weights + bias
    allocation_probs = jax.nn.softmax(output[:, : N_BIDDERS + 1], axis=1)
    allocations = allocation_probs[:, :N_BIDDERS]
    payment_fraction = jax.nn.sigmoid(output[:, N_BIDDERS + 1 :])
    payments = payment_fraction * allocations * bids
    return allocations, payments


def truthful_utilities(params: ParamTree, values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return truthful utilities, allocations, and payments."""
    allocations, payments = neural_auction(params, values)
    utilities = values * allocations - payments
    return utilities, allocations, payments


def revenue_and_regret(
    params: ParamTree,
    values: jax.Array,
    misreport_grid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute expected revenue and grid-based ex post regret."""
    truthful_u, _, payments = truthful_utilities(params, values)
    revenue = jnp.mean(jnp.sum(payments, axis=1))

    regrets = []
    max_regrets = []
    n_grid = misreport_grid.shape[0]
    for bidder in range(N_BIDDERS):
        tiled_reports = jnp.repeat(values[:, None, :], n_grid, axis=1)
        deviating_reports = tiled_reports.at[:, :, bidder].set(misreport_grid[None, :])
        flat_reports = deviating_reports.reshape((-1, N_BIDDERS))
        dev_allocations, dev_payments = neural_auction(params, flat_reports)
        dev_alloc_i = dev_allocations[:, bidder].reshape((-1, n_grid))
        dev_pay_i = dev_payments[:, bidder].reshape((-1, n_grid))
        dev_utility = values[:, bidder, None] * dev_alloc_i - dev_pay_i
        gain = jnp.max(dev_utility, axis=1) - truthful_u[:, bidder]
        regret = jnp.maximum(gain, 0.0)
        regrets.append(jnp.mean(regret))
        max_regrets.append(jnp.max(regret))

    truthful_ir_violation = jnp.max(jnp.maximum(-truthful_u, 0.0))
    return revenue, jnp.array(regrets), jnp.array(max_regrets), truthful_ir_violation


def training_loss(
    params: ParamTree,
    values: jax.Array,
    multipliers: jax.Array,
    rho: jax.Array,
    misreport_grid: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Augmented-Lagrangian loss: negative revenue plus regret penalties."""
    revenue, regrets, _, _ = revenue_and_regret(params, values, misreport_grid)
    penalty = jnp.sum(multipliers * regrets) + 0.5 * rho * jnp.sum(regrets**2)
    return -revenue + penalty, (revenue, regrets)


@jax.jit
def adam_step(
    params: ParamTree,
    first_moment: ParamTree,
    second_moment: ParamTree,
    key: jax.Array,
    step: jax.Array,
    multipliers: jax.Array,
    rho: jax.Array,
    misreport_grid: jax.Array,
) -> tuple[ParamTree, ParamTree, ParamTree, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One Adam update on a fresh batch of valuation profiles."""
    key, batch_key = random.split(key)
    values = random.uniform(batch_key, (BATCH_SIZE, N_BIDDERS))
    (objective, (revenue, regrets)), grads = jax.value_and_grad(
        training_loss,
        has_aux=True,
    )(params, values, multipliers, rho, misreport_grid)

    first_moment = jax.tree_util.tree_map(
        lambda old, grad: ADAM_BETA_1 * old + (1.0 - ADAM_BETA_1) * grad,
        first_moment,
        grads,
    )
    second_moment = jax.tree_util.tree_map(
        lambda old, grad: ADAM_BETA_2 * old + (1.0 - ADAM_BETA_2) * (grad * grad),
        second_moment,
        grads,
    )
    step_float = step.astype(jnp.float32)
    first_hat = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - ADAM_BETA_1**step_float),
        first_moment,
    )
    second_hat = jax.tree_util.tree_map(
        lambda moment: moment / (1.0 - ADAM_BETA_2**step_float),
        second_moment,
    )
    params = jax.tree_util.tree_map(
        lambda weight, m_hat, v_hat: weight
        - LEARNING_RATE * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS),
        params,
        first_hat,
        second_hat,
    )
    return params, first_moment, second_moment, key, objective, revenue, regrets


def train_auction() -> tuple[ParamTree, dict[str, np.ndarray]]:
    """Train the neural auction and return a compact diagnostic history."""
    key = random.PRNGKey(2026)
    params = init_params(key)
    first_moment = jax.tree_util.tree_map(jnp.zeros_like, params)
    second_moment = jax.tree_util.tree_map(jnp.zeros_like, params)
    train_grid = jnp.linspace(0.0, 1.0, MISREPORT_POINTS)
    multipliers = jnp.zeros((N_BIDDERS,))
    rho = RHO_START

    log_steps: list[int] = []
    log_objective: list[float] = []
    log_revenue: list[float] = []
    log_mean_regret: list[float] = []
    log_rho: list[float] = []

    for step in range(1, TRAIN_STEPS + 1):
        params, first_moment, second_moment, key, objective, revenue, regrets = adam_step(
            params,
            first_moment,
            second_moment,
            key,
            jnp.array(step),
            multipliers,
            jnp.array(rho),
            train_grid,
        )

        if step % AUGMENT_UPDATE_EVERY == 0:
            multipliers = jnp.maximum(0.0, multipliers + rho * regrets)
            rho = min(rho * RHO_GROWTH, RHO_MAX)

        if step == 1 or step % LOG_EVERY == 0:
            log_steps.append(step)
            log_objective.append(float(objective))
            log_revenue.append(float(revenue))
            log_mean_regret.append(float(jnp.max(regrets)))
            log_rho.append(float(rho))

    return params, {
        "step": np.array(log_steps),
        "objective": np.array(log_objective),
        "revenue": np.array(log_revenue),
        "mean_regret": np.array(log_mean_regret),
        "rho": np.array(log_rho),
    }


def myerson_revenue_on_values(values: np.ndarray) -> float:
    """Sample revenue for the two-bidder Myerson reserve auction."""
    high = np.max(values, axis=1)
    low = np.min(values, axis=1)
    payments = np.where(high >= RESERVE_PRICE, np.maximum(low, RESERVE_PRICE), 0.0)
    return float(np.mean(payments))


def second_price_revenue_on_values(values: np.ndarray) -> float:
    """Sample revenue for the second-price auction without a reserve."""
    return float(np.mean(np.min(values, axis=1)))


def audit_auction(params: ParamTree) -> dict[str, float | np.ndarray]:
    """Evaluate the learned auction on fresh valuation profiles."""
    audit_values = random.uniform(random.PRNGKey(99), (AUDIT_SAMPLE_SIZE, N_BIDDERS))
    audit_grid = jnp.linspace(0.0, 1.0, AUDIT_MISREPORT_POINTS)
    revenue, mean_regrets, max_regrets, ir_violation = revenue_and_regret(
        params,
        audit_values,
        audit_grid,
    )
    values_np = np.asarray(audit_values)
    return {
        "values": values_np,
        "neural_revenue": float(revenue),
        "mean_regrets": np.asarray(mean_regrets),
        "max_regrets": np.asarray(max_regrets),
        "ir_violation": float(ir_violation),
        "myerson_sample_revenue": myerson_revenue_on_values(values_np),
        "second_price_sample_revenue": second_price_revenue_on_values(values_np),
    }


def mechanism_surfaces(params: ParamTree, n_grid: int = 90) -> dict[str, np.ndarray]:
    """Evaluate learned allocation and payment surfaces on a square grid."""
    grid = np.linspace(0.0, 1.0, n_grid)
    v2, v1 = np.meshgrid(grid, grid)
    values = np.column_stack([v1.ravel(), v2.ravel()])
    allocations, payments = neural_auction(params, jnp.asarray(values))
    return {
        "grid": grid,
        "v1": v1,
        "v2": v2,
        "allocation_1": np.asarray(allocations[:, 0]).reshape((n_grid, n_grid)),
        "total_payment": np.asarray(jnp.sum(payments, axis=1)).reshape((n_grid, n_grid)),
    }


def audit_table(audit: dict[str, float | np.ndarray]) -> pd.DataFrame:
    """Build the report table for revenue, regret, and IR diagnostics."""
    mean_regrets = np.asarray(audit["mean_regrets"], dtype=float)
    max_regrets = np.asarray(audit["max_regrets"], dtype=float)
    rows = [
        {
            "Mechanism": "Neural auction",
            "Revenue": f"{float(audit['neural_revenue']):.4f}",
            "Largest mean bidder regret": f"{float(np.max(mean_regrets)):.4f}",
            "Max regret": f"{float(np.max(max_regrets)):.4f}",
            "Max IR violation": f"{float(audit['ir_violation']):.2e}",
        },
        {
            "Mechanism": "Myerson reserve auction",
            "Revenue": f"{float(audit['myerson_sample_revenue']):.4f}",
            "Largest mean bidder regret": "0.0000",
            "Max regret": "0.0000",
            "Max IR violation": "0.00e+00",
        },
        {
            "Mechanism": "Second-price auction",
            "Revenue": f"{float(audit['second_price_sample_revenue']):.4f}",
            "Largest mean bidder regret": "0.0000",
            "Max regret": "0.0000",
            "Max IR violation": "0.00e+00",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    print("Training a small RegretNet-style auction...")
    params, history = train_auction()
    audit = audit_auction(params)
    surfaces = mechanism_surfaces(params)

    neural_revenue = float(audit["neural_revenue"])
    myerson_sample = float(audit["myerson_sample_revenue"])
    myerson_exact = myerson_expected_revenue()
    second_price_exact = second_price_expected_revenue()
    mean_regrets = np.asarray(audit["mean_regrets"], dtype=float)
    max_regrets = np.asarray(audit["max_regrets"], dtype=float)
    max_mean_regret = float(np.max(mean_regrets))
    max_regret = float(np.max(max_regrets))

    setup_style()
    report = ModelReport(
        "Deep Learning for Optimal Auction Design",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "A seller knows the distribution of bidder values but not the realized values. "
        "The design problem is to choose an auction that earns revenue while still making "
        "truthful bidding optimal.\n\n"
        "For one item and IID uniform values, Myerson's reserve-price auction is the exact "
        "answer. Duetting et al. study the harder case where the auction itself is a neural "
        "network. The network maps reported values into allocations and payments, then "
        "training penalizes profitable misreports.\n\n"
        "This tutorial keeps the environment small enough to run locally. The computation "
        "trains a two-bidder neural auction and audits it against the Myerson benchmark."
    )

    report.add_equations(r"""
There are two risk-neutral bidders and one item. Bidder $i$ has value
$v_i \sim U[0,1]$. A direct mechanism asks each bidder to report a value
$b_i$. Given the report vector $b=(b_1,b_2)$, the mechanism returns allocation
probabilities $x_i(b)$ and payments $p_i(b)$.

Feasibility means the item is not allocated more than once:

$$
0 \leq x_i(b) \leq 1,
\qquad
x_1(b)+x_2(b) \leq 1.
$$

The neural network enforces this by applying a softmax to three outcomes:
bidder 1 wins, bidder 2 wins, or no sale. Payments are also bounded by the
allocated report in this tutorial, so truthful individual rationality is built
into the parameterization.

If bidder $i$ has true value $v_i$ but the report profile is $b$, utility is

$$
u_i(v_i,b;\theta)=v_i x_i(b;\theta)-p_i(b;\theta).
$$

Here $\theta$ denotes the neural network parameter vector.

The seller wants revenue, but the mechanism is useful only if bidders want to
tell the truth. Dominant-strategy incentive compatibility says that truthful
reporting must weakly dominate every one-player misreport:

$$
u_i(v_i,(v_i,b_{-i});\theta) \geq
u_i(v_i,(b_i',b_{-i});\theta)
\quad \forall i,v_i,b_i',b_{-i}.
$$

Regret turns that many-inequality condition into one diagnostic number. At a
value profile $v$, bidder $i$'s ex post regret is the best gain from lying:

$$
R_i(v;\theta) =
\max_{b_i' \in [0,1]}
\{u_i(v_i,(b_i',v_{-i});\theta)
{}- u_i(v_i,v;\theta)\}.
$$

If $R_i(v;\theta)=0$, the bidder cannot improve at that profile. Duetting et
al. train on a sample of value profiles $v^1,\ldots,v^L$, so the constraint is
the empirical average regret:

$$
\mathrm{rgt}_i(\theta) =
\frac{1}{L}\sum_{\ell=1}^{L} R_i(v^{\ell};\theta).
$$

Revenue on the same sample is

$$
\widehat{\mathrm{Rev}}(\theta) =
\frac{1}{L}\sum_{\ell=1}^{L}\sum_{i=1}^{2}
p_i(v^{\ell};\theta).
$$

The ideal learning problem is therefore revenue maximization subject to
truthfulness constraints:

$$
\max_{\theta} \widehat{\mathrm{Rev}}(\theta)
\quad \mathrm{subject\ to}\quad
\mathrm{rgt}_i(\theta)=0,\ i=1,2.
$$

This is why the loss is not just negative revenue. Pure revenue maximization
would let the network exploit bidders by making truthful reporting unattractive.
The constraint must push the network toward mechanisms that are hard to
manipulate.

The code uses an augmented Lagrangian:

$$
\mathcal{L}(\theta,\lambda,\rho) =
{}-\widehat{\mathrm{Rev}}(\theta)
{}+ \sum_{i=1}^{2} \lambda_i \mathrm{rgt}_i(\theta)
{}+ \frac{\rho}{2}\sum_{i=1}^{2}\mathrm{rgt}_i(\theta)^2.
$$

Here $\lambda_i$ is the Lagrange multiplier for bidder $i$'s regret constraint and $\rho$ is the penalty weight.

The first term rewards revenue. The multiplier term prices each bidder's
regret violation. The quadratic term makes large violations increasingly
expensive. As training proceeds, the multipliers and penalty weight rise when
regret remains positive.

The one-item uniform benchmark gives a clean audit. Myerson's virtual value is

$$
\phi(v)=v-\frac{1-F(v)}{f(v)}=2v-1,
$$

so the optimal reserve is $r^{\ast}=1/2$. The exact auction sells to the
highest bidder only when the highest value exceeds this reserve.
""")

    report.add_model_setup(
        "The paper studies flexible multi-bidder, multi-item settings. This tutorial uses "
        "the smallest benchmark where the answer is known.\n\n"
        "| Object | Value | Role |\n"
        "|---|---:|---|\n"
        "| Bidders | 2 | Strategic agents |\n"
        "| Items | 1 | Single allocation probability plus no-sale option |\n"
        "| Values | IID $U[0,1]$ | Private values known only to bidders |\n"
        f"| Myerson reserve $r^{{\\ast}}$ | {RESERVE_PRICE:.2f} | Exact optimal reserve for uniform values |\n"
        f"| Myerson revenue | {myerson_exact:.4f} | Analytical benchmark |\n"
        f"| Plain second-price revenue | {second_price_exact:.4f} | No-reserve benchmark |\n"
        f"| Neural net | 2-{HIDDEN_UNITS}-{HIDDEN_UNITS}-5 tanh MLP | Allocation logits and payment fractions |\n"
        f"| Training steps | {TRAIN_STEPS:,} | Adam updates with JAX autodiff |\n"
        f"| Misreport grid | {MISREPORT_POINTS} points | Inner regret search during training |"
    )

    report.add_solution_method(
        "The neural mechanism is a differentiable direct mechanism. The allocation head "
        "chooses among bidder 1, bidder 2, and no sale. The payment head chooses how much "
        "of each allocated report to charge. This keeps the object close to auction theory: "
        "the network is not predicting labels; it is choosing an allocation rule and a "
        "payment rule.\n\n"
        "The hard part is incentive compatibility. The algorithm approximates each bidder's "
        "best lie by a grid search. That makes regret differentiable through the neural "
        "mechanism except at grid argmax switches, which is enough for this small tutorial.\n\n"
        "```text\n"
        "Algorithm: local RegretNet-style auction training\n"
        "Inputs:\n"
        "    value distribution F on [0,1]^2\n"
        "    neural mechanism m_theta(b) = (x_theta(b), p_theta(b))\n"
        "    misreport grid B = {0, 1/(K-1), ..., 1}\n"
        "    batch size L, steps T, initial rho, multipliers lambda_i = 0\n"
        "Outputs:\n"
        "    trained theta, revenue audit, regret audit\n\n"
        "For t = 1,...,T:\n"
        "1. Draw v^1,...,v^L iid from F.\n\n"
        "2. Truthful pass. For each sample ell, evaluate\n"
        "       (x^ell, p^ell) = m_theta(v^ell)\n"
        "       u_i^ell = v_i^ell x_i^ell - p_i^ell\n"
        "       Rev_hat(theta) = (1/L) sum_ell sum_i p_i^ell\n\n"
        "3. Misreport pass. For each bidder i, sample ell, and b in B, form\n"
        "       report profile (b, v_-i^ell)\n"
        "       u_i^ell(b) = v_i^ell x_i(b, v_-i^ell; theta)\n"
        "                    - p_i(b, v_-i^ell; theta)\n\n"
        "4. Grid regret. Collapse the misreport utilities to\n"
        "       R_i^ell(theta) = max_b in B {u_i^ell(b) - u_i^ell}\n"
        "       rgt_i(theta) = (1/L) sum_ell max{R_i^ell(theta), 0}\n\n"
        "5. Training objective. Minimize\n"
        "       L(theta; lambda, rho) = -Rev_hat(theta)\n"
        "                              + sum_i lambda_i rgt_i(theta)\n"
        "                              + (rho/2) sum_i rgt_i(theta)^2\n\n"
        "6. Gradient step. Update theta with Adam using grad_theta L(theta; lambda, rho).\n\n"
        "7. Constraint update every M steps:\n"
        "       lambda_i <- max{0, lambda_i + rho rgt_i(theta)}\n"
        "       rho <- min{rho_growth rho, rho_max}\n\n"
        "8. Final audit. Draw fresh values and recompute Rev_hat, rgt_i, and max regret\n"
        "   with a finer grid B_audit.\n"
        "```\n\n"
        "The audit is the important part. A neural auction can earn more than Myerson on "
        "a finite regret grid if it leaves small profitable deviations. The revenue number "
        "and the regret number have to be read together."
    )

    fig1, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    extent = [0.0, 1.0, 0.0, 1.0]
    alloc_image = axes[0].imshow(
        surfaces["allocation_1"],
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    axes[0].plot([0.0, 1.0], [0.0, 1.0], color="white", linestyle="--", linewidth=1.2)
    axes[0].axhline(RESERVE_PRICE, color="white", linestyle=":", linewidth=1.6)
    axes[0].set_xlabel("Bidder 2 value")
    axes[0].set_ylabel("Bidder 1 value")
    axes[0].set_title("Learned Probability that Bidder 1 Wins")
    fig1.colorbar(alloc_image, ax=axes[0], fraction=0.046, pad=0.04)

    pay_image = axes[1].imshow(
        surfaces["total_payment"],
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    axes[1].plot([0.0, 1.0], [0.0, 1.0], color="white", linestyle="--", linewidth=1.2)
    axes[1].axhline(RESERVE_PRICE, color="white", linestyle=":", linewidth=1.6)
    axes[1].axvline(RESERVE_PRICE, color="white", linestyle=":", linewidth=1.6)
    axes[1].set_xlabel("Bidder 2 value")
    axes[1].set_ylabel("Bidder 1 value")
    axes[1].set_title("Learned Total Payment")
    fig1.colorbar(pay_image, ax=axes[1], fraction=0.046, pad=0.04)

    report.add_results(
        "The learned allocation is close to reserve-price logic. Bidder 1 wins mostly "
        "when its value is above both the rival value and the reserve. The dashed diagonal "
        "marks equal values. The dotted reserve lines show where Myerson starts selling."
    )
    report.add_figure(
        "figures/learned-mechanism.png",
        "Learned allocation and payment surfaces for the neural auction",
        fig1,
    )

    fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes2[0].plot(history["step"], history["revenue"], linewidth=2.0, label="Neural auction")
    axes2[0].axhline(myerson_exact, color="black", linestyle=":", linewidth=1.8, label="Myerson")
    axes2[0].axhline(
        second_price_exact,
        color="0.45",
        linestyle="--",
        linewidth=1.4,
        label="Second price",
    )
    axes2[0].set_xlabel("Gradient step")
    axes2[0].set_ylabel("Batch revenue")
    axes2[0].set_title("Revenue During Training")
    axes2[0].legend(fontsize=8)

    axes2[1].semilogy(
        history["step"],
        np.maximum(history["mean_regret"], 1e-6),
        linewidth=2.0,
        color="crimson",
    )
    axes2[1].set_xlabel("Gradient step")
    axes2[1].set_ylabel("Largest mean bidder regret")
    axes2[1].set_title("Regret Constraint During Training")

    report.add_results(
        "Training pushes revenue up first and then spends penalty weight reducing regret. "
        f"On the final audit sample, revenue is {neural_revenue:.4f}. The exact Myerson "
        f"benchmark is {myerson_exact:.4f}. The learned auction is not certified optimal; "
        f"its largest mean regret on the audit grid is {max_mean_regret:.4f}."
    )
    report.add_figure(
        "figures/training-diagnostics.png",
        "Revenue and regret diagnostics during augmented-Lagrangian training",
        fig2,
    )

    report.add_table(
        "tables/revenue-regret-audit.csv",
        "Revenue and Regret Audit",
        audit_table(audit),
        description=(
            "The neural auction is evaluated on fresh value draws and a finer misreport "
            "grid than the one used during training. The two benchmark auctions are DSIC, "
            "so their regret entries are zero by construction."
        ),
    )

    report.add_takeaway(
        "The Duetting et al. idea is to learn an auction as a constrained prediction "
        "problem. Revenue is the objective. Regret is the incentive-compatibility check.\n\n"
        "In the single-item uniform benchmark, Myerson gives the exact answer: sell to the "
        "highest value above reserve $r^{\\ast}=1/2$ and charge the larger of the reserve "
        "and the second value. The neural auction recovers the same broad shape, but the "
        f"audit still reports nonzero grid regret. That is the main lesson: learned "
        "mechanisms need incentive audits, not only revenue comparisons."
    )

    report.add_references([
        "[Duetting, P., Feng, Z., Narasimhan, H., Parkes, D., and Ravindranath, S. S. (2019). Optimal Auctions through Deep Learning. *Proceedings of Machine Learning Research*, 97, 1706-1715.](https://proceedings.mlr.press/v97/duetting19a.html)",
        "[Myerson, R. B. (1981). Optimal Auction Design. *Mathematics of Operations Research*, 6(1), 58-73.](https://doi.org/10.1287/moor.6.1.58)",
        "[Krishna, V. (2009). *Auction Theory*, 2nd ed. Academic Press.](https://shop.elsevier.com/books/auction-theory/krishna/978-0-12-374507-1)",
    ])

    report.write("README.md")
    print(
        "Generated: README.md + "
        f"{len(report._figures)} figures + {len(report._tables)} tables "
        f"(audit revenue={neural_revenue:.4f}, mean regret={max_mean_regret:.4f}, "
        f"max regret={max_regret:.4f}, Myerson sample revenue={myerson_sample:.4f})"
    )


if __name__ == "__main__":
    main()
