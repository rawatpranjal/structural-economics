#!/usr/bin/env python3
"""Inverse RL for the Rust-style bus engine replacement problem.

The tutorial compares a Rust nested fixed-point estimator with a maximum
causal entropy inverse reinforcement learning objective. In this finite-state
logit model, both optimize the same soft Bellman policy likelihood.
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib.output import ModelReport
from lib.plotting import save_thumbnail, setup_style


EULER_GAMMA = 0.5772156649015329


def build_transition_matrices(x: np.ndarray, delta_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Build replacement and no-replacement mileage transition matrices."""
    n = len(x)
    keep = np.zeros((n, n))
    for i, current in enumerate(x):
        increments = x - current
        valid = increments >= -1e-12
        keep[i, valid] = np.exp(-np.maximum(increments[valid], 0.0)) * (1.0 - np.exp(-delta_x))
        keep[i, -1] += max(0.0, 1.0 - keep[i].sum())
        keep[i] = keep[i] / keep[i].sum()

    replace = np.tile(keep[0], (n, 1))
    return replace, keep


def reward_features(x: np.ndarray) -> np.ndarray:
    """Return action-dependent reward features.

    Action 0 is replacement and has normalized reward zero. Action 1 is keeping
    the old engine and has features (1, mileage).
    """
    features = np.zeros((len(x), 2, 2))
    features[:, 1, 0] = 1.0
    features[:, 1, 1] = x
    return features


def rewards_from_theta(theta: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Map reward weights into state-action rewards."""
    return np.einsum("xaf,f->xa", features, theta)


def solve_soft_bellman(
    rewards: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    tol: float = 1e-10,
    max_iter: int = 5_000,
) -> dict[str, np.ndarray | float | int | bool]:
    """Solve the entropy-regularized Bellman fixed point."""
    q_values = np.zeros_like(rewards)
    transitions = (F_replace, F_keep)

    for iteration in range(1, max_iter + 1):
        inclusive = logsumexp(q_values, axis=1) + EULER_GAMMA
        q_new = np.empty_like(q_values)
        for action, transition in enumerate(transitions):
            q_new[:, action] = rewards[:, action] + beta * (transition @ inclusive)
        error = float(np.max(np.abs(q_new - q_values)))
        q_values = q_new
        if error < tol:
            break

    log_policy = q_values - logsumexp(q_values, axis=1, keepdims=True)
    policy = np.exp(log_policy)
    return {
        "q_values": q_values,
        "policy": policy,
        "p_replace": policy[:, 0],
        "log_policy": log_policy,
        "iterations": iteration,
        "error": error,
        "converged": error < tol,
    }


def solve_for_theta(
    theta: np.ndarray,
    features: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    tol: float = 1e-10,
) -> dict[str, np.ndarray | float | int | bool]:
    """Solve the soft Bellman system for a candidate reward vector."""
    rewards = rewards_from_theta(theta, features)
    return solve_soft_bellman(rewards, F_replace, F_keep, beta, tol=tol)


def draw_next_state(rng: np.random.Generator, row: np.ndarray) -> int:
    """Draw a next-state index from a transition row."""
    return int(rng.choice(len(row), p=row))


def simulate_buses(
    theta: np.ndarray,
    features: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    n_buses: int,
    n_periods: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate panel data from the solved replacement model."""
    solution = solve_for_theta(theta, features, F_replace, F_keep, beta)
    p_replace = np.asarray(solution["p_replace"])
    rng = np.random.default_rng(seed)

    x_index = np.zeros((n_buses, n_periods), dtype=int)
    mileage = np.zeros((n_buses, n_periods))
    replace = np.zeros((n_buses, n_periods), dtype=int)

    current = np.zeros(n_buses, dtype=int)
    for t in range(n_periods):
        x_index[:, t] = current
        mileage[:, t] = x[current]
        replace[:, t] = rng.uniform(size=n_buses) < p_replace[current]

        next_state = np.empty(n_buses, dtype=int)
        for i in range(n_buses):
            transition = F_replace[current[i]] if replace[i, t] else F_keep[current[i]]
            next_state[i] = draw_next_state(rng, transition)
        current = next_state

    return {
        "x_index": x_index,
        "mileage": mileage,
        "replace": replace,
        "action_index": 1 - replace,
        "p_replace": p_replace,
    }


def panel_log_likelihood(log_policy: np.ndarray, data: dict[str, np.ndarray]) -> float:
    """Choice log likelihood for observed actions given observed states."""
    states = data["x_index"].ravel()
    actions = data["action_index"].ravel()
    return float(np.sum(log_policy[states, actions]))


def estimate_nfxp(
    data: dict[str, np.ndarray],
    features: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate reward parameters by nested fixed-point maximum likelihood."""
    bounds = [(0.5, 3.5), (-0.35, -0.03)]

    def objective(theta: np.ndarray) -> float:
        solution = solve_for_theta(theta, features, F_replace, F_keep, beta, tol=1e-8)
        return -panel_log_likelihood(np.asarray(solution["log_policy"]), data)

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-10, "gtol": 1e-6},
    )
    solution = solve_for_theta(result.x, features, F_replace, F_keep, beta)
    log_likelihood = panel_log_likelihood(np.asarray(solution["log_policy"]), data)
    return {
        "theta": result.x,
        "objective": float(result.fun),
        "log_likelihood": log_likelihood,
        "success": bool(result.success),
        "iterations": int(result.nit),
        "message": str(result.message),
        "solution": solution,
    }


def estimate_mce_irl(
    data: dict[str, np.ndarray],
    features: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate reward parameters by the maximum causal entropy objective."""
    bounds = [(0.5, 3.5), (-0.35, -0.03)]

    def causal_entropy_loss(theta: np.ndarray) -> float:
        solution = solve_for_theta(theta, features, F_replace, F_keep, beta, tol=1e-8)
        return -panel_log_likelihood(np.asarray(solution["log_policy"]), data)

    result = minimize(
        causal_entropy_loss,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 120, "ftol": 1e-10, "gtol": 1e-6},
    )
    solution = solve_for_theta(result.x, features, F_replace, F_keep, beta)
    log_likelihood = panel_log_likelihood(np.asarray(solution["log_policy"]), data)
    return {
        "theta": result.x,
        "objective": float(result.fun),
        "log_likelihood": log_likelihood,
        "success": bool(result.success),
        "iterations": int(result.nit),
        "message": str(result.message),
        "solution": solution,
    }


def observed_state_weights(data: dict[str, np.ndarray], n_states: int) -> np.ndarray:
    """Count how often each mileage state appears in the panel."""
    counts = np.bincount(data["x_index"].ravel(), minlength=n_states).astype(float)
    return counts / counts.sum()


def conditional_feature_moments(
    data: dict[str, np.ndarray],
    features: np.ndarray,
    policy: np.ndarray,
) -> pd.DataFrame:
    """Compare demonstrated and policy-implied reward features at observed states."""
    states = data["x_index"].ravel()
    actions = data["action_index"].ravel()
    demonstrated = features[states, actions].mean(axis=0)
    expected = np.sum(policy[states, :, None] * features[states], axis=1).mean(axis=0)
    names = ["Keep action", "Keep mileage"]
    return pd.DataFrame({
        "Feature": names,
        "Demonstrations": demonstrated,
        "MCE policy": expected,
        "Gap": demonstrated - expected,
    })


def finite_horizon_feature_occupancy(
    data: dict[str, np.ndarray],
    features: np.ndarray,
    policy: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
) -> pd.DataFrame:
    """Compare discounted feature occupancy in demonstrations and the learned policy."""
    _, n_periods = data["replace"].shape
    discounts = beta ** np.arange(n_periods)
    demonstrated = np.zeros(features.shape[-1])
    for t, discount in enumerate(discounts):
        states_t = data["x_index"][:, t]
        actions_t = data["action_index"][:, t]
        demonstrated += discount * features[states_t, actions_t].mean(axis=0)

    transitions = policy[:, 0, None] * F_replace + policy[:, 1, None] * F_keep
    state_dist = np.bincount(data["x_index"][:, 0], minlength=features.shape[0]).astype(float)
    state_dist = state_dist / state_dist.sum()
    expected = np.zeros(features.shape[-1])
    expected_feature_by_state = np.sum(policy[:, :, None] * features, axis=1)
    for discount in discounts:
        expected += discount * (state_dist @ expected_feature_by_state)
        state_dist = state_dist @ transitions

    names = ["Keep action", "Keep mileage"]
    return pd.DataFrame({
        "Feature": names,
        "Demonstrations": demonstrated,
        "MCE policy": expected,
        "Gap": demonstrated - expected,
    })


def main() -> None:
    os.chdir(TUTORIAL_DIR)

    beta = 0.90
    theta_true = np.array([2.00, -0.15])
    x_min, x_max, delta_x = 0.0, 15.0, 0.25
    n_buses, n_periods = 1_500, 35

    x = np.arange(x_min, x_max + delta_x, delta_x)
    F_replace, F_keep = build_transition_matrices(x, delta_x)
    features = reward_features(x)
    true_solution = solve_for_theta(theta_true, features, F_replace, F_keep, beta)
    data = simulate_buses(
        theta_true,
        features,
        x,
        F_replace,
        F_keep,
        beta,
        n_buses=n_buses,
        n_periods=n_periods,
        seed=12,
    )

    nfxp = estimate_nfxp(
        data,
        features,
        F_replace,
        F_keep,
        beta,
        start=np.array([1.7, -0.12]),
    )
    mce = estimate_mce_irl(
        data,
        features,
        F_replace,
        F_keep,
        beta,
        start=np.array([2.4, -0.22]),
    )

    nfxp_solution = nfxp["solution"]
    mce_solution = mce["solution"]
    p_true = np.asarray(true_solution["p_replace"])
    p_nfxp = np.asarray(nfxp_solution["p_replace"])
    p_mce = np.asarray(mce_solution["p_replace"])
    policy_mce = np.asarray(mce_solution["policy"])
    q_true = np.asarray(true_solution["q_values"])
    q_mce = np.asarray(mce_solution["q_values"])

    repair_rate = float(data["replace"].mean())
    average_mileage = float(data["mileage"].mean())
    state_weights = observed_state_weights(data, len(x))
    log_likelihood_gap = abs(float(nfxp["log_likelihood"]) - float(mce["log_likelihood"]))
    max_policy_gap = float(np.max(np.abs(p_nfxp - p_mce)))

    if not bool(nfxp["success"]) or not bool(mce["success"]):
        raise RuntimeError("NFXP and MCE-IRL optimizers must both converge.")
    if log_likelihood_gap > 1e-4:
        raise RuntimeError(f"Likelihood gap is too large: {log_likelihood_gap:.3e}")
    if max_policy_gap > 1e-5:
        raise RuntimeError(f"Policy gap is too large: {max_policy_gap:.3e}")

    print("Inverse RL Rust bus tutorial")
    print(f"  NFXP estimate: {nfxp['theta']}")
    print(f"  MCE-IRL estimate: {mce['theta']}")
    print(f"  Log likelihood gap: {log_likelihood_gap:.3e}")
    print(f"  Max replacement-policy gap: {max_policy_gap:.3e}")

    setup_style()
    report = ModelReport(
        "Inverse Reinforcement Learning for the Rust Bus Problem",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Rust estimates payoffs from replacement choices by solving a dynamic discrete "
        "choice model. Maximum causal entropy inverse reinforcement learning estimates "
        "a reward from demonstrations. In this finite-state logit setting, the two "
        "routes meet at the same object: a soft Bellman policy likelihood.\n\n"
        "The example uses the same bus engine replacement environment as the dynamic "
        "choice tutorial. Mileage is the state. The action is to replace the engine or "
        "keep it for another period. The data are simulated from known payoffs, so the "
        "exercise is an equivalence check rather than an identification claim about "
        "all inverse reinforcement learning estimators."
    )

    report.add_equations(
        r"""
Let $x_t \in X$ denote mileage. The action $a_t=0$ replaces the engine and
$a_t=1$ keeps it. Replacement flow utility is normalized to zero:

$$r_\theta(x,\text{replace})=0.$$

The keep reward is linear in action-dependent features:

$$r_\theta(x,\text{keep})=\theta_0+\theta_1 x,\qquad \theta_1<0.$$

Write a demonstrated history as

$$\tau_i=(x_{i0},a_{i0},x_{i1},a_{i1},\ldots).$$

The reward features are action dependent. Replacement has feature vector
$f(x,\text{replace})=(0,0)$, while keeping has $f(x,\text{keep})=(1,x)$, so
$r_\theta(x,a)=\theta^\top f(x,a)$.

Maximum causal entropy IRL asks for a policy that earns reward while remaining
as random as possible, conditional on the observed state history:

$$\max_\pi\ E_\pi\left[\sum_{t\geq 0}\beta^t r_\theta(x_t,a_t)\right]+H_c(\pi),$$

where the causal entropy term is

$$H_c(\pi)=E_\pi\left[-\sum_{t\geq 0}\beta^t\log\pi(a_t\mid x_t)\right].$$

The inverse problem chooses reward weights so demonstrated and policy-implied
discounted features match:

$$\widehat\mu_D=\frac{1}{N}\sum_{i=1}^N\sum_{t\geq 0}\beta^t f(x_{it},a_{it})
=E_{\pi_\theta}\left[\sum_{t\geq 0}\beta^t f(x_t,a_t)\right].$$

A compact Lagrangian view puts the feature-matching multipliers in the reward:

$$\mathcal L(\pi,\theta)=H_c(\pi)+\theta^\top\left(E_\pi\left[\sum_{t\geq 0}\beta^t f(x_t,a_t)\right]-\widehat\mu_D\right).$$

For a fixed $\theta$, the policy problem has the causal logit solution. Define

$$Q_\theta(x,a)=r_\theta(x,a)+\beta\sum_{x'}F_a(x'\mid x)\left[\log\sum_{b\in\{0,1\}}\exp Q_\theta(x',b)+\gamma\right].$$

Then

$$\pi_\theta(a\mid x)=\frac{\exp Q_\theta(x,a)}{\sum_{b\in\{0,1\}}\exp Q_\theta(x,b)}.$$

Rust-style NFXP solves the Bellman equation inside the likelihood and maximizes

$$\ell(\theta)=\sum_{i,t}\log\pi_\theta(a_{it}\mid x_{it}).$$

The implementation estimates $\theta$ by this conditional demonstration
likelihood. The MCE view interprets the same first-order condition as
reward-feature matching. With the reward normalization, transitions, discount
factor, and logit scale fixed, the MCE-IRL policy and Rust logit DDC likelihood
are the same object in this example. Sanghvi et al. (2021) formalize this
connection by showing that MCE-IRL and NFXP share the same objective form,
causal-logit policy, and feature-count gradient in this class of models.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $\\theta_0$ | {theta_true[0]:.2f} | Keep-engine reward intercept |\n"
        f"| $\\theta_1$ | {theta_true[1]:.2f} | Mileage reward slope |\n"
        f"| Mileage states | {len(x)} | Grid for $x \\in [{x_min:g},{x_max:g}]$ |\n"
        f"| Actions | 2 | Replace or keep |\n"
        f"| Transition law | Exponential increments | Replacement resets to the low-mileage transition |\n"
        f"| Buses | {n_buses} | Simulated panel units |\n"
        f"| Periods | {n_periods} | Observations per bus |\n"
        f"| Ground truth | Known | Data are simulated from $\\theta=({theta_true[0]:.2f},{theta_true[1]:.2f})$ |"
    )

    report.add_solution_method(
        "Both estimators call the same soft Bellman solver. The only difference is the "
        "interpretation of the outer objective. NFXP calls it a dynamic discrete choice "
        "likelihood. MCE-IRL calls it the likelihood of demonstrated actions under the "
        "maximum-causal-entropy policy.\n\n"
        "```text\n"
        "Algorithm: NFXP and MCE-IRL equivalence check\n"
        "Input: mileage grid X, transitions F_replace and F_keep, beta, demonstrations\n"
        "Build reward features f(x, replace) = (0, 0) and f(x, keep) = (1, x)\n"
        "for each candidate theta proposed by the optimizer:\n"
        "    set r_theta(x, a) = f(x, a)' theta\n"
        "    solve the soft Bellman fixed point for Q_theta(x, a)\n"
        "    form pi_theta(a | x) from the softmax of Q_theta\n"
        "    evaluate sum log pi_theta(a_it | x_it)\n"
        "choose theta that maximizes the demonstration likelihood\n"
        "compare NFXP and MCE-IRL estimates, likelihoods, and policies\n"
        "```\n\n"
        "The comparison is deliberately narrow. The reward normalization is fixed, the "
        "transition law is known, and the data come from the same Rust-style model. "
        "Under those conditions, reward recovery and structural dynamic choice "
        "estimation solve the same numerical problem."
    )

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1a.plot(x, q_true[:, 0], label="Replace, true")
    ax1a.plot(x, q_true[:, 1], label="Keep, true")
    ax1a.plot(x, q_mce[:, 0], "--", label="Replace, MCE-IRL")
    ax1a.plot(x, q_mce[:, 1], "--", label="Keep, MCE-IRL")
    ax1a.set_xlabel("Mileage")
    ax1a.set_ylabel("Soft Q value")
    ax1a.set_title("Soft Bellman Values")
    ax1a.legend()
    ax1b.plot(x, p_true, label="True")
    ax1b.plot(x, p_nfxp, "--", label="NFXP")
    ax1b.plot(x, p_mce, ":", label="MCE-IRL")
    ax1b.set_xlabel("Mileage")
    ax1b.set_ylabel("Replacement probability")
    ax1b.set_title("Replacement Rule")
    ax1b.legend()
    fig1.tight_layout()
    report.add_results(
        "The learned MCE-IRL values line up with the Rust dynamic choice values after "
        "using the same reward normalization. The policy panel is the main check: NFXP "
        "and MCE-IRL choose the same mileage-specific replacement rule."
    )
    report.add_figure(
        "figures/value-policy-equivalence.png",
        "Soft values and replacement policies from NFXP and MCE-IRL",
        fig1,
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.bar(x, state_weights, width=delta_x * 0.9, alpha=0.8)
    ax2.set_xlabel("Mileage")
    ax2.set_ylabel("Share of observed bus-periods")
    ax2.set_title("Observed State Support")
    report.add_results(
        "Most likelihood weight comes from low and middle mileage states. High mileage "
        "states still discipline the slope, but they are rare because replacement resets "
        "the engine."
    )
    report.add_figure(
        "figures/state-support.png",
        "Observed mileage support in the simulated panel",
        fig2,
    )

    conditional_moments = conditional_feature_moments(data, features, policy_mce)
    occupancy_moments = finite_horizon_feature_occupancy(
        data,
        features,
        policy_mce,
        F_replace,
        F_keep,
        beta,
    )
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    labels = occupancy_moments["Feature"].to_numpy()
    positions = np.arange(len(labels))
    width = 0.35
    ax3.bar(positions - width / 2, occupancy_moments["Demonstrations"], width, label="Demonstrations")
    ax3.bar(positions + width / 2, occupancy_moments["MCE policy"], width, label="MCE policy")
    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Discounted feature count")
    ax3.set_title("Reward Feature Occupancy")
    ax3.legend()
    report.add_results(
        "The IRL view is a reward-feature view of the same choices. The demonstrated "
        "feature occupancies are close to the occupancies implied by the recovered "
        "maximum-causal-entropy policy."
    )
    report.add_figure(
        "figures/feature-occupancy.png",
        "Discounted feature occupancy in demonstrations and the MCE policy",
        fig3,
    )

    theta_nfxp = np.asarray(nfxp["theta"])
    theta_mce = np.asarray(mce["theta"])
    estimates = pd.DataFrame({
        "Parameter": ["theta_0", "theta_1"],
        "True": theta_true,
        "NFXP": theta_nfxp,
        "NFXP error": theta_nfxp - theta_true,
        "MCE-IRL": theta_mce,
        "MCE-IRL error": theta_mce - theta_true,
        "NFXP minus MCE-IRL": theta_nfxp - theta_mce,
    })
    report.add_table(
        "tables/parameter-estimates.csv",
        "Reward parameter estimates",
        estimates.round(6),
        description=(
            "Both methods recover the same reward weights. The finite-sample estimates "
            "are close to the known simulation truth."
        ),
    )

    diagnostics = pd.DataFrame({
        "Diagnostic": [
            "NFXP success",
            "MCE-IRL success",
            "NFXP log likelihood",
            "MCE-IRL log likelihood",
            "Absolute log likelihood difference",
            "Max replacement-policy difference",
            "Repair rate",
            "Average mileage",
            "Bellman iterations at truth",
            "Bellman error at MCE-IRL estimate",
        ],
        "Value": [
            float(bool(nfxp["success"])),
            float(bool(mce["success"])),
            float(nfxp["log_likelihood"]),
            float(mce["log_likelihood"]),
            log_likelihood_gap,
            max_policy_gap,
            repair_rate,
            average_mileage,
            float(true_solution["iterations"]),
            float(mce_solution["error"]),
        ],
    })
    report.add_table(
        "tables/equivalence-diagnostics.csv",
        "Equivalence diagnostics",
        diagnostics,
        description=(
            "The likelihood and policy gaps are numerical, not economic. They confirm "
            "that the two labels describe the same optimization problem here."
        ),
    )

    report.add_table(
        "tables/conditional-feature-moments.csv",
        "Conditional reward-feature moments",
        conditional_moments.round(6),
        description=(
            "These moments compare demonstrated reward features with the features "
            "predicted by the MCE policy at the observed mileage states."
        ),
    )
    report.add_table(
        "tables/discounted-feature-occupancy.csv",
        "Discounted reward-feature occupancy",
        occupancy_moments.round(6),
        description=(
            "The finite-horizon occupancy calculation starts from the same initial "
            "state distribution as the simulated bus panel."
        ),
    )

    report.add_takeaway(
        "In this controlled Rust-style replacement model, NFXP and MCE-IRL are not two "
        "different estimators of different objects. They solve the same soft Bellman "
        "likelihood with a different vocabulary. NFXP emphasizes structural payoffs and "
        "continuation values. MCE-IRL emphasizes reward features and demonstrations. "
        "The equivalence depends on the finite-state logit model, known transitions, "
        "and the replacement reward normalization."
    )

    report.add_references([
        "[Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. *Econometrica*, 55(5), 999-1033.](https://doi.org/10.2307/1911259)",
        "[Ziebart, B. D., Maas, A., Bagnell, J. A., and Dey, A. K. (2008). Maximum Entropy Inverse Reinforcement Learning. *AAAI*.](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf)",
        "[Sanghvi, N., Usami, S., Sharma, M., Groeger, J., and Kitani, K. (2021). Inverse Reinforcement Learning with Explicit Policy Estimates. *AAAI*.](https://arxiv.org/abs/2103.02863)",
        "[EconIRL documentation: inverse reinforcement learning estimators for economics.](https://econirl.readthedocs.io/en/latest/)",
    ])
    report.write()
    save_thumbnail("figures/value-policy-equivalence.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
