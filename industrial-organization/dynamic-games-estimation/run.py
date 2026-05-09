#!/usr/bin/env python3
"""Dynamic-game estimation from first-stage CCPs and forward values."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.output import ModelReport
from lib.plotting import setup_style


EULER_GAMMA = 0.5772156649015329


def state_space(q_max: int) -> list[tuple[int, int]]:
    """Firm-view states: own quality and rival quality."""
    return [(own, rival) for own in range(q_max + 1) for rival in range(q_max + 1)]


def transition_probs(q: int, action: int, q_max: int) -> dict[int, float]:
    """Quality transition for one firm."""
    if action == 1:
        up = min(q + 1, q_max)
        return {up: 0.65, q: 0.35} if up != q else {q: 1.0}
    down = max(q - 1, 0)
    return {down: 0.12, q: 0.88} if down != q else {q: 1.0}


def flow_payoff(theta: np.ndarray, state: tuple[int, int], action: int) -> float:
    """Current payoff before private action shocks."""
    quality_value, gap_incentive, investment_cost = theta
    own, rival = state
    catch_up_gap = max(rival - own, 0)
    return quality_value * own - investment_cost * action + gap_incentive * catch_up_gap * action


def solve_soft_mpe(
    theta: np.ndarray,
    q_max: int,
    beta: float,
    tol: float = 1e-10,
    max_iter: int = 1_000,
) -> dict[str, np.ndarray | int | float | bool]:
    """Solve a symmetric logit Markov-perfect investment game for simulation truth."""
    states = state_space(q_max)
    index = {state: i for i, state in enumerate(states)}
    n_states = len(states)
    W = np.zeros(n_states)
    policy = np.full(n_states, 0.5)

    for iteration in range(1, max_iter + 1):
        W_new = np.zeros(n_states)
        policy_new = np.zeros(n_states)
        for s_idx, state in enumerate(states):
            own, rival = state
            rival_invest = policy[index[(rival, own)]]
            values = []
            for action in [0, 1]:
                continuation = 0.0
                for rival_action, rival_prob in [(0, 1.0 - rival_invest), (1, rival_invest)]:
                    for next_own, own_prob in transition_probs(own, action, q_max).items():
                        for next_rival, rival_state_prob in transition_probs(rival, rival_action, q_max).items():
                            continuation += (
                                rival_prob
                                * own_prob
                                * rival_state_prob
                                * W[index[(next_own, next_rival)]]
                            )
                values.append(flow_payoff(theta, state, action) + beta * continuation)
            policy_new[s_idx] = expit(values[1] - values[0])
            W_new[s_idx] = logsumexp(values) + EULER_GAMMA

        error = float(np.max(np.abs(W_new - W)))
        W = 0.40 * W_new + 0.60 * W
        policy = 0.40 * policy_new + 0.60 * policy
        if error < tol:
            break

    return {
        "W": W,
        "policy": policy,
        "iterations": iteration,
        "error": error,
        "converged": error < tol,
    }


def simulate_markets(
    true_policy: np.ndarray,
    q_max: int,
    n_markets: int,
    n_periods: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate a two-firm panel and return firm-view observations."""
    states = state_space(q_max)
    index = {state: i for i, state in enumerate(states)}
    rng = np.random.default_rng(seed)
    qualities = rng.integers(0, q_max + 1, size=(n_markets, 2))
    rows = []

    for t in range(n_periods):
        actions = np.zeros((n_markets, 2), dtype=int)
        for m in range(n_markets):
            q1, q2 = qualities[m]
            p1 = true_policy[index[(q1, q2)]]
            p2 = true_policy[index[(q2, q1)]]
            actions[m, 0] = rng.uniform() < p1
            actions[m, 1] = rng.uniform() < p2
            rows.append({"market": m, "period": t, "own_quality": q1, "rival_quality": q2, "invest": actions[m, 0]})
            rows.append({"market": m, "period": t, "own_quality": q2, "rival_quality": q1, "invest": actions[m, 1]})

        for m in range(n_markets):
            for firm in [0, 1]:
                probs = transition_probs(int(qualities[m, firm]), int(actions[m, firm]), q_max)
                qualities[m, firm] = rng.choice(list(probs.keys()), p=list(probs.values()))

    return pd.DataFrame(rows)


def estimate_first_stage_ccp(data: pd.DataFrame, q_max: int) -> dict[str, np.ndarray]:
    """Empirical investment probabilities by firm-view state with Laplace smoothing."""
    states = state_space(q_max)
    index = {state: i for i, state in enumerate(states)}
    counts = np.zeros(len(states))
    invest_counts = np.zeros(len(states))
    for row in data.itertuples(index=False):
        s_idx = index[(int(row.own_quality), int(row.rival_quality))]
        counts[s_idx] += 1
        invest_counts[s_idx] += int(row.invest)
    ccp = (invest_counts + 1.0) / (counts + 2.0)
    return {"ccp": ccp, "counts": counts, "invest_counts": invest_counts}


def policy_transition_and_flow(
    theta: np.ndarray,
    ccp: np.ndarray,
    q_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected transition matrix and current payoff under the first-stage policy."""
    states = state_space(q_max)
    index = {state: i for i, state in enumerate(states)}
    n_states = len(states)
    transition = np.zeros((n_states, n_states))
    expected_flow = np.zeros(n_states)
    entropy = -(ccp * np.log(ccp) + (1.0 - ccp) * np.log(1.0 - ccp)) + EULER_GAMMA

    for s_idx, state in enumerate(states):
        own, rival = state
        own_policy = ccp[s_idx]
        rival_policy = ccp[index[(rival, own)]]
        expected_flow[s_idx] = (
            (1.0 - own_policy) * flow_payoff(theta, state, 0)
            + own_policy * flow_payoff(theta, state, 1)
            + entropy[s_idx]
        )
        for own_action, own_prob in [(0, 1.0 - own_policy), (1, own_policy)]:
            for rival_action, rival_prob in [(0, 1.0 - rival_policy), (1, rival_policy)]:
                for next_own, own_state_prob in transition_probs(own, own_action, q_max).items():
                    for next_rival, rival_state_prob in transition_probs(rival, rival_action, q_max).items():
                        transition[s_idx, index[(next_own, next_rival)]] += (
                            own_prob * rival_prob * own_state_prob * rival_state_prob
                        )
    return transition, expected_flow


def evaluate_ccp_values(theta: np.ndarray, ccp: np.ndarray, q_max: int, beta: float) -> np.ndarray:
    """Policy-evaluation value under fixed first-stage CCPs."""
    transition, expected_flow = policy_transition_and_flow(theta, ccp, q_max)
    return np.linalg.solve(np.eye(len(ccp)) - beta * transition, expected_flow)


def implied_policy(theta: np.ndarray, ccp: np.ndarray, q_max: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Choice probabilities implied by CCP value evaluation."""
    states = state_space(q_max)
    index = {state: i for i, state in enumerate(states)}
    W = evaluate_ccp_values(theta, ccp, q_max, beta)
    p_model = np.zeros(len(states))

    for s_idx, state in enumerate(states):
        own, rival = state
        rival_policy = ccp[index[(rival, own)]]
        values = []
        for action in [0, 1]:
            continuation = 0.0
            for rival_action, rival_prob in [(0, 1.0 - rival_policy), (1, rival_policy)]:
                for next_own, own_prob in transition_probs(own, action, q_max).items():
                    for next_rival, rival_state_prob in transition_probs(rival, rival_action, q_max).items():
                        continuation += (
                            rival_prob
                            * own_prob
                            * rival_state_prob
                            * W[index[(next_own, next_rival)]]
                        )
            values.append(flow_payoff(theta, state, action) + beta * continuation)
        p_model[s_idx] = expit(values[1] - values[0])
    return p_model, W


def pseudo_log_likelihood(theta: np.ndarray, data: pd.DataFrame, ccp: np.ndarray, q_max: int, beta: float) -> float:
    """Forward-value CCP pseudo log likelihood."""
    if theta[0] < 0.0 or theta[0] > 1.5 or theta[1] < 0.0 or theta[1] > 1.0 or theta[2] < 0.2 or theta[2] > 2.0:
        return -1e9
    states = state_space(q_max)
    index = {state: i for i, state in enumerate(states)}
    p_model, _ = implied_policy(theta, ccp, q_max, beta)
    state_idx = np.array([index[(int(row.own_quality), int(row.rival_quality))] for row in data.itertuples(index=False)])
    y = data["invest"].to_numpy(dtype=float)
    p = np.clip(p_model[state_idx], 1e-8, 1.0 - 1e-8)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def estimate_forward_ccp(data: pd.DataFrame, ccp: np.ndarray, q_max: int, beta: float, start: np.ndarray) -> dict[str, object]:
    """Estimate structural parameters using first-stage CCP policy evaluation."""
    objective = lambda theta: -pseudo_log_likelihood(theta, data, ccp, q_max, beta)
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 350, "xatol": 1e-5, "fatol": 1e-4, "disp": False},
    )
    p_model, values = implied_policy(np.asarray(result.x, dtype=float), ccp, q_max, beta)
    return {
        "theta": np.asarray(result.x, dtype=float),
        "log_likelihood": -float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
        "p_model": p_model,
        "values": values,
    }


def forward_simulated_values(
    theta: np.ndarray,
    ccp: np.ndarray,
    q_max: int,
    beta: float,
    n_paths: int,
    horizon: int,
    seed: int,
) -> np.ndarray:
    """Monte Carlo forward values under the first-stage CCP policy."""
    states = state_space(q_max)
    index = {state: i for i, state in enumerate(states)}
    entropy = -(ccp * np.log(ccp) + (1.0 - ccp) * np.log(1.0 - ccp)) + EULER_GAMMA
    rng = np.random.default_rng(seed)
    values = np.zeros(len(states))

    for s_idx, start_state in enumerate(states):
        path_values = np.zeros(n_paths)
        for path in range(n_paths):
            own, rival = start_state
            discounted = 1.0
            total = 0.0
            for _ in range(horizon):
                current_idx = index[(own, rival)]
                rival_idx = index[(rival, own)]
                own_action = int(rng.uniform() < ccp[current_idx])
                rival_action = int(rng.uniform() < ccp[rival_idx])
                total += discounted * (flow_payoff(theta, (own, rival), own_action) + entropy[current_idx])
                own_probs = transition_probs(own, own_action, q_max)
                rival_probs = transition_probs(rival, rival_action, q_max)
                own = int(rng.choice(list(own_probs.keys()), p=list(own_probs.values())))
                rival = int(rng.choice(list(rival_probs.keys()), p=list(rival_probs.values())))
                discounted *= beta
            path_values[path] = total
        values[s_idx] = path_values.mean()
    return values


def main() -> None:
    q_max = 3
    beta = 0.90
    theta_true = np.array([0.70, 0.40, 1.00])
    truth = solve_soft_mpe(theta_true, q_max, beta)
    true_policy = np.asarray(truth["policy"], dtype=float)
    data = simulate_markets(true_policy, q_max, n_markets=1_000, n_periods=30, seed=3)
    first_stage = estimate_first_stage_ccp(data, q_max)
    estimate = estimate_forward_ccp(data, np.asarray(first_stage["ccp"]), q_max, beta, start=np.array([0.55, 0.30, 0.85]))

    simulated_values = forward_simulated_values(
        np.asarray(estimate["theta"]),
        np.asarray(first_stage["ccp"]),
        q_max,
        beta,
        n_paths=1_000,
        horizon=70,
        seed=44,
    )
    exact_values = np.asarray(estimate["values"], dtype=float)
    forward_rmse = float(np.sqrt(np.mean((simulated_values - exact_values) ** 2)))
    policy_rmse = float(np.sqrt(np.mean((np.asarray(estimate["p_model"]) - true_policy) ** 2)))
    first_stage_rmse = float(np.sqrt(np.mean((np.asarray(first_stage["ccp"]) - true_policy) ** 2)))

    print("Quality investment game estimation tutorial")
    print(f"  Truth solve converged: {truth['converged']} in {truth['iterations']} iterations")
    print(f"  Estimated theta: {np.asarray(estimate['theta'])}")
    print(f"  Policy RMSE: {policy_rmse:.4f}")
    print(f"  Forward simulation value RMSE: {forward_rmse:.4f}")

    setup_style()
    report = ModelReport(
        "Quality-Ladder Dynamic Game: Estimating with CCPs",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Two firms invest to climb a product-quality ladder. A laggard may spend to catch up. "
        "A leader may save the investment cost.\n\n"
        "The object is the payoff vector: the value of quality, the cost of investment, and "
        "the catch-up incentive. The data contain firm qualities and investment choices.\n\n"
        "A likelihood needs continuation values because today's investment changes tomorrow's "
        "state. The CCP estimator first estimates investment rates by state. It then evaluates "
        "values under those policies, instead of solving a new MPE at each trial parameter."
    )

    report.add_equations(
        r"""
The firm-view state is $\omega=(q_i,q_j)$, where $q_i$ is own quality and
$q_j$ is rival quality. Both qualities lie on the four-rung ladder
$\{0,1,2,3\}$. Firm $i$ chooses $a_i\in\{0,1\}$, where one means invest.
Flow payoff is

$$\pi_i(\omega,a_i;\theta) = \theta_q q_i - \theta_c a_i + \theta_g \max\{q_j-q_i,0\} a_i .$$

The gap term raises the investment payoff when the rival leads.

First-stage CCPs estimate the state-specific investment rate
$p(\omega)=\Pr(a_i=1\mid \omega)$. Holding CCPs fixed gives a policy transition
$\hat P$ and expected flow payoff $\bar\pi_\theta(\omega;\hat p)$. The logit
action shock is integrated out. The value under the first-stage policy is

$$W_\theta = \bar\pi_\theta(\hat p) + \beta \hat P W_\theta.$$

Choice-specific values use the rival's first-stage CCP and $W_\theta$:

$$v_\theta(a_i,\omega) = \pi_i(\omega,a_i;\theta) + \beta E_{\hat p_j}\left[W_\theta(\omega')\mid \omega,a_i\right].$$

The second-stage pseudo likelihood scores observed investment choices, where
$d_{it}\in\{0,1\}$ is the observed investment indicator for firm $i$ at period
$t$ and $\Lambda(\cdot)$ is the logistic CDF:

$$\ell(\theta)=\sum_{i,t} d_{it}\log \Lambda[v_\theta(1,\omega_{it})-v_\theta(0,\omega_{it})] +(1-d_{it})\log\{1-\Lambda[v_\theta(1,\omega_{it})-v_\theta(0,\omega_{it})]\}.$$
"""
    )

    report.add_model_setup(
        f"| Object | Value | Role |\n"
        f"|--------|-------|------|\n"
        f"| Firms | 2 | Symmetric competitors observed as own-rival state pairs |\n"
        f"| Quality rungs | 0 to {q_max} | Product-quality ladder used as the dynamic state |\n"
        f"| Discount factor | {beta:.2f} | Weight on future market position |\n"
        f"| True $\\theta_q$ | {theta_true[0]:.2f} | Value of own quality |\n"
        f"| True $\\theta_g$ | {theta_true[1]:.2f} | Catch-up incentive when the rival leads |\n"
        f"| True $\\theta_c$ | {theta_true[2]:.2f} | Investment cost |\n"
        f"| Markets | 1,000 | Independent simulated two-firm markets |\n"
        f"| Periods | 30 | Panel length used for first-stage CCPs |"
    )

    report.add_solution_method(
        "The code solves the soft MPE once to create data with known truth.\n\n"
        "Estimation then treats the first-stage CCPs as future play. For each candidate payoff vector, it "
        "evaluates the value of following those policies. The resulting invest and no-invest "
        "values enter a logit likelihood.\n\n"
        "```text\n"
        "Algorithm: CCP forward-value estimation for the quality ladder game\n"
        "Input: panel of qualities and investment choices, transition law, beta\n"
        "First stage:\n"
        "  Estimate p_hat(omega)=Pr(invest | omega) from state frequencies with smoothing\n"
        "Second stage for each candidate theta:\n"
        "  Build the transition matrix induced by p_hat for the two firms\n"
        "  Solve W_theta = expected_flow_theta(p_hat) + beta P_hat W_theta\n"
        "  Compute invest and no-invest values using W_theta and rival p_hat\n"
        "  Score the observed investment choices with the logit likelihood\n"
        "Output: payoff estimates and policy fit\n"
        "```\n\n"
        "The first-stage CCPs describe how rivals move. The second stage changes payoffs "
        "while holding those policies fixed. Each likelihood evaluation is a linear "
        "policy-evaluation solve, not a new equilibrium solve."
    )

    states = state_space(q_max)
    own_grid = np.array([s[0] for s in states]).reshape(q_max + 1, q_max + 1)
    rival_grid = np.array([s[1] for s in states]).reshape(q_max + 1, q_max + 1)
    _ = own_grid, rival_grid
    fig1, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    images = [
        true_policy.reshape(q_max + 1, q_max + 1),
        np.asarray(first_stage["ccp"]).reshape(q_max + 1, q_max + 1),
        np.asarray(estimate["p_model"]).reshape(q_max + 1, q_max + 1),
    ]
    titles = ["True CCP", "First-stage CCP", "Model-implied CCP"]
    for ax, image, title in zip(axes, images, titles):
        im = ax.imshow(image, origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Rival quality")
        ax.set_ylabel("Own quality")
        for own in range(q_max + 1):
            for rival in range(q_max + 1):
                ax.text(rival, own, f"{image[own, rival]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig1.colorbar(im, ax=axes, fraction=0.025)
    report.add_results(
        "The heatmaps show the investment pattern. Firms invest more when the rival is ahead. "
        "Top-quality firms invest less because current quality already pays. The model-implied "
        "CCPs smooth empirical frequencies through the payoff vector."
    )
    report.add_figure(
        "figures/ccp-heatmaps.png",
        "True, first-stage, and model-implied investment CCPs",
        fig1,
        description="Rows are own quality and columns are rival quality.",
    )

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.scatter(true_policy, np.asarray(first_stage["ccp"]), label=f"First stage, RMSE {first_stage_rmse:.3f}")
    ax2.scatter(true_policy, np.asarray(estimate["p_model"]), label=f"Model, RMSE {policy_rmse:.3f}")
    ax2.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax2.set_xlabel("True investment CCP")
    ax2.set_ylabel("Estimated or implied CCP")
    ax2.set_title("Policy Fit by State")
    ax2.legend()
    report.add_results(
        "Policy fit matters because the likelihood uses state-level investment probabilities. "
        f"The model-implied RMSE against truth is **{policy_rmse:.3f}**. The first-stage "
        f"empirical RMSE is **{first_stage_rmse:.3f}**."
    )
    report.add_figure(
        "figures/policy-fit.png",
        "State-level policy fit against true CCPs",
        fig2,
        description="Known truth makes it possible to separate first-stage sampling error from second-stage structural fit.",
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    ax3.scatter(exact_values, simulated_values)
    lo = min(exact_values.min(), simulated_values.min())
    hi = max(exact_values.max(), simulated_values.max())
    ax3.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("Exact CCP value")
    ax3.set_ylabel("Forward-simulated value")
    ax3.set_title("Forward Simulation Check")
    report.add_results(
        "Forward values connect observed policies to dynamic payoffs. Exact policy evaluation "
        "and Monte Carlo forward simulation target the same values. The simulation RMSE is "
        f"**{forward_rmse:.3f}** "
        "with 1,000 paths per state and a 70-period horizon."
    )
    report.add_figure(
        "figures/forward-values.png",
        "Exact versus simulated forward values",
        fig3,
        description="The check compares matrix policy evaluation with simulated paths under the same CCPs.",
    )

    parameter_df = pd.DataFrame(
        {
            "Parameter": ["Quality value theta_q", "Gap incentive theta_g", "Investment cost theta_c"],
            "True": theta_true,
            "Estimate": np.asarray(estimate["theta"]),
            "Error": np.asarray(estimate["theta"]) - theta_true,
        }
    )
    report.add_table(
        "tables/parameter-recovery.csv",
        "Known-truth parameter recovery",
        parameter_df.round(5),
        description="The estimates match the data-generating payoffs.",
    )

    diagnostics = pd.DataFrame(
        {
            "Diagnostic": [
                "Truth MPE converged",
                "Truth MPE iterations",
                "Second-stage success",
                "Second-stage iterations",
                "Second-stage log likelihood",
                "First-stage policy RMSE",
                "Model policy RMSE",
                "Forward simulation value RMSE",
                "Firm-period observations",
            ],
            "Value": [
                float(bool(truth["converged"])),
                float(truth["iterations"]),
                float(bool(estimate["success"])),
                float(estimate["iterations"]),
                float(estimate["log_likelihood"]),
                first_stage_rmse,
                policy_rmse,
                forward_rmse,
                float(len(data)),
            ],
        }
    )
    report.add_table(
        "tables/estimator-diagnostics.csv",
        "Computation and estimator diagnostics",
        diagnostics,
        description="The synthetic panel comes from one equilibrium solve. Estimation then uses CCPs and forward values.",
    )

    report.add_takeaway(
        "In this quality game, empirical investment rates show where firms try to catch up.\n\n"
        "CCP estimation turns those rates into continuation values and a choice likelihood. "
        "The second stage recovers payoffs without solving a new MPE for every parameter guess."
    )

    report.add_references(
        [
            "[Aguirregabiria, V. and Mira, P. (2007). Sequential Estimation of Dynamic Discrete Games. *Econometrica*, 75(1), 1-53.](https://doi.org/10.1111/j.1468-0262.2007.00731.x)",
            "[Bajari, P., Benkard, C. L., and Levin, J. (2007). Estimating Dynamic Models of Imperfect Competition. *Econometrica*, 75(5), 1331-1370.](https://doi.org/10.1111/j.1468-0262.2007.00796.x)",
            "[Pesendorfer, M. and Schmidt-Dengler, P. (2008). Asymptotic Least Squares Estimators for Dynamic Games. *Review of Economic Studies*, 75(3), 901-928.](https://doi.org/10.1111/j.1467-937X.2008.00496.x)",
        ]
    )
    report.write("README.md")


if __name__ == "__main__":
    main()
