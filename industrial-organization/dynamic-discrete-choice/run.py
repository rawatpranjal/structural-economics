#!/usr/bin/env python3
"""Dynamic Discrete Choice: Rust-Style Bus Engine Replacement.

Solves and estimates a small dynamic discrete choice model with Type-I extreme
value shocks. The state is bus mileage, and the action is whether to replace the
engine. The tutorial compares full-solution maximum likelihood with a simple
Hotz-Miller conditional-choice-probability estimator.

Reference: Rust (1987); Hotz and Miller (1993).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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


def solve_ddc(
    theta: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    tol: float = 1e-10,
    max_iter: int = 5_000,
) -> dict[str, np.ndarray | float | int | bool]:
    """Solve the conditional value function contraction."""
    flow_replace = np.zeros_like(x)
    flow_keep = theta[0] + theta[1] * x
    values = np.zeros((len(x), 2))

    for iteration in range(1, max_iter + 1):
        inclusive = logsumexp(values, axis=1) + EULER_GAMMA
        next_replace = F_replace @ inclusive
        next_keep = F_keep @ inclusive
        values_new = np.column_stack([
            flow_replace + beta * next_replace,
            flow_keep + beta * next_keep,
        ])
        error = float(np.max(np.abs(values_new - values)))
        values = values_new
        if error < tol:
            break

    log_denom = logsumexp(values, axis=1)
    p_replace = np.exp(values[:, 0] - log_denom)
    return {
        "values": values,
        "p_replace": p_replace,
        "flow_keep": flow_keep,
        "iterations": iteration,
        "error": error,
        "converged": error < tol,
    }


def draw_next_state(rng: np.random.Generator, row: np.ndarray) -> int:
    """Draw a next-state index from a transition row."""
    return int(rng.choice(len(row), p=row))


def simulate_buses(
    theta: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    n_buses: int,
    n_periods: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Simulate panel data from the solved DDC model."""
    solution = solve_ddc(theta, x, F_replace, F_keep, beta)
    p_replace = np.asarray(solution["p_replace"])
    rng = np.random.default_rng(seed)

    x_index = np.zeros((n_buses, n_periods), dtype=int)
    mileage = np.zeros((n_buses, n_periods))
    replace = np.zeros((n_buses, n_periods), dtype=int)

    current = np.zeros(n_buses, dtype=int)
    for t in range(n_periods):
        x_index[:, t] = current
        mileage[:, t] = x[current]
        draws = rng.uniform(size=n_buses)
        replace[:, t] = draws < p_replace[current]

        next_state = np.empty(n_buses, dtype=int)
        for i in range(n_buses):
            row = F_replace[current[i]] if replace[i, t] else F_keep[current[i]]
            next_state[i] = draw_next_state(rng, row)
        current = next_state

    return {
        "x_index": x_index,
        "mileage": mileage,
        "replace": replace,
        "p_replace": p_replace,
    }


def panel_log_likelihood(p_replace: np.ndarray, data: dict[str, np.ndarray]) -> float:
    """Choice log likelihood for observed replacement decisions."""
    p = np.clip(p_replace[data["x_index"]], 1e-8, 1.0 - 1e-8)
    y = data["replace"]
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def estimate_full_solution(
    data: dict[str, np.ndarray],
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate structural parameters by full-solution maximum likelihood."""
    bounds = [(0.5, 3.5), (-0.35, -0.03)]

    def objective(theta: np.ndarray) -> float:
        solution = solve_ddc(theta, x, F_replace, F_keep, beta, tol=1e-8)
        return -panel_log_likelihood(np.asarray(solution["p_replace"]), data)

    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80})
    return {"theta": result.x, "objective": result.fun, "success": result.success}


def estimate_first_stage_logit(data: dict[str, np.ndarray]) -> np.ndarray:
    """Fit a reduced-form logit replacement rule in mileage and mileage squared."""
    mileage = data["mileage"].ravel()
    y = data["replace"].ravel()
    X = np.column_stack([np.ones_like(mileage), mileage, mileage**2])

    def objective(gamma: np.ndarray) -> float:
        eta = X @ gamma
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(objective, np.zeros(3), method="BFGS")
    return result.x


def hotz_miller_ccp(
    theta: np.ndarray,
    p_hat: np.ndarray,
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Map structural parameters and first-stage CCPs into model-implied CCPs."""
    p_replace = np.clip(p_hat, 1e-5, 1.0 - 1e-5)
    p_keep = 1.0 - p_replace
    flow_keep = theta[0] + theta[1] * x

    expected_flow = (
        p_replace * (-np.log(p_replace))
        + p_keep * (flow_keep - np.log(p_keep))
        + EULER_GAMMA
    )
    policy_transition = p_replace[:, None] * F_replace + p_keep[:, None] * F_keep
    ex_ante_value = np.linalg.solve(np.eye(len(x)) - beta * policy_transition, expected_flow)

    value_replace = beta * (F_replace @ ex_ante_value)
    value_keep = flow_keep + beta * (F_keep @ ex_ante_value)
    return expit(value_replace - value_keep)


def estimate_ccp(
    data: dict[str, np.ndarray],
    x: np.ndarray,
    F_replace: np.ndarray,
    F_keep: np.ndarray,
    beta: float,
    start: np.ndarray,
) -> dict[str, object]:
    """Estimate structural parameters using a Hotz-Miller-style CCP step."""
    gamma = estimate_first_stage_logit(data)
    design_grid = np.column_stack([np.ones_like(x), x, x**2])
    p_hat = expit(design_grid @ gamma)
    bounds = [(0.5, 3.5), (-0.35, -0.03)]

    def objective(theta: np.ndarray) -> float:
        p_model = hotz_miller_ccp(theta, p_hat, x, F_replace, F_keep, beta)
        return -panel_log_likelihood(p_model, data)

    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80})
    return {"theta": result.x, "objective": result.fun, "success": result.success, "p_hat": p_hat}


def main() -> None:
    beta = 0.90
    theta_true = np.array([2.00, -0.15])
    x_min, x_max, delta_x = 0.0, 15.0, 0.25
    n_buses, n_periods = 1_500, 35

    x = np.arange(x_min, x_max + delta_x, delta_x)
    F_replace, F_keep = build_transition_matrices(x, delta_x)
    solution_true = solve_ddc(theta_true, x, F_replace, F_keep, beta)
    data = simulate_buses(theta_true, x, F_replace, F_keep, beta, n_buses, n_periods, seed=12)

    full_est = estimate_full_solution(data, x, F_replace, F_keep, beta, start=np.array([1.7, -0.12]))
    ccp_est = estimate_ccp(data, x, F_replace, F_keep, beta, start=np.array([1.7, -0.12]))
    solution_full = solve_ddc(np.asarray(full_est["theta"]), x, F_replace, F_keep, beta)
    solution_ccp = solve_ddc(np.asarray(ccp_est["theta"]), x, F_replace, F_keep, beta)

    repair_rate = float(data["replace"].mean())
    average_mileage = float(data["mileage"].mean())
    high_mileage_share = float(np.mean(data["mileage"] >= 10.0))
    print("Dynamic discrete choice tutorial")
    print(f"  VFI converged in {solution_true['iterations']} iterations")
    print(f"  Simulated repair rate: {repair_rate:.3f}")
    print(f"  Full-solution estimate: {full_est['theta']}")
    print(f"  CCP estimate: {ccp_est['theta']}")

    setup_style()
    report = ModelReport(
        "Bus Engine Replacement and Dynamic Choice",
        "Continuation values, replacement hazards, and CCP estimation in a Rust-style model.",
        include_reproduce=False,
        show_figure_captions=False,
    )

    report.add_overview(
        "Engine replacement is a clean dynamic discrete choice problem: the action is "
        "discrete, but the cost of that action is paid through future states. A bus "
        "operator who keeps an old engine receives the current keep payoff and lets mileage "
        "drift upward. Replacing the engine sacrifices that current keep payoff but resets "
        "the bus toward a low-mileage state. The replacement hazard therefore summarizes "
        "both current maintenance costs and the continuation value of a fresher engine.\n\n"
        "The example is the estimation-side complement to the dynamic IO models in "
        "[dynamic entry and exit](../dynamic-entry-exit/) and "
        "[Markov-perfect investment](../dynamic-games/). Here there is one decision maker "
        "rather than strategic firms, so the focus is on recovering payoff parameters from "
        "observed choices. The tutorial solves the model, simulates a panel with known "
        "truth, and compares nested fixed-point maximum likelihood with a Hotz-Miller "
        "conditional-choice-probability (CCP) estimator."
    )

    report.add_equations(
        r"""
Let $x_t \in X$ denote mileage at the start of period $t$. The action
$a_t=1$ replaces the engine and $a_t=0$ keeps it. Replacement flow utility is
normalized to zero:

$$u(x,1) = 0,$$

and the keep payoff is

$$u(x,0) = \theta_0 + \theta_1 x, \qquad \theta_1 < 0.$$

The transition matrix $F_a(x' \mid x)$ gives next period's mileage. Replacement
uses $F_1$ and is close to the transition from a new engine; keeping uses $F_0$
and lets mileage drift upward. With additive Type-I extreme value shocks, the
conditional value functions satisfy

$$v_a(x) = u(x,a) + \beta \sum_{x'} F_a(x' \mid x)
\left[\log\left(\exp(v_1(x')) + \exp(v_0(x'))\right) + \gamma\right],$$

where $\gamma$ is Euler's constant. The replacement probability is

$$P_\theta(1 \mid x) =
\frac{\exp(v_1(x))}{\exp(v_1(x))+\exp(v_0(x))}.$$

For panel observations $(x_{it}, d_{it})$, where $d_{it}=1$ means replacement,
the full-solution likelihood is

$$\ell(\theta)=\sum_{i,t}
d_{it}\log P_\theta(1 \mid x_{it})
+ (1-d_{it})\log[1-P_\theta(1 \mid x_{it})].$$

The CCP estimator starts from a first-stage estimate $\hat p(x)$ of
$\Pr(a=1 \mid x)$. Given $\hat p$, form the policy transition

$$\hat F(x' \mid x)=\hat p(x)F_1(x' \mid x)+[1-\hat p(x)]F_0(x' \mid x).$$

For any candidate $\theta$, the Hotz-Miller ex ante value solves the linear
system

$$W_\theta =
\bar u_\theta(\hat p)+\beta \hat F W_\theta,$$

where $\bar u_\theta(\hat p)$ includes the keep payoff and the logit entropy
terms implied by $\hat p$. The model-implied replacement probability is then

$$P_\theta^{HM}(1 \mid x)
=\Lambda\left(\beta F_1 W_\theta
-\theta_0-\theta_1 x-\beta F_0 W_\theta\right),$$

with $\Lambda(z)=1/(1+\exp(-z))$.
"""
    )

    report.add_model_setup(
        f"| Parameter | Value | Description |\n"
        f"|-----------|-------|-------------|\n"
        f"| $\\beta$ | {beta} | Discount factor |\n"
        f"| $\\theta_0$ | {theta_true[0]:.2f} | Keep-engine payoff intercept |\n"
        f"| $\\theta_1$ | {theta_true[1]:.2f} | Mileage cost slope |\n"
        f"| Mileage states | {len(x)} | Grid for $x \\in [{x_min:g},{x_max:g}]$ |\n"
        f"| Transition law | Exponential increments | Replacement resets to the low-mileage transition |\n"
        f"| Buses | {n_buses} | Simulated panel units |\n"
        f"| Periods | {n_periods} | Observations per bus |\n"
        f"| Ground truth | Known | Data are simulated from $\\theta=({theta_true[0]:.2f},{theta_true[1]:.2f})$ |"
    )

    report.add_solution_method(
        "The nested fixed-point estimator treats the dynamic program as part of the "
        "likelihood. Every candidate $\\theta$ implies a replacement hazard only after the "
        "conditional value functions have been solved.\n\n"
        "```text\n"
        "Algorithm: nested fixed-point likelihood for replacement\n"
        "Input: grid X, transitions F_0 and F_1, discount beta, panel choices (x_it, d_it)\n"
        "Output: structural estimate theta and implied policy P_theta(1 | x)\n"
        "for each candidate theta proposed by the outer optimizer:\n"
        "    initialize conditional values v_0(x), v_1(x)\n"
        "    repeat:\n"
        "        inclusive(x) = log(exp(v_1(x)) + exp(v_0(x))) + gamma\n"
        "        update v_1(x) = beta * sum_x' F_1(x' | x) inclusive(x')\n"
        "        update v_0(x) = theta_0 + theta_1 x + beta * sum_x' F_0(x' | x) inclusive(x')\n"
        "        error = sup_x,a |v_a^{new}(x) - v_a^{old}(x)|\n"
        "    until error < epsilon\n"
        "    compute P_theta(1 | x) from the logit choice formula\n"
        "    evaluate the panel choice likelihood\n"
        "choose theta that maximizes the likelihood\n"
        "```\n\n"
        "The Hotz-Miller estimator moves the dynamic-programming burden into a first-stage "
        "policy estimate. In this run the first stage is a flexible logit in mileage and "
        "mileage squared; the known data-generating policy is held out for comparison, not "
        "used in estimation.\n\n"
        "```text\n"
        "Algorithm: Hotz-Miller CCP estimator\n"
        "Input: same grid, transitions, beta, and panel choices\n"
        "Output: structural estimate theta_CCP and implied policy P_theta^HM(1 | x)\n"
        "Estimate first-stage CCPs p_hat(x) = Pr(d=1 | x)\n"
        "Build the policy transition F_hat = p_hat F_1 + (1 - p_hat) F_0\n"
        "for each candidate theta:\n"
        "    construct expected flow payoffs under p_hat, including logit entropy terms\n"
        "    solve (I - beta F_hat) W_theta = expected_flow_theta\n"
        "    recover P_theta^HM(1 | x) from replacement and keep continuation values\n"
        "    evaluate the panel choice likelihood\n"
        "choose theta that maximizes the likelihood\n"
        "```\n\n"
        "The first estimator is direct but repeatedly solves a Bellman fixed point. The "
        "second estimator avoids that nested value-function iteration after the first "
        "stage, at the cost of relying on the smoothed CCPs."
    )

    values = np.asarray(solution_true["values"])
    p_true = np.asarray(solution_true["p_replace"])
    p_full = np.asarray(solution_full["p_replace"])
    p_ccp = np.asarray(solution_ccp["p_replace"])
    p_hat = np.asarray(ccp_est["p_hat"])

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1a.plot(x, values[:, 0], label="Replace, $a=1$")
    ax1a.plot(x, values[:, 1], label="Keep, $a=0$")
    ax1a.set_xlabel("Mileage")
    ax1a.set_ylabel("Conditional value")
    ax1a.set_title("Conditional Values")
    ax1a.legend()
    ax1b.plot(x, p_true, label="True")
    ax1b.plot(x, p_hat, "--", label="First-stage logit")
    ax1b.set_xlabel("Mileage")
    ax1b.set_ylabel("Replacement probability")
    ax1b.set_title("Replacement Hazard")
    ax1b.legend()
    fig1.tight_layout()
    report.add_results(
        "The first object to inspect is the replacement hazard, not the parameter vector. "
        "The keep value starts high because a low-mileage engine is still useful. "
        "As mileage rises, the keep payoff falls and replacement becomes a way to buy a "
        "better future state. The first-stage logit follows the true hazard where the "
        "simulated panel has mass, but it is only an approximation to the dynamic policy."
    )
    report.add_figure(
        "figures/value-and-ccp.png",
        "Value functions and replacement probabilities",
        fig1,
        description=(
            "The data-generating replacement probability is the benchmark curve. The "
            "estimated first-stage CCP is deliberately shown beside it because the CCP "
            "estimator lives or dies by this smoothing step."
        ),
    )

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    periods = np.arange(1, n_periods + 1)
    for bus in range(6):
        ax2.plot(periods, data["mileage"][bus], alpha=0.85, linewidth=1.6)
        repair_periods = periods[data["replace"][bus] == 1]
        ax2.scatter(repair_periods, data["mileage"][bus, data["replace"][bus] == 1], s=25)
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Mileage")
    ax2.set_title("Simulated Bus Histories")
    report.add_results(
        "The panel makes the identification problem concrete. Low and medium mileage "
        "states are observed often because buses return there after replacement. Very high "
        "mileage states are scarce: in this simulation only "
        f"**{high_mileage_share:.2%}** of bus-periods have mileage at least 10. "
        "That is where estimated hazards are expected to separate from the known policy."
    )
    report.add_figure(
        "figures/simulated-histories.png",
        "Mileage histories for six simulated buses",
        fig2,
        description=(
            "Mileage drifts upward under the keep action and falls after replacement. The "
            "points are replacement decisions, not exogenous maintenance shocks."
        ),
    )

    fig3, ax3 = plt.subplots()
    ax3.plot(x, p_true, label="True")
    ax3.plot(x, p_full, "--", label="Full-solution ML")
    ax3.plot(x, p_ccp, ":", label="CCP")
    ax3.set_xlabel("Mileage")
    ax3.set_ylabel("Replacement probability")
    ax3.set_title("Estimated Replacement Policies")
    ax3.legend()
    report.add_results(
        "Because the data are simulated, the true policy can stay on the graph as a "
        "ground-truth reference. Both estimators recover the economically important shape: "
        "replacement is rare for fresh engines and rises sharply once mileage makes keeping "
        "the engine costly. The remaining disagreement is largest in sparsely visited "
        "states; it should be read as finite-sample and first-stage approximation error, "
        "not as a different economic mechanism."
    )
    report.add_figure(
        "figures/estimated-policies.png",
        "Policy rules implied by the true and estimated parameters",
        fig3,
        description=(
            "The full-solution and CCP policies are almost on top of the truth over the "
            "states that carry most of the simulated likelihood."
        ),
    )

    theta_full = np.asarray(full_est["theta"])
    theta_ccp = np.asarray(ccp_est["theta"])
    estimate_table = pd.DataFrame({
        "Parameter": ["theta_0", "theta_1"],
        "True": theta_true,
        "Full-solution ML": theta_full,
        "Full ML error": theta_full - theta_true,
        "CCP": theta_ccp,
        "CCP error": theta_ccp - theta_true,
    })
    report.add_table(
        "tables/parameter-estimates.csv",
        "Structural parameter estimates",
        estimate_table.round(5),
        description=(
            "The estimates are close to the data-generating parameters. The full-solution "
            "estimator pays for a fresh fixed point at each trial value; the CCP estimator "
            "uses the first-stage policy to turn continuation values into a linear solve."
        ),
    )

    moments = pd.DataFrame({
        "Moment": [
            "Repair rate",
            "Average mileage",
            "Share with mileage >= 10",
            "VFI iterations",
            "Full ML success",
            "CCP success",
        ],
        "Value": [
            repair_rate,
            average_mileage,
            high_mileage_share,
            float(solution_true["iterations"]),
            float(bool(full_est["success"])),
            float(bool(ccp_est["success"])),
        ],
    })
    report.add_table(
        "tables/simulation-moments.csv",
        "Simulation and solver diagnostics",
        moments,
        description=(
            "The moments summarize the simulated panel and the numerical solve. The high-mileage "
            "share indicates how much likelihood information is available in the region where "
            "the replacement probability is already near one."
        ),
    )

    report.add_takeaway(
        "Dynamic discrete choice turns observed hazards into statements about current payoffs "
        "and continuation values. In the replacement problem, a high mileage bus is replaced "
        "not only because keeping it is costly today, but because replacement changes the "
        "distribution of tomorrow's state. Nested fixed-point likelihood estimates that object "
        "directly. CCP estimation is faster because it learns part of the policy first, but "
        "then the quality of the structural step depends on how well those first-stage CCPs "
        "approximate the true replacement hazard."
    )

    report.add_references([
        "[Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher. *Econometrica*, 55(5), 999-1033.](https://doi.org/10.2307/1911259)",
        "[Hotz, V. J. and Miller, R. A. (1993). Conditional Choice Probabilities and the Estimation of Dynamic Models. *Review of Economic Studies*, 60(3), 497-529.](https://doi.org/10.2307/2298122)",
        "[Aguirregabiria, V. and Mira, P. (2010). Dynamic Discrete Choice Structural Models: A Survey. *Journal of Econometrics*, 156(1), 38-67.](https://doi.org/10.1016/j.jeconom.2009.09.007)",
    ])
    report.write()
    save_thumbnail("figures/estimated-policies.png", "figures/thumb.png")


if __name__ == "__main__":
    main()
